"""The operating system, checked once per process (this module's first import: app.main at the start, and each
spawned worker), and the operations that differ by it. Windows, Linux and macOS are supported; on any other system
the app warns at the start and runs on the fallbacks. Code branches on the flags here (OS, IS_WINDOWS, IS_LINUX,
IS_MACOS), never on sys.platform / os.name / platform.system() itself.

Where the Windows API is used, the other systems have a fallback:
- elevated process (app.main refuses it): IsUserAnAdmin / effective uid 0 (root)
- the app's taskbar identity and 1 ms timer resolution (app.window): Windows only; the others need neither
- console windows of child programs (FFmpeg, the audio engine) kept hidden: Windows only; the others open none
- revealing a file: Explorer, the file selected / Finder, the file selected / the folder in the file manager
- the window's occlusion (ui.window_visibility): user32 + DWM / whether Qt has the window exposed
- the scratch cursor poller (audioengine.python.cursor_poller): Win32 cursor + waitable timer / the GUI's mouse events
- the physical core count for sklearn (analyzer.runtime): GetLogicalProcessorInformationEx / loky's own
- a key another program holds as a global hotkey (Settings > Shortcuts): RegisterHotKey refused / not checked
- paths of MAX_PATH (260) characters or more: refused while Windows' long paths are off (RtlAreLongPathsEnabled) /
  no limit elsewhere
- a sync client's folder (cloud_sync_folder): the cloud files API's sync roots, then the clients' known folders /
  the known folders (macOS ~/Library/CloudStorage, iCloud Drive, Dropbox's info.json); a file kept online only
  (online_only): its recall attributes / SF_DATALESS / not told on Linux

A path compared with another is compared by its path_key: one spelling of the file however each is written.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unicodedata
from functools import lru_cache

WINDOWS, LINUX, MACOS, OTHER = "windows", "linux", "macos", "other"
SUPPORTED = (WINDOWS, LINUX, MACOS)
NAMES = {WINDOWS: "Windows", LINUX: "Linux", MACOS: "macOS"}


def _detect() -> str:
    if sys.platform == "win32":
        return WINDOWS
    if sys.platform.startswith("linux"):
        return LINUX
    if sys.platform == "darwin":
        return MACOS
    return OTHER


OS = _detect()
IS_WINDOWS = OS == WINDOWS
IS_LINUX = OS == LINUX
IS_MACOS = OS == MACOS
IS_SUPPORTED = OS in SUPPORTED


def os_name() -> str:
    """The system's name, for messages."""
    return NAMES.get(OS, sys.platform)


def is_elevated() -> bool:
    """Running as administrator (Windows) / root (elsewhere)."""
    if IS_WINDOWS:
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid is not None and geteuid() == 0)


def prepare_app_process(app_id: str) -> None:
    """Windows: the taskbar groups the app's windows under ``app_id``, and the system timer runs at 1 ms (until
    the app exits) for steady playback timing. Elsewhere nothing is needed: the timers are fine-grained already."""
    if not IS_WINDOWS:
        return
    import atexit
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        atexit.register(lambda: ctypes.windll.winmm.timeEndPeriod(1))
    except Exception:
        pass


def register_audio_thread() -> bool:
    """The calling thread as a real-time audio thread: on Windows registered with MMCSS as "Pro Audio" (the
    scheduler serves it ahead of busy threads, as the audio engine's own); True if done. Elsewhere nothing (the
    thread's Qt priority applies; real-time scheduling would need privileges the app does not have)."""
    if not IS_WINDOWS:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        avrt = ctypes.WinDLL("avrt")
        avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        task_index = wintypes.DWORD(0)
        return bool(avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task_index)))
    except Exception:
        return False


def lower_process_priority() -> None:
    """The calling process below the others (a background job: the GUI and the audio go first): Windows' below
    normal priority class, elsewhere a nice of 10."""
    try:
        if IS_WINDOWS:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(10)
    except Exception:
        pass


def hidden_subprocess_kwargs() -> dict:
    """Popen keyword arguments that keep a console program's window from opening (Windows); none elsewhere."""
    if not IS_WINDOWS:
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0), "startupinfo": startupinfo}


def _windows_program(*parts: str) -> str:
    """A Windows program by its full path under the Windows folder: a bare name is looked for in the current
    folder first, where another program of that name could be."""
    return os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", *parts)


# Windows virtual keys of the named keys in Qt's portable key text (characters are looked up in the layout)
_VK = {"Left": 0x25, "Up": 0x26, "Right": 0x27, "Down": 0x28, "Space": 0x20, "Esc": 0x1B, "Tab": 0x09,
       "Backspace": 0x08, "Return": 0x0D, "Enter": 0x0D, "Ins": 0x2D, "Del": 0x2E, "Home": 0x24, "End": 0x23,
       "PgUp": 0x21, "PgDown": 0x22, **{f"F{n}": 0x6F + n for n in range(1, 25)}}
_HOTKEY_MODS = {"Alt": 0x1, "Ctrl": 0x2, "Shift": 0x4, "Meta": 0x8}
_HOTKEY_PROBE_ID = 0xB7A1


def hotkey_taken(key: str) -> bool:
    """Whether another program holds ``key`` (one combination in Qt's portable text, "Ctrl+Alt+R") as a global
    hotkey, so its presses never reach the app (a graphics card overlay holds Alt+R, for one): Windows, where the key
    is registered and released at once and refused when held. False elsewhere (no such check) and for a key not
    understood."""
    if not IS_WINDOWS or not key:
        return False
    parts = key.split("+")
    name = parts[-1] or "+"   # "Ctrl++": the plus key
    mods = 0
    for part in parts[:-1] if parts[-1] else parts[:-2]:
        if part not in _HOTKEY_MODS:
            return False
        mods |= _HOTKEY_MODS[part]
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    vk = _VK.get(name)
    if vk is None:
        if len(name) != 1:
            return False
        user32.VkKeyScanW.argtypes = [ctypes.c_wchar]
        user32.VkKeyScanW.restype = ctypes.c_short
        scan = user32.VkKeyScanW(name)
        if scan == -1:
            return False
        vk = scan & 0xFF
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    if user32.RegisterHotKey(None, _HOTKEY_PROBE_ID, mods | 0x4000, vk):   # MOD_NOREPEAT
        user32.UnregisterHotKey(None, _HOTKEY_PROBE_ID)
        return False
    return ctypes.get_last_error() == 1409   # ERROR_HOTKEY_ALREADY_REGISTERED


WINDOWS_MAX_PATH = 260     # MAX_PATH: a path this long or longer opens only with long paths on


def path_key(path: str | os.PathLike) -> str:
    """The path spelled one way for comparing it with another: absolute, its separators and dots normalized, its
    case folded on Windows (os.path.normcase), its Unicode composed (NFC) on macOS, whose file systems take a name's
    decomposed and composed forms as the same name (HFS+ and Finder give NFD: a Japanese name's dakuten, a Korean
    one's syllables); not on Windows and Linux, where they are two names. Links are not followed."""
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    return unicodedata.normalize("NFC", key) if IS_MACOS else key


@lru_cache(maxsize=1)
def long_paths_enabled() -> bool:
    """This process opens paths of WINDOWS_MAX_PATH characters or more: on Windows when LongPathsEnabled is set (the
    program declares itself long path aware: Python's and the package's PyInstaller program do), checked by
    RtlAreLongPathsEnabled; always elsewhere."""
    if not IS_WINDOWS:
        return True
    try:
        import ctypes
        check = ctypes.windll.ntdll.RtlAreLongPathsEnabled
        check.restype = ctypes.c_ubyte      # BOOLEAN
        return bool(check())
    except Exception:
        return False


def path_too_long(path: str | os.PathLike) -> bool:
    """``path`` cannot be opened here: WINDOWS_MAX_PATH characters or more on Windows with long paths off."""
    return IS_WINDOWS and len(os.path.abspath(os.fspath(path))) >= WINDOWS_MAX_PATH and not long_paths_enabled()


_DATALESS = 0x40000000                 # macOS st_flags SF_DATALESS: a file whose data is in the cloud
# Windows: FILE_ATTRIBUTE_OFFLINE, _RECALL_ON_OPEN, _RECALL_ON_DATA_ACCESS: a placeholder its sync client downloads
_ONLINE_ONLY = 0x00001000 | 0x00040000 | 0x00400000
# folders a sync client makes, by their names: Google Drive for desktop's drive (its own drive letter, not a sync
# root Windows knows), and the clients' default folders elsewhere
_SYNCED_NAMES = {"My Drive": "Google Drive", "내 드라이브": "Google Drive", "マイドライブ": "Google Drive",
                 "Shared drives": "Google Drive", "공유 드라이브": "Google Drive", "共有ドライブ": "Google Drive"}


def _windows_sync_root(path: str) -> str:
    """The provider of the Windows sync root holding ``path`` (OneDrive, iCloud, Dropbox ...: the cloud files API,
    CfGetSyncRootInfoByPath), "" for none."""
    import ctypes
    from ctypes import wintypes
    try:
        cldapi = ctypes.WinDLL("cldapi")
    except OSError:
        return ""
    call = cldapi.CfGetSyncRootInfoByPath
    call.argtypes = [wintypes.LPCWSTR, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    call.restype = ctypes.c_long

    class ProviderInfo(ctypes.Structure):    # CF_SYNC_ROOT_PROVIDER_INFO
        _fields_ = [("status", wintypes.ULONG), ("name", ctypes.c_wchar * 256), ("version", ctypes.c_wchar * 256)]

    info, size = ProviderInfo(), wintypes.DWORD(0)
    probe = path
    while probe and not os.path.exists(probe):      # a folder not made yet: its nearest made parent
        parent = os.path.dirname(probe)
        probe = "" if parent == probe else parent
    if not probe or call(probe, 2, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(size)) != 0:   # PROVIDER
        return ""
    return info.name.strip() or "a sync client"


def _known_sync_roots() -> list[tuple[str, str]]:
    """(folder, service) of the sync clients' folders found by their settings and default places."""
    home = os.path.expanduser("~")
    roots = []
    if IS_WINDOWS:
        roots += [(os.environ[v], "OneDrive") for v in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
                  if os.environ.get(v)]
        roots.append((os.path.join(home, "iCloudDrive"), "iCloud Drive"))
        dropbox_infos = [os.path.join(os.environ.get(v, ""), "Dropbox", "info.json") for v in ("APPDATA", "LOCALAPPDATA")]
    else:
        dropbox_infos = [os.path.join(home, ".dropbox", "info.json")]
    if IS_MACOS:
        roots.append((os.path.join(home, "Library", "Mobile Documents"), "iCloud Drive"))
        cloud = os.path.join(home, "Library", "CloudStorage")      # File Provider: OneDrive-…, GoogleDrive-…, Box-…
        try:
            roots += [(os.path.join(cloud, name), name.split("-", 1)[0]) for name in os.listdir(cloud)]
        except OSError:
            pass
    for info in dropbox_infos:
        try:
            import json
            with open(info, encoding="utf-8") as f:
                roots += [(account["path"], "Dropbox") for account in json.load(f).values()
                          if isinstance(account, dict) and account.get("path")]
        except (OSError, ValueError, AttributeError):
            pass
    roots += [(os.path.join(home, name), name) for name in ("Dropbox", "OneDrive", "Box")]
    return roots


def cloud_sync_folder(path: str | os.PathLike) -> str:
    """The sync service whose folder holds ``path`` ("OneDrive", "Dropbox", "iCloud Drive", "Google Drive", ...),
    "" for none: Windows' sync roots first, then the clients' known folders, then a folder named as one."""
    key = path_key(path)
    if IS_WINDOWS:
        found = _windows_sync_root(os.path.abspath(os.fspath(path)))
        if found:
            return found
    for root, service in _known_sync_roots():
        root_key = path_key(root)
        if key == root_key or key.startswith(root_key.rstrip(os.sep) + os.sep):
            return service
    parts = os.path.abspath(os.fspath(path)).replace("\\", "/").split("/")
    return next((_SYNCED_NAMES[p] for p in parts if p in _SYNCED_NAMES), "")


def online_only(path: str | os.PathLike) -> bool:
    """The file's data is not on this device (a sync client's placeholder: opening it downloads it first). False
    when it cannot be told (Linux) or the file is not there."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return False
    if IS_WINDOWS:
        return bool(getattr(st, "st_file_attributes", 0) & _ONLINE_ONLY)
    if IS_MACOS:
        return bool(getattr(st, "st_flags", 0) & _DATALESS)
    return False


def reveal_file(path: str) -> None:
    """Show the file in the system's file manager: selected in Explorer / Finder, its folder elsewhere."""
    if IS_WINDOWS:
        subprocess.Popen([_windows_program("explorer.exe"), "/select,", os.path.normpath(path)])
    elif IS_MACOS:
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path) or "."])
