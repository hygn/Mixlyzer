"""The operating system, checked once per process (this module's first import: app.main at the start, and each
spawned worker), and the operations that differ by it. Windows, Linux and macOS are supported; on any other system
the app warns at the start and runs on the fallbacks. Code branches on the flags here (OS, IS_WINDOWS, IS_LINUX,
IS_MACOS, MEMORY_SYNC), never on sys.platform / os.name / platform.system() itself.

Where the Windows API is used, the other systems have a fallback:
- elevated process (app.main refuses it): IsUserAnAdmin / effective uid 0 (root)
- the app's taskbar identity and 1 ms timer resolution (app.window): Windows only; the others need neither
- console windows of child programs (FFmpeg, tasklist) kept hidden: Windows only; the others open none
- the running processes: tasklist / ps
- revealing a file: Explorer, the file selected / Finder, the file selected / the folder in the file manager
- the window's occlusion (utils.window_visibility): user32 + DWM / whether Qt has the window exposed
- the scratch cursor poller (core.audio.cursor_poller): Win32 cursor + waitable timer / the GUI's mouse events
- the physical core count for sklearn (analyzer_core.utils): GetLogicalProcessorInformationEx / loky's own
- a key another program holds as a global hotkey (Settings > Shortcuts): RegisterHotKey refused / not checked
External Sync's memory reads (PyMemoryEditor) are Windows and Linux only: on macOS (MEMORY_SYNC False) the library
is not imported, the settings tab is not shown and the setting is always off.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys

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
# External Sync reads another program's memory (PyMemoryEditor): Windows and Linux
MEMORY_SYNC = OS in (WINDOWS, LINUX)


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


def hidden_subprocess_kwargs() -> dict:
    """Popen keyword arguments that keep a console program's window from opening (Windows); none elsewhere."""
    if not IS_WINDOWS:
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0), "startupinfo": startupinfo}


def list_processes() -> list[dict[str, int | str]]:
    """The running processes as {"name", "pid"} (tasklist on Windows, ps elsewhere); [] when it cannot be read."""
    command = ["tasklist", "/FO", "CSV", "/NH"] if IS_WINDOWS else ["ps", "-A", "-o", "pid=,comm="]
    try:
        out = subprocess.check_output(command, text=True, encoding="utf-8", errors="replace",
                                      **hidden_subprocess_kwargs())
    except Exception:
        return []
    rows: list[dict[str, int | str]] = []
    if IS_WINDOWS:
        for row in csv.reader(io.StringIO(out)):
            if len(row) < 2:
                continue
            try:
                pid = int(str(row[1]).strip())
            except ValueError:
                continue
            name = str(row[0]).strip()
            if name:
                rows.append({"name": name, "pid": pid})
    else:
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            # ps gives the command's path on some systems: its file name is the process name
            name = os.path.basename(parts[1].strip())
            if name:
                rows.append({"name": name, "pid": pid})
    return rows


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


def reveal_file(path: str) -> None:
    """Show the file in the system's file manager: selected in Explorer / Finder, its folder elsewhere."""
    if IS_WINDOWS:
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif IS_MACOS:
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path) or "."])
