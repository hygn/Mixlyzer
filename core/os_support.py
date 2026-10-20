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


def reveal_file(path: str) -> None:
    """Show the file in the system's file manager: selected in Explorer / Finder, its folder elsewhere."""
    if IS_WINDOWS:
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    elif IS_MACOS:
        subprocess.Popen(["open", "-R", path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path) or "."])
