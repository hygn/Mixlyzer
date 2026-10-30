"""The diagnostics (the menu's Diagnostics window: read there, then copied or saved by the user, never sent by the
app): what a report of a playback, device or decoding problem needs, as plain text. The app's and the system's
versions, FFmpeg's, the audio setup (app.audio_report's lines) with the output devices and MIDI inputs the engine
lists, each output's buffer since the start (``BufferRecord``: the time it played, its dropouts, its last state),
and the console's last lines (``CONSOLE``: the app's and the audio engine's, recorded from main()'s start whether
the log file is written or not). The user's home folder is written ~ and their account's name <user>.

Kept light at import (main() installs CONSOLE first, and a spawned worker imports the main module again): the
audio and the engine's modules are imported by ``report``. Its words are not translated (as the logs)."""
from __future__ import annotations

import getpass
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

from app.startup_console import ConsoleTap
from system import os_support
from system.version import APP_VERSION

CONSOLE_LINES = 3000


class ConsoleRecord:
    """The console's last CONSOLE_LINES lines with their times, through taps on sys.stdout / sys.stderr (main()
    installs them before the splash's: every stream wrapped since writes through them). Any thread."""

    def __init__(self, kept: int = CONSOLE_LINES) -> None:
        self._lines: deque[tuple[float, bool, str]] = deque(maxlen=kept)
        self._lock = threading.Lock()

    def install(self) -> None:
        sys.stdout = ConsoleTap(sys.stdout, self)
        sys.stderr = ConsoleTap(sys.stderr, self, error=True)

    def add(self, text: str, error: bool = False) -> None:
        text = text.rstrip()
        if text.strip():
            with self._lock:
                self._lines.append((time.time(), error, text))

    def lines(self) -> list[str]:
        """Each line with its time; "!" marks an error stream's."""
        with self._lock:
            kept = list(self._lines)
        return [f"{datetime.fromtimestamp(at):%H:%M:%S} {'!' if error else ' '} {text}" for at, error, text in kept]


CONSOLE = ConsoleRecord()

# an output's state as reported, without the observations' arrays and sums (BufferRecord sums those itself)
_NOT_STATE = {"hist", "observed", "outputs", "seconds", "mean_ms", "dropout_sec"}


class BufferRecord:
    """Each output's buffer since the start, from the player's buffer reports (``add``: EventBus.sig_output_buffer,
    every 0.25 s, on the GUI thread): the time it played and its dropouts summed, its state (device, API, period,
    latency, late wake-ups, longest gap, lowest level, threads) as last reported."""

    def __init__(self) -> None:
        self.outputs: dict[str, dict] = {}

    def add(self, _queued_ms: float, stats, capacity_ms: float) -> None:
        if not stats:
            return
        if "seconds" in stats:
            self._add("master", stats, stats, capacity_ms)
        for bus, state in (stats.get("outputs") or {}).items():
            if state is not None:
                self._add(bus, state, state.get("observed"), float(state.get("capacity_ms", 0.0)))

    def _add(self, name: str, state: dict, observed: dict | None, capacity_ms: float) -> None:
        out = self.outputs.setdefault(name, {"played_sec": 0.0, "dropout_sec": 0.0})
        out["state"] = {k: v for k, v in state.items() if k not in _NOT_STATE}
        out["capacity_ms"] = capacity_ms
        if observed:
            out["played_sec"] += float(observed.get("seconds", 0.0))
            out["dropout_sec"] += float(observed.get("dropout_sec", 0.0))

    def lines(self) -> list[str]:
        if not self.outputs:
            return ["no buffer reported yet (nothing played)"]
        out = []
        for name, d in self.outputs.items():
            out.append(f"{name}: played {d['played_sec']:.1f} s, dropouts {d['dropout_sec'] * 1000:.1f} ms, "
                       f"buffer {d['capacity_ms']:.1f} ms")
            out += [f"  {k}: {_value(v)}" for k, v in sorted(d["state"].items())]
        return out


def _value(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def _run(args: list[str]) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, timeout=10, **os_support.hidden_subprocess_kwargs())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"not run: {exc}"
    return proc.stdout.decode("utf-8", "replace")


def redact(text: str) -> str:
    """The user's home folder as ~, their account's name as <user> (a device named after them, a path outside
    the home folder)."""
    home = os.path.expanduser("~")
    for form in {home, home.replace("\\", "/")}:
        if len(form) > 3:
            text = re.sub(re.escape(form), "~", text, flags=re.IGNORECASE)
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if len(user) >= 3:
        text = re.sub(rf"(?<![\w]){re.escape(user)}(?![\w])", "<user>", text, flags=re.IGNORECASE)
    return text


def report(cfg, player, buffers: BufferRecord, console: ConsoleRecord = CONSOLE) -> str:
    """The diagnostics now: ``cfg`` the settings in use, ``player`` the PlayerController, ``buffers`` what it
    reported since the start."""
    import numpy as np
    import PySide6
    from PySide6 import QtCore

    from analyzer.audio.decoder import _ffmpeg_command
    from app.audio_report import audio_setup_lines
    from system import i18n

    ffmpeg = _ffmpeg_command()
    ffmpeg_version = _run([ffmpeg, "-hide_banner", "-version"]).splitlines()
    current, devices, channels = player.output_devices()
    midi = player.midi_ports()
    sections = [
        ("Mixlyzer", [f"version {APP_VERSION} ({'package' if getattr(sys, 'frozen', False) else 'source'})",
                      f"made {datetime.now():%Y-%m-%d %H:%M:%S}", f"interface language {i18n.language()}",
                      f"engine {'native' if player.native else 'Python'}"]),
        ("System", [f"{os_support.os_name()}: {platform.platform()}", f"machine {platform.machine()}, processor "
                    f"{platform.processor() or 'unknown'}, {os.cpu_count()} logical processors",
                    f"Python {sys.version.split()[0]}, Qt {QtCore.qVersion()}, PySide6 {PySide6.__version__}, "
                    f"NumPy {np.__version__}",
                    "long paths " + ("on" if os_support.long_paths_enabled() else "off (Windows' 260 characters)")]),
        ("FFmpeg", [ffmpeg] + ffmpeg_version[:1]),
        ("Audio setup", audio_setup_lines(cfg, player.native)),
        ("Output devices", [f"{'*' if name == current else ' '} {name or 'the default device'}: "
                            f"{channels.get(name, '?')} channels" for name in devices] or ["none listed"]),
        ("MIDI inputs", midi or ["none"]),
        ("Output buffers since the start", buffers.lines()),
    ]
    lines = console.lines()
    sections.append((f"Console (last {len(lines)} lines)", lines))
    return redact("\n\n".join(f"== {title}\n" + "\n".join(body) for title, body in sections) + "\n")
