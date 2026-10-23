"""The startup splash's console: the lines the app prints while it starts (its own "[Startup] ...", the audio
engine's "[AudioEngine] ...", warnings), drawn as a terminal's last lines: the time since the start, the line's
source as a coloured tag, its text (a finished task with its time right-aligned, a failure in the danger colour).

``ConsoleTap`` wraps sys.stdout / sys.stderr (the print hook, installed later, writes through it), so the lines go
on to the console and the log as before; lines printed on other threads (the engine's console relay) are queued
and drawn at the splash's next repaint. Kept light: PySide6 and the theme only (the startup imports it first)."""
from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme

LINES_KEPT = 200
ROWS = 9

# a line's source by its "[Tag]": its tag's text and colour (a theme token)
_TAGS = {"Startup": ("START", "accent"), "AudioEngine": ("AUDIO", "success"), "Player": ("PLAYER", "warning")}
_TAG_RE = re.compile(r"^\[([A-Za-z][\w ]{0,23})\]\s*(.*)$")
_DONE_RE = re.compile(r"^(.+): (\d+\.\d+) s$")   # run_startup's "[Startup] <task>: 1.23 s"
_FAILED_RE = re.compile(r"\b(failed|error|refused|did not|not opened|cannot|unreadable|not keeping pace)\b",
                        re.IGNORECASE)


@dataclass(frozen=True)
class ConsoleLine:
    at: float           # seconds since the console started
    tag: str            # "" none
    text: str
    duration: str = ""  # a finished task's time ("1.23 s")
    failed: bool = False


class ConsoleTap:
    """A stream that passes everything on to ``stream`` (None: a windowed app has none) and keeps the whole lines
    written for the splash. Any thread."""

    def __init__(self, stream, console: "ConsoleLines", error: bool = False) -> None:
        self._stream = stream
        self._console = console
        self._error = error
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if self._stream is not None:
            try:
                self._stream.write(text)
            except Exception:
                pass
        with self._lock:
            self._partial += text
            *lines, self._partial = self._partial.split("\n")
        for line in lines:
            self._console.add(line, self._error)
        return len(text)

    def flush(self) -> None:
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def __getattr__(self, name):   # encoding, isatty, fileno ...: the wrapped stream's
        if self._stream is None:
            raise AttributeError(name)
        return getattr(self._stream, name)


class ConsoleLines:
    """The lines kept (the last LINES_KEPT), parsed; ``changed`` set when one came."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._lines: deque[ConsoleLine] = deque(maxlen=LINES_KEPT)
        self._lock = threading.Lock()
        self.changed = threading.Event()
        self._taps: list[tuple[str, object]] = []

    def install(self) -> None:
        """sys.stdout and sys.stderr through taps (taken out again by ``uninstall``)."""
        for name, error in (("stdout", False), ("stderr", True)):
            stream = getattr(sys, name)
            self._taps.append((name, stream))
            setattr(sys, name, ConsoleTap(stream, self, error))

    def uninstall(self) -> None:
        """The taps out of the way: a stream wrapped since (the print hook) keeps writing through its tap, which now
        only passes the text on."""
        for name, stream in self._taps:
            tap = getattr(sys, name)
            if isinstance(tap, ConsoleTap) and tap._console is self:
                setattr(sys, name, stream)
        self._taps.clear()
        self.add = lambda _text, _error=False: None   # the taps still wrapped: nothing kept any more

    def add(self, text: str, error: bool = False) -> None:
        text = text.rstrip()
        if not text.strip():
            return
        tag, body = "", text.strip()
        m = _TAG_RE.match(body)
        if m:
            tag, body = m.group(1), m.group(2)
        duration = ""
        if tag == "Startup":
            done = _DONE_RE.match(body)
            if done:
                body, duration = done.group(1), f"{done.group(2)} s"
        failed = bool(_FAILED_RE.search(body)) or (error and not tag)
        line = ConsoleLine(time.perf_counter() - self._t0, tag, body, duration, failed)
        with self._lock:
            self._lines.append(line)
        self.changed.set()

    def last(self, n: int) -> list[ConsoleLine]:
        with self._lock:
            return list(self._lines)[-n:]


class StartupConsole(QtWidgets.QWidget):
    """The last ROWS lines, painted (no layout request per line: its size is fixed)."""

    PAD_X, PAD_Y, ROW_GAP = 12, 10, 3

    def __init__(self, lines: ConsoleLines, parent=None) -> None:
        super().__init__(parent)
        self._lines = lines
        # the text in the app's font (its Korean / Japanese glyphs its own, not a fixed font's fallback), the times
        # in a fixed one (their digits in columns)
        self._font = QtGui.QFont(QtWidgets.QApplication.font())
        self._font.setPointSizeF(8.5)
        self._time_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self._time_font.setPointSizeF(8.0)
        self._tag_font = QtGui.QFont(self._font)
        self._tag_font.setPointSizeF(7.0)
        self._tag_font.setBold(True)
        self._tag_font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 0.6)
        fm = QtGui.QFontMetrics(self._font)
        self._row_h = fm.height() + self.ROW_GAP
        self._time_w = QtGui.QFontMetrics(self._time_font).horizontalAdvance("+00.00")
        tag_fm = QtGui.QFontMetrics(self._tag_font)
        self._tag_w = max(tag_fm.horizontalAdvance(t) for t, _c in _TAGS.values()) + 12
        self.setFixedHeight(2 * self.PAD_Y + ROWS * self._row_h - self.ROW_GAP)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, False)

    def refresh(self) -> None:
        """Repainted when a line came (the splash calls this at its updates)."""
        if self._lines.changed.is_set():
            self._lines.changed.clear()
            self.update()

    def paintEvent(self, _event) -> None:
        theme = current_theme()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QtGui.QPen(QtGui.QColor(theme.border), 1.0))
        p.setBrush(QtGui.QColor(theme.base))
        p.drawRoundedRect(rect, 8, 8)

        lines = self._lines.last(ROWS)
        fm = QtGui.QFontMetrics(self._font)
        x0 = self.PAD_X
        x_tag = x0 + self._time_w + 10
        x_text = x_tag + self._tag_w + 8
        right = self.width() - self.PAD_X
        # the newest at the bottom; the older the fainter
        first_row = ROWS - len(lines)
        for k, line in enumerate(lines):
            row = first_row + k
            top = self.PAD_Y + row * self._row_h
            mid = top + (self._row_h - self.ROW_GAP) / 2
            base = top + fm.ascent()
            age = len(lines) - 1 - k
            p.setOpacity(max(0.4, 1.0 - 0.09 * age))

            p.setFont(self._time_font)
            p.setPen(QtGui.QColor(theme.text_disabled))
            p.drawText(QtCore.QRectF(x0, top, self._time_w, fm.height()), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                       f"+{line.at:.2f}")

            if line.tag:
                label, token = _TAGS.get(line.tag, (line.tag.upper()[:6], "text_dim"))
                color = QtGui.QColor(getattr(theme, token))
                pill = QtCore.QRectF(x_tag, mid - 7.5, self._tag_w, 15)
                fill = QtGui.QColor(color)
                fill.setAlphaF(0.16)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(fill)
                p.drawRoundedRect(pill, 7.5, 7.5)
                p.setFont(self._tag_font)
                p.setPen(color)
                p.drawText(pill, QtCore.Qt.AlignCenter, label)

            p.setFont(self._font)
            text_right = right
            if line.duration:
                p.setPen(QtGui.QColor(theme.text_dim))
                w = fm.horizontalAdvance(line.duration)
                p.drawText(QtCore.QPointF(right - w, base), line.duration)
                text_right = right - w - 10
            x = x_text
            if line.duration:   # a finished task: a check mark before it, a leader to its time
                p.setPen(QtGui.QColor(theme.success))
                p.drawText(QtCore.QPointF(x_text, base), "✓")
                x += fm.horizontalAdvance("✓ ")
            color = theme.danger if line.failed else (theme.text if age == 0 else theme.text_dim)
            p.setPen(QtGui.QColor(color))
            shown = fm.elidedText(line.text, QtCore.Qt.ElideRight, max(0, int(text_right - x)))
            p.drawText(QtCore.QPointF(x, base), shown)
            if line.duration:
                # a dotted leader from the title to its time
                end = x + fm.horizontalAdvance(shown) + 8
                if text_right - end > 12:
                    pen = QtGui.QPen(QtGui.QColor(theme.text_disabled), 1.0, QtCore.Qt.DotLine)
                    p.setPen(pen)
                    p.drawLine(QtCore.QPointF(end, base - 3), QtCore.QPointF(text_right, base - 3))
        if not lines:
            p.setOpacity(0.6)
            p.setPen(QtGui.QColor(theme.text_disabled))
            p.setFont(self._font)
            p.drawText(QtCore.QPointF(x0, self.PAD_Y + fm.ascent()), "▍")
        p.end()
