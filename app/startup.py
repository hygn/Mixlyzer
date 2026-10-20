"""Startup: the slow work done before the main window opens -- the libraries imported, the audio
kernels' JIT, FFmpeg's first run, the search readers loaded, the library's search keys cached --
then the window built, behind a splash screen showing the task (what it is doing, its progress)
and the overall progress.

Kept light (PySide6 and the theme only): a spawned worker process imports the app's main module,
and with it this one."""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme

# fraction of the task done, what it is doing ("" keeps the last)
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class StartupTask:
    title: str
    weight: float                        # its share of the overall progress (seconds, typical)
    run: Callable[[Progress], Any]
    # the splash takes the app's events between updates (repainted only while the window is being
    # built: an event then could reach the half-built window)
    pump_events: bool = True


class StartupSplash(QtWidgets.QWidget):
    """The task (title, what it is doing, its progress bar) and the overall progress bar below it."""

    _PAINT_INTERVAL = 0.03   # s between repaints in a fast loop

    def __init__(self) -> None:
        super().__init__(None, QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint)
        theme = current_theme()
        self.setObjectName("startupSplash")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"QWidget#startupSplash {{ background: {theme.window}; "
                           f"border: 1px solid {theme.border}; }}")
        self.setWindowTitle("Mixlyzer")
        self.setWindowIcon(QtGui.QIcon("assets/images/mixlyzer.png"))

        icon = QtWidgets.QLabel()
        icon.setPixmap(QtGui.QPixmap("assets/images/mixlyzer.png").scaled(
            48, 48, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        name = QtWidgets.QLabel("Mixlyzer")
        font = name.font()
        font.setPointSizeF(font.pointSizeF() * 1.8)
        font.setBold(True)
        name.setFont(font)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(icon)
        header.addSpacing(8)
        header.addWidget(name)
        header.addStretch(1)

        self.lbl_task = QtWidgets.QLabel("Starting")
        self.lbl_detail = QtWidgets.QLabel("")
        self.lbl_detail.setStyleSheet(f"color: {theme.text_dim};")
        # the task's progress above the overall progress
        self.bar_task = self._bar()
        self.bar_overall = self._bar()
        bars = QtWidgets.QVBoxLayout()
        bars.addWidget(self.bar_task)
        bars.addWidget(self.bar_overall)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.addLayout(header)
        layout.addSpacing(10)
        layout.addWidget(self.lbl_task)
        layout.addWidget(self.lbl_detail)
        layout.addSpacing(6)
        layout.addLayout(bars)
        self.setFixedWidth(520)
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            self.move(screen.availableGeometry().center() - self.rect().center())
        self.pump_events = True
        self._painted = 0.0

    @staticmethod
    def _bar() -> QtWidgets.QProgressBar:
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 1000)
        bar.setFormat("%p%")
        bar.setAlignment(QtCore.Qt.AlignCenter)
        return bar

    def set_task(self, title: str) -> None:
        self.lbl_task.setText(title)
        self.lbl_detail.setText("")
        self._paint(force=True)

    def set_progress(self, task: float, overall: float, detail: str = "") -> None:
        self.bar_task.setValue(round(1000 * min(max(task, 0.0), 1.0)))
        self.bar_overall.setValue(round(1000 * min(max(overall, 0.0), 1.0)))
        if detail:
            width = self.lbl_detail.width() or self.width() - 40
            self.lbl_detail.setText(self.lbl_detail.fontMetrics().elidedText(detail, QtCore.Qt.ElideMiddle, width))
        self._paint()

    def _paint(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._painted < self._PAINT_INTERVAL:
            return
        self._painted = now
        if self.pump_events:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        else:
            QtCore.QCoreApplication.sendPostedEvents(self, 0)   # its own layout / update requests only
            self.repaint()


def run_startup(splash: StartupSplash, tasks: list[StartupTask]) -> list:
    """Run the tasks in order on the splash; their results."""
    total = sum(t.weight for t in tasks) or 1.0
    done, results = 0.0, []
    for task in tasks:
        splash.pump_events = task.pump_events
        splash.set_task(task.title)

        def progress(fraction: float, detail: str = "", task: StartupTask = task, done: float = done) -> None:
            fraction = min(max(float(fraction), 0.0), 1.0)
            splash.set_progress(fraction, (done + task.weight * fraction) / total, detail)

        progress(0.0)
        t0 = time.perf_counter()
        results.append(task.run(progress))
        print(f"[Startup] {task.title}: {time.perf_counter() - t0:.2f} s")
        done += task.weight
        progress(1.0)
    return results


# the app's heavy imports in dependency order, with their usual time (s, measured)
_IMPORTS = (
    ("numpy", 0.17), ("scipy.signal", 1.17), ("numba", 0.36), ("sklearn.metrics", 0.31), ("pyqtgraph", 0.34),
    ("analyzer_core.global_analyzer", 0.36), ("core.player", 0.05), ("ui.pane", 0.08), ("ui.config.window", 0.05),
    ("app.window", 0.09),
)


def _import_libraries(progress: Progress) -> None:
    total = sum(w for _m, w in _IMPORTS)
    done = 0.0
    for module, weight in _IMPORTS:
        progress(done / total, module)
        importlib.import_module(module)
        done += weight


def _compile_audio_kernels(progress: Progress) -> None:
    """The playback's numba kernels: from numba's disk cache, compiled when it has none (the first
    start after an install or a numba update takes longer)."""
    from core.audio.cursor_poller import warm_cursor_poller
    from core.audio.dsp import warm_scratch_numba
    from core.audio.timestretch import warm_timestretch_numba
    kernels = (("Scratch reader", 0.7, warm_scratch_numba), ("Cursor poller", 0.05, warm_cursor_poller),
               ("Time stretch", 0.25, warm_timestretch_numba))
    done = 0.0
    for name, weight, warm in kernels:
        progress(done, name)
        warm()
        done += weight


def _warm_ffmpeg(progress: Progress) -> None:
    from core.audio.decoder import warm_ffmpeg_decoder
    progress(0.0, "ffmpeg: a short silent decode")
    warm_ffmpeg_decoder()


def _load_readers(progress: Progress) -> None:
    from core import reading
    reading.warm_up(progress)


def _search_keys(libpath: str) -> Callable[[Progress], None]:
    def run(progress: Progress) -> None:
        """Every stored reading's search keys into reading.keys' cache, so the library opens and
        searches without building them."""
        from core import reading
        from core.library_handler import LibraryDB
        progress(0.0, "library.db: readings")
        db = LibraryDB(os.path.join(libpath, "library.db"))
        db.connect()
        try:
            stored = db.all_readings()
        finally:
            db.close()
        texts = [text for fields in stored.values() for texts in fields.values() for text in texts]
        for i, text in enumerate(texts):
            if i % 256 == 0:
                progress(0.1 + 0.9 * i / len(texts), f"Search keys: {len(stored):,} tracks, {i:,} / {len(texts):,} readings")
            reading.keys(text)
    return run


def _build_window(progress: Progress):
    from app.window import AppWindow
    return AppWindow(progress=progress)


def startup_tasks(cfg) -> list[StartupTask]:
    """What the app does before its window opens, in order; the last gives the window."""
    # weights: the measured times (s) of a start with the files in the OS cache
    return [
        StartupTask("Loading libraries", 3.0, _import_libraries),
        StartupTask("Compiling audio kernels (JIT)", 0.4, _compile_audio_kernels),
        StartupTask("Starting FFmpeg", 0.1, _warm_ffmpeg),
        StartupTask("Loading search readers", 2.1, _load_readers),
        StartupTask("Loading the library", 0.1, _search_keys(cfg.libconfig.libpath)),
        StartupTask("Building the window", 2.3, _build_window, pump_events=False),
    ]
