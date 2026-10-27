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

from app.startup_console import ConsoleLines, StartupConsole
from ui.theme import current_theme
from system.i18n import tr
from system.resource_paths import resource_path

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


def launch_screen() -> QtGui.QScreen | None:
    """The screen the app starts on: the one under the mouse (where it was launched from), else the primary."""
    return QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos()) or QtGui.QGuiApplication.primaryScreen()


def center_on_screen(widget: QtWidgets.QWidget, screen: QtGui.QScreen | None) -> None:
    """``widget`` centred on ``screen``'s free area (its top-left corner kept in it when it is larger)."""
    if screen is None:
        return
    area = screen.availableGeometry()
    size = widget.frameGeometry().size()
    widget.move(area.x() + max(0, (area.width() - size.width()) // 2),
                area.y() + max(0, (area.height() - size.height()) // 2))


class StartupSplash(QtWidgets.QWidget):
    """The task (title, what it is doing, its progress bar), the overall progress bar below it and, under them, the
    console's last lines (``lines``: app.startup_console, the app's and the audio engine's)."""

    _PAINT_INTERVAL = 0.03   # s between repaints in a fast loop

    def __init__(self, lines: ConsoleLines | None = None) -> None:
        super().__init__(None, QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint)
        theme = current_theme()
        self.setObjectName("startupSplash")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QWidget#startupSplash {{ background: {theme.window}; border: 1px solid {theme.border}; }}\n"
            f"QProgressBar {{ border: none; border-radius: 2px; background: {theme.raised}; }}\n"
            f"QProgressBar::chunk {{ border-radius: 2px; background: {theme.accent}; }}\n"
            f"QProgressBar#overall::chunk {{ background: {theme.success}; }}")
        self.setWindowTitle("Mixlyzer")
        self.setWindowIcon(QtGui.QIcon(str(resource_path("assets/images/mixlyzer.png"))))

        icon = QtWidgets.QLabel()
        icon.setPixmap(QtGui.QPixmap(str(resource_path("assets/images/mixlyzer.png"))).scaled(
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

        # the task's title with its percent; what it is doing; its bar; the overall bar with its percent
        self.lbl_task = QtWidgets.QLabel(tr("Starting"))
        bold = self.lbl_task.font()
        bold.setBold(True)
        self.lbl_task.setFont(bold)
        self.lbl_task_pct = self._percent()
        self.lbl_detail = QtWidgets.QLabel("")
        self.lbl_detail.setStyleSheet(f"color: {theme.text_dim};")
        self.bar_task = self._bar(4)
        self.bar_overall = self._bar(4)
        self.bar_overall.setObjectName("overall")
        self.lbl_overall = QtWidgets.QLabel(tr("Overall"))
        self.lbl_overall.setStyleSheet(f"color: {theme.text_dim};")
        self.lbl_overall_pct = self._percent()
        task_row = QtWidgets.QHBoxLayout()
        task_row.addWidget(self.lbl_task, 1)
        task_row.addWidget(self.lbl_task_pct)
        overall_row = QtWidgets.QHBoxLayout()
        overall_row.addWidget(self.lbl_overall, 1)
        overall_row.addWidget(self.lbl_overall_pct)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addSpacing(12)
        layout.addLayout(task_row)
        layout.addWidget(self.lbl_detail)
        layout.addSpacing(2)
        layout.addWidget(self.bar_task)
        layout.addSpacing(8)
        layout.addLayout(overall_row)
        layout.addSpacing(2)
        layout.addWidget(self.bar_overall)
        self.console = None
        if lines is not None:
            self.console = StartupConsole(lines)
            layout.addSpacing(14)
            layout.addWidget(self.console)
        self.setFixedWidth(720)
        self.adjustSize()
        center_on_screen(self, launch_screen())
        self.pump_events = True
        self._painted = 0.0

    @staticmethod
    def _bar(height: int) -> QtWidgets.QProgressBar:
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        bar.setFixedHeight(height)
        return bar

    @staticmethod
    def _percent() -> QtWidgets.QLabel:
        """A percent's label, fixed in size (it changes every repaint)."""
        label = QtWidgets.QLabel("0%")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setFixedSize(label.fontMetrics().horizontalAdvance("100%") + 4, label.fontMetrics().height())
        label.setStyleSheet(f"color: {current_theme().text_dim};")
        return label

    def set_task(self, title: str) -> None:
        self.lbl_task.setText(title)
        self.lbl_detail.setText("")
        self._paint(force=True)

    def set_progress(self, task: float, overall: float, detail: str = "") -> None:
        task, overall = min(max(task, 0.0), 1.0), min(max(overall, 0.0), 1.0)
        self.bar_task.setValue(round(1000 * task))
        self.bar_overall.setValue(round(1000 * overall))
        self.lbl_task_pct.setText(f"{100 * task:.0f}%")
        self.lbl_overall_pct.setText(f"{100 * overall:.0f}%")
        if detail:
            width = self.lbl_detail.width() or self.width() - 44
            self.lbl_detail.setText(self.lbl_detail.fontMetrics().elidedText(detail, QtCore.Qt.ElideMiddle, width))
        self._paint()

    def _paint(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._painted < self._PAINT_INTERVAL:
            return
        self._painted = now
        if self.console is not None:
            self.console.refresh()
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


# The GUI process does no heavy linear algebra (the analyses run in worker processes): numpy's and scipy's
# OpenBLAS, each set aside buffers for every thread they may use when loaded (about 1 GB of memory committed on 16
# threads), are given one. Set before they load (main(), this process only: a frozen app's workers run the entry
# script too), taken back once they have, so the worker processes, spawned with this environment, keep theirs.
_BLAS_THREADS = "OPENBLAS_NUM_THREADS"
_blas_threads_set = False


def one_blas_thread() -> None:
    """Before numpy and scipy are imported: their OpenBLAS with one thread in this process (the user's own
    setting kept)."""
    global _blas_threads_set
    if _BLAS_THREADS not in os.environ:
        os.environ[_BLAS_THREADS] = "1"
        _blas_threads_set = True


def _release_blas_threads() -> None:
    global _blas_threads_set
    if _blas_threads_set:
        os.environ.pop(_BLAS_THREADS, None)
        _blas_threads_set = False


# the app's heavy imports in dependency order, with their usual time (s, measured)
_IMPORTS = (
    ("numpy", 0.17), ("scipy.signal", 1.17), ("numba", 0.36), ("sklearn.metrics", 0.31), ("pyqtgraph", 0.34),
    ("analyzer.pipeline.track", 0.36), ("core.playback.player", 0.05), ("ui.pane", 0.08), ("ui.config.window", 0.05),
    ("app.window", 0.09),
)


def _import_libraries(progress: Progress) -> None:
    total = sum(w for _m, w in _IMPORTS)
    done = 0.0
    for module, weight in _IMPORTS:
        progress(done / total, module)
        importlib.import_module(module)
        done += weight
    _release_blas_threads()   # numpy's and scipy's OpenBLAS loaded


def _compile_audio_kernels(progress: Progress) -> None:
    """The playback's numba kernels: from numba's disk cache, compiled when it has none (the first
    start after an install or a numba update takes longer)."""
    from audioengine.python.cursor_poller import warm_cursor_poller
    from audioengine.python.dsp import warm_scratch_numba
    from audioengine.python.timestretch import warm_timestretch_numba
    kernels = ((tr("Scratch reader"), 0.7, warm_scratch_numba), (tr("Cursor poller"), 0.05, warm_cursor_poller),
               (tr("Time stretch"), 0.25, warm_timestretch_numba))
    done = 0.0
    for name, weight, warm in kernels:
        progress(done, name)
        warm()
        done += weight


def _warm_ffmpeg(progress: Progress) -> None:
    from analyzer.audio.decoder import warm_ffmpeg_decoder
    progress(0.0, tr("ffmpeg: a short silent decode"))
    warm_ffmpeg_decoder()


def _load_readers(progress: Progress) -> None:
    from core.search import reading
    reading.warm_up(progress)


def _open_library(libpath: str, held: dict) -> Callable[[Progress], None]:
    def run(progress: Progress) -> None:
        """The library's owner started (core.library.service: its thread reads the library), then every stored
        reading's search keys into reading.keys' cache, so the library opens and searches without building them.
        The window takes the owner (``held``)."""
        from core.search import reading
        from core.library.service import LibraryService
        from core.library import jobs
        progress(0.0, tr("library.db: readings"))
        library = LibraryService(libpath)
        library.write(jobs.check_stores)    # the stores beside library.db made to agree with it, first
        library.refresh()
        library.wait()
        held["library"] = library
        stored = library.snapshot.readings
        texts = [text for fields in stored.values() for texts in fields.values() for text in texts]
        for i, text in enumerate(texts):
            if i % 256 == 0:
                progress(0.1 + 0.9 * i / len(texts), tr("Search keys: {tracks} tracks, {done} / {total} readings").format(
                    tracks=f"{len(stored):,}", done=f"{i:,}", total=f"{len(texts):,}"))
            reading.keys(text)
    return run


def _build_window(held: dict, cfg) -> Callable[[Progress], object]:
    def run(progress: Progress):
        from app.window import AppWindow
        from core.settings import store
        settings = store.SettingsStore(cfg)     # the settings as the start read them: their one owner from here
        store.install(settings)
        return AppWindow(settings, progress=progress, library=held.get("library"))
    return run


def startup_tasks(cfg) -> list[StartupTask]:
    """What the app does before its window opens, in order; the last gives the window."""
    held: dict = {}     # the library's owner, from its task to the window's
    # weights: the measured times (s) of a start with the files in the OS cache
    tasks = [
        StartupTask(tr("Loading libraries"), 3.0, _import_libraries),
        StartupTask(tr("Compiling audio kernels (JIT)"), 0.4, _compile_audio_kernels),
        StartupTask(tr("Starting FFmpeg"), 0.1, _warm_ffmpeg),
        StartupTask(tr("Loading search readers"), 1.5, _load_readers),
        StartupTask(tr("Loading the library"), 0.1, _open_library(cfg.libconfig.libpath, held)),
        StartupTask(tr("Building the window"), 2.3, _build_window(held, cfg), pump_events=False),
    ]
    from audioengine.native.client import available
    if cfg.playbackconfig.audio_engine == "native" and available():
        # the native engine plays: the Python engine's kernels are loaded only if it falls back (AppWindow)
        tasks = [t for t in tasks if t.run is not _compile_audio_kernels]
    return tasks
