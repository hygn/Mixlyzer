from __future__ import annotations

import os
from typing import Callable

from PySide6 import QtCore

from core.workers.reading_refresh import ReadingRefreshRequest, ReadingRefreshResult, ReadingRefreshWorker


class LibraryReadingRefresh(QtCore.QObject):
    """Read the library's readings again (Settings > Search), one run at a time: a request during a
    run starts once it ends, with the settings of then. The process ends with the app (an
    unfinished batch is rolled back; the tracks after it keep their readings)."""

    started = QtCore.Signal(object)
    progress = QtCore.Signal(object, int, str)
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(object, str)

    def __init__(self, *, cfg_getter: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._cfg_getter = cfg_getter
        self._pending = False
        self._active: ReadingRefreshRequest | None = None
        self._worker: ReadingRefreshWorker | None = None

    @QtCore.Slot(object)
    def refresh_requested(self, _payload: object = None) -> None:
        self._pending = True
        if self._worker is None:
            self._start_next()

    def _start_next(self) -> None:
        if self._worker is not None or not self._pending:
            return
        self._pending = False
        cfg = self._cfg_getter()
        request = ReadingRefreshRequest(
            db_path=os.path.join(str(cfg.libconfig.libpath), "library.db"),
            languages=tuple(cfg.searchconfig.db_languages),
            han_run=cfg.searchconfig.han_run(),
        )
        worker = ReadingRefreshWorker(request, parent=self)
        worker.progress.connect(self._relay_progress)
        worker.finished.connect(self._job_finished)
        worker.failed.connect(self._job_failed)
        self._active = request
        self._worker = worker
        self.started.emit(request)
        worker.start()

    @QtCore.Slot(int, int)
    def _relay_progress(self, done: int, total: int) -> None:
        if self._active is not None:
            value = 100 * done // total if total else 100
            self.progress.emit(self._active, value, f"Reading the library: {done} / {total} tracks")

    @QtCore.Slot(object)
    def _job_finished(self, result: ReadingRefreshResult) -> None:
        request = self._release()
        if request is not None:
            self.finished.emit(request, result)
        QtCore.QTimer.singleShot(0, self._start_next)

    @QtCore.Slot(str)
    def _job_failed(self, message: str) -> None:
        request = self._release()
        if request is not None:
            self.failed.emit(request, str(message))
        QtCore.QTimer.singleShot(0, self._start_next)

    def _release(self) -> ReadingRefreshRequest | None:
        request, worker = self._active, self._worker
        self._active, self._worker = None, None
        if worker is not None:
            worker.deleteLater()
        return request
