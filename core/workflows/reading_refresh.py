from __future__ import annotations

from typing import Callable

from PySide6 import QtCore

from foundation.i18n import tr
from core.library import jobs
from core.workers.reading_refresh import (
    BATCH_TRACKS, ReadingRefreshRequest, ReadingRefreshResult, ReadingRefreshWorker,
)


class LibraryReadingRefresh(QtCore.QObject):
    """Read the library's readings again (Settings > Search), one run at a time: a request during a
    run starts once it ends, with the settings of then. The library's owner gives the texts and writes what the
    process read, a batch of tracks a job (the process ends with the app: nothing of its own is written)."""

    started = QtCore.Signal(object)
    progress = QtCore.Signal(object, int, str)
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(object, str)

    def __init__(self, *, cfg_getter: Callable[[], object], library, parent=None) -> None:
        """``library``: the library's owner (core.library.service)."""
        super().__init__(parent)
        self._cfg_getter = cfg_getter
        self._library = library
        self._pending = False
        self._busy = False          # a run from the texts' reading to its last write
        self._uids: list[str] = []
        self._active: ReadingRefreshRequest | None = None
        self._worker: ReadingRefreshWorker | None = None

    @QtCore.Slot(object)
    def refresh_requested(self, _payload: object = None) -> None:
        self._pending = True
        if not self._busy:
            self._start_next()

    def _start_next(self) -> None:
        if self._busy or not self._pending:
            return
        self._pending, self._busy = False, True
        cfg = self._cfg_getter()
        languages, han_run = tuple(cfg.searchconfig.db_languages), cfg.searchconfig.han_run()
        self._library.read(jobs.reading_texts, done=lambda found: self._run(found, languages, han_run),
                           failed=lambda exc: self._fail_before_run(str(exc)))

    def _fail_before_run(self, message: str) -> None:
        self._busy = False
        self.failed.emit(None, message)
        QtCore.QTimer.singleShot(0, self._start_next)

    def _run(self, found: tuple, languages: tuple, han_run: int) -> None:
        uids, texts = found
        self._uids = list(uids)
        request = ReadingRefreshRequest(texts=tuple(texts), languages=languages, han_run=han_run, tracks=len(uids))
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
            self.progress.emit(self._active, value, tr("Reading the library: {done} / {total} tracks").format(done=done, total=total))

    @QtCore.Slot(object)
    def _job_finished(self, read: dict) -> None:
        """The texts read: written by the library's owner, a batch of tracks a job, then the works' names."""
        request = self._release()
        uids, changed = self._uids, [0]
        for start in range(0, len(uids), BATCH_TRACKS):
            self._library.write(jobs.apply_readings, uids[start:start + BATCH_TRACKS], read,
                                done=lambda n: changed.__setitem__(0, changed[0] + n))

        def done(_r) -> None:
            self._busy = False
            if request is not None:
                self.finished.emit(request, ReadingRefreshResult(tracks=len(uids), changed_fields=changed[0]))
            QtCore.QTimer.singleShot(0, self._start_next)

        self._library.write(jobs.apply_work_readings, read, done=done,
                            failed=lambda exc: self._job_failed(str(exc), request))

    @QtCore.Slot(str)
    def _job_failed(self, message: str, request: ReadingRefreshRequest | None = None) -> None:
        request = self._release() or request
        self._busy = False
        if request is not None:
            self.failed.emit(request, str(message))
        QtCore.QTimer.singleShot(0, self._start_next)

    def _release(self) -> ReadingRefreshRequest | None:
        request, worker = self._active, self._worker
        self._active, self._worker = None, None
        if worker is not None:
            worker.deleteLater()
        return request
