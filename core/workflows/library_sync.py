"""Settings > Library's Sync Now: library.db made to agree with every track's stored analysis (core.library.jobs.
sync_index: its BPM, key and segment tables written from its marks), a batch of tracks a job of the library's owner,
so the GUI's jobs run between."""
from __future__ import annotations

from PySide6 import QtCore

from core.library import jobs


class LibraryIndexSync(QtCore.QObject):
    """One run at a time (asked again while it runs: nothing more)."""

    progress = QtCore.Signal(int, int)        # tracks done, of
    finished = QtCore.Signal(object)          # jobs.IndexSync, the run's sums
    failed = QtCore.Signal(str)

    def __init__(self, library, parent=None) -> None:
        """``library``: the library's owner (core.library.service)."""
        super().__init__(parent)
        self._library = library
        self._pending: list[str] | None = None      # the tracks left; None: not running
        self._total = 0
        self._sum = jobs.IndexSync()

    @property
    def running(self) -> bool:
        return self._pending is not None

    @QtCore.Slot()
    def sync_requested(self) -> None:
        if self.running:
            return
        self._pending = [track.uid for track in self._library.snapshot.tracks if track.uid]
        self._total, self._sum = len(self._pending), jobs.IndexSync()
        self.progress.emit(0, self._total)
        self._next()

    def _next(self) -> None:
        if not self._pending:
            done = self._sum
            print(f"[Library] library.db synced with the stored analyses: {done.changed} of {self._total} tracks "
                  f"changed, {done.unread} without a stored analysis that reads")
            if done.changed:        # the snapshot and the library's filters follow, once
                self._library.write(jobs.index_synced, done=lambda _r: self._finish(done), failed=self._failed)
            else:
                self._finish(done)
            return
        batch, self._pending = self._pending[:jobs.SYNC_BATCH], self._pending[jobs.SYNC_BATCH:]
        self._library.write(jobs.sync_index, batch, done=self._batch_done, failed=self._failed)

    def _batch_done(self, result: jobs.IndexSync) -> None:
        self._sum = jobs.IndexSync(self._sum.changed + result.changed, self._sum.unread + result.unread)
        self.progress.emit(self._total - len(self._pending or ()), self._total)
        self._next()

    def _finish(self, done: jobs.IndexSync) -> None:
        self._pending = None
        self.finished.emit(done)

    def _failed(self, exc: Exception) -> None:
        self._pending = None
        self.failed.emit(str(exc))
