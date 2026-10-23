from __future__ import annotations

from dataclasses import dataclass
import os
import sqlite3
from typing import Callable

from PySide6 import QtCore

from core.i18n import tr
from core.music_search import store
from core.workers.music_index import MusicIndexRequest, MusicIndexResult, MusicIndexWorker

MAX_PROCESSES = 4


@dataclass(frozen=True)
class MusicIndexRun:
    total: int          # tracks to index
    processes: int


@dataclass(frozen=True)
class MusicIndexSummary:
    indexed: int
    failed: tuple[str, ...]
    dropped: int


def index_status(libpath: str) -> tuple[int, int]:
    """(library tracks indexed for Search by Music, library tracks)."""
    try:
        con = sqlite3.connect(f"file:{os.path.join(str(libpath), 'library.db')}?mode=ro", uri=True)
        try:
            uids = {u for (u,) in con.execute("SELECT uid FROM tracks")}
        finally:
            con.close()
    except sqlite3.Error:
        return 0, 0
    return len(uids & store.indexed_uids(store.path_for(libpath))), len(uids)


class LibraryMusicIndex(QtCore.QObject):
    """Index Now (Settings > Search > Search by Music): the library tracks not in the music search's index (analyzed
    before it existed) indexed, the tracks gone from the library dropped from it; one run at a time, its tracks
    shared by up to MAX_PROCESSES processes (a quarter of the cores). The analysis indexes what it analyzes."""

    started = QtCore.Signal(object)               # MusicIndexRun
    progress = QtCore.Signal(object, int, str)    # run, percent, text
    finished = QtCore.Signal(object, object)      # run, MusicIndexSummary
    failed = QtCore.Signal(object, str)

    def __init__(self, *, cfg_getter: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._cfg_getter = cfg_getter
        self._run: MusicIndexRun | None = None
        self._workers: list[MusicIndexWorker] = []
        self._done: dict[int, int] = {}
        self._results: list[MusicIndexResult] = []
        self._errors: list[str] = []
        self._dropped = 0

    def running(self) -> bool:
        return self._run is not None

    @QtCore.Slot(object)
    def index_requested(self, _payload: object = None) -> None:
        if self._run is not None:
            return
        libpath = str(self._cfg_getter().libconfig.libpath)
        search_db = store.path_for(libpath)
        try:
            con = sqlite3.connect(f"file:{os.path.join(libpath, 'library.db')}?mode=ro", uri=True)
            try:
                tracks = [(u, p, int(s or 0), float(m or 0.0)) for u, p, s, m in
                          con.execute("SELECT uid, path, file_size, file_mtime FROM tracks ORDER BY added_ts, uid")]
            finally:
                con.close()
        except sqlite3.Error as exc:
            run = MusicIndexRun(0, 0)
            self.started.emit(run)
            self.failed.emit(run, str(exc))
            return
        have = store.indexed_uids(search_db)
        uids = {t[0] for t in tracks}
        todo = [t for t in tracks if t[0] not in have and t[1] and os.path.isfile(t[1])]
        drop = tuple(sorted(have - uids))
        n = max(1, min(MAX_PROCESSES, (os.cpu_count() or 4) // 4, len(todo)))
        run = MusicIndexRun(len(todo), n)
        self._run, self._done, self._results, self._errors, self._dropped = run, {}, [], [], len(drop)
        self.started.emit(run)
        if not todo and not drop:
            QtCore.QTimer.singleShot(0, self._maybe_finish)
            return
        for k in range(n):
            share = tuple(todo[k::n])
            worker = MusicIndexWorker(MusicIndexRequest(search_db, share, drop if k == 0 else ()), parent=self)
            worker.progress.connect(lambda done, k=k: self._on_progress(k, done))
            worker.finished.connect(self._on_finished)
            worker.failed.connect(self._on_failed)
            self._workers.append(worker)
            self._done[k] = 0
            worker.start()

    def _on_progress(self, share: int, done: int) -> None:
        if self._run is None:
            return
        self._done[share] = done
        total = self._run.total
        n = sum(self._done.values())
        self.progress.emit(self._run, 100 * n // total if total else 100,
                           tr("Indexing for Search by Music: {done} / {total} tracks").format(done=n, total=total))

    @QtCore.Slot(object)
    def _on_finished(self, result: MusicIndexResult) -> None:
        self._results.append(result)
        self._maybe_finish()

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self._errors.append(str(message))
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        run = self._run
        if run is None or len(self._results) + len(self._errors) < len(self._workers):
            return
        for worker in self._workers:
            worker.deleteLater()
        self._workers = []
        self._run = None
        if self._errors:
            self.failed.emit(run, self._errors[0])
            return
        self.finished.emit(run, MusicIndexSummary(
            indexed=sum(r.indexed for r in self._results),
            failed=tuple(p for r in self._results for p in r.failed),
            dropped=self._dropped))
