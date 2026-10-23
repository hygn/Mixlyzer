"""Index Now (Settings > Search > Search by Music): the library tracks analyzed before the music search existed get
their index (core.music_search.indexing: reference lines and landmarks; about 6 s a track) in spawned processes at a
lower priority, each its share of the tracks; each writes its tracks to music_search.db itself (SQLite serializes
the writes). The analysis indexes the tracks it analyzes."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any

from PySide6 import QtCore

from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog
from core.i18n import tr


@dataclass(frozen=True)
class MusicIndexRequest:
    search_db: str
    tracks: tuple[tuple[str, str, int, float], ...]   # (uid, path, file size, file time) to index
    drop: tuple[str, ...] = ()                         # uids gone from the library: out of the index


@dataclass(frozen=True)
class MusicIndexResult:
    indexed: int
    failed: tuple[str, ...]    # the paths that could not be read


def index_tracks(request: MusicIndexRequest, progress=None) -> MusicIndexResult:
    """Index ``request.tracks``; ``progress(done)`` after each."""
    from threadpoolctl import threadpool_limits
    from analyzer_core.global_analyzer import fast_load
    from core.music_search import indexing, store

    con = store.open_writer(request.search_db)
    failed: list[str] = []
    done = 0
    try:
        with con:
            for uid in request.drop:
                store.drop(con, uid)
        for uid, path, size, mtime in request.tracks:
            try:
                with threadpool_limits(1):
                    entry = indexing.index_audio(fast_load(path, target_sr=indexing.SR, stereo=True))
                with con:
                    store.put(con, uid, size, mtime, entry)
            except Exception:
                failed.append(path)
            done += 1
            if progress is not None:
                progress(done)
        return MusicIndexResult(indexed=done - len(failed), failed=tuple(failed))
    finally:
        con.close()


def _music_index_process_entry(request: MusicIndexRequest, queue: Any) -> None:
    install_parent_watchdog()
    from core.os_support import lower_process_priority
    lower_process_priority()
    try:
        result = index_tracks(request, lambda done: queue.put({"type": "progress", "done": done}))
        queue.put({"type": "result", "payload": result})
    except Exception:
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "done"})


class MusicIndexWorker(SpawnProcessWorker):
    """Qt-facing worker for one share of an Index Now."""

    progress = QtCore.Signal(int)           # tracks done of this share
    finished = QtCore.Signal(object)        # MusicIndexResult
    failed = QtCore.Signal(str)

    def __init__(self, request: MusicIndexRequest, parent: QtCore.QObject | None = None) -> None:
        self._result: MusicIndexResult | None = None
        self._error_emitted = False
        super().__init__(
            target=_music_index_process_entry,
            args=(request,),
            daemon=True,
            process_name="Mixlyzer-music-index",
            parent=parent,
        )

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "progress":
            self.progress.emit(int(message["done"]))
        elif kind == "result":
            self._result = message.get("payload")
        elif kind == "error" and not self._error_emitted:
            self._error_emitted = True
            self.failed.emit(str(message.get("traceback", "Unknown music index error")))

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        if self._error_emitted:
            return
        if not graceful or self._result is None:
            self._error_emitted = True
            self.failed.emit(tr("The music search index process exited unexpectedly ({code}).").format(code=exitcode))
            return
        self.finished.emit(self._result)
