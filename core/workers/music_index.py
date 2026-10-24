"""Index Now (Settings > Search > Search by Music): the library tracks analyzed before the music search or the
variants existed get their index in spawned processes at a lower priority, each its share of the tracks: Search by
Music's (core.music_search.indexing: reference lines and landmarks from the audio; about 3 s a track) and the
variants' fingerprint (core.variants: from the stored features, no audio; about 20 ms a track). Each writes its
tracks to music_search.db / variants.db itself (SQLite serializes the writes). The analysis indexes the tracks it
analyzes."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any

from PySide6 import QtCore

from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog
from core.i18n import tr


@dataclass(frozen=True)
class IndexTrack:
    uid: str
    path: str            # the audio file
    size: int            # its size and time, kept with Search by Music's index
    mtime: float
    music: bool          # Search by Music's index to make (from the audio)
    variants: str        # the stored features (NPZ) to fingerprint for the variants, or ""


@dataclass(frozen=True)
class MusicIndexRequest:
    search_db: str
    variants_db: str
    tracks: tuple[IndexTrack, ...]
    drop: tuple[str, ...] = ()            # uids gone from the library: out of Search by Music's index
    drop_variants: tuple[str, ...] = ()   # and out of the variants'


@dataclass(frozen=True)
class MusicIndexResult:
    indexed: int
    failed: tuple[str, ...]    # the paths that could not be read


def index_tracks(request: MusicIndexRequest, progress=None) -> MusicIndexResult:
    """Index ``request.tracks``; ``progress(done)`` after each."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    from analyzer_core.global_analyzer import fast_load
    from core.music_search import indexing, store
    from core.variants import fingerprint as variant_fp, store as variant_store

    con = store.open_writer(request.search_db)
    vcon = variant_store.open_writer(request.variants_db)
    failed: list[str] = []
    done = 0
    try:
        with con:
            for uid in request.drop:
                store.drop(con, uid)
        with vcon:
            for uid in request.drop_variants:
                variant_store.drop(vcon, uid)
        for track in request.tracks:
            try:
                if track.variants:
                    with np.load(track.variants, allow_pickle=False) as npz:
                        features = {k: npz[k] for k in variant_fp.FEATURES}
                    variant_store.put(vcon, track.uid, features)
                if track.music:
                    with threadpool_limits(1):
                        entry = indexing.index_audio(fast_load(track.path, target_sr=indexing.SR, stereo=True))
                    with con:
                        store.put(con, track.uid, track.size, track.mtime, entry)
            except Exception:
                failed.append(track.path)
            done += 1
            if progress is not None:
                progress(done)
        return MusicIndexResult(indexed=done - len(failed), failed=tuple(failed))
    finally:
        con.close()
        vcon.close()


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
