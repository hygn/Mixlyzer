"""Index Now (Settings > Search > Search by Music): the library tracks analyzed before the music search or the
variants existed get their index. Search by Music's (core.music_search.indexing: reference lines and landmarks from
the audio; about 3 s a track) is made in spawned processes at a lower priority, each its share of the tracks, and
handed back track by track; the library's owner stores each (core.library.jobs.store_music_entry), makes the
variants' fingerprints from the stored features (jobs.fingerprint: no audio, about 20 ms a track) and drops the
tracks gone (jobs.drop_gone): core.workflows.music_index runs it. The processes never open the library. The analysis
indexes the tracks it analyzes."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any

from PySide6 import QtCore

from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog
from foundation.i18n import tr
from core.library.records import IndexTrack


@dataclass(frozen=True)
class MusicIndexRequest:
    tracks: tuple[IndexTrack, ...]       # a share of the tracks whose audio is indexed


# ---- the process: the audio read, nothing of the library ----

def index_audio_tracks(request: MusicIndexRequest, report) -> None:
    """Each track's Search by Music entry from its audio: ``report(uid, entry)``, the entry None when the audio
    could not be read."""
    from threadpoolctl import threadpool_limits
    from analyzer.pipeline.track import fast_load
    from core.music_search import indexing

    for track in request.tracks:
        try:
            with threadpool_limits(1):
                entry = indexing.index_audio(fast_load(track.path, target_sr=indexing.SR, stereo=True))
        except Exception:
            entry = None
        report(track.uid, entry)


def _music_index_process_entry(request: MusicIndexRequest, queue: Any) -> None:
    install_parent_watchdog()
    from foundation.os_support import lower_process_priority
    lower_process_priority()
    try:
        index_audio_tracks(request, lambda uid, entry: queue.put({"type": "entry", "uid": uid, "entry": entry}))
        queue.put({"type": "result"})
    except Exception:
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "done"})


class MusicIndexWorker(SpawnProcessWorker):
    """Qt-facing worker for one share of an Index Now: each track's entry handed back as it is made."""

    entry = QtCore.Signal(str, object)      # a track's uid, its entry (None: its audio could not be read)
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, request: MusicIndexRequest, parent: QtCore.QObject | None = None) -> None:
        self._ended = False
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
        if kind == "entry":
            self.entry.emit(str(message["uid"]), message.get("entry"))
        elif kind == "result":
            self._ended = True
        elif kind == "error" and not self._error_emitted:
            self._error_emitted = True
            self.failed.emit(str(message.get("traceback", "Unknown music index error")))

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        if self._error_emitted:
            return
        if not graceful or not self._ended:
            self._error_emitted = True
            self.failed.emit(tr("The music search index process exited unexpectedly ({code}).").format(code=exitcode))
            return
        self.finished.emit()
