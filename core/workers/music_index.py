"""Index Now (Settings > Search > Search by Music): the library tracks analyzed before the music search, the
variants or the mix candidates existed get their index. Search by Music's (analyzer.pipeline.music_index: reference
lines and landmarks from the audio; about 3 s a track) and the mix candidates' beats' features (analyzer.pipeline.
mix_features: on the stored beats given; about 6 s a track) are made in spawned processes at a lower priority, each
its share of the tracks, and handed back track by track; the library's owner stores each
(core.library.jobs.store_music_entry, store_mix_features), makes the variants' fingerprints from the stored analyses
(jobs.fingerprint: no audio, about 20 ms a track) and drops the tracks gone (jobs.drop_gone):
core.workflows.music_index runs it. The processes never open the library. The analysis indexes the tracks it
analyzes."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any

from PySide6 import QtCore

from system.process_worker import SpawnProcessWorker, install_parent_watchdog
from system.i18n import tr
from core.library.records import IndexTrack


@dataclass(frozen=True)
class MusicIndexRequest:
    tracks: tuple[IndexTrack, ...]       # a share of the tracks whose audio is indexed


# ---- the process: the audio read, nothing of the library ----

def index_audio_tracks(request: MusicIndexRequest, report, report_mix=lambda uid, features: None) -> None:
    """Each track's Search by Music entry from its audio: ``report(uid, entry)``, the entry None when the audio
    could not be read; and the mix candidates' beats' features on the beats given: ``report_mix(uid, features)``,
    None when not read."""
    from threadpoolctl import threadpool_limits
    from analyzer.pipeline.music_index import index_track
    from analyzer.pipeline.mix_features import mix_features_of_track

    for track in request.tracks:
        if track.music:
            try:
                with threadpool_limits(1):
                    entry = index_track(track.path)
            except Exception:
                entry = None
            report(track.uid, entry)
        if track.mix_beats is not None:
            try:
                with threadpool_limits(1):
                    features = mix_features_of_track(track.path, track.mix_beats)
            except Exception:
                features = None
            report_mix(track.uid, features)


def _music_index_process_entry(request: MusicIndexRequest, queue: Any) -> None:
    install_parent_watchdog()
    from system.os_support import lower_process_priority
    lower_process_priority()
    try:
        index_audio_tracks(request, lambda uid, entry: queue.put({"type": "entry", "uid": uid, "entry": entry}),
                           lambda uid, features: queue.put({"type": "mix", "uid": uid, "features": features}))
        queue.put({"type": "result"})
    except Exception:
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "done"})


class MusicIndexWorker(SpawnProcessWorker):
    """Qt-facing worker for one share of an Index Now: each track's entry handed back as it is made."""

    entry = QtCore.Signal(str, object)      # a track's uid, its entry (None: its audio could not be read)
    mix = QtCore.Signal(str, object)        # a track's uid, its beats' features (None: not read)
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
        elif kind == "mix":
            self.mix.emit(str(message["uid"]), message.get("features"))
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
