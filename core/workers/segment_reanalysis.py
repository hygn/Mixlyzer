"""Segment reanalysis subprocess worker: the editor's partial reanalyses (analyzer.pipeline: a tempo segment, the
keys, the phrases, the JumpCUEs) on the track's current marks (a `ReanalysisRequest`), their results in the editor's
form (a `ReanalysisResult`)."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PySide6 import QtCore

from analyzer.music.marks import JumpCueGroup, Phrase
from analyzer.pipeline.beatgrid import reanalyze_segment
from analyzer.pipeline.jumpcues import reanalyze_jumpcues
from analyzer.pipeline.keys import reanalyze_full as reanalyze_keys, reanalyze_selection as reanalyze_key_selection
from analyzer.pipeline.phrases import reanalyze_phrases
from system.process_worker import SpawnProcessWorker, install_parent_watchdog
from core.settings.config import config
from core.workers.analysis_settings import analysis_settings
from core.workers.analysis_texts import error_text, reanalysis_stage
from system.i18n import tr
from core.analysis_library.jump_cues import jump_cue_groups_of_pairs

# what is read again
BEAT = "beat"                  # a tempo segment (its row)
KEY_STATIC = "key_static"      # the key of a span
KEY_DYNAMIC = "key_dynamic"    # the keys of the whole track
JUMPCUE = "jumpcue"            # the JumpCUEs of the whole track
PHRASE = "phrase"              # the phrases of the whole track


@dataclass(frozen=True)
class ReanalysisRequest:
    """An editor's partial reanalysis: what is read again (``kind``) and the track's current marks it reads."""
    kind: str
    beats: np.ndarray
    tempo_segments: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    path: Optional[str] = None                  # None: the deck's track
    segment_index: int = 0                      # BEAT: the row read again
    prev_bpm: Optional[float] = None            # BEAT: the reference BPM
    key_segments: Optional[np.ndarray] = None   # KEY_*: the current keys
    selection: Optional[tuple[float, float]] = None   # KEY_STATIC: the span (s)
    phrases: Optional[tuple[Phrase, ...]] = None      # KEY_*, JUMPCUE: the current phrases (UNASSIGNED left out)
    duration_sec: float = 0.0


@dataclass(frozen=True)
class ReanalysisResult:
    """A partial reanalysis's result: the marks it gives (None: not this kind's)."""
    beats: Optional[np.ndarray] = None
    tempo_segments: Optional[np.ndarray] = None
    key_segments: Optional[np.ndarray] = None
    key_segments_auto: Optional[np.ndarray] = None    # KEY_DYNAMIC: the new automatic result
    key_selection: Optional[tuple] = None
    jump_cues: Optional[tuple[JumpCueGroup, ...]] = None   # also the new automatic result
    phrases: Optional[list[Phrase]] = None


class SegmentReanalysisWorker(SpawnProcessWorker):
    """Worker that recomputes tempo or key segments in a subprocess."""

    finished = QtCore.Signal(object)     # ReanalysisResult
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str, float)
    status = QtCore.Signal(str)

    def __init__(
        self,
        path: str,
        cfg: config,
        request: ReanalysisRequest,
        taskid: int,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(
            target=_segment_reanalysis_entry,
            args=(path, cfg, request, taskid),
            daemon=True,
            process_name=f"Mixlyzer-segment-analysis-{taskid}",
            parent=parent,
        )
        self._result_emitted = False
        self._error_emitted = False

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        mtype = message.get("type")
        if mtype == "progress":
            self.progress.emit(message.get("status", ""), message.get("progress", 0.0))
        elif mtype == "status":
            self.status.emit(message.get("status", ""))
        elif mtype == "result" and not self._result_emitted:
            self._result_emitted = True
            self.finished.emit(message.get("payload"))
        elif mtype == "error" and not self._error_emitted:
            self._error_emitted = True
            self.error.emit(message.get("message", "Unknown error"))

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        if not graceful and not self._error_emitted and not self._result_emitted:
            self._error_emitted = True
            self.error.emit(tr("Segment analysis process exited unexpectedly ({code}).").format(code=exitcode))


def _segment_reanalysis_entry(
    path: str,
    cfg: config,
    request: ReanalysisRequest,
    taskid: int,
    queue=None,
) -> None:
    install_parent_watchdog()

    beats = np.asarray(request.beats, dtype=float)
    tempo_segments = np.asarray(request.tempo_segments, dtype=float)
    key_segments = None
    if request.key_segments is not None:
        try:
            key_segments = np.asarray(request.key_segments, dtype=float)
        except Exception:
            key_segments = None
    phrases = list(request.phrases) if request.phrases is not None else None

    def progress(stage: str, fraction: float) -> None:
        queue.put({"type": "progress", "taskid": taskid, "status": reanalysis_stage(stage), "progress": float(fraction)})

    def done(text: str, result: ReanalysisResult) -> None:
        queue.put({"type": "progress", "taskid": taskid, "status": text, "progress": 0.95})
        queue.put({"type": "result", "taskid": taskid, "payload": result})

    try:
        settings = analysis_settings(cfg)
        kind = request.kind
        if kind in (BEAT, KEY_STATIC, KEY_DYNAMIC, JUMPCUE) and beats.size == 0:
            raise ValueError(f"the beats are required for this reanalysis ({kind})")
        if kind == BEAT:
            new_beats, new_segments = reanalyze_segment(path, settings, beats, tempo_segments, request.segment_index,
                                                        prev_bpm=request.prev_bpm, progress=progress)
            done(tr("Segment updated"), ReanalysisResult(beats=np.asarray(new_beats, dtype=float),
                                                         tempo_segments=np.asarray(new_segments, dtype=float)))
        elif kind == KEY_STATIC:
            if key_segments is None or key_segments.ndim != 2 or key_segments.shape[1] < 4:
                raise ValueError("key_segments must be provided for key reanalysis")
            updated_segments, picked = reanalyze_key_selection(
                path, settings, beats, request.duration_sec, request.selection, key_segments,
                tempo_segments=tempo_segments, phrases=phrases, progress=progress)
            done(tr("Key segment updated"), ReanalysisResult(key_segments=np.asarray(updated_segments, dtype=float),
                                                             key_selection=tuple(picked)))
        elif kind == KEY_DYNAMIC:
            updated_segments = np.asarray(reanalyze_keys(
                path, settings, beats, tempo_segments=tempo_segments, phrases=phrases, progress=progress),
                dtype=float)
            # the new automatic result (the key optimizer learns from where the user's keys differ)
            done(tr("Key segment updated"), ReanalysisResult(key_segments=updated_segments,
                                                             key_segments_auto=updated_segments.copy()))
        elif kind == JUMPCUE:
            pairs = reanalyze_jumpcues(path, settings, beats, tempo_segments=tempo_segments, phrases=phrases,
                                       progress=progress)
            done(tr("JumpCUE Updated"), ReanalysisResult(jump_cues=tuple(jump_cue_groups_of_pairs(pairs))))
        elif kind == PHRASE:
            rows = reanalyze_phrases(path, settings, beats, tempo_segments=tempo_segments, progress=progress)
            done(tr("Phrases Updated"), ReanalysisResult(phrases=list(rows)))
        else:
            raise ValueError(f"Unknown analyze type: {kind}")
    except Exception as exc:
        traceback.print_exc()
        queue.put(
            {
                "type": "error",
                "taskid": taskid,
                "message": error_text(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        queue.put({"type": "done", "taskid": taskid})
