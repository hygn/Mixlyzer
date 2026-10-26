"""Segment reanalysis subprocess worker: the editor's partial reanalyses (analyzer.pipeline: a tempo segment, the
keys, the phrases, the JumpCUEs) on the track's current marks, their results in the editor's form."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
from PySide6 import QtCore

from analyzer.pipeline.beatgrid import reanalyze_segment
from analyzer.pipeline.jumpcues import reanalyze_jumpcues
from analyzer.pipeline.keys import reanalyze_full as reanalyze_keys, reanalyze_selection as reanalyze_key_selection
from analyzer.pipeline.phrases import reanalyze_phrases
from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog
from core.config import config
from core.workers.analysis_settings import analysis_settings
from core.workers.analysis_texts import error_text, reanalysis_stage
from foundation.i18n import tr
from utils.jump_cues import build_jump_cues_np


class SegmentReanalysisWorker(SpawnProcessWorker):
    """Worker that recomputes tempo or key segments in a subprocess."""

    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str, float)
    status = QtCore.Signal(str)

    def __init__(
        self,
        path: str,
        cfg: config,
        beats,
        duration,
        segments,
        key_segments,
        selection,
        segment_index: int,
        taskid: int,
        prev_bpm=None,
        analyze_type: str = "beat",
        phrases=None,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(
            target=_segment_reanalysis_entry,
            args=(
                path,
                cfg,
                int(segment_index),
                beats,
                duration,
                segments,
                key_segments,
                selection,
                taskid,
                prev_bpm,
                analyze_type,
                phrases,
            ),
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
            self.finished.emit(dict(message.get("payload", {}) or {}))
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
    segment_index: int,
    beats,
    duration,
    segments,
    key_segments,
    selection,
    taskid: int,
    prev_bpm=None,
    analyze_type: str = "beat",
    phrases=None,
    queue=None,
) -> None:
    install_parent_watchdog()

    beats_arr = np.asarray(beats, dtype=float)
    tempo_segments_arr = np.asarray(segments, dtype=float)
    key_segments_arr = None
    if key_segments is not None:
        try:
            key_segments_arr = np.asarray(key_segments, dtype=float)
        except Exception:
            key_segments_arr = None

    def progress(stage: str, fraction: float) -> None:
        queue.put({"type": "progress", "taskid": taskid, "status": reanalysis_stage(stage), "progress": float(fraction)})

    def done(text: str, payload: dict) -> None:
        queue.put({"type": "progress", "taskid": taskid, "status": text, "progress": 0.95})
        queue.put({"type": "result", "taskid": taskid, "payload": payload})

    try:
        settings = analysis_settings(cfg)
        if analyze_type == "beat":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for beat reanalysis")
            new_beats, new_segments = reanalyze_segment(
                path, settings, beats_arr, tempo_segments_arr, segment_index, prev_bpm=prev_bpm, progress=progress)
            done(tr("Segment updated"), {"beats_time_sec": new_beats.tolist(), "tempo_segments": new_segments.tolist()})
        elif analyze_type == "key_static":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for key reanalysis")
            if key_segments_arr is None or key_segments_arr.ndim != 2 or key_segments_arr.shape[1] < 4:
                raise ValueError("key_segments must be provided for key reanalysis")
            updated_segments, picked = reanalyze_key_selection(
                path, settings, beats_arr, duration, selection, key_segments_arr,
                tempo_segments=tempo_segments_arr, phrases=phrases, progress=progress)
            done(tr("Key segment updated"), {"key_segments": updated_segments.tolist(), "key_selection": list(picked)})
        elif analyze_type == "key_dynamic":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for key reanalysis")
            updated_segments = np.asarray(reanalyze_keys(
                path, settings, beats_arr, tempo_segments=tempo_segments_arr, phrases=phrases, progress=progress),
                dtype=float)
            # the new automatic result (the key optimizer learns from where the user's keys differ)
            done(tr("Key segment updated"), {"key_segments": updated_segments.tolist(),
                                             "key_segments_auto": updated_segments.tolist()})
        elif analyze_type == "jumpcue":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for JumpCUE reanalysis")
            pairs = reanalyze_jumpcues(path, settings, beats_arr, tempo_segments=tempo_segments_arr, phrases=phrases,
                                       progress=progress)
            jump_cues_np = build_jump_cues_np(pairs, canonicalize_labels=True, merge_coincident=True)
            done(tr("JumpCUE Updated"), {"jump_cues_np": jump_cues_np, "jump_cues_auto_np": jump_cues_np})
        elif analyze_type == "phrase":
            rows = reanalyze_phrases(path, settings, beats_arr, tempo_segments=tempo_segments_arr, progress=progress)
            done(tr("Phrases Updated"), {"phrases": rows})
        else:
            raise ValueError(f"Unknown analyze type: {analyze_type}")
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
