"""Segment reanalysis subprocess worker."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
from PySide6 import QtCore

from analyzer_core.editor.beatgrid import (
    reanalyze_segment_from_file as beat_reanalyze_segment_from_file,
)
from analyzer_core.editor.keystrip import (
    reanalyze_segment_from_file as key_reanalyze_segment_from_file,
    reanalyze_full as key_reanalyze_full,
    update_key_segments_with_selection,
)
from analyzer_core.editor.jumpcue import reanalyze_jumpCUE
from core.config import config
from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog
from utils.keystrip import build_keystrip_buffer


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
        use_only_prev_bpm: bool = False,
        analyze_type: str = "beat",
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
                bool(use_only_prev_bpm),
                analyze_type,
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
            self.error.emit(f"Segment analysis process exited unexpectedly ({exitcode}).")


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
    use_only_prev_bpm: bool = False,
    analyze_type: str = "beat",
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

    def _progress(status: str, progress: float) -> None:
        queue.put(
            {
                "type": "progress",
                "taskid": taskid,
                "status": status,
                "progress": float(progress),
            }
        )

    try:
        if analyze_type == "beat":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for beat reanalysis")
            segments_arr = tempo_segments_arr
            new_beats, new_segments = beat_reanalyze_segment_from_file(
                path,
                cfg,
                beats_arr,
                segments_arr,
                segment_index,
                prev_bpm=prev_bpm,
                use_only_prev_bpm=use_only_prev_bpm,
                progress_cb=_progress,
            )
            queue.put(
                {
                    "type": "result",
                    "taskid": taskid,
                    "payload": {
                        "beats_time_sec": new_beats.tolist(),
                        "tempo_segments": new_segments.tolist(),
                    },
                }
            )
        elif analyze_type  == "key_static":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for key reanalysis")
            if key_segments_arr is None or key_segments_arr.ndim != 2 or key_segments_arr.shape[1] < 4:
                raise ValueError("key_segments must be provided for key reanalysis")
            if selection is None:
                raise ValueError("selection end must be greater than start")
            sel_start, sel_end = map(float, selection)
            if sel_end <= sel_start:
                raise ValueError("selection end must be greater than start")
            beats_arr = np.concatenate(([0.], beats_arr, [duration]))
            beat_start_index = int(np.searchsorted(beats_arr, sel_start, side="left"))
            beat_end_index = int(np.searchsorted(beats_arr, sel_end, side="right"))-1
            beat_start_index = max(0, min(beat_start_index, max(beats_arr.size - 1, 0)))
            beat_end_index = max(
                beat_start_index + 1,
                min(max(beat_end_index, beat_start_index + 1), beats_arr.size),
            )
            selection_tuple = key_reanalyze_segment_from_file(
                path,
                cfg,
                beats_arr,
                beat_start_index,
                beat_end_index,
                progress_cb=_progress,
            )
            updated_segments = update_key_segments_with_selection(
                key_segments_arr,
                selection_tuple,
                beat_times=beats_arr,
            )
            total_duration = float(np.max(updated_segments[:, 3])) if updated_segments.size else 0.0
            key_img = build_keystrip_buffer(updated_segments, total_duration)
            queue.put(
                {
                    "type": "result",
                    "taskid": taskid,
                    "payload": {
                        "key_segments": updated_segments.tolist(),
                        "key_image": key_img.tolist() if key_img is not None else None,
                        "key_selection": list(selection_tuple),
                    },
                }
            )
        elif analyze_type == "key_dynamic":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for key reanalysis")
            updated_segments = key_reanalyze_full(
                path,
                cfg,
                beats_arr,
                progress_cb=_progress,
            )
            updated_segments = np.asarray(updated_segments, dtype=float)
            total_duration = float(np.max(updated_segments[:, 3])) if updated_segments.size else 0.0
            key_img = build_keystrip_buffer(updated_segments, total_duration)
            queue.put(
                {
                    "type": "result",
                    "taskid": taskid,
                    "payload": {
                        "key_segments": updated_segments.tolist(),
                        "key_image": key_img.tolist() if key_img is not None else None,
                    },
                }
            )
        elif analyze_type == "jumpcue":
            if beats_arr.size == 0:
                raise ValueError("beats_time_sec is required for JumpCUE reanalysis")
            jc_dict = reanalyze_jumpCUE(path, cfg, beats_arr, progress_cb=_progress)
            queue.put(
                {
                    "type": "result",
                    "taskid": taskid,
                    "payload": jc_dict,
                }
            )
        else:
            raise ValueError(f"Unknown analyze type: {analyze_type}")
    except Exception as exc:
        traceback.print_exc()
        queue.put(
            {
                "type": "error",
                "taskid": taskid,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        queue.put({"type": "done", "taskid": taskid})
