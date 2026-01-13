from __future__ import annotations

import multiprocessing as mp
import traceback
from queue import Empty
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
from core.analysis_worker import _install_parent_watchdog
from core.config import config
from utils.keystrip import build_keystrip_buffer


class SegmentReanalysisWorker(QtCore.QObject):
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
        super().__init__(parent)
        self._path = path
        self._config = cfg
        self._beats = beats
        self._segments = segments
        self._key_segments = key_segments
        self._selection = selection
        self._segment_index = int(segment_index)
        self._taskid = taskid
        self._prev_bpm = prev_bpm
        self._use_only_prev_bpm = bool(use_only_prev_bpm)
        self._analyze_type = analyze_type
        self._ctx = mp.get_context("spawn")
        self._queue: Any | None = None
        self._process: mp.Process | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll_queue)
        self._result_emitted = False
        self._error_emitted = False
        self._duration = duration

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Worker already started.")
        self._queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_segment_reanalysis_entry,
            args=(
                self._path,
                self._config,
                self._segment_index,
                self._beats,
                self._duration,
                self._segments,
                self._key_segments,
                self._selection,
                self._taskid,
                self._queue,
                self._prev_bpm,
                self._use_only_prev_bpm,
                self._analyze_type,
            ),
        )
        self._process.daemon = True
        self._process.start()
        self._timer.start()

    def stop(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1)
        self._cleanup()

    def _poll_queue(self) -> None:
        if self._queue is None:
            return
        while True:
            queue = self._queue
            if queue is None:
                return
            try:
                message = queue.get_nowait()
            except Empty:
                break
            mtype = message.get("type")
            if mtype == "progress":
                self.progress.emit(message.get("status", ""), message.get("progress", 0.0))
            elif mtype == "status":
                self.status.emit(message.get("status", ""))
            elif mtype == "result":
                if not self._result_emitted:
                    payload = dict(message.get("payload", {}) or {})
                    self.finished.emit(payload)
                    self._result_emitted = True
            elif mtype == "error":
                if not self._error_emitted:
                    self.error.emit(message.get("message", "Unknown error"))
                    self._error_emitted = True
                break
            elif mtype == "done":
                self._timer.stop()
                self._finalize_process()
                break

    def _finalize_process(self) -> None:
        if self._process is not None:
            self._process.join(timeout=1)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._queue is not None:
            self._queue.close()
            self._queue.join_thread()
            self._queue = None
        if self._process is not None:
            self._process.close()
            self._process = None

    def __del__(self):
        self.stop()


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
    queue,
    prev_bpm=None,
    use_only_prev_bpm: bool = False,
    analyze_type: str = "beat",
) -> None:
    _install_parent_watchdog()

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
