"""Full-track analysis subprocess worker: the analyzer's pipeline (analyzer.pipeline.track) run on a track, and
what the library stores of it made from its results (the track's row, the stored features, the index rows)."""

import os
import time
import traceback
from typing import Any, Callable, Dict, Optional

import numpy as np
from PySide6 import QtCore
from threadpoolctl import threadpool_limits

from analyzer.audio.decoder import get_total_samples
from analyzer.pipeline.track import analyze_track
from core.tags.reader import cover_data, extract_tags, read_tag_sources
from system.process_worker import SpawnProcessWorker, install_parent_watchdog
from core.settings.config import config
from core.library.segments import build_bpm_segments, build_key_segments, build_phrase_index_rows
from core.workers.analysis_settings import analysis_settings
from core.workers.analysis_texts import error_text, track_stage
from system.i18n import tr
from core.analysis_library.cue_points import build_cue_points_np
from core.analysis_library.jump_cues import build_jump_cues_np
from core.analysis_library.phrases import build_phrase_segments_np


def _file_stats(path: str) -> tuple[int, float]:
    try:
        st = os.stat(path)
        return int(st.st_size), float(st.st_mtime)
    except FileNotFoundError:
        return 0, 0.0


def analyze_for_library(path: str, cfg: config, force_analyze: bool = False,
                        progress: Optional[Callable[[str, str, float], None]] = None) -> Dict[str, Any]:
    """The track analyzed, as the library's owner stores it (core.library.jobs.store_analysis): its features, its
    row's properties and ``store`` (the index rows, Search by Music's entry, the Variants' fingerprints, the cover as
    the file holds it).
    ``progress(status, task, fraction)``: each stage in words (the status line's, "" for none; the task's)."""
    def report(status: str, task: str, fraction: float) -> None:
        if progress is not None:
            progress(status, task, fraction)

    title, artist, album, comment = extract_tags(path)
    sources = read_tag_sources(path, title, artist, album, comment)
    result = analyze_track(path, analysis_settings(cfg),
                           progress=lambda stage, fraction: report(*track_stage(stage), fraction))
    features = dict(result.wave_overview)
    features["duration_sec"] = result.duration_sec
    features["tempo_segments"] = result.tempo_segments
    features["beats_time_sec"] = result.beats
    features["tempo_global"] = result.main_tempo
    features["phrase_segments_np"] = build_phrase_segments_np(result.phrases)
    features["cue_points_np"] = build_cue_points_np(result.cue_points)
    features["jump_cues_np"] = build_jump_cues_np(result.jump_pairs, canonicalize_labels=True, merge_coincident=True)
    # The automatic result is kept beside the user's JumpCUEs (which edits
    # replace): the JumpCUE optimizer learns from where the two differ.
    features["jump_cues_auto_np"] = {k: np.array(v, copy=True) for k, v in features["jump_cues_np"].items()}
    features["key_segments"] = result.key_segments
    # the automatic result kept beside the user's keys (which edits replace): the key optimizer learns from
    # the tracks whose keys differ from it
    features["key_segments_auto"] = np.array(result.key_segments, dtype=float, copy=True)
    features["sr"] = result.sample_rate

    report("", tr("Processing Metadata"), 0.85)
    fsize, fmtime = _file_stats(path)
    properties = {
        "path": path,
        "title": title,
        "artist": artist,
        "album": album,
        "bpm": result.main_tempo,
        "key": result.main_key,
        "duration_sec": float(result.duration_sec),
        "total_samples": int(get_total_samples(path) or 0),
        "rating": 0,
        "added_ts": int(time.time()),
        "comment": comment,
        "file_mtime": float(fmtime),
        "file_size": int(fsize),
        "tag_sources": sources,
    }
    bpm_segments = build_bpm_segments(result.tempo_segments)
    key_segments_db = build_key_segments(result.key_segments)
    phrase_sections = build_phrase_index_rows(result.phrases, bpm_segments, key_segments_db)
    # stored by the library's owner: the track's row merged with the library's (a reanalysis keeps the user's
    # metadata), its uid given there; the cover as the file holds it (its thumbnail)
    return {"features": features, "properties": properties, "update_db": True,
            "store": {"force": bool(force_analyze), "bpm_segments": bpm_segments, "key_segments": key_segments_db,
                      "phrase_sections": phrase_sections, "music_entry": result.music_index,
                      "variants": result.variants, "cover": cover_data(path)}}


def _analysis_process_entry(
    path: str,
    cfg: config,
    force_analyze: bool,
    taskid: int,
    queue,
) -> None:
    install_parent_watchdog()

    def progress(status: str, task: str, fraction: float) -> None:
        if status:
            queue.put({"type": "status", "taskid": taskid, "status": status})
        queue.put({"type": "progress", "taskid": taskid, "status": task, "progress": float(fraction)})

    try:
        # One BLAS thread: the analysis runs its own threads (HPSS, the CQTs, the tracker's seeds), and
        # BLAS threads on top of them only competed and spun (measured: 30 -> 25 s a track, CPU 47 -> 36 s,
        # every output bit-identical: no BLAS product here splits its sums across threads).
        with threadpool_limits(limits=1, user_api="blas"):
            payload = analyze_for_library(path, cfg, force_analyze, progress)
        payload["taskid"] = taskid
        queue.put({"type": "result", "taskid": taskid, "payload": payload})
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


class AnalysisWorker(SpawnProcessWorker):
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str, float)
    status = QtCore.Signal(str)

    def __init__(
        self,
        path: str,
        cfg: config,
        taskid: int,
        force_analyze: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(
            target=_analysis_process_entry,
            args=(path, cfg, force_analyze, taskid),
            daemon=True,
            process_name=f"Mixlyzer-analysis-{taskid}",
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
            self.error.emit(tr("Analysis process exited unexpectedly ({code}).").format(code=exitcode))
