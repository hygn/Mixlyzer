"""Full-track analysis subprocess worker: the analyzer's pipeline (analyzer.pipeline.track) run on a track, and
what the library stores of it made from its results (the track's row, the stored analysis, the index rows)."""

import os
import time
import traceback
from typing import Any, Callable, Dict, Optional

from PySide6 import QtCore
from threadpoolctl import threadpool_limits

from analyzer.audio.decoder import get_total_samples
from analyzer.pipeline.track import LAZY_MODULES, analyze_track
from core.tags.reader import cover_data, extract_tags, read_tag_sources
from system.process_worker import ResidentJobWorker, ResidentPool, install_parent_watchdog
from core.settings.config import config
from core.library.segments import track_index
from core.workers.analysis_settings import analysis_settings
from core.workers.analysis_texts import error_text, track_stage
from system.i18n import tr
from core.analysis_library.stored import arrays_of


def _file_stats(path: str) -> tuple[int, float]:
    try:
        st = os.stat(path)
        return int(st.st_size), float(st.st_mtime)
    except FileNotFoundError:
        return 0, 0.0


def analyze_for_library(path: str, cfg: config,
                        progress: Optional[Callable[[str, str, float], None]] = None) -> Dict[str, Any]:
    """The track analyzed, as the library's owner stores it (core.library.jobs.store_analysis): its stored analysis
    (``arrays``: core.analysis_library.stored), its row's properties and ``store`` (the index rows, Search by Music's
    entry, the Variants' fingerprints, the mix candidates' beats' features, the cover as the file holds it).
    ``progress(status, task, fraction)``: each stage in words (the status line's, "" for none; the task's)."""
    def report(status: str, task: str, fraction: float) -> None:
        if progress is not None:
            progress(status, task, fraction)

    title, artist, album, comment = extract_tags(path)
    sources = read_tag_sources(path, title, artist, album, comment)
    result = analyze_track(path, analysis_settings(cfg),
                           progress=lambda stage, fraction: report(*track_stage(stage), fraction))

    report("", tr("Processing Metadata"), 0.85)
    fsize, fmtime = _file_stats(path)
    # library.db's rows of the track, by the rule the editor's save and the library's sync write them by
    index = track_index(result.tempo_segments, result.key_segments, result.phrases)
    properties = {
        "path": path,
        "title": title,
        "artist": artist,
        "album": album,
        "bpm": index.bpm,
        "key": index.key,
        "duration_sec": float(result.duration_sec),
        "total_samples": int(get_total_samples(path) or 0),
        "rating": 0,
        "added_ts": int(time.time()),
        "comment": comment,
        "file_mtime": float(fmtime),
        "file_size": int(fsize),
        "tag_sources": sources,
    }
    # stored by the library's owner: the track's row merged with the library's (a track in it keeps the user's
    # metadata), its uid given there, and its rows made from the marks (``index``); the cover as the file holds it
    # (its thumbnail)
    return {"arrays": arrays_of(result), "properties": properties, "update_db": True,
            "store": {"index": index, "music_entry": result.music_index,
                      "variants": result.variants, "mix": result.mix_features, "cover": cover_data(path)}}


def _analysis_process_entry(
    path: str,
    cfg: config,
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
            payload = analyze_for_library(path, cfg, progress)
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


_pool: ResidentPool | None = None


def analysis_pool() -> ResidentPool:
    """The resident full-analysis processes (system.process_worker.ResidentPool; made on the GUI thread at the first
    analysis): as many as the analyses run at once (AnalysisWorkflow keeps that count), each ending after 10 minutes
    without a track and replaced after 20 tracks; while it waits for its first, it imports what an analysis reaches
    (librosa's modules too)."""
    global _pool
    if _pool is None:
        _pool = ResidentPool("Mixlyzer-analysis", idle_sec=600.0, max_jobs=20,
                             warm=("analyzer.pipeline.track", "core.workers.analysis", *LAZY_MODULES))
    return _pool


class AnalysisWorker(ResidentJobWorker):
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str, float)
    status = QtCore.Signal(str)

    def __init__(
        self,
        path: str,
        cfg: config,
        taskid: int,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(
            pool=analysis_pool(),
            target=_analysis_process_entry,
            args=(path, cfg, taskid),
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
