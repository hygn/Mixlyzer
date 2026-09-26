"""Orchestration for full-track analysis jobs."""

from __future__ import annotations

import os
from typing import Callable

from PySide6 import QtCore

from analyzer_core.global_analyzer import extract_tags, getAlbumArt
from core.analysis_lib_handler import FeatureNPZStore
from core.config import load_cfg
from core.library_handler import LibraryDB
from core.workers.analysis import AnalysisWorker


ResultCallback = Callable[[dict], None]
StatusCallback = Callable[[str], None]
AlbumArtCallback = Callable[[object], None]
CancelReanalysisCallback = Callable[[str], None]


class AnalysisWorkflow(QtCore.QObject):
    """Own full-analysis requests, cache lookup, workers, and task progress."""

    def __init__(
        self,
        *,
        taskmanager,
        result_callback: ResultCallback,
        reanalysis_result_callback: ResultCallback,
        album_art_callback: AlbumArtCallback,
        cancel_reanalysis_callback: CancelReanalysisCallback,
        status_callback: StatusCallback,
        parent: QtCore.QObject | None = None,
        cfg_loader=load_cfg,
        albumart_provider=getAlbumArt,
        tag_extractor=extract_tags,
    ) -> None:
        super().__init__(parent)
        self.taskmanager = taskmanager
        self._result_callback = result_callback
        self._reanalysis_result_callback = reanalysis_result_callback
        self._album_art_callback = album_art_callback
        self._cancel_reanalysis_callback = cancel_reanalysis_callback
        self._status_callback = status_callback
        self._cfg_loader = cfg_loader
        self._albumart_provider = albumart_provider
        self._tag_extractor = tag_extractor
        self._workers: dict[int, AnalysisWorker] = {}
        self._contexts: dict[int, dict] = {}
        self.current_path: str | None = None

    def analyze(self, path: str) -> None:
        self._start(path, force_analyze=False)

    def reanalyze(self, path: str) -> None:
        self._start(path, force_analyze=True)

    def cancel_all(self, reason: str = "Analysis canceled") -> None:
        for taskid, worker in list(self._workers.items()):
            worker.stop()
            worker.deleteLater()
            try:
                self.taskmanager.rmtask(taskid, reason)
            except Exception:
                pass
            self._workers.pop(taskid, None)
            self._contexts.pop(taskid, None)

    def _start(self, path: str, *, force_analyze: bool) -> None:
        if self._find_inflight_task(path) is not None:
            self._show_status(f"Analysis already in progress: {os.path.basename(path)}")
            return

        self._cancel_reanalysis_callback(
            "Segment reanalysis canceled (track changed)"
        )
        self._show_status(f"Analyzing: {os.path.basename(path)}")
        self.current_path = path

        cfg = self._cfg_loader()
        thumbnail = self._albumart_provider(path)
        title, _artist, _album, _comment = self._tag_extractor(path)
        task_info = self.taskmanager.addtask(
            songname=title,
            thumbnail=thumbnail,
            status="Loading Track",
            progress=0.0,
        )
        taskid = task_info.taskid

        database = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
        database.connect()
        try:
            track = database.get(path)
        finally:
            database.close()
        auto_load = track is not None

        if track is not None and track.uid and not force_analyze:
            try:
                store = FeatureNPZStore(
                    base_dir=cfg.libconfig.libpath,
                    compressed=True,
                )
                features = store.load(track.uid)
            except (FileNotFoundError, ValueError):
                pass
            else:
                self._album_art_callback(thumbnail)
                self._result_callback(
                    {
                        "features": features,
                        "properties": track.to_meta(),
                        "update_db": False,
                        "taskid": taskid,
                        "auto_load": True,
                    }
                )
                return

        worker = AnalysisWorker(
            path,
            cfg,
            taskid,
            force_analyze=force_analyze,
            parent=self,
        )
        self._workers[taskid] = worker
        self._contexts[taskid] = {
            "path": path,
            "force": force_analyze,
            "auto_load": auto_load,
        }
        worker.progress.connect(
            lambda status, progress, tid=taskid: self._on_progress(
                tid, status, progress
            )
        )
        worker.status.connect(
            lambda status, tid=taskid: self._on_status(tid, status)
        )
        worker.error.connect(
            lambda message, tid=taskid: self._on_error(tid, message)
        )
        worker.finished.connect(
            lambda payload, tid=taskid: self._on_success(tid, payload)
        )
        worker.start()

    def _find_inflight_task(self, path: str) -> int | None:
        normalized = os.path.normcase(os.path.normpath(path))
        for taskid, context in self._contexts.items():
            context_path = str(context.get("path") or "").strip()
            if context_path and os.path.normcase(os.path.normpath(context_path)) == normalized:
                return taskid
        return None

    def _on_progress(self, taskid: int, status: str, progress: float) -> None:
        if taskid in self._workers:
            self.taskmanager.updatetask(taskid, status, float(progress))

    def _on_status(self, taskid: int, status: str) -> None:
        if not status:
            return
        path = self._contexts.get(taskid, {}).get("path")
        basename = os.path.basename(path) if path else ""
        self._show_status(f"{status}: {basename}" if basename else status)

    def _on_error(self, taskid: int, message: str) -> None:
        self.taskmanager.rmtask(taskid, message)
        path = self._contexts.get(taskid, {}).get("path")
        basename = os.path.basename(path) if path else ""
        prefix = f"Error ({basename})" if basename else "Error"
        self._show_status(f"{prefix}: {message}")
        print(f"[AnalysisWorker] Error for task {taskid}: {message}")
        self._finalize(taskid)

    def _on_success(self, taskid: int, payload: dict) -> None:
        try:
            context = self._contexts.get(taskid, {})
            result = dict(payload)
            result.setdefault("auto_load", context.get("auto_load", True))
            callback = (
                self._reanalysis_result_callback
                if context.get("force")
                else self._result_callback
            )
            callback(result)
        finally:
            self._finalize(taskid)

    def _finalize(self, taskid: int) -> None:
        worker = self._workers.pop(taskid, None)
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        self._contexts.pop(taskid, None)

    def _show_status(self, message: str) -> None:
        try:
            self._status_callback(message)
        except Exception:
            pass
