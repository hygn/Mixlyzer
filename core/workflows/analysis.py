"""Orchestration for full-track analysis jobs.

A request names its listener: the deck loading the track (AppWindow.load_track: it plays the file at once and takes
the analysis's results when they come: a library track's stored analysis at once) or none (into the library
alone: a file dropped on the library). A file being analyzed takes another request's listener too."""

from __future__ import annotations

import os
from collections import deque
from typing import Callable, Optional

from PySide6 import QtCore

from core.tags.reader import extract_tags, getAlbumArt
from core.analysis_library import jobs as analysis_jobs
from core.library import jobs
from dataclasses import replace

from core.settings.store import current
from core.workers.analysis import AnalysisWorker
from system.i18n import tr
from system.os_support import online_only, path_key, path_too_long


# the library's and a request's listener: the result (its StoredAnalysis ``analysis``, properties) of its track
ResultCallback = Callable[[dict], None]
StatusCallback = Callable[[str], None]
# (the track, a reanalysis, its analysis settings for the run or None, the request's listener or None)
MissingFileCallback = Callable[[object, bool, Optional[dict], Optional[ResultCallback]], None]


class AnalysisWorkflow(QtCore.QObject):
    """Own full-analysis requests, cache lookup, workers, and task progress."""

    def __init__(
        self,
        *,
        library,
        taskmanager,
        result_callback: ResultCallback,
        reanalysis_result_callback: ResultCallback,
        status_callback: StatusCallback,
        missing_file_callback: MissingFileCallback | None = None,
        parent: QtCore.QObject | None = None,
        cfg_loader=current,
        albumart_provider=getAlbumArt,
        tag_extractor=extract_tags,
    ) -> None:
        super().__init__(parent)
        self._library = library          # core.library.service.LibraryService
        self.taskmanager = taskmanager
        self._result_callback = result_callback
        self._reanalysis_result_callback = reanalysis_result_callback
        self._status_callback = status_callback
        self._missing_file_callback = missing_file_callback
        self._cfg_loader = cfg_loader
        self._albumart_provider = albumart_provider
        self._tag_extractor = tag_extractor
        self._workers: dict[int, AnalysisWorker] = {}
        self._contexts: dict[int, dict] = {}     # every request in flight, running or waiting
        self._waiting: deque[int] = deque()      # the requests waiting to start, in their order

    def analyze(self, path: str, listener: ResultCallback | None = None) -> None:
        """The track's analysis: a library track's stored one, else a new one; to ``listener`` (a deck) and
        the library (result_callback)."""
        self._start(path, force_analyze=False, listener=listener)

    def reanalyze(self, path: str, analysis: dict | None = None) -> None:
        """A fresh full analysis; ``analysis``: analysis settings (analysisconfig fields) for this run only,
        over the saved ones (which stay as they are)."""
        self._start(path, force_analyze=True, analysis=analysis)

    def library_track(self, path: str):
        """The library's track of the file ``path`` (however the path spells it), or None."""
        return self._library.snapshot.find_file(path)

    def cancel_all(self, reason: str | None = None) -> None:
        reason = reason or tr("Analysis canceled")
        for taskid in list(self._waiting):    # not started: their tasks only
            self._waiting.remove(taskid)
            self._contexts.pop(taskid, None)
            try:
                self.taskmanager.rmtask(taskid, reason)
            except Exception:
                pass
        for taskid, worker in list(self._workers.items()):
            worker.stop()
            worker.deleteLater()
            try:
                self.taskmanager.rmtask(taskid, reason)
            except Exception:
                pass
            self._workers.pop(taskid, None)
            self._contexts.pop(taskid, None)

    def _start(self, path: str, *, force_analyze: bool, analysis: dict | None = None,
               listener: ResultCallback | None = None) -> None:
        if path_too_long(path):
            # Python, soundfile and the tags would fail on it one by one: said once, before anything opens it
            self._show_status(tr("The path is too long for Windows ({count} characters) while Windows' long paths "
                                 "are off: {path}").format(count=len(os.path.abspath(path)), path=path))
            return
        if online_only(path):    # a sync client's placeholder: read anyway, said (its download takes a while)
            self._show_status(tr("{file} is kept online only: it is downloaded before it is read.").format(
                file=os.path.basename(path)))
        cfg = self._cfg_loader()
        if analysis:
            cfg = replace(cfg, analysisconfig=replace(cfg.analysisconfig, **analysis))
        # a file in the library goes by the path it is stored under, however ``path`` spells it (a drop's
        # separators, another case, a link)
        track = self.library_track(path)
        if track is not None:
            path = track.path
            if not os.path.isfile(path):
                # the track's file is gone: no guessing, the user sets its new path
                self._show_status(tr("File not found: {path}").format(path=path))
                if self._missing_file_callback is not None:
                    self._missing_file_callback(track, force_analyze, analysis, listener)
                return

        inflight = self._find_inflight_task(path)
        if inflight is not None:
            # its result goes to this request's listener too
            if listener is not None:
                self._contexts[inflight].setdefault("listeners", []).append(listener)
                if inflight in self._waiting:      # now the deck's: ahead of the others waiting
                    self._waiting.remove(inflight)
                    self._waiting.appendleft(inflight)
            self._show_status(tr("Analysis already in progress: {file}").format(file=os.path.basename(path)))
            return
        if track is not None and listener is None and not force_analyze:   # in the library: analyzed
            self._show_status(tr("Already in the library: {file}").format(file=os.path.basename(path)))
            return

        thumbnail = self._albumart_provider(path)
        title, _artist, _album, _comment = self._tag_extractor(path)
        task_info = self.taskmanager.addtask(
            songname=title,
            thumbnail=thumbnail,
            status=tr("Loading Track"),
            progress=0.0,
        )
        taskid = task_info.taskid

        if track is not None and track.uid and not force_analyze:
            # its stored analysis, read on the library thread; none readable: analyzed
            def loaded(analysis) -> None:
                if analysis is None:
                    self._analyze(path, cfg, taskid, thumbnail, force_analyze, listener)
                    return
                self._show_status(tr("Loaded from the library: {file}").format(file=os.path.basename(path)))
                result = {"analysis": analysis, "properties": track.to_meta(), "update_db": False,
                          "taskid": taskid, "album_art": thumbnail}
                self._result_callback(result)
                if listener is not None:
                    listener(result)

            self._library.read(analysis_jobs.analysis, track.uid, done=loaded)
            return
        self._analyze(path, cfg, taskid, thumbnail, force_analyze, listener)

    def _analyze(self, path: str, cfg, taskid: int, thumbnail, force_analyze: bool,
                 listener: ResultCallback | None) -> None:
        """A full analysis asked for: started now while fewer than analysisconfig.max_parallel_analyses run, else
        waiting its turn (the deck's track ahead of the files only going into the library)."""
        self._contexts[taskid] = {
            "path": path,
            "cfg": cfg,
            "force": force_analyze,
            "listeners": [listener] if listener is not None else [],
            "album_art": thumbnail,
        }
        if listener is not None:
            self._waiting.appendleft(taskid)
        else:
            self._waiting.append(taskid)
        self._start_waiting()
        if taskid in self._waiting:
            self.taskmanager.updatetask(taskid, tr("Waiting for a free analysis slot"), 0.0)
            self._show_status(tr("Waiting to analyze: {file}").format(file=os.path.basename(path)))

    def _parallel_limit(self) -> int:
        """The full analyses run at once (Settings > Analysis > Global), as the settings say now."""
        try:
            return max(1, int(self._cfg_loader().analysisconfig.max_parallel_analyses))
        except Exception:
            return 1

    @QtCore.Slot(object, object)
    def settings_changed(self, _old, _new) -> None:
        """The settings saved (core.settings.store): more analyses allowed at once start the waiting ones now (fewer
        leave the running ones to finish)."""
        self._start_waiting()

    def _start_waiting(self) -> None:
        """The waiting analyses started, in their order, while there is room."""
        limit = self._parallel_limit()
        while self._waiting and len(self._workers) < limit:
            taskid = self._waiting.popleft()
            if taskid in self._contexts:
                self._start_worker(taskid)

    def _start_worker(self, taskid: int) -> None:
        context = self._contexts[taskid]
        path = context["path"]
        self._show_status(tr("Analyzing: {file}").format(file=os.path.basename(path)))
        worker = AnalysisWorker(
            path,
            context.pop("cfg"),
            taskid,
            parent=self,
        )
        self._workers[taskid] = worker
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
        normalized = path_key(path)
        for taskid, context in self._contexts.items():
            context_path = str(context.get("path") or "").strip()
            if context_path and path_key(context_path) == normalized:
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
        prefix = tr("Error ({file})").format(file=basename) if basename else tr("Error")
        self._show_status(f"{prefix}: {message}")
        print(f"[AnalysisWorker] Error for task {taskid}: {message}")
        self._finalize(taskid)

    def _on_success(self, taskid: int, payload: dict) -> None:
        """The analysis's results, stored by the library's owner (jobs.store_analysis: the analysis process does
        not open the library), then handed on (the deck, the library); the request stays in flight until then."""
        if payload.get("store") is None:
            self._deliver(taskid, payload)
            return
        self.taskmanager.updatetask(taskid, tr("Saving Result"), 0.95)
        self._library.write(
            jobs.store_analysis, payload, done=lambda stored: self._deliver(taskid, stored),
            failed=lambda exc: self._on_error(taskid, tr("The analysis could not be stored: {error}").format(
                error=exc)))

    def _deliver(self, taskid: int, payload: dict) -> None:
        try:
            context = self._contexts.get(taskid, {})
            result = dict(payload)
            result.setdefault("album_art", context.get("album_art"))
            callback = (
                self._reanalysis_result_callback
                if context.get("force")
                else self._result_callback
            )
            callback(result)
            for listener in context.get("listeners", []):
                listener(result)
        finally:
            self._finalize(taskid)

    def _finalize(self, taskid: int) -> None:
        worker = self._workers.pop(taskid, None)
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        self._contexts.pop(taskid, None)
        self._start_waiting()      # its slot to the next waiting

    def _show_status(self, message: str) -> None:
        try:
            self._status_callback(message)
        except Exception:
            pass
