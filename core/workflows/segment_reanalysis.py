from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
from PySide6 import QtCore

from core.tags.reader import getAlbumArt
from core.settings.config import load_cfg
from core.workers.segment_reanalysis import JUMPCUE, KEY_DYNAMIC, PHRASE, ReanalysisRequest, ReanalysisResult, \
    SegmentReanalysisWorker
from core.workflows.event_bus import EventBus
from core.track.model import DataModel
from system.i18n import tr


StatusCallback = Callable[[str], None]
PathGetter = Callable[[], Optional[str]]


class SegmentReanalysisWorkflow(QtCore.QObject):
    """Orchestrates beat/key reanalysis requests outside of the main window; their results go to the loaded track
    (``model``) as edits."""

    def __init__(
        self,
        *,
        bus: EventBus,
        model: DataModel,
        taskmanager,
        get_current_path: PathGetter,
        status_callback: Optional[StatusCallback] = None,
        parent: QtCore.QObject | None = None,
        cfg_loader=load_cfg,
        albumart_provider=getAlbumArt,
    ) -> None:
        super().__init__(parent)
        self.bus = bus
        self.model = model
        self.taskmanager = taskmanager
        self._get_current_path = get_current_path
        self._status_callback = status_callback or (lambda _msg: None)
        self._cfg_loader = cfg_loader
        self._albumart_provider = albumart_provider
        self._workers: dict[int, SegmentReanalysisWorker] = {}
        self._contexts: dict[int, dict] = {}

        self.bus.sig_segment_reanalyze_requested.connect(self._on_request)

    def cancel_all(self, reason: str | None = None) -> None:
        if not self._workers:
            return
        reason = reason or tr("Segment reanalysis canceled")
        for taskid, worker in list(self._workers.items()):
            if worker is not None:
                worker.stop()
                worker.deleteLater()
            try:
                self.taskmanager.rmtask(taskid, reason)
            except Exception:
                pass
            self._workers.pop(taskid, None)
            self._contexts.pop(taskid, None)
        self.bus.sig_segment_reanalyze_state.emit(True)
        self._show_status(reason)

    # internal helpers
    def _on_request(self, request: ReanalysisRequest | None) -> None:
        if request is None:
            return
        path = str(request.path) if request.path else self._get_current_path()
        if not path:
            self._show_status(tr("Load a track before segment reanalysis"))
            return

        kind = request.kind
        beats = np.asarray(request.beats, dtype=float)
        segments_input = np.asarray(request.tempo_segments, dtype=float)
        seg_index = 0 if kind in (JUMPCUE, PHRASE) else int(request.segment_index)
        selection = self._parse_selection(request.selection) if kind.startswith("key") else None
        key_segments_arr = None
        if request.key_segments is not None:
            try:
                key_segments_arr = np.asarray(request.key_segments, dtype=float)
            except Exception:
                key_segments_arr = None
        valid, seg_index = self._validate_request(
            analyze_type=kind,
            segments_input=segments_input,
            beats=beats,
            key_segments=key_segments_arr,
            seg_index=seg_index,
            selection=selection,
        )
        if not valid:
            return

        cfg = self._cfg_loader()
        thumb = self._albumart_provider(path)
        props = getattr(self.model, "properties", {}) or {}
        songname = props.get("title") or os.path.basename(path)
        if kind == JUMPCUE:
            task_song = f"{songname} (JumpCUE)"
            task_status = tr("JumpCUE reanalyze")
        elif kind == PHRASE:
            task_song = tr("{song} (Phrase)").format(song=songname)
            task_status = tr("Phrase reanalyze")
        elif kind.startswith("key"):
            task_song = tr("{song} (Key)").format(song=songname)
            task_status = tr("Key reanalyze")
        else:
            task_song = tr("{song} (Seg {index})").format(song=songname, index=seg_index + 1)
            task_status = tr("Segment reanalyze")
        task = self.taskmanager.addtask(
            songname=task_song,
            thumbnail=thumb,
            status=task_status,
            progress=0.0,
        )

        worker = SegmentReanalysisWorker(
            path=path,
            cfg=cfg,
            request=replace(request, beats=beats, tempo_segments=segments_input, key_segments=key_segments_arr,
                            selection=selection, segment_index=seg_index),
            taskid=task.taskid,
            parent=self,
        )
        worker.finished.connect(lambda result, tid=task.taskid: self._on_worker_finished(tid, result))
        worker.error.connect(lambda message, tid=task.taskid: self._on_worker_error(tid, message))
        worker.progress.connect(
            lambda status, prog, tid=task.taskid: self.taskmanager.updatetask(tid, status or "", prog)
        )
        worker.status.connect(
            lambda status, tid=task.taskid: self.taskmanager.updatetask(tid, status or "", 0.0)
        )
        worker.start()
        self._workers[task.taskid] = worker
        self._contexts[task.taskid] = {"analyze_type": kind}
        self.bus.sig_segment_reanalyze_state.emit(False)

    def _validate_request(
        self,
        *,
        analyze_type: str,
        segments_input: np.ndarray,
        beats: np.ndarray,
        key_segments: np.ndarray | None,
        seg_index: int,
        selection,
    ) -> tuple[bool, int]:
        seg_out = seg_index
        if analyze_type == JUMPCUE:
            if beats.size == 0:
                self._show_status(tr("No beatgrid available for JumpCUE reanalysis"))
                return False, seg_out
            return True, max(seg_out, 0)
        if analyze_type == PHRASE:
            if beats.size == 0:
                self._show_status(tr("No beatgrid available for phrase reanalysis"))
                return False, seg_out
            if segments_input.ndim != 2 or segments_input.shape[0] == 0:
                self._show_status(tr("No tempo segments available for phrase reanalysis"))
                return False, seg_out
            return True, max(seg_out, 0)
        if analyze_type == KEY_DYNAMIC:
            if segments_input.ndim != 2 or segments_input.shape[0] == 0:
                self._show_status(tr("No tempo segments available for key reanalysis"))
                return False, seg_out
            if key_segments is None or key_segments.size == 0:
                self._show_status(tr("No key segments available for reanalysis"))
                return False, seg_out
        elif analyze_type.startswith("key"):
            if key_segments is None or key_segments.ndim != 2 or key_segments.shape[0] == 0:
                self._show_status(tr("No key segments available for reanalysis"))
                return False, seg_out
            if selection is None:
                self._show_status("Select a key range to reanalyze")
                return False, seg_out
            try:
                sel_start = float(selection[0])
            except Exception:
                self._show_status(tr("Invalid key selection"))
                return False, seg_out
            idx = int(np.searchsorted(key_segments[:, 2], sel_start, side="right") - 1)
            seg_out = int(np.clip(idx, 0, len(key_segments) - 1))
        else:
            if segments_input.ndim != 2 or segments_input.shape[0] == 0 or not (0 <= seg_index < len(segments_input)):
                self._show_status("Select a valid segment to reanalyze")
                return False, seg_out
        return True, seg_out

    def _parse_selection(self, selection_payload):
        if selection_payload is None:
            return None
        try:
            sel_start, sel_end = selection_payload
            selection = (float(sel_start), float(sel_end))
        except Exception:
            selection = None
        return selection

    def _on_worker_finished(self, taskid: int, result: ReanalysisResult) -> None:
        worker = self._workers.pop(taskid, None)
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        ctx = self._contexts.pop(taskid, {})
        analyze_type = ctx.get("analyze_type", "beat")
        if analyze_type.startswith("key"):
            self._handle_key_result(taskid, result)
            self._show_status(tr("Key reanalysis completed"))
        elif analyze_type == JUMPCUE:
            self._handle_jumpCUE_result(taskid, result)
            self._show_status(tr("JumpCUE reanalysis completed"))
        elif analyze_type == PHRASE:
            self._handle_phrase_result(taskid, result)
            self._show_status(tr("Phrase reanalysis completed"))
        else:
            self._handle_segment_result(taskid, result)
            self._show_status(tr("Segment reanalysis completed"))
        if not self._workers:
            self.bus.sig_segment_reanalyze_state.emit(True)
    
    # A result is an edit of the loaded track like any other (core.track.model: undoable, saved with the editor's
    # next save); an automatic result beside it (a whole-track reanalysis's) is what the optimizers compare the
    # user's edits with.

    def _handle_jumpCUE_result(self, taskid: int, result: ReanalysisResult):
        if result.jump_cues is not None:
            self.model.set_automatic(jump_cues=result.jump_cues)
            try:
                self.model.set_jump_cues(list(result.jump_cues))
            except ValueError as exc:
                self._show_status(tr("JumpCUE reanalysis failed: {error}").format(error=exc))
        self.taskmanager.updatetask(taskid, tr("JumpCUE reanalysis finished"), 1.0)
        self.taskmanager.rmtask(taskid)

    def _handle_phrase_result(self, taskid: int, result: ReanalysisResult) -> None:
        phrases = result.phrases
        if isinstance(phrases, list):
            # the CUE points stay: the Phrase page's CUE button places them from the new phrases
            self.model.set_phrases(phrases, normalize=True)
        self.taskmanager.updatetask(taskid, tr("Phrase reanalysis finished"), 1.0)
        self.taskmanager.rmtask(taskid)

    def _handle_key_result(self, taskid: int, result: ReanalysisResult) -> None:
        key_segments = np.asarray(result.key_segments if result.key_segments is not None else (), dtype=float)
        auto = np.asarray(result.key_segments_auto if result.key_segments_auto is not None else (), dtype=float)
        if auto.ndim == 2 and auto.size:
            self.model.set_automatic(key_segments=auto)
        if key_segments.ndim == 2 and key_segments.size:
            self.model.set_key_segments(key_segments)
        self.taskmanager.updatetask(taskid, tr("Key reanalysis finished"), 1.0)
        self.taskmanager.rmtask(taskid)

    def _handle_segment_result(self, taskid: int, result: ReanalysisResult) -> None:
        beats = np.asarray(result.beats if result.beats is not None else (), dtype=float)
        segments = np.asarray(result.tempo_segments if result.tempo_segments is not None else (), dtype=float)
        if beats.size and segments.ndim == 2 and segments.size:
            self.model.set_beatgrid(beats, segments)
        self.taskmanager.updatetask(taskid, tr("Segment reanalysis finished"), 1.0)
        self.taskmanager.rmtask(taskid)

    def _on_worker_error(self, taskid: int, message: str) -> None:
        worker = self._workers.pop(taskid, None)
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        self._contexts.pop(taskid, None)
        self.taskmanager.rmtask(taskid, message)
        self._show_status(tr("Segment reanalysis failed: {error}").format(error=message))
        if not self._workers:
            self.bus.sig_segment_reanalyze_state.emit(True)

    def _show_status(self, message: str) -> None:
        try:
            self._status_callback(message)
        except Exception:
            pass
