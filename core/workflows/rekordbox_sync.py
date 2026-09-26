from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from PySide6 import QtCore

from core.library_handler import TrackRow
from core.workers.rekordbox_sync import (
    RekordboxSyncRequest,
    RekordboxSyncResult,
    RekordboxSyncWorker,
)


class RekordboxXmlSync(QtCore.QObject):
    """Queue Rekordbox sync requests and own each worker thread."""

    started = QtCore.Signal(object)
    progress = QtCore.Signal(object, int, str)
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(object, str)

    def __init__(self, *, cfg_getter: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._cfg_getter = cfg_getter
        self._queue: list[RekordboxSyncRequest] = []
        self._active_request: RekordboxSyncRequest | None = None
        self._thread: QtCore.QThread | None = None
        self._worker: RekordboxSyncWorker | None = None

    @QtCore.Slot(object)
    def sync_incremental(self, rows: object) -> None:
        track_rows = self._coerce_rows(rows)
        if track_rows is None:
            return
        request = self._request_from_config(mode="incremental", rows=tuple(track_rows))
        if request is not None:
            self._enqueue(request)

    @QtCore.Slot(object)
    def sync_requested(self, payload: object) -> None:
        if isinstance(payload, dict):
            full_rebuild = bool(payload.get("full_rebuild", True))
            show_progress = bool(payload.get("show_progress", False))
        else:
            full_rebuild = bool(payload)
            show_progress = False
        request = self._request_from_config(
            mode="library",
            full_rebuild=full_rebuild,
            show_progress=show_progress,
        )
        if request is not None:
            self._enqueue(request)

    @QtCore.Slot(str)
    def sync_track_requested(self, uid: str) -> None:
        track_uid = str(uid or "").strip()
        if not track_uid:
            return
        request = self._request_from_config(mode="track", uid=track_uid)
        if request is not None:
            self._enqueue(request)

    def _request_from_config(
        self, *, mode: str, **kwargs
    ) -> RekordboxSyncRequest | None:
        cfg = self._cfg_getter()
        libcfg = getattr(cfg, "libconfig", None) if cfg is not None else None
        if libcfg is None or not bool(
            getattr(libcfg, "rekordbox_sync_enabled", False)
        ):
            return None
        xml_path_text = str(
            getattr(libcfg, "rekordbox_xml_path", "") or ""
        ).strip()
        if not xml_path_text:
            return None
        return RekordboxSyncRequest(
            library_dir=Path(str(libcfg.libpath)).expanduser(),
            xml_path=Path(xml_path_text).expanduser(),
            mode=mode,
            **kwargs,
        )

    def _enqueue(self, request: RekordboxSyncRequest) -> None:
        self._queue.append(request)
        if self._thread is None:
            self._start_next()

    def _start_next(self) -> None:
        if self._thread is not None or not self._queue:
            return
        request = self._queue.pop(0)
        thread = QtCore.QThread(self)
        worker = RekordboxSyncWorker(request)
        worker.moveToThread(thread)
        worker.progress.connect(self._relay_progress)
        worker.finished.connect(self._job_finished)
        worker.failed.connect(self._job_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        thread.started.connect(worker.run)
        self._active_request = request
        self._thread = thread
        self._worker = worker
        self.started.emit(request)
        thread.start()

    @QtCore.Slot(int, str)
    def _relay_progress(self, value: int, message: str) -> None:
        if self._active_request is not None:
            self.progress.emit(self._active_request, int(value), str(message))

    @QtCore.Slot(object)
    def _job_finished(self, result: RekordboxSyncResult) -> None:
        if self._active_request is not None:
            self.finished.emit(self._active_request, result)

    @QtCore.Slot(str)
    def _job_failed(self, message: str) -> None:
        if self._active_request is not None:
            self.failed.emit(self._active_request, str(message))

    @QtCore.Slot()
    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._active_request = None
        QtCore.QTimer.singleShot(0, self._start_next)

    @staticmethod
    def _coerce_rows(rows: object) -> list[TrackRow] | None:
        if rows is None:
            return []
        result: list[TrackRow] = []
        try:
            iterator: Iterable = iter(rows)
        except TypeError:
            return None
        for row in iterator:
            if isinstance(row, TrackRow):
                result.append(row)
            elif isinstance(row, dict):
                try:
                    result.append(TrackRow(**row))
                except Exception:
                    continue
        return result
