"""Full-track analysis subprocess worker."""

import traceback
from typing import Any, Dict

from PySide6 import QtCore

from analyzer_core.global_analyzer import precompute_features
from core.config import config
from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog


class _TaskManagerProxy:
    """Proxy object passed into the analyzer process to forward progress updates."""

    def __init__(self, queue: Any):
        self._queue = queue

    def updatetask(self, taskid: int, status: str, progress: float, send_sig: bool = True):
        self._queue.put(
            {
                "type": "progress",
                "taskid": taskid,
                "status": status,
                "progress": float(progress),
            }
        )
        return None


def _analysis_process_entry(
    path: str,
    cfg: config,
    force_analyze: bool,
    taskid: int,
    queue,
) -> None:
    install_parent_watchdog()
    proxy = _TaskManagerProxy(queue)
    payload: Dict[str, Any] | None = None
    try:
        for item in precompute_features(
            path, cfg, proxy, taskid, force_analyze=force_analyze
        ):
            if isinstance(item, dict) and "status" in item:
                queue.put(
                    {
                        "type": "status",
                        "taskid": taskid,
                        "status": item["status"],
                    }
                )
            else:
                payload = dict(item)
        if payload is None:
            raise RuntimeError("No features produced.")
        queue.put(
            {
                "type": "result",
                "taskid": taskid,
                "payload": payload,
            }
        )
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
            self.error.emit(f"Analysis process exited unexpectedly ({exitcode}).")
