import multiprocessing as mp
import os
import threading
import time
import traceback
from queue import Empty
from typing import Any, Dict

from PySide6 import QtCore

from analyzer_core.global_analyzer import precompute_features
from core.config import config


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
    _install_parent_watchdog()
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
    except Exception as exc:  # pragma: no cover - defensive path
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


def _install_parent_watchdog() -> None:
    parent = mp.parent_process()
    if parent is None:
        return

    def _monitor() -> None:
        while True:
            try:
                alive = parent.is_alive()
            except Exception:
                alive = False
            if not alive:
                os._exit(1)
            time.sleep(0.5)

    thread = threading.Thread(target=_monitor, name="ParentWatchdog", daemon=True)
    thread.start()


class AnalysisWorker(QtCore.QObject):
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
        super().__init__(parent)
        self._path = path
        self._config = cfg
        self._taskid = taskid
        self._force_analyze = force_analyze
        self._ctx = mp.get_context("spawn")
        self._queue: Any | None = None
        self._process: mp.Process | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll_queue)
        self._result_emitted = False
        self._error_emitted = False

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Worker already started.")
        self._queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_analysis_process_entry,
            args=(
                self._path,
                self._config,
                self._force_analyze,
                self._taskid,
                self._queue,
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
