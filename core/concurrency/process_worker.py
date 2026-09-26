"""Shared Qt lifecycle for spawned worker processes."""

from __future__ import annotations

import multiprocessing as mp
import os
from queue import Empty
import threading
import time
from typing import Any, Callable

from PySide6 import QtCore


def install_parent_watchdog() -> None:
    """Exit a child promptly when its owning application process disappears."""
    parent = mp.parent_process()
    if parent is None:
        return

    def monitor() -> None:
        while True:
            try:
                alive = parent.is_alive()
            except Exception:
                alive = False
            if not alive:
                os._exit(1)
            time.sleep(0.5)

    threading.Thread(target=monitor, name="ParentWatchdog", daemon=True).start()


class SpawnProcessWorker(QtCore.QObject):
    """Spawn a process and drain its message queue from the Qt event loop."""

    def __init__(
        self,
        *,
        target: Callable[..., None],
        args: tuple[Any, ...],
        daemon: bool,
        process_name: str,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._target = target
        self._args = args
        self._daemon = bool(daemon)
        self._process_name = str(process_name)
        self._ctx = mp.get_context("spawn")
        self._queue: Any | None = None
        self._process: mp.Process | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll_queue)

    @property
    def process_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Worker already started.")
        self._queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=self._target,
            args=(*self._args, self._queue),
            name=self._process_name,
        )
        self._process.daemon = self._daemon
        self._process.start()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._cleanup()

    def _poll_queue(self) -> None:
        if self._queue is None:
            return
        while self._queue is not None:
            try:
                message = self._queue.get_nowait()
            except Empty:
                break
            self._handle_process_message(message)
            if message.get("type") == "done":
                self._finalize_process(graceful=True)
                return

        process = self._process
        if process is not None and not process.is_alive():
            self._finalize_process(graceful=False)

    def _finalize_process(self, *, graceful: bool) -> None:
        self._timer.stop()
        process = self._process
        exitcode = None
        if process is not None:
            process.join(timeout=1.0)
            exitcode = process.exitcode
        self._cleanup()
        self._process_exited(graceful=graceful, exitcode=exitcode)

    def _cleanup(self) -> None:
        if self._queue is not None:
            self._queue.close()
            self._queue.join_thread()
            self._queue = None
        if self._process is not None:
            self._process.close()
            self._process = None

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        raise NotImplementedError

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        pass

    def __del__(self):
        try:
            self.stop()
        except RuntimeError:
            # Qt may destroy the owned QTimer before Python finalizes this wrapper.
            pass
