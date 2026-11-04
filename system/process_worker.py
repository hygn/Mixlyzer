"""Shared Qt lifecycle for spawned worker processes."""

from __future__ import annotations

import multiprocessing as mp
import os
from queue import Empty
import threading
import time
from typing import Any, Callable

from PySide6 import QtCore

GRACEFUL_END_SEC = 30.0     # a process that said it is done, given this long to end by itself before it is terminated


_watchdog = threading.Event()   # set once the watchdog runs (a resident worker's jobs each call install)


def install_parent_watchdog() -> None:
    """Exit a child promptly when its owning application process disappears (once per process)."""
    parent = mp.parent_process()
    if parent is None or _watchdog.is_set():
        return
    _watchdog.set()

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
        # Process.start() runs on a starter thread (see start); the lock guards the
        # hand-over between it and stop()
        self._start_lock = threading.Lock()
        self._started = False
        self._start_failed = False
        self._abandoned = False
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
        # On Windows Process.start() returns only once the child has read all it is
        # sent, and the child reads its arguments after re-importing the app's main
        # module: seconds of a frozen UI. Started off the GUI thread instead; the
        # queue is polled once it has started.
        threading.Thread(
            target=self._start_process,
            args=(self._process, self._queue),
            name=f"{self._process_name}-start",
            daemon=True,
        ).start()
        self._timer.start()

    def _start_process(self, process: mp.Process, queue: Any) -> None:
        """Starter thread: start the process; when stop() came first, end it here."""
        failed = False
        try:
            process.start()
        except Exception:
            failed = True
        with self._start_lock:
            self._started = True
            self._start_failed = failed
            abandoned = self._abandoned
        if abandoned:
            _end_process(process, queue, started=not failed, wait=0.0)

    def stop(self) -> None:
        self._timer.stop()
        with self._start_lock:
            if self._process is not None and not self._started:
                # still starting: the starter thread ends the process once it is up
                self._abandoned = True
                self._process = None
                self._queue = None
                return
        process, queue = self._process, self._queue
        self._process = self._queue = None
        if process is not None:
            _release(process, queue, started=not self._start_failed, wait=0.0)

    def _poll_queue(self) -> None:
        if self._queue is None:
            return
        with self._start_lock:
            started, failed = self._started, self._start_failed
        if not started:
            return
        if failed:
            self._finalize_process(graceful=False)
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
        """The process done (``graceful``: it said so) or gone. One that said it is done is left to end by itself
        on a thread of its own (terminated if it has not in GRACEFUL_END_SEC): waited for here, a process still
        ending when the CPU is full (a batch analysis) held the GUI and was closed while it ran (a ValueError)."""
        self._timer.stop()
        process, queue = self._process, self._queue
        self._process = self._queue = None
        started = process is not None and not self._start_failed
        exitcode = None
        if started and not graceful:
            process.join(timeout=1.0)       # gone already: at once
            exitcode = process.exitcode
        if process is not None:
            _release(process, queue, started=started, wait=GRACEFUL_END_SEC if graceful else 0.0)
        self._process_exited(graceful=graceful, exitcode=exitcode)

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


def _end_process(process: mp.Process, queue: Any, *, started: bool, wait: float) -> None:
    """A process released: given ``wait`` seconds to end by itself, then terminated, then closed with its queue."""
    if started:
        process.join(timeout=wait)
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
    try:
        process.close()
    except ValueError:
        pass  # still running after terminate: left to the OS
    if queue is not None:
        queue.close()
        queue.join_thread()


def _release(process: mp.Process, queue: Any, *, started: bool, wait: float) -> None:
    """``_end_process`` on a thread of its own (never the GUI's waiting); in place while Python ends (no thread)."""
    try:
        threading.Thread(target=_end_process, args=(process, queue), kwargs={"started": started, "wait": wait},
                         name=f"{process.name}-end", daemon=True).start()
    except RuntimeError:
        _end_process(process, queue, started=started, wait=0.0)


# ---------------------------------------------------------------- resident workers
# A resident process runs one job after another, so a job pays no process start, no imports and no first-use costs
# (numba, the models: about 7 s of a full analysis). Between jobs it hands its resident memory
# back (system.os_support.trim_memory: a few MB while it waits, the next job no slower); after ``idle_sec`` without a
# job it ends, and after ``max_jobs`` jobs it is replaced (a long session's leaks never pile up). A job's target is
# what a SpawnProcessWorker would run (a module-level callable taking its args and then a queue): it is called with a
# queue-like sink tagging its messages with the job, and its "done" ends the job. A job stopped while it runs ends
# its process (a fresh one takes the next job).

class _JobSink:
    """The ``queue`` a job's target puts its messages on: they go out tagged with the job."""

    def __init__(self, out_queue: Any, job_id: int) -> None:
        self._out, self._job = out_queue, job_id

    def put(self, message: dict[str, Any]) -> None:
        self._out.put((self._job, message))


def _resident_main(cmd_queue: Any, out_queue: Any, idle_sec: float | None, max_jobs: int,
                   warm: tuple[str, ...]) -> None:
    """A resident process: its warm-up imports, then jobs from ``cmd_queue`` until one is None, ``idle_sec`` pass
    without one, or ``max_jobs`` are done."""
    import gc
    import importlib
    import traceback

    from system.os_support import trim_memory
    install_parent_watchdog()
    for name in warm:
        try:
            importlib.import_module(name)
        except Exception:
            traceback.print_exc()
    done = 0
    while True:
        try:
            job = cmd_queue.get(timeout=idle_sec) if idle_sec else cmd_queue.get()
        except Empty:
            break
        if job is None:
            break
        job_id, target, args = job
        sink = _JobSink(out_queue, job_id)
        try:
            target(*args, sink)
        except BaseException as exc:      # a target ends its own errors; a stray one still ends the job
            traceback.print_exc()
            sink.put({"type": "error", "message": repr(exc)})
            sink.put({"type": "done"})
        done += 1
        gc.collect()
        trim_memory()
        out_queue.put((job_id, {"type": "__idle__"}))
        if max_jobs and done >= max_jobs:
            break


class _Resident:
    def __init__(self, process: mp.Process, cmd_queue: Any, out_queue: Any) -> None:
        self.process, self.cmd_queue, self.out_queue = process, cmd_queue, out_queue
        self.job: ResidentJobWorker | None = None
        self.starting = True


class ResidentPool(QtCore.QObject):
    """Resident processes taking jobs (``ResidentJobWorker``): a job goes to an idle one, else a new one is started
    (the caller keeps the count of jobs at once: the analysis workflow its limit). Lives on the GUI thread."""

    def __init__(self, name: str, *, idle_sec: float | None, max_jobs: int, warm: tuple[str, ...] = (),
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._name, self._idle_sec, self._max_jobs, self._warm = name, idle_sec, int(max_jobs), tuple(warm)
        self._ctx = mp.get_context("spawn")
        self._residents: list[_Resident] = []
        self._count = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

    def _spawn(self) -> _Resident:
        self._count += 1
        cmd_queue, out_queue = self._ctx.Queue(), self._ctx.Queue()
        process = self._ctx.Process(target=_resident_main,
                                    args=(cmd_queue, out_queue, self._idle_sec, self._max_jobs, self._warm),
                                    name=f"{self._name}-{self._count}", daemon=True)
        resident = _Resident(process, cmd_queue, out_queue)

        def start() -> None:               # Process.start() off the GUI thread (as SpawnProcessWorker.start)
            try:
                process.start()
            except Exception:
                resident.process = None
            resident.starting = False
        threading.Thread(target=start, name=f"{process.name}-start", daemon=True).start()
        self._residents.append(resident)
        self._timer.start()
        return resident

    def prewarm(self) -> None:
        """One resident started ahead of its first job (none when there is one)."""
        if not self._residents:
            self._spawn()

    def submit(self, worker: ResidentJobWorker) -> None:
        resident = next((r for r in self._residents if r.job is None and r.process is not None
                         and (r.starting or r.process.is_alive())), None) or self._spawn()
        resident.job = worker
        worker._resident = resident
        resident.cmd_queue.put((worker.job_id, worker._target, worker._args))

    def cancel(self, worker: ResidentJobWorker) -> None:
        """A job stopped: its process ended (a job cannot be interrupted), the next job takes another."""
        resident = worker._resident
        if resident is not None and resident.job is worker:
            self._drop(resident)

    def _drop(self, resident: _Resident) -> None:
        if resident in self._residents:
            self._residents.remove(resident)
        resident.job = None
        if resident.process is not None:
            _release(resident.process, None, started=not resident.starting, wait=0.0)

    def _poll(self) -> None:
        for resident in list(self._residents):
            if resident.starting:
                continue
            while True:
                try:
                    job_id, message = resident.out_queue.get_nowait()
                except (Empty, OSError, ValueError, EOFError):
                    break
                worker = resident.job
                if worker is None or worker.job_id != job_id or message.get("type") == "__idle__":
                    continue
                worker._deliver(message)
                if message.get("type") == "done":
                    # free for the next job at once: the process takes it from its queue once it has tidied up
                    # (waited for "__idle__", the next track of a batch started a new process each time)
                    resident.job = None
            process = resident.process
            if process is None or not process.is_alive():
                worker = resident.job
                exitcode = process.exitcode if process is not None else None
                self._drop(resident)
                if worker is not None and exitcode == 0 and not worker._heard and not worker._ended:
                    # it ended on its idle time as the job came: the job, never begun, goes to a new one
                    worker._resident = None
                    self.submit(worker)
                elif worker is not None:
                    worker._resident_died(exitcode)
        if not self._residents:
            self._timer.stop()

    def shutdown(self) -> None:
        """Every resident told to end and waited for here (the app closing: a release thread still ending one when
        Python ends would close its handle under multiprocessing's own exit join)."""
        self._timer.stop()
        residents, self._residents = list(self._residents), []
        for resident in residents:
            try:
                resident.cmd_queue.put(None)
            except (OSError, ValueError):
                pass
        for resident in residents:
            if resident.process is not None:
                _end_process(resident.process, None, started=not resident.starting, wait=2.0)


_job_ids = iter(range(1, 1 << 62))


class ResidentJobWorker(QtCore.QObject):
    """SpawnProcessWorker's interface for a job a ``ResidentPool`` runs: start / stop, and the subclass's
    _handle_process_message / _process_exited called as there."""

    def __init__(self, *, pool: ResidentPool, target: Callable[..., None], args: tuple[Any, ...],
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pool, self._target, self._args = pool, target, args
        self.job_id = next(_job_ids)
        self._resident: _Resident | None = None
        self._ended = False
        self._heard = False               # a message of the job came (it began)

    @property
    def process_pid(self) -> int | None:
        resident = self._resident
        return resident.process.pid if resident is not None and resident.process is not None else None

    def start(self) -> None:
        if self._resident is not None or self._ended:
            raise RuntimeError("Worker already started.")
        self._pool.submit(self)

    def stop(self) -> None:
        if not self._ended:
            self._ended = True
            self._pool.cancel(self)

    def _deliver(self, message: dict[str, Any]) -> None:
        if self._ended:
            return
        self._heard = True
        self._handle_process_message(message)
        if message.get("type") == "done":
            self._ended = True
            self._process_exited(graceful=True, exitcode=0)

    def _resident_died(self, exitcode: int | None) -> None:
        if not self._ended:
            self._ended = True
            self._process_exited(graceful=False, exitcode=exitcode)

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        raise NotImplementedError

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        pass
