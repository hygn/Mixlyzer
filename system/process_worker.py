"""Shared Qt lifecycle for spawned worker processes."""

from __future__ import annotations

import enum
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


class _Start(enum.Enum):
    """A process's life as its ``_Launch`` holds it."""
    STARTING = enum.auto()      # Process.start() running on the launch's thread
    UP = enum.auto()
    FAILED = enum.auto()        # Process.start() raised
    LET_GO = enum.auto()        # released while it started: ended by the launch's thread once it is up
    ENDED = enum.auto()         # released: ended


class _Launch:
    """One process started on a thread of its own, and its hand-over with the one who started it: on Windows
    Process.start() returns only once the child has read all it is sent, and the child reads its arguments after
    re-importing the app's main module (seconds of a frozen UI on the GUI thread). Released while it starts, it is
    ended by that thread once it is up (closed at once, it ran anyway)."""

    def __init__(self, process: mp.Process, queue: Any) -> None:
        self.process, self.queue = process, queue     # ``queue`` closed with the process (None: none)
        self._lock = threading.Lock()
        self._state = _Start.STARTING
        threading.Thread(target=self._start, name=f"{process.name}-start", daemon=True).start()

    @property
    def state(self) -> _Start:
        with self._lock:
            return self._state

    def _start(self) -> None:
        try:
            self.process.start()
            state = _Start.UP
        except Exception:
            state = _Start.FAILED
        with self._lock:
            let_go = self._state is _Start.LET_GO
            self._state = _Start.ENDED if let_go else state
        if let_go:
            _end_process(self.process, self.queue, started=state is _Start.UP, wait=0.0)

    def release(self, *, wait: float, here: bool = False) -> None:
        """The process ended, once: still starting, by its thread once it is up; else given ``wait`` seconds to end
        by itself, then terminated (on a thread of its own, or ``here``)."""
        with self._lock:
            state = self._state
            if state in (_Start.LET_GO, _Start.ENDED):
                return
            self._state = _Start.LET_GO if state is _Start.STARTING else _Start.ENDED
        if state is _Start.STARTING:
            return
        if here:
            _end_process(self.process, self.queue, started=state is _Start.UP, wait=wait)
        else:
            _release(self.process, self.queue, started=state is _Start.UP, wait=wait)


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
        self._launch: _Launch | None = None     # the run's process (None: none running); a new one each start
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll_queue)

    @property
    def process_pid(self) -> int | None:
        return self._launch.process.pid if self._launch is not None else None

    def start(self) -> None:
        if self._launch is not None:
            raise RuntimeError("Worker already started.")
        queue = self._ctx.Queue()
        process = self._ctx.Process(target=self._target, args=(*self._args, queue), name=self._process_name)
        process.daemon = self._daemon
        self._launch = _Launch(process, queue)      # the queue polled once it has started
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        launch, self._launch = self._launch, None
        if launch is not None:
            launch.release(wait=0.0)

    def _poll_queue(self) -> None:
        launch = self._launch
        if launch is None:
            return
        state = launch.state
        if state is _Start.STARTING:
            return
        if state is _Start.FAILED:
            self._finalize_process(graceful=False)
            return
        while self._launch is launch:      # a handler may stop the worker
            try:
                message = launch.queue.get_nowait()
            except Empty:
                break
            self._handle_process_message(message)
            if message.get("type") == "done":
                self._finalize_process(graceful=True)
                return
        if self._launch is launch and not launch.process.is_alive():
            self._finalize_process(graceful=False)

    def _finalize_process(self, *, graceful: bool) -> None:
        """The process done (``graceful``: it said so) or gone. One that said it is done is left to end by itself
        on a thread of its own (terminated if it has not in GRACEFUL_END_SEC): waited for here, a process still
        ending when the CPU is full (a batch analysis) held the GUI and was closed while it ran (a ValueError)."""
        self._timer.stop()
        launch, self._launch = self._launch, None
        exitcode = None
        if launch is not None:
            if launch.state is _Start.UP and not graceful:
                launch.process.join(timeout=1.0)       # gone already: at once
                exitcode = launch.process.exitcode
            launch.release(wait=GRACEFUL_END_SEC if graceful else 0.0)
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
# its process (a fresh one takes the next job); stopped once it has answered (a "result" or an "error": a target's
# last message before its "done"), it is only let go: its process ends it and takes the next.

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
    def __init__(self, launch: _Launch, cmd_queue: Any, out_queue: Any) -> None:
        self.launch, self.cmd_queue, self.out_queue = launch, cmd_queue, out_queue
        self.job: ResidentJobWorker | None = None


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
        resident = _Resident(_Launch(process, None), cmd_queue, out_queue)
        self._residents.append(resident)
        self._timer.start()
        return resident

    def prewarm(self) -> None:
        """One resident started ahead of its first job (none when there is one)."""
        if not self._residents:
            self._spawn()

    def _usable(self, resident: _Resident) -> bool:
        """Idle and starting or alive."""
        state = resident.launch.state
        return resident.job is None and (
            state is _Start.STARTING or (state is _Start.UP and resident.launch.process.is_alive()))

    def submit(self, worker: ResidentJobWorker) -> None:
        resident = next((r for r in self._residents if self._usable(r)), None) or self._spawn()
        resident.job = worker
        worker._resident = resident
        resident.cmd_queue.put((worker.job_id, worker._target, worker._args))

    def cancel(self, worker: ResidentJobWorker) -> None:
        """A job stopped: its process ended (a job cannot be interrupted), the next job takes another."""
        resident = worker._resident
        if resident is not None and resident.job is worker:
            self._drop(resident)

    def _drop(self, resident: _Resident) -> None:
        """The resident out of the pool and its process ended (once: a second drop does nothing)."""
        if resident not in self._residents:
            return
        self._residents.remove(resident)
        resident.job = None
        resident.launch.release(wait=0.0)

    def _poll(self) -> None:
        for resident in list(self._residents):
            state = resident.launch.state
            if state is _Start.STARTING:
                continue
            while resident in self._residents:     # a handler may stop its job: the resident dropped, its process
                try:                               # closed by the release thread
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
            if resident not in self._residents:
                continue
            process = resident.launch.process
            if state is not _Start.UP or not process.is_alive():      # gone, or its start failed
                worker = resident.job
                exitcode = process.exitcode if state is _Start.UP else None
                self._drop(resident)
                if worker is not None and exitcode == 0 and worker._state is _JobState.WAITING:
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
        for resident in residents:      # one still starting is ended by its launch once up (it reads the None too)
            resident.launch.release(wait=2.0, here=True)


_job_ids = iter(range(1, 1 << 62))


class _JobState(enum.Enum):
    """A ``ResidentJobWorker``'s job."""
    NEW = enum.auto()
    WAITING = enum.auto()       # given to the pool, nothing heard of it yet
    RUNNING = enum.auto()       # a message of it came: it began
    ANSWERED = enum.auto()      # its "result" or "error" came: only its "done" is left
    ENDED = enum.auto()         # done, its process gone, or stopped: nothing more heard


class ResidentJobWorker(QtCore.QObject):
    """SpawnProcessWorker's interface for a job a ``ResidentPool`` runs: start / stop, and the subclass's
    _handle_process_message / _process_exited called as there."""

    def __init__(self, *, pool: ResidentPool, target: Callable[..., None], args: tuple[Any, ...],
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pool, self._target, self._args = pool, target, args
        self.job_id = next(_job_ids)
        self._resident: _Resident | None = None
        self._state = _JobState.NEW

    @property
    def process_pid(self) -> int | None:
        resident = self._resident
        return resident.launch.process.pid if resident is not None else None

    def start(self) -> None:
        if self._state is not _JobState.NEW:
            raise RuntimeError("Worker already started.")
        self._state = _JobState.WAITING
        self._pool.submit(self)

    def stop(self) -> None:
        """No more of the job heard. Before its answer the job is canceled (its process ended: a job cannot be
        interrupted); after it (stopped from the answer's handler, before the "done"), its process stays the pool's,
        busy until that "done", then taking the next job."""
        if self._state is _JobState.ENDED:
            return
        answered, self._state = self._state is _JobState.ANSWERED, _JobState.ENDED
        if not answered:
            self._pool.cancel(self)

    def _deliver(self, message: dict[str, Any]) -> None:
        if self._state is _JobState.ENDED:
            return
        if message.get("type") in ("result", "error"):
            self._state = _JobState.ANSWERED
        elif self._state is _JobState.WAITING:
            self._state = _JobState.RUNNING
        self._handle_process_message(message)
        if message.get("type") == "done":
            self._state = _JobState.ENDED
            self._process_exited(graceful=True, exitcode=0)

    def _resident_died(self, exitcode: int | None) -> None:
        if self._state is not _JobState.ENDED:
            self._state = _JobState.ENDED
            self._process_exited(graceful=False, exitcode=exitcode)

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        raise NotImplementedError

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        pass
