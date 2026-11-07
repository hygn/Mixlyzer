"""A spawned worker's lifecycle (system.process_worker): a process that said it is done but is slow to end (a full CPU
in a batch analysis) is not closed while it runs, nor waited for on the GUI thread; stop() ends a running one."""
import multiprocessing as mp
import os
import threading
import time
import unittest
from multiprocessing.context import SpawnProcess
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore  # noqa: E402

from system import process_worker  # noqa: E402

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


def _slow_to_end(linger: float, queue) -> None:
    """Says it is done, then lingers before it ends (module level: spawned)."""
    queue.put({"type": "done"})
    time.sleep(linger)


def _runs(_seconds: float, queue) -> None:
    time.sleep(_seconds)


class _Worker(process_worker.SpawnProcessWorker):
    def __init__(self, target, *args):
        super().__init__(target=target, args=args, daemon=True, process_name="test-worker")
        self.exited = []

    def _handle_process_message(self, message) -> None:
        self.process = self._launch.process

    def _process_exited(self, *, graceful: bool, exitcode) -> None:
        self.exited.append(graceful)


_Start = process_worker._Start


def _closed(process) -> bool:
    try:
        process.is_alive()
    except ValueError:      # "process object is closed"
        return True
    return False


def _until(done, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not done() and time.monotonic() < deadline:
        APP.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.01)


class ProcessWorkerTests(unittest.TestCase):
    def test_done_while_still_running_not_closed_nor_waited_for(self):
        worker = _Worker(_slow_to_end, 3.0)
        worker.start()
        _until(lambda: worker.exited, 60.0)
        self.assertEqual(worker.exited, [True])     # no ValueError: not closed while it ran
        self.assertTrue(worker.process.is_alive())  # finished here at once, the process still lingering
        _until(lambda: _closed(worker.process), 15.0)
        self.assertTrue(_closed(worker.process))    # released once it ended

    def test_stop_ends_a_running_process(self):
        worker = _Worker(_runs, 30.0)
        worker.start()
        _until(lambda: worker._launch.state is _Start.UP, 60.0)
        process = worker._launch.process
        worker.stop()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                alive = process.is_alive()
            except ValueError:        # closed: it has ended
                alive = False
            if not alive:
                break
            time.sleep(0.05)
        self.assertFalse(alive)

    def test_started_again_after_a_run_waits_for_its_own_process(self):
        """A run's start is its own: a worker started again once its run is over (or stopped) waits for the new
        process, never read as gone before it has started (the last run's started flag)."""
        worker = _Worker(_slow_to_end, 0.0)
        worker.start()
        _until(lambda: worker.exited, 60.0)
        real = SpawnProcess._Popen

        def slow(process_obj):      # a start slower than the first poll (a loaded machine)
            time.sleep(0.5)
            return real(process_obj)

        with mock.patch.object(SpawnProcess, "_Popen", staticmethod(slow)):
            worker.start()
            _until(lambda: len(worker.exited) == 2, 60.0)
        self.assertEqual(worker.exited, [True, True])


def _job(tag: str, seconds: float, queue) -> None:
    """A resident job: says which process ran it, then done (module level: spawned)."""
    time.sleep(seconds)
    queue.put({"type": "result", "tag": tag, "pid": os.getpid()})
    queue.put({"type": "done"})


class _Job(process_worker.ResidentJobWorker):
    def __init__(self, pool, tag: str, seconds: float = 0.0, *, stop_on_result: bool = False):
        super().__init__(pool=pool, target=_job, args=(tag, seconds))
        self.results, self.exited = [], []
        self._stop_on_result = stop_on_result

    def _handle_process_message(self, message) -> None:
        if message.get("type") == "result":
            self.results.append(message)
            if self._stop_on_result:      # as the workflows do: stopped from the result's handler, before "done"
                self.stop()

    def _process_exited(self, *, graceful: bool, exitcode) -> None:
        self.exited.append(graceful)


class ResidentPoolTests(unittest.TestCase):
    """system.process_worker.ResidentPool: one process runs job after job (no start for the next), a stopped job
    ends its process and the next goes to a new one, an idle process ends by itself."""

    def test_jobs_one_after_another_in_one_process(self):
        pool = process_worker.ResidentPool("test-resident", idle_sec=60.0, max_jobs=10)
        self.addCleanup(pool.shutdown)
        first = _Job(pool, "a")
        first.start()
        _until(lambda: first.exited, 60.0)
        second = _Job(pool, "b")
        second.start()
        _until(lambda: second.exited, 60.0)
        self.assertEqual((first.exited, second.exited), ([True], [True]))
        self.assertEqual(first.results[0]["pid"], second.results[0]["pid"])

    def test_a_stopped_job_ends_its_process_and_the_next_gets_another(self):
        pool = process_worker.ResidentPool("test-resident", idle_sec=60.0, max_jobs=10)
        self.addCleanup(pool.shutdown)
        warm = _Job(pool, "warm")
        warm.start()
        _until(lambda: warm.exited, 60.0)
        long = _Job(pool, "long", 30.0)
        long.start()
        _until(lambda: long._resident is not None and long._resident.launch.state is not _Start.STARTING, 10.0)
        process = long._resident.launch.process
        long.stop()
        _until(lambda: _closed(process) or not process.is_alive(), 15.0)
        self.assertTrue(_closed(process) or not process.is_alive())
        after = _Job(pool, "after")
        after.start()
        _until(lambda: after.exited, 60.0)
        self.assertEqual(after.exited, [True])
        self.assertNotEqual(after.results[0]["pid"], warm.results[0]["pid"])
        self.assertEqual(long.exited, [])       # a stopped job reports nothing

    def test_a_job_stopped_after_its_result_leaves_its_process_to_the_next(self):
        pool = process_worker.ResidentPool("test-resident", idle_sec=60.0, max_jobs=10)
        self.addCleanup(pool.shutdown)
        first = _Job(pool, "a", stop_on_result=True)
        first.start()
        _until(lambda: first.results, 60.0)
        resident = first._resident
        _until(lambda: resident.job is None, 10.0)       # its "done" frees it
        self.assertIn(resident, pool._residents)
        second = _Job(pool, "b")
        second.start()
        _until(lambda: second.exited, 60.0)
        self.assertEqual(second.exited, [True])
        self.assertEqual(first.results[0]["pid"], second.results[0]["pid"])

    def test_a_job_stopped_while_its_process_starts_leaves_no_process(self):
        pool = process_worker.ResidentPool("test-resident", idle_sec=60.0, max_jobs=10)
        self.addCleanup(pool.shutdown)
        job = _Job(pool, "a", 5.0)
        # stopped while Process.start() makes the child (a few ms; held here until the stop is done): the Process
        # object past its closed check, its child not yet made
        entered, gate = threading.Event(), threading.Event()
        real = SpawnProcess._Popen

        def held(process_obj):
            entered.set()
            gate.wait(30.0)
            return real(process_obj)

        with mock.patch.object(SpawnProcess, "_Popen", staticmethod(held)):
            job.start()
            resident = job._resident
            name = resident.launch.process.name
            self.assertTrue(entered.wait(30.0))
            job.stop()
            self.assertEqual(pool._residents, [])
            time.sleep(0.3)                     # a release thread's close() done meanwhile
            gate.set()
            _until(lambda: resident.launch.state is _Start.ENDED, 60.0)
        # the children multiprocessing still knows running (a Process object closed while it started reads as
        # closed, its child running all the same)
        running = lambda: [p.name for p in mp.active_children() if p.name == name]   # noqa: E731
        _until(lambda: not running(), 15.0)
        self.assertEqual(running(), [])         # never left running the canceled job
        self.assertEqual(job.exited, [])

    def test_an_idle_process_ends_by_itself(self):
        pool = process_worker.ResidentPool("test-resident", idle_sec=1.0, max_jobs=10)
        self.addCleanup(pool.shutdown)
        job = _Job(pool, "a")
        job.start()
        _until(lambda: job.exited, 60.0)
        _until(lambda: not pool._residents, 20.0)
        self.assertEqual(pool._residents, [])


if __name__ == "__main__":
    unittest.main()
