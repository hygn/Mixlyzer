"""A spawned worker's lifecycle (system.process_worker): a process that said it is done but is slow to end (a full CPU
in a batch analysis) is not closed while it runs, nor waited for on the GUI thread; stop() ends a running one."""
import os
import time
import unittest

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
        self.process = self._process

    def _process_exited(self, *, graceful: bool, exitcode) -> None:
        self.exited.append(graceful)


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
        _until(lambda: worker._started, 60.0)
        process = worker._process
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


if __name__ == "__main__":
    unittest.main()
