"""The full analyses' queue (core.workflows.analysis): at most analysisconfig.max_parallel_analyses at once, the others
waiting in their order, the deck's track ahead of them; a slot freed starts the next; a cancel drops the waiting."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from core.workflows import analysis as workflow

APP = QApplication.instance() or QApplication([])


class _Worker(QtCore.QObject):
    """An analysis process stand-in: started, stopped, and finished or failed by the test."""
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str, float)
    status = QtCore.Signal(str)
    started = []

    def __init__(self, path, cfg, taskid, parent=None):
        super().__init__(parent)
        self.path, self.taskid = path, taskid

    def start(self):
        _Worker.started.append(self.path)

    def stop(self):
        pass


class _Tasks:
    def __init__(self):
        self.next, self.status, self.removed = 0, {}, {}

    def addtask(self, **_kw):
        self.next += 1
        return SimpleNamespace(taskid=self.next)

    def updatetask(self, taskid, status, progress):
        self.status[taskid] = status

    def rmtask(self, taskid, reason=""):
        self.removed[taskid] = reason


class AnalysisQueueTests(unittest.TestCase):
    def setUp(self):
        _Worker.started = []
        self.limit = 2
        self.tasks = _Tasks()
        cfg = lambda: SimpleNamespace(analysisconfig=SimpleNamespace(max_parallel_analyses=self.limit))   # noqa: E731
        library = SimpleNamespace(snapshot=SimpleNamespace(find_file=lambda _p: None))
        patcher = mock.patch.object(workflow, "AnalysisWorker", _Worker)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.flow = workflow.AnalysisWorkflow(
            library=library, taskmanager=self.tasks, result_callback=lambda r: None,
            reanalysis_result_callback=lambda r: None, status_callback=lambda s: None, cfg_loader=cfg,
            albumart_provider=lambda p: None, tag_extractor=lambda p: (os.path.basename(p), "", "", ""))

    def ask(self, name, deck=False):
        self.flow.analyze(name, listener=(lambda r: None) if deck else None)

    def finish(self, path):
        taskid = next(t for t, c in self.flow._contexts.items() if c["path"] == path)
        self.flow._finalize(taskid)

    def test_at_most_the_limit_at_once_then_in_order(self):
        with mock.patch.object(workflow.os.path, "isfile", return_value=True):
            for name in ("a.wav", "b.wav", "c.wav", "d.wav"):
                self.ask(name)
        self.assertEqual(_Worker.started, ["a.wav", "b.wav"])
        self.assertEqual(sorted(self.tasks.status.values()), ["Waiting for a free analysis slot"] * 2)
        self.finish("a.wav")
        self.assertEqual(_Worker.started, ["a.wav", "b.wav", "c.wav"])
        self.limit = 3                      # the setting raised: the next start takes it
        self.finish("b.wav")
        self.assertEqual(_Worker.started, ["a.wav", "b.wav", "c.wav", "d.wav"])

    def test_the_decks_track_goes_first(self):
        self.limit = 1
        self.ask("a.wav")
        self.ask("b.wav")
        self.ask("deck.wav", deck=True)
        self.ask("c.wav")
        self.ask("c.wav", deck=True)        # a file waiting, then loaded onto the deck: ahead of the others
        self.finish("a.wav")
        self.assertEqual(_Worker.started, ["a.wav", "c.wav"])
        self.finish("c.wav")
        self.assertEqual(_Worker.started, ["a.wav", "c.wav", "deck.wav"])

    def test_cancel_drops_the_waiting(self):
        self.limit = 1
        for name in ("a.wav", "b.wav", "c.wav"):
            self.ask(name)
        self.flow.cancel_all("stop")
        self.assertEqual(len(self.tasks.removed), 3)        # the running one and both waiting
        self.assertEqual((self.flow._waiting, self.flow._contexts), (workflow.deque(), {}))
        self.ask("d.wav")
        self.assertEqual(_Worker.started, ["a.wav", "d.wav"])


if __name__ == "__main__":
    unittest.main()
