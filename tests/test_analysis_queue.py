"""The full analyses' queue (core.workflows.analysis): at most analysisconfig.max_parallel_analyses at once, the others
waiting in their order, the deck's track ahead of them; a slot freed starts the next; a cancel drops the waiting."""
import os
import unittest
import zipfile
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

    def test_a_limit_raised_starts_the_waiting_at_once(self):
        self.limit = 1
        for name in ("a.wav", "b.wav", "c.wav", "d.wav"):
            self.ask(name)
        self.assertEqual(_Worker.started, ["a.wav"])
        self.limit = 3                      # the settings saved (core.settings.store's changed)
        self.flow.settings_changed(None, None)
        self.assertEqual(_Worker.started, ["a.wav", "b.wav", "c.wav"])
        self.limit = 1                      # lowered: the running ones go on
        self.flow.settings_changed(None, None)
        self.assertEqual(len(self.flow._workers), 3)

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

    def test_a_new_file_being_analyzed_loads_without_saying_it_is_already(self):
        said = []
        self.flow._status_callback = said.append
        self.ask("a.wav")                   # into the library
        self.ask("a.wav")                   # again: nothing new to do, said
        self.assertEqual(said[-1], "Analysis already in progress: a.wav")
        got = []
        self.flow.analyze("a.wav", listener=got.append)    # onto the deck: it takes the result when done
        self.assertEqual(said[-1], "Analyzing: a.wav")
        self.assertEqual(_Worker.started, ["a.wav"])
        self.flow._deliver(next(iter(self.flow._contexts)), {"properties": {"path": "a.wav"}})
        self.assertEqual(len(got), 1)


class AnalysisIdTests(unittest.TestCase):
    """Every analysis known by its id: asked of by it (waiting, running, saving), told when it ends (its result, or
    its error, a cancel's too); another program's (``request``) stored only when asked, or when joined by one that
    stores; a file the library holds not analyzed for it."""

    def setUp(self):
        _Worker.started = []
        self.tasks = _Tasks()
        self.writes, self.results, self.ended = [], [], []
        self.held = set()
        library = SimpleNamespace(
            snapshot=SimpleNamespace(find_file=lambda p: SimpleNamespace(uid="u", path=p) if p in self.held else None),
            write=lambda job, payload, done, failed: self.writes.append((payload, done, failed)))
        patcher = mock.patch.object(workflow, "AnalysisWorker", _Worker)
        patcher.start()
        self.addCleanup(patcher.stop)
        cfg = SimpleNamespace(analysisconfig=SimpleNamespace(max_parallel_analyses=1))
        self.flow = workflow.AnalysisWorkflow(
            library=library, taskmanager=self.tasks, result_callback=self.results.append,
            reanalysis_result_callback=lambda r: None, status_callback=lambda s: None, cfg_loader=lambda: cfg,
            albumart_provider=lambda p: None, tag_extractor=lambda p: (os.path.basename(p), "", "", ""))
        self.flow.ended.connect(lambda *args: self.ended.append(args))

    def taskid(self, analysis_id):
        return self.flow._ids[analysis_id]

    @staticmethod
    def payload(path):
        return {"properties": {"path": path, "title": "t"}, "update_db": True, "taskid": 0,
                "arrays": {"beats_time_sec": [0.5, 1.0]}, "store": {"index": None}}

    def test_ids_and_states(self):
        first = self.flow.analyze("a.wav")
        second = self.flow.request("b.wav", store=False)
        self.assertRegex(first, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first, second)
        self.assertEqual(self.flow.analyze("a.wav"), first)                  # asked again: the same analysis
        self.assertEqual(self.flow.state(first).state, workflow.RUNNING)
        waiting = self.flow.state(second)
        self.assertEqual((waiting.state, waiting.position), (workflow.WAITING, 1))
        self.flow._on_progress(self.taskid(first), "Beats", 0.5)
        self.assertEqual(self.flow.state(first).progress, 0.5)
        self.flow._on_success(self.taskid(first), self.payload("a.wav"))
        self.assertEqual(self.flow.state(first).state, workflow.SAVING)
        payload, done, _failed = self.writes[-1]
        done(dict(payload, analysis="stored"))
        self.assertIsNone(self.flow.state(first))
        self.assertEqual(self.ended[0][0], first)
        self.assertEqual((self.ended[0][1]["analysis"], self.ended[0][2]), ("stored", ""))
        self.assertIsNone(self.flow.state("f" * 32))

    def test_another_programs_analysis_not_stored(self):
        analysis_id = self.flow.request("a.wav", store=False)
        self.flow._on_success(self.taskid(analysis_id), self.payload("a.wav"))
        self.assertEqual(self.writes, [])                                   # the library never asked
        _id, result, error = self.ended[-1]
        self.assertEqual(error, "")
        self.assertNotIn("uid", result["properties"])
        self.assertEqual(list(result["analysis"].beats), [0.5, 1.0])         # as the library would read it
        self.assertFalse(result["update_db"])
        self.assertEqual(len(self.results), 1)                              # its task finished by the window

    def test_stored_when_joined_by_one_that_stores(self):
        analysis_id = self.flow.request("a.wav", store=False)
        self.assertEqual(self.flow.request("a.wav", store=True), analysis_id)
        self.flow._on_success(self.taskid(analysis_id), self.payload("a.wav"))
        self.assertEqual(len(self.writes), 1)
        other = self.flow.request("b.wav", store=False)
        deck = []
        self.assertEqual(self.flow.analyze("b.wav", listener=deck.append), other)   # loaded onto the deck
        self.assertTrue(self.flow._contexts[self.taskid(other)]["store"])

    def test_a_file_in_the_library_is_not_analyzed_for_it(self):
        self.held.add("a.wav")
        self.assertIsNone(self.flow.request("a.wav", store=True))
        self.assertEqual(_Worker.started, [])

    def test_errors_and_cancels_are_told(self):
        failing = self.flow.request("a.wav", store=False)
        waiting = self.flow.request("b.wav", store=False)
        with mock.patch("builtins.print"):
            self.flow._on_error(self.taskid(failing), "bad file")
        self.assertEqual(self.ended[-1], (failing, None, "bad file"))
        self.flow.cancel_all("stop")
        self.assertEqual(self.ended[-1], (waiting, None, "stop"))
        self.assertEqual(self.flow._ids, {})


class LoadedWhileReanalyzedTests(unittest.TestCase):
    """A library track being reanalyzed loads on the deck at once: its stored analysis now, the new one when done."""

    def setUp(self):
        _Worker.started = []
        self.tasks = _Tasks()
        self.stored = "the stored analysis"
        self.said, self.results, self.reanalyzed = [], [], []
        track = SimpleNamespace(uid="u", path="a.wav", to_meta=lambda: {"path": "a.wav"})
        self.read_error = None     # the read's exception (a damaged NPZ), else the stored analysis answered
        library = SimpleNamespace(snapshot=SimpleNamespace(find_file=lambda _p: track),
                                  read=lambda _job, _uid, done, failed=None:
                                  failed(self.read_error) if self.read_error is not None else done(self.stored))
        for patcher in (mock.patch.object(workflow, "AnalysisWorker", _Worker),
                        mock.patch.object(workflow.os.path, "isfile", return_value=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        cfg = SimpleNamespace(analysisconfig=SimpleNamespace(max_parallel_analyses=2))
        self.flow = workflow.AnalysisWorkflow(
            library=library, taskmanager=self.tasks, result_callback=self.results.append,
            reanalysis_result_callback=self.reanalyzed.append, status_callback=self.said.append,
            cfg_loader=lambda: cfg, albumart_provider=lambda p: None,
            tag_extractor=lambda p: (os.path.basename(p), "", "", ""))
        self.flow.reanalyze("a.wav")
        self.running = next(iter(self.flow._contexts))
        self.deck = []

    def load(self):
        self.flow.analyze("a.wav", listener=self.deck.append)

    def test_the_stored_analysis_at_once_then_the_new_one(self):
        self.load()
        self.assertEqual([r["analysis"] for r in self.deck], [self.stored])
        self.assertEqual(self.said[-1], "Loaded from the library: a.wav")
        self.assertNotIn("Analysis already in progress: a.wav", self.said)
        self.assertEqual(_Worker.started, ["a.wav"])                 # no second analysis
        self.flow._deliver(self.running, {"analysis": "new", "properties": {"path": "a.wav"}})
        self.assertEqual([r["analysis"] for r in self.reanalyzed], ["new"])     # to the deck by the window, if it holds it
        self.assertEqual(len(self.deck), 1)
        self.assertEqual(self.flow._contexts, {})

    def test_none_stored_to_read_the_reanalysis_gives_it(self):
        self.stored = None
        self.load()
        self.assertEqual(self.deck, [])
        self.assertEqual(_Worker.started, ["a.wav"])                 # joined, not a second analysis
        self.assertEqual(len(self.tasks.removed), 1)                 # its own task gone
        self.flow._deliver(self.running, {"analysis": "new", "properties": {"path": "a.wav"}})
        self.assertEqual([r["analysis"] for r in self.deck], ["new"])

    def test_a_stored_analysis_that_raises_is_taken_as_none(self):
        """The read failing with what the reader does not take for "none readable" (a damaged NPZ): as none read,
        the deck given the new analysis, its task not left at Loading Track."""
        self.read_error = zipfile.BadZipFile("File is not a zip file")
        with mock.patch("builtins.print"):
            self.load()
        self.assertEqual(len(self.tasks.removed), 1)                 # its own task gone (joined the reanalysis)
        self.flow._deliver(self.running, {"analysis": "new", "properties": {"path": "a.wav"}})
        self.assertEqual([r["analysis"] for r in self.deck], ["new"])

    def test_a_second_reanalysis_is_still_refused(self):
        self.flow.reanalyze("a.wav")
        self.assertEqual(self.said[-1], "Analysis already in progress: a.wav")
        self.assertEqual(_Worker.started, ["a.wav"])


if __name__ == "__main__":
    unittest.main()
