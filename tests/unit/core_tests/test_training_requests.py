"""core.workers.training_models and core.workers.optimizer: a Reoptimize run's request given the files it reads (the
model in use, the one it starts from, the user's retrained file where there is one, else the shipped one), a shipped
model never taken as an output, the dialogs' warnings by song count and start, and the optimizer process's entry
(a refusal sent as its words, an error as its traceback, progress worded) and the worker's handling of its messages.
Every model path is in a temporary folder; the shipped models are only looked up, never written."""
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analyzer.training import model_start as MS
from analyzer.training.text import TrainingError, msg
from core.workers import optimizer as worker
from core.workers import training_models as TM


class OutputTests(unittest.TestCase):
    def test_a_shipped_model_is_never_an_output(self):
        for name in ("beat_tcn", "beat_critic", "key_model", "beat_decoder", "jumpcue", "phrase", "refine_onset",
                     "downbeat_ctx"):
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                TM.prepare(name, {"output_path": "assets/weights/default_beat_tcn.npz"})
            self.assertIn("default_beat_tcn.npz", str(caught.exception))
        TM.check_output("assets/weights/beat_tcn.npz")

    def test_a_shipped_model_in_other_case_is_never_an_output(self):
        with self.assertRaises(ValueError):
            TM.check_output("assets/weights/DEFAULT_beat_tcn.npz")


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.out = self.dir / "beat_tcn.npz"
        self.shipped = self.dir / "default_beat_tcn.npz"

    def tearDown(self):
        self._tmp.cleanup()

    def prepare(self, name="beat_tcn", **values):
        return TM.prepare(name, {"output_path": str(self.out), **values})

    def test_without_a_retrained_model_everything_starts_from_the_shipped_one(self):
        for start in (MS.CURRENT, MS.SHIPPED):
            with self.subTest(start=start):
                values = self.prepare(start=start)
                self.assertEqual((values["current_path"], values["start_path"]), (str(self.shipped), str(self.shipped)))
        values = self.prepare(start=MS.SCRATCH)
        self.assertIsNone(values["start_path"])
        self.assertEqual(values["current_path"], str(self.shipped))

    def test_a_retrained_model_is_the_one_in_use(self):
        self.out.write_bytes(b"retrained")
        current = self.prepare(start=MS.CURRENT)
        self.assertEqual((current["current_path"], current["start_path"]), (str(self.out), str(self.out)))
        shipped = self.prepare(start=MS.SHIPPED)
        self.assertEqual((shipped["current_path"], shipped["start_path"]), (str(self.out), str(self.shipped)))
        self.assertEqual(self.prepare()["start_path"], str(self.out))              # no start: the current one

    def test_base_path_overrides_the_start(self):
        base = self.dir / "other" / "beat_tcn.npz"
        values = self.prepare(start=MS.SCRATCH, base_path=str(base))
        self.assertEqual(values["start_path"], str(base.with_name("default_beat_tcn.npz")))
        base.parent.mkdir()
        base.write_bytes(b"x")
        self.assertEqual(self.prepare(base_path=str(base))["start_path"], str(base))

    def test_the_request_given_is_not_changed(self):
        given = {"output_path": str(self.out), "start": MS.SHIPPED, "current_models": {"tcn": None}}
        before = {k: (dict(v) if isinstance(v, dict) else v) for k, v in given.items()}
        TM.prepare("beat_tcn", given)
        self.assertEqual(given, before)

    def test_decoder_costs_refitted_around_the_shipped_or_the_current(self):
        out = self.dir / "beat_decoder.json"
        out.write_text("{}")
        shipped = TM.prepare("beat_decoder", {"output_path": str(out), "start": MS.SHIPPED})
        self.assertEqual(shipped["start_path"], str(out.with_name("default_beat_decoder.json")))
        for start in (MS.CURRENT, MS.SCRATCH):                     # never from scratch: the current costs
            self.assertEqual(TM.prepare("beat_decoder", {"output_path": str(out), "start": start})["start_path"],
                             str(out))
        tcn = self.dir / "my_tcn.npz"
        tcn.write_bytes(b"x")
        values = TM.prepare("beat_decoder", {"output_path": str(out), "tcn_path": str(tcn)})
        self.assertEqual(values["tcn_path"], str(tcn))
        for key, name in (("critic_path", "beat_critic.npz"), ("downbeat_ctx_path", "downbeat_ctx_hgb.npz"),
                          ("refine_onset_path", "refine_onset_hgb.npz")):
            self.assertTrue(Path(values[key]).name in (name, "default_" + name), values[key])
            self.assertTrue(Path(values[key]).is_absolute())

    def test_every_current_model_named(self):
        tcn = self.dir / "beat_tcn_user.npz"
        tcn.write_bytes(b"x")
        values = self.prepare(current_models={"tcn": str(tcn)})
        self.assertEqual(set(values["current_models"]), {"downbeat_ctx", "tcn", "critic", "decoder", "refine_onset"})
        self.assertEqual(values["current_models"]["tcn"], str(tcn))
        self.assertNotIn("current_models", self.prepare())

    def test_critic_and_jumpcue_files(self):
        critic = TM.prepare("beat_critic", {"output_path": str(self.dir / "beat_critic.npz")})
        self.assertEqual(critic["current_path"], str(self.dir / "default_beat_critic.npz"))
        self.assertTrue(Path(critic["tcn_path"]).name.endswith("beat_tcn.npz"))
        jump = TM.prepare("jumpcue", {"output_path": str(self.dir / "jumpcue_jump_point.json")})
        self.assertEqual(jump["current_path"], str(self.dir / "default_jumpcue_jump_point.json"))
        self.assertTrue(Path(jump["phrase_model_file"]).name.endswith("phrase_analyzer.npz"))
        self.assertTrue(Path(jump["jump_point_model_path"]).name.endswith("jumpcue_jump_point.json"))
        self.assertNotIn("start_path", jump)

    def test_gradient_boosted_models_are_given_no_start(self):
        for name in ("refine_onset", "downbeat_ctx", "phrase"):
            values = TM.prepare(name, {"output_path": str(self.dir / f"{name}.npz")})
            self.assertNotIn("current_path", values)
            self.assertNotIn("start_path", values)

    def test_base_model(self):
        self.assertIsNone(TM.base_model(self.out, MS.SCRATCH))
        self.assertEqual(TM.base_model(self.out, MS.SHIPPED), self.shipped)
        self.assertEqual(TM.base_model(self.out, MS.CURRENT), self.shipped)
        self.out.write_bytes(b"x")
        self.assertEqual(TM.base_model(self.out, MS.CURRENT), self.out)
        self.assertEqual(TM.base_model(self.shipped, MS.SHIPPED), self.shipped)


class WarningTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "beat_tcn.npz"

    def tearDown(self):
        self._tmp.cleanup()

    def test_blocked_below_the_minimum(self):
        for kind, (blocked, _) in MS.MIN_TRACKS.items():
            with self.subTest(kind=kind):
                messages, is_blocked = TM.warnings(blocked - 1, kind, MS.CURRENT, self.out)
                self.assertTrue(is_blocked)
                self.assertEqual(len(messages), 1)
                self.assertIn(f"at least {blocked} songs with beat grids; the library has {blocked - 1}", messages[0])
        messages, _ = TM.warnings(0, "fine_tune", MS.CURRENT, self.out, songs=MS.KEY_SONGS)
        self.assertIn("songs with checked key annotations", messages[0])
        self.assertTrue(TM.warnings(0, "scratch", MS.SCRATCH, self.out)[0][0].startswith("Training from scratch"))

    def test_few_songs_warned(self):
        blocked, warned = MS.MIN_TRACKS["fine_tune"]
        messages, is_blocked = TM.warnings(blocked, "fine_tune", MS.SHIPPED, self.out)
        self.assertFalse(is_blocked)
        self.assertEqual(len(messages), 1)
        self.assertIn("overfitting", messages[0])
        self.assertEqual(TM.warnings(warned, "fine_tune", MS.SHIPPED, self.out), ([], False))
        blocked, warned = MS.MIN_TRACKS["scratch"]
        messages, _ = TM.warnings(blocked, "scratch", MS.SCRATCH, self.out)
        self.assertIn(f"about {MS.SHIPPED_TRAINING_TRACKS}", messages[0])
        self.assertIn("one to two hours", messages[1])
        messages, _ = TM.warnings(warned, "scratch", MS.SCRATCH, self.out, scratch_takes_hours=False)
        self.assertEqual(messages, ["Training from scratch discards what the shipped model learned."])

    def test_retrained_model_warnings(self):
        enough = MS.MIN_TRACKS["fine_tune"][1]
        self.assertEqual(TM.warnings(enough, "fine_tune", MS.CURRENT, self.out), ([], False))
        self.out.write_bytes(b"x")
        current, _ = TM.warnings(enough, "fine_tune", MS.CURRENT, self.out)
        self.assertIn("already retrained", current[0])
        shipped, _ = TM.warnings(enough, "fine_tune", MS.SHIPPED, self.out)
        self.assertIn("replaces the earlier retrained model", shipped[0])
        self.assertEqual(TM.warnings(enough, "fine_tune", MS.CURRENT, None), ([], False))

    def test_words(self):
        self.assertEqual(TM.words(msg("Key features {done}/{total}", done=1, total=2)), "Key features 1/2")


class FakeRequest(dict):
    def __init__(self, **values):
        super().__init__(values)


class ProcessEntryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "beat_tcn.npz"
        self.enterContext(mock.patch.object(worker, "install_parent_watchdog"))

    def tearDown(self):
        from analyzer.training import library_tracks
        library_tracks._tracks = None
        self._tmp.cleanup()

    def run_entry(self, optimize, name="beat_tcn", values=None):
        q = queue.Queue()
        definition = worker._OptimizerDefinition(FakeRequest, optimize)
        job = {"optimizer_name": name, "display_name": "Beat model",
               "request_values": values if values is not None else {"output_path": str(self.out)}}
        with mock.patch.dict(worker._OPTIMIZER_LOADERS, {"beat_tcn": lambda: definition}):
            worker._optimizer_process_entry("beat", (job,), ({"uid": "u"},), q)
        return [q.get_nowait() for _ in range(q.qsize())]

    def test_request_prepared_and_progress_worded(self):
        seen = []

        def optimize(request, *, feature_progress, optimize_progress):
            seen.append(request)
            feature_progress(10, (msg("Key features {done}/{total}", done=1, total=4), "second line"))
            optimize_progress(50.7, msg("Done"))
            return "result"
        messages = self.run_entry(optimize)
        self.assertEqual([m["type"] for m in messages],
                         ["stage_started", "feature_progress", "optimize_progress", "job_result", "done"])
        self.assertEqual(messages[0]["display_name"], "Beat model")
        self.assertEqual(messages[1]["text"], "Key features 1/4\nsecond line")
        self.assertEqual((messages[2]["value"], messages[2]["text"]), (50, "Done"))
        self.assertEqual(messages[3]["payload"], "result")
        self.assertEqual(seen[0]["current_path"], str(self.out.with_name("default_beat_tcn.npz")))

    def test_a_refusal_is_sent_as_its_words(self):
        def optimize(request, **_):
            raise TrainingError(msg("Retraining needs at least {needed} {songs} (the library has {count})",
                                    needed=20, songs=MS.BEAT_SONGS, count=3))
        messages = self.run_entry(optimize)
        self.assertEqual([m["type"] for m in messages], ["stage_started", "error", "done"])
        self.assertEqual(messages[1]["traceback"],
                         "Retraining needs at least 20 songs with beat grids (the library has 3)")

    def test_an_error_is_sent_with_its_traceback(self):
        def optimize(request, **_):
            raise ZeroDivisionError("boom")
        messages = self.run_entry(optimize)
        self.assertIn("Traceback", messages[1]["traceback"])
        self.assertIn("ZeroDivisionError: boom", messages[1]["traceback"])
        self.assertEqual(messages[-1]["type"], "done")

    def test_a_shipped_output_never_reaches_the_optimizer(self):
        optimize = mock.Mock()
        messages = self.run_entry(optimize, values={"output_path": str(self.out.with_name("default_beat_tcn.npz"))})
        optimize.assert_not_called()
        self.assertEqual([m["type"] for m in messages], ["stage_started", "error", "done"])
        self.assertIn("never overwritten", messages[1]["traceback"])

    def test_unknown_optimizer(self):
        messages = self.run_entry(mock.Mock(), name="nonexistent")
        self.assertIn("Unsupported parameter optimizer: nonexistent", messages[1]["traceback"])

    def test_every_optimizer_has_a_loader(self):
        for name, loader in worker._OPTIMIZER_LOADERS.items():
            with self.subTest(name=name):
                definition = loader()
                self.assertTrue(callable(definition.optimize))
                self.assertTrue(definition.request_type.__name__.endswith("OptimizationRequest"))


class WorkerMessageTests(unittest.TestCase):
    def worker(self, jobs=1):
        w = worker.OptimizerWorker("beat", [{"optimizer_name": "beat_tcn"}] * jobs, [])
        self.seen = {"finished": [], "failed": [], "feature": [], "stage": []}
        w.finished.connect(self.seen["finished"].append)
        w.failed.connect(self.seen["failed"].append)
        w.featureProgress.connect(lambda v, t: self.seen["feature"].append((v, t)))
        w.stageStarted.connect(lambda *a: self.seen["stage"].append(a))
        return w

    def test_one_result_is_the_payload(self):
        w = self.worker()
        w._handle_process_message({"type": "stage_started", "index": 1, "total": 1, "optimizer_name": "beat_tcn",
                                   "display_name": "Beat"})
        w._handle_process_message({"type": "feature_progress", "value": 5, "text": "x"})
        w._handle_process_message({"type": "job_result", "payload": "r"})
        w._process_exited(graceful=True, exitcode=0)
        self.assertEqual(self.seen, {"finished": ["r"], "failed": [], "feature": [(5, "x")],
                                     "stage": [(1, 1, "beat_tcn", "Beat")]})

    def test_several_results_are_a_tuple(self):
        w = self.worker(jobs=2)
        for p in ("a", "b"):
            w._handle_process_message({"type": "job_result", "payload": p})
        w._process_exited(graceful=True, exitcode=0)
        self.assertEqual(self.seen["finished"], [("a", "b")])

    def test_missing_results_or_a_crash_fail_once(self):
        w = self.worker(jobs=2)
        w._handle_process_message({"type": "job_result", "payload": "a"})
        w._process_exited(graceful=True, exitcode=0)
        self.assertEqual(len(self.seen["failed"]), 1)
        self.assertIn("1 of 2", self.seen["failed"][0])
        w = self.worker()
        w._process_exited(graceful=False, exitcode=3)
        w._process_exited(graceful=False, exitcode=3)
        self.assertEqual(len(self.seen["failed"]), 1)
        self.assertIn("(3)", self.seen["failed"][0])

    def test_an_error_message_fails_once_and_nothing_finishes(self):
        w = self.worker()
        w._handle_process_message({"type": "error", "traceback": "refused"})
        w._handle_process_message({"type": "error", "traceback": "again"})
        w._handle_process_message({"type": "job_result", "payload": "r"})
        w._process_exited(graceful=True, exitcode=0)
        self.assertEqual((self.seen["failed"], self.seen["finished"]), (["refused"], []))


if __name__ == "__main__":
    unittest.main()
