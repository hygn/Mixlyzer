"""training.__main__, the training's command line: a run's request given the files it reads (the model in use, the
one the decoder's costs are refitted around, the user's file where there is one, else the shipped one), a shipped
model never taken as an output, the request's fields from the settings and --set, a refusal said in its words, an
existing output copied aside before the run, the result reported. Every path is in a temporary folder; the
settings, the library and the optimizer are stand-ins."""
import contextlib
import dataclasses
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.settings.config import default_cfg
from training import __main__ as cli
from training import library_tracks
from training import model_start as MS
from training.text import TrainingError, msg


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_shipped_model_is_never_an_output(self):
        for name in cli.OPTIMIZERS:
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                cli.prepare(name, {"output_path": "assets/weights/default_beat_tcn.npz"})
            self.assertIn("default_beat_tcn.npz", str(caught.exception))
        with self.assertRaises(ValueError):
            cli.check_output("assets/weights/DEFAULT_beat_tcn.npz")     # in any letter case
        cli.check_output("assets/weights/beat_tcn.npz")

    def test_decoder_costs_refitted_around_the_shipped_or_the_current(self):
        out = self.dir / "beat_decoder.json"
        shipped = out.with_name("default_beat_decoder.json")
        self.assertEqual(cli.prepare("beat_decoder", {"output_path": str(out)})["start_path"], str(shipped))
        out.write_text("{}")
        self.assertEqual(cli.prepare("beat_decoder", {"output_path": str(out), "start": MS.SHIPPED})["start_path"],
                         str(shipped))
        values = cli.prepare("beat_decoder", {"output_path": str(out), "start": MS.CURRENT})
        self.assertEqual((values["start_path"], values["current_path"]), (str(out), str(out)))
        tcn = self.dir / "my_tcn.npz"
        tcn.write_bytes(b"x")
        values = cli.prepare("beat_decoder", {"output_path": str(out), "tcn_path": str(tcn)})
        self.assertEqual(values["tcn_path"], str(tcn))
        for key, name in (("critic_path", "beat_critic.npz"), ("downbeat_ctx_path", "downbeat_ctx_hgb.npz"),
                          ("refine_onset_path", "refine_onset_hgb.npz")):
            self.assertIn(Path(values[key]).name, (name, "default_" + name))
            self.assertTrue(Path(values[key]).is_absolute())

    def test_every_current_model_named(self):
        tcn = self.dir / "beat_tcn_user.npz"
        tcn.write_bytes(b"x")
        values = cli.prepare("downbeat_ctx", {"output_path": str(self.dir / "d.npz"), "current_models": {"tcn": str(tcn)}})
        self.assertEqual(set(values["current_models"]), {"downbeat_ctx", "tcn", "critic", "decoder", "refine_onset"})
        self.assertEqual(values["current_models"]["tcn"], str(tcn))

    def test_jumpcue_files(self):
        jump = cli.prepare("jumpcue", {"output_path": str(self.dir / "jumpcue_jump_point.json")})
        self.assertEqual(jump["current_path"], str(self.dir / "default_jumpcue_jump_point.json"))
        self.assertTrue(Path(jump["phrase_model_file"]).name.endswith("phrase_analyzer.npz"))
        self.assertTrue(Path(jump["jump_point_model_path"]).name.endswith("jumpcue_jump_point.json"))

    def test_networks_start_from_the_shipped_the_model_in_use_or_nothing(self):
        for name, file in (("key_model", "key_chord_cnn.npz"), ("beat_critic", "beat_critic.npz")):
            with self.subTest(name=name):
                in_use = self.dir / file
                out = self.dir / "elsewhere" / f"trial_{file}"                 # an output that is not the setting's
                shipped = in_use.with_name("default_" + file)
                values = cli.prepare(name, {"output_path": str(out), "current_path": str(in_use)})
                self.assertEqual((values["current_path"], values["start_path"]), (str(shipped), str(shipped)))
                in_use.write_bytes(b"x")                                        # the user's own model
                for start, expected in ((MS.CURRENT, str(in_use)), (MS.SHIPPED, str(shipped)), (MS.SCRATCH, None)):
                    values = cli.prepare(name, {"output_path": str(out), "current_path": str(in_use), "start": start})
                    self.assertEqual((values["current_path"], values["start_path"]), (str(in_use), expected))

    def test_decoder_costs_never_from_scratch(self):
        with self.assertRaises(ValueError):
            cli.prepare("beat_decoder", {"output_path": str(self.dir / "beat_decoder.json"), "start": MS.SCRATCH})

    def test_gradient_boosted_models_are_given_no_start(self):
        for name in ("refine_onset", "downbeat_ctx", "phrase"):
            values = cli.prepare(name, {"output_path": str(self.dir / f"{name}.npz")})
            self.assertNotIn("current_path", values)
            self.assertNotIn("start_path", values)

    def test_the_request_given_is_not_changed(self):
        given = {"output_path": str(self.dir / "d.npz"), "current_models": {"tcn": None}}
        before = {k: (dict(v) if isinstance(v, dict) else v) for k, v in given.items()}
        cli.prepare("downbeat_ctx", given)
        self.assertEqual(given, before)


def _args(**given):
    values = {"output": None, "cache": None, "start": None, "cv_folds": None, "set": None}
    values.update(given)
    return SimpleNamespace(**values)


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.analysis = default_cfg().analysisconfig

    def test_the_settings_files(self):
        values = cli.request_values("phrase", self.analysis, "library", _args())
        self.assertEqual(values["output_path"], cli.project_path(self.analysis.phrase_parameter_path))
        self.assertEqual(values["cache_dir"], cli.project_path("featurecache/phrase"))
        self.assertEqual(values["library_dir"], cli.project_path("library"))

    def test_the_beat_models_read_the_ones_in_use(self):
        onset = cli.request_values("refine_onset", self.analysis, "library", _args())
        self.assertEqual(onset["bpm_min"], float(self.analysis.bpm_min))
        self.assertEqual(onset["frames_cache_dir"], cli.project_path(cli.BEAT_CACHE))
        self.assertEqual(set(onset["current_models"]), set(cli._CURRENT_MODELS))
        decoder = cli.request_values("beat_decoder", self.analysis, "library", _args())
        self.assertNotIn("current_models", decoder)
        self.assertEqual(decoder["tcn_path"], cli.project_path(self.analysis.beat_tcn_parameter_path))

    def test_the_options_and_set(self):
        values = cli.request_values("beat_decoder", self.analysis, "library", _args(
            output="x.json", cache="c", start=MS.SHIPPED, cv_folds=3, set=["quick=true", "table_path=t.csv"]))
        self.assertEqual((values["output_path"], values["cache_dir"]), (cli.project_path("x.json"), cli.project_path("c")))
        self.assertEqual((values["start"], values["cv_folds"], values["quick"], values["table_path"]),
                         (MS.SHIPPED, 3, True, "t.csv"))

    def test_a_key_gt_file_from_the_apps_folder(self):
        values = cli.request_values("key_model", self.analysis, "library", _args(set=["key_gt=gt/keys.json"]))
        self.assertEqual(values["key_gt"], str(cli.project_path("gt/keys.json")))
        absolute = str(Path(tempfile.gettempdir()) / "keys.json")
        values = cli.request_values("key_model", self.analysis, "library", _args(set=[f"key_gt={absolute}"]))
        self.assertEqual(values["key_gt"], absolute)


@dataclasses.dataclass(frozen=True)
class FakeRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    seed: int = 0
    rebuild_cache: bool = False
    cv_folds: int = 0
    max_cache_workers: int | None = None


@dataclasses.dataclass(frozen=True)
class FakeResult:
    output_path: str
    boundary_f1: float
    skipped_tracks: tuple = ()


class MainTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.out = self.dir / "phrase_analyzer.npz"
        cfg = default_cfg()
        cfg.libconfig.libpath = str(self.dir / "library")
        self.enterContext(mock.patch("core.settings.config.load_cfg", return_value=cfg))
        self.enterContext(mock.patch.object(cli, "project_root", return_value=self.dir))
        self.tracks = [{"uid": "u", "title": "t", "artist": "a", "audio_path": Path("x"), "annotation": None}]
        self.enterContext(mock.patch.object(cli, "read_tracks", return_value=self.tracks))
        self.module = SimpleNamespace(PhraseOptimizationRequest=FakeRequest)
        self.enterContext(mock.patch.object(cli.importlib, "import_module", return_value=self.module))

    def tearDown(self):
        library_tracks._tracks = None
        self._tmp.cleanup()

    def run_main(self, optimize, *argv):
        self.module.optimize_phrase_parameters = optimize
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["phrase", "--output", str(self.out), *argv])
        return code, out.getvalue(), err.getvalue()

    def test_a_run_reported(self):
        seen = []

        def optimize(request, *, feature_progress, optimize_progress):
            seen.append((request, library_tracks.tracks()))
            feature_progress(10, (msg("Key features {done}/{total}", done=1, total=4), "second line"))
            feature_progress(10, (msg("Key features {done}/{total}", done=1, total=4), "second line"))
            optimize_progress(50, msg("Done"))
            return FakeResult(str(self.out), 0.91234)
        code, out, _err = self.run_main(optimize, "--set", "seed=7")
        self.assertEqual(code, 0)
        request, tracks = seen[0]
        self.assertEqual((request.output_path, request.seed), (self.out, 7))
        self.assertEqual([t["uid"] for t in tracks], ["u"])
        self.assertEqual(out.count("[features  10%] Key features 1/4 | second line"), 1)    # a line once
        self.assertIn("[training  50%] Done", out)
        self.assertIn("boundary_f1: 0.9123", out)

    def test_an_existing_output_copied_aside_first(self):
        self.out.write_bytes(b"old")
        code, out, _err = self.run_main(lambda request, **_: FakeResult(str(self.out), 1.0))
        self.assertEqual(code, 0)
        copies = list((self.dir / "backups").glob("phrase_analyzer.bak_*.npz"))
        self.assertEqual([c.read_bytes() for c in copies], [b"old"])
        self.assertIn("copied to", out)
        code, _out, _err = self.run_main(lambda request, **_: FakeResult(str(self.out), 1.0), "--no-backup")
        self.assertEqual(len(list((self.dir / "backups").glob("*"))), 1)

    def test_a_refusal_said_in_its_words(self):
        def optimize(request, **_):
            raise TrainingError(msg("Retraining needs at least {needed} {songs} (the library has {count})",
                                    needed=50, songs=MS.BEAT_SONGS, count=3))
        code, _out, err = self.run_main(optimize)
        self.assertEqual(code, 2)
        self.assertIn("Retraining needs at least 50 songs with beat grids (the library has 3)", err)

    def test_a_shipped_output_or_an_unknown_field_refused_before_the_run(self):
        optimize = mock.Mock()
        self.out = self.dir / "default_phrase_analyzer.npz"
        code, _out, err = self.run_main(optimize)
        self.assertEqual(code, 2)
        self.assertIn("never overwritten", err)
        self.out = self.dir / "phrase_analyzer.npz"
        code, _out, err = self.run_main(optimize, "--set", "steps=10")
        self.assertEqual(code, 2)
        self.assertIn("no field steps", err)
        optimize.assert_not_called()


class RegistryTests(unittest.TestCase):
    def test_every_optimizer_names_its_module(self):
        for name, spec in cli.OPTIMIZERS.items():
            with self.subTest(name=name):
                module = cli.importlib.import_module(f"training.{spec.module}")
                self.assertTrue(callable(getattr(module, spec.optimize)))
                self.assertTrue(dataclasses.is_dataclass(getattr(module, spec.request)))
                self.assertTrue(hasattr(default_cfg().analysisconfig, spec.setting))


if __name__ == "__main__":
    unittest.main()
