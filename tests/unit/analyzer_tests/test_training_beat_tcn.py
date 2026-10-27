"""analyzer.training.beat_tcn_optimizer: the frame TCN's hand-written backward pass against central differences (with
and without dropout), the frame targets, the learning-rate schedule, a new training's initialisation, and small runs
of the whole optimizer on synthetic tracks (the trainings in this process): the model written where asked and read by
the tracker's own loader, the refusals, and the held-out check that keeps a worse model from being written."""
import contextlib
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.beat.general import tcn as T
from analyzer.training import beat_tcn_optimizer as O
from analyzer.training import beat_frames, model_start as MS
from analyzer.training.beat_grid_check import GridCheck
from analyzer.training.text import TrainingError

from unit.analyzer_tests import training_fixtures as F

RNG_SEED = 20261027
SHIPPED_TCN = F.shipped("beat_tcn.npz")


def tiny_net(rng, channels=3, inputs=4):
    P = {"inp.weight": rng.normal(0, 0.5, (channels, inputs, 1)), "inp.bias": rng.normal(0, 0.1, channels)}
    for i in range(len(T.DILATIONS)):
        P[f"blocks.{i}.conv.weight"] = rng.normal(0, 0.4, (channels, channels, T.K))
        P[f"blocks.{i}.conv.bias"] = rng.normal(0, 0.1, channels)
        P[f"blocks.{i}.mix.weight"] = rng.normal(0, 0.4, (channels, channels, 1))
        P[f"blocks.{i}.mix.bias"] = rng.normal(0, 0.1, channels)
    P["out.weight"], P["out.bias"] = rng.normal(0, 0.5, (2, channels, 1)), rng.normal(0, 0.1, 2)
    return P


class GradientTests(unittest.TestCase):
    def check(self, dropout_seed):
        rng = np.random.default_rng(RNG_SEED)
        P = tiny_net(rng)
        x = rng.normal(0, 1, (4, 40))
        R = rng.normal(0, 1, (2, 40))          # the loss: a fixed linear functional of the logits

        def forward(P):
            return O._forward(P, x, None if dropout_seed is None else np.random.default_rng(dropout_seed))

        def loss(P):
            return float(np.sum(R * forward(P)[0]))
        _, saved = forward(P)
        G = O._backward(P, saved, R)
        self.assertEqual(set(G), set(P))
        for key in P:
            with self.subTest(key=key):
                self.assertEqual(G[key].shape, P[key].shape)
                picks = rng.choice(P[key].size, size=min(6, P[key].size), replace=False)
                numeric = F.numeric_grad(loss, P, key, picks)
                analytic = G[key].reshape(-1)[picks]
                scale = max(1e-3, float(np.abs(numeric).max()))
                np.testing.assert_allclose(analytic, numeric, atol=1e-4 * scale, rtol=1e-3)

    def test_backward_matches_central_differences(self):
        self.check(None)

    def test_backward_with_dropout_matches_central_differences(self):
        self.check(5)

    def test_forward_without_dropout_is_the_trackers_network(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        P = {k: v.astype(np.float32) for k, v in tiny_net(rng, inputs=77).items()}
        P["mu"], P["sd"] = rng.normal(0, 1, 77).astype(np.float32), rng.uniform(0.5, 2, 77).astype(np.float32)
        frames = rng.normal(0, 1, (60, 77)).astype(np.float32)
        logits, _ = O._forward(P, ((frames - P["mu"]) / P["sd"]).T)
        beat, down = T.activations(P, frames)
        np.testing.assert_allclose(1 / (1 + np.exp(-logits[0])), beat, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(1 / (1 + np.exp(-logits[1])), down, rtol=1e-4, atol=1e-6)

    def test_dropout_keeps_the_expected_activation(self):
        rng = np.random.default_rng(RNG_SEED + 2)
        P = tiny_net(rng, channels=16)
        x = rng.normal(0, 1, (4, 400))
        saved = O._forward(P, x, np.random.default_rng(3))[1]
        keep = np.concatenate([c[5].ravel() for c in saved[1]])
        self.assertAlmostEqual(float(np.mean(keep == 0)), O.DROPOUT, delta=0.01)
        self.assertTrue(np.allclose(keep[keep > 0], 1 / (1 - O.DROPOUT)))


class TargetTests(unittest.TestCase):
    def test_each_beat_frame_and_half_its_neighbours(self):
        ft = np.arange(100) * 0.01
        beats = np.array([0.10, 0.50, 0.90])
        rows = np.array([[0.0, 1.0, 150.0, 0.10, 2.0, 150.0]])      # bars of two: downbeats at 0.10, 0.90
        yb, yd = O.track_targets(ft, beats, rows)
        self.assertEqual(sorted(np.flatnonzero(yb == 1.0)), [10, 50, 90])
        self.assertEqual(sorted(np.flatnonzero(yb == 0.5)), [9, 11, 49, 51, 89, 91])
        self.assertEqual(float(yb.sum()), 6.0)
        self.assertEqual(sorted(np.flatnonzero(yd == 1.0)), [10, 90])
        self.assertEqual(float(yd.sum()), 4.0)

    def test_beats_off_the_frames_are_left_out(self):
        ft = np.arange(50) * 0.01
        yb, _ = O.track_targets(ft, np.array([-1.0, 0.2, 9.0]), np.zeros((0, 6)))
        self.assertEqual(np.flatnonzero(yb == 1.0).tolist(), [20])

    def test_widen_never_lowers_a_target(self):
        y = np.random.default_rng(RNG_SEED).random(200).astype(np.float32)
        w = O._widen(y)
        self.assertTrue(np.all(w >= y))
        self.assertTrue(np.all(w[1:] >= 0.5 * y[:-1]) and np.all(w[:-1] >= 0.5 * y[1:]))


class ScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine_decay(self):
        steps, lr = 200, 1e-3
        rates = np.array([O._rate(lr, s, steps, 0.1) for s in range(1, steps + 1)])
        w = 20
        self.assertTrue(np.all(np.diff(rates[:w]) > 0))
        self.assertAlmostEqual(rates[w - 1], lr)
        self.assertAlmostEqual(rates[w], lr)                        # the first step after the warmup: the full rate
        self.assertTrue(np.all(np.diff(rates[w:]) <= 1e-15))
        self.assertTrue(np.all(rates > 0) and np.all(rates <= lr + 1e-15))
        self.assertLess(rates[-1], 0.01 * lr)

    def test_no_warmup_starts_at_the_full_rate(self):
        self.assertAlmostEqual(O._rate(5e-4, 1, 600, 0.0), 5e-4)
        self.assertAlmostEqual(O._rate(5e-4, 1, 1, 0.0), 5e-4)
        self.assertGreater(O._rate(5e-4, 600, 600, 0.0), 0.0)


class InitTests(unittest.TestCase):
    def test_new_training_has_the_shipped_shapes_and_the_given_statistics(self):
        shipped = T.load_weights(SHIPPED_TCN)[0]
        mu, sd = np.arange(77.0), np.full(77, 2.0)
        P = O._init_net(mu, sd, 0)
        self.assertEqual({k: v.shape for k, v in P.items()}, {k: v.shape for k, v in shipped.items()})
        np.testing.assert_array_equal(P["mu"], mu.astype(np.float32))
        for k, v in P.items():
            if k not in ("mu", "sd"):
                fan_in = int(np.prod(P[k.replace(".bias", ".weight")].shape[1:]))
                self.assertLessEqual(float(np.abs(v).max()), 1 / np.sqrt(fan_in))
        self.assertFalse(np.array_equal(P["inp.weight"], O._init_net(mu, sd, 1)["inp.weight"]))
        np.testing.assert_array_equal(P["inp.weight"], O._init_net(mu, sd, 0)["inp.weight"])

    def test_input_statistics_of_the_training_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = np.random.default_rng(RNG_SEED).normal(3, 2, (500, 77)).astype(np.float16)
            np.save(Path(tmp) / "x.npy", x)
            mu, sd = O._input_stats(tmp)
            np.testing.assert_allclose(mu, x.astype(np.float64).mean(0), atol=1e-9)
            np.testing.assert_allclose(sd, x.astype(np.float64).std(0) + 1e-6, rtol=1e-6)


def sequential_train_all(nets, work, songs, steps, lr, max_workers, progress, span, label, warmup=0.0):
    """_train_all without its process pool: each training in turn, in this process."""
    q = queue.Queue()
    out = [O._train_one((P, str(work), np.asarray(songs), steps, lr, i, q, warmup)) for i, P in enumerate(nets)]
    if progress:
        progress(span[1], label)
    return out


SMALL = 6        # the songs of a small run (the minimum lowered to it: the real minimum is checked on its own)


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.out = self.dir / "weights" / "beat_tcn.npz"

    def tearDown(self):
        F.library_tracks._tracks = None
        self._tmp.cleanup()

    def tracks(self, n, frames=True):
        tracks = []
        for i in range(n):
            a = F.beat_annotation(duration=8.0, bpm=120.0 + i)
            t = F.given_track(self.dir, i, a)
            tracks.append(t)
            if frames:
                F.write_beat_frames(self.cache, t, F.synthetic_frames(int(8.0 / F.FRAME_SEC), a.beats, i))
        F.use_tracks(tracks)
        return tracks

    def small(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"fine_tune": (SMALL, SMALL), "scratch": (SMALL, SMALL)}))
        return self.tracks(SMALL)

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, start=MS.SHIPPED,
                      current_path=SHIPPED_TCN, start_path=SHIPPED_TCN, steps=2, max_cache_workers=1)
        values.update(kw)
        return O.BeatTcnOptimizationRequest(**values)

    def run_small(self, request, **patches):
        progress = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(O, "_train_all", sequential_train_all))
            stack.enter_context(mock.patch.object(O, "CROP", 96))
            stack.enter_context(mock.patch.object(O, "BATCH", 2))
            stack.enter_context(mock.patch.object(beat_frames, "spawn_pool", F.no_pool))
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(O, name, value))
            result = O.optimize_beat_tcn(request, optimize_progress=lambda v, t: progress.append(v))
        return result, progress

    def test_fine_tuned_model_written_and_read_by_the_trackers_loader(self):
        self.small()
        result, progress = self.run_small(self.request())
        self.assertTrue(result.saved)
        self.assertEqual(result.output_path, self.out.resolve())
        self.assertEqual((result.track_count, result.steps, result.skipped_tracks), (SMALL, 2, ()))
        nets, shipped = T.load_weights(str(self.out)), T.load_weights(SHIPPED_TCN)
        self.assertEqual(len(nets), len(shipped))
        self.assertEqual(set(nets[0]), set(shipped[0]))
        self.assertFalse(np.array_equal(nets[0]["inp.weight"], shipped[0]["inp.weight"]))     # it was trained
        np.testing.assert_array_equal(nets[0]["mu"], shipped[0]["mu"])                         # its statistics kept
        self.assertTrue(all(np.all(np.isfinite(v)) for P in nets for v in P.values()))
        self.assertTrue(0.0 <= result.beat_ap_before <= 1.0 and 0.0 <= result.beat_ap_after <= 1.0)
        with np.load(self.out, allow_pickle=False) as z:
            self.assertIn('"steps": 2', str(z["training_json"]))
        self.assertFalse((self.cache / "_tcn_training").exists())         # the training frames removed
        self.assertEqual(progress[-1], 100)

    def test_from_scratch_starts_from_the_librarys_statistics(self):
        self.small()
        result, _ = self.run_small(self.request(start=MS.SCRATCH, start_path=None, steps=1))
        nets = T.load_weights(str(self.out))
        self.assertEqual(result.track_count, SMALL)
        self.assertFalse(np.allclose(nets[0]["mu"], T.load_weights(SHIPPED_TCN)[0]["mu"]))

    def test_too_few_songs_refused_before_any_work(self):
        self.tracks(MS.MIN_TRACKS["fine_tune"][0] - 1, frames=False)
        with mock.patch.object(O, "ensure_frames", side_effect=AssertionError("no frames for a refused run")):
            with self.assertRaises(TrainingError):
                O.optimize_beat_tcn(self.request())
        self.assertFalse(self.out.exists())

    def test_songs_whose_frames_are_lost_count_against_the_minimum(self):
        tracks = self.small()
        (self.cache / f"{tracks[0]['uid']}.npz").write_bytes(b"not a cache")
        with mock.patch.object(O, "ensure_frames", lambda tracks, *a, **k: tracks):
            with self.assertRaises(TrainingError):
                O.optimize_beat_tcn(self.request())
        self.assertFalse(self.out.exists())
        self.assertFalse((self.cache / "_tcn_training").exists())

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_beat_tcn(self.request(output_path=self.dir / "beat_tcn.json"))

    def test_worse_held_out_grids_leave_the_model_unwritten(self):
        self.small()
        worse = GridCheck(songs=3, beats_before_sec=1.0, beats_after_sec=5.0, bars_before_sec=0.0, bars_after_sec=0.0)
        result, _ = self.run_small(self.request(cv_folds=2), grid_check=mock.Mock(return_value=worse))
        self.assertFalse(result.saved)
        self.assertTrue(result.cross_validated)
        self.assertFalse(self.out.exists())

    def test_keep_if_worse_writes_it_anyway(self):
        self.small()
        worse = GridCheck(songs=3, beats_before_sec=1.0, beats_after_sec=5.0, bars_before_sec=0.0, bars_after_sec=0.0)
        check = mock.Mock(return_value=worse)
        result, _ = self.run_small(self.request(cv_folds=2, keep_if_worse=True), grid_check=check)
        self.assertTrue(result.saved and self.out.is_file())
        held = check.call_args.args[0]
        self.assertEqual(len(held), SMALL // 2)               # half the songs held out (two folds)
        self.assertEqual(check.call_args.args[4].tcn[-len("held_out_model.npz"):], "held_out_model.npz")


if __name__ == "__main__":
    unittest.main()
