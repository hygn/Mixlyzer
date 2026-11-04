"""training.beat_tcn_optimizer: the torch network against the tracker's NumPy inference of its exported arrays, the
frame targets, the learning-rate schedule, the crops (each within one song), and small runs of the whole optimizer
on synthetic tracks on the CPU: the model written where asked and read by the tracker's own loader, the refusals, and
the held-out check that keeps a worse model from being written. The torch parts need the training group (skipped
without it)."""
import contextlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.beat.general import tcn as T
from training import beat_tcn_optimizer as O
from training import beat_frames, model_start as MS
from training.beat_grid_check import GridCheck
from training.text import TrainingError

from unit.training_tests import training_fixtures as F

RNG_SEED = 20261105
SHIPPED_TCN = F.shipped("beat_tcn.npz")
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "torch is the training group's (uv sync --group training)")


@needs_torch
class NetworkTests(unittest.TestCase):
    def test_state_dict_is_the_shipped_arrays(self):
        shipped = T.load_weights(SHIPPED_TCN)[0]
        P = O.export(O.network(), np.zeros(77), np.ones(77))
        self.assertEqual({k: v.shape for k, v in P.items()}, {k: v.shape for k, v in shipped.items()})

    def test_exported_arrays_are_the_trackers_network(self):
        import torch
        torch.manual_seed(RNG_SEED)
        net = O.network()
        rng = np.random.default_rng(RNG_SEED)
        mu, sd = rng.normal(0, 1, 77), rng.uniform(0.5, 2, 77)
        frames = rng.normal(0, 1, (700, 77)).astype(np.float32)
        P = O.export(net, mu, sd)
        self.assertLess(O.check_export(net, P, frames, torch.device("cpu")), 1e-5)

    def test_dropout_only_while_training(self):
        import torch
        net = O.network()
        x = torch.randn(1, 77, 300)
        net.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(net(x), net(x)))
        net.train()
        with torch.no_grad():
            self.assertFalse(torch.equal(net(x), net(x)))


class TargetTests(unittest.TestCase):
    def test_each_beat_frame_and_half_its_neighbours(self):
        ft = np.arange(100) * 0.01
        beats = np.array([0.10, 0.50, 0.90])
        rows = np.array([[0.0, 1.0, 150.0, 0.10, 2.0, 150.0]])      # bars of two: downbeats at 0.10, 0.90
        yb, yd = O.track_targets(ft, beats, rows)
        self.assertEqual(sorted(np.flatnonzero(yb == 1.0)), [10, 50, 90])
        self.assertEqual(sorted(np.flatnonzero(yb == 0.5)), [9, 11, 49, 51, 89, 91])
        self.assertEqual(sorted(np.flatnonzero(yd == 1.0)), [10, 90])
        self.assertEqual(float(yd.sum()), 4.0)

    def test_beats_off_the_frames_are_left_out(self):
        ft = np.arange(50) * 0.01
        yb, _ = O.track_targets(ft, np.array([-1.0, 0.2, 9.0]), np.zeros((0, 6)))
        self.assertEqual(np.flatnonzero(yb == 1.0).tolist(), [20])


class ScheduleTests(unittest.TestCase):
    def test_warmup_then_cosine_decay(self):
        steps = 200
        f = np.array([O._rate(s, steps, 0.1) for s in range(1, steps + 1)])
        w = 20
        self.assertTrue(np.all(np.diff(f[:w]) > 0))
        self.assertAlmostEqual(f[w - 1], 1.0)
        self.assertAlmostEqual(f[w], 1.0)
        self.assertTrue(np.all(np.diff(f[w:]) <= 1e-15))
        self.assertTrue(np.all(f > 0) and np.all(f <= 1 + 1e-15))
        self.assertLess(f[-1], 0.01)


@needs_torch
class BatchTests(unittest.TestCase):
    def test_each_crop_stays_in_its_song(self):
        import torch
        sizes = [400, 250, 900]
        x = torch.cat([torch.full((n, 77), float(i + 1)) for i, n in enumerate(sizes)]).half()
        y = torch.zeros((sum(sizes), 2)).half()
        data = O.TrainingFrames(x, y, np.r_[0, np.cumsum(sizes)], [{}] * 3)
        gen = torch.Generator()
        gen.manual_seed(RNG_SEED)
        off = torch.as_tensor(data.offsets)
        with mock.patch.object(O, "CROP", 200), mock.patch.object(O, "BATCH", 32):
            xb, yb = O._batch(data, off, torch.arange(3), torch.zeros(77), torch.ones(77), gen)
        self.assertEqual(tuple(xb.shape), (32, 77, 200))
        self.assertEqual(tuple(yb.shape), (32, 2, 200))
        for crop in xb:
            self.assertEqual(torch.unique(crop).numel(), 1)          # one song's frames only


@needs_torch
class RunTests(unittest.TestCase):
    SMALL = 6        # the songs of a small run (the minimum lowered to it: the real minimum is checked on its own)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.out = self.dir / "weights" / "beat_tcn.npz"
        self.out.parent.mkdir(parents=True)

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
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"scratch": (self.SMALL, self.SMALL)}))
        return self.tracks(self.SMALL)

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, steps=2, trainings=2,
                      max_cache_workers=1, device="cpu")
        values.update(kw)
        return O.BeatTcnOptimizationRequest(**values)

    def run_small(self, request, **patches):
        progress = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(O, "CROP", 96))
            stack.enter_context(mock.patch.object(O, "BATCH", 2))
            stack.enter_context(mock.patch.object(beat_frames, "spawn_pool", F.no_pool))
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(O, name, value))
            result = O.optimize_beat_tcn(request, optimize_progress=lambda v, t: progress.append(v))
        return result, progress

    def test_model_written_and_read_by_the_trackers_loader(self):
        self.small()
        result, progress = self.run_small(self.request())
        self.assertTrue(result.saved)
        self.assertEqual(result.output_path, self.out.resolve())
        self.assertEqual((result.track_count, result.trainings, result.steps, result.skipped_tracks),
                         (self.SMALL, 2, 2, ()))
        nets, shipped = T.load_weights(str(self.out)), T.load_weights(SHIPPED_TCN)
        self.assertEqual(len(nets), 2)
        self.assertEqual(set(nets[0]), set(shipped[0]))
        self.assertFalse(np.array_equal(nets[0]["inp.weight"], nets[1]["inp.weight"]))     # other seeds
        np.testing.assert_array_equal(nets[0]["mu"], nets[1]["mu"])                          # the songs' statistics
        self.assertFalse(np.allclose(nets[0]["mu"], shipped[0]["mu"]))
        self.assertTrue(all(np.all(np.isfinite(v)) for P in nets for v in P.values()))
        with np.load(self.out, allow_pickle=False) as z:
            self.assertIn('"steps": 2', str(z["training_json"]))
        self.assertEqual(progress[-1], 100)

    def test_too_few_songs_refused_before_any_work(self):
        self.tracks(3, frames=False)
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

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_beat_tcn(self.request(output_path=self.dir / "beat_tcn.json"))

    def test_worse_held_out_grids_leave_the_model_unwritten(self):
        self.small()
        worse = GridCheck(songs=3, beats_before_sec=1.0, beats_after_sec=5.0, bars_before_sec=0.0, bars_after_sec=0.0)
        check = mock.Mock(return_value=worse)
        result, _ = self.run_small(self.request(cv_folds=2), grid_check=check)
        self.assertFalse(result.saved)
        self.assertTrue(result.cross_validated)
        self.assertFalse(self.out.exists())
        trial = check.call_args.args[4].tcn                          # the held-out model, removed after the check
        self.assertFalse(Path(trial).exists())


if __name__ == "__main__":
    unittest.main()
