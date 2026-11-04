"""training.beat_critic_optimizer: the torch network against the tracker's NumPy inference of its exported arrays, the
training windows against the tracker's windows, the wrong grids and their labels, the batches, the learning-rate
schedule, the decoder's readings, and small runs of the whole optimizer on synthetic tracks on the CPU: the critic
written where asked and read by the tracker's loader, from the shipped critic and from scratch, the refusals, an
export that differs from the trained net refused, and the held-out check that keeps a worse critic from being
written. The torch parts need the training group (skipped without it)."""
import contextlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.beat.general import critic as C
from training import beat_critic_optimizer as O
from training import beat_frames, model_start as MS
from training import beat_tcn_optimizer
from training.beat_grid_check import GridCheck
from training.text import TrainingError

from unit.training_tests import training_fixtures as F

RNG_SEED = 20261027
SHIPPED_CRITIC = F.shipped("beat_critic.npz")
SHIPPED_TCN = F.shipped("beat_tcn.npz")
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "torch is the training group's (uv sync --group training)")


def windows_input(rng, batch=6, feats=O.FEATURES):
    x = rng.normal(0, 1, (batch, feats + C.SLOTS, C.WIN * C.SLOTS)).astype(np.float32)
    x[:, feats:] = np.eye(C.SLOTS)[np.arange(C.WIN * C.SLOTS) % C.SLOTS].T
    return x


@needs_torch
class NetworkTests(unittest.TestCase):
    def test_state_dict_is_the_shipped_arrays(self):
        shipped = C.load_weights(SHIPPED_CRITIC)[0]
        P = O.export(O.network())
        self.assertEqual({k: v.shape for k, v in P.items()}, {k: v.shape for k, v in shipped.items()})
        self.assertTrue(all(v.dtype == np.float32 for v in P.values()))

    def test_shipped_fold_model_read_and_exported_unchanged(self):
        for P in C.load_weights(SHIPPED_CRITIC):
            Q = O.export(O.network_of(P))
            self.assertEqual(set(Q), set(P))
            for k in P:
                np.testing.assert_array_equal(Q[k], P[k])

    def test_exported_arrays_are_the_trackers_critic(self):
        import torch
        torch.manual_seed(RNG_SEED)
        x = windows_input(np.random.default_rng(RNG_SEED))
        for net in (O.network(), O.network_of(C.load_weights(SHIPPED_CRITIC)[0])):
            with self.subTest(net=net):
                self.assertLess(O.check_export(net, O.export(net), x, torch.device("cpu")), 1e-4)

    def test_a_changed_array_is_seen(self):
        import torch
        net = O.network_of(C.load_weights(SHIPPED_CRITIC)[0])
        P = O.export(net)
        P["out.bias"] = P["out.bias"] + 0.01
        diff = O.check_export(net, P, windows_input(np.random.default_rng(RNG_SEED)), torch.device("cpu"))
        self.assertAlmostEqual(diff, 0.01, places=4)


class WindowTests(unittest.TestCase):
    def test_training_window_is_the_trackers_window(self):
        rng = np.random.default_rng(RNG_SEED)
        beats = F.steady_beats(30.0, 128.0)
        frames = F.synthetic_frames(int(30.0 / F.FRAME_SEC), beats, 3)
        tr = C.track_features(frames)
        rows = np.diff(tr["cs"], axis=0)
        for kind in ("ref", "half", "r32", "d+1"):
            g = beats if kind == "ref" else O.wrong_grid(kind, beats)
            for k0 in (0, int(rng.integers(1, g.size - C.WIN - 1)), g.size - C.WIN - 1):
                with self.subTest(kind=kind, k0=k0):
                    np.testing.assert_allclose(O.window(rows, tr["t0"], tr["hop"], g, k0), C._window(tr, g, k0).T,
                                               rtol=1e-3, atol=2e-3)


class GridTests(unittest.TestCase):
    REF = np.arange(64) * 0.5 + 1.0

    def test_wrong_readings(self):
        half = O.wrong_grid("half", self.REF)
        np.testing.assert_allclose(half, self.REF[:-1] + 0.25)
        np.testing.assert_allclose(O.wrong_grid("q1", self.REF)[:3], [1.125, 1.625, 2.125])
        self.assertAlmostEqual(np.median(np.diff(O.wrong_grid("r32", self.REF))), 0.5 * 2 / 3)
        self.assertAlmostEqual(np.median(np.diff(O.wrong_grid("r34", self.REF))), 0.5 * 4 / 3)
        drift = O.wrong_grid("d+1", self.REF)
        self.assertEqual(drift[0], self.REF[0])
        self.assertAlmostEqual(drift[-1] - drift[0], 1.01 * (self.REF[-1] - self.REF[0]))
        for kind in O.KINDS:
            with self.subTest(kind=kind):
                g = O.wrong_grid(kind, self.REF)
                self.assertTrue(np.all(np.diff(g) > 0))

    def test_labels(self):
        self.assertEqual(O.label(self.REF, self.REF[5:37]), 1)
        self.assertEqual(O.label(self.REF, self.REF[5:37] + 0.04), 1)                  # within 50 ms: right
        self.assertEqual(O.label(self.REF, O.wrong_grid("half", self.REF)[5:37]), 0)
        off = self.REF[5:37].copy()
        off[:3] += 0.1                                                                 # three far: neither
        self.assertEqual(O.label(self.REF, off), -1)
        off[3] += 0.1
        self.assertEqual(O.label(self.REF, off), 0)

    def test_local_edit_keeps_the_grid_outside_its_stretch(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(200):
            g = O.local_edit(rng, self.REF)
            self.assertTrue(np.all(np.diff(g) > 0))
            self.assertEqual(g[0], self.REF[0])
            self.assertEqual(g[-1], self.REF[-1])
            kept = np.isin(self.REF, g).sum()
            self.assertGreaterEqual(kept, self.REF.size - 32)
        short = self.REF[:10]
        self.assertIs(O.local_edit(rng, short), short)                                 # too short to edit

    def test_new_fold_models_have_the_shipped_shapes(self):
        like = C.load_weights(SHIPPED_CRITIC)[0]
        P = O._init_critic(like, 0)
        self.assertEqual({k: v.shape for k, v in P.items()}, {k: v.shape for k, v in like.items()})
        for k, v in P.items():
            fan_in = int(np.prod(like[k.replace(".bias", ".weight")].shape[1:]))
            self.assertLessEqual(float(np.abs(v).max()), 1 / np.sqrt(fan_in))


class BatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_labelled_windows_of_the_songs_alone(self):
        rng = np.random.default_rng(RNG_SEED)
        rows = [rng.normal(0, 1, (int(30.0 / F.FRAME_SEC), 4)).astype(np.float16) + 10 * i for i in range(3)]
        meta, offset = [], 0
        for r in rows:
            meta.append((offset, r.shape[0], 0.0, F.FRAME_SEC))
            offset += r.shape[0]
        x_all = np.concatenate(rows)
        refs = [F.steady_beats(30.0, 120.0 + 5 * i) for i in range(3)]
        decs = [[O.wrong_grid("r23", refs[0])], [], []]
        draw = O.sampler(x_all, meta, refs, decs, [0, 2], seed=1)
        x, y = draw()
        self.assertEqual(x.shape, (O.BATCH, 4 + C.SLOTS, C.WIN * C.SLOTS))
        self.assertEqual((x.dtype, y.dtype), (np.float32, np.float32))
        self.assertTrue(set(np.unique(y)) <= {0.0, 1.0})
        level = x[:, :4].mean((1, 2))                                                # song i's rows are near 10 i
        self.assertTrue(np.all((np.abs(level) < 3) | (np.abs(level - 20) < 3)))       # songs 0 and 2's alone
        self.assertTrue(np.any(np.abs(level) < 3) and np.any(np.abs(level - 20) < 3))
        again, y2 = O.sampler(x_all, meta, refs, decs, [0, 2], seed=1)()
        np.testing.assert_array_equal(x, again)                                      # a fold model's seed: its batches
        np.testing.assert_array_equal(y, y2)


class ScheduleTests(unittest.TestCase):
    def test_fine_tuning_decays_from_the_first_step(self):
        steps = 150
        f = np.array([O._rate(s, steps, 0.0) for s in range(1, steps + 1)])
        self.assertEqual(f[0], 1.0)
        self.assertTrue(np.all(np.diff(f) < 0))
        self.assertGreater(f[-1], 0.0)
        self.assertLess(f[-1], 0.001)

    def test_from_scratch_is_the_tcns_warmup_and_decay(self):
        for steps in (60, 6000):
            for s in (1, 5, steps // 10, steps // 10 + 1, steps // 2, steps):
                self.assertEqual(O._rate(s, steps, O.WARMUP), beat_tcn_optimizer._rate(s, steps, O.WARMUP))


class DecoderReadingTests(unittest.TestCase):
    def test_one_grid_per_training_of_the_beat_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = F.beat_annotation(duration=16.0, bpm=126.0)
            track = F.given_track(Path(tmp), 0, a)
            F.write_beat_frames(Path(tmp), track, F.synthetic_frames(int(16.0 / F.FRAME_SEC), a.beats, 0))
            uid, grids = O._decoder_grids((track, tmp, SHIPPED_TCN, 100.0))
            self.assertEqual(O._decoder_grids(({**track, "uid": F.uid_of(5)}, tmp, SHIPPED_TCN, 100.0)),
                             (F.uid_of(5), []))                                 # no frames: no readings
        self.assertEqual(uid, track["uid"])
        self.assertEqual(len(grids), 3)
        for g in grids:
            self.assertTrue(np.all(np.diff(g) > 0))
            self.assertTrue(np.all((g >= -1.0) & (g <= 17.0)))


@needs_torch
class RunTests(unittest.TestCase):
    SMALL = 4        # the songs of a small run (the minimum lowered to it: the real minimum is checked on its own)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.out = self.dir / "weights" / "beat_critic.npz"

    def tearDown(self):
        F.library_tracks._tracks = None
        self._tmp.cleanup()

    def tracks(self, n, frames=True):
        tracks = []
        for i in range(n):
            a = F.beat_annotation(duration=24.0, bpm=124.0 + i)
            t = F.given_track(self.dir, i, a)
            tracks.append(t)
            if frames:
                F.write_beat_frames(self.cache, t, F.synthetic_frames(int(24.0 / F.FRAME_SEC), a.beats, i))
        F.use_tracks(tracks)
        return tracks

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, start=MS.SHIPPED,
                      current_path=SHIPPED_CRITIC, start_path=SHIPPED_CRITIC, current_models={"tcn": SHIPPED_TCN},
                      steps=2, max_cache_workers=1, device="cpu")
        values.update(kw)
        return O.BeatCriticOptimizationRequest(**values)

    def run_small(self, request, **patches):
        progress = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(MS.MIN_TRACKS, {"fine_tune": (self.SMALL, self.SMALL),
                                                                "scratch": (self.SMALL, self.SMALL)}))
            stack.enter_context(mock.patch.object(O, "_decoder_grids", lambda job: (str(job[0]["uid"]), [])))
            stack.enter_context(mock.patch.object(O, "spawn_pool", F.thread_pool))
            stack.enter_context(mock.patch.object(O, "BATCH", 4))
            stack.enter_context(mock.patch.object(O, "CHECK_WINDOWS", 4))
            stack.enter_context(mock.patch.object(O, "KINDS", ("half", "r32")))     # fewer wrong grids heard
            stack.enter_context(mock.patch.object(beat_frames, "spawn_pool", F.no_pool))
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(O, name, value))
            result = O.optimize_beat_critic(request, optimize_progress=lambda v, t: progress.append(v))
        return result, progress

    def test_fine_tuned_critic_written_and_read_by_the_trackers_loader(self):
        self.tracks(self.SMALL)
        result, progress = self.run_small(self.request())
        self.assertTrue(result.saved)
        self.assertEqual(result.output_path, self.out.resolve())
        self.assertEqual((result.track_count, result.steps, result.skipped_tracks, result.device),
                         (self.SMALL, 2, (), "cpu"))
        nets, shipped = C.load_weights(str(self.out)), C.load_weights(SHIPPED_CRITIC)
        self.assertEqual(len(nets), len(shipped))
        self.assertEqual({k: v.shape for k, v in nets[0].items()}, {k: v.shape for k, v in shipped[0].items()})
        self.assertFalse(np.array_equal(nets[0]["out.weight"], shipped[0]["out.weight"]))
        self.assertLess(float(np.abs(nets[0]["out.weight"] - shipped[0]["out.weight"]).max()), 0.01)  # a few small steps
        self.assertTrue(all(np.all(np.isfinite(v)) for P in nets for v in P.values()))
        with np.load(self.out, allow_pickle=False) as z:
            self.assertTrue(all(k.startswith(("f0/", "f1/", "f2/", "f3/", "f4/")) for k in z.files))
        self.assertTrue(0.0 <= result.pairwise_before <= 1.0 and 0.0 <= result.pairwise_after <= 1.0)
        self.assertTrue(np.isfinite(result.final_loss) and result.seconds_per_fold > 0)
        self.assertFalse((self.cache / "_critic_training").exists())
        self.assertEqual(progress[-1], 100)

    def test_from_scratch(self):
        self.tracks(self.SMALL)
        result, _ = self.run_small(self.request(start=MS.SCRATCH, start_path=None, steps=1))
        self.assertEqual(result.steps, 1)
        nets, shipped = C.load_weights(str(self.out)), C.load_weights(SHIPPED_CRITIC)
        self.assertEqual(len(nets), len(shipped))
        self.assertFalse(np.allclose(nets[0]["inp.weight"], shipped[0]["inp.weight"], atol=0.01))   # not the shipped
        self.assertFalse(np.array_equal(nets[0]["inp.weight"], nets[1]["inp.weight"]))              # other seeds

    def test_an_export_that_differs_is_refused(self):
        self.tracks(self.SMALL)
        with self.assertRaises(RuntimeError):
            self.run_small(self.request(), check_export=lambda *a: 1.0)
        self.assertFalse(self.out.exists())
        self.assertFalse((self.cache / "_critic_training").exists())     # the mapped rows let go on the failure too

    def test_worse_held_out_grids_leave_the_critic_unwritten(self):
        self.tracks(self.SMALL)
        worse = GridCheck(songs=2, beats_before_sec=0.0, beats_after_sec=0.0, bars_before_sec=1.0, bars_after_sec=3.0)
        check = mock.Mock(return_value=worse)
        result, _ = self.run_small(self.request(cv_folds=2), grid_check=check)
        self.assertFalse(result.saved or self.out.exists())
        self.assertTrue(result.cross_validated)
        before, after = check.call_args.args[3:5]
        self.assertEqual((before.critic, before.tcn), (SHIPPED_CRITIC, SHIPPED_TCN))  # the critic and beat model in use
        self.assertEqual(after.tcn, SHIPPED_TCN)
        self.assertFalse(Path(after.critic).exists())                 # the held-out critic, removed after the check
        self.assertFalse((self.cache / "_critic_training").exists())

    def test_too_few_songs_refused(self):
        self.tracks(MS.MIN_TRACKS["fine_tune"][0] - 1, frames=False)
        with mock.patch.object(O, "ensure_frames", side_effect=AssertionError("no frames for a refused run")):
            with self.assertRaises(TrainingError):
                O.optimize_beat_critic(self.request())
        self.assertFalse(self.out.exists())

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_beat_critic(self.request(output_path=self.dir / "critic.json"))

    def test_the_models_in_use_must_be_given(self):
        for missing in ({"current_path": None}, {"current_models": None}, {"current_models": {"critic": SHIPPED_CRITIC}}):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                O.optimize_beat_critic(self.request(**missing))
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
