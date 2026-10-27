"""analyzer.training.beat_critic_optimizer: the metronome critic's hand-written backward pass (dilated 3x3 convolutions,
mean and max pooling) against central differences, its forward against the tracker's, the training windows against
the tracker's windows, the wrong grids and their labels, and small runs of the whole optimizer on synthetic tracks
(the fold models trained in this process): the critic written where asked and read by the tracker's loader."""
import contextlib
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.beat.general import critic as C
from analyzer.training import beat_critic_optimizer as O
from analyzer.training import beat_frames, model_start as MS
from analyzer.training.beat_grid_check import GridCheck
from analyzer.training.text import TrainingError

from unit.analyzer_tests import training_fixtures as F

RNG_SEED = 20261027
SHIPPED_CRITIC = F.shipped("beat_critic.npz")
SHIPPED_TCN = F.shipped("beat_tcn.npz")


def tiny_critic(rng, feats=3, inst=2, hidden=3):
    P = {"inst.weight": rng.normal(0, 0.5, (inst, feats, 1, 1)), "inst.bias": rng.normal(0, 0.1, inst),
         "inp.weight": rng.normal(0, 0.5, (hidden, inst + C.SLOTS, 1, 1)), "inp.bias": rng.normal(0, 0.1, hidden)}
    for k in range(len(C.DIL)):
        P[f"convs.{k}.weight"] = rng.normal(0, 0.3, (hidden, hidden, 3, 3))
        P[f"convs.{k}.bias"] = rng.normal(0, 0.1, hidden)
    P["out.weight"], P["out.bias"] = rng.normal(0, 0.5, (1, 2 * hidden)), rng.normal(0, 0.1, 1)
    return P


def tiny_input(rng, batch=2, feats=3):
    x = rng.normal(0, 1, (batch, feats + C.SLOTS, C.WIN * C.SLOTS))
    x[:, feats:] = np.eye(C.SLOTS)[np.arange(C.WIN * C.SLOTS) % C.SLOTS].T
    return x


class GradientTests(unittest.TestCase):
    def test_backward_matches_central_differences(self):
        rng = np.random.default_rng(RNG_SEED)
        P = tiny_critic(rng)
        x = tiny_input(rng)
        r = rng.normal(0, 1, 2)

        def loss(P):
            return float(r @ O._forward(P, x)[0])
        _, saved = O._forward(P, x)
        G = O._backward(P, saved, r)
        self.assertEqual(set(G), set(P))
        for key in P:
            with self.subTest(key=key):
                self.assertEqual(G[key].shape, P[key].shape)
                picks = rng.choice(P[key].size, size=min(6, P[key].size), replace=False)
                numeric = F.numeric_grad(loss, P, key, picks, eps=1e-6)
                scale = max(1e-3, float(np.abs(numeric).max()))
                np.testing.assert_allclose(G[key].reshape(-1)[picks], numeric, atol=2e-4 * scale, rtol=2e-3)

    def test_conv3x3_back_input_gradient(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        W, b = rng.normal(0, 1, (2, 3, 3, 3)), rng.normal(0, 1, 2)
        x = rng.normal(0, 1, (1, 3, 4, 10))
        g = rng.normal(0, 1, (1, 2, 4, 10))
        for d in (1, 3):
            _, db, dx = O._conv3x3_back(W, x, g, d)
            np.testing.assert_allclose(db, g.sum((0, 2, 3)))
            picks = rng.choice(x.size, 8, replace=False)
            numeric = F.numeric_grad(lambda P: float(np.sum(g * C._conv3x3(W, b, P["x"], d))), {"x": x}, "x", picks)
            np.testing.assert_allclose(dx.reshape(-1)[picks], numeric, rtol=1e-6, atol=1e-8)

    def test_forward_is_the_trackers_critic(self):
        P = C.load_weights(SHIPPED_CRITIC)[0]
        x = tiny_input(np.random.default_rng(RNG_SEED + 2), feats=81).astype(np.float32)
        np.testing.assert_allclose(O._forward(P, x)[0], C._forward(P, x), rtol=1e-5, atol=1e-5)


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


def sequential_train_all(nets, work, meta, refs, decs, songs, steps, lr, max_workers, progress, span, label, warmup=0.0):
    q = queue.Queue()
    return [O._train_one((P, str(work), meta, refs, decs, np.asarray(songs), steps, lr, i, q, warmup))
            for i, P in enumerate(nets)]


SMALL = 4


class RunTests(unittest.TestCase):
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
                      current_path=SHIPPED_CRITIC, start_path=SHIPPED_CRITIC, tcn_path=SHIPPED_TCN, steps=2,
                      max_cache_workers=1)
        values.update(kw)
        return O.BeatCriticOptimizationRequest(**values)

    def run_small(self, request, **patches):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(MS.MIN_TRACKS, {"fine_tune": (SMALL, SMALL), "scratch": (SMALL, SMALL)}))
            stack.enter_context(mock.patch.object(O, "_train_all", sequential_train_all))
            stack.enter_context(mock.patch.object(O, "_decoder_grids", lambda job: (str(job[0]["uid"]), [])))
            stack.enter_context(mock.patch.object(O, "spawn_pool", F.thread_pool))
            stack.enter_context(mock.patch.object(O, "BATCH", 4))
            stack.enter_context(mock.patch.object(O, "KINDS", ("half", "r32")))     # fewer wrong grids heard
            stack.enter_context(mock.patch.object(beat_frames, "spawn_pool", F.no_pool))
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(O, name, value))
            return O.optimize_beat_critic(request)

    def test_fine_tuned_critic_written_and_read_by_the_trackers_loader(self):
        self.tracks(SMALL)
        result = self.run_small(self.request())
        self.assertTrue(result.saved)
        self.assertEqual((result.track_count, result.steps, result.skipped_tracks), (SMALL, 2, ()))
        nets, shipped = C.load_weights(str(self.out)), C.load_weights(SHIPPED_CRITIC)
        self.assertEqual(len(nets), len(shipped))
        self.assertEqual({k: v.shape for k, v in nets[0].items()}, {k: v.shape for k, v in shipped[0].items()})
        self.assertFalse(np.array_equal(nets[0]["out.weight"], shipped[0]["out.weight"]))
        with np.load(self.out, allow_pickle=False) as z:
            self.assertTrue(z.files)
        self.assertTrue(0.0 <= result.pairwise_before <= 1.0 and 0.0 <= result.pairwise_after <= 1.0)
        self.assertFalse((self.cache / "_critic_training").exists())

    def test_from_scratch(self):
        self.tracks(SMALL)
        result = self.run_small(self.request(start=MS.SCRATCH, start_path=None, steps=1))
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(C.load_weights(str(self.out))), len(C.load_weights(SHIPPED_CRITIC)))

    def test_worse_held_out_grids_leave_the_critic_unwritten(self):
        self.tracks(SMALL)
        worse = GridCheck(songs=2, beats_before_sec=0.0, beats_after_sec=0.0, bars_before_sec=1.0, bars_after_sec=3.0)
        result = self.run_small(self.request(cv_folds=2), grid_check=mock.Mock(return_value=worse))
        self.assertFalse(result.saved or self.out.exists())
        self.assertTrue(result.cross_validated)

    def test_too_few_songs_refused(self):
        self.tracks(MS.MIN_TRACKS["fine_tune"][0] - 1, frames=False)
        with mock.patch.object(O, "ensure_frames", side_effect=AssertionError("no frames for a refused run")):
            with self.assertRaises(TrainingError):
                O.optimize_beat_critic(self.request())
        self.assertFalse(self.out.exists())

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_beat_critic(self.request(output_path=self.dir / "critic.json"))


if __name__ == "__main__":
    unittest.main()
