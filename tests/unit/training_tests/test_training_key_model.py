"""training.key_model_optimizer: the torch network against the analyzer's NumPy inference of its exported arrays (the
chord posteriors and the key evidence), its root equivariance, the loss against the bars' key cross-entropy and the
template-chord term computed with the analyzer's model, the optimizer's step and the learning-rate schedule, which
annotations are the user's (edited, legacy, still automatic, raw-looking), the template chords, the training songs and
crops, and small runs of the whole optimizer on synthetic tracks on the CPU: the model written where asked and read by
the analyzer's own loader, the decoder settings kept, the refusals, and the held-out check that keeps worse keys from
being written. The torch parts need the training group (skipped without it)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.key import chord_key as CK
from training import key_model_optimizer as O
from training import model_start as MS
from training.library_tracks import Annotation
from training.text import TrainingError, render

from unit.training_tests import training_fixtures as F

RNG_SEED = 20261027
SHIPPED_KEY = F.shipped("key_chord_cnn.npz")
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "torch is the training group's (uv sync --group training)")


def beat_inputs(rng, n):
    """[n, 36] beat inputs shaped as chord_key.pool_beats makes them (sqrt, each block over its max)."""
    x = np.sqrt(rng.random((n, 36)))
    return np.concatenate([b / b.max(axis=1, keepdims=True) for b in np.split(x, 3, axis=1)], axis=1).astype(np.float32)


def transposed(a, s, blocks):
    """Every pitch class (each 12-block) up ``s`` semitones."""
    return np.concatenate([np.roll(a[..., 12 * b:12 * (b + 1)], s, axis=-1) for b in range(blocks)], axis=-1)


def tiny_batch(rng, B=2, n=12):
    x = np.stack([beat_inputs(rng, n) for _ in range(B)])
    y = rng.integers(0, 24, (B, n))
    w = (rng.random((B, n)) < 0.8).astype(np.float32)
    bar = np.tile(np.arange(n) // 4, (B, 1))
    pos = rng.integers(0, 3, (B, n))
    return x, y, w, bar, pos, O.template_chords(x.reshape(-1, 36)).reshape(B, n, 24)


def numpy_loss(arrays, batch, distill):
    """The training loss with the analyzer's model: each bar's key evidence (the sum of its keyed beats') against the
    share of each key among them, averaged over the bars with a keyed beat, plus ``distill`` times KL(templates ||
    chords) averaged over every beat of the crops."""
    model = CK.ChordKeyModel(arrays=arrays)
    x, y, w, bar, pos, teach = batch
    ce, kl = [], []
    for b in range(x.shape[0]):
        logpc = model.chords(x[b])
        L = model.key_loglik(logpc, pos[b])
        for j in np.unique(bar[b]):
            sel = bar[b] == j
            if w[b][sel].sum() > 0:
                target = np.bincount(y[b][sel], weights=w[b][sel], minlength=24) / w[b][sel].sum()
                ce.append(-(target * CK._log_softmax((L[sel] * w[b][sel, None]).sum(0))).sum())
        kl.extend((teach[b] * (np.log(teach[b] + 1e-8) - logpc)).sum(-1))
    return float(np.mean(ce) + distill * np.mean(kl))


@needs_torch
class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.arrays = CK.read_model_arrays(SHIPPED_KEY)
        self.rng = np.random.default_rng(RNG_SEED)

    def test_state_dict_and_export_are_the_model_files_arrays(self):
        net = O.network(self.arrays)
        self.assertEqual(list(net.state_dict()), list(CK.WEIGHTS))
        P = O.export(net, self.arrays)
        self.assertEqual({k: (v.shape, v.dtype) for k, v in P.items()},
                         {k: (v.shape, v.dtype) for k, v in self.arrays.items()})
        for k in P:
            np.testing.assert_array_equal(P[k], self.arrays[k])

    def test_exported_arrays_are_the_analyzers_model(self):
        import torch
        arrays = {**self.arrays, **O._init_weights(self.arrays, 3)}      # other weights than the shipped
        net = O.network(arrays)
        with torch.no_grad():
            for k in O.TRAINED:
                getattr(net, k).mul_(4.0)                                  # sharper chords than a new model's
        P = O.export(net, arrays)
        x, pos = beat_inputs(self.rng, 600), self.rng.integers(0, 3, 600)
        self.assertLess(O.check_export(net, P, x, pos), 1e-5)
        self.assertFalse(np.array_equal(P["c1_w"], arrays["c1_w"]))
        model = CK.ChordKeyModel(arrays=P)
        with torch.no_grad():
            logpc, L = net(torch.as_tensor(x)[None], torch.as_tensor(pos)[None])
        np.testing.assert_allclose(logpc[0].numpy(), model.chords(x), atol=1e-4)
        np.testing.assert_allclose(L[0].numpy(), model.key_loglik(model.chords(x), pos), atol=1e-4)

    def test_transposed_input_transposes_the_chords_and_keys(self):
        import torch
        net = O.network(self.arrays)
        x, pos = beat_inputs(self.rng, 80), torch.as_tensor(self.rng.integers(0, 3, 80))[None]
        with torch.no_grad():
            logpc, L = net(torch.as_tensor(x)[None], pos)
            for s in (1, 5, 11):
                moved, moved_keys = net(torch.as_tensor(transposed(x, s, 3))[None], pos)
                np.testing.assert_allclose(moved.numpy(), transposed(logpc.numpy(), s, 2), atol=1e-5)
                np.testing.assert_allclose(moved_keys.numpy(), transposed(L.numpy(), s, 2), atol=1e-5)

    def test_loss_is_the_bars_keys_and_the_template_term(self):
        batch = tiny_batch(self.rng, B=3, n=20)
        loss = O.key_loss(O.network(self.arrays), batch, 2.0)
        self.assertAlmostEqual(float(loss.detach()), numpy_loss(self.arrays, batch, 2.0), places=5)
        no_template = O.key_loss(O.network(self.arrays), batch, 0.0)
        self.assertAlmostEqual(float(no_template.detach()), numpy_loss(self.arrays, batch, 0.0), places=5)

    def test_no_keyed_beat_no_loss(self):
        x, y, w, bar, pos, teach = tiny_batch(self.rng)
        self.assertIsNone(O.key_loss(O.network(self.arrays), (x, y, 0 * w, bar, pos, teach), 2.0))

    def test_one_step_is_adam_with_l2_on_every_weight(self):
        net = O.network(self.arrays)
        opt = O.optimizer(net, 1e-3)
        O.key_loss(net, tiny_batch(self.rng), 2.0).backward()
        before = {k: getattr(net, k).detach().clone().numpy() for k in O.TRAINED}
        grads = {k: getattr(net, k).grad.clone().numpy() for k in O.TRAINED}
        opt.step()
        for k in O.TRAINED:
            g = grads[k] + O.WEIGHT_DECAY * before[k]          # the first step: m / (1 - b1) = g, v / (1 - b2) = g^2
            np.testing.assert_allclose(getattr(net, k).detach().numpy(), before[k] - 1e-3 * g / (np.abs(g) + 1e-8),
                                       atol=1e-7, err_msg=k)


class ScheduleTests(unittest.TestCase):
    def test_tuning_decays_from_the_first_step(self):
        steps = 30
        f = np.array([O._rate(s, steps, 0.0) for s in range(1, steps + 1)])
        self.assertEqual(f[0], 1.0)
        np.testing.assert_allclose(f, 0.5 * (1 + np.cos(np.pi * np.arange(steps) / steps)))

    def test_from_scratch_warmed_up_then_decayed(self):
        steps = 100
        f = np.array([O._rate(s, steps, O.SCRATCH_WARMUP) for s in range(1, steps + 1)])
        self.assertTrue(np.all(np.diff(f[:30]) > 0))
        self.assertAlmostEqual(f[29], 1.0)
        self.assertTrue(np.all(np.diff(f[30:]) < 0))
        self.assertLess(f[-1], 0.01)

    def test_steps_are_epochs_of_batches(self):
        self.assertEqual(O.steps_of(33, 15), 15 * 3)
        self.assertEqual(O.steps_of(3, 2), 2)


class TemplateTests(unittest.TestCase):
    def beat(self, bass, notes):
        x = np.zeros((1, 36), np.float32)
        x[0, bass] = 1.0
        x[0, [12 + n for n in notes]] = 1.0
        return x

    def test_triads_named_by_their_templates(self):
        self.assertEqual(int(np.argmax(O.template_chords(self.beat(0, (0, 4, 7))))), 0)          # C major
        self.assertEqual(int(np.argmax(O.template_chords(self.beat(9, (9, 0, 4))))), 12 + 9)     # A minor
        self.assertEqual(int(np.argmax(O.template_chords(self.beat(7, (7, 11, 2))))), 7)         # G major
        p = O.template_chords(np.random.default_rng(RNG_SEED).random((50, 36)))
        np.testing.assert_allclose(p.sum(1), 1.0, rtol=1e-5)

    def test_silent_beat_is_finite(self):
        self.assertTrue(np.all(np.isfinite(O.template_chords(np.zeros((3, 36))))))


def keys(*rows):
    return np.asarray(rows, float)


class AnnotationTests(unittest.TestCase):
    def test_rows_sorted_by_start_and_cut_to_four_columns(self):
        r = O._rows([[2, 0, 50, 90, 7], [0, 1, 0, 50, 7]])
        np.testing.assert_array_equal(r, [[0, 1, 0, 50], [2, 0, 50, 90]])
        self.assertEqual(O._rows([]).shape, (0, 4))

    def test_same_keys_tolerates_float32_times_only(self):
        a = keys([0, 1, 0.0, 50.0], [2, 0, 50.0, 90.0])
        self.assertTrue(O._same_keys(a, a + [0, 0, 5e-4, -5e-4]))
        self.assertFalse(O._same_keys(a, a + [0, 0, 0.01, 0]))
        self.assertFalse(O._same_keys(a, a + [1, 0, 0, 0]))
        self.assertFalse(O._same_keys(a, a[:1]))

    def test_raw_looking(self):
        clean = keys([0, 0, 0, 60], [7, 0, 60, 120], [0, 0, 120, 180])
        self.assertFalse(O.raw_looking(clean))
        many = keys(*[[i % 2, 0, 10 * i, 10 * i + 10] for i in range(10)])
        self.assertTrue(O.raw_looking(many))
        fragments = keys([0, 0, 0, 60], [0, 0, 60, 62], [0, 0, 62, 63], [5, 0, 63, 120])
        self.assertTrue(O.raw_looking(fragments))                # short fragments beside the same key

    def test_the_users_keys_are_the_training_songs(self):
        auto = keys([0, 0, 0, 20])
        cases = {
            "edited": dict(key_segments=keys([9, 1, 0, 20]), key_segments_auto=auto),
            "still_auto": dict(key_segments=auto.copy(), key_segments_auto=auto),
            "legacy": dict(key_segments=keys([2, 0, 0, 10], [9, 1, 10, 20])),
            "raw": dict(key_segments=keys(*[[i % 3, 0, 2 * i, 2 * i + 2] for i in range(10)])),
            "no_keys": dict(key_segments=None),
            "no_tempo": dict(key_segments=keys([0, 0, 0, 20]), tempo_segments=None),
        }
        with tempfile.TemporaryDirectory() as tmp:
            given = []
            for i, (name, values) in enumerate(cases.items()):
                a = F.beat_annotation(duration=20.0)
                fields = {**a.__dict__, "phrase_starts": np.array([0.25, 8.25]), **values}
                given.append(F.given_track(Path(tmp), i, Annotation(**fields), title=name))
            F.use_tracks(given)
            try:
                ignored = []
                listed = O.list_key_tracks(Path(tmp), ignored)
            finally:
                F.library_tracks._tracks = None
        self.assertEqual([(t["title"], t["kind"]) for t in listed], [("edited", "edited"), ("legacy", "legacy")])
        reasons = {s.title: render(s.reason) for s in ignored}
        self.assertEqual(set(reasons), {"no_keys", "no_tempo"})
        self.assertEqual(reasons["no_keys"], "no key annotation")
        self.assertIn("no tempo rows", reasons["no_tempo"])
        np.testing.assert_array_equal(listed[0]["phrase_starts"], [0.25, 8.25])


class PositionTableTests(unittest.TestCase):
    """position_tables on songs whose chords the CNN reads as given: the tonic inside phrases, the dominant on the
    bars a phrase ends on."""

    def song(self, key, n=48):
        pos = np.tile(np.r_[np.zeros(4), np.full(8, 2), np.ones(4)], n // 16).astype(np.int64)  # first, inside, last
        y = np.full(n, key, np.int64)
        return O.Song(x=np.zeros((n, 36), np.float32), y=y, w=np.ones(n, np.float32), bar=np.arange(n) // 4, pos=pos,
                      teach=np.zeros((n, 24), np.float32), grid=None, duration=float(n))

    def chords_of(self, songs):
        """The chords as read: the key's tonic chord, its dominant on the phrases' last bars."""
        def chords(_model, x):
            s = next(s for s in songs if s.x is x)
            p = np.full((x.shape[0], 24), 1e-3)
            tonic = s.y % 12
            p[np.arange(x.shape[0]), tonic] = 1.0
            last = s.pos == 1
            p[last, tonic[last]] = 0.2
            p[np.flatnonzero(last), (tonic[last] + 7) % 12] = 1.0
            return np.log(p / p.sum(1, keepdims=True))
        return chords

    def test_the_last_bars_lean_to_the_dominant_in_every_key(self):
        songs = [self.song(0), self.song(7)]                    # C major and G major: the same chords relative to them
        arrays = CK.read_model_arrays(SHIPPED_KEY)
        with mock.patch.object(CK.ChordKeyModel, "chords", self.chords_of(songs)):
            tab = O.position_tables(arrays, songs, [0, 1])
        base = np.asarray(arrays["tab"], np.float32)
        np.testing.assert_array_equal(tab[:, 2], base[:, 2])          # the bars inside a phrase: the start's
        np.testing.assert_array_equal(tab[1], base[1])               # minor: no keyed beat, its rows kept
        lift = (tab[0, 1] - base[0, 2])
        self.assertGreater(lift[7] - lift[0], 3.0)                   # the dominant up against the tonic on last bars
        np.testing.assert_allclose(tab[0, 0], base[0, 2], atol=1e-5)  # the first bars read as the inside ones


def track_for(beats, rows, duration, ks, phrase_starts=(0.0,)):
    return {"beats": beats, "segments": rows, "phrase_starts": np.asarray(phrase_starts, float),
            "duration": duration, "keys": ks}


def key_frames(duration, rng):
    return rng.random((36, int(duration * CK.SR / CK.HOP) + 1)).astype(np.float32)


class SongTests(unittest.TestCase):
    def test_beats_labelled_by_the_key_they_fall_in(self):
        rng = np.random.default_rng(RNG_SEED)
        duration = 20.0
        beats = F.steady_beats(duration, 120.0)
        ks = keys([2, 0, 0.0, 10.0], [9, 1, 10.0, 15.0])            # D major, then A minor, then no key
        s = O.song_of(track_for(beats, F.tempo_rows(beats, duration, 120.0), duration, ks), key_frames(duration, rng))
        self.assertEqual(s.x.shape, (beats.size, 36))
        first, second, rest = beats < 10.0, (beats >= 10.0) & (beats < 15.0), beats >= 15.0
        self.assertTrue(np.all(s.y[first] == 2) and np.all(s.y[second] == 12 + 9))
        np.testing.assert_array_equal(s.w, (beats < 15.0).astype(np.float32))
        self.assertTrue(np.all(s.w[rest] == 0))
        np.testing.assert_array_equal(s.bar, np.arange(beats.size) // 4)
        self.assertEqual(s.teach.shape, (beats.size, 24))

    def test_too_few_beats_is_no_song(self):
        beats = np.arange(10) * 0.5
        self.assertIsNone(O.song_of(track_for(beats, F.tempo_rows(beats, 6.0, 120.0), 6.0, keys([0, 0, 0, 6])),
                                    key_frames(6.0, np.random.default_rng(0))))

    def test_crops_padded_with_unkeyed_beats(self):
        rng = np.random.default_rng(RNG_SEED)
        duration = 30.0
        beats = F.steady_beats(duration, 120.0)
        s = O.song_of(track_for(beats, F.tempo_rows(beats, duration, 120.0), duration, keys([0, 0, 0, 30])),
                      key_frames(duration, rng))
        x, y, w, bar, pos, teach = O._batch([s, s], [0, 1], rng)
        n = beats.size
        self.assertEqual(x.shape, (2, O.CROP, 36))
        self.assertTrue(np.all(w[:, :n] == 1) and np.all(w[:, n:] == 0))
        self.assertTrue(np.all(pos[:, n:] == 2))
        self.assertTrue(np.all(bar[:, n:] == bar[:, n - 1:n] + 1))     # the padding is a bar of its own
        self.assertTrue(np.all(teach[:, n:] == 0))

    def test_new_weights_within_their_fan_in_bounds(self):
        arrays = CK.read_model_arrays(SHIPPED_KEY)
        P = O._init_weights(arrays, 0)
        self.assertEqual(set(P), set(O.TRAINED))
        for k in O.TRAINED:
            self.assertEqual(P[k].shape, arrays[k].shape)
            fan_in = int(np.prod(arrays[k.replace("_b", "_w")].shape[1:]))
            self.assertLessEqual(float(np.abs(P[k]).max()), 1 / np.sqrt(fan_in))


SMALL = 4


class _Library(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.cache.mkdir()
        self.out = self.dir / "weights" / "key_chord_cnn.npz"
        self.enterContext(mock.patch.object(O, "spawn_pool", F.no_pool))

    def tearDown(self):
        F.library_tracks._tracks = None
        self._tmp.cleanup()

    def tracks(self, n, frames=True):
        rng = np.random.default_rng(RNG_SEED)
        tracks = []
        for i in range(n):
            duration = 24.0
            a = F.beat_annotation(duration=duration, bpm=120.0)
            a = Annotation(**{**a.__dict__, "phrase_starts": np.array([0.25]),
                              "key_segments": keys([i % 12, 0, 0, 12], [(i + 7) % 12, 1, 12, duration]),
                              "key_segments_auto": keys([0, 0, 0, duration])})
            t = F.given_track(self.dir, i, a)
            tracks.append(t)
            if frames:
                np.savez_compressed(self.cache / f"{t['uid']}.npz", cache_format=np.asarray(O.KEY_FRAME_CACHE_FORMAT),
                                    audio_mtime_ns=np.int64(t["audio_path"].stat().st_mtime_ns),
                                    frames=key_frames(duration, rng).astype(np.float16))
        F.use_tracks(tracks)
        return tracks

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, start=MS.SHIPPED,
                      current_path=SHIPPED_KEY, start_path=SHIPPED_KEY, epochs=1, device="cpu")
        values.update(kw)
        return O.KeyModelOptimizationRequest(**values)

    def small(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"fine_tune": (SMALL, SMALL), "scratch": (SMALL, SMALL)}))
        return self.tracks(SMALL)


class RefusalTests(_Library):
    """Refused before any training (no torch needed)."""

    def test_too_few_checked_songs_refused(self):
        self.tracks(MS.MIN_TRACKS["fine_tune"][0] - 1, frames=False)
        with mock.patch.object(O, "ensure_key_frames", side_effect=AssertionError("no features for a refused run")):
            with self.assertRaises(TrainingError) as caught:
                O.optimize_key_model(self.request())
        self.assertIn("songs with checked key annotations", str(caught.exception))
        self.assertFalse(self.out.exists())

    def test_songs_without_key_frames_count_against_the_minimum(self):
        tracks = self.small()
        (self.cache / f"{tracks[0]['uid']}.npz").unlink()
        with mock.patch.object(O, "ensure_key_frames", lambda tracks, *a, **k: tracks):
            with self.assertRaises(TrainingError):
                O.optimize_key_model(self.request())
        self.assertFalse(self.out.exists())

    def test_stale_key_frames_are_rebuilt(self):
        tracks = self.small()
        path = self.cache / f"{tracks[0]['uid']}.npz"
        self.assertIsNotNone(O.read_key_frames(path, tracks[0]))
        tracks[0]["audio_path"].write_bytes(b"RIFF and more")       # the audio changed since
        os.utime(tracks[0]["audio_path"], ns=(1, 1))
        self.assertIsNone(O.read_key_frames(path, tracks[0]))
        path.write_bytes(b"corrupt")
        self.assertIsNone(O.read_key_frames(path, tracks[0]))

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_key_model(self.request(output_path=self.dir / "key.json"))


@needs_torch
class RunTests(_Library):
    def test_retrained_model_written_and_read_by_the_analyzers_loader(self):
        self.small()
        progress = []
        result = O.optimize_key_model(self.request(), optimize_progress=lambda v, t: progress.append(v))
        self.assertTrue(result.saved)
        self.assertEqual((result.track_count, result.edited_count, result.legacy_count), (SMALL, SMALL, 0))
        self.assertEqual((result.steps, result.device), (1, "cpu"))
        self.assertTrue(np.isfinite(result.seconds_per_step))
        arrays, shipped = CK.read_model_arrays(self.out), CK.read_model_arrays(SHIPPED_KEY)
        with np.load(self.out, allow_pickle=False) as z:
            self.assertEqual(set(z.files), set(CK.WEIGHTS + CK.SETTINGS))
        for k in CK.SETTINGS + ("tab",):                     # the decoder settings and the chord tables' kind kept
            self.assertEqual((arrays[k].shape, arrays[k].dtype), (shipped[k].shape, shipped[k].dtype))
        for k in CK.SETTINGS:
            np.testing.assert_array_equal(arrays[k], shipped[k])
        np.testing.assert_array_equal(arrays["tab"][:, 2], shipped["tab"][:, 2])    # the bars inside a phrase: kept
        self.assertFalse(np.array_equal(arrays["c1_w"], shipped["c1_w"]))
        model = CK.ChordKeyModel(self.out)
        self.assertTrue(np.all(np.isfinite(model.chords(np.random.default_rng(0).random((20, 36), np.float32)))))
        self.assertTrue(0.0 <= result.accuracy_before <= 1.0 and 0.0 <= result.accuracy_after <= 1.0)
        self.assertEqual(progress[-1], 100)

    def test_progress_texts_are_marked_for_translation(self):
        self.small()
        texts = []
        O.optimize_key_model(self.request(), feature_progress=lambda v, t: texts.append(t),
                             optimize_progress=lambda v, t: texts.append(t))
        self.assertEqual([w for t in texts for w in F.plain_words(t)], [])

    def test_from_scratch(self):
        self.small()
        result = O.optimize_key_model(self.request(start=MS.SCRATCH, start_path=None))
        self.assertTrue(result.saved and self.out.is_file())
        self.assertEqual(result.epochs, 1)
        new = O._init_weights(CK.read_model_arrays(SHIPPED_KEY), 0)
        arrays = CK.read_model_arrays(self.out)
        self.assertLess(float(np.abs(arrays["c1_w"] - new["c1_w"]).max()), 0.05)     # one step from the new weights

    def test_worse_held_out_keys_leave_the_model_unwritten(self):
        self.small()
        with mock.patch.object(O, "evaluate", side_effect=[(0.9, 0.0), (0.5, 0.1)]):
            result = O.optimize_key_model(self.request(cv_folds=2))
        self.assertFalse(result.saved or self.out.exists())
        self.assertTrue(result.cross_validated)
        self.assertEqual((result.accuracy_before, result.accuracy_after, result.eval_songs), (0.9, 0.5, SMALL // 2))

    def test_an_export_unlike_the_trained_net_is_not_written(self):
        self.small()
        with mock.patch.object(O, "check_export", return_value=1.0):
            with self.assertRaises(RuntimeError):
                O.optimize_key_model(self.request())
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
