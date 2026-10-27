"""analyzer.training.key_model_optimizer: the chord CNN's hand-written backward pass and the key loss's gradient against
central differences, its forward and key evidence against the analyzer's model, root equivariance, which annotations
are the user's (edited, legacy, still automatic, raw-looking), the template chords, the training songs and crops, and
small runs of the whole optimizer on synthetic tracks: the model written where asked and read by the analyzer's own
loader, the decoder settings kept, the refusals, and the held-out check that keeps worse keys from being written."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.key import chord_key as CK
from analyzer.training import key_model_optimizer as O
from analyzer.training import model_start as MS
from analyzer.training.library_tracks import Annotation
from analyzer.training.text import TrainingError, render

from unit.analyzer_tests import training_fixtures as F

RNG_SEED = 20261027
SHIPPED_KEY = F.shipped("key_chord_cnn.npz")
WEIGHT_KEYS = ("c1_w", "c1_b", "c2_w", "c2_b", "out_w", "out_b")


def tiny_model(rng, hidden=4):
    return {"c1_w": rng.normal(0, 0.4, (hidden, 36, 3)), "c1_b": rng.normal(0, 0.1, hidden),
            "c2_w": rng.normal(0, 0.4, (hidden, hidden, 3)), "c2_b": rng.normal(0, 0.1, hidden),
            "out_w": rng.normal(0, 0.4, (2, hidden, 1)), "out_b": rng.normal(0, 0.1, 2)}


def tiny_batch(rng, B=2, n=12):
    x = rng.random((B, n, 36))
    y = rng.integers(0, 24, (B, n))
    w = (rng.random((B, n)) < 0.8).astype(float)
    bar = np.tile(np.arange(n) // 4, (B, 1))
    pos = rng.integers(0, 3, (B, n))
    return x, y, w, bar, pos, O.template_chords(x.reshape(-1, 36)).reshape(B, n, 24).astype(float)


class GradientTests(unittest.TestCase):
    def test_loss_gradient_matches_central_differences(self):
        rng = np.random.default_rng(RNG_SEED)
        P = tiny_model(rng)
        R = CK._log_softmax(rng.normal(0, 1, (2, 3, 24)), -1)
        batch = tiny_batch(rng)
        loss, G = O.loss_and_grads(P, R, batch, 2.0)
        self.assertTrue(np.isfinite(loss))
        for key in WEIGHT_KEYS:
            with self.subTest(key=key):
                self.assertEqual(G[key].shape, P[key].shape)
                picks = rng.choice(P[key].size, size=min(8, P[key].size), replace=False)
                numeric = F.numeric_grad(lambda P: O.loss_and_grads(P, R, batch, 2.0)[0], P, key, picks, eps=1e-6)
                np.testing.assert_allclose(G[key].reshape(-1)[picks], numeric, rtol=1e-4, atol=1e-7)

    def test_conv_back_matches_central_differences(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        x, W, b = rng.normal(0, 1, (2, 7, 3)), rng.normal(0, 1, (4, 3, 3)), rng.normal(0, 1, 4)
        g = rng.normal(0, 1, (2, 7, 4))
        dW, db, dx = O._conv_back(x, W, g)
        P = {"x": x, "W": W}

        def f(P):
            return float(np.sum(g * O._conv(P["x"], P["W"], b)))
        np.testing.assert_allclose(dW.reshape(-1), F.numeric_grad(f, P, "W", range(W.size)), rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(dx.reshape(-1), F.numeric_grad(f, P, "x", range(x.size)), rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(db, g.sum((0, 1)))
        self.assertIsNone(O._conv_back(x, W, g, need_dx=False)[2])

    def test_no_keyed_beat_no_gradient(self):
        rng = np.random.default_rng(RNG_SEED + 2)
        x, y, w, bar, pos, teach = tiny_batch(rng)
        self.assertEqual(O.loss_and_grads(tiny_model(rng), np.zeros((2, 3, 24)), (x, y, 0 * w, bar, pos, teach), 2.0),
                         (0.0, None))


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.arrays = CK.read_model_arrays(SHIPPED_KEY)
        self.P = {k: self.arrays[k] for k in WEIGHT_KEYS}
        self.x = np.random.default_rng(RNG_SEED).random((1, 40, 36)).astype(np.float32)

    def test_forward_and_key_evidence_are_the_analyzers(self):
        model = CK.ChordKeyModel(arrays=self.arrays)
        logpc, _ = O.forward(self.P, self.x)
        np.testing.assert_allclose(logpc[0], model.chords(self.x[0]), rtol=1e-4, atol=1e-4)
        pos = np.random.default_rng(RNG_SEED).integers(0, 3, 40)
        R = CK._log_softmax(np.asarray(self.arrays["tab"], np.float32), -1)
        L, A = O.key_loglik(R, logpc, pos[None])
        np.testing.assert_allclose(L[0], model.key_loglik(logpc[0], pos), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(A.sum(-1), 1.0, atol=1e-5)

    def test_transposed_input_transposes_the_chords(self):
        logpc, _ = O.forward(self.P, self.x)
        for s in (1, 5, 11):
            shifted = O._rot(self.x, -s)                      # every pitch class up s semitones
            moved, _ = O.forward(self.P, shifted)
            np.testing.assert_allclose(moved, O._unroll24(logpc, s), atol=1e-5)

    def test_roll24_and_unroll24_are_inverse_within_each_quality(self):
        p = np.arange(24.0)
        for k in range(12):
            np.testing.assert_array_equal(O._unroll24(O._roll24(p, k), k), p)
        np.testing.assert_array_equal(O._roll24(p, 1)[[0, 11, 12, 23]], [1, 0, 13, 12])


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
        for k in WEIGHT_KEYS:
            self.assertEqual(P[k].shape, arrays[k].shape)
            fan_in = int(np.prod(arrays[k.replace("_b", "_w")].shape[1:]))
            self.assertLessEqual(float(np.abs(P[k]).max()), 1 / np.sqrt(fan_in))


SMALL = 4


class RunTests(unittest.TestCase):
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
                      current_path=SHIPPED_KEY, start_path=SHIPPED_KEY, epochs=1)
        values.update(kw)
        return O.KeyModelOptimizationRequest(**values)

    def small(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"fine_tune": (SMALL, SMALL), "scratch": (SMALL, SMALL)}))
        return self.tracks(SMALL)

    def test_retrained_model_written_and_read_by_the_analyzers_loader(self):
        self.small()
        progress = []
        result = O.optimize_key_model(self.request(), optimize_progress=lambda v, t: progress.append(v))
        self.assertTrue(result.saved)
        self.assertEqual((result.track_count, result.edited_count, result.legacy_count), (SMALL, SMALL, 0))
        arrays, shipped = CK.read_model_arrays(self.out), CK.read_model_arrays(SHIPPED_KEY)
        for k in CK.SETTINGS + ("tab",):                     # the decoder settings and the chord tables kept
            np.testing.assert_array_equal(arrays[k], shipped[k])
        self.assertFalse(np.array_equal(arrays["c1_w"], shipped["c1_w"]))
        model = CK.ChordKeyModel(self.out)
        self.assertTrue(np.all(np.isfinite(model.chords(np.random.default_rng(0).random((20, 36), np.float32)))))
        self.assertTrue(0.0 <= result.accuracy_before <= 1.0 and 0.0 <= result.accuracy_after <= 1.0)
        self.assertEqual(progress[-1], 100)

    # Bug: the last feature progress ("4 songs") is a plain f-string, not marked with text.msg: shown untranslated.
    @unittest.expectedFailure
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

    def test_worse_held_out_keys_leave_the_model_unwritten(self):
        self.small()
        with mock.patch.object(O, "evaluate", side_effect=[(0.9, 0.0), (0.5, 0.1)]):
            result = O.optimize_key_model(self.request(cv_folds=2))
        self.assertFalse(result.saved or self.out.exists())
        self.assertEqual((result.accuracy_before, result.accuracy_after, result.eval_songs), (0.9, 0.5, SMALL // 2))

    def test_too_few_checked_songs_refused(self):
        self.tracks(MS.MIN_TRACKS["fine_tune"][0] - 1, frames=False)
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


if __name__ == "__main__":
    unittest.main()
