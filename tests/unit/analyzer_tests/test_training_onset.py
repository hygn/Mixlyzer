"""analyzer.training.onset_training and refine_onset_optimizer: the beat targets and frame samples of the onset models,
the tracks with a usable beat grid (and why the others are left out), the onset channel cache's validity checks,
the songs grouped for cross-validation, and small runs of the tempo-change onset optimizer on synthetic caches: the
model written where asked and read by the analyzer's own loader, the refusals, and the held-out check."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from analyzer.beat.learned_onset import (ONSET_CHANNEL_NAMES, ONSET_FEATURE_NAMES, load_refine_onset_model,
                                         onset_channels, refine_onset_activation)
from analyzer.training import beat_frames, model_start as MS
from analyzer.training import onset_training as OT
from analyzer.training import refine_onset_optimizer as O
from analyzer.training.beat_grid_check import GridCheck
from analyzer.training.library_tracks import Annotation
from analyzer.training.text import TrainingError, render

from unit.analyzer_tests import training_fixtures as F

RNG_SEED = 20261027


class TargetTests(unittest.TestCase):
    FT = np.arange(400) * 0.01
    BEATS = np.arange(0.5, 3.51, 0.5)

    def test_frames_inside_the_beats_and_their_gaussian_targets(self):
        inside, target = OT.beat_targets(self.FT, self.BEATS)
        self.assertEqual((self.FT[inside[0]], self.FT[inside[-1]]), (0.25, 3.75))       # half a beat either side
        self.assertTrue(np.all(np.diff(inside) == 1))
        on_beat = np.isin(np.round(self.FT[inside], 6), np.round(self.BEATS, 6))
        np.testing.assert_allclose(target[on_beat], 1.0)
        self.assertTrue(np.all(target[~on_beat] < 1.0))
        one_off = np.isin(np.round(self.FT[inside], 6), np.round(self.BEATS + 0.01, 6))
        np.testing.assert_allclose(target[one_off], np.exp(-0.5 / OT.TARGET_SIGMA_FRAMES ** 2))

    def test_subdivision_frames(self):
        frames = OT._subdivision_frames(self.FT, self.BEATS)
        expected = {int(round(100 * (a + p * 0.5))) for a in self.BEATS[:-1] for p in OT.SUBDIVISION_POSITIONS}
        self.assertEqual(set(frames.tolist()), expected)

    def test_samples_of_each_kind_weighted(self):
        x = np.random.default_rng(RNG_SEED).random((self.FT.size, 3)).astype(np.float32)
        s = OT.samples_from_inputs(x, self.FT, self.BEATS)
        self.assertEqual(s["x"].shape[0], s["y"].size)
        self.assertEqual(set(np.unique(s["kind"]).tolist()), {0, 1, 2})
        np.testing.assert_array_equal(s["w"], np.select([s["kind"] == 0, s["kind"] == 1],
                                                        [1.0, OT.SUBDIVISION_WEIGHT], OT.NEGATIVE_STRIDE))
        self.assertTrue(np.all(s["y"][s["kind"] != 0] == 0.0))
        self.assertTrue(np.all(s["y"][s["kind"] == 0] >= OT.TARGET_MIN))
        self.assertEqual(int(np.sum(s["y"] == 1.0)), self.BEATS.size)

    def test_cache_workers(self):
        self.assertEqual(OT._auto_cache_workers(None, 0), 1)
        self.assertEqual(OT._auto_cache_workers(8, 3), 3)
        self.assertEqual(OT._auto_cache_workers(0, 3), 1)
        self.assertLessEqual(OT._auto_cache_workers(None, 100), 4)


class BeatTrackTests(unittest.TestCase):
    def test_tracks_without_a_usable_grid_left_out_with_their_reason(self):
        good = F.beat_annotation()
        cases = [("good", good, True, {}),
                 ("unreadable", None, True, {"error": "bad npz"}),
                 ("no_analysis", None, True, {}),
                 ("no_audio", good, False, {}),
                 ("no_beats", Annotation(sample_rate=22050), True, {}),
                 ("no_rate", Annotation(beats=good.beats), True, {}),
                 ("few_beats", Annotation(beats=good.beats[:OT.MIN_BEATS - 1], sample_rate=22050), True, {}),
                 ("nan_beats", Annotation(beats=np.r_[good.beats, np.nan], sample_rate=22050), True, {}),
                 ("unsorted", Annotation(beats=good.beats[::-1].copy(), sample_rate=22050), True, {})]
        with tempfile.TemporaryDirectory() as tmp:
            F.use_tracks([F.given_track(Path(tmp), i, a, audio=audio, title=name, **extra)
                          for i, (name, a, audio, extra) in enumerate(cases)])
            try:
                ignored = []
                listed = OT._list_beat_tracks(Path(tmp), ignored)
            finally:
                F.library_tracks._tracks = None
        self.assertEqual([t["title"] for t in listed], ["good", "unsorted"])
        self.assertTrue(np.all(np.diff(listed[1]["beats"]) > 0))           # sorted for the training
        reasons = {s.title: render(s.reason) for s in ignored}
        self.assertEqual(reasons, {
            "unreadable": "invalid beat analysis: bad npz", "no_analysis": "analysis NPZ is missing",
            "no_audio": "audio file is missing", "no_beats": "invalid beat analysis: beats",
            "no_rate": "invalid beat analysis: sample rate", "few_beats": "invalid or insufficient beat grid",
            "nan_beats": "invalid or insufficient beat grid"})


def write_onset_cache(cache_dir: Path, track: dict, times, chans, **override) -> Path:
    values = dict(cache_format=np.asarray(OT.ONSET_FEATURE_CACHE_FORMAT), frame_times=np.asarray(times, float),
                  channels=chans, channel_names=np.asarray(ONSET_CHANNEL_NAMES, dtype="U64"),
                  use_hpss=np.bool_(True), frame_hop_length=np.int64(F.HOP), sample_rate=np.int64(track["sample_rate"]),
                  audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns))
    values.update(override)
    path = OT._cache_path(Path(cache_dir), str(track["uid"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return path


class OnsetCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.track = F.given_track(self.dir, 0, None, sample_rate=22050)
        self.ft = np.arange(50) * F.FRAME_SEC
        self.ch = np.zeros((50, len(ONSET_CHANNEL_NAMES)), np.float32)

    def tearDown(self):
        self._tmp.cleanup()

    def read(self, **override):
        path = write_onset_cache(self.dir, self.track, self.ft, self.ch, **override)
        return OT._read_current_cache(path, self.track, True, F.HOP)

    def test_current_cache_read(self):
        got = self.read()
        np.testing.assert_array_equal(got[0], self.ft)
        self.assertEqual(got[1].shape, self.ch.shape)

    def test_stale_or_foreign_caches_are_misses(self):
        bad = {"format": dict(cache_format=np.asarray("v0")),
               "names": dict(channel_names=np.asarray(["x"] * len(ONSET_CHANNEL_NAMES))),
               "hpss": dict(use_hpss=np.bool_(False)), "hop": dict(frame_hop_length=np.int64(512)),
               "rate": dict(sample_rate=np.int64(44100)), "mtime": dict(audio_mtime_ns=np.int64(1)),
               "dtype": dict(channels=self.ch.astype(np.float64)),
               "shape": dict(channels=self.ch[:, :3]),
               "nan": dict(channels=np.full_like(self.ch, np.nan)),
               "extra_key": dict(extra=np.zeros(1))}
        for name, override in bad.items():
            with self.subTest(name=name):
                self.assertIsNone(self.read(**override))
        OT._cache_path(self.dir, self.track["uid"]).write_bytes(b"PK\x03\x04 truncated")
        self.assertIsNone(OT._read_current_cache(OT._cache_path(self.dir, self.track["uid"]), self.track, True, F.HOP))
        self.assertIsNone(OT._read_current_cache(self.dir / "missing.npz", self.track, True, F.HOP))


class GroupTests(unittest.TestCase):
    def test_spellings_of_one_song_are_one_group(self):
        a = O._group_key({"uid": "u1", "artist": "The Band", "title": "Song A!"})
        b = O._group_key({"uid": "u2", "artist": "the band", "title": "song-a"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, O._group_key({"uid": "u3", "artist": "The Band", "title": "Song B"}))
        self.assertEqual(O._group_key({"uid": "u4", "artist": "", "title": "?!"}), "u4")
        self.assertEqual(O._group_key({"uid": "u5", "artist": "アーティスト", "title": "曲"}), "アーティスト曲")

    def test_samples_kept_and_reweighted_by_kind(self):
        rng = np.random.default_rng(RNG_SEED)
        ft = np.arange(2000) * F.FRAME_SEC
        beats = F.steady_beats(ft[-1], 120.0)
        ch = rng.random((ft.size, len(ONSET_CHANNEL_NAMES))).astype(np.float32)
        x, y, w = O._track_samples(ft, ch, beats, 3)
        x2, y2, w2 = O._track_samples(ft, ch, beats, 3)
        np.testing.assert_array_equal(x, x2)
        self.assertEqual(x.shape[1], len(ONSET_FEATURE_NAMES))
        self.assertEqual(x.dtype, np.float16)
        self.assertEqual(set(np.unique(y).tolist()), {0, 1})
        everything = OT.samples_from_inputs(np.zeros((ft.size, 1), np.float32), ft, beats)
        # the kept samples stand for all of them: their weights sum to about all samples' weight
        self.assertAlmostEqual(float(w.sum()) / float(everything["w"].sum()), 1.0, delta=0.15)


def small_model(seed):
    return HistGradientBoostingClassifier(max_iter=5, max_leaf_nodes=7, min_samples_leaf=5, random_state=seed)


SMALL = 5


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "onset"
        self.out = self.dir / "weights" / "refine_onset_hgb.npz"
        self.enterContext(mock.patch.object(O, "spawn_pool", F.no_pool))
        self.enterContext(mock.patch.object(O, "_model", small_model))

    def tearDown(self):
        F.library_tracks._tracks = None
        load_refine_onset_model.cache_clear()
        self._tmp.cleanup()

    def tracks(self, n, caches=True):
        tracks, self.frames = [], []
        for i in range(n):
            a = F.beat_annotation(duration=12.0, bpm=118.0 + 2 * i)
            t = F.given_track(self.dir, i, a)
            tracks.append(t)
            frames = F.synthetic_frames(int(12.0 / F.FRAME_SEC), a.beats, i)
            self.frames.append(frames)
            if caches:
                write_onset_cache(self.cache, {**t, "sample_rate": F.SR}, frames.frame_times, onset_channels(frames))
        F.use_tracks(tracks)
        return tracks

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, max_cache_workers=1)
        values.update(kw)
        return O.RefineOnsetOptimizationRequest(**values)

    def test_model_written_and_read_by_the_analyzers_loader(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"gbm": (SMALL, SMALL)}))
        self.tracks(SMALL)
        result = O.optimize_refine_onset(self.request())
        self.assertTrue(result.saved)
        self.assertEqual(result.track_count, SMALL)
        self.assertGreater(result.sample_count, 0)
        self.assertTrue(0.0 <= result.beat_average_precision <= 1.0)
        model, hop = load_refine_onset_model(str(self.out))
        self.assertEqual(hop, F.HOP)
        activation = refine_onset_activation(self.frames[0], self.out)
        self.assertEqual(activation.shape, (self.frames[0].n_frames,))
        self.assertTrue(np.all((activation >= 0) & (activation <= 1)))

    def test_progress_texts_are_marked_for_translation(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"gbm": (SMALL, SMALL)}))
        self.tracks(SMALL)
        texts = []
        O.optimize_refine_onset(self.request(), feature_progress=lambda v, t: texts.append(t),
                                optimize_progress=lambda v, t: texts.append(t))
        self.assertEqual([w for t in texts for w in F.plain_words(t)], [])

    def test_too_few_songs_refused_before_any_cache(self):
        self.tracks(MS.MIN_TRACKS["gbm"][0] - 1, caches=False)
        with self.assertRaises(TrainingError) as caught:
            O.optimize_refine_onset(self.request())
        self.assertIn(str(MS.MIN_TRACKS["gbm"][0]), str(caught.exception))
        self.assertFalse(self.out.exists())

    def test_worse_held_out_grids_leave_the_model_unwritten(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"gbm": (SMALL, SMALL)}))
        self.tracks(SMALL)
        worse = GridCheck(songs=2, beats_before_sec=0.0, beats_after_sec=2.0, bars_before_sec=0.0, bars_after_sec=0.0)
        check = mock.Mock(return_value=worse)
        with mock.patch.object(O, "grid_check", check), \
                mock.patch.object(beat_frames, "ensure_frames", lambda tracks, *a, **k: tracks):
            result = O.optimize_refine_onset(self.request(cv_folds=2, frames_cache_dir=self.dir / "frames"))
        self.assertFalse(result.saved or self.out.exists())
        self.assertIn(len(check.call_args.args[0]), (2, 3))                       # the held-out songs, checked
        trial = check.call_args.args[4].refine_onset
        self.assertFalse(os.path.exists(trial))                                   # the trial model removed

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_refine_onset(self.request(output_path=self.dir / "onset.json"))


if __name__ == "__main__":
    unittest.main()
