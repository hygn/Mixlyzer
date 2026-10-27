"""analyzer.training.downbeat_ctx_optimizer: the downbeat model's samples labelled by the user's bars, and small runs of
the optimizer on synthetic frame caches: the model written where asked and read by the tracker's own loader, songs
with fewer than two downbeats left out, the refusals, and the held-out check that keeps a worse model unwritten."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from analyzer.beat.general.bars import load_ctx_model
from analyzer.training import beat_frames, model_start as MS
from analyzer.training import downbeat_ctx_optimizer as O
from analyzer.training.beat_grid_check import GridCheck
from analyzer.training.library_tracks import Annotation
from analyzer.training.text import TrainingError, render

from unit.analyzer_tests import training_fixtures as F

SMALL = 5


def small_model(seed):
    return HistGradientBoostingClassifier(max_iter=5, max_leaf_nodes=7, min_samples_leaf=3, random_state=seed)


class SampleTests(unittest.TestCase):
    def test_labels_are_the_users_downbeats(self):
        for ts in (3, 4):
            with self.subTest(ts=ts):
                beats = F.steady_beats(12.0, 120.0)
                frames = F.synthetic_frames(int(12.0 / F.FRAME_SEC), beats, ts)
                x, y = O.track_samples(frames, beats, F.tempo_rows(beats, 12.0, 120.0, ts=ts))
                self.assertEqual(x.shape[0], y.size)
                self.assertEqual(x.dtype, np.float32)
                np.testing.assert_array_equal(np.flatnonzero(y), np.arange(0, y.size, ts))


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "frames"
        self.out = self.dir / "weights" / "downbeat_ctx_hgb.npz"
        self.enterContext(mock.patch.object(O, "_model", small_model))
        self.enterContext(mock.patch.object(beat_frames, "spawn_pool", F.no_pool))

    def tearDown(self):
        F.library_tracks._tracks = None
        self._tmp.cleanup()

    def tracks(self, n, frames=True, extra=()):
        tracks = []
        for i in range(n):
            a = F.beat_annotation(duration=10.0, bpm=120.0 + 3 * i)
            tracks.append(F.given_track(self.dir, i, a))
        for j, a in enumerate(extra):
            tracks.append(F.given_track(self.dir, n + j, a, title="extra"))
        for t in tracks:
            if frames:
                a = t["annotation"]
                F.write_beat_frames(self.cache, t, F.synthetic_frames(int(10.0 / F.FRAME_SEC), a.beats, 1))
        F.use_tracks(tracks)
        return tracks

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, max_cache_workers=1)
        values.update(kw)
        return O.DownbeatCtxOptimizationRequest(**values)

    def test_model_written_and_read_by_the_trackers_loader(self):
        beats = F.steady_beats(10.0, 120.0)
        one_bar = Annotation(beats=beats, sample_rate=F.SR, duration_sec=10.0,
                             tempo_segments=np.array([[0.0, 1.0, 120.0, beats[0], 4.0, 120.0]]))
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"gbm": (SMALL, SMALL)}))
        self.tracks(SMALL, extra=[one_bar])
        result = O.optimize_downbeat_ctx(self.request())
        self.assertTrue(result.saved)
        self.assertEqual(result.track_count, SMALL)
        self.assertEqual([render(s.reason) for s in result.skipped_tracks], ["fewer than two downbeats"])
        model = load_ctx_model(str(self.out))
        frames = F.synthetic_frames(int(10.0 / F.FRAME_SEC), beats, 0)
        x, _ = O.track_samples(frames, beats, F.tempo_rows(beats, 10.0, 120.0))
        p = model.predict_proba(x.astype(np.float64))[:, 1]
        self.assertEqual(p.shape, (x.shape[0],))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_worse_held_out_grids_leave_the_model_unwritten(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"gbm": (SMALL, SMALL)}))
        self.tracks(SMALL)
        worse = GridCheck(songs=2, beats_before_sec=0.0, beats_after_sec=0.0, bars_before_sec=0.0, bars_after_sec=4.0)
        check = mock.Mock(return_value=worse)
        with mock.patch.object(O, "grid_check", check):
            result = O.optimize_downbeat_ctx(self.request(cv_folds=2))
        self.assertFalse(result.saved or self.out.exists())
        self.assertFalse((self.cache / "_downbeat_ctx_held_out.npz").exists())      # the trial model removed
        self.assertTrue(check.call_args.args[4].downbeat_ctx.endswith("_downbeat_ctx_held_out.npz"))

    # Bug: the samples' progress ("Samples 3/5") is a plain f-string, not marked with text.msg: shown untranslated.
    @unittest.expectedFailure
    def test_progress_texts_are_marked_for_translation(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"gbm": (SMALL, SMALL)}))
        self.tracks(SMALL)
        texts = []
        O.optimize_downbeat_ctx(self.request(), feature_progress=lambda v, t: texts.append(t),
                                optimize_progress=lambda v, t: texts.append(t))
        self.assertEqual([w for t in texts for w in F.plain_words(t)], [])

    def test_too_few_songs_refused(self):
        self.tracks(MS.MIN_TRACKS["gbm"][0] - 1, frames=False)
        with mock.patch.object(O, "ensure_frames", side_effect=AssertionError("no frames for a refused run")):
            with self.assertRaises(TrainingError):
                O.optimize_downbeat_ctx(self.request())

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_downbeat_ctx(self.request(output_path=self.dir / "ctx.json"))


if __name__ == "__main__":
    unittest.main()
