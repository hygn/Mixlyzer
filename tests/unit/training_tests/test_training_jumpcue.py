"""training.jumpcue_optimizer on synthetic tracks (no audio: their caches written as the trainer writes them): the
model written in the app's format and read by its runtime, the held-out measures, the labels (reviewed, known
negatives), the tracks left out and the refusals."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.files import atomic_save_npz
from analyzer.jumpcue import pairs as P
from tests.unit.analyzer_tests.test_jumpcue_pairs import synthetic
from tests.unit.training_tests import training_fixtures as F
from training import jumpcue_optimizer as O
from training.library_tracks import Annotation
from training.text import TrainingError

CHORUS_HEADS = (48, 112, 160)       # the synthetic song's chorus runs' heads (test_jumpcue_pairs.LABELS)


def annotation(beats, rows, saved, auto) -> Annotation:
    return Annotation(beats=beats, sample_rate=22050, duration_sec=float(beats[-1] + 0.5), tempo_segments=rows,
                      jump_cue_groups=[[float(beats[b]) for b in g] for g in saved],
                      jump_cue_auto_groups=[[float(beats[b]) for b in g] for g in auto])


class JumpCueTrainingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def given(self, count: int, *, cached: bool = True, **extra) -> list[dict]:
        """``count`` songs: the user extended the analysis's chorus pair to the three chorus heads, kept its ending
        jump, removed its interlude-bridge group."""
        tracks = []
        for i in range(count):
            beats, rows, phrases, feats = synthetic(i)
            a = annotation(beats, rows, saved=[CHORUS_HEADS, (128, 176)],
                           auto=[CHORUS_HEADS[:2], (128, 176), (80, 144)])
            track = F.given_track(self.dir, i, a, added_ts=100 + i, **extra)
            if cached:
                key = O._cache_key(beats, rows)
                atomic_save_npz(self.cache / f"{track['uid']}.npz",
                                {"key": np.asarray(key), **feats},
                                compressed=False)
            tracks.append(track)
        return tracks

    def request(self, **values) -> O.JumpCueOptimizationRequest:
        return O.JumpCueOptimizationRequest(library_dir=self.dir, cache_dir=self.cache,
                                            output_path=self.dir / "out" / "jumpcue_pairs.npz",
                                            **{"cv_folds": 3, **values})

    def test_the_models_written_in_the_apps_format(self):
        tracks = self.given(6)
        missing = F.given_track(self.dir, 7, tracks[0]["annotation"], audio=False)
        none = F.given_track(self.dir, 8, Annotation())
        F.use_tracks([*tracks, missing, none])
        with mock.patch("analyzer.runtime.spawn_pool", F.no_pool):
            result = O.optimize_jumpcue(self.request())
        models = P.load_models(str(result.output_path))
        t = P.pair_track(*synthetic(9)[:3], synthetic(9)[3])
        cand = P.candidates(t)
        X, names = P.features(t, cand)
        self.assertEqual(models.pair_names, names)
        scores = P.pair_scores(X, models)
        self.assertTrue(np.all((scores > 0) & (scores < 1)))
        with np.load(result.output_path, allow_pickle=False) as z:
            self.assertFalse([k for k in z.files if k.startswith(("merge", "second"))])     # the pair scorer alone
        self.assertEqual((result.track_count, result.group_count, result.reviewed_groups), (6, 12, 6))
        self.assertEqual([s.uid for s in result.skipped_tracks], [missing["uid"]])     # the one without groups unlisted
        self.assertEqual(set(result.held_out["pair scorer"]), {"R@1", "R@4", "R@8", "pairs"})
        self.assertEqual(result.held_out["pair scorer"]["pairs"], 6 * 3)       # the extended chorus group's pairs
        self.assertEqual(set(result.held_out["automatic pair"]), {"tracks given one", "a reviewed pair", "the same jump"})

    def test_labels(self):
        beats, rows, phrases, feats = synthetic(0)
        t = P.pair_track(beats, rows, phrases, feats)
        tr = O._Track("u", "u", t, [frozenset(CHORUS_HEADS), frozenset({128, 176})],
                      [frozenset(CHORUS_HEADS[:2]), frozenset({128, 176}), frozenset({80, 144})], False)
        pos, neg = tr.labels()
        # the chorus group the user extended is reviewed, the ending jump left as found is not
        self.assertEqual(pos, {(48, 112): True, (48, 160): True, (112, 160): True, (128, 176): False})
        self.assertEqual(neg, {(80, 144)})
        tr.full = True                     # every group of the track judged
        self.assertTrue(all(tr.labels()[0].values()))
        rows_ = O._pair_rows(tr)
        self.assertTrue(set(map(tuple, rows_["pairs"][rows_["y"] > 0].tolist())) == set(pos))

    def test_a_whole_track_review_from_a_track(self):
        tracks = self.given(6)
        F.use_tracks(tracks)
        with mock.patch("analyzer.runtime.spawn_pool", F.no_pool):
            result = O.optimize_jumpcue(self.request(full_review_from=tracks[3]["uid"]))
        self.assertEqual(result.reviewed_groups, 6 + 3)        # the ending jumps of the last three tracks too
        with self.assertRaises(TrainingError):
            O.optimize_jumpcue(self.request(full_review_from="ffffffff"))

    def test_a_failed_analysis_leaves_the_track_out(self):
        tracks = self.given(7)
        F.use_tracks(tracks)
        stale = self.cache / f"{tracks[6]['uid']}.npz"
        stale.unlink()
        with mock.patch("analyzer.runtime.spawn_pool", F.thread_pool), \
                mock.patch("analyzer.pipeline.jumpcues.jumpcue_inputs", side_effect=ValueError("no audio")):
            result = O.optimize_jumpcue(self.request())
        self.assertEqual(result.track_count, 6)
        self.assertEqual([s.uid for s in result.skipped_tracks], [tracks[6]["uid"]])

    def test_refusals(self):
        F.use_tracks(self.given(4))
        with self.assertRaises(TrainingError):                 # too few songs
            O.optimize_jumpcue(self.request())
        F.use_tracks(self.given(6))
        with self.assertRaises(TrainingError):                 # no held-out scores to learn from
            O.optimize_jumpcue(self.request(cv_folds=1))


if __name__ == "__main__":
    unittest.main()
