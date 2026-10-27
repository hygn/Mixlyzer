"""analyzer.training.beat_grid_check: the end-to-end check's scores (unusable beats with the octave allowed and the
half-beat shift not, missed bars), the per-song changes and the not-worse rule, the model sets, the check's
aggregation, and the whole tracker scoring a synthetic song: the same models score the same, and the decoder
optimizer's own pipeline at the current costs scores what the full analysis does."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.training import beat_decoder_optimizer as D
from analyzer.training import beat_grid_check as G

from unit.analyzer_tests import training_fixtures as F

REF = np.arange(40) * 0.5 + 1.0


class UsableTests(unittest.TestCase):
    def test_same_grid_is_usable(self):
        np.testing.assert_array_equal(G.usable_error(REF, REF), 0.0)

    def test_an_octave_is_usable(self):
        half = G.usable_error(REF, REF[::2])                                     # half tempo
        self.assertLess(float(half[:-1].max()), 1e-9)          # the last beat lies past the analysis's last beat
        double = np.sort(np.r_[REF, REF[:-1] + 0.25])
        self.assertLess(float(G.usable_error(REF, double).max()), 1e-9)

    def test_a_half_beat_shift_is_not(self):
        err = G.usable_error(REF, REF + 0.25)
        self.assertTrue(np.all(err[1:-1] > G.USABLE_SEC))

    def test_small_offsets_measured(self):
        np.testing.assert_allclose(G.usable_error(REF, REF + 0.01), 0.01, atol=1e-9)

    def test_too_short_grids(self):
        self.assertTrue(np.all(np.isinf(G.usable_error(REF, REF[:1]))))
        self.assertEqual(G.usable_error(REF[:1], REF).size, 1)


class BarTests(unittest.TestCase):
    DOWN = REF[::4]

    def test_missed_bars_in_seconds(self):
        self.assertEqual(G.missed_bar_time(self.DOWN, self.DOWN + 0.05), 0.0)
        self.assertAlmostEqual(G.missed_bar_time(self.DOWN, self.DOWN + 0.5), 2.0 * self.DOWN.size)
        self.assertAlmostEqual(G.missed_bar_time(self.DOWN, []), 2.0 * self.DOWN.size)
        self.assertEqual(G.missed_bar_time(self.DOWN[:1], []), 0.0)
        one_wrong = self.DOWN.copy()
        one_wrong[3] += 1.0
        self.assertAlmostEqual(G.missed_bar_time(self.DOWN, one_wrong), 2.0)


class CheckTests(unittest.TestCase):
    def test_changes_and_totals(self):
        before = [[10.0, 2.0], [1.0, 0.0], [3.0, 3.0]]
        after = [[2.0, 2.0], [1.5, 0.0], [9.0, 3.0]]
        c = G.from_scores(["a", "b", "c"], before, after)
        self.assertEqual((c.songs, c.beats_before_sec, c.beats_after_sec, c.bars_before_sec), (3, 14.0, 12.5, 5.0))
        self.assertEqual([s.title for s in c.changes], ["a", "c"])            # b changed by less than a second
        self.assertEqual([s.title for s in c.better], ["a"])
        self.assertEqual([s.title for s in c.worse], ["c"])
        self.assertEqual(c.better[0].change, -8.0)
        self.assertTrue(c.not_worse)

    def test_not_worse_tolerates_rounding_only(self):
        self.assertTrue(G.GridCheck(1, 1.0, 1.0 + 1e-9, 0.0, 0.0).not_worse)
        self.assertFalse(G.GridCheck(1, 1.0, 1.001, 0.0, 0.0).not_worse)
        self.assertEqual(G.GridCheck(0, 0.0, 0.0, 0.0, 0.0).worse, [])

    def test_model_sets(self):
        m = G.ModelSet(tcn="a.npz", critic="c.npz")
        r = m.replace(tcn=Path("b.npz"), decoder=None)
        self.assertEqual((r.tcn, r.critic, r.decoder), ("b.npz", "c.npz", None))
        self.assertEqual(m.tcn, "a.npz")
        self.assertEqual(r.files(), ("None", "b.npz", "c.npz", "None"))

    def test_songs_scored_and_joined(self):
        rows = {"u0": ("A", [(1.0, 0.0), (0.0, 0.0)]), "u1": None, "u2": ("C", [(0.0, 2.0), (0.0, 4.0)])}
        progress = []
        with mock.patch.object(G, "spawn_pool", F.thread_pool), \
                mock.patch.object(G, "_score_song", lambda job: rows[job[0]["uid"]]):
            c = G.grid_check([{"uid": u} for u in rows], Path("."), 100.0, G.ModelSet(), G.ModelSet(), 1,
                             lambda v, t: progress.append(v), (10, 40))
        self.assertEqual((c.songs, c.wrong_before_sec, c.wrong_after_sec), (2, 3.0, 4.0))
        self.assertEqual(progress, [20, 30, 40])
        with mock.patch.object(G, "spawn_pool", F.thread_pool), mock.patch.object(G, "_score_song", lambda job: None):
            self.assertEqual(G.grid_check([{"uid": "u"}], Path("."), 100.0, G.ModelSet(), G.ModelSet(), 1, None,
                                          (0, 1)), G.GridCheck(0, 0.0, 0.0, 0.0, 0.0))


class TrackerTests(unittest.TestCase):
    def test_the_decoder_optimizers_pipeline_scores_what_the_full_analysis_does(self):
        models = G.ModelSet(downbeat_ctx=F.shipped("downbeat_ctx_hgb.npz"), tcn=F.shipped("beat_tcn.npz"),
                            critic=F.shipped("beat_critic.npz"), decoder=F.shipped("beat_decoder.json"),
                            refine_onset=F.shipped("refine_onset_hgb.npz"))
        knobs = D.TR.load_knobs(models.decoder)
        with tempfile.TemporaryDirectory() as tmp:
            a = F.beat_annotation(duration=16.0, bpm=126.0)
            track = F.given_track(Path(tmp), 0, a, beats=a.beats, segments=a.tempo_segments)
            F.write_beat_frames(Path(tmp), track, F.synthetic_frames(int(16.0 / F.FRAME_SEC), a.beats, 0))
            title, (before, after) = G._score_song((track, tmp, 100.0, models, models))
            uid, table = D._score_song((track, tmp, models.files()[:3] + (models.decoder,), models.refine_onset, 100.0,
                                        ((knobs.section,), (knobs.half_reset,), (knobs.odd_bar,))))
            self.assertIsNone(G._score_song(({**track, "uid": F.uid_of(9)}, tmp, 100.0, models, models)))
        self.assertEqual(title, "Song 0")
        self.assertEqual(before, after)
        self.assertTrue(all(np.isfinite(before)))
        self.assertEqual(uid, track["uid"])
        np.testing.assert_allclose(table.reshape(-1), before)


if __name__ == "__main__":
    unittest.main()
