"""analyzer.music.tempo_rows: tempo rows [start, end, bpm, inizio, ts, bpm_end] and the beats they give. Steady rows,
ramps, the legacy widths, and the rows a hand edit can leave (NaN, empty, reversed, overlapping)."""
import unittest

import numpy as np

from analyzer.music import tempo_rows as tr

RNG_SEED = 20261027


def beats_of(rows):
    return tr.rebuild_grid_from_segments(np.asarray(rows, dtype=float))


class EndBpmTests(unittest.TestCase):
    def test_widths(self):
        np.testing.assert_array_equal(tr.with_end_bpm([[0, 10, 120, 0]]), [[0, 10, 120, 0, 4, 120]])
        np.testing.assert_array_equal(tr.with_end_bpm([[0, 10, 120, 0, 3]]), [[0, 10, 120, 0, 3, 120]])
        np.testing.assert_array_equal(tr.with_end_bpm([[0, 10, 120, 0, 3, 140]]), [[0, 10, 120, 0, 3, 140]])
        for empty in ([], np.empty((0, 6)), [1, 2, 3]):
            self.assertEqual(tr.with_end_bpm(empty).shape, (0, 6))

    def test_unusable_end_tempo_is_steady(self):
        for end in (np.nan, np.inf, 0.0, -5.0):
            with self.subTest(end=end):
                row = [0, 10, 120, 0, 4, end]
                self.assertEqual(tr.segment_end_bpm(row), 120.0)
                self.assertFalse(tr.is_ramp_row(row))
                self.assertEqual(tr.with_end_bpm([row])[0, 5], 120.0)

    def test_ramp_tolerance(self):
        self.assertFalse(tr.is_ramp_row([0, 10, 120, 0, 4, 120 + 1e-9]))
        self.assertTrue(tr.is_ramp_row([0, 10, 120, 0, 4, 120.001]))


class RampTests(unittest.TestCase):
    ROW = np.array([10.0, 30.0, 120.0, 10.0, 4.0, 160.0])

    def test_tempo_line(self):
        self.assertAlmostEqual(tr.bpm_at(self.ROW, 10.0), 120.0)
        self.assertAlmostEqual(tr.bpm_at(self.ROW, 20.0), 140.0)
        self.assertAlmostEqual(tr.bpm_at(self.ROW, 30.0), 160.0)
        self.assertAlmostEqual(tr.bpm_at([5, 5, 100, 5, 4, 200], 5.0), 100.0)   # no length: its start tempo

    def test_beats_follow_the_local_tempo(self):
        beats = tr.ramp_beat_times(10.0, 30.0, 120.0, 160.0, 10.0)
        self.assertAlmostEqual(beats[0], 10.0)
        gaps = np.diff(beats)
        self.assertTrue(np.all(gaps > 0))
        self.assertTrue(np.all(np.diff(gaps) < 0))                     # accelerating: every gap shorter
        mids = 0.5 * (beats[1:] + beats[:-1])
        local = 60.0 / np.array([tr.bpm_at(self.ROW, t) for t in mids])
        np.testing.assert_allclose(gaps, local, rtol=2e-3)
        # the number of beats is the integral of the tempo
        self.assertAlmostEqual(len(beats), tr.beats_elapsed(self.ROW, 30.0), delta=1.0)

    def test_decelerating_and_steady_alike(self):
        down = tr.ramp_beat_times(0.0, 20.0, 160.0, 120.0, 0.0)
        self.assertTrue(np.all(np.diff(np.diff(down)) > 0))
        steady = tr.ramp_beat_times(0.0, 10.0, 120.0, 120.0, 0.25)
        np.testing.assert_allclose(np.diff(steady), 0.5)
        self.assertAlmostEqual(steady[0], 0.25)

    def test_anchor_outside_the_row(self):
        """The anchor may be a beat before or after the row: the tempo line is extended there."""
        inside = tr.ramp_beat_times(10.0, 30.0, 120.0, 160.0, 10.0)
        k = 5
        for anchor in (inside[k], inside[-k]):
            np.testing.assert_allclose(tr.ramp_beat_times(10.0, 30.0, 120.0, 160.0, anchor), inside, atol=1e-9)
        before = tr.ramp_beat_times(10.0, 30.0, 120.0, 160.0, 9.5)   # a beat half a second before the start
        self.assertTrue(np.all(before >= 10.0 - 1e-9))

    def test_empty_spans(self):
        self.assertEqual(tr.ramp_beat_times(10.0, 30.0, 120.0, 160.0, 10.0, limit=5.0).size, 0)
        zero = tr.ramp_beat_times(10.0, 10.0, 120.0, 160.0, 10.0, limit=12.0)   # no length: steady at its start
        np.testing.assert_allclose(np.diff(zero), 0.5)

    def test_shift_and_back(self):
        for beats in (0.5, 1.0, 3.0, -2.0):
            with self.subTest(beats=beats):
                moved = self.ROW.copy()
                moved[3] = tr.shift_row_beats(self.ROW, beats)
                self.assertAlmostEqual(tr.beats_elapsed(moved, moved[3]) - tr.beats_elapsed(self.ROW, self.ROW[3]),
                                       beats, places=9)
                back = moved.copy()
                back[3] = tr.shift_row_beats(moved, -beats)
                self.assertAlmostEqual(back[3], self.ROW[3], places=9)
        steady = np.array([0.0, 10.0, 120.0, 0.0, 4.0])
        self.assertAlmostEqual(tr.shift_row_beats(steady, 3), 1.5)

    def test_split_keeps_every_beat(self):
        rows = np.array([[0.0, 10.0, 120.0, 0.0, 4.0, 120.0], self.ROW, [30.0, 40.0, 160.0, 30.0, 4.0, 160.0]])
        whole = beats_of(rows)
        for t in (10.2, 17.31, 25.0, 29.99):
            with self.subTest(t=t):
                left, right = tr.split_row(self.ROW, t)
                self.assertEqual((left[1], right[0]), (t, t))
                self.assertAlmostEqual(left[5], right[2])
                split = np.vstack([rows[0], left, right, rows[2]])
                np.testing.assert_allclose(beats_of(split), whole, atol=1e-6)
        left5, right5 = tr.split_row([0.0, 10.0, 120.0, 0.0, 4.0], 4.0)   # a five-column row stays five columns
        self.assertEqual((len(left5), len(right5), left5[1], right5[0]), (5, 5, 4.0, 4.0))

    def test_retempo_keeps_the_beat_and_the_downbeat(self):
        row = np.array([0.0, 20.0, 120.0, 1.0, 4.0, 120.0])
        beats = tr.rebuild_grid_from_segments(row[None, :])
        keep = beats[10]
        out = tr.retempo_row(row, 130.0, 150.0, keep)
        new = tr.ramp_beat_times(0.0, 20.0, 130.0, 150.0, out[3], 20.0)
        self.assertLess(float(np.min(np.abs(new - keep))), 1e-9)       # the kept beat is still a beat
        # the downbeat: as many beats from the kept beat as it was (8: from 1.0 to 5.0 at 120 BPM)
        self.assertAlmostEqual(tr.beats_elapsed(out, keep) - tr.beats_elapsed(out, out[3]), 8.0, places=6)
        self.assertEqual((out[2], out[5]), (130.0, 150.0))


class RebuildTests(unittest.TestCase):
    def test_bad_input_gives_no_beats(self):
        for bad in (None, [], [1, 2, 3], [[0, 10, 120]], "rows", [["a", "b", "c", "d"]], np.empty((0, 6))):
            with self.subTest(bad=bad):
                self.assertEqual(tr.rebuild_grid_from_segments(bad).size, 0)

    def test_bad_rows_skipped(self):
        good = [10.0, 20.0, 120.0, 10.0, 4.0]
        for bad in ([0, 10, 0, 0, 4], [0, 10, -120, 0, 4], [0, 10, np.nan, 0, 4], [np.nan, 10, 120, 0, 4],
                    [0, np.inf, 120, 0, 4], [10, 5, 120, 10, 4], [5, 5, 120, 5, 4]):
            with self.subTest(bad=bad):
                np.testing.assert_array_equal(beats_of([bad, good]), beats_of([good]))

    def test_nan_inizio_falls_back_to_the_start(self):
        np.testing.assert_allclose(beats_of([[2.0, 4.0, 120.0, np.nan, 4.0]]), [2.0, 2.5, 3.0, 3.5, 4.0])

    def test_inizio_far_outside_the_row(self):
        np.testing.assert_allclose(beats_of([[2.0, 4.0, 120.0, 2.0 - 1000 * 0.5, 4.0]]), [2.0, 2.5, 3.0, 3.5, 4.0])
        np.testing.assert_allclose(beats_of([[2.0, 4.0, 120.0, 2.25 + 1000 * 0.5, 4.0]]), [2.25, 2.75, 3.25, 3.75])

    def test_shared_boundary_beat_once(self):
        beats = beats_of([[0.0, 2.0, 120.0, 0.0, 4.0], [2.0, 4.0, 120.0, 2.0, 4.0]])
        np.testing.assert_allclose(beats, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

    def test_last_row_ends_at_the_track_end(self):
        """No beat past the last row's end (a hair at most), so none past the track's end."""
        beats = beats_of([[0.0, 2.0, 120.0, 0.0, 4.0], [2.0, 3.9, 120.0, 2.0, 4.0]])
        self.assertLessEqual(beats[-1], 3.9 + 1e-6)

    def test_random_grids_are_sane(self):
        rng = np.random.default_rng(RNG_SEED)
        for _ in range(300):
            n = int(rng.integers(1, 6))
            edges = np.concatenate([[0.0], np.cumsum(rng.uniform(0.3, 40.0, n))])
            rows = []
            for k in range(n):
                bpm = float(rng.uniform(60.0, 300.0))
                end_bpm = bpm if rng.random() < 0.6 else float(rng.uniform(60.0, 300.0))
                inizio = edges[k] + float(rng.uniform(-3.0, 3.0))
                rows.append([edges[k], edges[k + 1], bpm, inizio, float(rng.integers(1, 9)), end_bpm])
            beats = beats_of(rows)
            self.assertTrue(np.all(np.isfinite(beats)))
            if beats.size > 1:
                self.assertGreaterEqual(float(np.min(np.diff(beats))), tr.MIN_BEAT_GAP_SEC - 1e-12)
            if beats.size:
                self.assertGreaterEqual(beats[0], -1e-9)
                self.assertLessEqual(beats[-1], edges[-1] + 1e-6)
            for row in rows:                                            # each row's beats at its own tempo
                inside = beats[(beats > row[0] + 0.05) & (beats < row[1] - 0.05)]
                if inside.size > 2 and not tr.is_ramp_row(row):
                    np.testing.assert_allclose(np.diff(inside), 60.0 / row[2], rtol=1e-6)


class PruneTests(unittest.TestCase):
    def test_prune(self):
        self.assertEqual(tr.prune_close_beats([]).size, 0)
        np.testing.assert_array_equal(tr.prune_close_beats([3.0]), [3.0])
        np.testing.assert_array_equal(tr.prune_close_beats([1.0, 0.0, 0.5]), [0.0, 0.5, 1.0])   # sorted
        np.testing.assert_array_equal(tr.prune_close_beats([0.0, 0.04, 0.08, 0.2]), [0.0, 0.08, 0.2])
        np.testing.assert_array_equal(tr.prune_close_beats([0.0, 0.05]), [0.0, 0.05])            # the gap itself kept
        np.testing.assert_array_equal(tr.prune_close_beats([[0.0, 1.0], [2.0, 3.0]]), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
