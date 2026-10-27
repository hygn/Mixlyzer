"""analyzer.music.beat_geometry: bars counted in beats (the playhead's bar.beat, the editor, the metronome), the
segment's time signature, the grid moves the editor makes."""
import unittest

import numpy as np

from analyzer.music import beat_geometry as bg
from analyzer.music import tempo_rows as tr
from analyzer.music.tempo_rows import rebuild_grid_from_segments


def grid(rows):
    rows = np.asarray(rows, dtype=float)
    return rebuild_grid_from_segments(rows), rows


class TimeSignatureTests(unittest.TestCase):
    def test_read_or_four(self):
        self.assertEqual(bg.segment_time_signature([0, 1, 120, 0, 3]), 3)
        self.assertEqual(bg.segment_time_signature([0, 1, 120, 0, 2.6]), 3)
        for row in ([0, 1, 120, 0], [0, 1, 120, 0, np.nan], [0, 1, 120, 0, 0], [0, 1, 120, 0, -2], [0, 1, 120]):
            with self.subTest(row=row):
                self.assertEqual(bg.segment_time_signature(row), 4)


class DownbeatTests(unittest.TestCase):
    def test_no_rows_four_from_the_first_beat(self):
        beats = np.arange(10) * 0.5
        np.testing.assert_array_equal(bg.downbeat_beat_indices(beats, None), [0, 4, 8])
        self.assertEqual(bg.downbeat_beat_indices([], [[0, 10, 120, 0, 4]]).size, 0)
        self.assertEqual(bg.downbeat_beat_indices(None, None).size, 0)

    def test_from_the_inizio_beat(self):
        beats, rows = grid([[0.0, 10.0, 120.0, 1.0, 4.0]])   # the downbeat on the third beat
        idx = bg.downbeat_beat_indices(beats, rows)
        self.assertEqual(idx[0], 2)
        self.assertTrue(np.all(np.diff(idx) == 4))
        self.assertTrue(np.all(np.isclose(beats[idx] - 1.0, np.round((beats[idx] - 1.0) / 2.0) * 2.0)))

    def test_each_rows_own_meter(self):
        beats, rows = grid([[0.0, 6.0, 120.0, 0.0, 3.0], [6.0, 13.9, 120.0, 6.0, 4.0]])
        idx = bg.downbeat_beat_indices(beats, rows)
        np.testing.assert_allclose(beats[idx], [0.0, 1.5, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0])

    def test_unsorted_and_nan_beats(self):
        beats, rows = grid([[0.0, 4.9, 120.0, 0.0, 4.0]])
        shuffled = np.concatenate([beats[::-1], [np.nan, np.inf]])
        np.testing.assert_allclose(bg.downbeat_times(shuffled, rows), [0.0, 2.0, 4.0])

    def test_a_beat_on_the_tracks_end(self):
        """The grid keeps a beat lying exactly on the last row's end (the track's end); it is still a beat of the
        bar: never beat 5 of a 4/4 bar, and a downbeat there is a downbeat as the segments' downbeats say."""
        beats, rows = grid([[0.0, 4.0, 120.0, 0.0, 4.0]])
        self.assertEqual(beats[-1], 4.0)
        np.testing.assert_allclose(bg.downbeat_times(beats, rows), bg.build_downbeats_from_segments(rows))
        self.assertEqual(bg.bar_beat_label(beats, 4.0, rows), "3.1")

    def test_beat_never_past_its_bar(self):
        """Away from the end: the beat number is 1 .. the row's meter, everywhere."""
        beats, rows = grid([[0.0, 7.9, 120.0, 0.5, 3.0], [7.9, 20.3, 133.0, 8.0, 4.0], [20.3, 31.9, 96.0, 20.4, 5.0]])
        idx = bg.downbeat_beat_indices(beats, rows)
        for t in beats[beats > 1.0]:
            bar, beat = bg.bar_beat_position(beats, t, downbeat_indices=idx)
            meter = bg.segment_time_signature(rows[min(int(np.searchsorted(rows[:, 0], t, side="right")) - 1, 2)])
            self.assertTrue(1 <= beat <= meter, (t, bar, beat, meter))

    def test_one_dimensional_rows(self):
        beats, rows = grid([[0.0, 4.0, 120.0, 0.0, 4.0, 120.0]])
        np.testing.assert_array_equal(bg.downbeat_beat_indices(beats, rows.ravel()),
                                      bg.downbeat_beat_indices(beats, rows))

    def test_no_drift_over_a_long_track(self):
        """An hour at an odd tempo: every downbeat is still exactly every fourth beat."""
        beats, rows = grid([[0.0, 3600.0, 173.913, 0.0, 4.0]])
        idx = bg.downbeat_beat_indices(beats, rows)
        self.assertTrue(np.all(np.diff(idx) == 4))
        self.assertEqual(idx[0], 0)

    def test_segment_fallback_without_beats(self):
        np.testing.assert_allclose(bg.downbeat_times([], [[0.0, 4.0, 120.0, 0.0, 4.0]]), [0.0, 2.0, 4.0])
        self.assertIsNone(bg.downbeat_times([], None))
        self.assertIsNone(bg.build_downbeats_from_segments([[0.0, 4.0, 0.0, 0.0, 4.0]]))


class BarBeatTests(unittest.TestCase):
    def setUp(self):
        self.beats, self.rows = grid([[0.0, 10.0, 120.0, 1.0, 4.0]])   # downbeats at 1, 3, 5, ...

    def test_positions(self):
        label = lambda t: bg.bar_beat_label(self.beats, t, self.rows)   # noqa: E731
        self.assertEqual(label(1.0), "1.1")
        self.assertEqual(label(1.49), "1.1")
        self.assertEqual(label(1.5), "1.2")
        self.assertEqual(label(2.9), "1.4")
        self.assertEqual(label(3.0), "2.1")
        self.assertEqual(label(100.0), label(self.beats[-1]))           # past the end: the last beat's

    def test_before_the_first_bar(self):
        self.assertEqual(bg.bar_beat_position(self.beats, 0.0, self.rows), (0, 2))   # beat 0: two before bar 1
        self.assertEqual(bg.bar_beat_position(self.beats, 0.5, self.rows), (0, 1))
        self.assertEqual(bg.bar_beat_position(self.beats, -5.0, self.rows), (0, 3))  # before any beat

    def test_nothing_to_count(self):
        self.assertIsNone(bg.bar_beat_position([], 1.0, self.rows))
        self.assertEqual(bg.bar_beat_label([], 1.0, self.rows), "")
        self.assertIsNone(bg.bar_beat_position(self.beats, 1.0, downbeat_indices=[]))

    def test_precomputed_indices_alike(self):
        idx = bg.downbeat_beat_indices(self.beats, self.rows)
        for t in np.linspace(-1, 11, 97):
            self.assertEqual(bg.bar_beat_position(self.beats, t, self.rows),
                             bg.bar_beat_position(self.beats, t, downbeat_indices=idx))


class GridMoveTests(unittest.TestCase):
    def test_canonical_inizio(self):
        self.assertAlmostEqual(tr.canonical_inizio(9.0, 0.0, 120.0), 1.0)          # bars of 2 s
        self.assertAlmostEqual(tr.canonical_inizio(-0.5, 0.0, 120.0), 1.5)
        self.assertAlmostEqual(tr.canonical_inizio(4.6, 0.0, 120.0, 3), 0.1)       # its own bar: 1.5 s
        self.assertEqual(tr.canonical_inizio(np.nan, 7.0, 120.0), 7.0)
        self.assertEqual(tr.canonical_inizio(3.0, 0.0, 0.0), 3.0)
        self.assertAlmostEqual(tr.canonical_inizio(9.0, 0.0, 120.0, np.nan), 1.0)  # no meter: four

    def test_shift_and_back(self):
        beats, rows = grid([[0.0, 10.0, 120.0, 0.0, 4.0], [10.0, 20.0, 140.0, 10.0, 4.0]])
        moved, moved_rows = tr.shift_grid_in_seg(beats, rows, 1, 0.1)
        self.assertTrue(np.array_equal(moved[moved < 10.0], beats[beats < 10.0]))   # the other row untouched
        np.testing.assert_allclose(moved[moved > 10.05], beats[(beats >= 10.0) & (beats < 19.95)] + 0.1, atol=2e-3)
        back, back_rows = tr.shift_grid_in_seg(moved, moved_rows, 1, -0.1)
        np.testing.assert_allclose(back, beats, atol=2e-3)
        np.testing.assert_allclose(back_rows, rows, atol=1e-9)
        self.assertTrue(np.array_equal(rows[1], [10.0, 20.0, 140.0, 10.0, 4.0]))    # the input not changed

    def test_shift_a_ramp(self):
        beats, rows = grid([[0.0, 20.0, 120.0, 0.0, 4.0, 160.0]])
        moved, moved_rows = tr.shift_grid_in_seg(beats, rows, 0, 0.1)
        self.assertEqual(moved.size, beats.size)
        self.assertTrue(np.all(np.diff(np.diff(moved[1:-1])) < 0))                  # still accelerating
        np.testing.assert_allclose(moved, rebuild_grid_from_segments(moved_rows), atol=1e-6)

    def test_shift_over_a_beat_refused(self):
        beats, rows = grid([[0.0, 10.0, 120.0, 0.0, 4.0]])
        with self.assertRaises(AssertionError):
            tr.shift_grid_in_seg(beats, rows, 0, 0.6)

    def test_period_phase_grid(self):
        main, sub = tr.build_grid_from_period_phase(100, 0.01, 10.0, 3.0)
        np.testing.assert_allclose(main, np.arange(3, 100, 10) * 0.01)
        self.assertEqual(sub.size, 0)
        main, sub = tr.build_grid_from_period_phase(100, 0.01, 10.0, -7.0, subdiv=2)
        np.testing.assert_allclose(main, np.arange(3, 100, 10) * 0.01)
        np.testing.assert_allclose(sub, np.arange(3, 100, 5) * 0.01)
        for period in (0.0, -1.0, np.nan):
            self.assertEqual(tr.build_grid_from_period_phase(100, 0.01, period, 0.0)[0].size, 0)


if __name__ == "__main__":
    unittest.main()
