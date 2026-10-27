import unittest

import numpy as np

from analyzer.beat.general import bars as B
from analyzer.beat.general import critic as C
from analyzer.beat.general import decode as D
from analyzer.beat.general import junctions as JN
from analyzer.beat.general import tcn as T
from analyzer.music.beat_geometry import downbeat_beat_indices
from analyzer.music.tempo_rows import rebuild_grid_from_segments
from system.resource_paths import resource_path


def _rows_check(test, rows, downbeat_index_list):
    beats = rebuild_grid_from_segments(rows)
    dbs = beats[np.asarray(downbeat_index_list, int)]
    out = B.to_rows(rows, beats, dbs)
    test.assertTrue(np.allclose(rebuild_grid_from_segments(out), beats, atol=1e-6))
    test.assertEqual(downbeat_beat_indices(beats, out).tolist(), sorted(downbeat_index_list))
    return out


class ToRowsTests(unittest.TestCase):
    def test_lone_two_beat_bar_inside_a_steady_section(self):
        rows = np.array([[0.0, 40.0, 120.0, 0.25, 4.0, 120.0]])
        beats = rebuild_grid_from_segments(rows)
        idx = list(range(0, 33, 4)) + list(range(34, beats.size, 4))           # bars of 4, one of 2, bars of 4
        out = _rows_check(self, rows, idx)
        # the bar count restarts with a new row at beat 34 (the 2-beat bar ends there); no row of its own
        self.assertEqual(out.shape[0], 2)
        self.assertAlmostEqual(out[1, 3], beats[34])

    def test_three_four_section_between_four_four(self):
        rows = np.array([[0.0, 60.0, 150.0, 0.1, 4.0, 150.0]])
        beats = rebuild_grid_from_segments(rows)
        idx = list(range(0, 40, 4)) + list(range(40, 76, 3)) + list(range(76, beats.size, 4))
        _rows_check(self, rows, idx)

    def test_bar_crossing_a_tempo_change_and_a_short_bridge(self):
        rows = np.array([[0.0, 20.03, 150.0, 0.0, 4.0, 150.0],
                         [20.03, 20.72, 175.0, 20.3, 4.0, 175.0],        # a two-beat bridge, no bar line inside
                         [20.72, 50.0, 180.0, 20.72, 4.0, 180.0]])
        beats = rebuild_grid_from_segments(rows)
        start = int(np.searchsorted(beats, 20.72 - 1e-6))
        idx = list(range(2, start, 4)) + list(range(start + 1, beats.size, 4))
        _rows_check(self, rows, idx)

    def test_bar_starting_on_the_last_beat_before_a_tempo_change_needs_no_row(self):
        rows = np.array([[0.0, 19.9, 150.0, 0.4, 3.0, 150.0],
                         [19.9, 40.0, 175.0, 20.0, 4.0, 175.0]])
        beats = rebuild_grid_from_segments(rows)
        last = int(np.searchsorted(beats, 19.9)) - 1                         # the 150 grid's last beat
        idx = list(range(1, last + 1, 3)) + list(range(last + 4, beats.size, 4))
        self.assertEqual(idx[list(idx).index(last) + 1], last + 4)           # a bar of 4 from the last 150 beat
        out = _rows_check(self, rows, idx)
        self.assertEqual(out.shape[0], 2)
        self.assertTrue(np.all(out[:, 1] - out[:, 0] >= 60.0 / out[:, 2]))

    def test_ramp_rows_keep_their_beats(self):
        rows = np.array([[0.0, 30.0, 160.0, 0.2, 4.0, 160.0],
                         [30.0, 36.0, 160.0, 30.2, 4.0, 172.0],
                         [36.0, 60.0, 172.0, 36.0, 4.0, 172.0]])
        beats = rebuild_grid_from_segments(rows)
        _rows_check(self, rows, list(range(1, beats.size, 4)))


class DecoderTests(unittest.TestCase):
    def test_partition_finds_one_steady_grid_on_a_clean_pulse(self):
        hop, t0 = 256 / 22050, 0.0
        n = int(60.0 / hop)
        act = np.zeros(n)
        P = 60.0 / 160.0
        for b in np.arange(0.3, 60.0, P):
            act[int(round(b / hop))] = 1.0
        octave = D.Octave(111.0)
        off, dur = t0 - 0.5 * hop, t0 + hop * (n - 1)
        val, fine = D.readers(act, hop, t0)
        base = D.coarse_tempos(act, hop, octave)
        rows = D.partition(act, hop, off, dur, D.with_ratios(base, octave))
        rows = D.partition(act, hop, off, dur, D.with_ratios(np.r_[D.refit_tempos(rows, act, hop, off), base], octave))
        rows = D.final_refit(D.phase_merge(rows, val), val)
        self.assertEqual(rows.shape[0], 1)
        self.assertAlmostEqual(rows[0, 2], 160.0, delta=0.05)

    def test_readings_outside_the_octave_are_folded_back(self):
        hop, t0 = 256 / 22050, 0.0
        n = int(40.0 / hop)
        down = np.zeros(n)
        P = 60.0 / 270.0
        grid = 0.5 + P * np.arange(int(39.0 / P))
        for i, b in enumerate(grid):
            if i % 6 == 1:                                                  # bar lines on the odd beats of the 270 grid
                down[int(round(b / hop))] = 1.0
        val = lambda t: np.zeros(np.asarray(t).size)
        rows = np.array([[0.0, 40.0, 270.0, 0.5, 4.0, 270.0],
                         [40.0, 50.0, 100.0, 40.2, 4.0, 100.0],
                         [50.0, 60.0, 150.0, 50.1, 4.0, 150.0],
                         [60.0, 70.0, 150.0, 60.1, 4.0, 170.0]])
        out = B.fold_into_octave(rows, 111.0, down, t0, hop, val)
        self.assertAlmostEqual(out[0, 2], 135.0)
        k = (grid[1] - out[0, 3]) / (60.0 / 135.0)                          # the phase that holds the bar lines
        self.assertAlmostEqual(k, round(k))
        self.assertAlmostEqual(out[1, 2], 200.0)
        self.assertAlmostEqual(out[1, 3], 40.2)
        self.assertTrue(np.array_equal(out[2:], rows[2:]))                  # in the octave, and a ramp: untouched

    def test_junction_rules(self):
        P = 0.4
        self.assertEqual(JN.classify(P, P, P), 'between')
        self.assertEqual(JN.classify(0.55 * P, P, P), 'half')
        self.assertEqual(JN.classify(0.3 * P, P, P), 'extreme')


class ModelTests(unittest.TestCase):
    def test_shipped_tcn_weights_run(self):
        P = T.load_weights(resource_path('assets/weights/default_beat_tcn.npz'))[0]
        x = np.random.default_rng(0).normal(size=(500, P['mu'].size)).astype(np.float32) * P['sd'] + P['mu']
        beat, down = T.activations(P, x)
        self.assertEqual(beat.shape, (500,))
        self.assertTrue(np.all((beat >= 0) & (beat <= 1)) and np.all((down >= 0) & (down <= 1)))

    def test_shipped_critic_weights_run(self):
        nets = C.load_weights(resource_path('assets/weights/default_beat_critic.npz'))
        self.assertEqual(len(nets), 5)
        cin = nets[0]['inst.weight'].shape[1]
        x = np.random.default_rng(0).normal(size=(3, cin + C.SLOTS, C.WIN * C.SLOTS)).astype(np.float32)
        x[:, -C.SLOTS:] = np.eye(C.SLOTS)[np.arange(C.WIN * C.SLOTS) % C.SLOTS].T[None]
        self.assertEqual(C._forward(nets[0], x).shape, (3,))

    def test_shipped_contextual_downbeat_model_loads(self):
        m = B.load_ctx_model(resource_path('assets/weights/default_downbeat_ctx_hgb.npz'))
        p = m.predict_proba(np.zeros((4, m.node_feature_idx.max() + 1)))[:, 1]
        self.assertEqual(p.shape, (4,))


if __name__ == '__main__':
    unittest.main()
