import unittest

import numpy as np

from ui.edit_panel.quantize import beat_quantize_grid, quantize_to_beat_or_edge


class EditQuantizeTests(unittest.TestCase):
    def test_grid_includes_exact_track_edges(self):
        grid = beat_quantize_grid(np.asarray([0.25, 0.75, 9.6]), 10.0)
        np.testing.assert_allclose(grid, [0.0, 0.25, 0.75, 9.6, 10.0])

    def test_quantizes_near_start_and_end_to_edges(self):
        beats = np.asarray([0.25, 0.75, 9.6])
        self.assertEqual(quantize_to_beat_or_edge(0.05, beats, 10.0), 0.0)
        self.assertEqual(quantize_to_beat_or_edge(9.9, beats, 10.0), 10.0)

    def test_real_beat_still_wins_when_closer(self):
        beats = np.asarray([0.25, 0.75, 9.6])
        self.assertEqual(quantize_to_beat_or_edge(0.2, beats, 10.0), 0.25)
        self.assertEqual(quantize_to_beat_or_edge(9.65, beats, 10.0), 9.6)


if __name__ == "__main__":
    unittest.main()
