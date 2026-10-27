"""analyzer.music.key_rows.main_key: the track's key as the analysis gives it (the key of the most beats)."""
import unittest

import numpy as np

from analyzer.music.key_rows import main_key


class MainKeyTests(unittest.TestCase):
    def test_the_key_of_the_most_beats(self):
        rows = [[0, 0, 0.0, 10.0], [9, 1, 10.0, 12.0]]      # C for most of the time, A minor at the end
        self.assertEqual(main_key(rows, np.arange(0, 12, 0.5)), 0)
        self.assertEqual(main_key(rows, [1.0, 10.5, 11.0]), 21)   # beats, not time: A minor holds two of three

    def test_a_tie_the_lowest(self):
        self.assertEqual(main_key([[9, 1, 0, 1], [2, 0, 1, 2]], [0.5, 1.5]), 2)

    def test_boundaries(self):
        rows = [[0, 0, 0, 1], [7, 0, 1, 2]]
        self.assertEqual(main_key(rows, [1.0]), 7)      # on a boundary: the later row's
        self.assertEqual(main_key(rows, [2.0]), 7)      # the track's end: the last row's
        self.assertIsNone(main_key(rows, [2.5, -1.0]))  # outside every row

    def test_rows_out_of_order_and_not_finite(self):
        rows = [[7, 0, 1, 2], [np.nan, 0, 0, 1], [0, 1, 0, 1]]
        self.assertEqual(main_key(rows, [0.5, 0.6, 1.5]), 12)

    def test_nothing(self):
        self.assertIsNone(main_key(np.empty((0, 4)), [1.0]))
        self.assertIsNone(main_key([[0, 0, 0, 1]], []))
        self.assertIsNone(main_key([[0, 0]], [0.5]))


if __name__ == "__main__":
    unittest.main()
