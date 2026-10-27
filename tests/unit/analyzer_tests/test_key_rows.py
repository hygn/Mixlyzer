"""analyzer.music.key_rows.main_key: the track's key, the one the static key analysis gives (the key held most in the
intro, outro and interlude phrases, else in the track)."""
import unittest

import numpy as np

from analyzer.music.key_rows import main_key
from analyzer.music.marks import Phrase


class MainKeyTests(unittest.TestCase):
    ROWS = [[0, 0, 0.0, 10.0], [9, 1, 10.0, 12.0]]      # C for most of the time, A minor at the end

    def test_the_key_held_most_in_the_mix_phrases(self):
        self.assertEqual(main_key(self.ROWS), 0)
        outro = (Phrase(0.0, 10.0, "VERSE"), Phrase(10.0, 12.0, "OUTRO"))
        self.assertEqual(main_key(self.ROWS, outro, 12.0), 21)     # where the track is mixed
        self.assertEqual(main_key(self.ROWS, (Phrase(0.0, 12.0, "intro"),), 12.0), 0)   # labels in any case

    def test_without_mix_phrases_the_track(self):
        self.assertEqual(main_key(self.ROWS, (Phrase(0.0, 12.0, "CHORUS"),), 12.0), 0)
        self.assertEqual(main_key([[0, 0, 0, 1], [9, 1, 1, 4]]), 21)                  # to the rows' end
        # mix phrases holding no row (past the rows): the track's
        self.assertEqual(main_key(self.ROWS, (Phrase(20.0, 30.0, "OUTRO"),), 30.0), 0)

    def test_a_tie_the_first(self):
        self.assertEqual(main_key([[9, 1, 0, 1], [2, 0, 1, 2]]), 21)

    def test_rows_out_of_order_and_not_finite(self):
        rows = [[7, 0, 1, 3], [np.nan, 0, 0, 1], [0, 1, 0, 1]]
        self.assertEqual(main_key(rows), 7)

    def test_nothing(self):
        self.assertIsNone(main_key(np.empty((0, 4))))
        self.assertIsNone(main_key([[0, 0]]))
        self.assertIsNone(main_key([[np.nan, 0, 0, 1]]))

    def test_the_static_analysis_gives_it(self):
        """chord_key's static analysis takes this key of the dynamic analysis's rows."""
        from analyzer.key.chord_key import static_key
        outro = (Phrase(10.0, 12.0, "OUTRO"),)
        self.assertEqual(static_key(np.asarray(self.ROWS), outro, 12.0), (9, 1))
        self.assertEqual(static_key(np.asarray(self.ROWS), (), 12.0), (0, 0))


if __name__ == "__main__":
    unittest.main()
