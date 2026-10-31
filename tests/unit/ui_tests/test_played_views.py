"""The overview's and the main view's BPM and keys as heard (Settings > View, viewconfig.played_tempo_key): the keys
moved by the semitones (the mode kept), the tempos times the speed (the times kept), and only while the setting
is on (views.base.shown_played)."""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from views.base import shown_played
from views.render.keystrip import key_segment_rows, scaled_tempo_segments, transposed


class PlayedTests(unittest.TestCase):
    def test_keys_moved_their_mode_kept(self):
        keys = np.asarray([[0.0, 0.0, 0.0, 10.0], [9.0, 1.0, 10.0, 20.0]])      # C, then A minor
        moved = transposed(keys, 3)
        self.assertEqual([r[0] for r in key_segment_rows(moved)], [3, 12 + 0])     # D#, C minor
        self.assertEqual([r[0] for r in key_segment_rows(transposed(keys, -2))], [10, 12 + 7])   # A#, G minor
        self.assertIs(transposed(keys, 0), keys)
        self.assertEqual(keys[0, 0], 0.0)                                         # the track's own untouched

    def test_tempos_scaled_their_times_kept(self):
        rows = np.asarray([[0.0, 10.0, 120.0, 0.0, 4.0, 120.0], [10.0, 20.0, 120.0, 10.0, 4.0, 140.0]])
        scaled = scaled_tempo_segments(rows, 1.25)
        self.assertEqual(scaled[:, 2].tolist(), [150.0, 150.0])
        self.assertEqual(scaled[:, 5].tolist(), [150.0, 175.0])
        self.assertEqual(scaled[:, :2].tolist(), rows[:, :2].tolist())
        self.assertIs(scaled_tempo_segments(rows, 1.0), rows)

    def test_shown_only_while_the_setting_is_on(self):
        for on, shown in ((True, (1.1, 2)), (False, (1.0, 0))):
            settings = SimpleNamespace(viewconfig=SimpleNamespace(played_tempo_key=on))
            with mock.patch("core.settings.store.current", return_value=settings):
                self.assertEqual(shown_played((1.1, 2)), shown)


if __name__ == "__main__":
    unittest.main()
