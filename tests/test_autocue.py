import unittest

import numpy as np

from analyzer.phrase.cue_points import add_autocue_cue_point, autocue_time_sec
from core.analysis_library.cue_points import build_cue_points_np, extract_cue_points
from analyzer.music.marks import Phrase


class AutoCueTests(unittest.TestCase):
    @staticmethod
    def _grid(first_downbeat: float) -> tuple[np.ndarray, np.ndarray]:
        beats = np.arange(0.1, 8.0, 0.5)
        segments = np.asarray([[0.0, 8.0, 120.0, first_downbeat, 4.0]])
        return beats, segments

    def test_detected_lead_in_end_has_priority(self):
        beats, segments = self._grid(0.1)
        phrases = [
            Phrase(0.0, 2.1, "LEAD_IN"),
            Phrase(2.1, 8.0, "INTRO"),
        ]
        self.assertAlmostEqual(autocue_time_sec(phrases, beats, segments), 2.1)

    def test_short_silence_before_first_downbeat_cues_the_downbeat(self):
        beats, segments = self._grid(0.1)
        self.assertAlmostEqual(autocue_time_sec([], beats, segments), 0.1)

    def test_missing_initial_downbeat_cues_zero(self):
        beats, segments = self._grid(1.6)
        self.assertEqual(autocue_time_sec([], beats, segments), 0.0)

    def test_exactly_one_beat_before_downbeat_is_not_short_preroll(self):
        beats = np.arange(0.5, 8.0, 0.5)
        segments = np.asarray([[0.0, 8.0, 120.0, 0.5, 4.0]])
        self.assertEqual(autocue_time_sec([], beats, segments), 0.0)

    def test_zero_autocue_is_stored(self):
        points = add_autocue_cue_point([], 0.0)
        restored = extract_cue_points({"cue_points_np": build_cue_points_np(points)})
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].time_sec, 0.0)
        self.assertEqual(restored[0].label, "AUTOCUE")


if __name__ == "__main__":
    unittest.main()
