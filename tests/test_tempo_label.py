import unittest

from views import tempo_label
from views.tempo_label import TOUCH_FADE_PX, TOUCHED_OPACITY, touched_opacity

PX = 0.01   # x units per pixel
HOLD = 0.0  # right of the playhead (left of every label here unless said)


class TouchedOpacityTests(unittest.TestCase):
    def test_apart_full_touching_dimmed(self):
        here = (1.0, 1.5)
        self.assertEqual(touched_opacity(here, (1.5 + 30 * PX, 2.0), HOLD, PX), 1.0)
        self.assertAlmostEqual(touched_opacity(here, (1.5, 2.0), HOLD, PX), TOUCHED_OPACITY)
        self.assertAlmostEqual(touched_opacity(here, (1.49, 2.0), HOLD, PX), TOUCHED_OPACITY)   # overlapping

    def test_fades_over_the_last_pixels(self):
        half = touched_opacity((1.0, 1.5), (1.5 + 0.5 * TOUCH_FADE_PX * PX, 2.0), HOLD, PX)
        self.assertAlmostEqual(half, 1.0 - 0.5 * (1.0 - TOUCHED_OPACITY))

    def test_no_next_label_no_dimming(self):
        self.assertEqual(touched_opacity((1.0, 1.5), None, HOLD, PX), 1.0)
        self.assertEqual(touched_opacity(None, (1.5, 2.0), HOLD, PX), 1.0)

    def test_passed_stays_dimmed(self):
        # the next tempo's label held right of the playhead: this one is past, however far it has moved on
        self.assertEqual(touched_opacity((0.2, 0.7), (1.0, 1.5), 1.0, PX), TOUCHED_OPACITY)
        self.assertEqual(touched_opacity((0.2, 0.7), (0.9, 1.5), 1.0, PX), TOUCHED_OPACITY)   # pushed past it
        self.assertEqual(touched_opacity((0.2, 0.7), (1.2, 1.5), 1.0, PX), 1.0)               # not come yet

    def test_pixel_size_unknown(self):
        self.assertEqual(touched_opacity((1.0, 1.5), (1.6, 2.0), HOLD, 0.0), 1.0)
        self.assertEqual(touched_opacity((1.0, 1.5), (1.5, 2.0), HOLD, 0.0), TOUCHED_OPACITY)

    def test_contiguous_tempos_meet_at_one_beat(self):
        rows = tempo_label.tempo_rows([[0.0, 10.02, 149.0, 0.0, 4], [10.02, 20.0, 205.1, 10.02, 4]])
        shown = tempo_label.snap_to_beats(rows, [0.0, 5.0, 10.0, 15.0, 20.0])
        self.assertEqual(shown[0][1], shown[1][0])   # the first is pushed out where the next comes in


if __name__ == "__main__":
    unittest.main()
