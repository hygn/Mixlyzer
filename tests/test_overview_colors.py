import unittest

import numpy as np

from views.render.color import oklab_to_srgb, srgb_to_oklab
from views.render import wave

FLOOR = wave.RGB_OVERVIEW_CHROMA_FLOOR   # the least chroma drawn


def flat_bands(n):
    return np.ones((n, wave.WAVE_BAND_COUNT))


class OverviewColorTests(unittest.TestCase):
    def colors(self, values, bands=None, duration=100.0):
        values = np.asarray(values, dtype=np.float64)
        bands = flat_bands(values.shape[0]) if bands is None else bands
        return srgb_to_oklab(wave.overview_colors(values, bands, np.full(values.shape[0], 0.8), duration))

    def test_oklab_round_trip(self):
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [128, 64, 200]])
        np.testing.assert_array_equal(oklab_to_srgb(srgb_to_oklab(rgb)), rgb)

    def test_the_usual_balance_is_not_grey_a_bass_section_warm(self):
        balanced = np.tile([0.5, 0.3, 0.1], (400, 1))
        bass = np.tile([1.5, 0.3, 0.1], (400, 1))   # the low band three times its usual level
        lab = self.colors(np.concatenate([balanced, bass, balanced]))
        self.assertAlmostEqual(np.hypot(lab[100, 1], lab[100, 2]), FLOOR, delta=0.01)
        self.assertGreater(lab[600, 1], 0.05)                        # red-ward (+a) where the bass is up

    def test_hues_taking_turns_do_not_average_to_grey(self):
        values = np.tile([[2.0, 0.3, 0.1], [0.5, 0.3, 0.4]], (500, 1))   # bass-led and highs-led in turn
        lab = self.colors(values, duration=20.0)                     # 25 columns smoothed
        chroma = np.hypot(lab[100:900, 1], lab[100:900, 2])
        self.assertGreater(float(chroma.min()), 0.15)
        self.assertLess(float(np.ptp(lab[100:900, 1:], axis=0).max()), 0.05)   # one colour: no flicker

    def test_brightness_moves_the_colour(self):
        values = np.tile([0.5, 0.3, 0.1], (1200, 1))                 # one balance of the three levels
        bands = flat_bands(1200)
        bands[400:800, 6:9] *= 4.0                                   # brighter inside the mid range only
        bands[800:, :2] *= 4.0                                       # darker: the lowest bands up
        lab = self.colors(values, bands)
        self.assertLess(lab[600, 2], -0.05)                          # blue-ward (-b)
        self.assertGreater(lab[1000, 1], 0.05)                       # red-ward (+a)

    def test_one_lightness_a_leading_band_saturated_silence_neutral(self):
        rng = np.random.default_rng(0)
        values = np.array([0.5, 0.3, 0.1]) * rng.uniform(0.85, 1.15, (1000, 3))   # around one balance
        values[:50] = 0.0                                            # silence
        values[600:900] = [0.05, 0.05, 2.0]                          # the highs far above their usual level
        bands = flat_bands(1000)
        bands[:50] = 0.0
        lab = self.colors(values, bands)
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        # one lightness (a colour out of the sRGB gamut is clipped, its lightness moving; the floor's is not)
        self.assertAlmostEqual(float(lab[10, 0]), wave.RGB_OVERVIEW_LIGHTNESS, delta=0.01)
        in_gamut = chroma < FLOOR + 0.01
        self.assertGreater(int(in_gamut[50:].sum()), 300)
        self.assertAlmostEqual(float(np.median(lab[in_gamut, 0])), wave.RGB_OVERVIEW_LIGHTNESS, delta=0.01)
        self.assertGreater(float(chroma[750]), 0.15)
        self.assertLess(float(lab[750, 2]), 0.0)                     # blue-ward (-b)
        self.assertLess(float(chroma[10]), 0.01)
        self.assertGreater(float(chroma[50:].min()), FLOOR - 0.01)   # never grey

    def test_smoothing_follows_seconds(self):
        values = np.tile([0.5, 0.3, 0.1], (1000, 1))
        values[500] = [5.0, 0.3, 0.1]                                # one loud bass column
        short = self.colors(values, duration=500.0)                  # a column 0.5 s: none smoothed
        long = self.colors(values, duration=10.0)                    # 100 columns a second: 50 smoothed
        self.assertGreater(np.hypot(*short[500, 1:]), np.hypot(*long[500, 1:]) + 0.05)


if __name__ == "__main__":
    unittest.main()
