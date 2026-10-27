"""The scores themselves (no audio needed: always run)."""
import unittest

import numpy as np

from . import metrics as m


class MetricTests(unittest.TestCase):
    def test_f_measure(self):
        beats = np.arange(100) * 0.5
        self.assertEqual(m.f_measure(beats, beats), 1.0)
        self.assertEqual(m.f_measure(beats, beats + 0.02), 1.0)
        self.assertEqual(m.f_measure(beats, beats + 0.03), 0.0)
        self.assertAlmostEqual(m.f_measure(beats, beats[::2]), 2 * 50 / 150)        # half the beats
        self.assertAlmostEqual(m.f_measure(beats, np.sort(np.r_[beats, beats + 0.25])), 2 * 100 / 300)
        self.assertEqual(m.f_measure([], []), 1.0)
        self.assertEqual(m.f_measure(beats, []), 0.0)
        self.assertEqual(m.f_measure(beats, [np.nan, np.inf]), 0.0)

    def test_one_estimate_one_reference(self):
        self.assertEqual(m.matches([1.0, 1.01], [1.005], 0.025), 1)
        self.assertEqual(m.matches([1.0], [0.99, 1.01], 0.025), 1)

    def test_offset(self):
        beats = np.arange(200) * 0.47 + 0.3
        for shift in (0.0, 0.026, -0.04, 0.059):
            self.assertAlmostEqual(m.offset(beats, beats + shift), shift, places=9)
        self.assertEqual(m.offset(beats, beats + 0.2), 0.0)                          # past the limit: none
        self.assertEqual(m.offset(beats[:5], beats[:5] + 0.03), 0.0)                 # too few to tell

    def test_keys(self):
        ref = [[0, 0, 0.0, 50.0], [2, 0, 50.0, 100.0]]                                  # C, then D
        self.assertAlmostEqual(m.key_agreement(ref, ref, 100.0), 1.0)
        self.assertAlmostEqual(m.key_agreement(ref, [[0, 0, 0.0, 100.0]], 100.0), 0.5)
        self.assertAlmostEqual(m.key_agreement(ref, [[9, 1, 0.0, 50.0], [11, 1, 50.0, 100.0]], 100.0), 1.0)  # Am, Bm
        self.assertAlmostEqual(m.key_agreement(ref, [[0, 1, 0.0, 100.0]], 100.0), 0.0)  # C minor: another scale
        self.assertAlmostEqual(m.key_agreement(ref, [], 100.0), 0.0)
        self.assertAlmostEqual(m.key_agreement(ref, [[0, 0, 1.0, 51.0], [2, 0, 51.0, 101.0]], 100.0, 1.0), 1.0)
        gaps = [[0, 0, 0.0, 40.0], [2, 0, 60.0, 100.0]]                                 # 40 - 60 s: not scored
        self.assertAlmostEqual(m.key_agreement(gaps, [[0, 0, 0.0, 55.0], [2, 0, 55.0, 100.0]], 100.0), 1.0)

    def test_scales(self):
        self.assertEqual(m.scale_of(0, 0), m.scale_of(9, 1))                           # C = Am
        self.assertEqual(m.scale_of(6, 0), m.scale_of(3, 1))                           # Gb = Ebm
        self.assertEqual((m.scale_of(7, 0) - m.scale_of(0, 0)) % 12, 1)                # a fifth up: one sharp more
        self.assertEqual(len({m.scale_of(p, 0) for p in range(12)}), 12)

    def test_phrases(self):
        beats = np.arange(400) * 0.5
        self.assertEqual(m.boundary_f([16.0, 32.0], [16.2, 31.6], beats), 1.0)
        self.assertEqual(m.boundary_f([16.0, 32.0], [17.0], beats), 0.0)
        self.assertEqual(m.boundary_f([], [], beats), 1.0)
        ref = [(0, 16, "INTRO"), (16, 48, "VERSE"), (48, 64, "CHORUS")]
        self.assertAlmostEqual(m.label_agreement(ref, ref, 64.0), 1.0)
        self.assertAlmostEqual(m.label_agreement(ref, [(0, 64, "VERSE")], 64.0), 0.5)


if __name__ == "__main__":
    unittest.main()
