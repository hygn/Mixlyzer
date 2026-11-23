"""ReplayGain's loudness (analyzer.audio.loudness): ITU-R BS.1770-4 against EBU Tech 3341's minimum requirements
(cases 1 - 5: stereo 1 kHz sines, -23.0 LUFS +- 0.1 LU), at the rates library files have."""
import unittest

import numpy as np

from analyzer.audio.loudness import integrated_loudness, replaygain_db


def sine(db: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    y = 10.0 ** (db / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    return np.stack([y, y], axis=1)


def parts(rate: int, *spec) -> np.ndarray:
    return np.concatenate([sine(db, sec, rate) for db, sec in spec], axis=0)


class Tech3341Tests(unittest.TestCase):
    def check(self, pcm, rate, expected):
        got = integrated_loudness(pcm, rate)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, expected, delta=0.1)

    def test_cases_1_to_5(self):
        for rate in (44100, 48000, 96000):
            with self.subTest(rate=rate):
                self.check(sine(-23.0, 20.0, rate), rate, -23.0)                                  # case 1
                self.check(sine(-33.0, 20.0, rate), rate, -33.0)                                  # case 2
                self.check(parts(rate, (-36, 10), (-23, 60), (-36, 10)), rate, -23.0)             # case 3: relative gate
                self.check(parts(rate, (-72, 10), (-36, 10), (-23, 60), (-36, 10), (-72, 10)),    # case 4: both gates
                           rate, -23.0)
                self.check(parts(rate, (-26, 20), (-20, 20.1), (-26, 20)), rate, -23.0)          # case 5

    def test_the_standards_coefficients_at_48k(self):
        # BS.1770-4 Tables 1 and 2
        from analyzer.audio.loudness import _k_weighting
        np.testing.assert_allclose(_k_weighting(48000), [
            [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
            [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]], rtol=0, atol=1e-12)

    def test_silence_and_short(self):
        self.assertIsNone(integrated_loudness(np.zeros((48000 * 5, 2)), 48000))
        self.assertIsNone(integrated_loudness(sine(-20.0, 0.2, 48000), 48000))     # shorter than one block

    def test_replaygain(self):
        self.assertAlmostEqual(replaygain_db(-23.0), 5.0)
        self.assertAlmostEqual(replaygain_db(-8.0), -10.0)
        self.assertIsNone(replaygain_db(None))


if __name__ == "__main__":
    unittest.main()
