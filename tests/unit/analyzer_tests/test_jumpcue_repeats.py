"""analyzer.jumpcue.repeats on a synthetic harmonic spectrogram (no audio): a passage of chords found again later as it
is and a semitone up (a modulated last chorus), nothing found where nothing repeats, the semitone axis."""
import unittest

import numpy as np

from analyzer.jumpcue import repeats as R

SR, N_FFT, HOP = 22050, 2048, 512
BEAT = 0.5                                  # seconds
BEATS = 240
PASSAGE = (32, 64)                          # beats 32..63 come again at 112 as they are and at 176 a semitone up
SAME, UP = 112, 176


def spectrogram(seed: int = 0, repeat: bool = True) -> np.ndarray:
    """[F, T] the harmonic magnitude of a song of random three-note chords, a new one every two beats, each note with
    four partials; the passage copied at SAME and, transposed +1, at UP."""
    rng = np.random.default_rng(seed)
    chords = [rng.choice(np.arange(48, 72), 3, replace=False) for _ in range(BEATS // 2)]
    notes = [chords[b // 2] for b in range(BEATS)]
    if repeat:
        for k in range(PASSAGE[1] - PASSAGE[0]):
            notes[SAME + k] = notes[PASSAGE[0] + k]
            notes[UP + k] = notes[PASSAGE[0] + k] + 1
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    frames = int(BEATS * BEAT * SR / HOP) + 1
    t = np.arange(frames) * HOP / SR
    M = np.full((freqs.size, frames), 1e-4)
    for b in range(BEATS):
        cols = (t >= b * BEAT) & (t < (b + 1) * BEAT)
        spectrum = np.zeros(freqs.size)
        for m in notes[b]:
            f0 = 440.0 * 2 ** ((m - 69) / 12)
            for h in range(1, 5):
                spectrum += np.exp(-0.5 * ((freqs - h * f0) / 3.0) ** 2) / h
        M[:, cols] += spectrum[:, None]
    return M


class RepeatTests(unittest.TestCase):
    def test_a_passage_found_again_as_it_is_and_a_semitone_up(self):
        beats = np.arange(BEATS) * BEAT
        found = R.repeats(spectrogram(), SR, N_FFT, HOP, beats)
        self.assertTrue(len(found))
        for lag in (SAME - PASSAGE[0], UP - PASSAGE[0]):
            with self.subTest(lag=lag):
                on = [r for r in found if abs((r[2] - r[0]) - lag) <= R.NMS_LAG]
                self.assertTrue(on, f"no repeat on the lag {lag}")
                a0, a1, b0, b1, z = max(on, key=lambda r: r[1] - r[0])
                # the passage covered, its ends reaching at most WINDOW beats beyond (the window's average)
                self.assertTrue(PASSAGE[0] - R.WINDOW <= a0 <= PASSAGE[0])
                self.assertTrue(PASSAGE[1] - 1 <= a1 <= PASSAGE[1] - 1 + R.WINDOW)
                self.assertGreaterEqual(z, R.Z_MIN)
        self.assertTrue(np.all(found[:, 2] > found[:, 0]))       # the earlier passage first
        self.assertTrue(np.all(np.diff(found[:, 4]) <= 0))        # best first

    def test_a_grid_running_past_the_audio(self):
        beats = np.arange(BEATS + 6) * BEAT                      # six beats past the audio's end
        M = spectrogram()
        self.assertTrue(np.all(np.isfinite(R.pitch_beats(M, SR, N_FFT, HOP, beats))))
        self.assertTrue(len(R.repeats(M, SR, N_FFT, HOP, beats)))   # a NaN there once hid every repeat

    def test_nothing_found_where_nothing_repeats(self):
        beats = np.arange(BEATS) * BEAT
        found = R.repeats(spectrogram(repeat=False), SR, N_FFT, HOP, beats)
        self.assertFalse(any(r[1] - r[0] + 1 >= 24 for r in found))     # no long repeat
        self.assertEqual(R.repeats(spectrogram(), SR, N_FFT, HOP, beats[:10]).shape, (0, 5))   # too few beats

    def test_the_semitone_axis(self):
        W = R._semitone_weights(SR, N_FFT)
        self.assertEqual(W.shape, (R.HI - R.LO + 1, N_FFT // 2 + 1))
        sums = W.sum(1)
        self.assertTrue(np.all(np.isclose(sums, 1.0) | (sums == 0.0)))      # a semitone narrower than a bin: empty
        self.assertTrue(np.allclose(sums[55 - R.LO:], 1.0))                   # every one from G3 up has its bins
        a4 = np.argmax(W[69 - R.LO])                     # A4: the bin nearest 440 Hz
        self.assertAlmostEqual(np.fft.rfftfreq(N_FFT, 1.0 / SR)[a4], 440.0, delta=11.0)


if __name__ == "__main__":
    unittest.main()
