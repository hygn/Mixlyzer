"""The analysis profiles (assets/analysis/profiles.json, core.workers.analysis_settings.analysis_profile): each one
read as the analyzer's AnalysisProfile, Full the whole analysis, the CQT approximation's gains shipped and checked."""
import os
import tempfile
import unittest

import numpy as np

from analyzer.audio.stft_cqt import GAIN_BINS, load_gains, pseudo_cqt
from analyzer.settings import AnalysisProfile
from core.settings.config import default_cfg
from core.workers.analysis_settings import CQT_GAINS_FILE, analysis_profile, analysis_settings
from system.resource_paths import resource_path


class ProfileTests(unittest.TestCase):
    def test_full_is_the_whole_analysis_and_the_default(self):
        self.assertEqual(analysis_profile("full"), AnalysisProfile())
        self.assertEqual(default_cfg().analysisconfig.analysis_profile, "full")
        self.assertEqual(analysis_settings(default_cfg()).profile, AnalysisProfile())

    def test_each_profile_does_less(self):
        balanced, fast = analysis_profile("balanced"), analysis_profile("fast")
        self.assertEqual(balanced.beat_trainings, 1)
        self.assertFalse(balanced.metronome_critic)
        self.assertFalse(balanced.key_tuning_estimate)
        self.assertEqual(balanced.cqt_gains, "")
        self.assertEqual(fast.cqt_gains, str(resource_path(CQT_GAINS_FILE)))

    def test_an_unknown_name_is_full(self):
        self.assertEqual(analysis_profile("turbo"), AnalysisProfile())


class GainsTests(unittest.TestCase):
    def test_the_shipped_gains_read(self):
        gains = load_gains(str(resource_path(CQT_GAINS_FILE)))
        self.assertEqual({k: v.shape for k, v in gains.items()}, {k: (n,) for k, n in GAIN_BINS.items()})

    def test_bad_gains_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.npz")
            np.savez(path, mix=np.ones(10), melody=np.ones(36))
            with self.assertRaises(ValueError):
                load_gains(path)
            path2 = os.path.join(d, "g2.npz")
            np.savez(path2, mix=-np.ones(252), melody=np.ones(36))
            with self.assertRaises(ValueError):
                load_gains(path2)

    def test_a_tone_lands_in_its_bin(self):
        # a 440 Hz tone: the approximated CQT's loudest bin is A4's (C4 + 9 semitones, 12 a octave)
        sr, hop = 22050, 256
        y = np.sin(2 * np.pi * 440.0 * np.arange(sr * 2) / sr).astype(np.float32)
        C = pseudo_cqt(y, sr, hop, 261.6256, 36, 12, np.ones(36))
        self.assertEqual(C.shape, (36, 1 + y.size // hop))
        self.assertEqual(int(np.argmax(C[:, C.shape[1] // 2])), 9)


if __name__ == "__main__":
    unittest.main()
