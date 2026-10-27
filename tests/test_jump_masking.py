import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from analyzer.jumpcue.jump_point_model import (
    JumpMaskingConfig,
    JumpPointModel,
    compute_jump_masking_context,
    jump_point_beat_features,
)


class JumpMaskingTests(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.beats = np.array([1.0, 2.0])
        self.fade = self.sr // 250
        self.audio = np.zeros(self.sr * 3)
        self.audio[self.sr:self.sr + self.fade] = (
            0.03 * np.sin(2 * np.pi * 3000 * np.arange(self.fade) / self.sr)
        )

    def score(self, audio, **kwargs):
        return compute_jump_masking_context(audio, self.sr, self.beats, **kwargs).pair_score(0, 1)

    def with_burst(self, offset=0, members=(1, 2)):
        audio = self.audio.copy()
        burst = np.random.default_rng(0).normal(0, 0.3, int(self.sr * 0.008))
        for member in members:
            start = member * self.sr + offset
            audio[start:start + burst.size] += burst
        return audio

    def test_nearby_shared_attack_masks_fixed_splice_error(self):
        self.assertGreater(self.score(self.with_burst()), self.score(self.audio) + 0.5)

    def test_attack_late_in_the_beat_gets_no_masking_credit(self):
        self.assertAlmostEqual(self.score(self.with_burst(int(self.sr * 0.08))), self.score(self.audio))

    def test_both_directions_must_be_masked(self):
        self.assertGreater(self.score(self.with_burst()), self.score(self.with_burst(members=(1,))))
        ctx = compute_jump_masking_context(self.with_burst(members=(1,)), self.sr, self.beats)
        self.assertEqual(ctx.pair_score(0, 1), ctx.pair_score(1, 0))

    def test_low_band_masker_does_not_hide_high_band_error(self):
        scores = []
        n = int(self.sr * 0.03)
        for frequency in (180, 3000):
            audio = self.audio.copy()
            masker = 0.3 * np.sin(2 * np.pi * frequency * np.arange(n) / self.sr)
            for member in (1, 2):
                start = member * self.sr - int(self.sr * 0.02)
                audio[start:start + n] += masker
            scores.append(self.score(audio))
        self.assertGreater(scores[1], scores[0] + 0.3)

    def test_stereo_phase_cancellation_does_not_hide_error(self):
        stereo = np.column_stack((self.audio, -self.audio))
        self.assertLess(self.score(stereo), 0.1)
        self.assertEqual(self.score(stereo.mean(axis=1)), 1.0)

    def test_identical_continuations_and_silence_have_no_error(self):
        self.assertEqual(self.score(np.zeros_like(self.audio)), 1.0)
        repeated = self.with_burst()
        repeated[2 * self.sr:2 * self.sr + self.fade] = repeated[self.sr:self.sr + self.fade]
        self.assertEqual(self.score(repeated), 1.0)

    def test_score_is_invariant_to_track_gain(self):
        audio = self.with_burst()
        self.assertAlmostEqual(self.score(audio), self.score(audio * 0.01), places=6)

    def test_larger_error_is_less_masked(self):
        audio = self.with_burst()
        bigger = audio.copy()
        bigger[self.sr:self.sr + self.fade] += self.audio[self.sr:self.sr + self.fade] * 4
        self.assertLess(self.score(bigger), self.score(audio))

    def test_bounds_and_empty_audio_are_finite_and_unusable(self):
        ctx = compute_jump_masking_context(np.zeros(0), self.sr, [0, 1])
        self.assertEqual(ctx.pair_score(0, 1), 0.0)
        self.assertEqual(ctx.pair_score(-1, 0), 0.0)
        empty = compute_jump_masking_context(np.zeros(0), self.sr, [])
        self.assertEqual(empty.heads.shape[0], 0)
        with self.assertRaises(ValueError):
            JumpMaskingConfig(hop_ms=0)

    def test_crossfade_duration_is_configurable(self):
        cfg = JumpMaskingConfig(crossfade_ms=6)
        ctx = compute_jump_masking_context(self.audio, self.sr, self.beats, config=cfg)
        self.assertEqual(ctx.heads.shape[1], int(self.sr * 0.006))


class MaskingModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.sr = 22050
        self.beats = np.arange(40) * 0.25 + 0.1
        self.points = [8, 24]
        self.audio = np.zeros(self.sr * 11)
        fade = self.sr // 250
        error = 0.03 * np.sin(2 * np.pi * 3000 * np.arange(fade) / self.sr)
        burst = np.random.default_rng(0).normal(0, 0.3, int(self.sr * 0.008))
        for d in (-1, 0, 1):
            start = int(round(self.beats[self.points[0] + d] * self.sr))
            self.audio[start:start + fade] += error
        for p in self.points:
            start = int(round(self.beats[p + 1] * self.sr))
            self.audio[start:start + burst.size] += burst
        self.frames = SimpleNamespace(
            sample_rate=self.sr, frame_times=np.arange(1100) * 0.01,
            librosa_onset=np.zeros(1100), melody_onset=np.zeros(1100), mfcc=np.zeros((20, 1100)),
        )
        self.ctx = SimpleNamespace(
            beat_mel=np.ones((40, 2)) / np.sqrt(2), beat_db=np.zeros(40),
        )

    def choose(self, model, feats):
        # Hold the existing long/lyric scores constant to isolate the new feature.
        def constant_scores(ctx, span, lag, config):
            return [SimpleNamespace(beat_a=k, score=1.0, acoustic_pre=1.0, lyric_pre=1.0)
                    for k in range(*span)]
        with patch("analyzer.jumpcue.jump_point_model.jump_point_scores", side_effect=constant_scores):
            return model.choose(self.ctx, feats, self.points, lambda _: True)

    def test_masking_features_change_final_jump_choice(self):
        feats = jump_point_beat_features(self.frames, self.beats, None, np.zeros(40), audio=self.audio)
        model = JumpPointModel(["masking_min", "masking_mean"], np.array([0.35, 0.15]), 0.6, 1, 1, 2)
        self.assertEqual(self.choose(model, feats), (1, (0, 1)))

    def test_legacy_model_and_missing_pcm_remain_usable(self):
        feats = jump_point_beat_features(self.frames, self.beats, None, np.zeros(40))
        self.assertIsNone(feats.masking)
        legacy = JumpPointModel(["long_min", "share"], np.array([1.0, 1.0]), 0.6, 1, 1, 2)
        self.assertIsNotNone(self.choose(legacy, feats))
        extended = JumpPointModel(["masking_min"], np.array([0.35]), 0.6, 1, 1, 2)
        self.assertIsNotNone(self.choose(extended, feats))

    def test_bundled_model_registers_new_features(self):
        path = Path(__file__).resolve().parents[1] / "assets/weights/default_jumpcue_jump_point.json"
        model = JumpPointModel.load(path)
        self.assertEqual(len(model.names), model.weights.size)
        self.assertIn("masking_min", model.names)
        self.assertIn("masking_mean", model.names)
        # The masking proxy did not improve the choice; it stays registered with zero weight.
        for name in ("masking_min", "masking_mean"):
            self.assertEqual(model.weights[model.names.index(name)], 0.0)
        self.assertIn("zero", json.loads(path.read_text(encoding="utf-8"))["feature_extensions"]["masking"]["weight_source"])


if __name__ == "__main__":
    unittest.main()
