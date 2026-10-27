from dataclasses import asdict
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from analyzer.jumpcue.JumpCUE import JumpCueConfig, JumpCueEngine
from analyzer.jumpcue.jump_point_model import (
    PAIR_SCORES, JumpContextCache, JumpPointFeatures, JumpPointModel, compute_jump_masking_context,
)
from analyzer.jumpcue.self_corr_wrapper import SimilarLink, SelfCorrelationReport
from analyzer.jumpcue.vocal_context import JumpContextConfig, VocalContext, jump_point_scores


def context_fixture(seed=3):
    rng = np.random.default_rng(seed)
    n = 128
    mel = np.tile(rng.normal(size=(32, 12)), (4, 1)) + rng.normal(0, 0.01, (n, 12))
    mel /= np.linalg.norm(mel, axis=1, keepdims=True)
    ceps = np.tile(rng.normal(size=(32, 4, 6)), (4, 1, 1))
    ceps /= np.linalg.norm(ceps, axis=-1, keepdims=True)
    frames = np.arange(n * 32) / 128.0
    active = rng.random(frames.size) > 0.3
    ctx = VocalContext(
        beats_time=np.arange(n) * 0.25 + 0.1, frame_times=frames,
        vocal_ratio=active.astype(float), activity=active, gap_activity=active,
        beat_mel=mel.astype(np.float32), beat_db=rng.normal(-12, 4, n),
        beat_ceps=ceps.astype(np.float32), beat_ceps_active=rng.random((n, 4)) > 0.2,
        lyric_null=0.1, shift_tol=1, min_active=2,
    )
    feats = JumpPointFeatures(
        boundary=rng.random(n), mfcc_change=rng.random(n), onset=rng.random(n),
        melody_onset=rng.random(n), downbeat=(np.arange(n) % 4 == 0).astype(float),
    )
    return ctx, feats


class OriginalModel(JumpPointModel):
    """Reference candidate calculations from before the reuse change."""
    def _beat_features(self, ctx, feats, cache=None):
        db = np.asarray(ctx.beat_db, dtype=float)
        fall = np.array([np.median(db[k - 8:k]) - db[k] if k >= 8 else 0.0 for k in range(db.size)])
        return {
            "boundary": feats.boundary, "mfcc_change": feats.mfcc_change, "onset": feats.onset,
            "melody_onset": feats.melody_onset,
            "break_depth": np.array([fall[k:k + 4].max() for k in range(fall.size)]),
            "downbeat": feats.downbeat,
        }

    def _shifts(self, ctx, beat_feats, points, masking=None, cache=None):
        pairs = list(itertools.combinations(range(points.size), 2))
        out = []
        mel = ctx.beat_mel
        for d in range(-self.shift_beats, self.shift_beats + 1):
            ks = points + d
            if ks.min() < self.edge_beats or ks.max() + self.edge_beats >= mel.shape[0]:
                continue
            pair = {name: {} for name in PAIR_SCORES}
            for i, j in pairs:
                s = jump_point_scores(ctx, (int(ks[j]), int(ks[j]) + 1), int(ks[j] - ks[i]))
                pair["long"][(i, j)] = s[0].score if s else -1.0
                pair["pre"][(i, j)] = max(s[0].acoustic_pre, 0.0) * s[0].lyric_pre if s else 0.0
                pair["short"][(i, j)] = float(np.mean(
                    [mel[ks[i] + t] @ mel[ks[j] + t] for t in range(self.short_beats)]))
                pair["masking"][(i, j)] = masking._compute_pair_score(int(ks[i]), int(ks[j])) if masking else 0.0
            out.append((d, ks, {name: values[ks] for name, values in beat_feats.items()}, pair))
        return pairs, out


class OriginalEngine(JumpCueEngine):
    def _pair_curves(self, C, early, late, length, vocal):
        after = np.full(length, -np.inf)
        before = np.full(length, -np.inf)
        for sc in jump_point_scores(vocal, (late, late + length), late - early, self.config.jump_context):
            d = sc.beat_a - late
            after[d] = sc.score
            penalty = max(sc.acoustic_post, 0.0) * sc.lyric_post - sc.score
            before[d] = max(sc.acoustic_pre, 0.0) * sc.lyric_pre - penalty
        if not np.isfinite(after).any():
            d = np.arange(length)
            after = C[early + d, late + d].astype(float)
            prev = np.maximum(d - 1, 0)
            before = C[early + prev, late + prev].astype(float)
        return after, before


def model_fixture(model_type=JumpPointModel):
    path = Path(__file__).resolve().parents[1] / "assets/weights/default_jumpcue_jump_point.json"
    model = JumpPointModel.load(path)
    return model_type(model.names, model.weights, model.short_threshold, 4, 16, model.short_beats)


class JumpContextCacheTests(unittest.TestCase):
    def setUp(self):
        self.ctx, self.feats = context_fixture()
        self.cache = JumpContextCache(self.ctx)

    def assert_scores_equal(self, actual, expected):
        self.assertEqual([s.beat_a for s in actual], [s.beat_a for s in expected])
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(list(asdict(a).values()), list(asdict(b).values()), rtol=0, atol=1e-12)

    def test_partial_ranges_and_track_edges_match_original_scores(self):
        for span in ((0, 128), (35, 42), (26, 51), (100, 128)):
            self.assert_scores_equal(self.cache.scores(span, 24), jump_point_scores(self.ctx, span, 24))

    def test_overlapping_range_only_computes_missing_beats(self):
        with patch("analyzer.jumpcue.jump_point_model.jump_point_scores", wraps=jump_point_scores) as scorer:
            self.cache.scores((45, 65), 24)
            self.cache.scores((50, 70), 24)
            self.cache.score(31, 55)
            self.assertEqual([c.args[1] for c in scorer.call_args_list], [(45, 65), (65, 70)])

    def test_scoring_parameters_are_part_of_cache_key(self):
        first = self.cache.scores((48, 54), 24)
        cfg = JumpContextConfig(cut_weight=2.0)
        changed = self.cache.scores((48, 54), 24, cfg)
        self.assert_scores_equal(changed, jump_point_scores(self.ctx, (48, 54), 24, cfg))
        self.assertTrue(any(a.score != b.score for a, b in zip(first, changed)))
        cfg.cut_weight = 0.0
        self.assert_scores_equal(self.cache.scores((48, 54), 24, cfg),
                                 jump_point_scores(self.ctx, (48, 54), 24, cfg))

    def test_large_local_window_preserves_range_dependent_scores(self):
        cfg = JumpContextConfig(window_beats=2, local_beats=4)
        for span in ((40, 60), (44, 45)):
            self.assert_scores_equal(self.cache.scores(span, 24, cfg), jump_point_scores(self.ctx, span, 24, cfg))

    def test_break_features_are_reused_and_match_previous_formula(self):
        for ref in (8, 5):
            level = np.asarray(self.ctx.beat_db, dtype=float)
            fall = np.zeros(level.size)
            for k in range(ref, level.size):
                fall[k] = np.median(level[k - ref:k]) - level[k]
            np.testing.assert_array_equal(self.cache.level_fall(ref), fall)
            np.testing.assert_array_equal(self.cache.break_depth(ref),
                                          [fall[k:k + 4].max() for k in range(fall.size)])
            self.assertIs(self.cache.level_fall(ref), self.cache.level_fall(ref))
        self.assertIs(self.cache.beat_features(self.feats), self.cache.beat_features(self.feats))

    def test_masking_pair_fft_runs_once_for_both_directions(self):
        pcm = np.random.default_rng(1).normal(0, 0.1, (8000 * 3, 2)).astype(np.float32)
        masking = compute_jump_masking_context(pcm, 8000, [1, 2])
        with patch.object(masking, "_compute_pair_score", wraps=masking._compute_pair_score) as compute:
            first = masking.pair_score(0, 1)
            self.assertEqual(masking.pair_score(1, 0), first)
            self.assertEqual(masking.pair_score(0, 1), first)
            self.assertEqual(compute.call_count, 1)

    def test_final_model_candidates_and_choice_match_original(self):
        pcm = np.random.default_rng(1).normal(0, 0.1, (8000 * 33, 2)).astype(np.float32)
        self.feats.masking = compute_jump_masking_context(pcm, 8000, self.ctx.beats_time)
        current, original = model_fixture(), model_fixture(OriginalModel)
        for points in ([28, 60], [28, 60, 92], [18, 50, 82]):
            points = np.array(points)
            a = current._shifts(self.ctx, current._beat_features(self.ctx, self.feats), points, self.feats.masking)
            b = original._shifts(self.ctx, original._beat_features(self.ctx, self.feats), points, self.feats.masking)
            self.assertEqual(a[0], b[0])
            for ar, br in zip(a[1], b[1]):
                self.assertEqual(ar[0], br[0])
                for name in PAIR_SCORES:
                    np.testing.assert_allclose(list(ar[3][name].values()), list(br[3][name].values()),
                                               rtol=0, atol=1e-12)
            allowed = lambda ks: min(ks) >= 17 and all(k % 2 == 0 for k in ks)
            self.assertEqual(current.choose(self.ctx, self.feats, points, allowed, cache=self.cache),
                             original.choose(self.ctx, self.feats, points, allowed))

    def test_rule_scores_are_reused_by_final_model(self):
        model = model_fixture()
        engine = JumpCueEngine()
        engine._score_cache = self.cache
        C = self.ctx.beat_mel @ self.ctx.beat_mel.T
        with patch("analyzer.jumpcue.jump_point_model.jump_point_scores", wraps=jump_point_scores) as scorer:
            engine._pair_curves(C, 24, 56, 12, self.ctx)
            model.choose(self.ctx, self.feats, [28, 60], lambda _: True, cache=self.cache)
            # Rule stage already covered every shift (56..64) of the model.
            self.assertEqual(scorer.call_count, 1)

    def test_multiple_model_groups_reuse_track_features_and_pair_scores(self):
        model = model_fixture()
        with patch("analyzer.jumpcue.jump_point_model.jump_point_scores", wraps=jump_point_scores) as scorer:
            model.choose(self.ctx, self.feats, [28, 60], lambda _: True, cache=self.cache)
            calls = scorer.call_count
            model.choose(self.ctx, self.feats, [29, 61], lambda _: True, cache=self.cache)
            self.assertEqual(scorer.call_count - calls, 1)  # only the new end beat
            self.assertEqual(scorer.call_args.args[1], (65, 66))

    def test_another_track_cannot_use_existing_context_cache(self):
        other, feats = context_fixture(seed=4)
        with self.assertRaises(ValueError):
            model_fixture().choose(other, feats, [28, 60], lambda _: True, cache=self.cache)
        self.assertIsNot(JumpContextCache(other).level_fall(), self.cache.level_fall())

    def test_engine_run_resets_cache_even_when_no_links_are_found(self):
        engine = JumpCueEngine()
        engine._score_cache = self.cache
        engine._pair_cache[(24, 56, 12)] = (np.ones(12), np.ones(12))
        report = SelfCorrelationReport(beat_ssm=None, links=[])
        with patch("analyzer.jumpcue.JumpCUE.analyze_self_correlation",
                   return_value=(report, np.zeros((12, 128)))):
            result = engine.run(np.zeros(128), 8000, self.ctx.beats_time)
        self.assertEqual(result.pairs, [])
        self.assertIsNone(engine._score_cache)
        self.assertEqual(engine._pair_cache, {})

    def test_group_building_and_final_pairs_match_original(self):
        cfg = JumpCueConfig(min_point_time_sec=0, min_point_gap_sec=2)
        current, original = JumpCueEngine(cfg), OriginalEngine(cfg)
        current._jump_model, original._jump_model = model_fixture(), model_fixture(OriginalModel)
        current._jump_features = original._jump_features = self.feats
        link = SimilarLink(lag_beats=32, lag_sec=8, score=0.9, a=(12.1, 18.1), b=(4.1, 10.1),
                           a_beat=(48, 72), b_beat=(16, 40))
        report = SelfCorrelationReport(beat_ssm=None, links=[link])
        a = current._build_pairs(report, self.ctx.beats_time, self.ctx.beat_mel.T, self.ctx)
        b = original._build_pairs(report, self.ctx.beats_time, self.ctx.beat_mel.T, self.ctx)
        self.assertTrue(a)
        self.assertEqual([p.as_dict() for p in a], [p.as_dict() for p in b])


if __name__ == "__main__":
    unittest.main()
