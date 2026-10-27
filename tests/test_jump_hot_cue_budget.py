import itertools
import unittest

import numpy as np

from analyzer.jumpcue.JumpCUE import JumpCueConfig, JumpCueEngine, RepeatGroup


class LargestAllowedModel:
    """Stand-in for the learned model: keeps the largest allowed member set."""

    def choose(self, ctx, feats, points, allowed, *, cache=None):
        n = len(points)
        for size in range(n, 1, -1):
            for S in itertools.combinations(range(n), size):
                if allowed([int(points[i]) for i in S]):
                    return 0, S
        return None


def engine(**overrides):
    cfg = JumpCueConfig(min_point_time_sec=0.0, min_point_gap_sec=0.0, **overrides)
    eng = JumpCueEngine(cfg)
    eng._jump_model = LargestAllowedModel()
    return eng


def group(first, count, score):
    return RepeatGroup(starts=[first + 40 * i for i in range(count)], length=32, offset=0, score=score)


class HotCueBudgetTests(unittest.TestCase):
    beats = np.arange(400) * 0.5

    def test_model_points_fill_slots_strongest_first(self):
        out = engine()._model_points([group(0, 6, 3.0), group(10, 6, 2.0), group(20, 3, 1.0)], self.beats)
        self.assertEqual([len(g.starts) for g in out], [6, 2])
        self.assertLessEqual(sum(len(g.starts) for g in out), 8)

    def test_slot_count_is_configurable(self):
        out = engine(hot_cue_slots=4)._model_points([group(0, 3, 2.0), group(10, 3, 1.0)], self.beats)
        self.assertEqual([len(g.starts) for g in out], [3])

    def test_rule_path_keeps_whole_groups_that_fit(self):
        out = engine()._fit_hot_cues([group(0, 6, 3.0), group(10, 3, 2.0), group(20, 2, 1.0)])
        self.assertEqual([len(g.starts) for g in out], [6, 2])


if __name__ == "__main__":
    unittest.main()
