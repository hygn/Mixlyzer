"""The analyzer's compiled kernels against the code they replace: the same floats, bit for bit."""
import unittest

import numpy as np
from scipy.ndimage import median_filter

import analyzer.beat.downbeat_features as downbeat
import analyzer.phrase.phrase_analyzer as phrase
from analyzer.audio.running_median import median_filter_1d


class RunningMedianTests(unittest.TestCase):
    def test_equals_scipy(self):
        rng = np.random.default_rng(0)
        for shape in ((5, 1), (1, 1), (3, 7), (7, 3), (20, 40), (64, 300)):
            for size in (1, 3, 31):
                for kind in ("normal", "ties", "zeros"):
                    x = np.abs(rng.standard_normal(shape)).astype(np.float32)
                    if kind == "ties":
                        x = np.round(x * 3) / 3
                    if kind == "zeros":
                        x[:, ::2] = 0
                    for axis in (0, 1):
                        with self.subTest(shape=shape, size=size, kind=kind, axis=axis):
                            want = median_filter(x, size=(1, size) if axis == 1 else (size, 1), mode="reflect")
                            np.testing.assert_array_equal(median_filter_1d(x, size, axis), want)


class TreeKernelTests(unittest.TestCase):
    def test_equals_the_numpy_path(self):
        rng = np.random.default_rng(1)
        trees, nodes = 40, 15                       # complete trees of depth 3 (7 splits, 8 leaves)
        left = np.tile(np.r_[np.arange(1, 14, 2), np.zeros(8)], trees).astype(np.int32)
        right = np.tile(np.r_[np.arange(2, 15, 2), np.zeros(8)], trees).astype(np.int32)
        is_leaf = np.tile(np.r_[np.zeros(7), np.ones(8)], trees).astype(bool)
        clf = phrase.NumpyHistGradientBoostingClassifier(
            baseline=rng.standard_normal(3), tree_classes=np.arange(trees) % 3,
            tree_offsets=np.arange(trees + 1) * nodes, node_value=rng.standard_normal(trees * nodes),
            node_feature_idx=rng.integers(0, 6, trees * nodes), node_num_threshold=rng.standard_normal(trees * nodes),
            node_missing_go_to_left=rng.random(trees * nodes) < 0.5, node_left=left, node_right=right,
            node_is_leaf=is_leaf,
        )
        x = rng.standard_normal((200, 6))
        x[rng.random(x.shape) < 0.1] = np.nan
        fast = clf.raw_predict(x)
        kernel, phrase._raw_predict_trees_numba = phrase._raw_predict_trees_numba, None
        try:
            want = clf.raw_predict(x)
        finally:
            phrase._raw_predict_trees_numba = kernel
        np.testing.assert_array_equal(fast, want)

    def test_a_model_walking_outside_its_arrays_is_refused(self):
        """A model file is outside data and the compiled walk has no bounds checks: a class, a child or an
        offset out of range, a loop in a tree, or rows narrower than the features read are refused."""
        nodes = 3   # a split and two leaves
        good = dict(baseline=np.zeros(2), tree_classes=np.array([1]), tree_offsets=np.array([0, nodes]),
                    node_value=np.zeros(nodes), node_feature_idx=np.array([4, 0, 0]),
                    node_num_threshold=np.zeros(nodes), node_missing_go_to_left=np.zeros(nodes, bool),
                    node_left=np.array([1, 0, 0]), node_right=np.array([2, 0, 0]),
                    node_is_leaf=np.array([False, True, True]))
        clf = phrase.NumpyHistGradientBoostingClassifier(**good)
        self.assertEqual(clf.raw_predict(np.zeros((2, 5))).shape, (2, 2))
        with self.assertRaises(ValueError):
            clf.raw_predict(np.zeros((2, 4)))   # feature 4 read
        for bad in (dict(tree_classes=np.array([2])), dict(node_left=np.array([0, 0, 0])),
                    dict(node_right=np.array([3, 0, 0])), dict(tree_offsets=np.array([0, 2])),
                    dict(node_feature_idx=np.array([-1, 0, 0]))):
            with self.subTest(next(iter(bad))), self.assertRaises(ValueError):
                phrase.NumpyHistGradientBoostingClassifier(**{**good, **bad})


class NeighborMedianTests(unittest.TestCase):
    @staticmethod
    def reference(values, radius):
        signal = np.asarray(values, dtype=np.float64)
        out = np.zeros_like(signal)
        for i in range(signal.size):
            nb = np.concatenate([signal[max(0, i - radius):i], signal[i + 1:min(signal.size, i + radius + 1)]])
            out[i] = float(np.median(nb)) if nb.size else signal[i]
        return out

    @staticmethod
    def recurrence(onset):
        out = np.zeros_like(onset)
        for i in range(onset.size):
            periodic = [onset[o] for off in (-8, -4, 4, 8) if 0 <= (o := i + off) < onset.size]
            nearby = [onset[o] for off in (-3, -2, -1, 1, 2, 3) if 0 <= (o := i + off) < onset.size]
            if periodic and nearby:
                out[i] = -abs(onset[i] - float(np.median(periodic))) + abs(onset[i] - float(np.median(nearby)))
        return out

    def test_equals_np_median(self):
        rng = np.random.default_rng(2)
        for n in (0, 1, 2, 3, 9, 17, 400):
            for kind in ("normal", "ties", "nan"):
                x = rng.standard_normal(n)
                if kind == "ties":
                    x = np.round(x * 2) / 2
                if kind == "nan" and n > 2:
                    x[n // 2] = np.nan
                with self.subTest(n=n, kind=kind):
                    for radius in (2, 4):
                        np.testing.assert_array_equal(downbeat._neighbor_median(x, radius), self.reference(x, radius))
                    if downbeat._onset_recurrence_numba is not None:
                        np.testing.assert_array_equal(downbeat._onset_recurrence_numba(x), self.recurrence(x))


if __name__ == "__main__":
    unittest.main()
