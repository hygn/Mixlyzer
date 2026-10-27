import unittest

import numpy as np

from analyzer.jumpcue.jump_point_model import (
    feature_term_value,
    feature_vector,
)


class JumpFeatureTermTests(unittest.TestCase):
    def test_base_interaction_and_square(self):
        base = {"long_min": 0.5, "short_min": 0.8}
        self.assertEqual(feature_term_value(base, "long_min"), 0.5)
        self.assertAlmostEqual(feature_term_value(base, "long_min*short_min"), 0.4)
        self.assertAlmostEqual(feature_term_value(base, "short_min^2"), 0.64)

    def test_vector_preserves_model_order_and_is_contiguous(self):
        vector = feature_vector(
            {"a": 2.0, "b": 3.0},
            ["b", "a*b", "a^2"],
        )
        np.testing.assert_array_equal(vector, np.array([3.0, 6.0, 4.0]))
        self.assertTrue(vector.flags.c_contiguous)

    def test_unknown_term_fails_fast(self):
        with self.assertRaises(KeyError):
            feature_term_value({"a": 1.0}, "a*missing")


if __name__ == "__main__":
    unittest.main()
