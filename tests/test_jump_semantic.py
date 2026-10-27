import unittest

import numpy as np

from analyzer.jumpcue.jump_point_model import (
    context_cosine,
    semantic_pair_scores,
)


class JumpSemanticTests(unittest.TestCase):
    def test_identical_audio_context_scores_one(self):
        rng = np.random.default_rng(3)
        phrase = rng.normal(size=(48, 18))
        phrase[24:40] = phrase[4:20]
        scores = semantic_pair_scores(phrase, 4, 24, 16)
        self.assertAlmostEqual(scores["harmony_post"], 1.0, places=7)
        self.assertAlmostEqual(scores["chroma_post"], 1.0, places=7)

    def test_chroma_progression_allows_one_consistent_transposition(self):
        rng = np.random.default_rng(4)
        chroma = rng.normal(size=(40, 12))
        chroma[20:36] = np.roll(chroma[2:18], 5, axis=1)
        direct = context_cosine(chroma, 2, 20, 16)
        invariant = context_cosine(
            chroma, 2, 20, 16, transposition_invariant=True
        )
        self.assertLess(direct, 0.8)
        self.assertAlmostEqual(invariant, 1.0, places=7)

    def test_missing_or_truncated_context_is_neutral(self):
        phrase = np.ones((20, 18))
        self.assertEqual(semantic_pair_scores(None, 2, 4)["harmony_post"], 0.0)
        self.assertEqual(context_cosine(phrase, 10, 12, 16), 0.0)


if __name__ == "__main__":
    unittest.main()
