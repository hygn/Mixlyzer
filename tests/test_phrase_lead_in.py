import unittest

import numpy as np

from analyzer.phrase import lead_in
from analyzer.phrase.phrase_analyzer import _decode_labels


class LeadInTests(unittest.TestCase):
    def test_candidates_are_first_downbeats_after_beat_zero(self):
        downbeat = np.zeros(40, dtype=bool)
        downbeat[[0, 3, 7, 11, 15, 19, 23]] = True
        self.assertEqual(lead_in.candidates(downbeat).tolist(), [3, 7, 11, 15, 19])

    def test_insert_drops_boundaries_within_a_bar_after_the_end(self):
        bounds = np.array([0, 9, 17, 33, 100])
        self.assertEqual(lead_in.insert_lead_in(bounds, 5).tolist(), [0, 5, 9, 17, 33, 100])
        self.assertEqual(lead_in.insert_lead_in(bounds, 9).tolist(), [0, 9, 17, 33, 100])
        self.assertEqual(lead_in.insert_lead_in(bounds, 15).tolist(), [0, 15, 33, 100])
        self.assertEqual(lead_in.insert_lead_in(bounds, None).tolist(), bounds.tolist())

    def test_label_mask_puts_lead_in_only_on_the_first_segment(self):
        labels = ["CHORUS", "INTRO", "LEAD_IN"]
        logp = np.log(np.full((3, 3), 1 / 3))
        logp[1, 2] = 0.0  # the label model prefers LEAD_IN in the middle
        transition = np.zeros((5, 5))
        common = dict(label_weight=1.0, transition_weight=0.0, length_weight=0.0)
        bounds = np.array([0, 4, 20, 40])
        with_lead_in = _decode_labels(labels, transition, np.zeros(3), np.ones(3), logp, bounds,
                                      allowed=lead_in.label_mask(labels, 3, True), **common)
        without = _decode_labels(labels, transition, np.zeros(3), np.ones(3), logp, bounds,
                                 allowed=lead_in.label_mask(labels, 3, False), **common)
        self.assertEqual(with_lead_in[0], "LEAD_IN")
        self.assertNotIn("LEAD_IN", with_lead_in[1:])
        self.assertNotIn("LEAD_IN", without)

    def test_features_have_fixed_shapes_with_few_downbeats(self):
        rng = np.random.default_rng(0)
        n = 30
        downbeat = np.zeros(n, dtype=bool)
        downbeat[[1, 5]] = True
        cands, loc, presence = lead_in.lead_in_features(
            rng.normal(size=(n, 133)), np.full(n, 0.5), downbeat, np.arange(n) * 0.5, np.array([0, 9, n]), 48)
        self.assertEqual(cands.tolist(), [1, 5])
        self.assertEqual(loc.shape, (lead_in.N_CANDIDATES, len(lead_in.LOCATION_FEATURES)))
        self.assertTrue(np.isnan(loc[2:]).all())
        self.assertEqual(presence.shape, (len(lead_in.PRESENCE_FEATURES),))


if __name__ == "__main__":
    unittest.main()
