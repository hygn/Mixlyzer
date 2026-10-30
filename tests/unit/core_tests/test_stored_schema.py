"""core.analysis_library.stored: the stored format's version (stable, dev) every {uid}.npz carries."""
import unittest

import numpy as np

from core.analysis_library import stored
from core.analysis_library.stored import SCHEMA, Schema, TrackEdits, apply_edits, migratable, schema_of


def _edits() -> TrackEdits:
    return TrackEdits(beats=np.array([0.5, 1.0]), tempo_segments=np.array([[0.0, 2.0, 120.0, 0.5, 4.0, 120.0]]),
                      key_segments=np.empty((0, 4)), phrases=(), jump_cues=(), cue_points=())


class SchemaTests(unittest.TestCase):
    def test_order_and_production(self):
        self.assertLess(Schema(0, 0), Schema(0, 1))
        self.assertLess(Schema(0, 9), Schema(1, 0))
        self.assertTrue(Schema(1, 0).production)
        self.assertFalse(Schema(0, 1).production)

    def test_migratable(self):
        self.assertTrue(migratable(SCHEMA))
        self.assertTrue(migratable(Schema(0, 0)))               # a release's format, older
        self.assertFalse(migratable(Schema(SCHEMA.stable, SCHEMA.dev + 1)))   # newer
        self.assertFalse(migratable(Schema(SCHEMA.stable + 1, 0)))
        if SCHEMA.dev > 1:                                      # an earlier development format: not kept up with
            self.assertFalse(migratable(Schema(SCHEMA.stable, SCHEMA.dev - 1)))

    def test_unversioned_arrays(self):
        self.assertIsNone(schema_of({}))
        self.assertIsNone(stored.StoredAnalysis.from_arrays({}).schema)

    def test_the_editors_save_writes_the_version(self):
        arrays = {}
        apply_edits(arrays, _edits())
        self.assertEqual(schema_of(arrays), SCHEMA)
        self.assertEqual(stored.StoredAnalysis.from_arrays(arrays).schema, SCHEMA)
        # as the store keeps them: 0-d integer arrays
        kept = {k: np.asarray(v) for k, v in arrays.items() if not isinstance(v, dict)}
        self.assertEqual(schema_of(kept), SCHEMA)

    def test_a_newer_format_is_not_written_over(self):
        newer = {stored.SCHEMA_STABLE_KEY: np.int32(SCHEMA.stable), stored.SCHEMA_DEV_KEY: np.int32(SCHEMA.dev + 1)}
        before = dict(newer)
        with self.assertRaises(ValueError):
            apply_edits(newer, _edits())
        self.assertEqual(newer, before)


if __name__ == "__main__":
    unittest.main()
