import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.settings import config
from analyzer.training.jumpcue_parameter_optimizer import load_cache, save_cache


class UnreadableConfigTests(unittest.TestCase):
    """A config.json that cannot be read is kept aside (its settings not lost) and the defaults written."""

    def setUp(self):
        self.cwd = os.getcwd()
        self.dir = tempfile.TemporaryDirectory()
        os.chdir(self.dir.name)

    def tearDown(self):
        os.chdir(self.cwd)
        config._unreadable_kept.clear()
        self.dir.cleanup()

    def test_kept_aside(self):
        for name, data in (("json", b'{"libconfig": {'), ("utf8", b"\xff\xfe{}"), ("not an object", b"[1, 2]")):
            with self.subTest(name):
                config._unreadable_kept.clear()
                Path("config.json").write_bytes(data)
                cfg = config.load_cfg()
                kept = dict(config.unreadable_settings()).get(config.CONFIG_FILE)
                self.assertIsNotNone(kept)
                self.assertEqual(kept.read_bytes(), data)
                self.assertEqual(json.loads(Path("config.json").read_text(encoding="utf-8")),
                                 json.loads(json.dumps(cfg.to_dict())))
                kept.unlink()

    def test_missing_is_no_warning(self):
        config.load_cfg()
        self.assertEqual(config.unreadable_settings(), [])
        self.assertTrue(Path("config.json").is_file())


class JumpCueCacheTests(unittest.TestCase):
    def test_round_trip_without_pickle(self):
        record = {"key": "k", "harmony": None, "records": [{
            "members": np.array([3, 9]), "pairs": [(0, 1)],
            "cands": [{"d": -2, "member": {"boundary": np.array([0.5, np.nan])}, "long": {(0, 1): -1.0},
                       "short": {(0, 1): np.float64(0.25)}, "pre": {}}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uid.npz"
            save_cache(path, record)
            with np.load(path, allow_pickle=False) as archive:   # a plain NPZ: readable without pickle
                self.assertIn("tree", archive.files)
            back = load_cache(path)
        cand = back["records"][0]["cands"][0]
        self.assertEqual(back["records"][0]["pairs"], [(0, 1)])
        self.assertEqual((cand["d"], cand["long"], cand["short"]), (-2, {(0, 1): -1.0}, {(0, 1): 0.25}))
        np.testing.assert_array_equal(cand["member"]["boundary"], [0.5, np.nan])
        np.testing.assert_array_equal(back["records"][0]["members"], [3, 9])
        self.assertIsNone(back["harmony"])


if __name__ == "__main__":
    unittest.main()
