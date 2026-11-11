import json
import os
import tempfile
import unittest
from pathlib import Path

from core.settings import config


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


if __name__ == "__main__":
    unittest.main()
