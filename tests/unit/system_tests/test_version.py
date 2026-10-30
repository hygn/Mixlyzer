"""system.version: the app's version is pyproject.toml's (the diagnostics name it)."""
import tomllib
import unittest
from pathlib import Path

from system.version import APP_VERSION

ROOT = Path(__file__).resolve().parents[3]


class VersionTests(unittest.TestCase):
    def test_pyprojects(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(APP_VERSION, project["version"])


if __name__ == "__main__":
    unittest.main()
