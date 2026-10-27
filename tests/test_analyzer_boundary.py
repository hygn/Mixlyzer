"""The analyzer is complete in its directory: it imports third-party packages and itself, never the app's code."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PACKAGES = {"app", "audioengine", "core", "migration", "system", "third_party", "ui", "views"}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


class AnalyzerBoundaryTests(unittest.TestCase):
    def test_no_app_imports(self):
        found = [f"{path.relative_to(ROOT)}:{line} {name}"
                 for path in sorted((ROOT / "analyzer").rglob("*.py"))
                 for line, name in _imports(path) if name.split(".")[0] in APP_PACKAGES]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
