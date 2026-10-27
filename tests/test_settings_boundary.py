"""The settings' one owner (core.settings.store): config.json and deck.json are read at the start alone (app.main,
before the window: the UI language, the library's version check, then the store made of it) and written by the
store alone. Outside core/settings/ no code names load_cfg, save_cfg or save_deck (an import or a call) but app.main's
one read: the others read ``current()`` / the store's ``current`` and write through ``save`` / ``deck_changed``."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the app's code (benchmark/, experiments/ and interchange/ are research; debug/ the user's own scripts)
SCANNED = ("app", "core", "ui", "views", "system", "analyzer", "third_party", "audioengine", "migration", "build")
VENDORED = ("audioengine/native/third_party/", "audioengine/native/target/")
FILE_FUNCTIONS = {"load_cfg", "save_cfg", "save_deck"}
OWNER = "core/settings/"
ALLOWED = {"app/main.py": {"load_cfg"}}      # the start's one read, before there is a store


def _violations() -> list[str]:
    found = []
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith((OWNER, *VENDORED)):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
            allowed = ALLOWED.get(rel, set())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.Name):
                    names = [node.id]
                elif isinstance(node, ast.Attribute):
                    names = [node.attr]
                for name in names:
                    if name in FILE_FUNCTIONS and name not in allowed:
                        found.append(f"{rel}:{node.lineno}: {name}")
    return found


class SettingsBoundaryTests(unittest.TestCase):
    def test_the_settings_files_have_one_owner(self):
        self.assertEqual(_violations(), [])


if __name__ == "__main__":
    unittest.main()
