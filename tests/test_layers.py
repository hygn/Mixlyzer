"""The packages' layers: each imports only its own layer and those below it (function-level imports included).

    0 system                          imports nothing of the app
    1 analyzer, audioengine           the analysis and the audio engines (neither imports the other)
    2 core, third_party, migration    the app's features
    3 views                          the plots and what they draw (views.render, the viewport timeline)
    4 ui                              the interface: the widgets, dialogs and panels, the views among them
    5 app                             the entry point and the window
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYER = {"system": 0, "analyzer": 1, "audioengine": 1, "core": 2, "third_party": 2, "migration": 2,
         "views": 3, "ui": 4, "app": 5}
APART = {("analyzer", "audioengine"), ("audioengine", "analyzer")}   # same layer, never each other
SKIP = ("audioengine/native/third_party",)


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


class LayerTests(unittest.TestCase):
    def test_no_import_upwards(self):
        found = []
        for package, layer in LAYER.items():
            for path in sorted((ROOT / package).rglob("*.py")):
                rel = path.relative_to(ROOT).as_posix()
                if rel.startswith(SKIP):
                    continue
                for line, name in _imports(path):
                    target = name.split(".")[0]
                    if target in LAYER and (LAYER[target] > layer or (package, target) in APART):
                        found.append(f"{rel}:{line} {name}")
        self.assertEqual(found, [])

    def test_the_interface_names_controls_not_engines(self):
        """The interface (ui, views) names the decks' and the mixer's controls (audioengine.controls) and the volume
        curve (audioengine.volume), never an engine's modules: what it asks of the engines goes through the player
        (core.playback.player)."""
        allowed = ("audioengine.controls", "audioengine.volume")
        found = []
        for package in ("ui", "views"):
            for path in sorted((ROOT / package).rglob("*.py")):
                rel = path.relative_to(ROOT).as_posix()
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        names = [node.module] if node.module != "audioengine" else \
                            [f"audioengine.{a.name}" for a in node.names]
                    elif isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    else:
                        continue
                    for name in names:
                        if name.split(".")[0] == "audioengine" and not any(
                                name == a or name.startswith(a + ".") for a in allowed):
                            found.append(f"{rel}:{node.lineno} {name}")
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
