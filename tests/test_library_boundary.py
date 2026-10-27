"""The library's boundary (core/library/__init__.py, core/analysis_library/__init__.py): its files (library.db, the
indexes beside it, the {uid}.npz analyses) are read and written by core.library and core.analysis_library alone.
Outside them, code asks the library's owner (core.library.service) to run a job (core.library.jobs,
core.analysis_library.jobs) and gets records (core.library.records) back: no SQL, no import of the modules inside
(db, tieups, variants, music_index, thumbnails; the analyses' store), no path of a track's {uid}.npz
(``analysis_path``). A type named in annotations only (under ``if TYPE_CHECKING:``) is no use of them."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the app's code (benchmark/, experiments/ and interchange/ are research, tests/ build their own libraries)
SCANNED = ("app", "core", "ui", "views", "system", "analyzer", "third_party", "audioengine", "migration", "build")
VENDORED = ("audioengine/native/third_party/", "audioengine/native/target/")
INSIDE = ("core.library.db", "core.library.tieups", "core.library.variants", "core.library.music_index",
          "core.library.thumbnails", "core.analysis_library.store")
LIBRARY = ("core/library/", "core/analysis_library/")
# files outside the library allowed SQLite, each for a database of its own that is not the library's
OWN_SQLITE = {
    "core/tieups/web.py": "the Wikimedia answers' cache, cache/tieup_web.db",
}


def _files():
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith((*LIBRARY, *VENDORED)):
                continue
            yield rel, path


def _type_checking_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for child in node.body:
                lines.update(range(child.lineno, (child.end_lineno or child.lineno) + 1))
    return lines


def _imports(tree: ast.AST):
    """(line, module) of every import, ``from core.library import db`` as core.library.db."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.lineno, node.module
            for alias in node.names:
                yield node.lineno, f"{node.module}.{alias.name}"


def _violations():
    inside = set(INSIDE)
    found = []
    for rel, path in _files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
        typing_only = _type_checking_lines(tree)
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else \
                node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
            if name == "analysis_path":
                found.append(f"{rel}:{node.lineno} names a track's analysis file: ask "
                             "core.analysis_library.jobs for what it holds")
        for line, module in _imports(tree):
            if line in typing_only:
                continue
            top = module.split(".")[0]
            if top == "sqlite3" and rel not in OWN_SQLITE:
                found.append(f"{rel}:{line} imports sqlite3: ask the library's owner (core.library.jobs)")
            elif any(module == m or module.startswith(m + ".") for m in inside):
                found.append(f"{rel}:{line} imports {module}: inside the library, use core.library.jobs / "
                             "core.analysis_library.jobs")
    return found


class LibraryBoundaryTests(unittest.TestCase):
    def test_library_files_only_through_its_owner(self):
        self.assertEqual(_violations(), [])

    def test_the_stored_analyses_below_the_library(self):
        """core.analysis_library is below core.library (its owner's service and jobs use it): it imports nothing of
        core.library, so the two never import each other."""
        found = []
        for path in sorted((ROOT / "core" / "analysis_library").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            for line, module in _imports(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if module == "core.library" or module.startswith("core.library."):
                    found.append(f"{rel}:{line} imports {module}")
        self.assertEqual(found, [])

    def test_the_stored_analyses_used_by_their_surface(self):
        """Outside the library, core.analysis_library is used through its jobs and its types (StoredAnalysis,
        TrackEdits, ...) and the stored-form conversions named here; what a mark is and how it is shown is
        core.marks'."""
        surface = {"core.analysis_library.jobs", "core.analysis_library.stored",
                   "core.analysis_library.cue_points.as_stored",
                   "core.analysis_library.jump_cues.jump_cue_groups_of_pairs"}
        found = []
        for rel, path in _files():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [f"{node.module}.{a.name}" for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for name in names:
                    if not name.startswith("core.analysis_library."):
                        continue
                    if not any(name == s or name.startswith(s + ".") for s in surface):
                        found.append(f"{rel}:{node.lineno} {name}")
        self.assertEqual(found, [])

    def test_the_marks_vocabulary_knows_no_storage(self):
        """core.marks (the marks' vocabulary and rules) imports nothing of the library, the stored analyses or the
        loaded track: they use it."""
        found = []
        for path in sorted((ROOT / "core" / "marks").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            for line, module in _imports(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if module.split(".")[:2] in (["core", "library"], ["core", "analysis_library"], ["core", "track"]):
                    found.append(f"{rel}:{line} imports {module}")
        self.assertEqual(found, [])

    def test_the_exceptions_are_still_needed(self):
        """An exception no file needs any more is taken out of the list."""
        for rel in OWN_SQLITE:
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8-sig"))
            self.assertTrue(any(module == "sqlite3" for _line, module in _imports(tree)), rel)


if __name__ == "__main__":
    unittest.main()
