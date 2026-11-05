"""The GUI process stays out of the analysis's own libraries: the analyses run in worker processes, Search by Music loads what it
needs when it starts. In a process of its own, as the app's start imports the window."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEAVY = ("numba", "llvmlite", "sklearn", "librosa", "joblib")


class GuiImportsTests(unittest.TestCase):
    def test_the_window_loads_none_of_them(self):
        code = (f"import sys, app.window; print('LOADED', sorted(m for m in {HEAVY!r} if m in sys.modules))")
        done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=180,
                              env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
        lines = [line for line in done.stdout.splitlines() if line.startswith("LOADED")]
        self.assertTrue(lines, done.stderr[-800:])
        self.assertEqual(lines[-1], "LOADED []")

    def test_a_worker_still_reaches_them_by_the_same_names(self):
        """What the code outside takes from the packages made lazy: the detector, the methods the index is made by."""
        code = ("from analyzer.phrase import detect_phrase_segments; "
                "from analyzer.search.music import landmarks, melody; "
                "from analyzer.search.music.search import MusicMatch, SearchResult, search_audio; "
                "from analyzer.search.music.results import MusicMatch as M; "
                "from analyzer.pipeline.track import LAZY_MODULES; "
                "from core.library.music_index import METHOD; "
                "assert MusicMatch is M; "
                "assert METHOD == f'melody={melody.METHOD};landmarks={landmarks.METHOD}'; "
                "assert callable(detect_phrase_segments) and callable(search_audio) and LAZY_MODULES; "
                "print('OK')")
        done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=180,
                              env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
        self.assertIn("OK", done.stdout, done.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
