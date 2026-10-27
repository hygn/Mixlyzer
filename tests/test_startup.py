import os
import unittest
from unittest import mock

from app import startup


class BlasThreadTests(unittest.TestCase):
    def test_one_thread_while_loading_then_taken_back(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
            startup.one_blas_thread()
            self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")
            with mock.patch.object(startup, "_IMPORTS", ()):
                startup._import_libraries(lambda *a, **k: None)
            self.assertNotIn("OPENBLAS_NUM_THREADS", os.environ)   # the worker processes keep theirs

    def test_the_users_own_setting_kept(self):
        with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "6"}):
            startup.one_blas_thread()
            with mock.patch.object(startup, "_IMPORTS", ()):
                startup._import_libraries(lambda *a, **k: None)
            self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "6")


class ConsoleTests(unittest.TestCase):
    def test_lines_parsed_and_passed_on(self):
        import io
        from app.startup_console import ConsoleLines, ConsoleTap
        lines = ConsoleLines()
        out = io.StringIO()
        tap = ConsoleTap(out, lines)
        print("[Startup] Loading libraries: 2.81 s", file=tap)
        tap.write("[AudioEngine] master on ")   # print's pieces: a line is kept once it ends
        self.assertEqual(len(lines.last(10)), 1)
        tap.write("DDJ-400\n[AudioEngine] device reopen failed: gone\nplain\n")
        got = lines.last(10)
        self.assertEqual(out.getvalue(), "[Startup] Loading libraries: 2.81 s\n[AudioEngine] master on DDJ-400\n"
                                         "[AudioEngine] device reopen failed: gone\nplain\n")
        self.assertEqual([(l.tag, l.text, l.duration, l.failed) for l in got], [
            ("Startup", "Loading libraries", "2.81 s", False),
            ("AudioEngine", "master on DDJ-400", "", False),
            ("AudioEngine", "device reopen failed: gone", "", True),
            ("", "plain", "", False),
        ])

    def test_installed_and_taken_out(self):
        import sys
        from app.startup_console import ConsoleLines, ConsoleTap
        before = sys.stdout
        lines = ConsoleLines()
        lines.install()
        try:
            self.assertIsInstance(sys.stdout, ConsoleTap)
            print("[Startup] OS: test")
        finally:
            lines.uninstall()
        self.assertIs(sys.stdout, before)
        print("after")
        self.assertEqual([l.text for l in lines.last(10)], ["OS: test"])


if __name__ == "__main__":
    unittest.main()
