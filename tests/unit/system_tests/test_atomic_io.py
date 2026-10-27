"""system.atomic_io: a file is replaced whole or not at all, and no temporary file is left behind either way."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from system import atomic_io


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def leftovers(self, target: Path) -> list[str]:
        return sorted(p.name for p in target.parent.iterdir() if p != target)

    def test_text_written_as_given(self):
        target = self.dir / "a.txt"
        text = "line 1\r\nline 2\nUnicode: é あ \U0001F3B5\n"
        atomic_io.atomic_write_text(target, text)
        self.assertEqual(target.read_bytes(), text.encode("utf-8"))   # no newline translation
        atomic_io.atomic_write_text(target, "x", encoding="utf-16")
        self.assertEqual(target.read_text(encoding="utf-16"), "x")
        self.assertEqual(self.leftovers(target), [])

    def test_json_round_trip_and_replace(self):
        target = self.dir / "c.json"
        target.write_text("old", encoding="utf-8")
        payload = {"a": [1, 2.5, None, True], "名": "値", "nested": {"x": {}}}
        atomic_io.atomic_write_json(target, payload)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), payload)
        self.assertIn("名", target.read_text(encoding="utf-8"))   # ensure_ascii off by default
        self.assertEqual(self.leftovers(target), [])

    def test_parent_folders_made(self):
        target = self.dir / "x" / "y" / "z.json"
        atomic_io.atomic_write_json(target, [])
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), [])

    def test_no_suffix_and_dotted_names(self):
        for name in ("plain", ".hidden", "a.b.c"):
            with self.subTest(name=name):
                target = self.dir / f"in-{name}" / name
                atomic_io.atomic_write_text(target, name)
                self.assertEqual(target.read_text(encoding="utf-8"), name)
                self.assertEqual(self.leftovers(target), [])

    def test_failure_leaves_the_old_file(self):
        target = self.dir / "keep.json"
        target.write_text('{"kept": 1}', encoding="utf-8")
        with self.assertRaises(TypeError):
            atomic_io.atomic_write_json(target, {"bad": object()})   # fails half-way through the dump
        self.assertEqual(target.read_text(encoding="utf-8"), '{"kept": 1}')
        self.assertEqual(self.leftovers(target), [])
        with self.assertRaises(RuntimeError):
            with atomic_io.atomic_output_path(target) as temporary:
                temporary.write_text("half", encoding="utf-8")
                raise RuntimeError("the writer failed")
        self.assertEqual(target.read_text(encoding="utf-8"), '{"kept": 1}')
        self.assertEqual(self.leftovers(target), [])

    def test_temporary_beside_the_target(self):
        """A rename within one folder: never across file systems."""
        target = self.dir / "t.bin"
        with atomic_io.atomic_output_path(target) as temporary:
            self.assertEqual(temporary.parent, target.parent)
            self.assertNotEqual(temporary, target)
            temporary.write_bytes(b"\x00\x01")
        self.assertEqual(target.read_bytes(), b"\x00\x01")

    def test_replace_tried_again_while_refused(self):
        """Windows refuses a replace while another process holds the file for a moment."""
        target = self.dir / "busy.txt"
        real = os.replace
        calls = []

        def refuse_twice(src, dst):
            calls.append(dst)
            if len(calls) <= 2:
                raise PermissionError("held")
            real(src, dst)

        with mock.patch.object(atomic_io.os, "replace", refuse_twice), mock.patch.object(atomic_io, "REPLACE_WAIT_SEC", 0):
            atomic_io.atomic_write_text(target, "done")
        self.assertEqual((len(calls), target.read_text(encoding="utf-8")), (3, "done"))
        self.assertEqual(self.leftovers(target), [])

    def test_replace_gives_up(self):
        target = self.dir / "held.txt"
        target.write_text("old", encoding="utf-8")
        with mock.patch.object(atomic_io.os, "replace", side_effect=PermissionError("held")) as replace, \
                mock.patch.object(atomic_io, "REPLACE_WAIT_SEC", 0):
            with self.assertRaises(PermissionError):
                atomic_io.atomic_write_text(target, "new")
        self.assertEqual(replace.call_count, atomic_io.REPLACE_TRIES)
        self.assertEqual(target.read_text(encoding="utf-8"), "old")
        self.assertEqual(self.leftovers(target), [])

    def test_other_errors_not_retried(self):
        target = self.dir / "x.txt"
        with mock.patch.object(atomic_io.os, "replace", side_effect=FileNotFoundError("gone")) as replace:
            with self.assertRaises(FileNotFoundError):
                atomic_io.atomic_write_text(target, "x")
        self.assertEqual(replace.call_count, 1)


if __name__ == "__main__":
    unittest.main()
