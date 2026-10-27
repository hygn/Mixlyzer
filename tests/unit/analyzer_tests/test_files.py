"""analyzer.files: an NPZ / NPY written whole under the name asked for, whatever its suffix (numpy given a path not
ending in ".npz" / ".npy" writes beside it, which left the file replaced empty), and nothing left beside it."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyzer.files import atomic_save_npy, atomic_save_npz


class AtomicSaveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def leftovers(self, target: Path) -> list[str]:
        return sorted(p.name for p in self.dir.iterdir() if p != target)

    def test_npz_under_any_name(self):
        for name in ("model.npz", "model.NPZ", "model", "model.weights"):
            for compressed in (True, False):
                with self.subTest(name=name, compressed=compressed):
                    target = self.dir / name
                    target.write_bytes(b"the old model")
                    atomic_save_npz(target, {"w": np.arange(4.0)}, compressed=compressed)
                    with np.load(target, allow_pickle=False) as z:
                        np.testing.assert_array_equal(z["w"], np.arange(4.0))
                    self.assertEqual(self.leftovers(target), [])
                    target.unlink()

    def test_npy_under_any_name(self):
        for name in ("values.npy", "values.NPY", "values"):
            with self.subTest(name=name):
                target = self.dir / name
                atomic_save_npy(target, np.arange(3))
                np.testing.assert_array_equal(np.load(target, allow_pickle=False), np.arange(3))
                self.assertEqual(self.leftovers(target), [])
                target.unlink()

    def test_check_refusing_leaves_the_old_file(self):
        target = self.dir / "model.npz"
        target.write_bytes(b"the old model")

        def check(written: Path) -> None:
            with np.load(written, allow_pickle=False) as z:
                self.assertIn("w", z.files)       # the written file is what check reads
            raise ValueError("not a model")

        with self.assertRaises(ValueError):
            atomic_save_npz(target, {"w": np.zeros(2)}, check=check)
        self.assertEqual(target.read_bytes(), b"the old model")
        self.assertEqual(self.leftovers(target), [])


if __name__ == "__main__":
    unittest.main()
