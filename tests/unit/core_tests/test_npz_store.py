"""core.analysis_library.store: {uid}.npz, the file format of a track's stored analysis. Only NumPy values, never a
pickle; a uid is the only file name it makes (no path from outside); written whole or not at all."""
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path

import numpy as np

from core.analysis_library.store import FeatureNPZStore

UID = "0f8fad5b-d9cb-469f-a165-70867728950e"


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.store = FeatureNPZStore(str(self.dir / "lib"))

    def files(self):
        return sorted(p.name for p in (self.dir / "lib").iterdir())

    def test_round_trip(self):
        features = {"beats": np.arange(5, dtype=np.float64), "sr": 22050, "flag": True, "rows": [[1, 2], [3, 4]],
                    "block": {"start": np.array([0.5], np.float32), "inner": {"x": (1, 2, 3)}}, "f16": np.float16(1.5)}
        self.store.save(UID, features)
        got = self.store.load(UID)
        self.assertEqual(sorted(got), ["beats", "block.inner.x", "block.start", "f16", "flag", "rows", "sr"])
        np.testing.assert_array_equal(got["beats"], np.arange(5))
        self.assertEqual(got["beats"].dtype, np.float64)
        self.assertEqual(got["block.start"].dtype, np.float32)                   # dtypes kept
        self.assertEqual(got["sr"].shape, ())
        np.testing.assert_array_equal(got["rows"], [[1, 2], [3, 4]])
        self.assertEqual(self.files(), [f"{UID}.npz"])

    def test_what_is_not_stored(self):
        self.store.save(UID, {"keep": 1, "text": "a title", "bytes": b"\x00", "none": None,
                              "objects": [1, None], "ragged": [[1], [2, 3]], "dict_in_list": [{"a": 1}]})
        self.assertEqual(sorted(self.store.load(UID)), ["keep"])
        self.store.save(UID, {"labels": ["INTRO", "VERSE"], "mixed": [1, "x"]})   # a list of texts: a text array
        got = self.store.load(UID)
        self.assertEqual(got["labels"].tolist(), ["INTRO", "VERSE"])
        self.assertEqual(got["mixed"].tolist(), ["1", "x"])                       # NumPy's coercion: all texts
        with self.assertRaises(ValueError):
            self.store.save(UID, {"only": "text"})
        with self.assertRaises(ValueError):
            self.store.save(UID, {})
        with self.assertRaises(TypeError):
            self.store.save(UID, [("beats", 1)])

    def test_keys_made_safe(self):
        self.store.save(UID, {"a b/c\\d:e": 1, "": 2, "\u66f2": 3})
        got = self.store.load(UID)
        self.assertIn("a_b_c_d_e", got)
        self.assertIn("k", got)                                                  # an empty key gets a name
        self.assertTrue(all("/" not in k and "\\" not in k for k in got))

    def test_uid_is_the_only_file_name(self):
        for bad in ("../escape", "..\\escape", "/abs/path", "C:\\x", UID + "/../x", "x", "",
                    str(uuid.uuid1()), UID + ".npz"):
            with self.subTest(uid=bad):
                with self.assertRaises(ValueError):
                    self.store.save(bad, {"a": 1})
                with self.assertRaises(ValueError):
                    self.store.load(bad)
                with self.assertRaises(ValueError):
                    self.store.load_keys(bad, ("a",))
                self.assertFalse(self.store.exists(bad))
                self.store.delete(bad)                                           # nothing to delete: no error
        self.assertFalse((self.dir / "escape.npz").exists())
        self.assertEqual(self.files(), [])

    def test_one_file_per_track_whatever_the_spelling(self):
        self.store.save(UID.upper(), {"a": 1})
        self.store.save("{" + UID + "}", {"a": 2})
        self.assertEqual(self.files(), [f"{UID}.npz"])
        self.assertEqual(int(self.store.load(UID)["a"]), 2)

    def test_load_keys(self):
        self.store.save(UID, {"beats": [1.0], "wave": [2.0], "phrase": {"start": [0.0], "end": [1.0]}, "x": 0})
        self.assertEqual(sorted(self.store.load_keys(UID, ("beats",), ("phrase.",))),
                         ["beats", "phrase.end", "phrase.start"])
        self.assertEqual(self.store.load_keys(UID, ()), {})
        self.assertIsNone(self.store.load_keys(str(uuid.uuid4()), ("beats",)))

    def test_missing_and_delete(self):
        other = str(uuid.uuid4())
        with self.assertRaises(FileNotFoundError):
            self.store.load(other)
        self.assertFalse(self.store.exists(other))
        self.store.save(other, {"a": 1})
        self.assertTrue(self.store.exists(other.upper()))
        self.store.delete(other)
        self.store.delete(other)                                                 # twice: no error
        self.assertFalse(self.store.exists(other))

    def test_a_pickled_array_is_never_read(self):
        """An NPZ from elsewhere holding an object array (a pickle) is refused, not unpickled."""
        path = Path(self.store.path(UID))
        np.savez(path, ok=np.arange(3), evil=np.array([{"a": 1}], dtype=object))
        with self.assertRaises(ValueError):
            self.store.load(UID)
        got = self.store.load_keys(UID, ("ok",))                                # the safe arrays alone still read
        np.testing.assert_array_equal(got["ok"], [0, 1, 2])
        with self.assertRaises(ValueError):
            self.store.load_keys(UID, ("evil",))

    def test_not_an_npz(self):
        path = Path(self.store.path(UID))
        for data in (b"", b"not a zip", b"PK\x03\x04" + bytes(30)):
            path.write_bytes(data)
            with self.subTest(data=data[:8]), self.assertRaises(Exception) as caught:
                self.store.load(UID)
            self.assertNotIsInstance(caught.exception, (SystemExit, KeyboardInterrupt))

    def test_compressed_or_not(self):
        plain = FeatureNPZStore(str(self.dir / "plain"), compressed=False)
        plain.save(UID, {"a": np.zeros(1000)})
        self.store.save(UID, {"a": np.zeros(1000)})
        with zipfile.ZipFile(plain.path(UID)) as z:
            self.assertEqual({i.compress_type for i in z.infolist()}, {zipfile.ZIP_STORED})
        with zipfile.ZipFile(self.store.path(UID)) as z:
            self.assertEqual({i.compress_type for i in z.infolist()}, {zipfile.ZIP_DEFLATED})

    def test_replaced_whole(self):
        self.store.save(UID, {"a": 1, "b": 2})
        self.store.save(UID, {"a": 3})
        self.assertEqual(sorted(self.store.load(UID)), ["a"])                   # no key of the old file left
        self.assertEqual(self.files(), [f"{UID}.npz"])                           # no temporary left

    def test_keys_made_alike_refused(self):
        """Two keys made safe alike would lose one of the values: refused, the stored file as it was."""
        self.store.save(UID, {"a": 1})
        for features in ({"a/b": 1, "a_b": 2}, {"x": {"y": 1}, "x.y": 2}):
            with self.subTest(features=list(features)), self.assertRaises(ValueError):
                self.store.save(UID, features)
        self.assertEqual(sorted(self.store.load(UID)), ["a"])


if __name__ == "__main__":
    unittest.main()
