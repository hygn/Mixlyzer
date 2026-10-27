"""system.ids: a track's, a playlist's, a work's id is a UUIDv4 in one spelling (it names the {uid}.npz and the
thumbnail on disk, so nothing else may pass)."""
import unittest
import uuid

from system.ids import canonical_uuid4, canonical_uuid4_or_none

V4 = "0f8fad5b-d9cb-469f-a165-70867728950e"


class CanonicalUuid4Tests(unittest.TestCase):
    def test_one_spelling(self):
        for text in (V4, V4.upper(), f"  {V4}\n", V4.replace("-", ""), "{" + V4 + "}", f"urn:uuid:{V4}",
                     uuid.UUID(V4)):
            with self.subTest(text=text):
                self.assertEqual(canonical_uuid4(text), V4)

    def test_other_versions_refused(self):
        for version in (1, 3, 5):
            u = {1: uuid.uuid1, 3: lambda: uuid.uuid3(uuid.NAMESPACE_DNS, "x"),
                 5: lambda: uuid.uuid5(uuid.NAMESPACE_DNS, "x")}[version]()
            with self.subTest(version=version), self.assertRaises(ValueError):
                canonical_uuid4(u)
        with self.assertRaises(ValueError):
            canonical_uuid4("00000000-0000-0000-0000-000000000000")   # nil: no version

    def test_never_a_path(self):
        """What reaches a file name: no separators, no traversal, no Windows device names."""
        for text in ("../" + V4, V4 + "/..", V4 + ".npz", "..\\x", "CON", "", " ", "a" * 32, V4[:-1], V4 + "0",
                     f"{V4}\x00", "g" + V4[1:]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                canonical_uuid4(text)

    def test_random_ids_are_their_own_spelling(self):
        for _ in range(200):
            text = str(uuid.uuid4())
            self.assertEqual(canonical_uuid4(text.upper()), text)

    def test_or_none(self):
        for empty in (None, "", "   ", 0):
            self.assertIsNone(canonical_uuid4_or_none(empty))
        self.assertEqual(canonical_uuid4_or_none(V4.upper()), V4)
        with self.assertRaises(ValueError):
            canonical_uuid4_or_none("not an id")


if __name__ == "__main__":
    unittest.main()
