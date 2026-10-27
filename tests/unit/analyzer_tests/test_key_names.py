"""analyzer.music.key_names: the 24 keys (0 - 11 major from C, 12 - 23 minor from Cm) in the Alphanumeric notation
(1A - 12B) and by name. Checked by music theory, not by the table itself."""
import unittest

from analyzer.music.key_names import ALPHANUMERIC_LABELS, CLASSICAL_LABELS, KEY_DISPLAY_LABELS, idx_to_labels


def number(label: str) -> int:
    return int(label[:-1])


class KeyNameTests(unittest.TestCase):
    def test_every_key_once(self):
        for labels in (ALPHANUMERIC_LABELS, CLASSICAL_LABELS, KEY_DISPLAY_LABELS):
            self.assertEqual(len(labels), 24)
            self.assertEqual(len(set(labels)), 24)
        self.assertEqual({number(a) for a in ALPHANUMERIC_LABELS}, set(range(1, 13)))
        self.assertTrue(all(a.endswith("B") for a in ALPHANUMERIC_LABELS[:12]))   # major: B
        self.assertTrue(all(a.endswith("A") for a in ALPHANUMERIC_LABELS[12:]))   # minor: A
        self.assertTrue(all(c.endswith("m") for c in CLASSICAL_LABELS[12:]))

    def test_relative_keys_share_a_number(self):
        """A major key and its relative minor (a minor third below) have one key signature."""
        for tonic in range(12):
            major = ALPHANUMERIC_LABELS[tonic]
            minor = ALPHANUMERIC_LABELS[12 + (tonic + 9) % 12]
            self.assertEqual(number(major), number(minor), (CLASSICAL_LABELS[tonic], minor))

    def test_a_fifth_up_is_one_number_up(self):
        for mode in (0, 12):
            for tonic in range(12):
                here = number(ALPHANUMERIC_LABELS[mode + tonic])
                fifth = number(ALPHANUMERIC_LABELS[mode + (tonic + 7) % 12])
                self.assertEqual(fifth, here % 12 + 1)

    def test_reference_points(self):
        self.assertEqual(idx_to_labels(0), ("8B", "C"))
        self.assertEqual(idx_to_labels(21), ("8A", "Am"))
        self.assertEqual(idx_to_labels(-1), idx_to_labels(23))      # wraps
        self.assertEqual(idx_to_labels(24), idx_to_labels(0))
        self.assertEqual(idx_to_labels(2.9), idx_to_labels(2))      # truncated, as an index
        self.assertEqual(KEY_DISPLAY_LABELS[9], "A (11B)")


if __name__ == "__main__":
    unittest.main()
