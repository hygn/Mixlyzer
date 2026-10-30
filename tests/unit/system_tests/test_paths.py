"""system.os_support's paths: one key for a file however its path is written (case on Windows, a name's NFD and NFC
forms on macOS: 初音ミク's dakuten, 소녀시대's syllables), and a path Windows cannot open with its long paths off."""
import os
import unicodedata
import unittest
from unittest import mock

from system import os_support

NFC = "がくっぽいど 소녀시대.mp3"
NFD = unicodedata.normalize("NFD", NFC)


class PathKeyTests(unittest.TestCase):
    def test_separators_and_dots(self):
        here = os.path.abspath("music")
        self.assertEqual(os_support.path_key(os.path.join(here, "a", "..", "b.mp3")),
                         os_support.path_key(os.path.join(here, ".", "b.mp3")))

    def test_nfd_and_nfc_one_name_on_macos_only(self):
        self.assertNotEqual(NFC, NFD)
        with mock.patch.object(os_support, "IS_MACOS", True):
            self.assertEqual(os_support.path_key(NFD), os_support.path_key(NFC))
        with mock.patch.object(os_support, "IS_MACOS", False):
            self.assertNotEqual(os_support.path_key(NFD), os_support.path_key(NFC))

    @unittest.skipUnless(os_support.IS_WINDOWS, "case folded on Windows only")
    def test_case_on_windows(self):
        self.assertEqual(os_support.path_key(r"C:\Music\Aimer.mp3"), os_support.path_key("c:/music/AIMER.MP3"))


class PathTooLongTests(unittest.TestCase):
    long_path = os.path.abspath(os.path.join(*["x" * 50] * 6, "君の知らない物語.mp3"))

    def test_refused_on_windows_with_long_paths_off(self):
        with mock.patch.object(os_support, "IS_WINDOWS", True), \
                mock.patch.object(os_support, "long_paths_enabled", return_value=False):
            self.assertTrue(os_support.path_too_long(self.long_path))
            self.assertFalse(os_support.path_too_long(os.path.abspath("君の知らない物語.mp3")))

    def test_taken_with_long_paths_on_and_elsewhere(self):
        with mock.patch.object(os_support, "IS_WINDOWS", True), \
                mock.patch.object(os_support, "long_paths_enabled", return_value=True):
            self.assertFalse(os_support.path_too_long(self.long_path))
        with mock.patch.object(os_support, "IS_WINDOWS", False):
            self.assertFalse(os_support.path_too_long(self.long_path))


if __name__ == "__main__":
    unittest.main()
