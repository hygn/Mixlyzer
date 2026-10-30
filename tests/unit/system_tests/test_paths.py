"""system.os_support's paths: one key for a file however its path is written (case on Windows, a name's NFD and NFC
forms on macOS: 初音ミク's dakuten, 소녀시대's syllables), a path Windows cannot open with its long paths off, a sync
client's folder and a file kept online only."""
import json
import os
import shutil
import tempfile
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


class CloudSyncTests(unittest.TestCase):
    """A sync client's folder by its settings (Dropbox's info.json, OneDrive's variable), by a folder named as one
    (Google Drive's drive), the cloud files API left out (its answer this machine's own); a file kept online only by
    its attributes."""

    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        dropbox = os.path.join(self.home, "Work Dropbox")
        info_dir = os.path.join(self.home, ".dropbox")
        os.makedirs(info_dir)
        with open(os.path.join(info_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump({"business": {"path": dropbox}}, f)
        for patcher in (mock.patch.object(os.path, "expanduser", lambda p: p.replace("~", self.home, 1)),
                        mock.patch.object(os_support, "IS_WINDOWS", False),
                        mock.patch.object(os_support, "IS_MACOS", False),
                        mock.patch.dict(os.environ, {"OneDrive": os.path.join(self.home, "OneDrive - Company")})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.dropbox = dropbox

    def test_the_services_folders(self):
        self.assertEqual(os_support.cloud_sync_folder(os.path.join(self.dropbox, "DJ", "library")), "Dropbox")
        self.assertEqual(os_support.cloud_sync_folder(os.path.join(self.home, "Box", "library")), "Box")
        self.assertEqual(os_support.cloud_sync_folder(os.path.join("G:", os.sep, "My Drive", "library")),
                         "Google Drive")
        self.assertEqual(os_support.cloud_sync_folder(os.path.join(self.home, "Music", "library")), "")
        self.assertEqual(os_support.cloud_sync_folder(self.dropbox + " backup"), "")      # a name's prefix is not it
        with mock.patch.object(os_support, "IS_WINDOWS", True), \
                mock.patch.object(os_support, "_windows_sync_root", return_value=""):
            self.assertEqual(os_support.cloud_sync_folder(os.path.join(self.home, "OneDrive - Company", "x")),
                             "OneDrive")
        with mock.patch.object(os_support, "IS_WINDOWS", True), \
                mock.patch.object(os_support, "_windows_sync_root", return_value="iCloud"):
            self.assertEqual(os_support.cloud_sync_folder(os.path.join(self.home, "Music")), "iCloud")

    def test_online_only(self):
        stat = mock.Mock(st_file_attributes=0x00400000, st_flags=0x40000000)
        with mock.patch.object(os_support.os, "stat", return_value=stat):
            for windows, macos, expected in ((True, False, True), (False, True, True), (False, False, False)):
                with mock.patch.object(os_support, "IS_WINDOWS", windows), \
                        mock.patch.object(os_support, "IS_MACOS", macos):
                    self.assertEqual(os_support.online_only("ヴヴヴ.m4a"), expected)
        with mock.patch.object(os_support.os, "stat", return_value=mock.Mock(st_file_attributes=0x20)), \
                mock.patch.object(os_support, "IS_WINDOWS", True):
            self.assertFalse(os_support.online_only("ヴヴヴ.m4a"))                  # an archive bit: on the device
        self.assertFalse(os_support.online_only(os.path.join(self.home, "missing.flac")))


if __name__ == "__main__":
    unittest.main()
