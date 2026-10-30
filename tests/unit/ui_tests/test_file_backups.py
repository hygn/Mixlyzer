"""Settings > Library > Backups (ui.config.backups): the tags' and rekordbox's backups listed with their size, the
selected ones deleted (a folder backup too, its day folder when left empty), a path that is no backup refused. The
backup folders in a temporary folder (never the app folder's)."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from unit.ui_tests.qt_env import APP, isolated  # noqa: F401  (the application the widgets need)

from PySide6 import QtWidgets

from core.tags import backup as tag_backup
from third_party import rekordbox_phrases
from ui.config import backups

_ISOLATION = None


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


class FileBackupTests(unittest.TestCase):
    def setUp(self):
        self.app = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.app, ignore_errors=True)
        for patcher in (mock.patch.object(tag_backup, "app_folder", return_value=self.app),
                        mock.patch.object(rekordbox_phrases, "app_folder", return_value=self.app)):
            patcher.start()
            self.addCleanup(patcher.stop)
        tags = self.app / "backups" / "tags" / "20261031"
        tags.mkdir(parents=True)
        (tags / "ヴヴヴ 012715.mxtags").write_bytes(b"x" * 3000)
        (tags / "ヴヴヴ 013000.flac").write_bytes(b"x" * 200_000)
        rb = self.app / "backups" / "rekordbox" / "20261030"
        (rb / "97db3 235959").mkdir(parents=True)
        (rb / "97db3 235959" / "ANLZ0000.DAT").write_bytes(b"PMAI" + bytes(96))
        (rb / "97db3 235959" / "ANLZ0000.EXT").write_bytes(b"PMAI" + bytes(196))

    def test_listed_and_deleted(self):
        rows = backups.file_backups()
        self.assertEqual(sorted(path.name for _k, path, _t, _s in rows),
                         ["97db3 235959", "ヴヴヴ 012715.mxtags", "ヴヴヴ 013000.flac"])
        self.assertEqual({path.name: size for _k, path, _t, size in rows}["97db3 235959"], 300)
        folder = next(path for _k, path, _t, _s in rows if path.is_dir())
        backups.delete_backup(folder)
        self.assertFalse(folder.parent.exists())                 # its day folder, left empty, gone too
        mxtags = next(path for _k, path, _t, _s in backups.file_backups() if path.suffix == ".mxtags")
        backups.delete_backup(mxtags)
        self.assertEqual([path.name for _k, path, _t, _s in backups.file_backups()], ["ヴヴヴ 013000.flac"])
        self.assertTrue((self.app / "backups" / "tags" / "20261031").is_dir())   # still holds one
        outside = self.app / "elsewhere.txt"
        outside.write_text("not a backup")
        with self.assertRaises(OSError):
            backups.delete_backup(outside)
        self.assertTrue(outside.exists())

    def test_the_page(self):
        dialog = QtWidgets.QWidget()
        group = backups.build_backups_group(dialog)
        dialog.refresh_file_backups()
        tree = dialog.tree_file_backups
        self.assertEqual(tree.topLevelItemCount(), 3)
        self.assertIn("3 backups", group.findChildren(QtWidgets.QLabel)[0].text())
        tree.topLevelItem(0).setSelected(True)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            next(b for b in group.findChildren(QtWidgets.QPushButton) if b.text().startswith("Delete")).click()
        self.assertEqual(tree.topLevelItemCount(), 2)


if __name__ == "__main__":
    unittest.main()
