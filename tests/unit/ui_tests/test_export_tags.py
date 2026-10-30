"""The Export tab's Write Tags (ui.library.tabs.export): each change confirmed first (the file's value and the one
written), nothing written when cancelled, the backup asked for made before the write and put back by Restore from
Backup, no confirmation when the file already holds the values. A real file in a temporary folder; the backups in
another (never the app folder's)."""
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from unit.ui_tests.qt_env import APP, isolated

from PySide6 import QtWidgets

from core.library import jobs
from core.library.records import TrackRow
from core.tags import backup as tag_backup
from core.tags import writer as tw
from core.tags.writer import read_tags
from ui.library.tabs import export as export_tab
from ui.library.tabs.export import ExportTrackPanel

_ISOLATION = None
UID = "0f8fad5b-d9cb-469f-a165-70867728950e"


def wait_until(done, timeout_ms: int = 5000) -> None:
    """Qt's events run (the worker threads' signals delivered) until ``done()``; the wait a sleep that lets the
    worker threads run (QTest.qWait keeps the GIL)."""
    for _ in range(timeout_ms // 5):
        APP.processEvents()
        if done():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


class _Library:
    def __init__(self, row: TrackRow):
        self.row = row

    def read(self, job, *args, done):
        if job is jobs.tag_values:
            done((self.row, ""))

    def call(self, job, *args, **kwargs):
        return {} if job is jobs.tag_sources else None


class WriteTagsTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.app = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.app, ignore_errors=True)
        for patcher in (mock.patch.object(tag_backup, "app_folder", return_value=self.app),
                        mock.patch.object(ExportTrackPanel, "_backup", "tags")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.file = self.dir / "ヴヴヴ.flac"
        sf.write(self.file, (0.1 * np.sin(np.arange(22050) / 10)).astype(np.float32), 22050, format="FLAC")
        self.original = self.file.read_bytes()
        row = TrackRow(uid=UID, path=str(self.file), title="ヴヴヴ", artist="vα-liv", album="", comment="")
        self.panel = ExportTrackPanel(_Library(row))
        self.panel.set_track(row)
        for field, box in self.panel.tag_fields.items():
            box.setChecked(field in ("title", "artist"))
        self.shown = []

    def dialog(self, accept: bool, backup: str, uncheck: tuple = ()):
        def exec_(dialog):
            self.shown.append(dialog)
            for kind, radio in dialog._choices.items():
                radio.setChecked(kind == backup)
            for box in dialog.findChildren(QtWidgets.QCheckBox):
                if box.text() in uncheck:
                    box.setChecked(False)
            return QtWidgets.QDialog.Accepted if accept else QtWidgets.QDialog.Rejected
        return mock.patch.object(export_tab.TagWriteDialog, "exec", exec_)

    def write(self, accept: bool = True, backup: str = "tags", uncheck: tuple = ()) -> None:
        with self.dialog(accept, backup, uncheck):
            self.panel.write_tags()
            wait_until(lambda: self.panel._tags_writing is None)

    def test_confirmed_written_with_a_tag_backup_and_put_back(self):
        self.write(backup="tags")
        (dialog,) = self.shown
        self.assertEqual(dialog.changes, [("title", "Title", [("vorbis:TITLE", "TITLE", "")], "ヴヴヴ", ""),
                                          ("artist", "Artist", [("vorbis:ARTIST", "ARTIST", "")], "vα-liv", "")])
        boxes = dialog.findChildren(QtWidgets.QCheckBox)
        self.assertEqual([(b.text(), b.isChecked()) for b in boxes], [("TITLE", True), ("ARTIST", True)])
        texts = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
        self.assertEqual([t for t in texts if t in ("(empty)", "ヴヴヴ", "vα-liv")],
                         ["(empty)", "ヴヴヴ", "(empty)", "vα-liv"])         # each change whole, the file's above
        self.assertEqual({k: read_tags(self.file).get(k) for k in ("title", "artist")},
                         {"title": "ヴヴヴ", "artist": "vα-liv"})
        (saved,) = list((self.app / "backups" / "tags").rglob("*.mxtags"))
        note = self.panel.tag_note.text()
        self.assertTrue(note.replace("​", "").endswith(f"\nIts backup: {saved}"))     # on a line of its own
        self.assertIn(f"\\​{saved.name}", note)                                    # breakable at its separators
        self.assertEqual(self.panel.tag_note.toolTip(), str(saved))
        with mock.patch.object(QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(saved), "")), \
                mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            self.panel.restore_tags()
            wait_until(lambda: self.panel._tags_writing is None)
        self.assertEqual(self.file.read_bytes(), self.original)

    def test_a_whole_file_backup(self):
        self.write(backup="file")
        (saved,) = list((self.app / "backups" / "tags").rglob("*.flac"))
        self.assertEqual(saved.read_bytes(), self.original)
        self.assertEqual(ExportTrackPanel._backup, "file")     # the next confirmation's choice

    def test_cancelled_nothing_written(self):
        self.write(accept=False)
        self.assertEqual(len(self.shown), 1)
        self.assertEqual(self.file.read_bytes(), self.original)
        self.assertFalse((self.app / "backups").exists())

    def test_every_place_written_shown_and_the_formats_own(self):
        # a store's M4A: its comment blank in ©cmt (what rekordbox reads) and held in desc (what the library reads):
        # here FLAC's COMMENT and DESCRIPTION. The confirmation shows both places by name, warns that the format's
        # own place is written too; the same value in the library still writes it there
        tw.write_tags(self.file, {"comment": "JASRAC / ASOBINOTES"}, {"comment": ["vorbis:DESCRIPTION"]})
        tw.write_tags(self.file, {"comment": " "}, {"comment": ["vorbis:COMMENT"]})
        for field, box in self.panel.tag_fields.items():
            box.setChecked(field == "comment")
        self.panel._library.row = TrackRow(uid=UID, path=str(self.file), comment="JASRAC / ASOBINOTES [ヴヴヴ OP]")
        self.write(accept=False)
        warning = "Also written to COMMENT, the field DJ software (rekordbox ...) reads. It holds now: (empty)"
        places = [("vorbis:COMMENT", "COMMENT", ""), ("vorbis:DESCRIPTION", "DESCRIPTION", "JASRAC / ASOBINOTES")]
        self.assertEqual(self.shown[0].changes,
                         [("comment", "Comment", places, "JASRAC / ASOBINOTES [ヴヴヴ OP]", warning)])
        self.assertIn(warning, [label.text() for label in self.shown[0].findChildren(QtWidgets.QLabel)])
        self.assertEqual([b.text() for b in self.shown[0].findChildren(QtWidgets.QCheckBox)],
                         ["COMMENT", "DESCRIPTION"])
        self.panel._library.row = TrackRow(uid=UID, path=str(self.file), comment="JASRAC / ASOBINOTES")
        self.write()
        self.assertEqual(self.shown[1].changes, [("comment", "Comment", places, "JASRAC / ASOBINOTES", warning)])
        self.assertEqual(tw.read_spellings(self.file)["vorbis:COMMENT"], ["JASRAC / ASOBINOTES"])
        self.write()                                    # both places hold it now: nothing to confirm
        self.assertEqual(len(self.shown), 2)
        self.assertEqual(self.panel.tag_note.text(), "The file's tags already hold these values.")

    def test_an_unchecked_place_left_as_it_is(self):
        tw.write_tags(self.file, {"comment": "JASRAC / ASOBINOTES"}, {"comment": ["vorbis:DESCRIPTION"]})
        tw.write_tags(self.file, {"comment": " "}, {"comment": ["vorbis:COMMENT"]})
        for field, box in self.panel.tag_fields.items():
            box.setChecked(field in ("title", "comment"))
        self.panel._library.row = TrackRow(uid=UID, path=str(self.file), title="ヴヴヴ",
                                           comment="JASRAC / ASOBINOTES [ヴヴヴ OP]")
        self.write(uncheck=("COMMENT", "TITLE"))
        found = tw.read_spellings(self.file)
        self.assertEqual((found["vorbis:COMMENT"], found["vorbis:DESCRIPTION"]),
                         ([" "], ["JASRAC / ASOBINOTES [ヴヴヴ OP]"]))
        self.assertNotIn("vorbis:TITLE", found)                 # its one place unchecked: the field not written

    def test_nothing_checked_no_write(self):
        with self.dialog(True, ""):
            self.panel.write_tags()
            wait_until(lambda: bool(self.shown))
        dialog = self.shown[0]
        write = dialog.findChild(QtWidgets.QDialogButtonBox).buttons()
        accept = next(b for b in write if b.text() == "Write Tags")
        for box in dialog.findChildren(QtWidgets.QCheckBox):
            box.setChecked(False)
        self.assertFalse(accept.isEnabled())
        dialog.findChildren(QtWidgets.QCheckBox)[0].setChecked(True)
        self.assertTrue(accept.isEnabled())
        wait_until(lambda: self.panel._tags_writing is None)

    def test_nothing_changing_no_confirmation(self):
        self.write(backup="")
        self.shown.clear()
        written = self.file.read_bytes()
        self.write()
        self.assertEqual(self.shown, [])
        self.assertEqual(self.file.read_bytes(), written)
        self.assertEqual(self.panel.tag_note.text(), "The file's tags already hold these values.")


if __name__ == "__main__":
    unittest.main()
