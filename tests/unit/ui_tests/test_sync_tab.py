"""The Sync tab (ui.library.tabs.sync): Write Tags, each change confirmed first (the file's value and the one
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

from analyzer.music.marks import CuePoint, JumpCue, JumpCueGroup
from core.analysis_library import jobs as analysis_jobs
from core.analysis_library.stored import StoredAnalysis
from core.library import jobs
from core.library.records import TrackRow
from third_party import serato
from core.tags import backup as tag_backup
from core.tags import writer as tw
from core.tags.writer import read_tags
from ui.library.tabs import sync as sync_tab
from ui.library.tabs.sync import SyncTrackPanel
from ui.theme import current_theme

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
        self.analysis = None     # the stored analysis answered (None: none read)

    def read(self, job, *args, done):
        if job is jobs.tag_values:
            done((self.row, ""))
        elif job is analysis_jobs.analysis and self.analysis is not None:
            done(self.analysis)

    def call(self, job, *args, **kwargs):
        return {} if job is jobs.tag_sources else None


class WriteTagsTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.app = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.app, ignore_errors=True)
        for patcher in (mock.patch.object(tag_backup, "app_folder", return_value=self.app),
                        mock.patch.object(SyncTrackPanel, "_backup", "tags")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.file = self.dir / "ヴヴヴ.flac"
        sf.write(self.file, (0.1 * np.sin(np.arange(22050) / 10)).astype(np.float32), 22050, format="FLAC")
        self.original = self.file.read_bytes()
        row = TrackRow(uid=UID, path=str(self.file), title="ヴヴヴ", artist="vα-liv", album="", comment="")
        self.panel = SyncTrackPanel(_Library(row))
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
        return mock.patch.object(sync_tab.TagWriteDialog, "exec", exec_)

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
        self.assertEqual(SyncTrackPanel._backup, "file")     # the next confirmation's choice

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

    def test_serato_beatgrid_and_hot_cues(self):
        cue = lambda label, point, comment="": JumpCue(label, comment, point, point, point)   # noqa: E731
        period = 60 / 174
        beats = 0.37 + period * np.arange(64)
        self.panel._library.analysis = StoredAnalysis(
            duration_sec=23.0, beats=beats, tempo_segments=np.array([[0.37, beats[-1], 174, 0.37 + period, 3, 174]]),
            jump_cues=(JumpCueGroup((cue("A", 2.5, "サビ"), cue("B", 10.0), cue("I", 12.0)), 0),))
        for box in self.panel.tag_fields.values():
            box.setChecked(False)
        self.panel.cb_serato.setChecked(True)
        self.write()
        (dialog,) = self.shown
        changes = {field: (olds, new, warning) for field, _name, olds, new, warning in dialog.changes}
        self.assertEqual(set(changes), {"serato_beatgrid", "serato_tempo_map", "serato_markers"})
        self.assertEqual(changes["serato_beatgrid"][1], "1 grid marker from 0:00.715, 174.00 BPM at the end")
        self.assertEqual(changes["serato_tempo_map"][1], "1 tempo segment from 0:00.715, 174.00 BPM at the end")
        self.assertEqual(changes["serato_markers"][1], "A 0:02.500 サビ; B 0:10.000 B · BPM locked")
        self.assertIn("past H", changes["serato_markers"][2])                     # the JumpCUE I left out
        self.assertIn("4-beat bars", changes["serato_beatgrid"][2])               # 3/4 bars
        got = read_tags(self.file)
        markers = serato.parse_beatgrid(got["serato_beatgrid"].encode("latin-1"))
        self.assertAlmostEqual(markers[0][0], 0.37 + period, places=4)
        self.assertEqual([(c.slot, c.name) for c in serato.parse_cues(got["serato_markers"].encode("latin-1"))],
                         [(0, "サビ"), (1, "B")])
        self.shown.clear()
        self.write()                                                              # written: nothing to confirm
        self.assertEqual(self.shown, [])
        # Serato analyzed it since (its own Tempo Map, read before the beatgrid: 15 segments at half the tempo):
        # ours written over it
        own = serato.build_tempo_map([(0.19, 52.0), (24.5, 76.0), (60.0, 102.6)])
        tw.write_tags(self.file, {"serato_tempo_map": own})
        self.write()
        (olds, new, _warning), = [(o, n, w) for f, _name, o, n, w in self.shown[0].changes if f == "serato_tempo_map"]
        self.assertEqual(olds[0][2], "3 tempo segments from 0:00.190, 102.60 BPM at the end")
        anchor, segments = serato.parse_tempo_map(read_tags(self.file)["serato_tempo_map"].encode("latin-1"))
        self.assertAlmostEqual(anchor, 0.37 + period, places=6)
        self.assertEqual(len(segments), 1)

    def test_serato_options(self):
        cue = lambda label, point: JumpCue(label, "", point, point, point)   # noqa: E731
        beats = 0.5 * np.arange(64)
        self.panel._library.analysis = StoredAnalysis(
            duration_sec=32.0, beats=beats, tempo_segments=np.array([[0.0, 32.0, 120, 0.0, 4, 120]]),
            jump_cues=(JumpCueGroup((cue("A", 2.0), cue("B", 8.0)), 0),),
            cue_points=(CuePoint(4.0, "Intro"), CuePoint(16.0, "Drop")))
        for box in self.panel.tag_fields.values():
            box.setChecked(False)
        self.panel.cb_serato.setChecked(True)
        lib = mock.Mock(serato_hot_cues="both", serato_keep_cues=False, serato_bpm_lock=False)
        with mock.patch.object(sync_tab.settings_store, "current", return_value=mock.Mock(libconfig=lib)):
            self.write()
        raw = read_tags(self.file)["serato_markers"].encode("latin-1")
        self.assertEqual([(c.slot, c.name) for c in serato.parse_cues(raw)],
                         [(0, "A"), (1, "B"), (2, "Intro"), (3, "Drop")])
        self.assertFalse(serato.bpm_locked(raw))

    def test_an_edited_analysis_read_again_and_kept_hot_cues_said(self):
        # A-D written; then D removed in the editor (the stored analysis changed): read again, not the first one's;
        # with the file's hot cues kept, the D written before stays, which the note says
        cue = lambda label, point: JumpCue(label, "", point, point, point)   # noqa: E731
        beats = 0.5 * np.arange(64)
        stored = lambda labels: StoredAnalysis(   # noqa: E731
            duration_sec=32.0, beats=beats, tempo_segments=np.array([[0.0, 32.0, 120, 0.0, 4, 120]]),
            jump_cues=(JumpCueGroup(tuple(cue(x, 2.0 + 4 * i) for i, x in enumerate(labels)), 0),))
        for box in self.panel.tag_fields.values():
            box.setChecked(False)
        self.panel.cb_serato.setChecked(True)
        self.panel._library.analysis = stored("ABCD")
        self.write()
        self.panel._library.analysis = stored("ABC")
        lib = mock.Mock(serato_hot_cues="jumpcues", serato_keep_cues=True, serato_bpm_lock=True)
        with mock.patch.object(sync_tab.settings_store, "current", return_value=mock.Mock(libconfig=lib)):
            self.write()
        self.assertIn("The file's hot cues D stay", self.panel.tag_note.text())
        lib.serato_keep_cues = False
        with mock.patch.object(sync_tab.settings_store, "current", return_value=mock.Mock(libconfig=lib)):
            self.write()
        raw = read_tags(self.file)["serato_markers"].encode("latin-1")
        self.assertEqual([c.slot for c in serato.parse_cues(raw)], [0, 1, 2])

    def test_a_file_held_by_another_program_then_written(self):
        # Windows refuses to replace a file another program holds (Serato with the track loaded): said; the next
        # write's note back in the text's colour
        with mock.patch.object(sync_tab, "write_tags", side_effect=PermissionError(13, "Access is denied")):
            self.write(backup="")
        self.assertIn("Another program has the file open", self.panel.tag_note.text())
        self.assertIn(current_theme().danger, self.panel.tag_note.styleSheet())
        self.assertEqual(self.file.read_bytes(), self.original)
        self.write(backup="")
        self.assertEqual(self.panel.tag_note.text(), "The tags were written to the file.")
        self.assertIn(current_theme().text, self.panel.tag_note.styleSheet())
        self.assertNotIn(current_theme().danger, self.panel.tag_note.styleSheet())

    def test_nothing_changing_no_confirmation(self):
        self.write(backup="")
        self.shown.clear()
        written = self.file.read_bytes()
        self.write()
        self.assertEqual(self.shown, [])
        self.assertEqual(self.file.read_bytes(), written)
        self.assertEqual(self.panel.tag_note.text(), "The file's tags already hold these values.")


class RekordboxSyncTests(unittest.TestCase):
    """Sync to Rekordbox XML: the request an edit's save sends, its outcome shown; off without the setting."""

    def setUp(self):
        from core.workflows.event_bus import EventBus
        self.bus = EventBus()
        self.asked = []
        self.bus.sig_rekordbox_sync_track_requested.connect(self.asked.append)
        self.lib = mock.Mock(rekordbox_sync_enabled=True, rekordbox_xml_path="rekordbox.xml")
        patcher = mock.patch.object(sync_tab.settings_store, "current", return_value=mock.Mock(libconfig=self.lib))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.row = TrackRow(uid=UID, path="ヴヴヴ.flac", title="ヴヴヴ")
        self.panel = SyncTrackPanel(_Library(self.row), bus=self.bus)
        self.panel.set_track(self.row)

    def request(self, uid=UID):
        return mock.Mock(mode="track", uid=uid, xml_path=Path("rekordbox.xml"))

    def test_the_tracks_request_and_its_outcome(self):
        self.assertTrue(self.panel.btn_rekordbox_sync.isEnabled())
        self.panel.sync_rekordbox()
        self.assertEqual(self.asked, [UID])
        self.bus.sig_rekordbox_sync_finished.emit(self.request(), mock.Mock(output_path=Path("rekordbox.xml")))
        self.assertTrue(self.panel.rekordbox_note.text().startswith("Written into rekordbox"))
        # a sync the tab did not ask for (an edit saved) shown nowhere here
        self.bus.sig_rekordbox_sync_failed.emit(self.request(), "disk full")
        self.assertNotIn("disk full", self.panel.rekordbox_note.text())
        self.panel.sync_rekordbox()
        self.bus.sig_rekordbox_sync_failed.emit(self.request(), "disk full")
        self.assertIn("disk full", self.panel.rekordbox_note.text())
        self.assertIn(current_theme().danger, self.panel.rekordbox_note.styleSheet())

    def test_off_without_the_setting(self):
        self.lib.rekordbox_sync_enabled = False
        self.panel.set_track(self.row)
        self.assertFalse(self.panel.btn_rekordbox_sync.isEnabled())
        self.assertIn("Settings > Library > Rekordbox XML", self.panel.rekordbox_note.text())
        self.panel.sync_rekordbox()
        self.assertEqual(self.asked, [])
        self.lib.rekordbox_sync_enabled = True
        self.panel.set_track(self.row)
        self.assertTrue(self.panel.btn_rekordbox_sync.isEnabled())
        self.assertEqual(self.panel.rekordbox_note.text(), "")


if __name__ == "__main__":
    unittest.main()
