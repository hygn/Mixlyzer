"""The Export tab's JAMS (ui.library.tabs.export): the track's stored marks and its audio file's identity written as
a JAMS file into the destination folder (on a thread of its own), also when asked before its analysis came; a
missing audio file or one FFmpeg cannot read said, nothing written; the file's path and the track's uid each left out
when unchecked, the checks kept in deckconfig (deck.json); a file there overwritten only when the user says so. The audio's hash is analyzer.audio.decoder's
(tests/test_audio_packet_hash.py), patched here."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from unit.ui_tests.qt_env import APP, isolated  # noqa: F401

from PySide6 import QtWidgets

from analyzer.audio.decoder import AudioPacketHash
from analyzer.music.marks import CuePoint, Phrase
from core.analysis_library.stored import StoredAnalysis
from core.library.records import TrackRow
from core.settings.config import default_cfg
from third_party.jams import FORMAT
from ui.library.tabs import export as export_tab
from ui.library.tabs.export import ExportTrackPanel
from unit.ui_tests.test_export_tempo_map import UID, _Library, finish

_ISOLATION = None
AUDIO = AudioPacketHash("cd" * 32, 10, "flac", 44100, "stereo", "n7.1-test")


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


class ExportJamsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        boxes = mock.patch.multiple(QtWidgets.QMessageBox, information=mock.DEFAULT, warning=mock.DEFAULT,
                                    critical=mock.DEFAULT, question=mock.DEFAULT)
        self.boxes = boxes.start()
        self.addCleanup(boxes.stop)
        hashed = mock.patch("ui.library.tabs.export.audio_packet_hash", return_value=AUDIO)
        self.hashed = hashed.start()
        self.addCleanup(hashed.stop)
        self.cfg = default_cfg()       # the settings the tab keeps its choices in (no deck.json written)
        settings = mock.patch.object(export_tab.settings_store, "current", return_value=self.cfg)
        settings.start()
        self.addCleanup(settings.stop)
        self.source = Path(self.dir.name) / "ヴヴヴ.flac"
        self.source.write_bytes(b"0123456789")
        self.out = Path(self.dir.name) / "ヴヴヴ.jams"

    def panel(self) -> tuple[ExportTrackPanel, _Library]:
        library = _Library(StoredAnalysis(
            duration_sec=40.0, beats=0.5 * np.arange(64), key_segments=np.asarray([[9.0, 1.0, 0.0, 32.0]]),
            tempo_segments=np.asarray([[0.0, 32.0, 120.0, 0.0, 4.0, 120.0]]),
            phrases=(Phrase(0.0, 16.0, "INTRO"), Phrase(16.0, 32.0, "CHORUS")), cue_points=(CuePoint(16.0, "C"),)))
        panel = ExportTrackPanel(library)
        panel.set_track(TrackRow(uid=UID, path=str(self.source), title="ヴヴヴ", artist="ア", total_samples=1764000))
        panel.dest_edit.setText(self.dir.name)
        return panel, library

    def test_written_when_the_analysis_comes(self):
        panel, library = self.panel()
        panel.export_jams()                       # asked before the analysis came
        self.assertFalse(self.out.exists())
        library.answer()
        finish(panel)
        self.boxes["information"].assert_called_once()
        self.hashed.assert_called_once_with(str(self.source))
        doc = json.loads(self.out.read_text(encoding="ascii"))
        self.assertEqual(doc["file_metadata"]["title"], "ヴヴヴ")
        by = {a["namespace"]: a for a in doc["annotations"]}
        self.assertEqual(len(by["beat"]["data"]), 64)
        self.assertEqual([o["value"] for o in by["key_mode"]["data"]], ["A:minor"])
        self.assertEqual([o["value"] for o in by["segment_open"]["data"]], ["INTRO", "CHORUS"])
        mx = doc["sandbox"]["mixlyzer"]
        self.assertEqual((mx["format"], mx["uid"]), (FORMAT, UID))
        self.assertEqual(mx["file"]["path"], str(self.source))
        self.assertEqual(mx["file"]["size"], 10)
        self.assertEqual((mx["file"]["audio"]["packets_sha256"], mx["file"]["audio"]["samples"]),
                         ("cd" * 32, 1764000))

    def test_path_and_uid_each_left_out_when_unchecked(self):
        panel, library = self.panel()
        library.answer()
        panel.chk_jams_path.setChecked(False)
        panel.export_jams()
        finish(panel)
        mx = json.loads(self.out.read_text(encoding="ascii"))["sandbox"]["mixlyzer"]
        self.assertNotIn("path", mx["file"])
        self.assertEqual(mx["uid"], UID)
        panel.chk_jams_path.setChecked(True)
        panel.chk_jams_uid.setChecked(False)
        self.boxes["question"].return_value = QtWidgets.QMessageBox.StandardButton.Yes
        panel.export_jams()
        finish(panel)
        mx = json.loads(self.out.read_text(encoding="ascii"))["sandbox"]["mixlyzer"]
        self.assertNotIn("uid", mx)
        self.assertEqual(mx["file"]["path"], str(self.source))

    def test_the_choices_kept_for_the_next_start(self):
        panel, _library = self.panel()
        self.assertTrue(panel.chk_jams_path.isChecked() and panel.chk_jams_uid.isChecked())    # the defaults
        panel.chk_jams_path.setChecked(False)
        deck = self.cfg.deckconfig
        self.assertEqual((deck.export_jams_path, deck.export_jams_uid), (False, True))
        again, _library = self.panel()
        self.assertFalse(again.chk_jams_path.isChecked())
        self.assertTrue(again.chk_jams_uid.isChecked())

    def test_no_audio_file_nothing_written(self):
        panel, library = self.panel()
        library.answer()
        self.source.unlink()
        panel.export_jams()
        self.boxes["warning"].assert_called_once()
        self.assertFalse(self.out.exists())
        self.assertTrue(panel.btn_jams.isEnabled())

    def test_audio_not_read_said(self):
        panel, library = self.panel()
        library.answer()
        self.hashed.return_value = None
        panel.export_jams()
        finish(panel)
        self.boxes["warning"].assert_called_once()
        self.assertFalse(self.out.exists())
        self.assertTrue(panel.btn_jams.isEnabled())

    def test_a_file_there_kept_unless_overwritten(self):
        panel, library = self.panel()
        library.answer()
        self.out.write_bytes(b"kept")
        self.boxes["question"].return_value = QtWidgets.QMessageBox.StandardButton.No
        panel.export_jams()
        self.assertEqual(self.out.read_bytes(), b"kept")
        self.boxes["question"].return_value = QtWidgets.QMessageBox.StandardButton.Yes
        panel.export_jams()
        finish(panel)
        self.assertTrue(self.out.read_text(encoding="ascii").startswith("{"))


if __name__ == "__main__":
    unittest.main()
