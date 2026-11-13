"""The Export tab's DAW Tempo Map (ui.library.tabs.export): the track's stored marks written as a MIDI file into the
destination folder (on a thread of its own), also when asked before its analysis came (the shortcut right after the
track is chosen); a track without a beatgrid said, nothing written; a file there overwritten only when the user says
so; at the tempo changed (in % or as a target BPM from the track's BPM) or the track's own."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from unit.ui_tests.qt_env import APP, isolated  # noqa: F401

from PySide6 import QtCore, QtWidgets

from core.analysis_library.stored import StoredAnalysis
from core.library.records import TrackRow
from ui.library.tabs.export import ExportTrackPanel

_ISOLATION = None
UID = "0f8fad5b-d9cb-469f-a165-70867728950e"


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


class _Library:
    """The library's owner: an analysis read answered when ``answer`` is called (as the library thread would)."""

    def __init__(self, analysis):
        self.analysis, self.waiting = analysis, []

    def read(self, _job, _uid, done):
        self.waiting.append(done)

    def answer(self):
        waiting, self.waiting = self.waiting, []
        for done in waiting:
            done(self.analysis)


def finish(panel: ExportTrackPanel) -> None:
    """Until what the panel is making (on its thread) is written and said."""
    deadline = time.monotonic() + 20.0
    while panel._busy and time.monotonic() < deadline:
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
        time.sleep(0.005)
    assert not panel._busy, "the export did not finish"


def tempos(data: bytes) -> list[float]:
    """The file's tempo events in BPM."""
    out, pos = [], 0
    while (pos := data.find(b"\xff\x51\x03", pos)) >= 0:
        out.append(60e6 / int.from_bytes(data[pos + 3:pos + 6], "big"))
        pos += 6
    return out


def analysis(beats) -> StoredAnalysis:
    return StoredAnalysis(duration_sec=40.0, beats=np.asarray(beats, dtype=float),
                          tempo_segments=np.asarray([[0.0, 32.0, 120.0, 0.0, 4.0, 120.0]]),
                          key_segments=np.asarray([[9.0, 1.0, 0.0, 32.0]]))


class ExportTempoMapTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        boxes = mock.patch.multiple(QtWidgets.QMessageBox, information=mock.DEFAULT, warning=mock.DEFAULT,
                                    critical=mock.DEFAULT, question=mock.DEFAULT)
        self.boxes = boxes.start()
        self.addCleanup(boxes.stop)

    def panel(self, beats, bpm=None, key=None) -> tuple[ExportTrackPanel, _Library]:
        library = _Library(analysis(beats))
        panel = ExportTrackPanel(library)
        panel.set_track(TrackRow(uid=UID, path=str(Path(self.dir.name) / "ヴヴヴ.flac"), title="ヴヴヴ", bpm=bpm,
                                 key=key))
        panel.dest_edit.setText(self.dir.name)
        return panel, library

    def test_written_when_the_analysis_comes(self):
        panel, library = self.panel(0.5 * np.arange(64))
        panel.export_tempo_map()                # asked before the analysis came
        out = Path(self.dir.name) / "ヴヴヴ_tempo_map.mid"
        self.assertFalse(out.exists())
        library.answer()
        finish(panel)
        self.assertTrue(out.read_bytes().startswith(b"MThd"))
        self.boxes["information"].assert_called_once()

    def test_no_beatgrid(self):
        panel, library = self.panel([])
        library.answer()
        panel.export_tempo_map()
        finish(panel)
        self.boxes["warning"].assert_called_once()
        self.assertEqual(list(Path(self.dir.name).iterdir()), [])

    def test_at_the_changed_tempo_or_the_tracks(self):
        panel, library = self.panel(0.5 * np.arange(64), bpm=120.0)
        library.answer()
        panel.spn_bpm.setValue(150.0)           # the target BPM: +25 % of the track's stored 120
        self.assertAlmostEqual(panel.spn_tempo.value(), 25.0)
        self.assertEqual(panel.sld_tempo.value(), 2500)
        panel.export_tempo_map()
        finish(panel)
        changed = Path(self.dir.name) / "ヴヴヴ_tempo_map_150BPM.mid"      # named by the target BPM
        self.assertAlmostEqual(tempos(changed.read_bytes())[0], 150.0, places=2)
        panel.chk_map_tempo.setChecked(False)   # the track's own tempo
        panel.export_tempo_map()
        finish(panel)
        self.assertAlmostEqual(tempos((Path(self.dir.name) / "ヴヴヴ_tempo_map.mid").read_bytes())[0], 120.0,
                               places=2)
        # the deck's WIDE range: +100 % at most; the slider's bottom the least change, never a standstill
        panel.sld_tempo.setValue(panel.sld_tempo.maximum())
        self.assertAlmostEqual(panel.speed(), 2.0)
        panel.sld_tempo.setValue(panel.sld_tempo.minimum())
        self.assertAlmostEqual(panel.spn_tempo.value(), -99.99)
        self.assertGreater(panel.speed(), 0.0)
        self.assertEqual(panel.spn_tempo.text(), "-99.99 %")

    def test_the_key_shift_and_the_target_key(self):
        panel, library = self.panel(0.5 * np.arange(64), key=21)          # A minor (8A)
        library.answer()
        self.assertEqual(panel.btn_target_key.key(), 21)
        panel.spn_key.setValue(2.0)
        self.assertEqual(panel.sld_key.value(), 200)
        self.assertEqual(panel.btn_target_key.key(), 23)                  # B minor
        panel.sld_key.setValue(-40)                                       # a fraction: the nearest semitone's key
        self.assertAlmostEqual(panel.key_shift(), -0.4)
        self.assertEqual(panel.btn_target_key.key(), 21)
        panel.btn_target_key.setKey(16)                                   # E minor: 5 down, not 7 up
        self.assertAlmostEqual(panel.key_shift(), -5.0)
        panel.btn_target_key.setKey(15)                                   # D# minor, a tritone: down, as it was
        self.assertAlmostEqual(panel.key_shift(), -6.0)
        panel.spn_key.setValue(0.0)
        panel.btn_target_key.setKey(15)                                   # from the track's own: up
        self.assertAlmostEqual(panel.key_shift(), 6.0)
        panel.sld_key.setValue(panel.sld_key.maximum())
        self.assertAlmostEqual(panel.key_shift(), 6.0)                    # the deck's KEY: ±6
        self.assertEqual(panel.spn_key.text(), "+6.00 semitones")
        # the key aimed at is the track's mode's: the popup shows only the minor keys' row
        popup = panel.btn_target_key._popup
        popup.show_for(panel.btn_target_key, 21, True)
        self.assertEqual({idx >= 12 for idx, cell in popup._cells.items() if cell.isVisibleTo(popup)}, {True})
        popup.hide()

    def test_a_track_without_a_key(self):
        panel, library = self.panel(0.5 * np.arange(64))
        library.answer()
        self.assertFalse(panel.btn_target_key.isEnabled())
        panel.spn_key.setValue(-2.0)                                      # shifted all the same
        self.assertAlmostEqual(panel.key_shift(), -2.0)

    def test_at_the_changed_tempo_and_key_or_the_tracks(self):
        panel, library = self.panel(0.5 * np.arange(64), bpm=120.0, key=21)
        library.answer()
        panel.spn_bpm.setValue(150.0)
        panel.spn_key.setValue(2.0)
        panel.export_tempo_map()
        finish(panel)
        both = (Path(self.dir.name) / "ヴヴヴ_tempo_map_150BPM_key+2.mid").read_bytes()   # named by both changes
        self.assertAlmostEqual(tempos(both)[0], 150.0, places=2)
        self.assertIn("Bm (10A)".encode(), both)                          # A minor two semitones up
        self.assertNotIn("Am (8A)".encode(), both)
        panel.chk_map_key.setChecked(False)                               # the track's own keys
        panel.export_tempo_map()
        finish(panel)
        tempo_only = (Path(self.dir.name) / "ヴヴヴ_tempo_map_150BPM.mid").read_bytes()
        self.assertIn("Am (8A)".encode(), tempo_only)
        panel.chk_map_key.setChecked(True)
        panel.chk_map_tempo.setChecked(False)                             # the key alone
        panel.spn_key.setValue(-1.5)                                      # its keys the nearest semitone's
        panel.export_tempo_map()
        finish(panel)
        key_only = (Path(self.dir.name) / "ヴヴヴ_tempo_map_key-1.5.mid").read_bytes()
        self.assertAlmostEqual(tempos(key_only)[0], 120.0, places=2)
        self.assertIn("Gm (6A)".encode(), key_only)                      # -1.5 to -2: G minor

    def test_a_file_there_kept_unless_overwritten(self):
        panel, library = self.panel(0.5 * np.arange(64))
        library.answer()
        out = Path(self.dir.name) / "ヴヴヴ_tempo_map.mid"
        out.write_bytes(b"kept")
        self.boxes["question"].return_value = QtWidgets.QMessageBox.StandardButton.No
        panel.export_tempo_map()
        self.assertEqual(out.read_bytes(), b"kept")
        self.boxes["question"].return_value = QtWidgets.QMessageBox.StandardButton.Yes
        panel.export_tempo_map()
        finish(panel)
        self.assertTrue(out.read_bytes().startswith(b"MThd"))


if __name__ == "__main__":
    unittest.main()
