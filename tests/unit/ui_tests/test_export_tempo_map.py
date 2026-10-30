"""The Export tab's DAW Tempo Map (ui.library.tabs.export): the track's stored marks written as a MIDI file into the
destination folder, also when asked before its analysis came (the shortcut right after the track is chosen); a
track without a beatgrid said, nothing written; a file there overwritten only when the user says so."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from unit.ui_tests.qt_env import APP, isolated  # noqa: F401

from PySide6 import QtWidgets

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

    def panel(self, beats) -> tuple[ExportTrackPanel, _Library]:
        library = _Library(analysis(beats))
        panel = ExportTrackPanel(library)
        panel.set_track(TrackRow(uid=UID, path=str(Path(self.dir.name) / "ヴヴヴ.flac"), title="ヴヴヴ"))
        panel.dest_edit.setText(self.dir.name)
        return panel, library

    def test_written_when_the_analysis_comes(self):
        panel, library = self.panel(0.5 * np.arange(64))
        panel.export_tempo_map()                # asked before the analysis came
        out = Path(self.dir.name) / "ヴヴヴ_tempo_map.mid"
        self.assertFalse(out.exists())
        library.answer()
        self.assertTrue(out.read_bytes().startswith(b"MThd"))
        self.boxes["information"].assert_called_once()

    def test_no_beatgrid(self):
        panel, library = self.panel([])
        library.answer()
        panel.export_tempo_map()
        self.boxes["warning"].assert_called_once()
        self.assertEqual(list(Path(self.dir.name).iterdir()), [])

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
        self.assertTrue(out.read_bytes().startswith(b"MThd"))


if __name__ == "__main__":
    unittest.main()
