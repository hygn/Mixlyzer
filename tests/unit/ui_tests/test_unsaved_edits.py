"""The main window's question before edits not saved are lost (app.window.AppWindow._with_edits_resolved, the loads,
the unload, a reanalysis of the deck's track, an External Sync load, quitting): none asked without edits not saved;
Save runs what was asked once the edits are stored (nothing when no save could start), Discard runs it at once,
Cancel nothing; an External Sync load has no Cancel and takes the last track asked while the box is open. A stand-in
window: the question's box answered by a mock, the editor's save a stub."""
import unittest
from unittest import mock

from unit.ui_tests.qt_env import APP  # noqa: F401  (offscreen Qt, one QApplication)

from PySide6 import QtWidgets
from PySide6.QtWidgets import QMessageBox

from app.window import AppWindow


class _Editor:
    def __init__(self, can_save=True):
        self.can_save = can_save
        self.pending = None

    def save_then(self, after):
        if not self.can_save:
            return False
        self.pending = after
        return True

    def finish(self):
        after, self.pending = self.pending, None
        after()


class _Window(QtWidgets.QWidget):
    """What _with_edits_resolved and the guarded paths read of the window."""
    _with_edits_resolved = AppWindow._with_edits_resolved
    _load_track_from_external_sync = AppWindow._load_track_from_external_sync

    def __init__(self, unsaved=True, can_save=True):
        super().__init__()
        self.model = mock.Mock(unsaved=unsaved)
        self.pane = mock.Mock(track_edit=_Editor(can_save))
        self._asking_edits = False
        self._external_pending = None
        self.messages = []
        self.loaded = []
        self.player = mock.Mock()

    def statusBar(self):
        return mock.Mock(showMessage=self.messages.append)

    def _track_exists_in_library(self, path):
        return True

    def _load(self, path):
        self.loaded.append(path)


def answering(button, during=None):
    """QMessageBox.exec answering ``button`` (``during(box)`` run while it is open)."""
    def exec_(box):
        if during is not None:
            during(box)
        return button
    return mock.patch.object(QMessageBox, "exec", exec_)


class UnsavedEditsTests(unittest.TestCase):
    def test_nothing_asked_without_unsaved_edits(self):
        w = _Window(unsaved=False)
        ran = []
        with mock.patch.object(QMessageBox, "exec", side_effect=AssertionError("asked")):
            w._with_edits_resolved(lambda: ran.append(1))
        self.assertEqual(ran, [1])

    def test_save_runs_it_once_stored(self):
        w = _Window()
        ran = []
        with answering(QMessageBox.Save):
            w._with_edits_resolved(lambda: ran.append(1))
        self.assertEqual(ran, [])                     # the save is running
        w.pane.track_edit.finish()
        self.assertEqual(ran, [1])

    def test_discard_runs_it_cancel_does_not(self):
        w = _Window()
        ran = []
        with answering(QMessageBox.Discard):
            w._with_edits_resolved(lambda: ran.append("discard"))
        with answering(QMessageBox.Cancel):
            w._with_edits_resolved(lambda: ran.append("cancel"))
        self.assertEqual(ran, ["discard"])

    def test_a_save_that_cannot_start_does_nothing(self):
        w = _Window(can_save=False)
        ran = []
        with answering(QMessageBox.Save):
            w._with_edits_resolved(lambda: ran.append(1))
        self.assertEqual(ran, [])
        self.assertEqual(len(w.messages), 1)

    def test_external_sync_has_no_cancel_and_takes_the_last_track(self):
        w = _Window()
        seen = []

        def during(box):
            seen.append(bool(box.standardButtons() & QMessageBox.Cancel))
            w._load_track_from_external_sync("b.flac", 3.0)        # another load while the box is open
            w._load_track_from_external_sync("c.flac", 4.0)
        with answering(QMessageBox.Discard, during):
            w._load_track_from_external_sync("a.flac", 1.0)
        self.assertEqual(seen, [False])
        self.assertEqual(w.loaded, ["c.flac"])
        w.player.seek.assert_called_once_with(4.0)

    def test_external_sync_loads_even_when_no_save_can_start(self):
        w = _Window(can_save=False)
        with answering(QMessageBox.Save):
            w._load_track_from_external_sync("a.flac", 0.0)
        self.assertEqual(w.loaded, ["a.flac"])


if __name__ == "__main__":
    unittest.main()
