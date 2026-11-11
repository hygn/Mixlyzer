"""The main window's question before edits not saved are lost (app.window.AppWindow._with_edits_resolved, the loads,
the unload, a reanalysis of the deck's track, quitting): none asked without edits not saved; Save runs what was asked
once the edits are stored (nothing when no save could start), Discard runs it at once, Cancel nothing. A stand-in
window: the question's box answered by a mock, the editor's save a stub."""
import unittest
from unittest import mock

from unit.ui_tests.qt_env import APP  # noqa: F401  (offscreen Qt, one QApplication)

from PySide6 import QtWidgets
from PySide6.QtWidgets import QMessageBox

from app.window import AppWindow, _Edits


class _Editor:
    def __init__(self, can_save=True):
        self.can_save = can_save
        self.pending = None
        self.saving = False

    def save_then(self, after):
        if not self.can_save:
            return False
        self.pending, self.saving = after, True
        return True

    def finish(self):
        after, self.pending, self.saving = self.pending, None, False
        after()


class _Window(QtWidgets.QWidget):
    """What _with_edits_resolved reads of the window."""
    _with_edits_resolved = AppWindow._with_edits_resolved

    def __init__(self, unsaved=True, can_save=True):
        super().__init__()
        self.model = mock.Mock(unsaved=unsaved)
        self.pane = mock.Mock(track_edit=_Editor(can_save))
        self._edits = _Edits.IDLE
        self.messages = []

    def statusBar(self):
        return mock.Mock(showMessage=self.messages.append)


def answering(button):
    """QMessageBox.exec answering ``button``."""
    return mock.patch.object(QMessageBox, "exec", lambda box: button)


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
        self.assertIs(w._edits, _Edits.IDLE)

    def test_a_save_that_cannot_start_does_nothing(self):
        w = _Window(can_save=False)
        ran = []
        with answering(QMessageBox.Save):
            w._with_edits_resolved(lambda: ran.append(1))
        self.assertEqual(ran, [])
        self.assertEqual(len(w.messages), 1)


if __name__ == "__main__":
    unittest.main()
