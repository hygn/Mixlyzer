"""The main window's question before edits not saved are lost (app.window.AppWindow._with_edits_resolved, the loads,
the unload, a reanalysis of the deck's track, an External Sync load, quitting): none asked without edits not saved;
Save runs what was asked once the edits are stored (nothing when no save could start), Discard runs it at once,
Cancel nothing; an External Sync load has no Cancel and takes the last track asked while the box is open. A stand-in
window: the question's box answered by a mock, the editor's save a stub."""
import time
import unittest
from unittest import mock

from unit.ui_tests.qt_env import APP  # noqa: F401  (offscreen Qt, one QApplication)

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QMessageBox

from app.window import AppWindow


class _Editor:
    def __init__(self, can_save=True):
        self.can_save = can_save
        self.pending = None
        self.failed = None
        self.saving = False

    def save_then(self, after, failed=None):
        if not self.can_save:
            return False
        self.pending, self.failed, self.saving = after, failed, True
        return True

    def finish(self):
        after, self.pending, self.saving = self.pending, None, False
        after()

    def fail(self):
        failed, self.pending, self.failed, self.saving = self.failed, None, None, False
        if failed is not None:
            failed()


class _Window(QtWidgets.QWidget):
    """What _with_edits_resolved and the guarded paths read of the window."""
    _with_edits_resolved = AppWindow._with_edits_resolved
    _load_track_from_external_sync = AppWindow._load_track_from_external_sync
    _external_load_waiting = AppWindow._external_load_waiting

    def __init__(self, unsaved=True, can_save=True):
        super().__init__()
        self.model = mock.Mock(unsaved=unsaved)
        self.pane = mock.Mock(track_edit=_Editor(can_save))
        self._asking_edits = False
        self._external_pending = None
        self._edits_resolved_for_close = False
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

    def test_external_sync_load_during_another_box_asked_after_it(self):
        w = _Window()
        with answering(QMessageBox.Cancel, lambda box: w._load_track_from_external_sync("b.flac", 2.0)):
            w._with_edits_resolved(lambda: None)                  # quitting: Cancel, the window stays
        self.assertEqual(w.loaded, [])
        asked = []
        with answering(QMessageBox.Discard, lambda box: asked.append(box.informativeText())):
            drain()                                               # that load asked for now (no Cancel)
        self.assertEqual(len(asked), 1)
        self.assertEqual(w.loaded, ["b.flac"])
        w.player.seek.assert_called_once_with(2.0)

    def test_external_sync_load_whose_save_failed_asked_again(self):
        w = _Window()
        with answering(QMessageBox.Save):
            w._load_track_from_external_sync("a.flac", 1.0)
        self.assertTrue(w.pane.track_edit.saving)
        w.pane.track_edit.fail()                                  # the edits kept, not saved
        self.assertEqual(w.loaded, [])
        with answering(QMessageBox.Discard):
            drain()
        self.assertEqual(w.loaded, ["a.flac"])


def drain() -> None:
    """The events posted (the single-shot timers of 0 ms) run."""
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        APP.processEvents(QtCore.QEventLoop.AllEvents, 20)
        time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
