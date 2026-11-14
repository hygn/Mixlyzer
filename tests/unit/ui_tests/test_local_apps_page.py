"""Settings > Library > Local Apps (offscreen Qt) on a local API whose server is not started: its status, Pair an App
(the pairing opened, an app asking shown with its permissions, a wrong code's tries left, the right code pairing it,
the dialog closed before an answer refusing it), the paired apps listed and unpaired after a confirmation."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from unit.ui_tests.qt_env import APP, drain, isolated

from PySide6 import QtCore, QtWidgets

from core.local_api.pairings import PairingStore, Phase, Verdict
from core.workflows.event_bus import EventBus
from ui.config.local_api import PairingDialog
from ui.config.window import SettingsDialog

_ISOLATION = None


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


class _Workflow(QtCore.QObject):
    ended = QtCore.Signal(str, object, str)


class _Api(QtCore.QObject):
    """The local API as the page sees it: its store, its status and its signals (no server)."""
    status_changed = QtCore.Signal()
    pairing_changed = QtCore.Signal()

    def __init__(self, store, listening=True):
        super().__init__()
        self.store = store
        self.store.on_change = self.pairing_changed.emit
        self.listening = listening

    def status(self):
        return SimpleNamespace(on=True, listening=self.listening, port=48700, error="")

    def open_pairing(self):
        self.store.open()

    def close_pairing(self):
        self.store.close()


class LocalAppsPageTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = PairingStore(os.path.join(self.dir.name, "pairings.json"))
        self.api = _Api(self.store)
        self.dialog = SettingsDialog(EventBus())
        self.dialog.set_local_api(self.api)

    def tearDown(self):
        if self.dialog._pairing_dialog is not None:
            self.dialog._pairing_dialog.close()
        self.dialog.close()
        self.dialog.deleteLater()
        drain()
        self.dialog = None
        self.dir.cleanup()

    def test_status_and_pairing(self):
        self.assertIn("127.0.0.1:48700", self.dialog.lbl_local_api_status.text())
        self.assertTrue(self.dialog.btn_pair_app.isEnabled())
        self.dialog._pair_app()
        pairing = self.dialog._pairing_dialog
        self.assertIsInstance(pairing, PairingDialog)
        self.assertIs(self.store.view().phase, Phase.OPEN)
        self.assertFalse(pairing.ed_code.isVisibleTo(pairing))
        asking = self.store.ask("Tool", ["read", "write"])
        drain()
        self.assertTrue(pairing.ed_code.isVisibleTo(pairing))
        self.assertIn("Tool", pairing.lbl_state.text())
        self.assertIn("Save to the library", pairing.lbl_state.text())
        pairing.ed_code.setText(f"{(int(asking.code) + 1) % 10000:04d}")
        pairing._enter()
        self.assertIn("2 tries left", pairing.lbl_state.text())
        pairing.ed_code.setText(asking.code)
        pairing._enter()
        drain()
        self.assertEqual(self.store.check(asking.issued.token)[0], Verdict.PAIRED)
        self.assertIn("Tool is paired.", pairing.lbl_state.text())
        self.assertEqual(self.dialog.tbl_paired_apps.rowCount(), 1)
        self.assertEqual(self.dialog.tbl_paired_apps.item(0, 0).text(), "Tool")
        # unpaired after a confirmation
        self.dialog.tbl_paired_apps.selectRow(0)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.No):
            self.dialog._unpair_apps()
        self.assertEqual(len(self.store.pairings()), 1)
        with mock.patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
            self.dialog._unpair_apps()
        drain()
        self.assertEqual(self.store.pairings(), [])
        self.assertEqual(self.dialog.tbl_paired_apps.rowCount(), 0)

    def test_closed_before_an_answer_refuses(self):
        self.dialog._pair_app()
        asking = self.store.ask("Tool", ["read"])
        self.dialog._pairing_dialog.close()
        self.assertEqual(self.store.check(asking.issued.token)[0], Verdict.DENIED)
        self.assertIs(self.store.view().phase, Phase.CLOSED)

    def test_off_until_listening(self):
        self.api.listening = False
        self.api.status = lambda: SimpleNamespace(on=False, listening=False, port=48700, error="")
        self.dialog._refresh_local_api()
        self.assertFalse(self.dialog.btn_pair_app.isEnabled())
        self.assertIn("Off", self.dialog.lbl_local_api_status.text())


if __name__ == "__main__":
    unittest.main()
