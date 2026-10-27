"""What the interface tests share: offscreen Qt and one QApplication, and `isolated()`, which keeps the app's widgets
off the user's files and devices while a test module runs: the current folder a temporary one (a relative path the
app writes lands there), the settings read as `default_cfg()`, the user's shortcut and MIDI files temporary ones, no
audio input listed, no key reported as held by another program."""
import contextlib
import os
import tempfile
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore
from PySide6.QtMultimedia import QAudioDevice
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])


class NoAudioInputs(QtCore.QObject):
    """QMediaDevices without devices: nothing is enumerated or opened."""
    audioInputsChanged = QtCore.Signal()

    @staticmethod
    def audioInputs() -> list:
        return []

    @staticmethod
    def defaultAudioInput() -> QAudioDevice:
        return QAudioDevice()


def isolated() -> contextlib.ExitStack:
    """Entered by a module's setUpModule, closed by its tearDownModule."""
    from core.controls import midi, shortcuts
    from core.settings.config import default_cfg
    from system import os_support
    import ui.edit_panel.phrase
    import ui.library.actions
    import ui.library.music
    import ui.music_search

    stack = contextlib.ExitStack()
    folder = stack.enter_context(tempfile.TemporaryDirectory())
    cwd = os.getcwd()
    os.chdir(folder)
    stack.callback(os.chdir, cwd)
    for patcher in (
        mock.patch.object(ui.library.music, "QMediaDevices", NoAudioInputs),
        mock.patch.object(ui.music_search, "QMediaDevices", NoAudioInputs),
        mock.patch.object(ui.music_search, "save_deck", lambda _cfg: None),
        mock.patch.object(ui.library.actions, "load_cfg", default_cfg),
        mock.patch.object(ui.edit_panel.phrase, "load_cfg", default_cfg),
        mock.patch.object(shortcuts, "USER_FILE", os.path.join(folder, "shortcuts.json")),
        mock.patch.object(midi, "USER_FILE", os.path.join(folder, "midi.json")),
        mock.patch.object(os_support, "hotkey_taken", lambda _keys: False),
    ):
        stack.enter_context(patcher)
    return stack


def drain(ms: int = 0) -> None:
    """Posted events and zero-delay timers run."""
    APP.processEvents(QtCore.QEventLoop.AllEvents, ms)
    APP.sendPostedEvents()
