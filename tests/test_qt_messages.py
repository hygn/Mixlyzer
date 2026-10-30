"""ui.qt_messages: Qt's notes that say nothing wrong left out of the console (the log, the diagnostics), made here as
the app makes them: pyqtgraph's plot widgets (its unique connection to the colour scheme), the audio inputs listed
(the media backend's version), a script no bundled font draws; every other message passed on."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from ui import qt_messages  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class QtMessagesTests(unittest.TestCase):
    def setUp(self):
        self.passed: list[tuple[str, str]] = []
        record = lambda _kind, context, message: self.passed.append((context.category or "", message))  # noqa: E731
        QtCore.qInstallMessageHandler(record)
        qt_messages.install()

    def tearDown(self):
        QtCore.qInstallMessageHandler(None)
        qt_messages._previous_handler = None

    def test_the_notes_left_out(self):
        import pyqtgraph as pg
        widgets = [pg.GraphicsLayoutWidget() for _ in range(2)]
        from PySide6.QtMultimedia import QMediaDevices
        QMediaDevices.audioInputs()
        label = QtWidgets.QLabel("বাংলা தமிழ்")
        label.grab()
        for category, message in self.passed:
            self.assertFalse(qt_messages.quiet(category, message))
        self.assertFalse([m for _c, m in self.passed if "QStyleHints" in m or "Qt multimedia with FFmpeg" in m])
        del widgets

    def test_the_notes_as_qt_words_them(self):
        # Qt 6.10's words (the media backend's said once a process, so not made again here)
        for category, message in (
                ("qt.core.qobject.connect", "QObject::connect(QStyleHints, QStyleHints): unique connections require "
                                            "a pointer to member function of a QObject subclass"),
                ("qt.multimedia.ffmpeg", "Using Qt multimedia with FFmpeg version 7.1.1 LGPL version 2.1 or later"),
                ("qt.text.font.db", "OpenType support missing for \"Noto Sans JP\", script 11")):
            self.assertTrue(qt_messages.quiet(category, message), message)
        self.assertFalse(qt_messages.quiet("qt.multimedia.ffmpeg", "Could not open the audio input"))

    def test_others_passed_on(self):
        QtCore.QMessageLogger().warning("ヴヴヴ: a warning of the app's own")
        self.assertIn("ヴヴヴ: a warning of the app's own", [m for _c, m in self.passed])


if __name__ == "__main__":
    unittest.main()
