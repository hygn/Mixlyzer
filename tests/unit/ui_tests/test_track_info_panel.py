"""The track info (ui.deck.track_info_panel): the loaded track dragged out of it, its file as a file manager's drag
would carry it, from the album art and from the info, none while nothing is loaded; the time row's forms, a click
going to the next, a drag from it switching nothing."""
import os
import tempfile
import unittest
from unittest import mock

from unit.ui_tests.qt_env import APP

from PySide6 import QtCore, QtGui, QtTest

import ui.deck.track_info_panel as panel_module
from ui.deck.track_info_panel import TrackInfoPanel


class _Drag:
    """QDrag recording what it carries instead of running a drag."""
    made: list["_Drag"] = []

    def __init__(self, source):
        self.source, self.data = source, None
        _Drag.made.append(self)

    def setMimeData(self, data):
        self.data = data

    def setPixmap(self, _pixmap):
        pass

    def exec(self, *_actions):
        return QtCore.Qt.CopyAction


def _drag_from(widget) -> None:
    QtTest.QTest.mousePress(widget, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(10, 10))
    move = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(40, 40), QtCore.QPointF(40, 40),
                             QtCore.Qt.NoButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    APP.sendEvent(widget, move)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(40, 40))


class TrackInfoDragTests(unittest.TestCase):
    def setUp(self):
        _Drag.made.clear()
        patcher = mock.patch.object(panel_module.QtGui, "QDrag", _Drag)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.panel = TrackInfoPanel()
        self.panel.resize(900, TrackInfoPanel.HEADER_H)
        self.panel.show()
        APP.processEvents()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.file = os.path.join(folder.name, "track.mp3")
        with open(self.file, "wb"):
            pass

    def test_the_album_art_and_the_info_drag_the_loaded_file(self):
        self.panel.set_file(self.file)
        for widget in (self.panel.cover_label, self.panel.info_area):
            _drag_from(widget)
        self.assertEqual(len(_Drag.made), 2)
        for drag in _Drag.made:
            self.assertIs(drag.source, self.panel)
            self.assertEqual([u.toLocalFile() for u in drag.data.urls()],
                             [QtCore.QUrl.fromLocalFile(self.file).toLocalFile()])

    def test_nothing_dragged_while_nothing_is_loaded(self):
        self.panel.set_file("")
        _drag_from(self.panel.cover_label)
        self.panel.set_file(self.file + ".missing")
        _drag_from(self.panel.info_area)
        self.assertEqual(_Drag.made, [])

    def _time_row(self) -> list[str]:
        return [cell.text() for cell in self.panel._time_cells]

    def test_a_click_on_the_time_switches_its_form(self):
        changed = []
        self.panel.time_display_changed.connect(changed.append)
        self.panel.set_time(65.5, 200.0)
        self.assertEqual(self._time_row(), ["01:05.50", "/ 03:20.00"])
        cell = self.panel._time_cells[0]
        QtTest.QTest.mouseClick(cell, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(5, 5))
        self.assertEqual(self._time_row(), ["−02:14.50", "/ 03:20.00"])
        QtTest.QTest.mouseClick(self.panel._time_cells[1], QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                                QtCore.QPoint(5, 5))
        self.assertEqual(self._time_row(), ["01:05.50", "/ −02:14.50"])
        QtTest.QTest.mouseClick(cell, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(5, 5))
        self.assertEqual(self._time_row(), ["01:05.50", "/ 03:20.00"])
        self.assertEqual(changed, ["remaining", "elapsed_remaining", "elapsed"])

    def test_a_drag_from_the_time_switches_nothing(self):
        self.panel.set_file(self.file)
        self.panel.set_time_display("remaining")
        _drag_from(self.panel._time_cells[0])
        self.assertEqual(len(_Drag.made), 1)
        self.assertEqual(self.panel._time_display, "remaining")


if __name__ == "__main__":
    unittest.main()
