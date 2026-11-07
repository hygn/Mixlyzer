"""A drag held on the main plot (a scratch) or the overview (a scrub) ends however the button is lost: let go over a
box opened while it is held (the release goes to the box), or another window taking the focus. It ended at the next
click before: the deck held, the waveform following a scrub, until then."""
import unittest

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from unit.ui_tests.qt_env import APP  # noqa: F401

from core.workflows.event_bus import EventBus
from ui.deck.mainplotvbx import ScratchViewBox
from views.overview import SeekOnlyViewBox


class HeldDragTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.events = []
        for name in ("sig_scratch_begin", "sig_scratch_end", "sig_scrub_begin", "sig_scrub_end"):
            getattr(self.bus, name).connect(lambda *args, name=name: self.events.append((name[4:], *args)))
        self.windows = []

    def tearDown(self):
        for window in self.windows:
            window.close()
            window.deleteLater()
        QtWidgets.QApplication.processEvents()

    def held(self, box):
        """The box in a window, the left button pressed at its middle."""
        win = pg.GraphicsLayoutWidget()
        self.windows.append(win)
        win.addPlot(viewBox=box).setXRange(0, 8)
        win.resize(800, 300)
        win.show()
        QTest.qWaitForWindowExposed(win)
        win.activateWindow()
        QtWidgets.QApplication.processEvents()
        QTest.mousePress(win.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, win.viewport().rect().center())
        QtWidgets.QApplication.processEvents()
        return win

    def let_go_over_a_box(self, win):
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Edits", "Save?", QtWidgets.QMessageBox.Ok, win)

        def answer():
            QTest.mouseRelease(box, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(5, 5))
            box.done(0)
        QtCore.QTimer.singleShot(50, answer)
        box.exec()
        QtWidgets.QApplication.processEvents()

    def scratch_box(self):
        return ScratchViewBox(self.bus, get_time=lambda: 10.0, get_duration=lambda: 100.0,
                              can_scratch=lambda: True, get_window=lambda: 8.0)

    def test_a_scratch_let_go_over_a_box_ends(self):
        win = self.held(self.scratch_box())
        self.let_go_over_a_box(win)
        self.assertEqual([e[0] for e in self.events], ["scratch_begin", "scratch_end"])
        self.assertEqual(self.events[-1][1], 10.0)                    # not moved: where it was grabbed
        self.assertIsNone(win.scene().mouseGrabberItem())

    def test_a_scratch_ends_when_another_window_takes_the_focus(self):
        self.held(self.scratch_box())
        other = QtWidgets.QWidget()
        self.windows.append(other)
        other.show()
        other.activateWindow()
        QtWidgets.QApplication.processEvents()
        self.assertEqual([e[0] for e in self.events], ["scratch_begin", "scratch_end"])

    def test_a_scrub_let_go_over_a_box_ends_where_it_was(self):
        box = SeekOnlyViewBox(bus=self.bus, get_duration=lambda: 100.0, get_time=lambda: 42.0)
        win = self.held(box)
        at = box._scrub_t
        self.assertIsNotNone(at)
        self.let_go_over_a_box(win)
        self.assertEqual([e[0] for e in self.events], ["scrub_begin", "scrub_end"])
        self.assertEqual(self.events[-1][1], at)

    def test_a_release_ends_once(self):
        win = self.held(self.scratch_box())
        QTest.mouseRelease(win.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                           win.viewport().rect().center())
        QtWidgets.QApplication.processEvents()
        self.assertEqual([e[0] for e in self.events], ["scratch_begin", "scratch_end"])


if __name__ == "__main__":
    unittest.main()
