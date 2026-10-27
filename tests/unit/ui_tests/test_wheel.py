"""The mouse wheel over an input field (ui.widgets.wheel): a spin box's value and a combo box's item stay, and the
scroll area holding them scrolls instead; with none around them, nothing changes."""
import unittest

from unit.ui_tests.qt_env import APP

from PySide6 import QtCore, QtGui, QtWidgets

from ui.widgets import wheel


def _wheel(widget: QtWidgets.QWidget, notches: int = -1) -> None:
    pos = QtCore.QPointF(widget.width() / 2, widget.height() / 2)
    QtWidgets.QApplication.sendEvent(widget, QtGui.QWheelEvent(
        pos, widget.mapToGlobal(pos), QtCore.QPoint(), QtCore.QPoint(0, 120 * notches), QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False))
    APP.processEvents()


class WheelTests(unittest.TestCase):
    def setUp(self):
        self._kept = (QtWidgets.QAbstractSpinBox.wheelEvent, QtWidgets.QComboBox.wheelEvent)
        wheel.install()

    def tearDown(self):
        QtWidgets.QAbstractSpinBox.wheelEvent, QtWidgets.QComboBox.wheelEvent = self._kept

    def test_the_page_scrolls_and_the_fields_keep_their_values(self):
        area = QtWidgets.QScrollArea()
        area.resize(300, 200)
        inner = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(inner)
        spin, dspin, combo = QtWidgets.QSpinBox(), QtWidgets.QDoubleSpinBox(), QtWidgets.QComboBox()
        spin.setValue(50)
        dspin.setValue(5.0)
        combo.addItems(list("abcdef"))
        combo.setCurrentIndex(2)
        for w in (spin, dspin, combo):
            lay.addWidget(w)
        for _ in range(40):
            lay.addWidget(QtWidgets.QLabel("x"))
        area.setWidget(inner)
        area.show()
        APP.processEvents()
        bar = area.verticalScrollBar()
        for w in (spin, dspin, combo):
            w.setFocus()
            before = bar.value()
            _wheel(w)
            self.assertGreater(bar.value(), before, type(w).__name__)
        self.assertEqual((spin.value(), dspin.value(), combo.currentIndex()), (50, 5.0, 2))
        area.close()

    def test_no_scroll_area_changes_nothing(self):
        spin = QtWidgets.QSpinBox()
        spin.setValue(3)
        spin.show()
        _wheel(spin, 1)
        self.assertEqual(spin.value(), 3)
        spin.close()


if __name__ == "__main__":
    unittest.main()
