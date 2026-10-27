import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from ui.widgets.live_label import LiveLabel  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _LayoutRequests(QtCore.QObject):
    def __init__(self, widget):
        super().__init__()
        self.count = 0
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.LayoutRequest:
            self.count += 1
        return False


class LiveLabelTests(unittest.TestCase):
    def setUp(self):
        self.parent = QtWidgets.QWidget()
        self.label = LiveLabel("Elapsed: 00:00.00", self.parent)   # no layout: requests go to the parent
        self.parent.show()
        app.processEvents()
        self.requests = _LayoutRequests(self.parent)

    def tearDown(self):
        self.parent.close()

    def test_a_new_text_no_wider_asks_for_no_layout(self):
        for text in ("Elapsed: 00:01.11", "Elapsed: 00:02.22", "Elapsed: 00:00.00"):
            self.label.setText(text)
            app.processEvents()
        self.assertEqual(self.label.text(), "Elapsed: 00:00.00")
        self.assertEqual(self.requests.count, 0)

    def test_a_wider_text_widens_its_size_hint(self):
        before = self.label.sizeHint().width()
        self.label.setText("Elapsed: 00:00.00 | Left: 00:00.00")
        app.processEvents()
        self.assertGreater(self.label.sizeHint().width(), before)
        self.assertGreater(self.requests.count, 0)
        width = self.label.sizeHint().width()
        self.label.setText("short")                                 # the hint keeps the widest text
        self.assertEqual(self.label.sizeHint().width(), width)


if __name__ == "__main__":
    unittest.main()
