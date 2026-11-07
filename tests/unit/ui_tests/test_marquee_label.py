"""The track info's MarqueeLabel (ui.deck.track_info_panel): squeezed to no width it waits for a resize, never
polling the event loop (a 0 ms timer started again and again kept the GUI thread busy); given room again it scrolls."""
import time
import unittest
from unittest import mock

from PySide6 import QtWidgets

from unit.ui_tests.qt_env import APP  # noqa: F401

from ui.deck.track_info_panel import MarqueeLabel


class MarqueeTests(unittest.TestCase):
    def setUp(self):
        self.holder = QtWidgets.QWidget()
        self.label = MarqueeLabel(self.holder)
        self.holder.show()

    def tearDown(self):
        self.holder.close()
        self.holder.deleteLater()

    def _spin(self, sec: float = 0.2) -> None:
        end = time.monotonic() + sec
        while time.monotonic() < end:
            APP.processEvents()

    def test_no_width_no_polling(self):
        self.label.setGeometry(0, 0, 0, 20)
        calls = []
        real = self.label._update_scroll_state
        with mock.patch.object(self.label, "_update_scroll_state", side_effect=lambda *a, **k: (calls.append(1), real(*a, **k))):
            self.label.set_marquee_text("A title far too long to fit in no room at all")
            self._spin()
        self.assertLess(len(calls), 5, f"{len(calls)} checks in 0.2 s")

    def test_room_again_scrolls(self):
        self.label.setGeometry(0, 0, 0, 20)
        self.label.set_marquee_text("A title far too long to fit in sixty pixels")
        self._spin(0.05)
        self.label.setGeometry(0, 0, 60, 20)
        self._spin(0.05)
        self.assertTrue(self.label._needs_scroll)
        self.assertTrue(self.label._timer.isActive())


if __name__ == "__main__":
    unittest.main()
