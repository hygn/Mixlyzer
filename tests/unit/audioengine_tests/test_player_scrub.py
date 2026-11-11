"""PlayerController's scrub (core.playback.player): the overview dragged moves the deck, its positions sent at most
every SCRUB_INTERVAL_MS, the last one when let go. The controller's own methods on a stand-in (no audio engine
started)."""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from core.playback.player import PlayerController  # noqa: E402

APP = QApplication.instance() or QApplication([])


class _Player:
    _on_scrub_begin = PlayerController._on_scrub_begin
    _on_scrub_update = PlayerController._on_scrub_update
    _send_scrub = PlayerController._send_scrub
    _on_scrub_end = PlayerController._on_scrub_end

    def __init__(self):
        self._scrub_pending = None
        self._scrub_sent = None
        self._scrub_timer = QtCore.QTimer()
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(PlayerController.SCRUB_INTERVAL_MS)
        self.sent = []

    def _send(self, name, *args):
        self.sent.append(name)


class ScrubTests(unittest.TestCase):
    def test_the_deck_scrubbed(self):
        player = _Player()
        player._on_scrub_begin()
        player._on_scrub_update(10.0)
        player._on_scrub_update(20.0)            # within the interval: kept for the timer
        player._scrub_timer.stop()
        player._send_scrub()
        player._on_scrub_end(30.0)
        self.assertEqual(player.sent, ["scrub_begin", "scrub_update", "scrub_update", "scrub_update", "scrub_end"])

    def test_let_go_where_last_sent(self):
        player = _Player()
        player._on_scrub_begin()
        player._on_scrub_update(10.0)
        player._on_scrub_end(10.0)               # held still there: no second seek to it
        self.assertEqual(player.sent, ["scrub_begin", "scrub_update", "scrub_end"])


if __name__ == "__main__":
    unittest.main()
