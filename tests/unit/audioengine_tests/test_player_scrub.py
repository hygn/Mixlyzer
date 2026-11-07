"""PlayerController's scrub (core.playback.player): the overview dragged moves the local deck, as every local deck
command, only while External Sync is off (it follows the other program's deck then: the drag moves the views
alone); a scrub begun before External Sync was turned on is still ended. The controller's own methods on a stand-in
(no audio engine started)."""
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
    _send_local = PlayerController._send_local

    def __init__(self, external: bool):
        self._external_sync_enabled = external
        self._scrub_pending = None
        self._scrub_sent = None
        self._scrub_local = False
        self._scrub_timer = QtCore.QTimer()
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(PlayerController.SCRUB_INTERVAL_MS)
        self.sent = []

    def _send(self, name, *args):
        self.sent.append(name)


class ScrubTests(unittest.TestCase):
    def scrub(self, player, *, external_on_midway=False):
        player._on_scrub_begin()
        player._on_scrub_update(10.0)
        if external_on_midway:
            player._external_sync_enabled = True
        player._on_scrub_update(20.0)
        player._scrub_timer.stop()
        player._send_scrub()
        player._on_scrub_end(30.0)

    def test_the_local_deck_scrubbed(self):
        player = _Player(external=False)
        self.scrub(player)
        self.assertEqual(player.sent, ["scrub_begin", "scrub_update", "scrub_update", "scrub_update", "scrub_end"])

    def test_nothing_sent_while_external_sync_follows(self):
        player = _Player(external=True)
        self.scrub(player)
        self.assertEqual(player.sent, [])

    def test_a_scrub_begun_before_external_sync_is_ended(self):
        player = _Player(external=False)
        self.scrub(player, external_on_midway=True)
        self.assertEqual(player.sent, ["scrub_begin", "scrub_update", "scrub_end"])


if __name__ == "__main__":
    unittest.main()
