"""The player's audio worker (core.playback.player._AudioWorker) and the deck's marks: a track decoded gets its own
marks, never the track before's (a library track's stored analysis may come after its audio: the deck played it on
the old beats, CUE points and JumpCUEs meanwhile); the same track decoded again (the engine restarted at another
format) keeps them. A stand-in feeder and deck, the decoder patched: no engine, no file."""
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from core.playback import player  # noqa: E402

APP = QApplication.instance() or QApplication([])
A_MARKS = ([1.0, 2.0], [True, False], [1.5], [3.0] + [float("nan")] * 25)
B_MARKS = ([0.5, 1.0], [True, False], [], [float("nan")] * 26)


class MarksTests(unittest.TestCase):
    def setUp(self):
        self.worker = player._AudioWorker()
        self.worker.feeder = mock.Mock(spec=["pause", "set_track", "seek", "rate", "ch"])
        self.worker.deck = mock.Mock()
        self.worker.rate, self.worker.ch = 48000, 2
        patcher = mock.patch.object(player, "decode_to_memmap", return_value=np.zeros((4800, 2), np.float32))
        patcher.start()
        self.addCleanup(patcher.stop)

    def decoded(self, path: str) -> None:
        """``path`` loaded and its decode handed over."""
        self.worker.run_command("load", (path,))
        deadline = time.monotonic() + 5.0
        while self.worker._decoding and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(0.005)
        self.assertFalse(self.worker._decoding)

    def marks_sent(self) -> list:
        return [c.args for c in self.worker.deck.set_marks.call_args_list]

    def test_the_next_track_never_gets_the_track_befores_marks(self):
        self.decoded("a.flac")
        self.worker.run_command("set_marks", (A_MARKS,))
        self.assertEqual(self.marks_sent(), [A_MARKS])
        self.decoded("b.flac")                      # its audio before its stored analysis
        self.assertEqual(self.marks_sent(), [A_MARKS], "b played on a's marks")
        self.worker.run_command("set_marks", (B_MARKS,))
        self.assertEqual(self.marks_sent()[-1], B_MARKS)

    def test_marks_come_before_the_audio_kept_for_it(self):
        self.worker.run_command("load", ("b.flac",))
        self.worker.run_command("set_marks", (B_MARKS,))   # while it decodes
        self.decoded_after_load()
        self.assertEqual(self.marks_sent()[-1], B_MARKS)

    def decoded_after_load(self):
        deadline = time.monotonic() + 5.0
        while self.worker._decoding and time.monotonic() < deadline:
            APP.processEvents()
            time.sleep(0.005)

    def test_the_same_track_decoded_again_keeps_its_marks(self):
        self.decoded("a.flac")
        self.worker.run_command("set_marks", (A_MARKS,))
        self.worker.feeder.rate, self.worker.feeder.ch = 44100, 2
        self.worker._on_format_changed(1.0)                # the engine restarted at another format
        self.decoded_after_load()
        self.assertEqual(self.marks_sent()[-1], A_MARKS)
        self.worker.deck.set_marks.reset_mock()
        self.worker.run_command("load", ("a.flac", True))  # the same track loaded again
        self.decoded_after_load()
        self.assertEqual(self.marks_sent(), [A_MARKS])

    def test_a_track_not_decoded_leaves_the_transport_off(self):
        """PLAY, STOP and CUE stay off: with no track they would do nothing."""
        shown = []
        self.worker.transport_enabled.connect(shown.append)
        with mock.patch.object(player, "decode_to_memmap", side_effect=RuntimeError("not audio")):
            self.worker.run_command("load", ("broken.flac",))
            self.decoded_after_load()
        self.assertEqual(shown, [False])
        self.assertFalse(self.worker._track_ready)


if __name__ == "__main__":
    unittest.main()
