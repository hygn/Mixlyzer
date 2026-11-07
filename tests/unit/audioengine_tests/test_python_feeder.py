"""The Python engine's PCMFeeder (audioengine.python.feeder): a JumpCUE armed on a track is disarmed whole when
another track is loaded (its candidate points would rebuild the jump and fire it on the new track). A stand-in sink:
nothing plays."""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from audioengine.python.feeder import PCMFeeder  # noqa: E402

# a QApplication (not a QCoreApplication): the suites run in one process, and the interface's tests after this one
# make widgets
APP = QApplication.instance() or QApplication([])


class JumpCueOnLoadTests(unittest.TestCase):
    def test_a_track_loaded_disarms_the_jumpcue(self):
        feeder = PCMFeeder(mock.Mock(), 44100, 2)
        feeder.set_track(np.zeros((44100 * 10, 2), np.float32))
        feeder.arm_jump(2.0, 6.0, starts_sec=[2.0, 4.0])          # a JumpCUE with its candidate points
        self.assertIsNotNone(feeder._jump_starts)
        feeder.set_track(np.zeros((44100 * 10, 2), np.float32))  # another track
        self.assertIsNone(feeder._jump_starts)
        self.assertIsNone(feeder._jump_start)
        self.assertIsNone(feeder._jump_dest)


class ScratchOnLoadTests(unittest.TestCase):
    def test_a_track_loaded_while_scratching_feeds_at_the_normal_interval(self):
        feeder = PCMFeeder(mock.Mock(), 44100, 2)
        feeder.set_track(np.zeros((44100 * 10, 2), np.float32))
        feeder.open()
        try:
            feeder.set_scratching(True)       # the record grabbed: fed every FLUSH_SCRATCH_MS
            self.assertEqual(feeder._flush_timer.interval(), PCMFeeder.FLUSH_SCRATCH_MS)
            feeder.set_track(np.zeros((44100 * 10, 2), np.float32))   # another track, the hand let go with it
            self.assertFalse(feeder._scratching)
            self.assertEqual(feeder._flush_timer.interval(), PCMFeeder.FLUSH_MS)
        finally:
            feeder.close()


if __name__ == "__main__":
    unittest.main()
