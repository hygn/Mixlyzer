"""The Beatgrid page's TAP (ui.edit_panel.beatgrid): a tap after a pause of over 2 s starts the taps anew (it was one
very slow beat that dragged the BPM down), and one within 60 ms of the last (faster than 1000 BPM: a bounce, a double
click) is no tap. The mixin's own method on a stand-in, the clock given."""
import unittest
from unittest import mock

from PySide6 import QtWidgets

from unit.ui_tests.qt_env import APP  # noqa: F401

from ui.edit_panel import beatgrid


class _Panel:
    _tapBPM_set = beatgrid.BeatgridEditMixin._tapBPM_set

    def __init__(self):
        self.prev_tap_time = None
        self.prev_tap_BPM = 0
        self.tempo_multiplier = 1.0
        self.inp_refBPM = QtWidgets.QLineEdit()


class TapTests(unittest.TestCase):
    def tap(self, panel, *times):
        for t in times:
            with mock.patch.object(beatgrid.time, "perf_counter", return_value=t):
                panel._tapBPM_set()
        return panel.inp_refBPM.text()

    def test_taps_give_their_bpm(self):
        self.assertEqual(self.tap(_Panel(), 0.0, 0.5, 1.0, 1.5), "120.00")

    def test_a_pause_starts_the_taps_anew(self):
        panel = _Panel()
        self.tap(panel, 0.0, 0.5, 1.0, 1.5)
        self.assertEqual(self.tap(panel, 10.0), "0")                  # a first tap again
        self.assertEqual(self.tap(panel, 10.4), "150.00")             # the new taps alone

    def test_a_bounce_is_no_tap(self):
        self.assertEqual(self.tap(_Panel(), 0.0, 0.5, 0.52, 1.0), "120.00")


if __name__ == "__main__":
    unittest.main()
