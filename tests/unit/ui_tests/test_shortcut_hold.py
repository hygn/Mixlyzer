"""The keyboard shortcuts' held keys (ui.shortcuts.ShortcutManager): a key held when the shortcuts are set again is let
go there (its release, matched against the action's keys, would no longer match: it repeated on), a momentary one's
handler told."""
import unittest

from unit.ui_tests.qt_env import APP  # noqa: F401

from PySide6 import QtWidgets

from ui.shortcuts import ShortcutManager


class HeldKeyTests(unittest.TestCase):
    def setUp(self):
        self.window = QtWidgets.QWidget()
        self.steps, self.cue = [], []
        handlers = {"playback.beat_forward": lambda: self.steps.append(1),
                    "playback.cue": lambda down: self.cue.append(down)}
        self.manager = ShortcutManager(self.window, handlers, lambda _text, _ms: None)
        self.manager.apply({"playback.beat_forward": ["Right"], "playback.cue": ["C"]})

    def tearDown(self):
        QtWidgets.QApplication.instance().removeEventFilter(self.manager)
        self.window.deleteLater()

    def test_a_repeating_key_held_stops_when_the_shortcuts_are_set_again(self):
        self.manager._shortcuts["playback.beat_forward"].activated.emit()     # Right pressed and held
        self.assertTrue(self.manager._hold_timer.isActive())
        self.manager.apply({"playback.beat_forward": ["L"], "playback.cue": ["C"]})   # Settings > Shortcuts' OK
        self.assertIsNone(self.manager._held)
        self.assertFalse(self.manager._hold_timer.isActive())

    def test_a_momentary_key_held_is_let_go_when_the_shortcuts_are_set_again(self):
        self.manager._shortcuts["playback.cue"].activated.emit()              # CUE held: previewing
        self.manager.apply({"playback.beat_forward": ["Right"], "playback.cue": ["V"]})
        self.assertEqual(self.cue, [True, False])


if __name__ == "__main__":
    unittest.main()
