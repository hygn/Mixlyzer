"""Tooltips app-wide (ui.widgets.tooltips): wrapped at MAX_WIDTH when wider, a line wrapped by hand joined (a new
sentence's or a list item's kept), a button's shortcut keys after it in parentheses, shown for a tagged widget."""
import unittest

from unit.ui_tests.qt_env import APP  # noqa: F401

from PySide6 import QtCore, QtGui, QtWidgets

from ui.widgets import tooltips


class TooltipTests(unittest.TestCase):
    def test_short_kept_on_one_line_with_its_keys(self):
        text = tooltips.formatted("Split", [QtGui.QKeySequence("Ctrl+K")])
        self.assertNotIn("<table", text)
        self.assertIn("Split (Ctrl+K)", text.replace("&nbsp;", " "))

    def test_long_wrapped_at_the_width(self):
        long = "Reanalyze the current key segment's key from the audio, " * 6
        text = tooltips.formatted(long)
        self.assertIn(f"width='{tooltips.MAX_WIDTH}'", text)

    def test_hand_wrapped_lines_joined_new_lines_kept(self):
        text = tooltips.formatted("Root and fifth: hold the root\nand fifth under the music.\nIn the right key ...")
        self.assertIn("root and fifth", text)            # a continuation joined
        self.assertIn("music.<br>In the right", text)     # a new sentence on its own line

    def test_rich_text_keys_inside_its_last_paragraph(self):
        text = tooltips.formatted("<p>The key a semitone up.</p>", [QtGui.QKeySequence("Shift+=")])
        self.assertTrue(text.rstrip().endswith("(Shift+=)</p>"), text)

    def test_a_tagged_widgets_keys_shown(self):
        button = QtWidgets.QToolButton()
        button.setToolTip("Split")
        tooltips.tag(button, "editor.split")
        tooltips.install()
        tooltips.set_shortcut_keys(lambda action: [QtGui.QKeySequence("Ctrl+K")] if action == "editor.split" else [])
        shown = []
        original = QtWidgets.QToolTip.showText
        try:
            QtWidgets.QToolTip.showText = staticmethod(lambda pos, text, *a: shown.append(text))
            event = QtGui.QHelpEvent(QtCore.QEvent.ToolTip, QtCore.QPoint(1, 1), QtCore.QPoint(1, 1))
            QtWidgets.QApplication.sendEvent(button, event)
        finally:
            QtWidgets.QToolTip.showText = original
            tooltips.set_shortcut_keys(None)
        self.assertEqual(len(shown), 1)
        self.assertIn("(Ctrl+K)", shown[0])


if __name__ == "__main__":
    unittest.main()
