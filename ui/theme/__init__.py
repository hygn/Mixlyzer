"""App theme: tokens -> palette + style sheet, applied the same way on every platform.

    from ui.theme import apply_theme, current_theme
    apply_theme(app)            # once, after QApplication is created
    current_theme().accent      # colours for code that paints / styles itself
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme.icons import write_icons
from ui.theme.palette import build_palette
from ui.theme.qss import build_stylesheet
from ui.theme.tokens import DARK, Theme

__all__ = ["DARK", "Theme", "apply_theme", "current_theme"]

_current: Theme = DARK


def current_theme() -> Theme:
    return _current


def apply_theme(app: QtWidgets.QApplication, theme: Theme = DARK) -> None:
    """Fusion everywhere (native styles ignore parts of a style sheet and differ per
    OS), then the theme's palette and style sheet."""
    global _current
    _current = theme
    app.setStyle("Fusion")
    app.setPalette(build_palette(theme))
    app.setStyleSheet(build_stylesheet(theme, write_icons(theme)))
    hints = QtGui.QGuiApplication.styleHints()
    if hasattr(hints, "setColorScheme"):
        hints.setColorScheme(QtCore.Qt.ColorScheme.Dark if theme.dark else QtCore.Qt.ColorScheme.Light)
