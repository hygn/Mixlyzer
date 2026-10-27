"""A window made at the start polished and laid out then, not as it is first shown: its first show held the main
view's frames while every widget took the style sheet (the settings dialog: ~30 of its ~50 ms)."""
from __future__ import annotations

from PySide6 import QtWidgets


def polish_ahead(window: QtWidgets.QWidget) -> None:
    """``window`` and every widget in it polished (the style sheet applied) and its layout activated, not shown."""
    for widget in (window, *window.findChildren(QtWidgets.QWidget)):
        widget.ensurePolished()
    if window.layout() is not None:
        window.layout().activate()
