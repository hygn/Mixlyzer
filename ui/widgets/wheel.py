"""The mouse wheel never changes an input field: over a spin box or a combo box it scrolls what holds them (the
settings' pages, a table) instead. A value moved by a wheel meant for the page was a setting changed unseen.

The deck's own wheel controls (the tempo and volume sliders, the knobs) keep theirs: they are no input fields."""
from __future__ import annotations

from PySide6 import QtGui, QtWidgets


def _scroll_instead(widget: QtWidgets.QWidget, event: QtGui.QWheelEvent) -> None:
    """The wheel given to the nearest scroll area holding ``widget`` (none: passed on to the parent as ignored)."""
    area = widget.parentWidget()
    while area is not None and not isinstance(area, QtWidgets.QAbstractScrollArea):
        area = area.parentWidget()
    if area is None:
        event.ignore()
        return
    viewport = area.viewport()
    QtWidgets.QApplication.sendEvent(viewport, QtGui.QWheelEvent(
        viewport.mapFromGlobal(event.globalPosition()), event.globalPosition(), event.pixelDelta(),
        event.angleDelta(), event.buttons(), event.modifiers(), event.phase(), event.inverted(), event.source()))
    event.accept()


def install() -> None:
    """Every spin box (numbers, times) and combo box made from Python from now on: the wheel scrolls past it. Called
    once at the start, before any window is made."""
    QtWidgets.QAbstractSpinBox.wheelEvent = _scroll_instead
    QtWidgets.QComboBox.wheelEvent = _scroll_instead
