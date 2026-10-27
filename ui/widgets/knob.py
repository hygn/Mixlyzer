"""A rotary knob (a DJ mixer's TRIM / EQ / a level): 0 .. 1, its neutral setting the middle (a level's: its 0 dB). It sends a request while the hand
turns it and shows what the player then reports (set_value), so a controller's knob moves it too."""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme


class Knob(QtWidgets.QWidget):
    """Dragged up / down (DRAG_PX the whole range), the wheel a step, a double-click back to the middle; ``moved``
    while the hand turns it. Drawn as a mixer's: the travel from -135 to +135 degrees, the setting as an arc from
    the neutral setting (``neutral``, the double-click's too) and a pointer."""

    moved = QtCore.Signal(float)
    DRAG_PX = 160.0
    WHEEL_STEP = 0.02

    def __init__(self, size: int = 20, parent=None, neutral: float = 0.5) -> None:
        super().__init__(parent)
        self._neutral = float(neutral)
        self._value = self._neutral
        self._drag: tuple[float, float] | None = None   # (y at the press, the value then)
        self.setFixedSize(size, size)
        self.setCursor(QtCore.Qt.SizeVerCursor)

    @property
    def held(self) -> bool:
        return self._drag is not None

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        """Shown (no request); not while the hand holds it."""
        if self._drag is None:
            self._set(value, emit=False)

    def _set(self, value: float, emit: bool) -> None:
        v = min(1.0, max(0.0, float(value)))
        if abs(v - self._value) < 1e-9:
            return
        self._value = v
        self.update()
        if emit:
            self.moved.emit(v)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self._drag = (event.position().y(), self._value)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag is not None:
            y0, v0 = self._drag
            self._set(v0 + (y0 - event.position().y()) / self.DRAG_PX, emit=True)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.isEnabled():
            self._set(self._neutral, emit=True)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta and self.isEnabled():
            self._set(self._value + (self.WHEEL_STEP if delta > 0 else -self.WHEEL_STEP), emit=True)
        event.accept()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        theme = current_theme()
        enabled = self.isEnabled()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(3, 3, -3, -3)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(theme.raised))
        p.drawEllipse(r)
        arc = r.adjusted(-1.5, -1.5, 1.5, 1.5)
        p.setPen(QtGui.QPen(QtGui.QColor(theme.border), 2.0))
        p.drawArc(arc, 225 * 16, -270 * 16)
        span = (self._value - 0.5) * 270.0
        arc_span = (self._value - self._neutral) * 270.0
        if abs(arc_span) > 0.5 and enabled:
            p.setPen(QtGui.QPen(QtGui.QColor(theme.accent), 2.0))
            p.drawArc(arc, int((90.0 - (self._neutral - 0.5) * 270.0) * 16), int(-arc_span * 16))
        angle = math.radians(90.0 - span)
        c = r.center()
        tip = QtCore.QPointF(c.x() + math.cos(angle) * r.width() * 0.42, c.y() - math.sin(angle) * r.height() * 0.42)
        p.setPen(QtGui.QPen(QtGui.QColor(theme.text if enabled else theme.text_disabled), 2.0, QtCore.Qt.SolidLine,
                            QtCore.Qt.RoundCap))
        p.drawLine(c, tip)
