"""A button's flash: an accent glow over it that fades, as the button lights when a keyboard shortcut works it.

Drawn over the button (its style sheet untouched: many editor buttons set their own colours)."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme

FLASH_MS = 260
FLASH_ALPHA = 0.7


class _Flash(QtWidgets.QWidget):
    def __init__(self, button: QtWidgets.QWidget):
        super().__init__(button)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self._level = 0.0
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setDuration(FLASH_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InQuad)
        self._anim.valueChanged.connect(self._set_level)
        self._anim.finished.connect(self.hide)
        self.hide()

    def start(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self._anim.stop()
        self._anim.start()

    def _set_level(self, level) -> None:
        self._level = float(level)
        self.update()

    def paintEvent(self, _event) -> None:
        color = QtGui.QColor(current_theme().accent)
        color.setAlphaF(FLASH_ALPHA * self._level)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 3.0, 3.0)
        p.end()


def flash(button: QtWidgets.QWidget) -> None:
    """Light ``button`` for a moment (again while lit: from full again)."""
    glow = getattr(button, "_shortcut_flash", None)
    if glow is None:
        glow = _Flash(button)
        button._shortcut_flash = glow
    glow.start()
