"""A one-line plain-text label for a text that changes every frame (a time, a status line). QLabel.setText
asks for a new layout on every change (its size hint may have changed), and the request climbs to the window:
the whole window was laid out again on every frame. This label repaints only, and asks for a layout only when
its text is wider than any it has shown (its size hint is that widest text)."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class LiveLabel(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)   # QLabel's own text stays empty: it never relayouts
        self._text = ""
        self._hint_width = 0
        self.setText(text)

    def setText(self, text: str) -> None:
        text = str(text or "")
        if text == self._text:
            return
        self._text = text
        width = self.fontMetrics().horizontalAdvance(text)
        if width > self._hint_width:
            self._hint_width = width
            self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._text

    def sizeHint(self) -> QtCore.QSize:
        m = self.contentsMargins()
        extra = 2 * self.margin()
        return QtCore.QSize(self._hint_width + m.left() + m.right() + extra,
                            self.fontMetrics().height() + m.top() + m.bottom() + extra)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange:
            self._hint_width = self.fontMetrics().horizontalAdvance(self._text)
            self.updateGeometry()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)   # frame and style sheet background
        painter = QtGui.QPainter(self)
        rect = self.contentsRect().adjusted(self.margin(), self.margin(), -self.margin(), -self.margin())
        self.style().drawItemText(painter, rect, int(self.alignment()), self.palette(), self.isEnabled(), self._text,
                                  self.foregroundRole())
