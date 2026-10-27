"""A one-line label that never widens its window: a text longer than the room the layout gives it is cut with "…"
(the whole text in its tooltip). For titles, paths and messages, whose length has no bound."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class ElidedLabel(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None, *,
                 mode: QtCore.Qt.TextElideMode = QtCore.Qt.ElideRight):
        """``mode``: where the text is cut (ElideMiddle for a path: its file name stays)."""
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = str(text or "")
        self._elide()

    def text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, self._mode, max(0, self.contentsRect().width()))
        super().setText(shown)
        self.setToolTip(self._full if shown != self._full else "")
