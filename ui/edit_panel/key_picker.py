"""Key picker popup: the 24 keys as Alphanumeric rows (1A..12A over 1B..12B), each in its
key colour and named both ways (Alphanumeric over classical); one click picks."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from views.render.color import key_cmap_color, key_label_text_color
from views.render.keystrip import key_segment_label
from analyzer.music.key_names import ALPHANUMERIC_LABELS, idx_to_labels
from system.i18n import tr

# key index for each Alphanumeric cell: ALPHANUMERIC_INDEX[(number, letter)]
ALPHANUMERIC_INDEX = {(int(c[:-1]), c[-1]): i for i, c in enumerate(ALPHANUMERIC_LABELS)}


class KeyPickerPopup(QtWidgets.QFrame):
    picked = QtCore.Signal(int)

    CELL_W, CELL_H = 44, 38

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Popup)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(2)
        self._cells: dict[int, QtWidgets.QToolButton] = {}
        for row, letter in enumerate(("A", "B")):
            for n in range(1, 13):
                idx = ALPHANUMERIC_INDEX[(n, letter)]
                cell = QtWidgets.QToolButton()
                cell.setText("\n".join(idx_to_labels(idx)))
                cell.setToolTip(key_segment_label(idx))
                cell.setFixedSize(self.CELL_W, self.CELL_H)
                cell.clicked.connect(lambda _=False, i=idx: self._pick(i))
                grid.addWidget(cell, row, n - 1)
                self._cells[idx] = cell
        self._current: int | None = None
        self._restyle()

    def _restyle(self) -> None:
        for idx, cell in self._cells.items():
            bg = "rgb({}, {}, {})".format(*key_cmap_color(idx))
            fg = "rgb({}, {}, {})".format(*key_label_text_color(idx))
            ring = fg if idx == self._current else bg
            cell.setStyleSheet(
                f"QToolButton {{ background-color: {bg}; color: {fg}; border: 2px solid {ring};"
                " border-radius: 2px; font-weight: bold; padding: 0; }"
                f"QToolButton:hover {{ border-color: {fg}; }}"
            )

    def show_for(self, anchor: QtWidgets.QWidget, current: int | None) -> None:
        """Open under ``anchor`` with ``current`` outlined."""
        self._current = current
        self._restyle()
        self.adjustSize()
        pos = anchor.mapToGlobal(QtCore.QPoint(0, anchor.height() + 2))
        screen = anchor.screen().availableGeometry() if anchor.screen() else None
        if screen is not None:
            pos.setX(min(pos.x(), screen.right() - self.width()))
            if pos.y() + self.height() > screen.bottom():
                pos.setY(anchor.mapToGlobal(QtCore.QPoint(0, 0)).y() - self.height() - 2)
        self.move(pos)
        self.show()

    def _pick(self, idx: int) -> None:
        self.hide()
        self.picked.emit(int(idx))


class KeyPickButton(QtWidgets.QToolButton):
    """A key value shown in its key colour; a click opens the KeyPickerPopup."""

    keyChanged = QtCore.Signal(int)

    def __init__(self, key: int = 0, parent=None):
        super().__init__(parent)
        self._key = int(key) % 24
        self._popup = KeyPickerPopup(self)
        self._popup.picked.connect(self.setKey)
        self.clicked.connect(lambda: self._popup.show_for(self, self._key))
        self.setToolTip(tr("Pick the key (Alphanumeric wheel)"))
        self._restyle()

    def key(self) -> int:
        return self._key

    def setKey(self, key: int) -> None:
        key = int(key) % 24
        if key == self._key:
            return
        self._key = key
        self._restyle()
        self.keyChanged.emit(key)

    def _restyle(self) -> None:
        from ui.theme import current_theme
        theme = current_theme()
        self.setText(key_segment_label(self._key))
        # Dressed like the text inputs beside it; the key colour is only a strip
        # on the left so a row of inputs keeps one visual weight.
        swatch = "rgb({}, {}, {})".format(*key_cmap_color(self._key))
        self.setStyleSheet(
            f"QToolButton {{ background-color: {theme.base}; color: {theme.text};"
            f" border: 1px solid {theme.border}; border-left: 4px solid {swatch};"
            f" border-radius: {theme.radius_small}px; padding: 0 6px; }}"
            f"QToolButton:hover {{ border-color: {theme.border_strong}; border-left-color: {swatch}; }}"
            f"QToolButton:disabled {{ background-color: {theme.window}; color: {theme.text_disabled};"
            f" border-color: {theme.separator}; border-left-color: {theme.separator}; }}"
        )
