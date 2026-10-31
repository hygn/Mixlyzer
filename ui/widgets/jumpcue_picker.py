"""JumpCUE picker, as the key picker (ui.edit_panel.key_picker) is: a button showing the cue picked with its group's
colour as a strip on its left, a click opening a popup of the track's JumpCUEs, each a cell in its group's colour
(its letter over its time, by letter, eight a row), one click picks; with a "none" cell first where none may be
picked (the Export tab's cue jumped from: no jump)."""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from analyzer.music.marks import JumpCue

Entry = tuple[JumpCue, tuple[int, int, int]]    # a cue and its group's colour (core.marks.colors.jump_cue_colors)


def cue_time(seconds: float) -> str:
    """A cue's time as m:ss.s."""
    minutes, rest = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def text_on(rgb: tuple[int, int, int]) -> str:
    """Black or white, whichever reads better on ``rgb`` (the higher contrast by WCAG's relative luminance)."""
    def linear(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * linear(rgb[0]) + 0.7152 * linear(rgb[1]) + 0.0722 * linear(rgb[2])
    return "rgb(0, 0, 0)" if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else "rgb(255, 255, 255)"


class JumpCuePickerPopup(QtWidgets.QFrame):
    picked = QtCore.Signal(object)       # the JumpCue, or None (the none cell)

    CELL_W, CELL_H, COLUMNS = 56, 38, 8

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Popup)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(2)

    def show_for(self, anchor: QtWidgets.QWidget, entries: list[Entry], current: Optional[JumpCue],
                 none_text: Optional[str]) -> None:
        """Open under ``anchor`` with ``entries``' cells (``current`` outlined), the none cell first when
        ``none_text`` is given."""
        from ui.theme import current_theme
        theme = current_theme()
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        row = 0
        if none_text:
            cell = QtWidgets.QToolButton()
            cell.setText(none_text)
            cell.setFixedHeight(self.CELL_H // 2 + 6)
            cell.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            ring = theme.text if current is None else theme.border
            cell.setStyleSheet(
                f"QToolButton {{ background-color: {theme.base}; color: {theme.text}; border: 2px solid {ring};"
                f" border-radius: 2px; padding: 0 6px; }} QToolButton:hover {{ border-color: {theme.text}; }}")
            cell.clicked.connect(lambda _=False: self._pick(None))
            self._grid.addWidget(cell, 0, 0, 1, self.COLUMNS)
            row = 1
        for k, (cue, rgb) in enumerate(entries):
            cell = QtWidgets.QToolButton()
            cell.setText(f"{cue.label}\n{cue_time(cue.point)}")
            name = cue.comment.strip()
            cell.setToolTip(f"{cue.label}  {cue_time(cue.point)}" + (f"  {name}" if name and name != cue.label else ""))
            cell.setFixedSize(self.CELL_W, self.CELL_H)
            bg, fg = "rgb({}, {}, {})".format(*rgb), text_on(rgb)
            ring = fg if cue == current else bg
            cell.setStyleSheet(
                f"QToolButton {{ background-color: {bg}; color: {fg}; border: 2px solid {ring};"
                " border-radius: 2px; font-weight: bold; padding: 0; }"
                f"QToolButton:hover {{ border-color: {fg}; }}")
            cell.clicked.connect(lambda _=False, c=cue: self._pick(c))
            self._grid.addWidget(cell, row + k // self.COLUMNS, k % self.COLUMNS)
        self.adjustSize()
        pos = anchor.mapToGlobal(QtCore.QPoint(0, anchor.height() + 2))
        screen = anchor.screen().availableGeometry() if anchor.screen() else None
        if screen is not None:
            pos.setX(min(pos.x(), screen.right() - self.width()))
            if pos.y() + self.height() > screen.bottom():
                pos.setY(anchor.mapToGlobal(QtCore.QPoint(0, 0)).y() - self.height() - 2)
        self.move(pos)
        self.show()

    def _pick(self, cue: Optional[JumpCue]) -> None:
        self.hide()
        self.picked.emit(cue)


class JumpCuePickButton(QtWidgets.QToolButton):
    """The JumpCUE picked, its group's colour a strip on the left; a click opens the JumpCuePickerPopup.
    ``none_text``: none may be picked (shown so); else a cue is always picked while there are any."""

    cueChanged = QtCore.Signal(object)    # the JumpCue, or None

    def __init__(self, none_text: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._none_text = none_text
        self._entries: list[Entry] = []
        self._cue: Optional[JumpCue] = None
        self._popup = JumpCuePickerPopup(self)
        self._popup.picked.connect(self.setCue)
        self.clicked.connect(lambda: self._popup.show_for(self, self._entries, self._cue, self._none_text))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._restyle()

    def setCues(self, entries: list[Entry]) -> None:
        """The cues to pick from (each with its group's colour), by letter; the one picked kept where it is among
        them, else none (or the first, where none may not be picked)."""
        self._entries = sorted(entries, key=lambda e: (e[0].label, e[0].point))
        cues = self.cues()
        self.setCue(self._cue if self._cue in cues else (None if self._none_text or not cues else cues[0]),
                    force=True)

    def cues(self) -> list[JumpCue]:
        return [cue for cue, _rgb in self._entries]

    def cue(self) -> Optional[JumpCue]:
        return self._cue

    def setCue(self, cue: Optional[JumpCue], force: bool = False) -> None:
        if cue == self._cue and not force:
            return
        changed = cue != self._cue
        self._cue = cue
        self._restyle()
        if changed:
            self.cueChanged.emit(cue)

    def _restyle(self) -> None:
        from ui.theme import current_theme
        theme = current_theme()
        rgb = next((rgb for cue, rgb in self._entries if cue == self._cue), None)
        if self._cue is not None:
            self.setText(f"{self._cue.label}  {cue_time(self._cue.point)}")
        else:
            self.setText(self._none_text or "-")
        # dressed like the text inputs beside it (as the key picker's button): the colour only a strip on the left
        swatch = "rgb({}, {}, {})".format(*rgb) if rgb else theme.separator
        self.setStyleSheet(
            f"QToolButton {{ background-color: {theme.base}; color: {theme.text};"
            f" border: 1px solid {theme.border}; border-left: 4px solid {swatch};"
            f" border-radius: {theme.radius_small}px; padding: 0 6px; text-align: left; }}"
            f"QToolButton:hover {{ border-color: {theme.border_strong}; border-left-color: {swatch}; }}"
            f"QToolButton:disabled {{ background-color: {theme.window}; color: {theme.text_disabled};"
            f" border-color: {theme.separator}; border-left-color: {theme.separator}; }}"
        )
