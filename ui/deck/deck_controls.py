"""The toolbar row's controls, one look per kind, so a control shows what it does: a mode is a segmented switch
(its choices joined, the one in use filled in the accent); a switch that turns something on is a lamp (outlined in
its colour while off, filled while on: CUE yellow as a mixer's, the rest blue). Each group is a block as the
settings' group boxes are (a frame, its title at its left, the controls after it, filling it); each part of the row
starts with its title (CH 1, MASTER: one width) and the blocks fill the window's width: the room left goes to the controls
that widen, about the same to each (``fill``), so a wide window neither makes one control a giant nor leaves a frame
empty, and a block of knobs alone keeps its width. Its margins are tight, so the row at its narrowest still shows
everything whole. What is lit is what the deck reports: a click is a request."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme

HEIGHT = 20
TITLE_W = 48
KNOB = 18   # a knob's size in a block
CONTROL_SHARE = 80  # px a widening control counts for beside a block's own width, when the row's room is shared (fill)
PADDING = 3         # a button's text from its sides
ROW_SPACING = 4     # between the row's blocks


def _rgba(color: str, alpha: float) -> str:
    c = QtGui.QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.2f})"


def _bold(widget: QtWidgets.QWidget, size: int = 8) -> None:
    f = widget.font()
    f.setPointSize(size)
    f.setBold(True)
    widget.setFont(f)


def text_room(button: QtWidgets.QAbstractButton, text: str) -> int:
    """The width ``button`` needs for ``text`` uncut: the text, its padding and the border."""
    return button.fontMetrics().horizontalAdvance(text) + 2 * PADDING + 2


class _Lit(QtWidgets.QToolButton):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        _bold(self)
        self.setFixedHeight(HEIGHT)
        self.setProperty("lit", False)
        self.setMinimumWidth(text_room(self, text))   # the row at its narrowest cuts no text

    def set_lit(self, lit: bool) -> None:
        if bool(self.property("lit")) != bool(lit):
            self.setProperty("lit", bool(lit))
            self.style().unpolish(self)
            self.style().polish(self)


class Segment(_Lit):
    """One choice of a mode (segments() joins them)."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        theme = current_theme()
        r = 4
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {theme.border}; background: {theme.base}; color: {theme.text_dim};"
            f" padding: 0 {PADDING}px; border-radius: 0; }}\n"
            f"QToolButton[place=first] {{ border-top-left-radius: {r}px; border-bottom-left-radius: {r}px; }}\n"
            f"QToolButton[place=last] {{ border-left: none; border-top-right-radius: {r}px;"
            f" border-bottom-right-radius: {r}px; }}\n"
            f"QToolButton[place=middle] {{ border-left: none; }}\n"
            f"QToolButton:hover {{ color: {theme.text}; background: {theme.raised}; }}\n"
            f"QToolButton[lit=true] {{ background: {theme.accent}; border-color: {theme.accent};"
            f" color: {theme.on_accent}; }}\n"
            f"QToolButton:disabled {{ color: {theme.text_disabled}; }}")


def segments(buttons: list[Segment]) -> QtWidgets.QWidget:
    """The choices of one mode joined into a segmented switch (as wide as its block gives it, its choices alike)."""
    box = QtWidgets.QWidget()
    box.setProperty("controls", len(buttons))
    lay = QtWidgets.QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    for k, b in enumerate(buttons):
        place = "first" if k == 0 else "last" if k == len(buttons) - 1 else "middle"
        b.setProperty("place", place)
        b.style().unpolish(b)
        b.style().polish(b)
        b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(b, 1)
    return box


class Lamp(_Lit):
    """A switch that turns something on, in ``color`` (outlined while off, filled while on)."""

    def __init__(self, text: str, color: str | None = None, parent=None) -> None:
        super().__init__(text, parent)
        theme = current_theme()
        color = color or theme.accent
        on_text = "#111111" if QtGui.QColor(color).lightnessF() > 0.55 else "#ffffff"
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {_rgba(color, 0.45)}; background: {theme.raised};"
            f" color: {_rgba(color, 0.85)}; border-radius: 3px; padding: 0 {PADDING}px; }}\n"
            f"QToolButton:hover {{ border-color: {color}; }}\n"
            f"QToolButton[lit=true] {{ background: {color}; border-color: {color}; color: {on_text}; }}\n"
            f"QToolButton:disabled {{ color: {theme.text_disabled}; border-color: {theme.border}; }}")


def label(text: str) -> QtWidgets.QLabel:
    """A group's or a knob's name, small."""
    lbl = QtWidgets.QLabel(text)
    _bold(lbl, 7)
    lbl.setStyleSheet(f"color: {current_theme().text_dim};")
    return lbl


def row_title(text: str) -> QtWidgets.QLabel:
    """A row's title (CH 1, MASTER), one width for every row."""
    lbl = QtWidgets.QLabel(text)
    _bold(lbl, 7)
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setFixedSize(TITLE_W, HEIGHT - 4)
    theme = current_theme()
    lbl.setStyleSheet(f"color: {theme.text}; background: {theme.border}; border-radius: 3px;")
    return lbl


BLOCK_H = HEIGHT + 2


def controls_in(widget: QtWidgets.QWidget) -> int:
    """How many controls that widen ``widget`` is or holds: a button or a combo box is one; a segmented switch, a block
    or a row of blocks says (its ``controls``); a knob or a label none."""
    if isinstance(widget, (QtWidgets.QAbstractButton, QtWidgets.QComboBox)):
        return 1
    return int(widget.property("controls") or 0)


def block(title: str, items: list, spacing: int = 3) -> QtWidgets.QFrame:
    """A group of controls (``items``: widgets, an int a gap), framed as the settings' group boxes are, one control
    high: its title at its left, the controls filling the room after it, its buttons, combo boxes and segmented switches
    widening with it (``controls_in``: knobs and labels keep their size). A block of knobs alone keeps its width (fill),
    so no frame is left empty."""
    box = QtWidgets.QFrame()
    box.setObjectName("deckBlock")
    theme = current_theme()
    box.setStyleSheet(f"QFrame#deckBlock {{ border: 1px solid {theme.border}; border-radius: 4px; }}")
    lay = QtWidgets.QHBoxLayout(box)
    lay.setContentsMargins(5, 0, 4, 0)
    lay.setSpacing(spacing)
    lay.addWidget(label(title), 0, QtCore.Qt.AlignVCenter)
    lay.addSpacing(3)
    controls = 0
    for item in items:
        if isinstance(item, int):
            lay.addSpacing(item)
            continue
        if isinstance(item, QtWidgets.QComboBox):   # one control high
            font = item.font()
            font.setPointSize(9)
            item.setFont(font)
            item.setFixedHeight(HEIGHT)
        widens = controls_in(item)
        controls += widens
        if widens:
            item.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(item, widens, QtCore.Qt.AlignVCenter)
    box.setProperty("controls", controls)
    box.setFixedHeight(BLOCK_H)
    box.setSizePolicy(QtWidgets.QSizePolicy.Expanding if controls else QtWidgets.QSizePolicy.Fixed,
                      QtWidgets.QSizePolicy.Fixed)
    return box


def fill(row: QtWidgets.QHBoxLayout, widgets: list) -> None:
    """``widgets`` along ``row``, the row filled end to end: the room shared by their widths and the controls in them
    that widen (CONTROL_SHARE each), so each control takes about the same and a wide block's one combo box no giant's
    share; one with no such control stays as wide as it is."""
    for w in widgets:
        controls = controls_in(w)
        row.addWidget(w, w.sizeHint().width() + CONTROL_SHARE * controls if controls else 0)


def blocks_row(widget: QtWidgets.QWidget, blocks: list) -> None:
    """``widget`` as a row of ``blocks`` (fill), wide as its row allows."""
    row = QtWidgets.QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(ROW_SPACING)
    fill(row, blocks)
    controls = sum(controls_in(b) for b in blocks)
    widget.setProperty("controls", controls)
    widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding if controls else QtWidgets.QSizePolicy.Fixed,
                         QtWidgets.QSizePolicy.Fixed)
