"""The toolbar rows' controls, one look per kind, so a control shows what it does: a mode is a segmented switch
(its choices joined, the one in use filled in the accent); a switch that turns something on is a lamp (outlined in
its colour while off, filled while on: CUE yellow as a mixer's, Beat FX ON red, the rest blue); a momentary action
is a pill outlined only (TAP). Each group is a block as the settings' group boxes are (a frame, its title on its
top edge, the controls in its middle); a row starts with its title (CH 1, MASTER: one width, so the rows line up)
and its blocks fill the window's width, each widened in proportion to its own (``fill``). What is lit is what the
deck reports: a click is a request."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme

HEIGHT = 20
TITLE_W = 58
KNOB = 18   # a knob's size in a block


def _rgba(color: str, alpha: float) -> str:
    c = QtGui.QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha:.2f})"


def _bold(widget: QtWidgets.QWidget, size: int = 8) -> None:
    f = widget.font()
    f.setPointSize(size)
    f.setBold(True)
    widget.setFont(f)


class _Lit(QtWidgets.QToolButton):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        _bold(self)
        self.setFixedHeight(HEIGHT)
        self.setProperty("lit", False)

    def set_lit(self, lit: bool) -> None:
        if bool(self.property("lit")) != bool(lit):
            self.setProperty("lit", bool(lit))
            self.style().unpolish(self)
            self.style().polish(self)


class Segment(_Lit):
    """One choice of a mode (segments() joins them)."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setMinimumWidth(40)
        theme = current_theme()
        r = 4
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {theme.border}; background: {theme.base}; color: {theme.text_dim};"
            f" padding: 0 6px; border-radius: 0; }}\n"
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
    box.setProperty("grow", True)
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
        self.setMinimumWidth(34)
        theme = current_theme()
        color = color or theme.accent
        on_text = "#111111" if QtGui.QColor(color).lightnessF() > 0.55 else "#ffffff"
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {_rgba(color, 0.45)}; background: {theme.raised};"
            f" color: {_rgba(color, 0.85)}; border-radius: 3px; padding: 0 6px; }}\n"
            f"QToolButton:hover {{ border-color: {color}; }}\n"
            f"QToolButton[lit=true] {{ background: {color}; border-color: {color}; color: {on_text}; }}\n"
            f"QToolButton:disabled {{ color: {theme.text_disabled}; border-color: {theme.border}; }}")


class Momentary(QtWidgets.QToolButton):
    """A momentary action: a pill outlined only."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        _bold(self)
        self.setFixedHeight(HEIGHT)
        self.setMinimumWidth(40)
        theme = current_theme()
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {theme.border_strong}; background: transparent; color: {theme.text};"
            f" border-radius: {HEIGHT // 2}px; padding: 0 10px; }}\n"
            f"QToolButton:hover {{ background: {theme.raised_hover}; }}\n"
            f"QToolButton:pressed {{ background: {theme.raised_pressed}; border-color: {theme.text}; }}\n"
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


def grows(widget: QtWidgets.QWidget) -> bool:
    """A control that widens with its block: a button, a combo box, or one marked ``grow`` (a segmented switch, a
    pad)."""
    return isinstance(widget, (QtWidgets.QAbstractButton, QtWidgets.QComboBox)) or bool(widget.property("grow"))


def block(title: str, items: list, spacing: int = 4) -> QtWidgets.QFrame:
    """A group of controls (``items``: widgets, an int a gap), framed as the settings' group boxes are, one control
    high: its title at its left, the controls in the room after it; as wide as its row gives it (fill), its buttons,
    combo boxes, segmented switches and pads widening with it (``grows``: knobs and labels keep their size; a block
    of none of them keeps them in its middle)."""
    box = QtWidgets.QFrame()
    box.setObjectName("deckBlock")
    theme = current_theme()
    box.setStyleSheet(f"QFrame#deckBlock {{ border: 1px solid {theme.border}; border-radius: 4px; }}")
    lay = QtWidgets.QHBoxLayout(box)
    lay.setContentsMargins(7, 0, 6, 0)
    lay.setSpacing(spacing)
    lay.addWidget(label(title), 0, QtCore.Qt.AlignVCenter)
    lay.addSpacing(6)
    growing = any(grows(item) for item in items if not isinstance(item, int))
    if not growing:
        lay.addStretch(1)
    for item in items:
        if isinstance(item, int):
            lay.addSpacing(item)
            continue
        if isinstance(item, QtWidgets.QComboBox):   # one control high
            font = item.font()
            font.setPointSize(9)
            item.setFont(font)
            item.setFixedHeight(HEIGHT)
        if grows(item):
            item.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            lay.addWidget(item, 1, QtCore.Qt.AlignVCenter)
        else:
            lay.addWidget(item, 0, QtCore.Qt.AlignVCenter)
    if not growing:
        lay.addStretch(1)
    box.setFixedHeight(BLOCK_H)
    box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    return box


def fill(row: QtWidgets.QHBoxLayout, widgets: list) -> None:
    """``widgets`` along ``row``, the room left shared in proportion to their widths (the row filled end to end,
    each block as it was, only wider)."""
    for w in widgets:
        row.addWidget(w, max(1, w.sizeHint().width()))


def blocks_row(widget: QtWidgets.QWidget, blocks: list) -> None:
    """``widget`` as a row of ``blocks`` (fill), wide as its row allows."""
    row = QtWidgets.QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    fill(row, blocks)
    widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
