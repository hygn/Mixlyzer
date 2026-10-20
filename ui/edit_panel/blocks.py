"""Building blocks of the edit pages: a page is blocks three button rows high -- the edit
modes, the tools, and a group of its own (time signatures, listening aids, labels) --
with a thin rule between neighbouring blocks.

Every page lays its blocks and tools out alike, so a function sits in the same place on each:
the modifiers of the tools' actions (edit modes) all at the far left, then the tools, then the
page's own group.
  row 1  moves in time (beatgrid shift, boundary nudges), then the reanalysis: the whole
         track (key_reanalyze_all icon), then the current segment (beatgrid_reanalyze icon);
  row 2  the value (reference BPM / key / beat counter, CUE points);
  row 3  split, merge previous, merge next, the page's own structure edit (downbeat, merge
         same key, unassign), then relative value changes (BPM x1/2 / x2, relative key).
The beatgrid's reanalysis follows its reference BPM, so it ends row 2 there."""
from __future__ import annotations

from PySide6 import QtWidgets

from ui.theme import current_theme


def rule_cell() -> QtWidgets.QWidget:
    """The thin vertical rule between two blocks (put it in a cell spanning the three rows)."""
    rule = QtWidgets.QFrame()
    rule.setFixedWidth(1)
    rule.setStyleSheet(f"background: {current_theme().border};")
    cell = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(cell)
    lay.setContentsMargins(4, 2, 4, 2)
    lay.addWidget(rule)
    return cell


def mode_column(top: QtWidgets.QAbstractButton, bottom: QtWidgets.QAbstractButton) -> QtWidgets.QWidget:
    """An edit mode's two buttons, one over the other, filling a column three button rows high
    (the panel gives them their heights: EditPanel._mode_pairs)."""
    cell = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(cell)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(top)
    lay.addStretch(1)
    lay.addWidget(bottom)
    return cell
