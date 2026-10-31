"""Tooltips app-wide (installed once at the start, ``install``): a widget's tooltip shown as rich text, wrapped at
MAX_WIDTH when it would be wider (a plain text's lines are never wrapped by Qt itself, so a long one ran across
the screen), with the shortcut of the action the widget works (``tag``: ui.shortcuts' action ids, its keys as
they are now: Settings > Shortcuts' changes too) after it in parentheses. A tooltip a widget draws itself (a
table's cells, a plot's items) is left as it is."""
from __future__ import annotations

import html
import re
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

MAX_WIDTH = 420           # px: wider tooltips wrap
SHORTCUT_PROPERTY = "shortcut_action"
_HAND_WRAP = re.compile(r"[ \t]*\n[ \t]*(?=[a-z(])")

_keys_of: Optional[Callable[[str], list[QtGui.QKeySequence]]] = None
_filter: Optional[QtCore.QObject] = None


def tag(widget: QtWidgets.QWidget, action_id: str) -> None:
    """``widget`` works shortcut ``action_id``: its keys are shown after its tooltip."""
    widget.setProperty(SHORTCUT_PROPERTY, action_id)


def set_shortcut_keys(keys_of: Optional[Callable[[str], list[QtGui.QKeySequence]]]) -> None:
    """Where an action's keys are read (ui.shortcuts.ShortcutManager.keys_of: the keys in use; None: no keys)."""
    global _keys_of
    _keys_of = keys_of


def formatted(text: str, keys: list[QtGui.QKeySequence] = ()) -> str:
    """``text`` (plain or rich) as the rich text shown: ``keys`` in parentheses after it, wrapped at MAX_WIDTH when
    it is wider."""
    rich = QtGui.Qt.mightBeRichText(text)
    if not rich:
        # a line wrapped by hand (the next one goes on in lower case) joined, so it wraps at the width; a line
        # starting anew (a sentence, a list's item) kept
        text = _HAND_WRAP.sub(" ", text)
    body = text if rich else html.escape(text).replace("\n", "<br>")
    if keys:
        names = ", ".join(k.toString(QtGui.QKeySequence.NativeText) for k in keys)
        hint = f" ({html.escape(names)})"
        end = body.rstrip()
        body = (end[:-4] + hint + "</p>") if end.lower().endswith("</p>") else body + hint
    doc = QtGui.QTextDocument()
    doc.setDefaultFont(QtWidgets.QToolTip.font())
    doc.setHtml(body)
    if doc.idealWidth() > MAX_WIDTH:
        return f"<table width='{MAX_WIDTH}' cellspacing='0' cellpadding='0'><tr><td>{body}</td></tr></table>"
    return body if rich else f"<qt>{body}</qt>"


class _ToolTips(QtCore.QObject):
    def eventFilter(self, obj, event) -> bool:
        if event.type() != QtCore.QEvent.ToolTip or not isinstance(obj, QtWidgets.QWidget):
            return False
        text = obj.toolTip()
        if not text:
            return False     # none of its own: its parent's, or one it draws itself
        action = obj.property(SHORTCUT_PROPERTY)
        keys = _keys_of(action) if (action and _keys_of is not None) else []
        QtWidgets.QToolTip.showText(event.globalPos(), formatted(text, keys), obj, QtCore.QRect(),
                                    obj.toolTipDuration())
        return True


def install() -> None:
    """The tooltips formatted from now on (the application made)."""
    global _filter
    if _filter is None:
        _filter = _ToolTips()
        QtWidgets.QApplication.instance().installEventFilter(_filter)
