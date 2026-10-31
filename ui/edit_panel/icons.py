"""Edit panel icons: transparent PNGs in assets/icons (beatgrid_<name>.png), and the library's playlist folders' in
the same glyphs' grammar (library_folder*.png)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PySide6 import QtCore, QtGui

from system.resource_paths import resource_path

# A tool whose action follows an edit mode shows the mode under its glyph (in the 72 px icon):
#   scope_all / scope_segment: three blocks like the scope buttons' icons, all filled (the whole
#     grid / track) or the middle one (the current segment);
#   step_beat / step_bar: one dot (a beat) or four (a bar), as under the BEAT / BAR buttons.
BADGES = ("scope_all", "scope_segment", "step_beat", "step_bar")
_BADGE_TOP, _BADGE_H, _BADGE_X0, _BADGE_X1, _BADGE_GAP = 55, 15, 6, 66, 5
_GLYPH_BOTTOM = _BADGE_TOP - 5    # the glyph ends a gap above the badge
_STROKE = 4.5                     # an unfilled block's outline
_COLOR = QtGui.QColor("#e6e6e6")


def _with_disabled(pm: QtGui.QPixmap) -> QtGui.QIcon:
    """The icon of ``pm``; the disabled state is the same glyph faded, so it greys
    out with its button on the dark theme."""
    ic = QtGui.QIcon(pm)
    if not pm.isNull():
        faded = QtGui.QPixmap(pm.size())
        faded.fill(QtGui.QColor(0, 0, 0, 0))
        p = QtGui.QPainter(faded)
        p.setOpacity(0.3)
        p.drawPixmap(0, 0, pm)
        p.end()
        ic.addPixmap(faded, QtGui.QIcon.Disabled)
    return ic


@lru_cache(maxsize=None)
def icon(name: str, inverted: bool = False) -> QtGui.QIcon:
    """Icon from assets/icons/<name>.png. ``inverted`` inverts the image's colours
    (alpha kept), for a button lit in a light colour."""
    pm = QtGui.QPixmap(str(resource_path(f"assets/icons/{name}.png")))
    if inverted and not pm.isNull():
        # the image's colour channels inverted as they are, alpha kept
        img = pm.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        img.invertPixels(QtGui.QImage.InvertRgb)
        pm = QtGui.QPixmap.fromImage(img)
    return _with_disabled(pm)


def _draw_badge(p: QtGui.QPainter, badge: str, width: int) -> None:
    """The badge across the middle 60 px of an icon ``width`` wide."""
    x0 = _BADGE_X0 + (width - 72) / 2
    span = _BADGE_X1 - _BADGE_X0
    if badge.startswith("scope_"):
        w = (span - 2 * _BADGE_GAP) / 3
        for k in range(3):
            rect = QtCore.QRectF(x0 + k * (w + _BADGE_GAP), _BADGE_TOP, w, _BADGE_H)
            if badge == "scope_all" or k == 1:
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(_COLOR)
                p.drawRoundedRect(rect, 2.5, 2.5)
            else:
                half = _STROKE / 2
                p.setBrush(QtCore.Qt.NoBrush)
                p.setPen(QtGui.QPen(_COLOR, _STROKE))
                p.drawRoundedRect(rect.adjusted(half, half, -half, -half), 2, 2)
        return
    d = 11.0  # four of them stay apart at 19 px
    n = 1 if badge == "step_beat" else 4
    step = (span - d) / 3
    first = x0 + span / 2 - d / 2 if n == 1 else x0
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(_COLOR)
    for k in range(n):
        p.drawEllipse(QtCore.QRectF(first + k * step, _BADGE_TOP + (_BADGE_H - d) / 2, d, d))


@lru_cache(maxsize=None)
def badged_icon_beside(name: str, badge: str, inverted: bool = False) -> QtGui.QIcon:
    """``name``'s square icon with ``badge`` beside it (a 2:1 icon, shown at 38 x 19): for a glyph
    that fills its icon's height and would have to shrink to take the badge under it; ``inverted``
    as icon()'s."""
    src = QtGui.QImage(str(resource_path(f"assets/icons/{name}.png")))
    if src.isNull():
        return QtGui.QIcon()
    out = QtGui.QImage(src.width() * 2, src.height(), QtGui.QImage.Format_ARGB32)
    out.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(out)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.drawImage(QtCore.QPoint(4, 0), src)
    p.translate(src.width() + 2, (src.height() - _BADGE_H) / 2 - _BADGE_TOP)
    _draw_badge(p, badge, src.width())
    p.end()
    return _with_disabled(_inverted(out, inverted))


def _inverted(img: QtGui.QImage, inverted: bool) -> QtGui.QPixmap:
    """``img`` as a pixmap, its colours inverted (alpha kept) where ``inverted``: dark on a lit button."""
    if inverted:
        img = img.convertToFormat(QtGui.QImage.Format_ARGB32)
        img.invertPixels(QtGui.QImage.InvertRgb)
    return QtGui.QPixmap.fromImage(img)


@lru_cache(maxsize=None)
def badged_icon(name: str, badge: str, inverted: bool = False) -> QtGui.QIcon:
    """``name``'s icon with ``badge`` (one of BADGES) under its glyph: the glyph is moved up
    into its top margin, scaled down only when it is too tall to fit above the badge; ``inverted``
    as icon()'s."""
    src = QtGui.QImage(str(resource_path(f"assets/icons/{name}.png")))
    if src.isNull():
        return QtGui.QIcon()
    src = src.convertToFormat(QtGui.QImage.Format_ARGB32)
    alpha = np.frombuffer(src.constBits(), np.uint8).reshape(src.height(), src.bytesPerLine())
    ink = alpha[:, 3:src.width() * 4:4] > 8
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if not rows.size:
        return icon(name, inverted)
    box = QtCore.QRect(int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))
    out = QtGui.QImage(src.size(), QtGui.QImage.Format_ARGB32)
    out.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(out)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
    if box.height() <= _GLYPH_BOTTOM:
        # whole pixels: the glyph as it is, its bottom at the gap above the badge
        p.drawImage(QtCore.QPoint(0, _GLYPH_BOTTOM - (box.top() + box.height())), src)
    else:
        scale = _GLYPH_BOTTOM / box.height()
        w = box.width() * scale
        target = QtCore.QRectF(box.center().x() + 0.5 - w / 2, 0, w, _GLYPH_BOTTOM)
        p.drawImage(target, src, QtCore.QRectF(box))
    _draw_badge(p, badge, src.width())
    p.end()
    return _with_disabled(_inverted(out, inverted))


def folder_icon(opened: bool) -> QtGui.QIcon:
    """A playlist folder in the library's sources panel: closed (library_folder.png) or open
    (library_folder_open.png)."""
    return icon("library_folder_open" if opened else "library_folder")
