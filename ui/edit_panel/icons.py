"""Edit panel icons: transparent PNGs in assets/icons (beatgrid_<name>.png)."""
from __future__ import annotations

from functools import lru_cache

from PySide6 import QtGui

from core.resource_paths import resource_path

@lru_cache(maxsize=None)
def icon(name: str, inverted: bool = False) -> QtGui.QIcon:
    """Icon from assets/icons/<name>.png; the disabled state is the same glyph faded,
    so it greys out with its button on the dark theme. ``inverted`` inverts the
    image's colours (alpha kept), for a button lit in a light colour."""
    pm = QtGui.QPixmap(str(resource_path(f"assets/icons/{name}.png")))
    if inverted and not pm.isNull():
        # the image's colour channels inverted as they are, alpha kept
        img = pm.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
        img.invertPixels(QtGui.QImage.InvertRgb)
        pm = QtGui.QPixmap.fromImage(img)
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
