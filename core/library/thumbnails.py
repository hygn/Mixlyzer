"""Album art thumbnails stored in the library: ``{libpath}/thumbs/{uid}.jpg``."""

from __future__ import annotations

import os
from typing import Optional

from PySide6 import QtCore, QtGui

THUMBS_DIR = "thumbs"
# Width (px) the art is downsampled to: sharp at twice the library's art column.
THUMB_WIDTH = 128
THUMB_QUALITY = 88


def thumbnail_path(libpath: str, uid: str) -> str:
    return os.path.join(libpath, THUMBS_DIR, f"{uid}.jpg")


def save_thumbnail(libpath: str, uid: str, art: Optional[QtGui.QImage]) -> bool:
    """Store ``art`` downsampled to THUMB_WIDTH as the track's thumbnail (replaced
    atomically); no art removes it. True when one was written."""
    if not uid:
        return False
    path = thumbnail_path(libpath, uid)
    if art is None or art.isNull():
        delete_thumbnail(libpath, uid)
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = art.convertToFormat(QtGui.QImage.Format_RGB32)
    if image.width() > THUMB_WIDTH:
        image = image.scaledToWidth(THUMB_WIDTH, QtCore.Qt.SmoothTransformation)
    temporary = f"{path}.tmp"
    if not image.save(temporary, "JPG", THUMB_QUALITY):
        return False
    os.replace(temporary, path)
    return True


def delete_thumbnail(libpath: str, uid: str) -> None:
    try:
        os.remove(thumbnail_path(libpath, uid))
    except (FileNotFoundError, ValueError):
        pass


def load_thumbnail(libpath: str, uid: str) -> Optional[QtGui.QImage]:
    if not uid:
        return None
    image = QtGui.QImage(thumbnail_path(libpath, uid))
    return None if image.isNull() else image
