"""The deck's record, turning as the track plays (33 1/3 rpm): the album art as its label (else the record image
Settings > View names, else a drawn record), one even white rim with a red stretch at 12 o'clock turning with it.
Its turns are drawn in coarse steps and kept, so a frame only sets a picture already made. The main window's track
info and Deck Monitor's decks show it."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from system.resource_paths import setting_path

RPM = 33.3333333
STEP_DEG = 2            # rotate in coarse steps to limit redraw work


class RecordDisc(QtWidgets.QLabel):
    """``diameter`` its size; ``record_image()`` the record image's path setting (used without album art)."""

    def __init__(self, diameter: int, record_image: Callable[[], Optional[str]], parent=None) -> None:
        super().__init__(parent)
        self.diameter = int(diameter)
        self._record_image = record_image
        self.setFixedSize(self.diameter, self.diameter)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self._rotation_cache: dict[int, QtGui.QPixmap] = {}
        self._last_cache_key: Optional[int] = None
        self._base_pix = QtGui.QPixmap()
        self._base_dpr = 1.0
        self._time = 0.0
        self._album_art: Optional[QtGui.QImage] = None
        self._load(None)

    def set_album_art(self, album_art: Optional[QtGui.QImage]) -> None:
        self._album_art = album_art
        self._load(album_art)

    def reload(self) -> None:
        """Drawn again (the record image setting changed)."""
        self._load(self._album_art)

    def set_time(self, time_sec: float) -> None:
        """Turned to where the track is."""
        self._time = time_sec
        base = self._base_pix
        if base.isNull():
            return
        dpr = self.devicePixelRatioF()
        if abs(dpr - self._base_dpr) > 1e-6:
            self._load(self._album_art)
            base = self._base_pix
            if base.isNull():
                return
        angle = ((time_sec * (RPM / 60) * 360) % 360)
        cache_key = int(round(angle / STEP_DEG) * STEP_DEG) % 360
        if cache_key == self._last_cache_key:
            return
        self._last_cache_key = cache_key
        rotated = self._rotation_cache.get(cache_key)
        if rotated is None:
            logical_size = self.size()
            pixel_w = max(1, int(round(logical_size.width() * dpr)))
            pixel_h = max(1, int(round(logical_size.height() * dpr)))
            rotated = QtGui.QPixmap(pixel_w, pixel_h)
            rotated.setDevicePixelRatio(dpr)
            rotated.fill(QtCore.Qt.transparent)

            painter = QtGui.QPainter(rotated)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.translate(logical_size.width() / 2, logical_size.height() / 2)
            painter.rotate(cache_key)
            painter.translate(-logical_size.width() / 2, -logical_size.height() / 2)
            painter.drawPixmap(0, 0, base)
            painter.end()
            self._rotation_cache[cache_key] = rotated
        self.setPixmap(rotated)

    def _load(self, album_art: Optional[QtGui.QImage]) -> None:
        size = self.diameter
        dpr = self.devicePixelRatioF()
        pixel_size = max(1, int(round(size * dpr)))
        base = QtGui.QPixmap(pixel_size, pixel_size)
        base.setDevicePixelRatio(dpr)
        base.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(base)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        outer_rect = QtCore.QRectF(2, 2, size - 4, size - 4)

        painter.setBrush(QtGui.QColor("#111"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#444"), 2))
        painter.drawEllipse(outer_rect)

        used_any = False
        if album_art is not None and not album_art.isNull():
            src_pm = QtGui.QPixmap.fromImage(album_art)
            w, h = src_pm.width(), src_pm.height()
            side = min(w, h)
            x = (w - side) // 2
            y = (h - side) // 2
            label_margin = int(size * 0.19)
            label_size = size - 2 * label_margin
            square = src_pm.copy(x, y, side, side).scaled(
                label_size,
                label_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            circ = QtGui.QPixmap(label_size, label_size)
            circ.fill(QtCore.Qt.transparent)
            pp = QtGui.QPainter(circ)
            pp.setRenderHint(QtGui.QPainter.Antialiasing, True)
            path = QtGui.QPainterPath()
            path.addEllipse(0, 0, label_size, label_size)
            pp.setClipPath(path)
            pp.drawPixmap(0, 0, square)
            pp.end()
            painter.drawPixmap(label_margin, label_margin, circ)
            used_any = True

        if not used_any:
            try:
                cfg_path = self._record_image()
            except Exception:
                cfg_path = None
            path = cfg_path or "assets/vinyl.png"
            vinyl = QtGui.QPixmap(setting_path(path)) if path else QtGui.QPixmap()
            if not vinyl.isNull():
                vinyl = vinyl.scaled(
                    size - 6,
                    size - 6,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                vx = (size - vinyl.width()) // 2
                vy = (size - vinyl.height()) // 2
                painter.drawPixmap(vx, vy, vinyl)
            else:
                painter.setBrush(QtGui.QColor("#333"))
                painter.setPen(QtGui.QPen(QtGui.QColor("#666"), 1))
                painter.drawEllipse(
                    outer_rect.adjusted(
                        size * 0.17, size * 0.17, -size * 0.17, -size * 0.17
                    )
                )
                painter.setPen(QtGui.QPen(QtGui.QColor("#555"), 1))
                for r in range(8, (size // 2) - 8, 3):
                    painter.drawEllipse(QtCore.QRectF(size / 2 - r, size / 2 - r, 2 * r, 2 * r))

        # Rim: one even white ring (balanced, so the spinning record never looks
        # off-centre), with the upright marker as a red stretch of the same ring
        # at 12 o'clock, turning with the record.
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor("#e6e6e6"), 2.5))
        painter.drawEllipse(outer_rect)
        marker_pen = QtGui.QPen(QtGui.QColor("#e5484d"), 2.5)
        marker_pen.setCapStyle(QtCore.Qt.FlatCap)
        painter.setPen(marker_pen)
        painter.drawArc(outer_rect, (90 - 10) * 16, 20 * 16)

        painter.end()

        self._base_pix = base
        self._base_dpr = dpr
        self._rotation_cache.clear()
        self._last_cache_key = None
        self.set_time(self._time)
