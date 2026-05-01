from __future__ import annotations

import math
from typing import Optional

from PySide6 import QtWidgets, QtGui, QtCore

from core.config import config


class MarqueeLabel(QtWidgets.QLabel):
    """QLabel variant that scrolls long text after a short pause."""

    HOLD_MS = 5000
    STEP_MS = 200
    PAD = "   "

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.STEP_MS)
        self._timer.timeout.connect(self._on_tick)
        self._full_text = ""
        self._buffer = ""
        self._scroll_idx = 0
        self._needs_scroll = False
        self._hold_ticks = 0
        self._pending_geometry_check = False

    def set_marquee_text(self, text: str):
        self._full_text = text or ""
        self._buffer = self._full_text + self.PAD
        self._scroll_idx = 0
        self._reset_hold()
        self.setText(self._full_text)
        self._update_scroll_state(force=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scroll_state()

    def _update_scroll_state(self, force: bool = False):
        if not self._full_text:
            self._timer.stop()
            return
        width = self.contentsRect().width()
        if width <= 0:
            if not self._pending_geometry_check:
                self._pending_geometry_check = True
                QtCore.QTimer.singleShot(
                    0, lambda: self._reset_geometry_flag_and_update()
                )
            return
        fm = self.fontMetrics()
        needs = fm.horizontalAdvance(self._full_text) > width
        changed = (needs != self._needs_scroll) or force
        self._needs_scroll = needs
        if not needs:
            self._timer.stop()
            self.setText(self._full_text)
            return
        if changed:
            self._reset_hold()
            self._scroll_idx = 0
            self.setText(self._full_text)
        if not self._timer.isActive():
            self._timer.start()

    def _on_tick(self):
        if not (self._needs_scroll and self._buffer):
            return
        if self._hold_ticks > 0:
            self._hold_ticks -= 1
            return
        self._scroll_idx = (self._scroll_idx + 1) % len(self._buffer)
        if self._scroll_idx == 0:
            self.setText(self._full_text)
            self._reset_hold()
            return
        rotated = self._buffer[self._scroll_idx :] + self._buffer[: self._scroll_idx]
        self.setText(rotated)

    def _reset_geometry_flag_and_update(self):
        self._pending_geometry_check = False
        self._update_scroll_state()

    def _reset_hold(self):
        self._hold_ticks = max(0, math.ceil(self.HOLD_MS / self.STEP_MS))


class VerticalPeakMeter(QtWidgets.QWidget):
    FLOOR_DBFS = -24.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._display_dbfs = self.FLOOR_DBFS
        self._target_dbfs = self.FLOOR_DBFS
        self._peak_hold_dbfs = self.FLOOR_DBFS
        self._hold_ms = 500
        self._fall_db_per_sec = 24.0
        self._peak_fall_db_per_sec = 36.0
        self._peak_hold_deadline = QtCore.QDeadlineTimer(self._hold_ms)
        self._anim = QtCore.QTimer(self)
        self._anim.setInterval(16)
        self._anim.timeout.connect(self._tick)
        self.setFixedWidth(10)
        self.setMinimumHeight(TrackInfoPanel.HEADER_H - 6)
        self.setMaximumHeight(TrackInfoPanel.HEADER_H - 6)
        self._anim.start()

    def set_level_dbfs(self, dbfs: float) -> None:
        value = max(self.FLOOR_DBFS, min(0.0, float(dbfs)))
        self._target_dbfs = value
        if value >= self._display_dbfs:
            self._display_dbfs = value
        if value >= self._peak_hold_dbfs:
            self._peak_hold_dbfs = value
            self._peak_hold_deadline = QtCore.QDeadlineTimer(self._hold_ms)
        self.update()

    def reset_level(self) -> None:
        self._display_dbfs = self.FLOOR_DBFS
        self._target_dbfs = self.FLOOR_DBFS
        self._peak_hold_dbfs = self.FLOOR_DBFS
        self.update()

    def _tick(self) -> None:
        dt = self._anim.interval() / 1000.0
        changed = False
        if self._display_dbfs > self._target_dbfs:
            next_dbfs = max(self._target_dbfs, self._display_dbfs - (self._fall_db_per_sec * dt))
            if abs(next_dbfs - self._display_dbfs) > 1e-6:
                self._display_dbfs = next_dbfs
                changed = True
        if self._peak_hold_deadline.hasExpired():
            next_peak = max(self._display_dbfs, self._peak_hold_dbfs - (self._peak_fall_db_per_sec * dt))
            if abs(next_peak - self._peak_hold_dbfs) > 1e-6:
                self._peak_hold_dbfs = next_peak
                changed = True
        if changed:
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(3, 2, -3, -2)
        painter.fillRect(rect, QtGui.QColor("#111"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#444"), 1))
        painter.drawRect(rect)

        fill_ratio = (self._display_dbfs - self.FLOOR_DBFS) / abs(self.FLOOR_DBFS)
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        fill_h = int(round(rect.height() * fill_ratio))
        if fill_h > 0:
            fill_rect = QtCore.QRect(rect.left() + 1, rect.bottom() - fill_h + 1, rect.width() - 1, fill_h - 1)
            grad_rect = rect.adjusted(1, 1, -1, -1)
            grad = QtGui.QLinearGradient(grad_rect.topLeft(), grad_rect.bottomLeft())
            grad.setColorAt(0.0, QtGui.QColor("#ff4d4d"))
            grad.setColorAt(0.35, QtGui.QColor("#ffcf40"))
            grad.setColorAt(1.0, QtGui.QColor("#34d399"))
            painter.save()
            painter.setClipRect(fill_rect)
            painter.fillRect(grad_rect, grad)
            painter.restore()

        peak_ratio = (self._peak_hold_dbfs - self.FLOOR_DBFS) / abs(self.FLOOR_DBFS)
        peak_ratio = max(0.0, min(1.0, peak_ratio))
        peak_y = rect.bottom() - int(round(rect.height() * peak_ratio))
        painter.setPen(QtGui.QPen(QtGui.QColor("#f8fafc"), 2))
        painter.drawLine(rect.left() + 1, peak_y, rect.right() - 1, peak_y)
        painter.end()


class TrackInfoPanel(QtWidgets.QWidget):
    """Header widget that displays central track information."""

    HEADER_H = 96

    def __init__(self, cfg: Optional[config] = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._record_rotation_cache: dict[int, QtGui.QPixmap] = {}
        self._record_last_cache_key: Optional[int] = None
        self._record_cache_step_deg = 2  # rotate in coarse steps to limit redraw work
        self._record_base_pix = QtGui.QPixmap()
        self._record_base_dpr = 1.0
        self._current_time = 0.0
        self._last_album_art: Optional[QtGui.QImage] = None

        self._init_ui()
        self._set_placeholder_cover()
        self._load_record_pixmap(None)

    def _init_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(QtCore.Qt.AlignVCenter)

        self.cover_label = QtWidgets.QLabel()
        self.cover_label.setFixedSize(self.HEADER_H, self.HEADER_H)
        self.cover_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cover_label.setScaledContents(False)
        self.cover_label.setStyleSheet("background:#222; border:1px solid #444; padding:2px 2px 4px 2px;")
        layout.addWidget(self.cover_label, 0, QtCore.Qt.AlignVCenter)

        center = QtWidgets.QWidget()
        center.setMinimumHeight(self.HEADER_H)
        center.setMaximumHeight(self.HEADER_H)
        grid = QtWidgets.QGridLayout(center)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        grid.setAlignment(QtCore.Qt.AlignVCenter)

        def _fix_label_h(lbl: QtWidgets.QLabel, *, ignore_width: bool = False):
            lbl.setWordWrap(False)
            h_policy = (
                QtWidgets.QSizePolicy.Ignored
                if ignore_width
                else QtWidgets.QSizePolicy.Preferred
            )
            lbl.setSizePolicy(h_policy, QtWidgets.QSizePolicy.Fixed)

        self.title_label = MarqueeLabel()
        self.artist_label = MarqueeLabel()
        self.l_key = QtWidgets.QLabel("KEY: --")
        self.l_cent = QtWidgets.QLabel("Cent: +000")
        for w in (self.l_key,):
            f = w.font()
            f.setPointSize(18)
            f.setBold(True)
            w.setFont(f)
            _fix_label_h(w)
        for w in (self.title_label,):
            f = w.font()
            f.setPointSize(18)
            f.setBold(True)
            w.setFont(f)
            _fix_label_h(w, ignore_width=True)
        for w in (self.artist_label,):
            f = w.font()
            f.setPointSize(13)
            f.setBold(False)
            w.setFont(f)
            _fix_label_h(w, ignore_width=True)

        self.l_bpm = QtWidgets.QLabel("BPM: --")
        self.l_time = QtWidgets.QLabel("00:00.00 / 00:00.00")
        for w in (self.l_bpm, self.l_time, self.l_cent):
            f = w.font()
            f.setPointSize(12)
            f.setBold(False)
            w.setFont(f)
            _fix_label_h(w)

        grid.addWidget(self.title_label, 0, 0, 1, 3, QtCore.Qt.AlignVCenter)
        grid.addWidget(self.artist_label, 1, 0, 1, 3, QtCore.Qt.AlignVCenter)
        grid.addWidget(self.l_bpm, 2, 0, 1, 1, QtCore.Qt.AlignVCenter)
        grid.addWidget(self.l_time, 2, 4, 1, 1, QtCore.Qt.AlignVCenter)
        grid.addWidget(self.l_key, 0, 4, 1, 1, QtCore.Qt.AlignVCenter)
        grid.addWidget(self.l_cent, 1, 4, 1, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(center, 1, QtCore.Qt.AlignVCenter)

        self.record_label = QtWidgets.QLabel()
        self.record_label.setFixedSize(self.HEADER_H-4, self.HEADER_H-4)
        self.record_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.record_label, 0, QtCore.Qt.AlignVCenter)

        self.peak_meter = VerticalPeakMeter()
        layout.addWidget(self.peak_meter, 0, QtCore.Qt.AlignVCenter)

    # External
    def set_title(self, text: str):
        self.title_label.set_marquee_text((text or "").strip())

    def set_artist(self, text: str):
        self.artist_label.set_marquee_text((text or "").strip())

    def set_bpm_text(self, text: str):
        self.l_bpm.setText(text)

    def set_time_text(self, text: str):
        self.l_time.setText(text)

    def set_key_text(self, text: str):
        self.l_key.setText(text)

    def set_cent_text(self, text: str):
        self.l_cent.setText(text)

    def update_album_art(self, album_art: Optional[QtGui.QImage]):
        self._last_album_art = album_art
        if album_art is not None and not album_art.isNull():
            target = self.cover_label.contentsRect().size()
            if not target.isValid():
                target = self.cover_label.size()
            pix = QtGui.QPixmap.fromImage(album_art).scaled(
                target,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self.cover_label.setPixmap(pix)
        else:
            self._set_placeholder_cover()
        self._load_record_pixmap(album_art)

    def update_record_pix(self, time_sec: float):
        self._current_time = time_sec
        base = self._record_base_pix
        if base.isNull():
            return
        dpr = self.record_label.devicePixelRatioF()
        if abs(dpr - self._record_base_dpr) > 1e-6:
            self._load_record_pixmap(self._last_album_art)
            base = self._record_base_pix
            if base.isNull():
                return
        angle = ((time_sec * (33.3333333 / 60) * 360) % 360)
        step = max(1, int(getattr(self, "_record_cache_step_deg", 2)))
        cache_key = int(round(angle / step) * step) % 360
        if cache_key == self._record_last_cache_key:
            return
        self._record_last_cache_key = cache_key
        rotated = self._record_rotation_cache.get(cache_key)
        if rotated is None:
            logical_size = self.record_label.size()
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
            self._record_rotation_cache[cache_key] = rotated
        self.record_label.setPixmap(rotated)

    def set_config(self, cfg: config):
        self.cfg = cfg
        self._load_record_pixmap(self._last_album_art)

    def set_output_peak_dbfs(self, dbfs: float):
        self.peak_meter.set_level_dbfs(dbfs)

    # Internal helper
    def _set_placeholder_cover(self):
        pix = QtGui.QPixmap(self.HEADER_H-4, self.HEADER_H-4)
        pix.fill(QtGui.QColor("#333"))
        painter = QtGui.QPainter(pix)
        painter.setPen(QtGui.QPen(QtGui.QColor("#666")))
        painter.drawRect(0, 0, self.HEADER_H - 5, self.HEADER_H - 5)
        painter.setPen(QtGui.QPen(QtGui.QColor("#aaa")))
        painter.drawText(pix.rect(), QtCore.Qt.AlignCenter, "No Art")
        painter.end()
        self.cover_label.setPixmap(pix)

    def _load_record_pixmap(
        self,
        album_art: Optional[QtGui.QImage],
        size: int | None = None,
        vinyl_path: Optional[str] = None,
    ):
        size = size or self.HEADER_H - 4
        dpr = self.record_label.devicePixelRatioF() if hasattr(self, "record_label") else 1.0
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
            cfg_path = (
                getattr(self.cfg.viewconfig, "record_img_path", None)
                if self.cfg is not None and getattr(self.cfg, "viewconfig", None) is not None
                else None
            )
            path = vinyl_path if vinyl_path is not None else cfg_path or "assets/vinyl.png"
            vinyl = QtGui.QPixmap(path) if path else QtGui.QPixmap()
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

        painter.setBrush(QtGui.QColor("#999"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ddd"), 1))
        painter.drawEllipse(QtCore.QRectF(size / 2 - 2, size / 2 - 2, 4, 4))
        painter.end()

        self._record_base_pix = base
        self._record_base_dpr = dpr
        self._record_rotation_cache.clear()
        self._record_last_cache_key = None
        self.update_record_pix(self._current_time)
