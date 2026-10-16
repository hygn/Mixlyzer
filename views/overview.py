from PySide6 import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg
from core.event_bus import EventBus
from utils.jumpcue_colors import get_cue_point_color
from utils.jump_cues import extract_jump_cue_graph
from utils.phrases import (
    abbreviate_phrase_labels,
    extract_phrase_segments,
    phrase_strip_segments,
)
from utils.cue_points import extract_cue_points
from utils.color import key_label_text_color
from utils.keystrip import key_segment_label, key_segment_rows, key_strip_segments
from utils.wave import render_wave_overview
from views.segment_strip import SegmentStrip
from core.config import load_cfg


# Pixel rows of the overview waveform image.
OVERVIEW_WAVE_HEIGHT = 64


def _screen_pixel_width() -> int:
    """Device pixels across the widest screen: the overview image is drawn at
    this many columns (one per pixel when the overview spans that screen)."""
    screens = QtGui.QGuiApplication.screens()
    return max((int(round(screen.geometry().width() * screen.devicePixelRatio())) for screen in screens), default=4096)


def _down_triangle_path() -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.moveTo(-0.5, -0.6)
    path.lineTo(0.5, -0.6)
    path.lineTo(0.0, 0.6)
    path.closeSubpath()
    return path


class SeekOnlyViewBox(pg.ViewBox):
    """Seek-only view box used for scrub interactions."""
    def __init__(self, bus, get_duration, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus              # def seek_cb(t: float) -> None
        self._get_duration = get_duration    # def get_duration() -> float
        self.setMouseEnabled(x=False, y=False)
        self.setDefaultPadding(0.0)
        if hasattr(self, 'setWheelEnabled'):
            self.setWheelEnabled(False)

    def wheelEvent(self, ev):
        ev.ignore()
    
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.bus.sig_scrub_begin.emit()
            t = self._map_scene_x_to_time(ev.scenePos().x())
            if t is not None:
                self.bus.sig_scrub_update.emit(t)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if ev.buttons() & QtCore.Qt.LeftButton:
            t = self._map_scene_x_to_time(ev.scenePos().x())
            if t is not None:
                self.bus.sig_scrub_update.emit(t)
        ev.accept()
    
    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            t = self._map_scene_x_to_time(ev.scenePos().x())
            if t is not None:
                self.bus.sig_scrub_end.emit(t)
        ev.accept()

    def _map_scene_x_to_time(self, scene_x):
        x = self.mapSceneToView(QtCore.QPointF(scene_x, 0)).x()
        if not np.isfinite(x):
            return None
        dur = float(self._get_duration() or 0.0)
        return float(np.clip(x, 0.0, max(0.0, dur)))


class OverviewWidget(QtWidgets.QWidget):
    def __init__(self, bus: EventBus, *, tl=None, model):
        super().__init__()
        self.bus = bus
        self.tl = tl
        self.model = model

        self.canvas = pg.GraphicsLayoutWidget(show=True)
        try:
            self.canvas.ci.layout.setContentsMargins(8, 8, 8, 8)
            self.canvas.ci.layout.setSpacing(0)
        except Exception:
            pass
        self.vb = SeekOnlyViewBox(
            bus= bus,
            get_duration=lambda: self.duration,
        )

        self.p = self.canvas.addPlot(row=0, col=0, viewBox=self.vb)
        self.p.setMenuEnabled(False)
        self.p.hideButtons()
        self.p.hideAxis("left")
        self.p.hideAxis("bottom")
        try:
            self.p.layout.setContentsMargins(0, 0, 0, 0)
            self.p.layout.setSpacing(0)
        except Exception:
            pass

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(4)
        layout.addWidget(self.canvas, 1)

        self._current_time = 0.0
        # Device-pixel column of the playhead last drawn; the overview is
        # redrawn only when the playhead moves to another pixel column.
        self._now_px: int | None = None

        self.img_wave = pg.ImageItem(); self.p.addItem(self.img_wave)
        if hasattr(self.img_wave, "setAutoDownsample"):
            self.img_wave.setAutoDownsample(True)
        try:
            self.img_wave.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        except Exception:
            pass
        self.img_wave.setOpts(interpolation='nearest')
        self.played_overlay = QtWidgets.QGraphicsRectItem()
        self.played_overlay.setPen(pg.mkPen(None))
        self.played_overlay.setBrush(pg.mkBrush(0, 0, 0, 140))
        self.played_overlay.setZValue(4)
        self.p.addItem(self.played_overlay)
        self.bpm_strip_items: list[tuple[pg.TextItem, float, float, str]] = []
        self.bpm_strip_bg = QtWidgets.QGraphicsRectItem()
        self.bpm_strip_bg.setPen(pg.mkPen(None))
        self.bpm_strip_bg.setBrush(pg.mkBrush(0, 0, 0, 235))
        self.bpm_strip_bg.setZValue(3)
        self.p.addItem(self.bpm_strip_bg)
        self.bpm_strip_boundaries = QtWidgets.QGraphicsPathItem(); self.p.addItem(self.bpm_strip_boundaries)
        line_pen = pg.mkPen((245, 245, 245, 230), width=1.2, cosmetic=True)
        self.bpm_strip_boundaries.setPen(line_pen)
        self.bpm_strip_boundaries.setZValue(8)
        self.jump_items: list[tuple[pg.ScatterPlotItem, pg.TextItem]] = []
        self.img_key = SegmentStrip(); self.p.addItem(self.img_key)
        # key segment boundaries and labels (labels only where the segment is wide enough)
        self.key_boundaries = QtWidgets.QGraphicsPathItem(); self.p.addItem(self.key_boundaries)
        self.key_boundaries.setPen(pg.mkPen((255, 255, 255, 220), width=1, cosmetic=True))
        self.key_boundaries.setZValue(7)
        self.key_labels: list[pg.TextItem] = []
        self._key_rows: list[tuple[int, float, float]] = []

        # Phrase overlay: translucent label band over the lower waveform, on a
        # dark backing so the band reads over the waveform.
        self.phrase_bg = QtWidgets.QGraphicsRectItem()
        self.phrase_bg.setPen(pg.mkPen(None))
        self.phrase_bg.setBrush(pg.mkBrush(0, 0, 0, 140))
        self.phrase_bg.setZValue(5)
        self.phrase_bg.setVisible(False)
        self.p.addItem(self.phrase_bg)
        self.img_phrase = SegmentStrip(); self.p.addItem(self.img_phrase)
        self.img_phrase.setOpacity(0.5)
        self.img_phrase.setZValue(6)
        self.phrase_boundaries = QtWidgets.QGraphicsPathItem(); self.p.addItem(self.phrase_boundaries)
        self.phrase_boundaries.setPen(pg.mkPen((255, 255, 255, 220), width=1, cosmetic=True))
        self.phrase_boundaries.setZValue(7)
        self.phrase_labels: list[pg.TextItem] = []
        self._phrase_visible = True
        self._phrase_band_h = 0.208
        self._phrase_font = QtGui.QFont()
        self._phrase_font.setPointSizeF(7.0)
        self._phrase_font.setBold(True)
        self._bpm_font = QtGui.QFont()
        self._bpm_font.setPointSizeF(8.0)
        self._bpm_font.setBold(True)
        self.cue_point_markers = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen((255, 40, 40), width=1.0),
            brush=pg.mkBrush(255, 40, 40, 240),
            pxMode=True,
            symbol=_down_triangle_path(),
        )
        self.cue_point_markers.setZValue(25)
        self.p.addItem(self.cue_point_markers)


        self.line_now = pg.InfiniteLine(angle=90, pen=pg.mkPen((255,200,50), width=2)); self.p.addItem(self.line_now)
        self.line_now.setZValue(30)

        self.duration = 0.0
        self.SEGMENT_H = 0.10
        self.BPM_H = 0.55
        self.KEY_H = 0.25
        self.WAVE_H = 1.15
        self.CUE_H = 0.1
        self._wave_levels = (0, 255)
        self._wave_top = self.KEY_H + self.WAVE_H
        self._tempo_segments = None
        self._segment_bottom = -(self.SEGMENT_H + self.BPM_H)
        self._bpm_bottom = -self.BPM_H
        self._key_bottom = 0.0
        self._cue_label_y = 0.95
        self._cue_block_bottom = 0.9
        self._cue_block_height = 0.1
        self._cue_marker_size = 10.0
        self._cue_font = QtGui.QFont()
        self._cue_font.setPointSizeF(8.0)
        self._cue_font.setBold(True)
        self._jump_cues: list[dict] = []
        self._jump_links: list[dict] = []
        self._wave_bottom = self.KEY_H

        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_duration_changed.connect(self._on_duration)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_updated)
        self.bus.sig_jumpcue_updated.connect(self._on_jumpcue_updated)
        self.bus.sig_phrase_segments_updated.connect(self._on_phrases_updated)
        self.bus.sig_cue_points_updated.connect(self._render_cue_points)

    def set_data(self) -> None:
        """Initialize overview visuals from the shared model."""
        features = self.model.features
        self.duration = float(self.model.duration_sec or 0.0)
        self._current_time = 0.0
        self._now_px = None
        strip_height = self.SEGMENT_H + self.BPM_H
        total_height = max(1.0, self.KEY_H + self.WAVE_H)
        self.p.setYRange(-strip_height, total_height, padding=0.0)
        self.p.setXRange(0.0, max(0.1, self.duration), padding=0.0)

        self._segment_bottom = -strip_height
        self._bpm_bottom = -self.BPM_H
        self._key_bottom = 0.0
        self.bpm_strip_bg.setRect(QtCore.QRectF(0.0, self._segment_bottom, self.duration, strip_height))
        wave_bottom = self.KEY_H
        wave_top = wave_bottom + self.WAVE_H
        self._wave_bottom = wave_bottom
        self._wave_top = wave_top
        self._cue_block_height = 0.1
        self._cue_block_bottom = wave_top - self._cue_block_height
        self._cue_label_y = self._cue_block_bottom

        self.refresh_waveform()

        self.update_key_strip()
        self._jump_cues, self._jump_links = extract_jump_cue_graph(features)
        self._render_jump_cues(self._jump_cues)
        self._tempo_segments = features.get("tempo_segments")
        self._render_tempo_strips(self._tempo_segments)
        self._render_phrases()
        self._render_cue_points()
        self._update_played_overlay()

    def refresh_waveform(self) -> None:
        """Draw the overview waveform from the stored band levels in the configured style."""
        levels = (self.model.features or {}).get("wave_overview_np")
        if levels is None or self.duration <= 0:
            self.img_wave.clear()
            return
        image = render_wave_overview(np.asarray(levels), np.asarray(self.model.features["wave_overview_ref"]),
                                     load_cfg().viewconfig.waveform_style, OVERVIEW_WAVE_HEIGHT,
                                     columns=_screen_pixel_width())
        self.img_wave.setImage(image, autoLevels=False, levels=self._wave_levels)
        self.img_wave.setRect(QtCore.QRectF(0.0, self._wave_bottom, self.duration, self._wave_top - self._wave_bottom))

    def _render_cue_points(self) -> None:
        points = extract_cue_points(self.model.features)
        if not points:
            self.cue_point_markers.setData([], [])
            return
        times = [float(point["time_sec"]) for point in points]
        colors = [get_cue_point_color(int(point.get("color_idx", 0) or 0)) for point in points]
        y = self._wave_top - 0.05
        self.cue_point_markers.setData(
            times, [y] * len(times),
            brush=[pg.mkBrush(*c, 240) for c in colors],
            pen=[pg.mkPen(c, width=1.0) for c in colors],
        )
    
    def _on_jumpcue_updated(self) -> None:
        feats = self.model.features
        self._jump_cues, self._jump_links = extract_jump_cue_graph(feats)
        self._render_jump_cues(self._jump_cues)

    def update_key_strip(self) -> None:
        """Draw the key strip from the current key segments (rendered just in time)."""
        key_segments = (self.model.features or {}).get("key_segments") if self.duration > 0 else None
        self.img_key.set_segments(key_strip_segments(key_segments), self._key_bottom, self.KEY_H)
        self._key_rows = key_segment_rows(key_segments)
        edges = [t for _idx, start, end in self._key_rows for t in (start, end)]
        self.key_boundaries.setPath(
            self._phrase_boundary_path(edges, self._key_bottom, self._key_bottom + self.KEY_H)
        )
        while len(self.key_labels) < len(self._key_rows):
            lbl = pg.TextItem("", anchor=(0.5, 0.5))
            lbl.setFont(self._phrase_font)
            lbl.setZValue(17)
            self.p.addItem(lbl)
            self.key_labels.append(lbl)
        label_y = self._key_bottom + self.KEY_H * 0.5
        for i, lbl in enumerate(self.key_labels):
            if i < len(self._key_rows):
                idx, start, end = self._key_rows[i]
                lbl.setText(key_segment_label(idx))
                lbl.setColor(key_label_text_color(idx))
                lbl.setPos((start + end) * 0.5, label_y)
            else:
                lbl.setText("")
        self._refresh_key_label_visibility()

    def _refresh_key_label_visibility(self) -> None:
        """Overview only: a key label shows when its segment is wider than the label."""
        duration = max(float(self.duration), 1e-6)
        view_w = max(float(self.canvas.width()), 1.0)
        for i, lbl in enumerate(self.key_labels):
            if i >= len(self._key_rows):
                lbl.setVisible(False)
                continue
            _idx, start, end = self._key_rows[i]
            seg_px = (end - start) / duration * view_w
            lbl.setVisible(seg_px >= lbl.boundingRect().width() + 8.0)

    # events
    def _on_time(self, t: float):
        self._current_time = float(np.clip(t, 0.0, max(self.duration, 0.0)))
        # The whole track spans the widget, so the playhead moves about one pixel
        # per duration / width seconds; skip frames where it stays on its pixel.
        now_px = self._time_to_device_px(self._current_time)
        if now_px is not None and now_px == self._now_px:
            return
        self._now_px = now_px
        self.line_now.setPos(self._current_time)
        self._update_played_overlay()

    def _time_to_device_px(self, t: float) -> int | None:
        vb = self.p.getViewBox()
        if vb is None or self.duration <= 0.0:
            return None
        point = vb.mapViewToDevice(QtCore.QPointF(t, 0.0))
        if point is None:
            return None
        return int(round(point.x() * float(self.devicePixelRatioF())))

    def _on_duration(self, d: float):
        self.duration = float(d)
        self._now_px = None
        self.p.setXRange(0.0, max(0.1, self.duration), padding=0.0)
        self.bpm_strip_bg.setRect(
            QtCore.QRectF(0.0, self._segment_bottom, self.duration, self.SEGMENT_H + self.BPM_H)
        )
        self._update_played_overlay()

    def _render_jump_cues(self, cues: list[dict]):
        # Cull to cues intersecting current view
        try:
            view_min, view_max = self.p.viewRange()[0]
        except Exception:
            view_min, view_max = -float("inf"), float("inf")

        # the jump point only (the start/end range is not shown)
        data: list[tuple[float, tuple[int, int, int], str]] = []
        for cue in cues:
            label = str(cue.get("label", "")).strip()
            if not label:
                continue
            color = cue["color"]
            j_pnt = float(cue.get("point", cue.get("start", 0.0)))
            if j_pnt < view_min - 0.1 or j_pnt > view_max + 0.1:
                continue
            data.append((j_pnt, color, label))

        if not data:
            for marker, text in self.jump_items:
                marker.setVisible(False)
                text.setVisible(False)
            return

        while len(self.jump_items) < len(data):
            marker = pg.ScatterPlotItem(symbol="s", size=0.0)
            text = pg.TextItem("", color="k", anchor=(0.5, 0.5))
            text.setFont(self._cue_font)
            marker.setZValue(19)
            text.setZValue(20)
            self.p.addItem(marker)
            self.p.addItem(text)
            self.jump_items.append((marker, text))

        for idx, (x_point, color, label) in enumerate(data):
            marker, text = self.jump_items[idx]
            marker.setData([x_point], [self._cue_label_y])
            marker.setBrush(pg.mkBrush(color[0], color[1], color[2], 255))
            marker.setPen(pg.mkPen(color=(color[0], color[1], color[2], 100), width=0.6))
            marker.setSize(self._cue_marker_size)

            text.setText(label)
            text.setPos(x_point, self._cue_label_y)
            text.setFont(self._cue_font)

            marker.setVisible(True)
            text.setVisible(True)

        for idx in range(len(data), len(self.jump_items)):
            marker, text = self.jump_items[idx]
            marker.setVisible(False)
            text.setVisible(False)

    def _on_beatgrid_updated(self) -> None:
        segments = self.model.features.get("tempo_segments")
        self._tempo_segments = segments
        self._render_tempo_strips(self._tempo_segments)

    def _render_tempo_strips(self, tempo_segments) -> None:
        self._clear_tempo_strip_items()
        arr = self._normalize_tempo_segments(tempo_segments)
        self.bpm_strip_bg.setRect(
            QtCore.QRectF(0.0, self._segment_bottom, self.duration, self.SEGMENT_H + self.BPM_H)
        )
        if arr.size == 0:
            return

        total = max(self.duration, 0.0)
        if total <= 0.0:
            return

        clipped = []
        for seg in arr:
            start = float(seg[0])
            end = float(seg[1])
            bpm = float(seg[2])
            if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(bpm)):
                continue
            start = float(np.clip(start, 0.0, total))
            end = float(np.clip(end, 0.0, total))
            if end - start <= 1e-3:
                continue
            clipped.append((start, end, bpm))
        if not clipped:
            return

        self._render_combined_tempo_strip(
            clipped,
            self._merge_bpm_display_segments(clipped),
        )

    def _clear_tempo_strip_items(self) -> None:
        for label, _start, _end, _text in self.bpm_strip_items:
            try:
                self.p.removeItem(label)
            except Exception:
                pass
        self.bpm_strip_items.clear()
        self.bpm_strip_boundaries.setPath(QtGui.QPainterPath())

    @staticmethod
    def _normalize_tempo_segments(tempo_segments) -> np.ndarray:
        if tempo_segments is None:
            return np.empty((0, 3), dtype=float)
        try:
            arr = np.asarray(tempo_segments, dtype=float)
        except Exception:
            return np.empty((0, 3), dtype=float)
        if arr.ndim == 1:
            cols = 5 if arr.size % 5 == 0 else 4 if arr.size % 4 == 0 else 3 if arr.size % 3 == 0 else 0
            if cols == 0:
                return np.empty((0, 3), dtype=float)
            arr = arr.reshape((-1, cols))
        if arr.ndim != 2 or arr.shape[1] < 3:
            return np.empty((0, 3), dtype=float)
        return arr[:, :3].astype(float, copy=False)

    @staticmethod
    def _format_bpm_label(bpm: float) -> str:
        rounded = round(float(bpm))
        if abs(float(bpm) - rounded) <= 0.05:
            return str(int(rounded))
        return f"{float(bpm):.1f}"

    def _merge_bpm_display_segments(self, segments: list[tuple[float, float, float]]) -> list[tuple[float, float, str]]:
        merged: list[tuple[float, float, str]] = []
        for start, end, bpm in segments:
            label = self._format_bpm_label(bpm)
            if merged and merged[-1][2] == label and abs(start - merged[-1][1]) <= 1e-3:
                prev_start, _prev_end, prev_label = merged[-1]
                merged[-1] = (prev_start, end, prev_label)
            else:
                merged.append((start, end, label))
        return merged

    def _render_combined_tempo_strip(
        self,
        segments: list[tuple[float, float, float]],
        bpm_segments: list[tuple[float, float, str]],
    ) -> None:
        strip_h = self.SEGMENT_H + self.BPM_H
        line_y = self._segment_bottom + strip_h * 0.34
        tick_top = self._segment_bottom + strip_h * 0.98
        tick_bottom = self._segment_bottom + strip_h * 0.02
        label_y = self._segment_bottom + strip_h * 0.73

        self.bpm_strip_boundaries.setPath(
            self._combined_tempo_line_path(segments, line_y, tick_top, tick_bottom)
        )

        label_pad = max(float(self.duration), 1.0) * 0.003
        for start, end, label_text in bpm_segments:
            label_x = min(end, start + label_pad)
            label = pg.TextItem(label_text, color=(245, 245, 245), anchor=(0.0, 0.5))
            label.setFont(self._bpm_font)
            label.setZValue(10)
            label.setPos(label_x, label_y)
            label.setVisible(self._strip_label_fits(start, end, label_text))
            self.p.addItem(label)
            self.bpm_strip_items.append((label, start, end, label_text))

    def _combined_tempo_line_path(
        self,
        segments: list[tuple[float, float, float]],
        y: float,
        tick_top: float,
        tick_bottom: float,
    ) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        if not segments:
            return path

        for start, end, _bpm in segments:
            if end <= start:
                continue
            path.moveTo(float(start), float(y))
            path.lineTo(float(end), float(y))

        first_start = float(segments[0][0])
        last_end = float(segments[-1][1])
        self._add_tempo_tick(path, first_start, tick_top, tick_bottom)
        for idx in range(1, len(segments)):
            prev_start, prev_end, prev_bpm = segments[idx - 1]
            start, end, bpm = segments[idx]
            if abs(float(start) - float(prev_end)) > 1e-3:
                self._add_tempo_tick(path, float(prev_end), tick_top, tick_bottom)
                self._add_tempo_tick(path, float(start), tick_top, tick_bottom)
                continue
            if self._format_bpm_label(prev_bpm) == self._format_bpm_label(bpm):
                self._add_tempo_tick(path, float(start), y, tick_bottom)
            else:
                self._add_tempo_tick(path, float(start), tick_top, tick_bottom)
        self._add_tempo_tick(path, last_end, tick_top, tick_bottom)
        return path

    @staticmethod
    def _add_tempo_tick(path: QtGui.QPainterPath, x: float, y0: float, y1: float) -> None:
        path.moveTo(float(x), float(y0))
        path.lineTo(float(x), float(y1))

    def _strip_label_fits(self, start: float, end: float, label: str) -> bool:
        duration = max(float(self.duration), 1e-6)
        view_w = max(float(self.canvas.width()), 1.0)
        px_w = (float(end) - float(start)) / duration * view_w
        return px_w >= max(28.0, 8.0 * len(label))

    def set_phrase_visible(self, visible: bool) -> None:
        self._phrase_visible = bool(visible)
        self._render_phrases()

    def _on_phrases_updated(self) -> None:
        self._render_phrases()
        self._render_cue_points()

    def _render_phrases(self) -> None:
        feats = self.model.features or {}
        phrases = extract_phrase_segments(feats, self.duration) if self._phrase_visible else []
        numbered = abbreviate_phrase_labels(phrases)
        band_bottom = self._wave_bottom
        band_h = self._phrase_band_h
        segments = phrase_strip_segments(phrases) if (self._phrase_visible and self.duration > 0) else []
        self.img_phrase.set_segments(segments, band_bottom, band_h)
        if not segments:
            self.phrase_bg.setVisible(False)
        else:
            self.phrase_bg.setRect(QtCore.QRectF(0.0, band_bottom, self.duration, band_h))
            self.phrase_bg.setVisible(True)

        # Thin boundary lines at every segment edge, within the band only.
        edges: list[float] = []
        for seg in phrases:
            edges.append(float(seg.get("start", 0.0)))
            edges.append(float(seg.get("end", 0.0)))
        self.phrase_boundaries.setPath(self._phrase_boundary_path(edges, band_bottom, band_bottom + band_h))

        while len(self.phrase_labels) < len(phrases):
            lbl = pg.TextItem("", color=(255, 255, 255), anchor=(0.5, 0.5))
            lbl.setFont(self._phrase_font)
            lbl.setZValue(17)
            self.p.addItem(lbl)
            self.phrase_labels.append(lbl)
        label_y = band_bottom + band_h * 0.5
        for idx, lbl in enumerate(self.phrase_labels):
            if idx < len(phrases):
                seg = phrases[idx]
                center = (float(seg.get("start", 0.0)) + float(seg.get("end", 0.0))) * 0.5
                lbl.setText(numbered[idx] if idx < len(numbered) else "")
                lbl.setPos(center, label_y)
                lbl.setVisible(True)
            else:
                lbl.setVisible(False)

    @staticmethod
    def _phrase_boundary_path(xs, y0: float, y1: float) -> QtGui.QPainterPath:
        xs = np.asarray(sorted(set(float(x) for x in xs)), dtype=float)
        if xs.size == 0:
            return QtGui.QPainterPath()
        x = np.repeat(xs, 3)
        y = np.empty(xs.size * 3, dtype=float)
        y[0::3] = y0
        y[1::3] = y1
        y[2::3] = np.nan
        x[2::3] = np.nan
        return pg.arrayToQPath(x, y, connect="finite")

    def _update_played_overlay(self) -> None:
        if self.played_overlay is None:
            return
        width = float(np.clip(self._current_time, 0.0, max(self.duration, 0.0)))
        bottom = self._segment_bottom
        top = max(bottom, self._wave_top)
        height = top - bottom
        self.played_overlay.setRect(QtCore.QRectF(0.0, bottom, width, height))
        self.played_overlay.setVisible(width > 0.0 and height > 0.0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._now_px = None
        QtCore.QTimer.singleShot(0, self._refresh_bpm_label_visibility)
        QtCore.QTimer.singleShot(0, self._refresh_key_label_visibility)

    def _refresh_bpm_label_visibility(self) -> None:
        for label, start, end, label_text in self.bpm_strip_items:
            if not label_text:
                label.setVisible(False)
                continue
            label.setVisible(self._strip_label_fits(start, end, label_text))

    # seeking by click/drag
    def _map_scene_pos_to_time(self, ev) -> float | None:
        vb = self.p.getViewBox()
        if vb is None:
            return None
        pos = ev.scenePos() if hasattr(ev, "scenePos") else QtCore.QPointF(*ev)
        x = vb.mapSceneToView(pos).x()
        if not np.isfinite(x):
            return None
        return float(np.clip(x, 0.0, max(0.0, self.duration)))
