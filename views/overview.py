from PySide6 import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg
from core.event_bus import EventBus
from utils.jumpcue_colors import get_jumpcue_pair_color
from utils.jump_cues import extract_jump_cue_pairs, extract_jump_cue_graph
from utils.phrases import (
    abbreviate_phrase_labels,
    build_phrase_strip_buffer,
    extract_phrase_segments,
    merge_fill_phrases_for_display,
)
from utils.cue_points import extract_cue_points


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
        self.jump_items: list[tuple[pg.BarGraphItem, pg.ScatterPlotItem, pg.TextItem]] = []
        self.img_key  = pg.ImageItem(); self.p.addItem(self.img_key)
        if hasattr(self.img_key, "setAutoDownsample"):
            self.img_key.setAutoDownsample(True)

        # Phrase overlay: translucent label band over the lower waveform.
        self.img_phrase = pg.ImageItem(); self.p.addItem(self.img_phrase)
        if hasattr(self.img_phrase, "setAutoDownsample"):
            self.img_phrase.setAutoDownsample(True)
        self.img_phrase.setOpts(interpolation="nearest")
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
        self._wave_preview_source = None
        self._wave_preview_cache = None
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
        self._selection_region: pg.LinearRegionItem | None = None
        self._selection_start_marker: pg.ScatterPlotItem | None = None
        self._selection_end_marker: pg.ScatterPlotItem | None = None
        self._selection_marker_y = 1.05
        self._selection_start: float | None = None
        self._selection_end: float | None = None
        self._jump_arrow_line: pg.PlotDataItem | None = None
        self._jump_arrow_head: pg.ArrowItem | None = None
        self._armed_jump: tuple[float, float] | None = None
        self._jump_pairs: list[dict] = []
        self._jump_cues: list[dict] = []
        self._jump_links: list[dict] = []
        self._wave_bottom = self.KEY_H

        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_duration_changed.connect(self._on_duration)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_updated)
        self.bus.sig_key_selection_changed.connect(self._on_selection_changed)
        self.bus.sig_jump_arm.connect(self._on_jump_arm)
        self.bus.sig_jump_disarm.connect(self._on_jump_disarm)
        self.bus.sig_jumpcue_updated.connect(self._on_jumpcue_updated)
        self.bus.sig_phrase_segments_updated.connect(self._on_phrases_updated)

    def set_data(self) -> None:
        """Initialize overview visuals from the shared model."""
        features = self.model.features
        self.duration = float(self.model.duration_sec or 0.0)
        self._current_time = 0.0
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
        self._selection_marker_y = wave_top - 0.012
        self._cue_block_height = 0.1
        self._cue_block_bottom = wave_top - self._cue_block_height
        self._cue_label_y = self._cue_block_bottom

        wave_data = features.get("wave_img_np_preview")

        if wave_data is not None:
            wave_arr = wave_data if isinstance(wave_data, np.ndarray) else np.asarray(wave_data)
            if wave_arr.size:
                if self._wave_preview_source is not wave_data:
                    slice_start = int(wave_arr.shape[1] / 2) if wave_arr.ndim == 3 else 0
                    trimmed = wave_arr[:, slice_start:, :] if wave_arr.ndim == 3 else wave_arr
                    self._wave_preview_cache = np.ascontiguousarray(trimmed)
                    self._wave_preview_source = wave_data
                if self._wave_preview_cache is not None:
                    self.img_wave.setImage(self._wave_preview_cache, autoLevels=False, levels=self._wave_levels)
                    self.img_wave.setRect(QtCore.QRectF(0.0, wave_bottom, self.duration, wave_top - wave_bottom))
        else:
            self._wave_preview_source = None
            self._wave_preview_cache = None

        key_np = features.get("overview_key_np") or features.get("key_np")
        if key_np is not None:
            key_arr = key_np if isinstance(key_np, np.ndarray) else np.asarray(key_np)
            if key_arr.size:
                if key_arr.dtype != np.uint8:
                    key_arr = np.clip(key_arr, 0, 255).astype(np.uint8)
            key_img = np.ascontiguousarray(key_arr)
            self.img_key.setImage(key_img, autoLevels=False, levels=self._wave_levels)
            self.img_key.setRect(QtCore.QRectF(0.0, self._key_bottom, self.duration, self.KEY_H))
        jump_pairs = extract_jump_cue_pairs(features)
        self._jump_pairs = jump_pairs
        self._jump_cues, self._jump_links = extract_jump_cue_graph(features)
        self._render_jump_cues(self._jump_cues)
        self._tempo_segments = features.get("tempo_segments")
        self._render_tempo_strips(self._tempo_segments)
        self._render_phrases()
        self._render_cue_points()
        self._apply_selection_graphics()
        self._clear_jump_arrow()
        self._update_played_overlay()

    def _render_cue_points(self) -> None:
        points = extract_cue_points(self.model.features)
        if not points:
            self.cue_point_markers.setData([], [])
            return
        times = [float(point["time_sec"]) for point in points]
        y = self._wave_top - 0.05
        self.cue_point_markers.setData(times, [y] * len(times))
    
    def _on_jumpcue_updated(self) -> None:
        feats = self.model.features
        jc_block = feats.get("jump_cues_np")
        extracted = feats.get("jump_cues_extracted")
        if extracted is None and jc_block is not None:
            extracted = extract_jump_cue_pairs({"jump_cues_np": jc_block})
        self._jump_pairs = extracted or []
        self._jump_cues, self._jump_links = extract_jump_cue_graph(feats)
        self._render_jump_cues(self._jump_cues)

    def update_key_image(self, key_img) -> None:
        if key_img is None:
            return
        arr = np.asarray(key_img)
        if arr.size == 0:
            return
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self.img_key.setImage(arr, autoLevels=False, levels=self._wave_levels)
        self.img_key.setRect(QtCore.QRectF(0.0, self._key_bottom, self.duration, self.KEY_H))

    # events
    def _on_time(self, t: float):
        self._current_time = float(np.clip(t, 0.0, max(self.duration, 0.0)))
        self.line_now.setPos(self._current_time)
        self._update_played_overlay()

    def _on_duration(self, d: float):
        self.duration = float(d)
        self.p.setXRange(0.0, max(0.1, self.duration), padding=0.0)
        self.bpm_strip_bg.setRect(
            QtCore.QRectF(0.0, self._segment_bottom, self.duration, self.SEGMENT_H + self.BPM_H)
        )
        self._update_played_overlay()

    def _on_jump_arm(self, payload) -> None:
        start, dest = self._resolve_jump_points(payload)
        if start is None or dest is None:
            self._clear_jump_arrow()
            return
        self._armed_jump = (start, dest)
        self._render_jump_arrow()

    def _on_jump_disarm(self) -> None:
        self._clear_jump_arrow()

    def _resolve_jump_points(self, payload) -> tuple[float | None, float | None]:
        if not isinstance(payload, dict):
            return (None, None)
        source = payload.get("source")
        target = payload.get("target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            return (None, None)

        def _pick_point(block):
            if not isinstance(block, dict):
                return None
            try:
                val = block.get("point", block.get("start"))
                return float(val) if val is not None else None
            except Exception:
                return None

        start = _pick_point(source)
        dest = _pick_point(target)

        if start is None or dest is None:
            return (None, None)
        if not (np.isfinite(start) and np.isfinite(dest)):
            return (None, None)
        dur = max(self.duration, 0.0)
        start = float(np.clip(start, 0.0, dur))
        dest = float(np.clip(dest, 0.0, dur))
        return (start, dest)

    def _render_jump_arrow(self) -> None:
        self._ensure_jump_arrow_items()
        if self._jump_arrow_line is None or self._jump_arrow_head is None:
            return
        if self._armed_jump is None:
            self._clear_jump_arrow_graphics()
            return

        start, dest = self._armed_jump
        if not (np.isfinite(start) and np.isfinite(dest)):
            self._clear_jump_arrow_graphics()
            return

        try:
            y_min, y_max = self.p.viewRange()[1]
        except Exception:
            y_min, y_max = 0.0, max(self._wave_top, 1.0)

        base_y = (y_min + y_max) * 0.5
        pad = 0.02
        base_y = float(np.clip(base_y, y_min + pad, y_max - pad))

        pen = pg.mkPen((220, 20, 20), width=2)
        self._jump_arrow_line.setData([start, dest], [base_y, base_y], pen=pen)
        self._jump_arrow_line.setVisible(True)

        angle = 180 if dest >= start else 0
        self._jump_arrow_head.setStyle(
            angle=angle,
            brush=pg.mkBrush(220, 20, 20),
            pen=pen,
            headLen=14,
            tipAngle=30,
            tailLen=None,
            tailWidth=6,
        )
        self._jump_arrow_head.setPos(dest, base_y)
        self._jump_arrow_head.setVisible(True)

    def _clear_jump_arrow(self) -> None:
        self._armed_jump = None
        self._clear_jump_arrow_graphics()

    def _clear_jump_arrow_graphics(self) -> None:
        self._ensure_jump_arrow_items()
        if self._jump_arrow_line is not None:
            self._jump_arrow_line.setData([], [])
            self._jump_arrow_line.setVisible(False)
        if self._jump_arrow_head is not None:
            self._jump_arrow_head.setVisible(False)

    def _ensure_jump_arrow_items(self) -> None:
        if self._jump_arrow_line is None:
            line = pg.PlotDataItem()
            line.setZValue(28)
            line.setVisible(False)
            self.p.addItem(line)
            self._jump_arrow_line = line
        if self._jump_arrow_head is None:
            head = pg.ArrowItem(angle=0)
            head.setZValue(29)
            head.setVisible(False)
            self.p.addItem(head)
            self._jump_arrow_head = head

    def _render_jump_cues(self, cues: list[dict]):
        # Cull to cues intersecting current view
        try:
            view_min, view_max = self.p.viewRange()[0]
        except Exception:
            view_min, view_max = -float("inf"), float("inf")

        data: list[tuple[float, float, float, tuple[int, int, int], str]] = []
        for cue in cues:
            label = str(cue.get("label", "")).strip()
            if not label:
                continue
            color = cue.get("color")
            if not isinstance(color, tuple) or len(color) != 3:
                color_idx = int(cue.get("color_index", cue.get("component_index", 0)) or 0)
                color = get_jumpcue_pair_color(color_idx)
            start = float(cue.get("start", 0.0))
            end = float(cue.get("end", start))
            j_pnt = float(cue.get("point", start))
            if end <= start:
                end = start + max(0.01, self.duration * 0.002)
            if end < view_min - 0.1 or start > view_max + 0.1:
                continue
            data.append((start, end, j_pnt, color, label))

        if not data:
            for bar, marker, text in self.jump_items:
                bar.setVisible(False)
                marker.setVisible(False)
                text.setVisible(False)
            return

        while len(self.jump_items) < len(data):
            bar = pg.BarGraphItem(
                x=[0.0],
                height=[0.0],
                width=0.01,
                y=self._cue_block_bottom,
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen(None),
            )
            marker = pg.ScatterPlotItem(symbol="s", size=0.0)
            text = pg.TextItem("", color="k", anchor=(0.5, 0.5))
            text.setFont(self._cue_font)
            bar.setZValue(18)
            marker.setZValue(19)
            text.setZValue(20)
            self.p.addItem(bar)
            self.p.addItem(marker)
            self.p.addItem(text)
            self.jump_items.append((bar, marker, text))

        for idx, (x_start, x_end, x_point, color, label) in enumerate(data):
            bar, marker, text = self.jump_items[idx]
            width = max(0.01, x_end - x_start)
            center = x_start + width * 0.5
            bar.setOpts(
                x=[center],
                y=self._cue_block_bottom,
                height=[self._cue_block_height],
                width=width,
                brush=pg.mkBrush(color[0], color[1], color[2], 30),
                pen=pg.mkPen(color=(color[0], color[1], color[2], 30), width=0.4),
            )
            marker.setData([x_point], [self._cue_label_y])
            marker.setBrush(pg.mkBrush(color[0], color[1], color[2], 255))
            marker.setPen(pg.mkPen(color=(color[0], color[1], color[2], 100), width=0.6))
            marker.setSize(self._cue_marker_size)

            text.setText(label)
            text.setPos(x_point, self._cue_label_y)
            text.setFont(self._cue_font)

            bar.setVisible(True)
            marker.setVisible(True)
            text.setVisible(True)

        for idx in range(len(data), len(self.jump_items)):
            bar, marker, text = self.jump_items[idx]
            bar.setVisible(False)
            marker.setVisible(False)
            text.setVisible(False)

    def _on_beatgrid_updated(self) -> None:
        segments = self.model.features.get("tempo_segments")
        self._tempo_segments = segments
        self._render_tempo_strips(self._tempo_segments)
        self._apply_selection_graphics()

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

    def _render_phrases(self) -> None:
        feats = self.model.features or {}
        raw_phrases = extract_phrase_segments(feats) if self._phrase_visible else []
        phrases = merge_fill_phrases_for_display(raw_phrases)
        numbered = abbreviate_phrase_labels(phrases)
        band_bottom = self._wave_bottom
        band_h = self._phrase_band_h
        strip = (
            build_phrase_strip_buffer(phrases, self.duration)
            if (self._phrase_visible and self.duration > 0)
            else None
        )
        if strip is None:
            self.img_phrase.clear()
        else:
            self.img_phrase.setImage(strip, autoLevels=False, levels=(0, 255))
            self.img_phrase.setRect(QtCore.QRectF(0.0, band_bottom, self.duration, band_h))

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

    def _on_selection_changed(self, payload) -> None:
        start = end = None
        if payload is not None:
            try:
                start, end = payload
            except Exception:
                start = end = None
        self._selection_start = self._normalize_selection_value(start)
        self._selection_end = self._normalize_selection_value(end)
        self._apply_selection_graphics()

    def _apply_selection_graphics(self) -> None:
        self._ensure_selection_items()
        start = self._selection_start
        end = self._selection_end
        has_range = start is not None and end is not None
        if self._selection_region is not None:
            if has_range:
                self._selection_region.setRegion((start, end))
                self._selection_region.setVisible(True)
            else:
                self._selection_region.setVisible(False)
        y = self._selection_marker_y
        self._update_marker_item(self._selection_start_marker, start, y)
        self._update_marker_item(self._selection_end_marker, end, y)

    def _hide_selection_graphics(self) -> None:
        if self._selection_region is not None:
            self._selection_region.setVisible(False)
        if self._selection_start_marker is not None:
            self._selection_start_marker.setVisible(False)
        if self._selection_end_marker is not None:
            self._selection_end_marker.setVisible(False)
        self._selection_start = None
        self._selection_end = None

    def _ensure_selection_items(self) -> None:
        if self._selection_region is None:
            region = pg.LinearRegionItem(values=(0.0, 0.0))
            region.setMovable(False)
            region.setBrush(pg.mkBrush(255, 215, 0, 40))
            pen = pg.mkPen(color=(255, 215, 0), width=1.4)
            for line in getattr(region, "lines", []):
                try:
                    line.setPen(pen)
                except Exception:
                    pass
            region.setZValue(24)
            region.setVisible(False)
            self.p.addItem(region)
            self._selection_region = region
        if self._selection_start_marker is None:
            start_marker = pg.ScatterPlotItem(
                size=14,
                pen=pg.mkPen(color=(0, 200, 255), width=1.2),
                brush=pg.mkBrush(0, 200, 255, 200),
                pxMode=True,
                symbol=self._make_triangle_symbol(point_down=True),
            )
            start_marker.setZValue(26)
            start_marker.setVisible(False)
            self.p.addItem(start_marker)
            self._selection_start_marker = start_marker
        if self._selection_end_marker is None:
            end_marker = pg.ScatterPlotItem(
                size=14,
                pen=pg.mkPen(color=(255, 140, 0), width=1.2),
                brush=pg.mkBrush(255, 140, 0, 200),
                pxMode=True,
                symbol=self._make_triangle_symbol(point_down=False),
            )
            end_marker.setZValue(26)
            end_marker.setVisible(False)
            self.p.addItem(end_marker)
            self._selection_end_marker = end_marker

    @staticmethod
    def _make_triangle_symbol(*, point_down: bool) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        if point_down:
            path.moveTo(0.0, -0.6)
            path.lineTo(-0.5, 0.6)
            path.lineTo(0.5, 0.6)
        else:
            path.moveTo(0.0, 0.6)
            path.lineTo(-0.5, -0.6)
            path.lineTo(0.5, -0.6)
        path.closeSubpath()
        return path

    def _normalize_selection_value(self, value):
        if value is None:
            return None
        try:
            val = float(value)
        except Exception:
            return None
        if not np.isfinite(val):
            return None
        return val

    def _update_marker_item(self, marker: pg.ScatterPlotItem | None, value: float | None, y: float) -> None:
        if marker is None:
            return
        if value is None:
            marker.setVisible(False)
            return
        marker.setData([value], [y])
        marker.setVisible(True)

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
        QtCore.QTimer.singleShot(0, self._refresh_bpm_label_visibility)

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
