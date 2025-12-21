from PySide6 import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg
from core.event_bus import EventBus
from utils.jump_cues import extract_jump_cue_pairs


class SeekOnlyViewBox(pg.ViewBox):
    """Seek-only view box used for scrub interactions."""
    def __init__(self, bus, get_duration, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus              # def seek_cb(t: float) -> None
        self._get_duration = get_duration    # def get_duration() -> float
        self.setMouseEnabled(x=False, y=False)
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
        self.vb = SeekOnlyViewBox(
            bus= bus,
            get_duration=lambda: self.duration,
        )

        self.p = self.canvas.addPlot(row=0, col=0, viewBox=self.vb)
        self.p.setMenuEnabled(False)
        self.p.hideButtons()
        self.p.hideAxis("left")
        self.p.hideAxis("bottom")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
        self.segment_regions: list[pg.LinearRegionItem] = []
        self.segment_colors = [
            (70, 130, 180, 40),
            (46, 139, 87, 40),
            (255, 140, 0, 40),
            (138, 43, 226, 40),
        ]
        self.jump_items: list[tuple[pg.BarGraphItem, pg.ScatterPlotItem, pg.TextItem]] = []
        self.jump_colors = [
            (34, 139, 34),
            (0, 153, 204),
            (220, 120, 20),
            (148, 0, 211),
            (210, 105, 30),
            (0, 128, 128),
        ]
        self.img_key  = pg.ImageItem(); self.p.addItem(self.img_key)
        if hasattr(self.img_key, "setAutoDownsample"):
            self.img_key.setAutoDownsample(True)
        
        self.line_now = pg.InfiniteLine(angle=90, pen=pg.mkPen((255,200,50), width=2)); self.p.addItem(self.line_now)

        self.duration = 0.0
        self.KEY_H = 0.25
        self.CUE_H = 0.1
        self._wave_preview_source = None
        self._wave_preview_cache = None
        self._wave_levels = (0, 255)
        self._wave_top = 1.0
        self._tempo_segments = None
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

        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_duration_changed.connect(self._on_duration)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_updated)
        self.bus.sig_key_selection_changed.connect(self._on_selection_changed)
        self.bus.sig_jump_arm.connect(self._on_jump_arm)
        self.bus.sig_jump_disarm.connect(self._on_jump_disarm)
        self.bus.sig_jumpcue_updated.connect(self._on_jumpcue_updated)

    def set_data(self) -> None:
        """Initialize overview visuals from the shared model."""
        features = self.model.features
        self.duration = float(self.model.duration_sec or 0.0)
        self._current_time = 0.0
        total_height = max(1.0, self.KEY_H + 1.0)
        self.p.setYRange(0.0, total_height + 0.02, padding=0.0)
        self.p.setXRange(0.0, max(0.1, self.duration), padding=0.0)

        wave_bottom = self.KEY_H
        wave_top = wave_bottom + 1.0
        self._wave_top = wave_top
        self._selection_marker_y = wave_top + 0.012
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
            self.img_key.setRect(QtCore.QRectF(0.0, 0.0, self.duration, self.KEY_H))
        jump_pairs = extract_jump_cue_pairs(features)
        self._jump_pairs = jump_pairs
        self._render_jump_cues(self._jump_pairs)
        self._tempo_segments = features.get("tempo_segments")
        self._render_segments(self._tempo_segments)
        self._apply_selection_graphics()
        self._clear_jump_arrow()
    
    def _on_jumpcue_updated(self) -> None:
        feats = self.model.features
        jc_block = feats.get("jump_cues_np")
        extracted = feats.get("jump_cues_extracted")
        if extracted is None and jc_block is not None:
            extracted = extract_jump_cue_pairs({"jump_cues_np": jc_block})
        self._jump_pairs = extracted or []
        self._render_jump_cues(self._jump_pairs)

    def update_key_image(self, key_img) -> None:
        if key_img is None:
            return
        arr = np.asarray(key_img)
        if arr.size == 0:
            return
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self.img_key.setImage(arr, autoLevels=False, levels=self._wave_levels)
        self.img_key.setRect(QtCore.QRectF(0.0, 0.0, self.duration, self.KEY_H))

    # events
    def _on_time(self, t: float):
        self._current_time = float(np.clip(t, 0.0, max(self.duration, 0.0)))
        self.line_now.setPos(self._current_time)

    def _on_duration(self, d: float):
        self.duration = float(d)
        self.p.setXRange(0.0, max(0.1, self.duration), padding=0.0)

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
        cue = payload.get("cue")
        label = str(payload.get("label", "")).strip()
        if not cue and label:
            for pair in self._jump_pairs:
                if (pair.get("forward", {}).get("label") == label) or (pair.get("backward", {}).get("label") == label):
                    cue = pair
                    break
        if not cue:
            return (None, None)

        def _pick_point(block):
            if not isinstance(block, dict):
                return None
            try:
                val = block.get("point", block.get("start"))
                return float(val) if val is not None else None
            except Exception:
                return None

        forward = cue.get("forward") or {}
        backward = cue.get("backward") or {}
        start = dest = None
        if forward.get("label") == label:
            start = _pick_point(forward)
            dest = _pick_point(backward)
        elif backward.get("label") == label:
            start = _pick_point(backward)
            dest = _pick_point(forward)
        else:
            start = _pick_point(forward)
            dest = _pick_point(backward)

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

        data: list[tuple[float, float, tuple[int, int, int], str]] = []
        for idx, pair in enumerate(cues):
            color = self.jump_colors[idx % len(self.jump_colors)]
            for role in ("forward", "backward"):
                point = pair.get(role) or {}
                label = str(point.get("label", "")).strip()
                if not label:
                    continue
                start = float(point.get("start", 0.0))
                end = float(point.get("end", start))
                j_pnt = float(point.get("point", start))
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
        self._render_segments(self._tempo_segments)
        self._apply_selection_graphics()

    def _render_segments(self, tempo_segments) -> None:
        for region in self.segment_regions:
            try:
                self.p.removeItem(region)
            except Exception:
                pass
        self.segment_regions.clear()

        if tempo_segments is None:
            return
        try:
            arr = np.asarray(tempo_segments, dtype=float)
        except Exception:
            return
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                return
            arr = arr.reshape((-1, 3))
        if arr.ndim < 2 or arr.shape[1] < 2:
            return

        total = max(self.duration, 0.0)
        if total <= 0.0:
            return

        for idx, seg in enumerate(arr):
            start = float(seg[0])
            end = float(seg[1])
            if not (np.isfinite(start) and np.isfinite(end)):
                continue
            if end <= start:
                continue
            start = float(np.clip(start, 0.0, total))
            end = float(np.clip(end, 0.0, total))
            if end - start <= 1e-3:
                continue
            region = pg.LinearRegionItem(values=(start, end))
            region.setMovable(False)
            color = self.segment_colors[idx % len(self.segment_colors)]
            region.setBrush(pg.mkBrush(*color))
            region.setZValue(4)
            self.p.addItem(region)
            self.segment_regions.append(region)

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
