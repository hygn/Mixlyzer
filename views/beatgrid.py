from PySide6 import QtCore
import pyqtgraph as pg
import numpy as np
from .base import ViewPlugin, register_view
from core.event_bus import EventBus

@register_view("BeatgridView")
class BeatgridView(ViewPlugin):
    def __init__(self, bus: EventBus, model, tl):
        super().__init__(bus, model, tl)
        self.plot = None
        self.beats_time = None   # np.ndarray of seconds [0..duration]
        self.downbeats_time = None
        self.duration = 0.0
        self.pen = pg.mkPen((200, 200, 200), width=1)
        self.downbeat_pen = pg.mkPen((255, 60, 60), width=2)
        self._beat_lines_item = None
        self._downbeat_lines_item = None

        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_window_changed.connect(self._on_window)
        self.bus.sig_center_changed.connect(self._on_center)
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_updated)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.setYRange(0.0, 1.0, padding=0.02)
        self._beat_lines_item = pg.PlotDataItem(pen=self.pen, connect="finite")
        self._downbeat_lines_item = pg.PlotDataItem(pen=self.downbeat_pen, connect="finite")
        plot.addItem(self._beat_lines_item)
        plot.addItem(self._downbeat_lines_item)
        self._beat_lines_item.setZValue(10)
        self._downbeat_lines_item.setZValue(11)
        vb = plot.getViewBox()
        if hasattr(vb, "sigRangeChanged"):
            vb.sigRangeChanged.connect(self._refresh_lines)

    def detach(self):
        if self.plot is None:
            return
        for item in (self._beat_lines_item, self._downbeat_lines_item):
            if item is not None and item.scene() is not None:
                self.plot.removeItem(item)
        self._beat_lines_item = None
        self._downbeat_lines_item = None
        vb = self.plot.getViewBox()
        try:
            vb.sigRangeChanged.disconnect(self._refresh_lines)
        except Exception:
            pass
        self.plot = None

    def render_initial(self):
        f = self.model.features or {}
        self.beats_time = f.get("beats_time_sec")  # seconds from start
        self.downbeats_time = self._build_downbeats_from_segments(f.get("tempo_segments"))
        self.duration = float(self.model.duration_sec or 0.0)
        self._refresh_lines()

    # bus callbacks
    def _on_time(self, _t: float):
        self._refresh_lines()

    def _on_window(self, _w: float):
        self._refresh_lines()

    def _on_center(self, _c: float):
        self._refresh_lines()

    def _on_features_loaded(self):
        self.render_initial()

    # core
    def _refresh_lines(self, *args):
        if self.plot is None or self._beat_lines_item is None or self._downbeat_lines_item is None:
            return
        if self.beats_time is None or len(self.beats_time) == 0:
            self._beat_lines_item.setData([], [])
            self._downbeat_lines_item.setData([], [])
            return

        try:
            (vxmin, vxmax), (y_min, y_max) = self.plot.viewRange()
        except Exception:
            return

        left = float(self.tl.center_t - self.tl.current_time)

        t = self.beats_time
        pos = left + t

        pad = max(0.0, (vxmax - vxmin) * 0.02)
        mask = (pos >= (vxmin - pad)) & (pos <= (vxmax + pad))
        visible_pos = pos[mask]
        self._beat_lines_item.setData(*self._build_vertical_segments(visible_pos, y_min, y_max))

        if self.downbeats_time is None or len(self.downbeats_time) == 0:
            self._downbeat_lines_item.setData([], [])
            return

        span = max(1e-6, float(y_max - y_min))
        cap = span * 0.08
        top_seg = (y_max - cap, y_max)
        bottom_seg = (y_min, y_min + cap)

        downbeat_pos = left + self.downbeats_time
        downbeat_mask = (downbeat_pos >= (vxmin - pad)) & (downbeat_pos <= (vxmax + pad))
        visible_downbeats = downbeat_pos[downbeat_mask]
        self._downbeat_lines_item.setData(*self._build_downbeat_segments(visible_downbeats, top_seg, bottom_seg))
    
    def _on_beatgrid_updated(self, bg_seg=None):
        f = self.model.features or {}
        self.beats_time = f.get("beats_time_sec")
        self.downbeats_time = self._build_downbeats_from_segments(f.get("tempo_segments"))
        self.duration = float(self.model.duration_sec or 0.0)
        self._refresh_lines()

    @staticmethod
    def _build_vertical_segments(xs, y0: float, y1: float):
        xs = np.asarray(xs, dtype=float)
        if xs.size == 0:
            return ([], [])
        x = np.repeat(xs, 3)
        y = np.empty(xs.size * 3, dtype=float)
        y[0::3] = y0
        y[1::3] = y1
        y[2::3] = np.nan
        x[2::3] = np.nan
        return x, y

    @classmethod
    def _build_downbeat_segments(cls, xs, top_seg, bottom_seg):
        xs = np.asarray(xs, dtype=float)
        if xs.size == 0:
            return ([], [])
        top_x, top_y = cls._build_vertical_segments(xs, float(top_seg[0]), float(top_seg[1]))
        bot_x, bot_y = cls._build_vertical_segments(xs, float(bottom_seg[0]), float(bottom_seg[1]))
        x = np.concatenate([top_x, bot_x])
        y = np.concatenate([top_y, bot_y])
        return x, y

    @staticmethod
    def _build_downbeats_from_segments(tempo_segments):
        if tempo_segments is None:
            return None
        arr = np.asarray(tempo_segments, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                return None
            arr = arr.reshape((-1, 3))
        if arr.shape[1] < 3:
            return None

        downbeats = []
        for seg in arr:
            start, end, bpm, inizio = seg[:4]
            if not np.isfinite(inizio) or not np.isfinite(end) or not np.isfinite(bpm):
                continue
            if bpm <= 0:
                continue
            t = max(0.0, float(inizio))
            stop = max(float(end), t)
            bar = 4.0 * 60.0 / float(bpm)
            if bar <= 0:
                continue
            while t <= stop + 1e-6:
                downbeats.append(t)
                t += bar
        if not downbeats:
            return None
        return np.asarray(sorted(set(downbeats)), dtype=float)
