from PySide6 import QtCore, QtGui, QtWidgets
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
        self.pen = pg.mkPen((200, 200, 200), width=1, cosmetic=True)
        self.downbeat_pen = pg.mkPen((255, 60, 60), width=2, cosmetic=True)
        self._beat_lines_item: QtWidgets.QGraphicsPathItem | None = None
        self._downbeat_lines_item: QtWidgets.QGraphicsPathItem | None = None
        self._cached_track_range = None

        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_window_changed.connect(self._on_window)
        self.bus.sig_center_changed.connect(self._on_center)
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_updated)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.setYRange(0.0, 1.0, padding=0.02)
        self._beat_lines_item = QtWidgets.QGraphicsPathItem()
        self._beat_lines_item.setPen(self.pen)
        self._downbeat_lines_item = QtWidgets.QGraphicsPathItem()
        self._downbeat_lines_item.setPen(self.downbeat_pen)
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
        self._refresh_lines(force=True)

    # bus callbacks
    def _on_time(self, _t: float):
        self._update_item_offset()
        self._refresh_lines_if_needed()

    def _on_window(self, _w: float):
        self._refresh_lines(force=True)

    def _on_center(self, _c: float):
        self._update_item_offset()
        self._refresh_lines_if_needed()

    def _on_features_loaded(self):
        self.render_initial()

    # core
    def _refresh_lines(self, *args, force: bool = False):
        if self.plot is None or self._beat_lines_item is None or self._downbeat_lines_item is None:
            return
        if self.beats_time is None or len(self.beats_time) == 0:
            self._beat_lines_item.setPath(QtGui.QPainterPath())
            self._downbeat_lines_item.setPath(QtGui.QPainterPath())
            self._cached_track_range = None
            return

        try:
            (vxmin, vxmax), (y_min, y_max) = self.plot.viewRange()
        except Exception:
            return

        left = float(self.tl.center_t - self.tl.current_time)
        pad = max(0.0, (vxmax - vxmin) * 0.02)
        track_min = float(vxmin - left - pad)
        track_max = float(vxmax - left + pad)
        cached = self._cached_track_range
        if (
            not force
            and cached is not None
            and track_min >= cached[0]
            and track_max <= cached[1]
        ):
            self._update_item_offset()
            return

        cache_pad = max(0.25, (track_max - track_min) * 0.5)
        cache_min = track_min - cache_pad
        cache_max = track_max + cache_pad

        visible_beats = self._slice_visible_times(self.beats_time, cache_min, cache_max)
        self._beat_lines_item.setPath(self._build_vertical_path(visible_beats, y_min, y_max))

        if self.downbeats_time is None or len(self.downbeats_time) == 0:
            self._downbeat_lines_item.setPath(QtGui.QPainterPath())
            self._cached_track_range = (cache_min, cache_max)
            self._update_item_offset()
            return

        span = max(1e-6, float(y_max - y_min))
        cap = span * 0.08
        top_seg = (y_max - cap, y_max)
        bottom_seg = (y_min, y_min + cap)

        visible_downbeats = self._slice_visible_times(self.downbeats_time, cache_min, cache_max)
        self._downbeat_lines_item.setPath(
            self._build_downbeat_path(visible_downbeats, top_seg, bottom_seg)
        )
        self._cached_track_range = (cache_min, cache_max)
        self._update_item_offset()
    
    def _on_beatgrid_updated(self, bg_seg=None):
        f = self.model.features or {}
        self.beats_time = f.get("beats_time_sec")
        self.downbeats_time = self._build_downbeats_from_segments(f.get("tempo_segments"))
        self.duration = float(self.model.duration_sec or 0.0)
        self._cached_track_range = None
        self._refresh_lines(force=True)

    def _refresh_lines_if_needed(self) -> None:
        if self.plot is None:
            return
        try:
            (vxmin, vxmax), _ = self.plot.viewRange()
        except Exception:
            return
        left = float(self.tl.center_t - self.tl.current_time)
        pad = max(0.0, (vxmax - vxmin) * 0.02)
        track_min = float(vxmin - left - pad)
        track_max = float(vxmax - left + pad)
        cached = self._cached_track_range
        if cached is None or track_min < cached[0] or track_max > cached[1]:
            self._refresh_lines(force=True)

    def _update_item_offset(self) -> None:
        left = float(self.tl.center_t - self.tl.current_time)
        for item in (self._beat_lines_item, self._downbeat_lines_item):
            if item is not None:
                item.setPos(left, 0.0)

    @staticmethod
    def _slice_visible_times(times, t_min: float, t_max: float):
        arr = np.asarray(times, dtype=float)
        if arr.size == 0:
            return arr
        lo = int(np.searchsorted(arr, t_min, side="left"))
        hi = int(np.searchsorted(arr, t_max, side="right"))
        return arr[lo:hi]

    @staticmethod
    def _build_vertical_path(xs, y0: float, y1: float):
        xs = np.asarray(xs, dtype=float)
        if xs.size == 0:
            return QtGui.QPainterPath()
        x = np.repeat(xs, 3)
        y = np.empty(xs.size * 3, dtype=float)
        y[0::3] = y0
        y[1::3] = y1
        y[2::3] = np.nan
        x[2::3] = np.nan
        return pg.arrayToQPath(x, y, connect="finite")

    @classmethod
    def _build_downbeat_path(cls, xs, top_seg, bottom_seg):
        xs = np.asarray(xs, dtype=float)
        if xs.size == 0:
            return QtGui.QPainterPath()
        top_path = cls._build_vertical_path(xs, float(top_seg[0]), float(top_seg[1]))
        bot_path = cls._build_vertical_path(xs, float(bottom_seg[0]), float(bottom_seg[1]))
        return top_path.united(bot_path)

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
