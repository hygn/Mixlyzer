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
        self._lines = {}         # {int index -> pg.InfiniteLine}
        self._downbeat_markers = {}  # {int index -> (top_item, bottom_item)}
        self.pen = pg.mkPen((200, 200, 200), width=1)
        self.downbeat_pen = pg.mkPen((255, 60, 60), width=2)

        self.bus.sig_time_changed.connect(self._on_time)
        self.bus.sig_window_changed.connect(self._on_window)
        self.bus.sig_center_changed.connect(self._on_center)
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_updated)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.setYRange(0.0, 1.0, padding=0.02)
        vb = plot.getViewBox()
        if hasattr(vb, "sigRangeChanged"):
            vb.sigRangeChanged.connect(self._refresh_lines)

    def detach(self):
        if self.plot is None:
            return
        for line in self._lines.values():
            if line.scene() is not None:
                line.scene().removeItem(line)
        self._lines.clear()
        self._clear_downbeat_markers()
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
        if self.plot is None or self.beats_time is None or len(self.beats_time) == 0:
            self._clear_downbeat_markers()
            return

        try:
            (vxmin, vxmax), _ = self.plot.viewRange()
        except Exception:
            return

        left = float(self.tl.center_t - self.tl.current_time)

        t = self.beats_time
        pos = left + t

        pad = max(0.0, (vxmax - vxmin) * 0.02)
        mask = (pos >= (vxmin - pad)) & (pos <= (vxmax + pad))
        visible_idx = np.nonzero(mask)[0]

        keep = set(int(i) for i in visible_idx)
        for idx in list(self._lines.keys()):
            if idx not in keep:
                line = self._lines.pop(idx, None)
                if line is not None and line.scene() is not None:
                    line.scene().removeItem(line)

        for idx in visible_idx:
            idx = int(idx)
            p = float(pos[idx])
            line = self._lines.get(idx)
            if line is None:
                line = pg.InfiniteLine(pos=p, angle=90, pen=self.pen, movable=False)
                self._lines[idx] = line
                self.plot.addItem(line)
            else:
                line.setPos(p)

        self._refresh_downbeat_markers(left, vxmin, vxmax, pad)

    def _refresh_downbeat_markers(self, left: float, vxmin: float, vxmax: float, pad: float) -> None:
        if self.plot is None:
            return
        if self.downbeats_time is None or len(self.downbeats_time) == 0:
            self._clear_downbeat_markers()
            return

        y_min, y_max = self.plot.viewRange()[1]
        span = max(1e-6, float(y_max - y_min))
        cap = span * 0.08
        top_seg = (y_max - cap, y_max)
        bottom_seg = (y_min, y_min + cap)

        pos = left + self.downbeats_time
        mask = (pos >= (vxmin - pad)) & (pos <= (vxmax + pad))
        visible_idx = np.nonzero(mask)[0]
        keep = set(int(i) for i in visible_idx)
        for idx in list(self._downbeat_markers.keys()):
            if idx not in keep:
                top_item, bot_item = self._downbeat_markers.pop(idx, (None, None))
                for item in (top_item, bot_item):
                    if item is not None:
                        try:
                            self.plot.removeItem(item)
                        except Exception:
                            pass

        for idx in visible_idx:
            idx = int(idx)
            p = float(pos[idx])
            marker = self._downbeat_markers.get(idx)
            if marker is None:
                top_item = pg.PlotDataItem([p, p], list(top_seg), pen=self.downbeat_pen)
                bot_item = pg.PlotDataItem([p, p], list(bottom_seg), pen=self.downbeat_pen)
                self._downbeat_markers[idx] = (top_item, bot_item)
                self.plot.addItem(top_item)
                self.plot.addItem(bot_item)
            else:
                top_item, bot_item = marker
                if top_item is not None:
                    top_item.setData([p, p], list(top_seg))
                if bot_item is not None:
                    bot_item.setData([p, p], list(bottom_seg))
    
    def _on_beatgrid_updated(self, bg_seg=None):
        f = self.model.features or {}
        self.beats_time = f.get("beats_time_sec")
        self.downbeats_time = self._build_downbeats_from_segments(f.get("tempo_segments"))
        self.duration = float(self.model.duration_sec or 0.0)
        self._refresh_lines()

    def _clear_downbeat_markers(self) -> None:
        if not self._downbeat_markers:
            return
        if self.plot is None:
            self._downbeat_markers.clear()
            return
        for top_item, bot_item in self._downbeat_markers.values():
            for item in (top_item, bot_item):
                if item is not None:
                    try:
                        self.plot.removeItem(item)
                    except Exception:
                        pass
        self._downbeat_markers.clear()

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
