from bisect import bisect_right
from PySide6 import QtCore
import numpy as np
import pyqtgraph as pg
from .base import ViewPlugin, register_view

@register_view("PlayHead")
class PlayHead(ViewPlugin):
    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen((255,200,50), width=2))
        self.beat_label = pg.TextItem("", color=(50, 120, 255), anchor=(1.0, 0.0))
        self._beats_time = ()
        self._timesignature = 4
        self._last_count = None
        self._beat_start_offset = 0
        self.plot = None

        self._set_beats(self.model.features)

        self.bus.sig_window_changed.connect(self.update_window)
        self.bus.sig_time_changed.connect(self.update_time)
        self.bus.sig_center_changed.connect(self._on_center_changed)
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_beatgrid_edited.connect(self._on_beatgrid_edited)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        self.plot.addItem(self.playhead)
        self.plot.addItem(self.beat_label)
        self.playhead.setPos(self.tl.center_t)
        self._update_beat_label()

    def detach(self):
        if self.plot is None:
            return
        self.plot.removeItem(self.playhead)
        if self.beat_label.scene() is not None:
            self.beat_label.scene().removeItem(self.beat_label)
        self.plot = None

    def update_time(self, _t: float):
        self._update_beat_label()

    def update_window(self, _w: float):
        self._update_playhead()

    def _on_center_changed(self, _c: float):
        self._update_playhead()

    def _on_features_loaded(self):
        self._set_beats(self.model.features)
        self._update_beat_label()

    def _on_beatgrid_edited(self, _payload=None):
        self._set_beats(self.model.features)
        self._update_beat_label()

    def _update_playhead(self):
        self.playhead.setPos(self.tl.center_t)
        self._update_label_position()

    def _update_beat_label(self):
        if self.plot is None:
            return
        raw_count = self._compute_beat_count() - self._beat_start_offset
        if raw_count <= 0:
            count = -1
        else:
            count = raw_count - 1
        if self._last_count != count:
            self._last_count = count
            bar = int(count/self._timesignature+1)
            bar = bar if bar >= 1 else bar -1
            beats = abs(count%self._timesignature) + 1
            bar_beats = str(f"{bar}.{beats}")
            self.beat_label.setText(bar_beats)
        self._update_label_position()

    def _compute_beat_count(self) -> int:
        if not self._beats_time:
            return 0
        current = float(getattr(self.tl, "current_time", 0.0))
        return bisect_right(self._beats_time, current)

    def _update_label_position(self):
        if self.plot is None:
            return
        try:
            (xmin, xmax), (ymin, ymax) = self.plot.viewRange()
        except Exception:
            xmin, xmax = 0.0, 1.0
            ymin, ymax = 0.0, 1.0

        x_range = float(xmax - xmin)
        y_range = float(ymax - ymin)
        x_pad = 0.01 * x_range if x_range else 0.0
        y_pad = 0.03 * y_range if y_range else 0.0

        x = float(self.tl.center_t) - x_pad
        y = float(ymax) - y_pad
        self.beat_label.setPos(x, y)

    def _set_beats(self, features):
        beats = features.get("beats_time_sec") if features else None
        ts = features.get("timesignature") if features else None
        if beats is None:
            self._beats_time = ()
            self._beat_start_offset = 0
            return
        try:
            values = tuple(float(b) for b in beats)
        except TypeError:
            self._beats_time = ()
            self._beat_start_offset = 0
            return
        self._beats_time = values if values else ()
        self._timesignature = ts if ts else 4

        self._beat_start_offset = 0
        downbeat = self._extract_first_downbeat(features.get("tempo_segments") if features else None)
        if downbeat is not None and self._beats_time:
            beats_arr = np.asarray(self._beats_time, dtype=float)
            if beats_arr.size:
                idx = int(np.argmin(np.abs(beats_arr - float(downbeat))))
                self._beat_start_offset = max(0, idx)

    @staticmethod
    def _extract_first_downbeat(tempo_segments):
        if tempo_segments is None:
            return None
        try:
            arr = np.asarray(tempo_segments, dtype=float)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            if arr.size % 3 != 0:
                return None
            arr = arr.reshape((-1, 3))
        if arr.ndim >= 2 and arr.shape[1] >= 1:
            starts = arr[:, 3]
            finite = starts[np.isfinite(starts)]
            if finite.size:
                return float(np.min(finite))
        return None
