from PySide6 import QtCore
import pyqtgraph as pg
from .base import ViewPlugin, register_view
from core.beat_geometry import downbeat_beat_indices, bar_beat_label

@register_view("PlayHead")
class PlayHead(ViewPlugin):
    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen((255,200,50), width=2))
        self.beat_label = pg.TextItem("", color=(50, 120, 255), anchor=(1.0, 0.0))
        self._beats_time = ()
        self._downbeats = ()
        self._last_label = None
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
        label = self._compute_bar_beat_label()
        if label != self._last_label:
            self._last_label = label
            self.beat_label.setText(label)
        self._update_label_position()

    def _compute_bar_beat_label(self) -> str:
        current = float(getattr(self.tl, "current_time", 0.0))
        return bar_beat_label(
            self._beats_time,
            current,
            downbeat_indices=self._downbeats,
        )

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
        if beats is None:
            self._beats_time = ()
            self._downbeats = ()
            return
        try:
            values = tuple(float(b) for b in beats)
        except TypeError:
            self._beats_time = ()
            self._downbeats = ()
            return
        self._beats_time = values if values else ()
        # Downbeat positions as beat indices (drift-free); cached for cheap
        # per-frame bar.beat lookups.
        self._downbeats = downbeat_beat_indices(
            self._beats_time,
            features.get("tempo_segments") if features else None,
        )
