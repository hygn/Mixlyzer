from PySide6 import QtGui
import pyqtgraph as pg

from .base import ViewPlugin, register_view
from utils.cue_points import extract_cue_points


def _down_triangle_path() -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.moveTo(-0.5, -0.6)
    path.lineTo(0.5, -0.6)
    path.lineTo(0.0, 0.6)
    path.closeSubpath()
    return path


@register_view("CUEPointView")
class CUEPointView(ViewPlugin):
    """Display standalone CUEPoints above the main waveform."""

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._times: list[float] = []
        self._markers = pg.ScatterPlotItem(
            size=14,
            pen=pg.mkPen((255, 40, 40), width=1.2),
            brush=pg.mkBrush(255, 40, 40, 240),
            pxMode=True,
            symbol=_down_triangle_path(),
        )
        self._markers.setZValue(110)
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_time_changed.connect(self._reposition)
        self.bus.sig_window_changed.connect(self._reposition)
        self.bus.sig_center_changed.connect(self._reposition)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.addItem(self._markers)

    def detach(self):
        if self.plot is not None and self._markers.scene() is not None:
            self.plot.removeItem(self._markers)
        self.plot = None

    def render_initial(self):
        self._load_points()
        self._reposition()

    def _on_features_loaded(self):
        self._load_points()
        self._reposition()

    def _load_points(self) -> None:
        self._times = [
            float(point["time_sec"])
            for point in extract_cue_points(self.model.features)
        ]

    def _reposition(self, *_args) -> None:
        if self.plot is None or not self._times:
            self._markers.setData([], [])
            return
        left = float(self.tl.center_t) - float(self.tl.current_time)
        # Visible window in track-time, with a small pad to avoid edge popping.
        window = max(0.1, float(self.tl.window_sec))
        visible_start = float(self.tl.current_time) - float(self.tl.center_t)
        visible_end = visible_start + window
        pad = window * 0.05
        visible_start -= pad
        visible_end += pad
        y = 0.985
        xs = [
            left + time_sec
            for time_sec in self._times
            if visible_start <= time_sec <= visible_end
        ]
        self._markers.setData(xs, [y] * len(xs))
