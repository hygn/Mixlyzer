from PySide6 import QtGui
import pyqtgraph as pg

from .base import ViewPlugin, emphasis_opacity, emphasis_size, register_view
from core.marks.colors import get_cue_point_color


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

    EDIT_PAGE = "cue"
    SIZE = 14

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._times: list[float] = []
        self._brushes: list = []
        self._pens: list = []
        self._markers = pg.ScatterPlotItem(
            size=self.SIZE,
            pen=pg.mkPen((255, 40, 40), width=1.2),
            brush=pg.mkBrush(255, 40, 40, 240),
            pxMode=True,
            symbol=_down_triangle_path(),
        )
        self._markers.setZValue(110)
        self.model.sig_loaded.connect(self._on_features_loaded)
        self.model.sig_cue_points_changed.connect(self._on_features_loaded)
        self.bus.sig_time_changed.connect(self._reposition)
        self.bus.sig_window_changed.connect(self._reposition)
        self.bus.sig_center_changed.connect(self._reposition)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.addItem(self._markers)

    def _apply_emphasis(self) -> None:
        self._markers.setSize(emphasis_size(self.emphasis, self.SIZE))
        self._markers.setOpacity(emphasis_opacity(self.emphasis))

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
        points = self.model.cue_points
        self._times = [float(point.time_sec) for point in points]
        colors = [get_cue_point_color(int(point.color_idx or 0)) for point in points]
        self._brushes = [pg.mkBrush(*c, 240) for c in colors]
        self._pens = [pg.mkPen(c, width=1.2) for c in colors]

    def _reposition(self, *_args) -> None:
        if self.plot is None or not self._times:
            self._markers.setData([], [])
            return
        left = self.tl.left_edge()
        # Visible window in track-time, with a small pad to avoid edge popping.
        window = max(0.1, float(self.tl.window_sec))
        visible_start = -self.tl.left_edge()
        visible_end = visible_start + window
        pad = window * 0.05
        visible_start -= pad
        visible_end += pad
        y = 0.985
        shown = [i for i, time_sec in enumerate(self._times) if visible_start <= time_sec <= visible_end]
        xs = [left + self._times[i] for i in shown]
        self._markers.setData(
            xs, [y] * len(xs),
            brush=[self._brushes[i] for i in shown],
            pen=[self._pens[i] for i in shown],
        )
