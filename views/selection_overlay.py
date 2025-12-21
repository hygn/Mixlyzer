import math
from PySide6 import QtGui
import pyqtgraph as pg
from .base import ViewPlugin, register_view


@register_view("SelectionOverlayView")
class SelectionOverlayView(ViewPlugin):
    """Displays start/end triangle markers for the current key selection."""

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._selection_start: float | None = None
        self._selection_end: float | None = None
        self._y_fallback = 0.9

        self._start_marker = self._create_marker(color=(0, 200, 255), point_down=True)
        self._end_marker = self._create_marker(color=(255, 140, 0), point_down=False)

        self.bus.sig_key_selection_changed.connect(self._on_selection_changed)
        self.bus.sig_time_changed.connect(self._on_view_changed)
        self.bus.sig_window_changed.connect(self._on_view_changed)
        self.bus.sig_center_changed.connect(self._on_view_changed)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        self.plot.addItem(self._start_marker)
        self.plot.addItem(self._end_marker)
        try:
            self.plot.sigRangeChanged.connect(self._on_range_changed)
        except Exception:
            pass

    def detach(self):
        if self.plot is None:
            return
        try:
            self.plot.sigRangeChanged.disconnect(self._on_range_changed)
        except Exception:
            pass
        for marker in (self._start_marker, self._end_marker):
            if marker.scene() is not None:
                marker.scene().removeItem(marker)
        self.plot = None

    def render_initial(self):
        self._update_markers()

    def _on_selection_changed(self, payload) -> None:
        start = end = None
        if payload is not None:
            try:
                start, end = payload
            except Exception:
                start = end = None
        self._selection_start = self._normalize_value(start)
        self._selection_end = self._normalize_value(end)
        self._update_markers()

    def _on_range_changed(self, *_args):
        self._update_markers()

    def _on_view_changed(self, *_args):
        self._update_markers()

    def _update_markers(self) -> None:
        if self.plot is None:
            self._start_marker.setVisible(False)
            self._end_marker.setVisible(False)
            return
        y = self._marker_y()
        left = self._left_edge()
        self._set_marker(self._start_marker, self._selection_start, left, y)
        self._set_marker(self._end_marker, self._selection_end, left, y)

    def _marker_y(self) -> float:
        if self.plot is None:
            return self._y_fallback
        try:
            (_, _), (ymin, ymax) = self.plot.viewRange()
        except Exception:
            ymin, ymax = 0.0, 1.0
        y_range = max(1e-3, ymax - ymin)
        y = ymax - 0.05 * y_range
        self._y_fallback = y
        return y

    def _left_edge(self) -> float:
        center = float(getattr(self.tl, "center_t", 0.0))
        current = float(getattr(self.tl, "current_time", 0.0))
        return center - current

    @staticmethod
    def _normalize_value(value):
        if value is None:
            return None
        try:
            val = float(value)
        except Exception:
            return None
        if not math.isfinite(val):
            return None
        return val

    @staticmethod
    def _set_marker(marker: pg.ScatterPlotItem, value: float | None, left: float, y: float) -> None:
        if value is None:
            marker.setVisible(False)
            return
        marker.setData([left + value], [y])
        marker.setVisible(True)

    @staticmethod
    def _create_marker(*, color: tuple[int, int, int], point_down: bool) -> pg.ScatterPlotItem:
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
        marker = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen(color=color, width=1.4),
            brush=pg.mkBrush(color[0], color[1], color[2], 220),
            pxMode=True,
            symbol=path,
        )
        marker.setZValue(120)
        marker.setVisible(False)
        return marker
