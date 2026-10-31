import math

from PySide6 import QtGui
import pyqtgraph as pg

from .base import PHRASE_LANE, ViewPlugin, register_view

CUE_ORANGE = (255, 140, 0)


def _up_triangle_path() -> QtGui.QPainterPath:
    """A triangle pointing up with its base on the origin (the symbol's y grows downwards)."""
    path = QtGui.QPainterPath()
    path.moveTo(-0.5, 0.0)
    path.lineTo(0.5, 0.0)
    path.lineTo(0.0, -0.9)
    path.closeSubpath()
    return path


@register_view("DeckCueView")
class DeckCueView(ViewPlugin):
    """The deck's cue point (the CUE button's, as the player's deck reports it: sig_deck_state), as a CDJ shows it:
    an orange triangle pointing up, its base on the phrase strip's top edge."""

    SIZE = 13

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._cue = math.nan
        self._marker = pg.ScatterPlotItem(
            size=self.SIZE,
            pen=pg.mkPen((60, 30, 0), width=1.0),
            brush=pg.mkBrush(*CUE_ORANGE, 255),
            pxMode=True,
            symbol=_up_triangle_path(),
        )
        self._marker.setZValue(108)   # over the waveform and the phrase band, under the CUE point markers
        bus.sig_deck_state.connect(self._on_deck_state)
        bus.sig_time_changed.connect(self._reposition)
        bus.sig_window_changed.connect(self._reposition)
        bus.sig_center_changed.connect(self._reposition)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.addItem(self._marker)

    def detach(self):
        if self.plot is not None and self._marker.scene() is not None:
            self.plot.removeItem(self._marker)
        self.plot = None

    def render_initial(self):
        self._reposition()

    def _on_deck_state(self, report) -> None:
        cue = float(report.cue)
        if cue != self._cue and not (math.isnan(cue) and math.isnan(self._cue)):
            self._cue = cue
            # at its track time; each frame only moves it (_reposition), off the view it is culled as any item
            self._marker.setData(*(([cue], [PHRASE_LANE[1]]) if math.isfinite(cue) else ([], [])))
            self._reposition()

    def _reposition(self, *_args) -> None:
        if self.plot is not None:
            self._marker.setPos(self.tl.left_edge(), 0.0)
