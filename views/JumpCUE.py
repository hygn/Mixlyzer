from __future__ import annotations

from typing import List

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .base import ViewPlugin, emphasis_opacity, emphasis_size, register_view
from utils.jump_cues import extract_jump_cue_graph


class JumpCueMarker(QtWidgets.QGraphicsItem):
    """A JumpCUE's squircle in its colour with its letter centred (black), a thin
    dark outline and a soft shadow; drawn in screen pixels, centred on its pos."""

    SIDE = 16.0         # px
    RADIUS = 0.3        # of SIDE, as the editor's squircles
    PAD = 4.0           # px each side of a letter wider than the square (AA ...)

    def __init__(self, label: str, rgb, font: QtGui.QFont):
        super().__init__()
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self._label = label
        self._fill = QtGui.QColor(*rgb)
        self._font = font
        width = max(self.SIDE, QtGui.QFontMetricsF(font).horizontalAdvance(label) + 2 * self.PAD)
        self._rect = QtCore.QRectF(-width / 2.0, -self.SIDE / 2.0, width, self.SIDE)

    def boundingRect(self) -> QtCore.QRectF:
        return self._rect.adjusted(-1.0, -1.0, 2.0, 2.5)  # outline and shadow

    def paint(self, painter: QtGui.QPainter, _option, _widget=None) -> None:
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        radius = self.SIDE * self.RADIUS
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 120))
        painter.drawRoundedRect(self._rect.translated(0.8, 1.4), radius, radius)
        painter.setBrush(self._fill)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 170), 1.0))
        painter.drawRoundedRect(self._rect, radius, radius)
        painter.setFont(self._font)
        painter.setPen(QtGui.QColor(0, 0, 0))
        painter.drawText(self._rect, QtCore.Qt.AlignCenter, self._label)


@register_view("JumpCUEView")
class JumpCUEView(ViewPlugin):
    """Display Jump CUEs as squircles with their letter at the jump point (the
    start/end range is not shown)."""

    TOP_PADDING = 0.0
    BLOCK_HEIGHT = 0.06
    EDIT_PAGE = "cue"

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._duration = 0.0
        self._cues: List[dict] = []
        # One entry per labelled cue, built when the cues change:
        # [point, marker, shown]. Playback only moves the markers by the view offset.
        self._entries: List[list] = []

        self._font = QtGui.QFont()
        self._font.setPointSizeF(9.0)
        self._font.setBold(True)

        self.bus.sig_time_changed.connect(self._refresh)
        self.bus.sig_window_changed.connect(self._refresh)
        self.bus.sig_center_changed.connect(self._refresh)
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_jumpcue_updated.connect(self._on_jumpcue_updated)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.setYRange(-0.05, 1.1, padding=0.0)
        vb = plot.getViewBox()
        if hasattr(vb, "sigRangeChanged"):
            vb.sigRangeChanged.connect(self._refresh)

    def detach(self):
        if self.plot is None:
            return
        vb = self.plot.getViewBox()
        try:
            if hasattr(vb, "sigRangeChanged"):
                vb.sigRangeChanged.disconnect(self._refresh)
        except Exception:
            pass
        self._clear_items()
        self.plot = None

    def render_initial(self):
        f = self.model.features or {}
        self._cues, _links = extract_jump_cue_graph(f)
        self._duration = float(self.model.duration_sec or 0.0)
        self._build_items()
        self._refresh()

    def _on_features_loaded(self):
        self.render_initial()

    def _on_jumpcue_updated(self, payload: dict | None = None) -> None:
        f = self.model.features or {}
        self._cues, _links = extract_jump_cue_graph(f)
        self._duration = float(self.model.duration_sec or 0.0)
        self._build_items()
        self._refresh()

    def _clear_items(self) -> None:
        for _point, marker, _shown in self._entries:
            if marker.scene() is not None and self.plot is not None:
                self.plot.removeItem(marker)
        self._entries = []

    def _build_items(self) -> None:
        """Create the marker of every cue."""
        self._clear_items()
        if self.plot is None:
            return
        for cue in self._cues:
            label = str(cue.get("label", "")).strip()
            if not label:
                continue
            point = float(cue.get("point", cue.get("start", 0.0)))
            marker = JumpCueMarker(label, cue["color"], self._font)
            # over the beatgrid lines (10, 11) and lane boundaries (15), which used to
            # cross the square and darken it; under the CUE point markers (110)
            marker.setZValue(105)
            marker.setVisible(False)
            self.plot.addItem(marker)
            self._entries.append([point, marker, False])
        self._apply_emphasis()

    def _apply_emphasis(self) -> None:
        for _point, marker, _shown in self._entries:
            marker.setScale(emphasis_size(self.emphasis, 1.0))
            marker.setOpacity(emphasis_opacity(self.emphasis))

    def _refresh(self, *args):
        """Place the cues near the view at the current offset; hide the rest."""
        if self.plot is None or not self._entries:
            return
        try:
            view_min, view_max = self.plot.viewRange()[0]
        except Exception:
            return

        left = self.tl.left_edge()
        label_y = 1.0 - self.TOP_PADDING - self.BLOCK_HEIGHT
        for entry in self._entries:
            point, marker, shown = entry
            show = view_min - 0.1 <= left + point <= view_max + 0.1
            if show:
                marker.setPos(left + point, label_y)
            if show != shown:
                marker.setVisible(show)
                entry[2] = show
