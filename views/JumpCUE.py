from __future__ import annotations

from typing import List

import pyqtgraph as pg
from PySide6 import QtGui
from pyqtgraph.graphicsItems.BarGraphItem import BarGraphItem
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
from pyqtgraph.graphicsItems.TextItem import TextItem

from .base import ViewPlugin, register_view
from utils.jumpcue_colors import get_jumpcue_pair_color
from utils.jump_cues import extract_jump_cue_pairs, extract_jump_cue_graph


@register_view("JumpCUEView")
class JumpCUEView(ViewPlugin):
    """Display Jump CUE segments as translucent blocks with square markers."""

    TOP_PADDING = 0.0
    BLOCK_HEIGHT = 0.06
    MARKER_SIZE = 10

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._duration = 0.0
        self._pairs: List[dict] = []
        self._cues: List[dict] = []
        # One entry per labelled cue, built when the cues change:
        # [start, end, point, bar, marker, text, shown]. The items are laid out in
        # track time; playback only moves them by the view offset.
        self._entries: List[list] = []

        self._font = QtGui.QFont()
        self._font.setPointSizeF(8.0)
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
        self._pairs = extract_jump_cue_pairs(f)
        self._cues, _links = extract_jump_cue_graph(f)
        self._duration = float(self.model.duration_sec or 0.0)
        self._build_items()
        self._refresh()

    def _on_features_loaded(self):
        self.render_initial()
    
    def _on_jumpcue_updated(self, payload: dict | None = None) -> None:
        f = self.model.features or {}
        jc_block = f.get("jump_cues_np")
        extracted = f.get("jump_cues_extracted")
        if extracted is None and jc_block is not None:
            extracted = extract_jump_cue_pairs({"jump_cues_np": jc_block})
        self._pairs = extracted or []
        self._cues, _links = extract_jump_cue_graph(f)
        self._duration = float(self.model.duration_sec or 0.0)
        self._build_items()
        self._refresh()

    def _clear_items(self) -> None:
        for _start, _end, _point, bar, marker, text, _shown in self._entries:
            for item in (bar, marker, text):
                if item.scene() is not None and self.plot is not None:
                    self.plot.removeItem(item)
        self._entries = []

    def _build_items(self) -> None:
        """Create the block, marker and label of every cue at its track time."""
        self._clear_items()
        if self.plot is None:
            return
        block_bottom = 1.0 - self.TOP_PADDING - self.BLOCK_HEIGHT
        label_y = block_bottom
        for cue in self._cues:
            label = str(cue.get("label", "")).strip()
            if not label:
                continue
            color = cue.get("color")
            if not isinstance(color, tuple) or len(color) != 3:
                color_idx = int(cue.get("color_index", cue.get("component_index", 0)) or 0)
                color = get_jumpcue_pair_color(color_idx)
            start = float(cue.get("start", 0.0))
            end = float(cue.get("end", start))
            point = float(cue.get("point", start))
            if end <= start:
                end = start + max(0.01, (self._duration or 1.0) * 0.002)
            width = max(0.01, end - start)

            bar = BarGraphItem(
                x=[start + width * 0.5],
                y=block_bottom,
                height=[self.BLOCK_HEIGHT],
                width=width,
                brush=pg.mkBrush(color[0], color[1], color[2], 30),
                pen=pg.mkPen(color=(color[0], color[1], color[2], 30), width=0.4),
            )
            marker = ScatterPlotItem(
                x=[point],
                y=[label_y],
                symbol="s",
                size=self.MARKER_SIZE,
                brush=pg.mkBrush(color[0], color[1], color[2], 255),
                pen=pg.mkPen(color=(color[0], color[1], color[2], 100), width=0.6),
            )
            text = TextItem(label, color="k", anchor=(0.5, 0.5))
            text.setFont(self._font)
            bar.setZValue(8)
            marker.setZValue(9)
            text.setZValue(10)
            for item in (bar, marker, text):
                item.setVisible(False)
                self.plot.addItem(item)
            self._entries.append([start, end, point, bar, marker, text, False])

    def _refresh(self, *args):
        """Place the cues near the view at the current offset; hide the rest."""
        if self.plot is None or not self._entries:
            return
        try:
            view_min, view_max = self.plot.viewRange()[0]
        except Exception:
            return

        left = float(getattr(self.tl, "center_t", 0.0) - getattr(self.tl, "current_time", 0.0))
        label_y = 1.0 - self.TOP_PADDING - self.BLOCK_HEIGHT
        for entry in self._entries:
            start, end, point, bar, marker, text, shown = entry
            show = not (left + end < view_min - 0.1 or left + start > view_max + 0.1)
            if show:
                bar.setPos(left, 0.0)
                marker.setPos(left, 0.0)
                text.setPos(left + point, label_y)
            if show != shown:
                bar.setVisible(show)
                marker.setVisible(show)
                text.setVisible(show)
                entry[6] = show
