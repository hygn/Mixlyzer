from __future__ import annotations

from typing import List, Tuple

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from pyqtgraph.graphicsItems.BarGraphItem import BarGraphItem
from pyqtgraph.graphicsItems.ScatterPlotItem import ScatterPlotItem
from pyqtgraph.graphicsItems.TextItem import TextItem

from .base import ViewPlugin, register_view
from utils.jump_cues import extract_jump_cue_pairs


@register_view("JumpCUEView")
class JumpCUEView(ViewPlugin):
    """Display Jump CUE segments as translucent blocks with square markers."""

    TOP_PADDING = 0.0
    BLOCK_HEIGHT = 0.06
    MARKER_SIZE = 10

    COLORS = [
        (34, 139, 34),
        (0, 153, 204),
        (220, 120, 20),
        (148, 0, 211),
        (210, 105, 30),
        (0, 128, 128),
    ]

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._duration = 0.0
        self._pairs: List[dict] = []
        self._items: List[Tuple[BarGraphItem, ScatterPlotItem, TextItem]] = []

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
        for bar, marker, text in self._items:
            if bar.scene() is not None:
                self.plot.removeItem(bar)
            if marker.scene() is not None:
                self.plot.removeItem(marker)
            if text.scene() is not None:
                self.plot.removeItem(text)
        self._items.clear()
        self.plot = None

    def render_initial(self):
        f = self.model.features or {}
        self._pairs = extract_jump_cue_pairs(f)
        self._duration = float(self.model.duration_sec or 0.0)
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
        self._duration = float(self.model.duration_sec or 0.0)
        self._refresh()

    def _refresh(self, *args):
        if self.plot is None:
            return

        try:
            (vxmin, vxmax), _ = self.plot.viewRange()
        except Exception:
            return

        left = float(getattr(self.tl, "center_t", 0.0) - getattr(self.tl, "current_time", 0.0))
        pad = max(0.0, (vxmax - vxmin) * 0.05)

        for bar, marker, text in self._items:
            bar.setVisible(False)
            marker.setVisible(False)
            text.setVisible(False)

        if not self._pairs:
            return

        try:
            view_min, view_max = self.plot.viewRange()[0]
        except Exception:
            view_min, view_max = -float("inf"), float("inf")

        visible_entries: list[tuple[float, float, float, Tuple[int, int, int], str]] = []
        for idx, pair in enumerate(self._pairs):
            color = self.COLORS[idx % len(self.COLORS)]
            for role in ("forward", "backward"):
                seg = pair.get(role) or {}
                label = str(seg.get("label", "")).strip()
                if not label:
                    continue
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                point = float(seg.get("point", start))
                if end <= start:
                    end = start + max(0.01, (self._duration or 1.0) * 0.002)

                x_start = left + start
                x_end = left + end
                x_point = left + point
                if x_end < view_min - 0.1 or x_start > view_max + 0.1:
                    continue
                visible_entries.append((x_start, x_end, x_point, color, label))

        if not visible_entries:
            return

        block_bottom = 1.0 - self.TOP_PADDING - self.BLOCK_HEIGHT
        label_y = block_bottom

        while len(self._items) < len(visible_entries):
            bar = BarGraphItem(
                x=[0.0],
                height=[0.0],
                width=0.01,
                y=block_bottom,
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen(None),
            )
            marker = ScatterPlotItem(symbol="s", size=0.0)
            text = TextItem("", color="k", anchor=(0.5, 0.5))
            text.setFont(self._font)
            bar.setZValue(8)
            marker.setZValue(9)
            text.setZValue(10)
            self.plot.addItem(bar)
            self.plot.addItem(marker)
            self.plot.addItem(text)
            self._items.append((bar, marker, text))

        for idx, (x_start, x_end, x_point, color, label) in enumerate(visible_entries):
            bar, marker, text = self._items[idx]
            width = max(0.01, x_end - x_start)
            center = x_start + width * 0.5

            bar.setOpts(
                x=[center],
                y=block_bottom,
                height=[self.BLOCK_HEIGHT],
                width=width,
                brush=pg.mkBrush(color[0], color[1], color[2], 30),
                pen=pg.mkPen(color=(color[0], color[1], color[2], 30), width=0.4),
            )
            marker.setData([x_point], [label_y])
            marker.setSize(self.MARKER_SIZE)
            marker.setBrush(pg.mkBrush(color[0], color[1], color[2], 255))
            marker.setPen(pg.mkPen(color=(color[0], color[1], color[2], 100), width=0.6))

            text.setText(label)
            text.setPos(x_point, label_y)

            bar.setVisible(True)
            marker.setVisible(True)
            text.setVisible(True)

        for idx in range(len(visible_entries), len(self._items)):
            bar, marker, text = self._items[idx]
            bar.setVisible(False)
            marker.setVisible(False)
            text.setVisible(False)
