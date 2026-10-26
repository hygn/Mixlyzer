from PySide6 import QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg
from .base import PHRASE_LANE, ViewPlugin, emphasis_opacity, emphasis_width, register_view
from .segment_strip import SegmentStrip
from core.marks.phrases import (
    number_phrase_labels,
    phrase_strip_segments,
)


def _vertical_lines_path(xs, y0: float, y1: float) -> QtGui.QPainterPath:
    """Build a path of vertical line segments at x positions, between y0 and y1."""
    xs = np.asarray(sorted(set(float(x) for x in xs)), dtype=float)
    if xs.size == 0:
        return QtGui.QPainterPath()
    x = np.repeat(xs, 3)
    y = np.empty(xs.size * 3, dtype=float)
    y[0::3] = y0
    y[1::3] = y1
    y[2::3] = np.nan
    x[2::3] = np.nan
    return pg.arrayToQPath(x, y, connect="finite")


def _sticky_label_left(
    phrase_start_x: float,
    phrase_end_x: float,
    viewport_center_x: float,
    label_width: float,
) -> float:
    """Keep a label at center until its phrase boundaries push it along."""
    return min(
        max(float(viewport_center_x), float(phrase_start_x)),
        float(phrase_end_x) - max(0.0, float(label_width)),
    )


@register_view("PhraseView")
class PhraseView(ViewPlugin):
    """Phrase (song-section) overlay: a translucent label band above the key strip."""

    BAND_BOTTOM, BAND_TOP = PHRASE_LANE
    EDIT_PAGE = "phrase"
    STRIP_OPACITY = 0.45

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot = None
        self.img = SegmentStrip()
        self.img.setOpacity(self.STRIP_OPACITY)
        self.img.setZValue(14)
        self.boundaries = QtWidgets.QGraphicsPathItem()
        self.boundaries.setPen(pg.mkPen((255, 255, 255, 220), width=1, cosmetic=True))
        self.boundaries.setZValue(15)
        self._labels: list[pg.TextItem] = []
        self._phrases: list[dict] = []
        self._numbered: list[str] = []
        self.duration = 0.0
        self._font = QtGui.QFont()
        self._font.setPointSizeF(8.0)
        self._font.setBold(True)

        self.bus.sig_time_changed.connect(self._on_view_changed)
        self.bus.sig_window_changed.connect(self._on_view_changed)
        self.bus.sig_center_changed.connect(self._on_view_changed)
        self.model.sig_loaded.connect(self._on_features_loaded)
        self.model.sig_phrases_changed.connect(self._on_phrases_updated)
        self.bus.sig_phrase_labels_changed.connect(self._on_phrases_updated)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.addItem(self.img)
        plot.addItem(self.boundaries)
        self._apply_emphasis()

    def _apply_emphasis(self) -> None:
        opacity = emphasis_opacity(self.emphasis)
        self.img.setOpacity(self.STRIP_OPACITY * opacity)
        self.boundaries.setPen(pg.mkPen((255, 255, 255, 220), width=emphasis_width(self.emphasis, 1.0), cosmetic=True))
        for item in (self.boundaries, *self._labels):
            item.setOpacity(opacity)

    def detach(self):
        if self.plot is None:
            return
        if self.img.scene() is not None:
            self.plot.removeItem(self.img)
        if self.boundaries.scene() is not None:
            self.plot.removeItem(self.boundaries)
        for label in self._labels:
            if label.scene() is not None:
                self.plot.removeItem(label)
        self._labels.clear()
        self.plot = None

    def render_initial(self):
        self._load_phrases()
        self._rebuild()

    def _on_features_loaded(self):
        self._load_phrases()
        self._rebuild()

    def _on_phrases_updated(self):
        self._load_phrases()
        self._rebuild()

    def _on_view_changed(self, *_args):
        self._reposition()

    def _load_phrases(self):
        self.duration = float(self.model.duration_sec or 0.0)
        self._phrases = self.model.phrases
        self._numbered = number_phrase_labels(self._phrases)

    def _rebuild(self):
        if self.plot is None:
            return
        segments = phrase_strip_segments(self._phrases) if self.duration > 0 else []
        self.img.set_segments(segments, self.BAND_BOTTOM, self.BAND_TOP - self.BAND_BOTTOM)

        # Thin boundary lines at every segment edge (kept even between same labels).
        edges: list[float] = []
        for seg in self._phrases:
            edges.append(float(seg.start))
            edges.append(float(seg.end))
        self.boundaries.setPath(_vertical_lines_path(edges, self.BAND_BOTTOM, self.BAND_TOP))

        # Pool TextItems for the numbered labels.
        while len(self._labels) < len(self._phrases):
            label = pg.TextItem("", color=(255, 255, 255), anchor=(0.0, 0.5))
            label.setFont(self._font)
            label.setZValue(16)
            self.plot.addItem(label)
            self._labels.append(label)
        for idx, label in enumerate(self._labels):
            if idx < len(self._phrases):
                disp = self._numbered[idx] if idx < len(self._numbered) else ""
                label.setText(disp)
                label.setVisible(True)
            else:
                label.setText("")
                label.setVisible(False)
        self._apply_emphasis()
        self._reposition()

    def _reposition(self):
        if self.plot is None:
            return
        left = self.tl.left_edge()
        self.img.setPos(left, 0.0)
        self.boundaries.setPos(left, 0.0)
        label_y = (self.BAND_BOTTOM + self.BAND_TOP) * 0.5
        viewport_center = float(self.tl.center_t)
        # Visible window in track-time, with a small pad to avoid edge popping.
        window = max(0.1, float(self.tl.window_sec))
        visible_start = -self.tl.left_edge()
        visible_end = visible_start + window
        pad = window * 0.05
        visible_start -= pad
        visible_end += pad
        try:
            x_per_pixel = abs(float(self.plot.getViewBox().viewPixelSize()[0]))
        except Exception:
            x_per_pixel = 0.0
        for idx, label in enumerate(self._labels):
            if idx >= len(self._phrases):
                label.setVisible(False)
                continue
            seg = self._phrases[idx]
            seg_start = float(seg.start)
            seg_end = float(seg.end)
            # Skip phrases entirely outside the viewport — no need to position
            # (and measure) labels that aren't on screen.
            if seg_end < visible_start or seg_start > visible_end:
                label.setVisible(False)
                continue
            phrase_start = left + seg_start
            phrase_end = left + seg_end
            label_width = float(label.boundingRect().width()) * x_per_pixel
            label_left = _sticky_label_left(
                phrase_start,
                phrase_end,
                viewport_center,
                label_width,
            )
            label.setPos(label_left, label_y)
            label.setVisible(True)
