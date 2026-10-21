from typing import Optional

import numpy as np
import pyqtgraph as pg

from core.tempo_rows import bpm_at, segment_end_bpm
from .base import ViewPlugin, emphasis_opacity, register_view
from .phrase import _sticky_label_left
from .playhead import _outlined_text_item

# a gradual tempo change, in the score's words (like BPM, not translated)
ACCEL, RIT = "accel.", "rit."
TEXT_COLOR = (235, 235, 235)
Z_VALUE = 12   # one above the beatgrid's lines (10, 11), under the CUE markers (105, 110)

TempoRow = tuple[float, float, float, float]   # start, end, start BPM, end BPM (steady: the same)


def bpm_text(bpm: float) -> str:
    """A tempo as shown: whole when within 0.05 BPM, else one decimal."""
    rounded = round(float(bpm))
    return str(int(rounded)) if abs(float(bpm) - rounded) <= 0.05 else f"{float(bpm):.1f}"


def tempo_rows(tempo_segments) -> list[TempoRow]:
    """A row per tempo shown: a ramp, or a steady tempo (neighbours showing the same one joined: a row split for
    its bar length, a phase reset at the same tempo, is no change of tempo; a ramp too small to show: steady)."""
    if tempo_segments is None:
        return []
    try:
        arr = np.asarray(tempo_segments, dtype=float)
    except (TypeError, ValueError):
        return []
    if arr.ndim != 2 or arr.shape[1] < 3:
        return []
    rows: list[TempoRow] = []
    for row in arr:
        start, end, bpm = float(row[0]), float(row[1]), float(row[2])
        if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(bpm)) or end <= start or bpm <= 0:
            continue
        end_bpm = segment_end_bpm(row)
        if bpm_text(end_bpm) == bpm_text(bpm):
            end_bpm = bpm
        prev = rows[-1] if rows else None
        if (prev is not None and prev[2] == prev[3] and end_bpm == bpm and bpm_text(prev[2]) == bpm_text(bpm)
                and abs(start - prev[1]) <= 1e-3):
            rows[-1] = (prev[0], end, prev[2], prev[3])
        else:
            rows.append((start, end, bpm, end_bpm))
    return rows


def tempo_text(row: TempoRow, now: Optional[float] = None) -> str:
    """A tempo's label: "174 BPM"; a ramp's "rit. 180→160 BPM" / "accel. ...", and with ``now`` (the playhead in
    it) the tempo there: "rit. 172.4 BPM"."""
    start, end, bpm, end_bpm = row
    if end_bpm == bpm:
        return f"{bpm_text(bpm)} BPM"
    direction = ACCEL if end_bpm > bpm else RIT
    if now is not None:
        return f"{direction} {bpm_at((start, end, bpm, start, 4, end_bpm), now):.1f} BPM"
    return f"{direction} {bpm_text(bpm)}→{bpm_text(end_bpm)} BPM"


@register_view("TempoLabelView")
class TempoLabelView(ViewPlugin):
    """The beatgrid's tempos at the top of the main plot, a label each as the key and phrase labels: it comes in
    with its tempo, is held right of the playhead while the tempo plays (the bar.beat label is left of it) and
    is pushed along by the tempo's end; a ramp's shows its start and end tempo, and the tempo at the playhead
    while it plays. Outlined as the bar.beat label; drawn just above the beatgrid's lines and under the CUE
    markers, so it never hides a cue. Part of the beatgrid's display (its emphasis the beatgrid page's)."""
    EDIT_PAGE = "beatgrid"

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot = None
        self._labels: list[pg.TextItem] = []
        self._rows: list[TempoRow] = []

        self.bus.sig_time_changed.connect(self._reposition)
        self.bus.sig_window_changed.connect(self._reposition)
        self.bus.sig_center_changed.connect(self._reposition)
        self.bus.sig_features_loaded.connect(self._rebuild)
        self.bus.sig_beatgrid_edited.connect(self._rebuild)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        self._rebuild()

    def detach(self):
        if self.plot is None:
            return
        for label in self._labels:
            if label.scene() is not None:
                self.plot.removeItem(label)
        self._labels.clear()
        self.plot = None

    def render_initial(self):
        self._rebuild()

    def _apply_emphasis(self) -> None:
        opacity = emphasis_opacity(self.emphasis)
        for label in self._labels:
            label.setOpacity(opacity)

    def _rebuild(self, *_args) -> None:
        if self.plot is None:
            return
        features = self.model.features if isinstance(self.model.features, dict) else {}
        self._rows = tempo_rows(features.get("tempo_segments"))
        while len(self._labels) < len(self._rows):
            label = _outlined_text_item(TEXT_COLOR, anchor=(0.0, 0.0))
            label.setZValue(Z_VALUE)
            self.plot.addItem(label)
            self._labels.append(label)
        for label in self._labels:
            label.setText("")
            label.setVisible(False)
        self._apply_emphasis()
        self._reposition()

    def _reposition(self, *_args) -> None:
        """Each tempo's label at its start, held right of the playhead while it plays, pushed by its end."""
        if self.plot is None:
            return
        try:
            (xmin, xmax), (ymin, ymax) = self.plot.viewRange()
        except Exception:
            xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
        x_pad = 0.01 * float(xmax - xmin)
        label_y = float(ymax) - 0.03 * float(ymax - ymin)
        try:
            x_per_pixel = abs(float(self.plot.getViewBox().viewPixelSize()[0]))
        except Exception:
            x_per_pixel = 0.0
        left = self.tl.left_edge()
        now = float(getattr(self.tl, "current_time", 0.0))
        hold = float(self.tl.center_t) + x_pad   # right of the playhead, as the bar.beat label is left of it
        visible_start = float(xmin) - left
        visible_end = float(xmax) - left
        for i, label in enumerate(self._labels):
            if i >= len(self._rows):
                label.setVisible(False)
                continue
            row = self._rows[i]
            start, end = row[0], row[1]
            if end < visible_start or start > visible_end:
                label.setVisible(False)
                continue
            text = tempo_text(row, now if start <= now < end else None)
            if label.textItem.toPlainText() != text:
                label.setText(text)
            width = float(label.boundingRect().width()) * x_per_pixel
            label.setPos(_sticky_label_left(left + start, left + end, hold, width), label_y)
            label.setVisible(True)
