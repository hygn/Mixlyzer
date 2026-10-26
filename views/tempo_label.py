from typing import Optional

import numpy as np
import pyqtgraph as pg

from analyzer.music.tempo_rows import bpm_at, segment_end_bpm
from .base import ViewPlugin, emphasis_opacity, register_view
from .phrase import _sticky_label_left
from .playhead import _outlined_text_item

# a gradual tempo change, in the score's words (like BPM, not translated)
ACCEL, RIT = "accel.", "rit."
TEXT_COLOR = (235, 235, 235)
Z_VALUE = 12   # one above the beatgrid's lines (10, 11), under the CUE markers (105, 110)
# A label the next tempo's label touches (pushing it out) is dimmed to this, so the eye goes to the tempo coming,
# and stays so once passed (the next one held at the playhead); it fades over the last pixels before they touch.
TOUCHED_OPACITY = 0.35
TOUCH_FADE_PX = 12.0

TempoRow = tuple[float, float, float, float]   # start, end, start BPM, end BPM (steady: the same)
ShownRow = tuple[float, float, TempoRow]       # where its label comes in and is pushed out (beats), the tempo


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


def snap_to_beats(rows: list[TempoRow], beats) -> list[ShownRow]:
    """Where each tempo's label comes in and is pushed out: its start and end moved to their nearest beats (a
    tempo changes at a beat; a row's edge can sit between, as the tracker's boundary); a tempo left without
    length there dropped. No beats: the rows' own edges."""
    try:
        beats = np.asarray(beats if beats is not None else [], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        beats = np.empty(0)
    beats = beats[np.isfinite(beats)]
    if beats.size == 0:
        return [(row[0], row[1], row) for row in rows]

    def nearest(t: float) -> float:
        i = int(np.searchsorted(beats, t))
        candidates = beats[max(0, i - 1):i + 1]
        return float(candidates[np.argmin(np.abs(candidates - t))])

    snapped = [(nearest(row[0]), nearest(row[1]), row) for row in rows]
    return [shown for shown in snapped if shown[1] > shown[0]]


def tempo_text(row: TempoRow, now: Optional[float] = None) -> str:
    """A tempo's label: "174 BPM"; a ramp's "rit. 180→160 BPM" / "accel. ...", and with ``now`` (the playhead in
    it) the tempo there: "rit. 172.4 BPM"."""
    start, end, bpm, end_bpm = row
    if end_bpm == bpm:
        return f"{bpm_text(bpm)} BPM"
    direction = ACCEL if end_bpm > bpm else RIT
    if now is not None:   # (its label comes in at a beat: the tempo held within the row)
        now = min(max(float(now), start), end)
        return f"{direction} {bpm_at((start, end, bpm, start, 4, end_bpm), now):.1f} BPM"
    return f"{direction} {bpm_text(bpm)}→{bpm_text(end_bpm)} BPM"


@register_view("TempoLabelView")
class TempoLabelView(ViewPlugin):
    """The beatgrid's tempos at the top of the main plot, a label each as the key and phrase labels: it comes in
    at the beat nearest its tempo's start, is held right of the playhead while the tempo plays (the bar.beat
    label is left of it) and is pushed along by the tempo's end, dimmed once the next tempo's label touches it
    and after;
    a ramp's shows its start and end tempo, and the tempo at the playhead while it plays. Outlined as the
    bar.beat label; drawn just above the beatgrid's lines and under the CUE markers, so it never hides a cue.
    Part of the beatgrid's display (its emphasis the beatgrid page's)."""
    EDIT_PAGE = "beatgrid"

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot = None
        self._labels: list[pg.TextItem] = []
        self._rows: list[ShownRow] = []
        self._touch: list[float] = []   # each label's opacity for the next one touching it (1: not touched)

        self.bus.sig_time_changed.connect(self._reposition)
        self.bus.sig_window_changed.connect(self._reposition)
        self.bus.sig_center_changed.connect(self._reposition)
        self.model.sig_loaded.connect(self._rebuild)
        self.model.sig_beatgrid_changed.connect(self._rebuild)

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
        for i, label in enumerate(self._labels):
            label.setOpacity(opacity * (self._touch[i] if i < len(self._touch) else 1.0))

    def _rebuild(self, *_args) -> None:
        if self.plot is None:
            return
        self._rows = snap_to_beats(tempo_rows(self.model.tempo_segments), self.model.beats)
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
        """Each tempo's label at its start beat, held right of the playhead while it plays, pushed by its end."""
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
        spans: list[Optional[tuple[float, float]]] = []   # each shown label's left and right edge
        for i, label in enumerate(self._labels):
            if i >= len(self._rows):
                label.setVisible(False)
                spans.append(None)
                continue
            start, end, row = self._rows[i]
            if end < visible_start or start > visible_end:
                label.setVisible(False)
                spans.append(None)
                continue
            text = tempo_text(row, now if start <= now < end else None)
            if label.textItem.toPlainText() != text:
                label.setText(text)
            width = float(label.boundingRect().width()) * x_per_pixel
            x = _sticky_label_left(left + start, left + end, hold, width)
            label.setPos(x, label_y)
            label.setVisible(True)
            spans.append((x, x + width))
        touch = [touched_opacity(here, after, hold, x_per_pixel) for here, after in zip(spans, spans[1:] + [None])]
        if touch != self._touch:
            self._touch = touch
            self._apply_emphasis()


def touched_opacity(here: Optional[tuple[float, float]], after: Optional[tuple[float, float]], hold: float,
                    x_per_pixel: float) -> float:
    """A label's opacity for the next label (``after``, its left and right edge; None: not shown) coming up to it:
    1 apart, TOUCHED_OPACITY touching and once passed (the next one at ``hold``, the place right of the playhead,
    or pushed past it: its tempo come), faded over the last TOUCH_FADE_PX before they touch."""
    if here is None or after is None:
        return 1.0
    if after[0] <= hold:
        return TOUCHED_OPACITY
    gap = after[0] - here[1]
    if x_per_pixel > 0.0:
        gap /= x_per_pixel
    elif gap > 0.0:
        return 1.0
    closeness = float(np.clip(1.0 - gap / TOUCH_FADE_PX, 0.0, 1.0))
    return 1.0 - closeness * (1.0 - TOUCHED_OPACITY)
