"""Where the marks just changed blinks once (the model's sig_change_made / last_changes: an edit, an undo or a redo, a
reanalysis's result), only over that stretch of the time axis, so a change off the screen, or too small to see, is seen
too. It is gone at once and asks for nothing; Settings > View turns it off (`viewconfig.highlight_changes`). Changes in a
row (a beatgrid nudged again and again) do not blink each: the latest stays on until they stop (`HOLD_MS`). A key's strip darkens (`LitBands`: the strip is at its full colour,
so only down is left that keeps its hue), a phrase's strip is at its full colour (the strip draws it faint), a beatgrid's
own lines thicken, a CUE point gets a white CUE marker, a JumpCUE a white JumpCUE with its letter; everything else stays as it is. A beatgrid's lines thicken in step with
what is drawn: the view's own paths and pens, again one device pixel to each side, so they lie on its pixels at any
position (a wider pen rounds on its own). `LitBands`, `Blinker` and the functions also blink the overview."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from analyzer.music.beat_geometry import downbeat_times
from core.marks.phrases import UNASSIGNED, phrase_color
from core.settings.store import current as current_settings
from core.track.changes import CUE, JUMPCUE, KEY, PHRASE

from .JumpCUE import JumpCUEView, JumpCueMarker, marker_font
from .base import KEY_LANE, PHRASE_LANE, WAVEFORM_LANE, ViewPlugin, register_view
from .beatgrid import BeatgridView
from .cue_points import _down_triangle_path
from .segment_strip import SegmentStrip

BLINK_MS = 90           # on, then off
BLINKS = 1
# A change this soon after the last is one of a row: it stays on, for this long after the last of them. So two blinks are
# never closer than this, under WCAG 2.3.1's three flashes in any second however fast the edits come; and a key held
# down (its repeats start after 0.25 to 1 s, 0.5 s by default on Windows) stays on from its first repeat.
HOLD_MS = 820
LIT_UNASSIGNED = (150, 150, 160)    # UNASSIGNED's own colour is nearly black: it would not show
LIT_DARKEN = 0.6        # how much of a key's strip goes dark: as far from its colour as a phrase's step (OKLab)
MIN_PX = 4             # the least width of a stretch (device pixels)
LINE_OFFSETS = (-1, 0, 1)   # device pixels: a beatgrid's lines while lit, the view's own lines again at each
CUE_Y = 0.985           # a CUE marker's height (views.cue_points)
MARKER_SIZE = 16
JUMP_CUE_Y = 1.0 - JumpCUEView.TOP_PADDING - JumpCUEView.BLOCK_HEIGHT     # a JumpCUE's height, as views.JumpCUE draws it
MARKER_Z = 112          # over a JumpCUE or CUE point already there


class LitBands(pg.GraphicsObject):
    """Rectangles (track seconds x plot units) darkening what is under them while lit; ``solid``: a plain fill, where
    there is nothing under them to darken (black)."""

    def __init__(self, solid: Optional[tuple[int, int, int]] = None):
        super().__init__()
        self._rects: list[QtCore.QRectF] = []
        self._bounds = QtCore.QRectF()
        self._lit = False
        self._fill = QtGui.QColor(*solid) if solid else QtGui.QColor.fromRgbF(0.0, 0.0, 0.0, LIT_DARKEN)

    def set_rects(self, rects) -> None:
        """``rects``: (x, y, width, height); none: nothing."""
        self.prepareGeometryChange()
        self._rects = [QtCore.QRectF(*r) for r in rects]
        self._bounds = QtCore.QRectF()
        for r in self._rects:
            self._bounds = self._bounds.united(r)
        self.update()

    def set_lit(self, lit: bool) -> None:
        self._lit = bool(lit)
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def paint(self, painter, *_args) -> None:
        if self._lit:
            for r in self._rects:
                painter.fillRect(r, self._fill)


class Blinker(QtCore.QObject):
    """A change told by ``run``: ``set_lit`` on and off, ``blinks`` times, then ``done``. One told within ``hold_ms`` of
    the last: on, with no blink, until ``hold_ms`` after the last, then off and ``done``."""

    def __init__(self, parent: QtCore.QObject, set_lit: Callable[[bool], None], interval_ms: int, hold_ms: int,
                 done: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self._set_lit, self._done = set_lit, done
        self._lit = False
        self._toggles_left = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._toggle)
        self._since_last = QtCore.QTimer(self)
        self._since_last.setSingleShot(True)
        self._since_last.setInterval(hold_ms)
        self._since_last.timeout.connect(self._quiet)

    def run(self, blinks: int) -> None:
        if self._since_last.isActive():
            self._timer.stop()
        else:
            self._toggles_left = 2 * blinks - 1
            self._timer.start()
        self._since_last.start()
        self._lit = True
        self._set_lit(True)

    def stop(self) -> None:
        self._timer.stop()
        self._since_last.stop()
        self._lit = False
        self._set_lit(False)

    def _toggle(self) -> None:
        self._lit = not self._lit
        self._set_lit(self._lit)
        self._toggles_left -= 1
        if self._toggles_left <= 0:
            self._timer.stop()
            self._finished()

    def _quiet(self) -> None:
        """No change for ``hold_ms``: what a row of them left on goes off (a single change's blink is over by then)."""
        if self._lit and not self._timer.isActive():
            self.stop()
            self._finished()

    def _finished(self) -> None:
        if self._done is not None:
            self._done()


def phrase_lit_segments(phrases, start: float, end: float, min_width: float) -> list[tuple[float, float, tuple[int, int, int]]]:
    """(start, end, colour) of the phrases in a stretch, at least ``min_width`` wide around its middle."""
    width = max(end - start, min_width)
    start, end = (start + end - width) / 2.0, (start + end + width) / 2.0
    segments = []
    for phrase in phrases:
        low, high = max(start, float(phrase.start)), min(end, float(phrase.end))
        if high > low:
            segments.append((low, high, LIT_UNASSIGNED if phrase.label == UNASSIGNED else phrase_color(phrase.label)))
    return segments


def band_rect(start: float, end: float, y0: float, y1: float, min_width: float) -> tuple[float, float, float, float]:
    """A rectangle between ``y0`` and ``y1`` over a stretch, at least ``min_width`` wide around its middle."""
    width = max(end - start, min_width)
    return (start + end - width) / 2.0, y0, width, y1 - y0


def _line_items(pen: tuple, z: float) -> list[QtWidgets.QGraphicsPathItem]:
    """The items of a beatgrid's lines while lit, one per LINE_OFFSETS: ``pen`` (colour, width), the view's."""
    items = []
    for _offset in LINE_OFFSETS:
        item = QtWidgets.QGraphicsPathItem()
        item.setPen(pg.mkPen(pen[0], width=pen[1], cosmetic=True))
        item.setZValue(z)
        item.setVisible(False)
        items.append(item)
    return items


@register_view("ChangeFlashView")
class ChangeFlashView(ViewPlugin):
    """Each kind of marks changed, in its lane, at its track times; each frame only moves it (left_edge)."""

    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot: pg.PlotItem | None = None
        self._band = LitBands()
        self._band.setZValue(5)         # over the key strip and the waveform; under the beat lines, which stay as they are
        self._phrase_strip = SegmentStrip()
        self._phrase_strip.setZValue(14.5)
        self._phrase_strip.setVisible(False)
        self._beats = _line_items(BeatgridView.BEAT_PEN, 10.5)            # each over the lines it thickens
        self._downbeats = _line_items(BeatgridView.DOWNBEAT_PEN, 11.5)
        self._marker = pg.ScatterPlotItem(size=MARKER_SIZE, pen=pg.mkPen((0, 0, 0), width=1),
                                          brush=pg.mkBrush(255, 255, 255), pxMode=True, symbol=_down_triangle_path())
        self._marker.setZValue(MARKER_Z)
        self._marker.setVisible(False)
        self._items = (self._band, self._phrase_strip, *self._beats, *self._downbeats, self._marker)
        self._jump_font = marker_font()
        self._jump_markers: list[tuple[float, JumpCueMarker]] = []
        self._blinker = Blinker(self, self._set_lit, BLINK_MS, HOLD_MS, done=lambda: self._draw(()))
        for sig in (bus.sig_time_changed, bus.sig_window_changed, bus.sig_center_changed):
            sig.connect(self._reposition)
        model.sig_change_made.connect(self._on_change)
        model.sig_loaded.connect(self._stop)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        for item in self._items:
            plot.addItem(item, ignoreBounds=True)

    def detach(self):
        self._blinker.stop()
        self._set_jump_markers(())
        if self.plot is not None and self._band.scene() is not None:
            for item in self._items:
                self.plot.removeItem(item)
        self.plot = None

    def render_initial(self):
        self._reposition()

    def _on_change(self) -> None:
        if not current_settings().viewconfig.highlight_changes:
            return
        self._draw(self.model.last_changes)
        self._blinker.run(BLINKS)

    def _stop(self) -> None:
        self._blinker.stop()
        self._draw(())

    def _draw(self, changes) -> None:
        """``changes`` (core.track.changes.Change): a key's stretch its strip darkened, a phrase's its strip at full
        colour, a beatgrid's its lines (none in it: the stretch over the waveform), a CUE point's or JumpCUE's a marker of
        its own shape at each end."""
        min_width = MIN_PX * (self.tl.pixel_sec if self.tl.pixel_sec > 0.0 else self.tl.window_sec / 1000.0)
        rects, phrases, beats, downbeats, cues, jump_cues = [], [], [], [], [], []
        for change in changes:
            if change.kind == CUE:
                cues += sorted({change.start, change.end})
            elif change.kind == JUMPCUE:
                jump_cues += sorted({change.start, change.end})
            elif change.kind == PHRASE:
                phrases += phrase_lit_segments(self.model.phrases, change.start, change.end, min_width)
            elif change.kind == KEY:
                rects.append(band_rect(change.start, change.end, *KEY_LANE, min_width))
            else:
                in_beats, in_downbeats = self._lines_in(change.start, change.end)
                beats += in_beats
                downbeats += in_downbeats
                if not in_beats and not in_downbeats:
                    rects.append(band_rect(change.start, change.end, *WAVEFORM_LANE, min_width))
        self._band.set_rects(rects)
        self._phrase_strip.set_segments(phrases, PHRASE_LANE[0], PHRASE_LANE[1] - PHRASE_LANE[0])
        beat_path, downbeat_path = self._line_paths(beats, downbeats)
        for item in self._beats:
            item.setPath(beat_path)
        for item in self._downbeats:
            item.setPath(downbeat_path)
        self._marker.setData(cues, [CUE_Y] * len(cues))
        self._set_jump_markers(jump_cues)
        self._reposition()

    def _set_jump_markers(self, points) -> None:
        """A white JumpCUE at each of ``points``, with its letter (none: one was taken out there)."""
        if self.plot is None:
            return
        for _point, marker in self._jump_markers:
            self.plot.removeItem(marker)
        letters = {float(cue.point): str(cue.label).strip() for group in self.model.jump_cues for cue in group.cues}
        self._jump_markers = []
        for point in points:
            marker = JumpCueMarker(letters.get(point, ""), (255, 255, 255), self._jump_font)
            marker.setZValue(MARKER_Z)
            marker.setVisible(False)
            self.plot.addItem(marker, ignoreBounds=True)
            self._jump_markers.append((point, marker))

    def _line_paths(self, beats, downbeats) -> tuple[QtGui.QPainterPath, QtGui.QPainterPath]:
        """``beats`` and ``downbeats`` as views.beatgrid draws them: its own paths over the view's height."""
        if self.plot is None:
            return QtGui.QPainterPath(), QtGui.QPainterPath()
        _x_range, (y_min, y_max) = self.plot.viewRange()
        cap = max(1e-6, y_max - y_min) * 0.08
        return (BeatgridView._build_vertical_path(beats, y_min, y_max),
                BeatgridView._build_downbeat_path(downbeats, (y_max - cap, y_max), (y_min, y_min + cap)))

    def _lines_in(self, start: float, end: float) -> tuple[list[float], list[float]]:
        """The beats (the downbeats among them, as the view draws them) and the downbeats inside a stretch."""
        down = downbeat_times(self.model.beats, self.model.tempo_segments)

        def inside(times) -> list[float]:
            return [float(t) for t in times if start <= float(t) < end]

        return inside(self.model.beats), inside(down if down is not None else ())

    def _set_lit(self, lit: bool) -> None:
        self._band.set_lit(lit)
        for item in (self._phrase_strip, *self._beats, *self._downbeats, self._marker):
            item.setVisible(lit)
        for _point, marker in self._jump_markers:
            marker.setVisible(lit)

    def _reposition(self, *_args) -> None:
        if self.plot is None:
            return
        left = self.tl.left_edge()
        for item in (self._band, self._phrase_strip, self._marker):
            item.setPos(left, 0.0)
        for point, marker in self._jump_markers:
            marker.setPos(left + point, JUMP_CUE_Y)
        pixel = self.tl.pixel_sec if self.tl.pixel_sec > 0.0 else self.tl.window_sec / 1000.0
        for items in (self._beats, self._downbeats):
            for item, offset in zip(items, LINE_OFFSETS):
                item.setPos(left + offset * pixel, 0.0)
