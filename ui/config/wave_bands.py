"""Settings > View's waveform bands, set by ear and eye (pyqtgraph, as the views): the master's spectrum as it plays
(the native engine's, audioengine.controls.spectrum_edges) under the low / mid / high ranges as three overlapping
range sliders (BandSpectrum), and under them a small waveform drawn as the main view will draw it, from the loaded
track at the playhead, with the settings being edited (WavePreview: each piece drawn once, with the settings it was
drawn with; a change shows from the next piece on). Both run only while shown, at a low rate and resolution."""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from audioengine.controls import spectrum_edges
from system.i18n import tr
from ui.theme import current_theme
from views.render.wave import THREE_BAND_HIGH, THREE_BAND_LOW, THREE_BAND_MID
from views.waveform import _WaveStyle, column_hop, render_columns

# the bands' edges and the spectrum shown, 20 Hz - 20 kHz (the engine's bins above it are not drawn)
F_MIN, F_MAX = 20.0, 20000.0
# the spectrum as shown: each bin's power (its density times its width: dB re full scale, a full-scale sine about
# -3 dB in its bin), from DB_BOTTOM to DB_TOP
DB_TOP, DB_BOTTOM = 0.0, -60.0
DB_FALL_PER_FRAME = 1.0                # the line falls at most this per frame (30 a second): readable, not jumpy
MIN_RATIO = 2.0 ** (1.0 / 24.0)        # a band at least a bin of the spectrum wide
# the bands' colours as the style draws them: RGB red / green / blue, 3Band rekordbox's
BAND_COLORS = {"rgb": ((235, 70, 70), (80, 210, 90), (80, 130, 255)),
               "3band": (THREE_BAND_LOW, THREE_BAND_MID, THREE_BAND_HIGH)}


class _InsideAxis(pg.AxisItem):
    """An axis whose labels at its ends are moved in to fit (the 20k at the right edge). pyqtgraph leaves out a
    label its bounding rect does not hold: the rect is widened by EDGE_PX each side for that, and the labels moved
    in after."""

    EDGE_PX = 40

    def boundingRect(self):
        return super().boundingRect().adjusted(-self.EDGE_PX, 0, self.EDGE_PX, 0)

    def generateDrawSpecs(self, p):
        specs = super().generateDrawSpecs(p)
        if specs is None:
            return None
        axis, ticks, texts = specs
        right = float(self.width())
        moved = []
        for rect, flags, text in texts:
            if rect.right() > right:
                rect = rect.translated(right - rect.right(), 0)
            if rect.left() < 0:
                rect = rect.translated(-rect.left(), 0)
            moved.append((rect, flags, text))
        return axis, ticks, moved


def _still_plot(height: int) -> pg.PlotWidget:
    """A plot that only shows: no pan, zoom, menu or buttons; its bottom axis's labels kept inside it."""
    plot = pg.PlotWidget(background=current_theme().base, axisItems={"bottom": _InsideAxis("bottom")})
    plot.setMouseEnabled(x=False, y=False)
    plot.setMenuEnabled(False)
    plot.hideButtons()
    plot.setFixedHeight(height)
    plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
    return plot


class BandSpectrum(QtWidgets.QWidget):
    """The spectrum (log frequency, dB) with the three bands over it, each a range slider as tall as it (pyqtgraph's
    LinearRegionItem: its edges or the whole band dragged; where they overlap, the higher band's on top).
    ``edited`` while dragging; ``shown`` when it comes into sight or leaves it (the spectrum is made only
    meanwhile)."""

    edited = QtCore.Signal(int, float, float)   # band (0 low, 1 mid, 2 high), its low and high edge (Hz)
    shown = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        theme = current_theme()
        self._plot = _still_plot(140)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        item = self._plot.getPlotItem()
        item.setLogMode(x=True, y=False)
        item.hideAxis("left")
        axis = item.getAxis("bottom")
        axis.setPen(pg.mkPen(theme.separator))
        axis.setTextPen(pg.mkPen(theme.text_dim))
        axis.setTicks([[(math.log10(hz), label) for hz, label in (
            (50, "50"), (100, "100"), (200, "200"), (500, "500"), (1000, "1k"), (2000, "2k"), (5000, "5k"),
            (10000, "10k"), (20000, "20k"))]])
        item.showGrid(x=True, y=False, alpha=0.25)
        # a hair wider than the bands' range: pyqtgraph leaves out a tick on the range's very edge (the 20k)
        item.setXRange(math.log10(F_MIN), math.log10(F_MAX * 1.0005), padding=0)
        item.setYRange(DB_BOTTOM, DB_TOP, padding=0)
        edges = np.asarray(spectrum_edges())
        self._centres = np.sqrt(edges[:-1] * edges[1:])
        self._width_db = 10.0 * np.log10(np.diff(edges))     # a bin's density (per Hz) to its power
        self._shown = self._centres <= F_MAX                  # the bins drawn
        fill = QtGui.QColor(theme.text)
        fill.setAlpha(40)
        self._curve = pg.PlotDataItem(pen=pg.mkPen(theme.text, width=1.2), fillLevel=DB_BOTTOM, brush=fill)
        item.addItem(self._curve)
        self._note = pg.TextItem("", color=theme.text_dim, anchor=(0.5, 0.5))
        self._note.setPos(0.5 * (math.log10(F_MIN) + math.log10(F_MAX)), 0.5 * (DB_TOP + DB_BOTTOM))
        item.addItem(self._note)
        bounds = (math.log10(F_MIN), math.log10(F_MAX))
        self._regions = []
        for band in range(3):
            region = pg.LinearRegionItem(values=(0.0, 1.0), orientation="vertical", bounds=bounds, swapMode="block")
            region.setZValue(10 + band)
            region.sigRegionChanged.connect(lambda r, band=band: self._region_moved(band, r))
            item.addItem(region)
            self._regions.append(region)
        self._db: Optional[np.ndarray] = None
        self._setting = False
        self.set_style("rgb")
        self.set_bands(((20.0, 200.0), (250.0, 1500.0), (1600.0, 11025.0)))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.shown.emit(True)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.shown.emit(False)

    # ---- what it shows ----
    def set_bands(self, bands) -> None:
        """The bands' edges (Hz) as set elsewhere (the number fields): the sliders follow, nothing emitted."""
        self._setting = True
        try:
            for region, (lo, hi) in zip(self._regions, bands):
                lo, hi = max(float(lo), F_MIN), min(float(hi), F_MAX)
                if not region.mouseHovering:
                    region.setRegion((math.log10(lo), math.log10(max(hi, lo * MIN_RATIO))))
        finally:
            self._setting = False

    def set_style(self, style: str) -> None:
        colors = BAND_COLORS.get(style, BAND_COLORS["rgb"])
        for region, color in zip(self._regions, colors):
            tint = QtGui.QColor(*color)
            tint.setAlpha(45)
            hover = QtGui.QColor(*color)
            hover.setAlpha(80)
            region.setBrush(tint)
            region.setHoverBrush(hover)
            for line in region.lines:
                line.setPen(pg.mkPen(color, width=2))
                line.setHoverPen(pg.mkPen(color, width=4))
            region.update()     # pyqtgraph's setBrush asks for no repaint: the parts not drawn again kept the old one

    def set_note(self, text: str) -> None:
        """A line shown where the spectrum is (none: "")."""
        self._note.setText(text)

    def add_frames(self, frames: np.ndarray) -> None:
        """New spectrum frames (``[k, SPECTRUM_BINS]`` density dB): the line (each bin's power) follows them,
        falling at most DB_FALL_PER_FRAME a frame."""
        for frame in frames:
            frame = np.asarray(frame, dtype=np.float64) + self._width_db
            self._db = frame if self._db is None else np.maximum(frame, self._db - DB_FALL_PER_FRAME)
        if len(frames):
            self._curve.setData(self._centres[self._shown], np.clip(self._db[self._shown], DB_BOTTOM, DB_TOP))

    def clear(self) -> None:
        self._db = None
        self._curve.setData([], [])

    def _region_moved(self, band: int, region: pg.LinearRegionItem) -> None:
        if self._setting:
            return
        lo, hi = (10.0 ** v for v in region.getRegion())
        if hi < lo * MIN_RATIO:
            hi = lo * MIN_RATIO
        self.edited.emit(band, round(lo, 1), round(hi, 1))


# ---- the waveform preview ----

PREVIEW_HEIGHT = 48
PREVIEW_TICK_MS = 100           # how often it looks whether more is to be drawn
PREVIEW_AHEAD = 0.25            # the share of its width ahead of the playhead (drawn before it is heard)
PREVIEW_BATCH_SEC = 0.5         # the most drawn at once (a piece)


class _PieceSignals(QtCore.QObject):
    done = QtCore.Signal(object, float, float, int)   # image [columns, H, 3] or None, start, end, the track's gen


class _PieceJob(QtCore.QRunnable):
    def __init__(self, pcm, rate: int, c0: int, c1: int, style: _WaveStyle, gen: int) -> None:
        super().__init__()
        self.pcm, self.rate, self.c0, self.c1, self.style, self.gen = pcm, rate, c0, c1, style, gen
        self.signals = _PieceSignals()

    def run(self) -> None:
        hop = column_hop(self.style, self.rate)
        try:
            image = render_columns(self.pcm, self.rate, self.c0, self.c1, self.style, height=PREVIEW_HEIGHT)
        except Exception as exc:
            print(f"[Settings] waveform preview not drawn: {exc}")
            image = None
        self.signals.done.emit(image, self.c0 * hop / self.rate, self.c1 * hop / self.rate, self.gen)


class WavePreview(QtWidgets.QWidget):
    """The loaded track's waveform around the playhead (a quarter of the width ahead of it), drawn as the main
    view draws it with ``style()`` (the settings being edited) a piece at a time, a piece ahead of the playhead on
    a thread of its own: each piece an image placed at its time, kept as it was drawn."""

    def __init__(self, style: Callable[[], tuple[_WaveStyle, float]], parent=None):
        """``style()``: the waveform's settings as edited, and the seconds the preview spans (the main view's
        width they are for)."""
        super().__init__(parent)
        self._style = style
        self._plot = _still_plot(PREVIEW_HEIGHT)
        self._plot.setBackground((0, 0, 0))
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot)
        item = self._plot.getPlotItem()
        item.hideAxis("left")
        item.hideAxis("bottom")
        item.setYRange(0, PREVIEW_HEIGHT, padding=0)
        self._view = item
        self._playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(current_theme().accent, width=1))
        self._playhead.setZValue(10)
        item.addItem(self._playhead)
        self._note = pg.TextItem(tr("Load a track to preview the waveform."), color=current_theme().text_dim,
                                 anchor=(0.5, 0.5))
        item.addItem(self._note)
        self._pcm = None
        self._rate = 0
        self._gen = 0
        self._time = 0.0
        self._pieces: list[tuple[float, float, pg.ImageItem]] = []
        self._until: Optional[float] = None    # drawn up to here (the run the playhead is in)
        self._busy = False
        self._span = 6.0
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(PREVIEW_TICK_MS)
        self._timer.timeout.connect(self._draw_more)
        self._follow()

    def set_track(self, pcm, rate: int) -> None:
        """The loaded track's audio (None: none): what was drawn of the last goes."""
        self._pcm, self._rate = pcm, int(rate or 0)
        self._gen += 1
        for _start, _end, image in self._pieces:
            self._view.removeItem(image)
        self._pieces.clear()
        self._until = None
        self._note.setVisible(pcm is None)

    def set_time(self, sec: float) -> None:
        self._time = float(sec)
        if self.isVisible():
            self._follow()

    def start(self) -> None:
        self._timer.start()
        self._follow()

    def stop(self) -> None:
        self._timer.stop()

    def _follow(self) -> None:
        """The view's span around the playhead (PREVIEW_AHEAD of it ahead)."""
        t = self._time
        self._view.setXRange(t - (1.0 - PREVIEW_AHEAD) * self._span, t + PREVIEW_AHEAD * self._span, padding=0)
        self._playhead.setPos(t)
        self._note.setPos(t, PREVIEW_HEIGHT / 2)

    def _draw_more(self) -> None:
        """The next piece, when the drawn part does not reach PREVIEW_AHEAD of the width ahead of the playhead."""
        if self._busy or self._pcm is None or self._rate <= 0:
            return
        style, span = self._style()
        if span != self._span:
            self._span = span
            self._follow()
        t = self._time
        if self._until is None or t > self._until + 1.0 or t < self._until - self._span:   # a jump: a new run
            self._until = max(0.0, t - 0.1)
        target = min(t + PREVIEW_AHEAD * self._span + 0.2, self._pcm.shape[0] / self._rate)
        if self._until >= target - 0.02:
            return
        hop = column_hop(style, self._rate)
        c0 = int(self._until * self._rate // hop)
        c1 = int(math.ceil(min(target, self._until + PREVIEW_BATCH_SEC) * self._rate / hop))
        if c1 <= c0:
            return
        self._busy = True
        job = _PieceJob(self._pcm, self._rate, c0, c1, style, self._gen)
        job.signals.done.connect(self._piece_done, QtCore.Qt.QueuedConnection)
        self._pool.start(job)

    @QtCore.Slot(object, float, float, int)
    def _piece_done(self, image, start: float, end: float, gen: int) -> None:
        self._busy = False
        if gen != self._gen:
            return
        self._until = end
        if image is not None and image.size:
            piece = pg.ImageItem(np.ascontiguousarray(image[:, ::-1]), axisOrder="col-major", levels=(0, 255))
            piece.setRect(QtCore.QRectF(start, 0.0, end - start, PREVIEW_HEIGHT))
            self._view.addItem(piece)
            self._pieces.append((start, end, piece))
        # what is out of sight goes
        gone = [p for p in self._pieces if p[1] < self._time - self._span]
        for _start, _end, piece in gone:
            self._view.removeItem(piece)
        self._pieces = [p for p in self._pieces if p not in gone]
