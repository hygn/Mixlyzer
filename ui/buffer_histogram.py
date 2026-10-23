"""The output buffer's histogram (Settings > Audio): how long the buffer spent at each level over the recent
window, so a buffer that nears empty now and then shows even when its average looks safe."""
from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ui.theme import current_theme


class BufferHistogram(QtWidgets.QWidget):
    """Time at each level, left (empty: a dropout) to right (the capacity and a little above), in columns of
    the widget's width; heights on a log scale (0.01 % .. 100 % of the time), so rare low levels show. Colours
    run as the buffer meter's: red near empty, green full. Lines: the capacity, and the 1 % low."""

    LOG_FLOOR = -4.0   # 0.01 % of the window's time: the lowest share drawn

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: np.ndarray | None = None   # share of the time per column (0..1)
        self._span_ms = 1.0                       # the level at the right edge
        self._capacity_ms = 0.0
        self._low_ms = 0.0
        self._micro = False
        self.setFixedHeight(72)
        self.setMinimumWidth(160)

    def set_data(self, seconds_per_bucket: np.ndarray, bucket_ms: float, capacity_ms: float, low_ms: float,
                 micro: bool) -> None:
        """``seconds_per_bucket``: the window's time at each ``bucket_ms`` level; the capacity and the 1 % low in
        ms; ``micro``: levels labelled in µs (else ms)."""
        total = float(seconds_per_bucket.sum())
        if total <= 0.0:
            self._columns = None
            self.update()
            return
        top = int(np.flatnonzero(seconds_per_bucket)[-1]) + 1
        self._span_ms = max(capacity_ms * 1.25, top * bucket_ms, bucket_ms)
        self._shares = seconds_per_bucket / total
        self._bucket_ms = bucket_ms
        self._columns = self._shares
        self._capacity_ms = capacity_ms
        self._low_ms = low_ms
        self._micro = micro
        self.update()

    def _label(self, ms: float) -> str:
        return f"{ms * 1000:,.0f} µs" if self._micro else f"{ms:.0f} ms"

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        theme = current_theme()
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(2, 2, -2, -14)
        painter.fillRect(rect, QtGui.QColor("#111"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#444"), 1))
        painter.drawRect(rect)
        if self._columns is None:
            return
        height = rect.height() - 2
        bottom = rect.bottom()
        width = rect.width() - 2
        colours = [QtGui.QColor("#ff4d4d"), QtGui.QColor("#ffcf40"), QtGui.QColor("#38d26b")]
        # the buckets into pixel columns: each bucket over its own width (a column holding several: their sum)
        cols = np.zeros(width)
        x0 = (np.arange(self._shares.size) * self._bucket_ms / self._span_ms * width).astype(int)
        x1 = np.maximum(x0 + 1, ((np.arange(self._shares.size) + 1) * self._bucket_ms / self._span_ms * width).astype(int))
        for share, a, b in zip(self._shares, x0, x1):
            if share > 0.0 and a < width:
                cols[a:min(b, width)] = np.maximum(cols[a:min(b, width)], share) if b - a > 1 else cols[a] + share
        for x, share in enumerate(cols):
            if share <= 0.0:
                continue
            h = (max(math.log10(share), self.LOG_FLOOR) - self.LOG_FLOOR) / -self.LOG_FLOOR
            h = max(1, int(round(h * height)))
            # red near empty, yellow, green from a third of the capacity on
            level = (x / width) * self._span_ms / max(self._capacity_ms, 1e-9)
            colour = colours[0] if level < 0.12 else colours[1] if level < 0.35 else colours[2]
            painter.fillRect(QtCore.QRect(rect.left() + 1 + x, bottom - h, 1, h), colour)

        def mark(ms: float, colour: str, dashed: bool) -> int:
            x = rect.left() + 1 + int(round(ms / self._span_ms * width))
            pen = QtGui.QPen(QtGui.QColor(colour), 1)
            if dashed:
                pen.setStyle(QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(x, rect.top() + 1, x, bottom)
            return x

        mark(self._capacity_ms, theme.text_dim, True)
        mark(self._low_ms, "#ffcf40", False)
        # the scale: 0 at the left, the capacity under its line
        painter.setPen(QtGui.QColor(theme.text_dim))
        font = painter.font()
        font.setPointSizeF(max(6.0, font.pointSizeF() * 0.75))
        painter.setFont(font)
        baseline = self.rect().bottom() - 1
        painter.drawText(rect.left(), baseline, "0")
        text = self._label(self._capacity_ms)
        cx = rect.left() + 1 + int(round(self._capacity_ms / self._span_ms * width))
        tw = painter.fontMetrics().horizontalAdvance(text)
        painter.drawText(max(rect.left(), min(cx - tw // 2, rect.right() - tw)), baseline, text)
