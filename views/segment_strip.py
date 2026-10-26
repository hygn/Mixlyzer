from PySide6 import QtCore, QtGui
import pyqtgraph as pg


class SegmentStrip(pg.GraphicsObject):
    """A colour band of time segments, drawn as one filled rectangle each (in track
    seconds), so its edges stay exact at any zoom (an image of the whole track
    blurs them when the view is narrow)."""

    def __init__(self):
        super().__init__()
        self._segments: list[tuple[float, float, QtGui.QColor]] = []
        self._y0 = 0.0
        self._h = 0.0
        self._bounds = QtCore.QRectF()

    def set_segments(self, segments, y0: float, h: float) -> None:
        """``segments``: (start, end, (r, g, b)); an empty list clears the strip."""
        self.prepareGeometryChange()
        self._segments = [(float(s), float(e), QtGui.QColor(*color)) for s, e, color in segments if e > s]
        self._y0, self._h = float(y0), float(h)
        if self._segments:
            start = min(s for s, _e, _c in self._segments)
            end = max(e for _s, e, _c in self._segments)
            self._bounds = QtCore.QRectF(start, self._y0, end - start, self._h)
        else:
            self._bounds = QtCore.QRectF()
        self.update()

    def clear(self) -> None:
        self.set_segments([], self._y0, self._h)

    def boundingRect(self) -> QtCore.QRectF:
        return self._bounds

    def paint(self, painter, *_args) -> None:
        painter.setPen(QtCore.Qt.NoPen)
        for start, end, color in self._segments:
            painter.fillRect(QtCore.QRectF(start, self._y0, end - start, self._h), color)
