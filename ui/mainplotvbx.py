from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore


class NoDragNoWheelViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=False, y=False)
        if hasattr(self, "setWheelEnabled"):
            self.setWheelEnabled(False)

    def wheelEvent(self, ev):
        ev.ignore()

    def mouseDragEvent(self, ev, *args, **kwargs):
        ev.ignore()

    def mouseClickEvent(self, ev, *args, **kwargs):
        ev.ignore()


class ScratchViewBox(NoDragNoWheelViewBox):
    """Main view box: holding the waveform with the left button holds the deck
    (scratch), dragging moves the track under the pointer, as on rekordbox."""

    def __init__(self, bus, get_time: Callable[[], float], get_duration: Callable[[], float],
                 can_scratch: Callable[[], bool], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus
        self._get_time = get_time
        self._get_duration = get_duration
        self._can_scratch = can_scratch
        self._grab: tuple[float, float] | None = None  # (pointer x, track time) at the press

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or not self._can_scratch() or self._get_duration() <= 0.0:
            ev.ignore()
            return
        x = self._view_x(ev.scenePos())
        if x is None:
            ev.ignore()
            return
        self._grab = (x, float(self._get_time()))
        self.bus.sig_scratch_begin.emit()
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._grab is None:
            ev.ignore()
            return
        t = self._time_at(ev.scenePos())
        if t is not None:
            self.bus.sig_scratch_update.emit(t)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._grab is None or ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        t = self._time_at(ev.scenePos())
        if t is not None:
            self.bus.sig_scratch_update.emit(t)
        self._grab = None
        self.bus.sig_scratch_end.emit(t if t is not None else float(self._get_time()))
        ev.accept()

    def _view_x(self, scene_pos) -> float | None:
        x = self.mapSceneToView(scene_pos).x()
        return float(x) if np.isfinite(x) else None

    def _time_at(self, scene_pos) -> float | None:
        """Track time under the pointer: the grabbed time moved against the drag
        (the view's x is in track seconds)."""
        x = self._view_x(scene_pos)
        if x is None or self._grab is None:
            return None
        x0, t0 = self._grab
        return float(np.clip(t0 - (x - x0), 0.0, max(0.0, float(self._get_duration()))))
