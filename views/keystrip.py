from PySide6 import QtCore
import numpy as np
import pyqtgraph as pg
from .base import ViewPlugin, register_view

@register_view("KeyStripView")
class KeyStripView(ViewPlugin):
    """Thin key color strip at the bottom of the main plot (y ∈ [-h, 0])."""
    def __init__(self, bus, model, tl, height=0.14):
        super().__init__(bus, model, tl)
        self.img = pg.ImageItem()
        self.key_np = None
        self.duration = 0.0
        self.h = float(height)

        self.bus.sig_time_changed.connect(self.update_time)
        self.bus.sig_window_changed.connect(self.update_window)
        self.bus.sig_features_loaded.connect(self.update_features)
        self.bus.sig_key_segments_updated.connect(self._on_key_segments_updated)

    def attach(self, plot: pg.PlotItem):
        plot.addItem(self.img)

    def detach(self):
        if self.img.scene() is not None:
            self.img.scene().removeItem(self.img)

    def render_initial(self):
        f = self.model.features
        self.key_np = f.get("key_np")
        self.duration = float(f.get("duration_sec") or 0.0)
        if self.key_np is not None and self.duration > 0:
            self.img.setImage(self.key_np, autoLevels=False, levels=(0, 255))
            self._set_rect()

    def update_time(self, _t: float):
        self._set_rect()

    def update_window(self, _w: float):
        self._set_rect()

    def update_features(self):
        self.render_initial()

    def _on_key_segments_updated(self, payload=None) -> None:
        key_img = None
        if isinstance(self.model.features, dict):
            key_img = self.model.features.get("key_np")
        if key_img is None:
            return
        arr = np.asarray(key_img)
        if arr.size == 0:
            return
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self.key_np = arr
        if self.duration <= 0:
            self.duration = float(self.model.duration_sec or 0.0)
        if self.key_np is not None and self.duration > 0:
            self.img.setImage(self.key_np, autoLevels=False, levels=(0, 255))
            self._set_rect()

    def _set_rect(self):
        if self.key_np is None or self.duration <= 0:
            return
        total_w = max(self.duration, self.tl.window_sec)
        left = self.tl.center_t - self.tl.current_time
        self.img.setRect(QtCore.QRectF(left, 0, total_w, self.h))
