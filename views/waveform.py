from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
from .base import ViewPlugin, register_view

@register_view("WaveformView")
class WaveformView(ViewPlugin):
    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.img = pg.ImageItem()
        if hasattr(self.img, "setAutoDownsample"):
            self.img.setAutoDownsample(True)
        try:
            self.img.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        except Exception:
            pass
        self.img.setOpts(interpolation='nearest')
        self.wave_np = None
        self.duration = 0.0
        self._wave_levels = (0, 255)
        self._rect_cache = None

        self.bus.sig_time_changed.connect(self.update_time)
        self.bus.sig_window_changed.connect(self.update_window)
        self.bus.sig_features_loaded.connect(self.update_features)

    def attach(self, plot: pg.PlotItem):
        plot.addItem(self.img)

    def detach(self):
        if self.img.scene() is not None:
            self.img.scene().removeItem(self.img)

    def render_initial(self):
        f = self.model.features
        self.duration = float(self.model.duration_sec)
        arr = f.get("wave_img_np")
        if arr is None:
            self.wave_np = None
            return
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self.wave_np = np.ascontiguousarray(arr)
        if self.wave_np is not None:
            self.img.setImage(self.wave_np, autoLevels=False, levels=self._wave_levels)
            self._set_rect(force=True)

    def update_time(self, _t: float):
        self._set_rect()

    def update_window(self, _w: float):
        self._set_rect()

    def update_features(self):
        self.render_initial()

    def _set_rect(self, force: bool = False):
        if self.wave_np is None:
            return
        left = self.tl.center_t - self.tl.current_time
        rect = QtCore.QRectF(left, 0.14, self.duration, 1.0-0.14)
        if not force and self._rect_cache is not None:
            if (
                abs(rect.x() - self._rect_cache.x()) < 1e-4
                and abs(rect.width() - self._rect_cache.width()) < 1e-4
            ):
                return
        self._rect_cache = rect
        self.img.setRect(rect)
