from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from utils.color import key_label_text_color
from utils.keystrip import key_segment_label, key_segment_rows, render_keystrip
from .base import KEY_LANE, ViewPlugin, register_view
from .phrase import _sticky_label_left, _vertical_lines_path


@register_view("KeyStripView")
class KeyStripView(ViewPlugin):
    """Key colour strip in the main plot's key lane (at the bottom, under the phrase
    band), with boundary lines and a key label per segment like the phrase band."""
    def __init__(self, bus, model, tl):
        super().__init__(bus, model, tl)
        self.plot = None
        self.img = pg.ImageItem()
        self.img.setOpts(interpolation="nearest")
        self.boundaries = QtWidgets.QGraphicsPathItem()
        self.boundaries.setPen(pg.mkPen((255, 255, 255, 220), width=1, cosmetic=True))
        self.boundaries.setZValue(15)
        self._labels: list[pg.TextItem] = []
        self._rows: list[tuple[int, float, float]] = []
        self._strip = None
        self.duration = 0.0
        self.y0 = float(KEY_LANE[0])
        self.h = float(KEY_LANE[1] - KEY_LANE[0])
        self._font = QtGui.QFont()
        self._font.setPointSizeF(8.0)
        self._font.setBold(True)

        self.bus.sig_time_changed.connect(self.update_time)
        self.bus.sig_window_changed.connect(self.update_window)
        self.bus.sig_center_changed.connect(self.update_window)
        self.bus.sig_features_loaded.connect(self.update_features)
        self.bus.sig_key_segments_updated.connect(self._on_key_segments_updated)

    def attach(self, plot: pg.PlotItem):
        self.plot = plot
        plot.addItem(self.img)
        plot.addItem(self.boundaries)

    def detach(self):
        if self.plot is None:
            return
        for item in (self.img, self.boundaries, *self._labels):
            if item.scene() is not None:
                self.plot.removeItem(item)
        self._labels.clear()
        self.plot = None

    def render_initial(self):
        f = self.model.features
        self.duration = float(f.get("duration_sec") or 0.0)
        self._render()

    def update_time(self, _t: float):
        self._set_rect()

    def update_window(self, *_args):
        self._set_rect()

    def update_features(self):
        self.render_initial()

    def _on_key_segments_updated(self, payload=None) -> None:
        if self.duration <= 0:
            self.duration = float(self.model.duration_sec or 0.0)
        self._render()

    def _render(self) -> None:
        """Draw the strip, boundaries and labels from the current key segments
        (rendered just in time)."""
        features = self.model.features if isinstance(self.model.features, dict) else {}
        img = render_keystrip(features, self.duration)
        self._strip = img
        self._rows = key_segment_rows(features.get("key_segments")) if img is not None else []
        if img is None or self.duration <= 0:
            self.img.clear()
        else:
            self.img.setImage(img, autoLevels=False, levels=(0, 255))

        edges = [t for _idx, start, end in self._rows for t in (start, end)]
        self.boundaries.setPath(_vertical_lines_path(edges, self.y0, self.y0 + self.h))

        if self.plot is not None:
            while len(self._labels) < len(self._rows):
                label = pg.TextItem("", anchor=(0.0, 0.5))
                label.setFont(self._font)
                label.setZValue(16)
                self.plot.addItem(label)
                self._labels.append(label)
        for i, label in enumerate(self._labels):
            if i < len(self._rows):
                idx = self._rows[i][0]
                label.setText(key_segment_label(idx))
                label.setColor(key_label_text_color(idx))
            else:
                label.setText("")
                label.setVisible(False)
        self._set_rect()

    def _set_rect(self):
        if self._strip is None or self.duration <= 0:
            for label in self._labels:
                label.setVisible(False)
            return
        total_w = max(self.duration, self.tl.window_sec)
        left = self.tl.center_t - self.tl.current_time
        self.img.setRect(QtCore.QRectF(left, self.y0, total_w, self.h))
        self.boundaries.setPos(left, 0.0)
        self._position_labels(left)

    def _position_labels(self, left: float) -> None:
        """Every segment on screen gets its label (short ones too), held at the
        view centre until the segment's edges push it along, as the phrase band does."""
        if self.plot is None or not self._labels:
            return
        window = max(0.1, float(self.tl.window_sec))
        visible_start = float(self.tl.current_time) - float(self.tl.center_t)
        visible_end = visible_start + window
        try:
            x_per_pixel = abs(float(self.plot.getViewBox().viewPixelSize()[0]))
        except Exception:
            x_per_pixel = 0.0
        label_y = self.y0 + self.h * 0.5
        center = float(self.tl.center_t)
        for i, label in enumerate(self._labels):
            if i >= len(self._rows):
                label.setVisible(False)
                continue
            _idx, start, end = self._rows[i]
            if end < visible_start or start > visible_end:
                label.setVisible(False)
                continue
            label_w = float(label.boundingRect().width()) * x_per_pixel
            label.setPos(_sticky_label_left(left + start, left + end, center, label_w), label_y)
            label.setVisible(True)
