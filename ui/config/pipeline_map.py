"""The analysis pipeline map of Settings > Analysis > Global: the analyses, the order they run in and what each
needs (assets/images/analysis_pipeline.svg; its block rectangles and its words in analysis_pipeline.json). Drawn
from the vector image at the screen's own pixel size, so it stays sharp at any display scale, fitted to the page's
width (its height following) and centred; the words are drawn over it in the UI's language (a translation too wide
for its room is drawn smaller). Clicking a block opens that analysis's settings tab."""

from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from system.i18n import tr, tr_mark
from system.resource_paths import resource_path
from ui.widgets.background_paint import BackgroundPaint

IMAGE = "assets/images/analysis_pipeline.svg"
REGIONS = "assets/images/analysis_pipeline.json"
MIN_WIDTH = 480
# block -> the settings tab it opens, by its English title (the audio decode / HPSS has no settings)
TARGETS = {
    "Waveform": tr_mark("View"),     # the waveform's bands are view settings
    "Beat grid": tr_mark("Beat"),
    "Phrase": tr_mark("Phrase"),
    "CUE points": tr_mark("CUE"),
    "JumpCUE": tr_mark("CUE"),
    "Key": tr_mark("Key"),
}
# the image's words (analysis_pipeline.json "texts"), for the translation catalogs
WORDS = (
    tr_mark("file"), tr_mark("audio"), tr_mark("beat grid"), tr_mark("phrases"),
    tr_mark("Audio"), tr_mark("decoded, HPSS"), tr_mark("Waveform"), tr_mark("overview levels"),
    tr_mark("Beat grid"), tr_mark("beats, tempo, bars"), tr_mark("Phrase"), tr_mark("sections + labels"),
    tr_mark("CUE points"), tr_mark("by phrase label"), tr_mark("JumpCUE"), tr_mark("jump points"),
    tr_mark("Key"), tr_mark("key sections"),
)


def _draw_map(p: QtGui.QPainter, size: QtCore.QSizeF, state) -> None:
    """The map at ``size`` (on the paint thread: its own SVG renderer, the widget's font and DPI from ``state``): the
    image as wide as it fits at its aspect, centred, its words over it in the UI's language."""
    path, image_size, texts, font, dpi = state
    renderer = QtSvg.QSvgRenderer(path)
    scale = min(size.width() / image_size.width(), size.height() / image_size.height())
    w, h = image_size.width() * scale, image_size.height() * scale
    origin = QtCore.QPointF((size.width() - w) / 2, (size.height() - h) / 2)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setRenderHint(QtGui.QPainter.TextAntialiasing)
    renderer.render(p, QtCore.QRectF(origin, QtCore.QSizeF(w, h)))
    p.translate(origin)
    for item in texts:
        _draw_text(p, item, scale, font, dpi)


def _draw_text(p: QtGui.QPainter, item: dict, scale: float, base: QtGui.QFont, dpi: float) -> None:
    text = tr(str(item.get("text", "")))
    size = float(item.get("size", 9.0)) * scale            # pixels
    room = float(item.get("room", 1e9)) * scale
    font = QtGui.QFont(base)
    font.setPointSizeF(size * 72.0 / max(1.0, dpi))
    font.setHintingPreference(QtGui.QFont.PreferNoHinting)   # spaced as the image's outlined words were
    width = QtGui.QFontMetricsF(font).horizontalAdvance(text)
    if width > room > 0:
        font.setPointSizeF(font.pointSizeF() * room / width)
        width = QtGui.QFontMetricsF(font).horizontalAdvance(text)
    x = float(item.get("x", 0.0)) * scale
    if item.get("align") == "center":
        x -= width / 2
    p.setFont(font)
    p.setPen(QtGui.QColor(str(item.get("color", "#d8d8d8"))))
    p.drawText(QtCore.QPointF(x, float(item.get("y", 0.0)) * scale), text)


class PipelineMap(QtWidgets.QWidget):
    """The pipeline image; ``tabRequested`` names the settings tab of the block clicked (its title, translated)."""

    tabRequested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setToolTip(tr("Numbers: the order the analyses run in. A wire: the analysis at the arrow needs that "
                           "result. Every analysis also reads the decoded audio (22050 Hz, HPSS). Click an analysis "
                           "to open its settings."))
        self._blocks: dict[str, QtCore.QRectF] = {}
        self._texts: list[dict] = []
        self._size = QtCore.QSizeF(1.0, 1.0)     # the image's own size
        self._scale = 1.0
        self._origin = QtCore.QPointF()
        self._hovered: str | None = None
        self._renderer = QtSvg.QSvgRenderer(str(resource_path(IMAGE)), self)    # its size and validity (GUI)
        # painted off the GUI thread (ui.widgets.background_paint): the image's 1000 dots and its words, ~10 ms
        self._paint = BackgroundPaint(self, _draw_map)
        if not self._renderer.isValid():
            return
        try:
            meta = json.loads(resource_path(REGIONS).read_text(encoding="utf-8"))
            width, height = (float(v) for v in meta["size"])
            self._blocks = {name: QtCore.QRectF(*rect) for name, rect in meta["blocks"].items()}
            self._texts = list(meta.get("texts", ()))
        except (OSError, ValueError, KeyError, TypeError):
            view = self._renderer.viewBoxF()
            width, height = view.width(), view.height()
            self._blocks = {}
        self._size = QtCore.QSizeF(width, height)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumWidth(MIN_WIDTH)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return round(width * self._size.height() / self._size.width())

    def sizeHint(self) -> QtCore.QSize:
        width = round(self._size.width())
        return QtCore.QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(MIN_WIDTH, self.heightForWidth(MIN_WIDTH))

    def _place(self) -> QtCore.QRectF:
        """The image's rectangle: as wide as the widget allows at its aspect, centred."""
        self._scale = min(self.width() / self._size.width(), self.height() / self._size.height())
        w, h = self._size.width() * self._scale, self._size.height() * self._scale
        self._origin = QtCore.QPointF((self.width() - w) / 2, (self.height() - h) / 2)
        return QtCore.QRectF(self._origin, QtCore.QSizeF(w, h))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._repaint()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._repaint()

    def _repaint(self) -> None:
        """The picture painted again at the widget's size, off the GUI thread (_draw_map); the widget then draws it
        as it is until its size changes."""
        if self.isVisible() and self._renderer.isValid():
            self._paint.request((str(resource_path(IMAGE)), QtCore.QSizeF(self._size), list(self._texts),
                                 QtGui.QFont(self.font()), float(self.logicalDpiY())))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        self._paint.draw_on(painter, QtCore.QRectF(self.rect()))
        painter.end()

    def block_at(self, pos: QtCore.QPointF) -> str | None:
        self._place()
        point = (pos - self._origin) / self._scale
        return next((name for name, rect in self._blocks.items() if rect.contains(point)), None)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        name = self.block_at(event.position())
        target = TARGETS.get(name or "")
        if name != self._hovered:
            self._hovered = name
            self.setCursor(QtCore.Qt.PointingHandCursor if target else QtCore.Qt.ArrowCursor)
            if target:
                QtWidgets.QToolTip.showText(event.globalPosition().toPoint(),
                                            tr("{name}: open the {tab} settings").format(name=tr(name), tab=tr(target)),
                                            self)
            else:
                QtWidgets.QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            target = TARGETS.get(self.block_at(event.position()) or "")
            if target:
                self.tabRequested.emit(tr(target))
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hovered = None
        self.unsetCursor()
        super().leaveEvent(event)
