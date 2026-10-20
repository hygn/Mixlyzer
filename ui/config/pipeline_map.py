"""The analysis pipeline map of Settings > Analysis > Global: the analyses, the order they run in and what each
needs (assets/images/analysis_pipeline.svg; its block rectangles and its words in analysis_pipeline.json). Drawn
from the vector image at the screen's own pixel size, so it stays sharp at any display scale; the words are drawn
over it in the UI's language (a translation too wide for its room is drawn smaller). Clicking a block opens that
analysis's settings tab."""

from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from core.i18n import tr, tr_mark
from core.resource_paths import resource_path

IMAGE = "assets/images/analysis_pipeline.svg"
REGIONS = "assets/images/analysis_pipeline.json"
SHOWN_WIDTH = 680
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
        self._scale = 1.0
        self._hovered: str | None = None
        self._renderer = QtSvg.QSvgRenderer(str(resource_path(IMAGE)), self)
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
        self._scale = SHOWN_WIDTH / width
        self.setFixedSize(SHOWN_WIDTH, round(height * self._scale))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self._renderer.isValid():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        self._renderer.render(painter, QtCore.QRectF(self.rect()))
        for item in self._texts:
            self._draw_text(painter, item)
        painter.end()

    def _draw_text(self, painter: QtGui.QPainter, item: dict) -> None:
        text = tr(str(item.get("text", "")))
        size = float(item.get("size", 9.0)) * self._scale            # pixels
        room = float(item.get("room", 1e9)) * self._scale
        font = QtGui.QFont(self.font())
        font.setPointSizeF(size * 72.0 / max(1.0, self.logicalDpiY()))
        font.setHintingPreference(QtGui.QFont.PreferNoHinting)   # spaced as the image's outlined words were
        width = QtGui.QFontMetricsF(font).horizontalAdvance(text)
        if width > room > 0:
            font.setPointSizeF(font.pointSizeF() * room / width)
            width = QtGui.QFontMetricsF(font).horizontalAdvance(text)
        x = float(item.get("x", 0.0)) * self._scale
        if item.get("align") == "center":
            x -= width / 2
        painter.setFont(font)
        painter.setPen(QtGui.QColor(str(item.get("color", "#d8d8d8"))))
        painter.drawText(QtCore.QPointF(x, float(item.get("y", 0.0)) * self._scale), text)

    def block_at(self, pos: QtCore.QPointF) -> str | None:
        point = QtCore.QPointF(pos.x() / self._scale, pos.y() / self._scale)
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
