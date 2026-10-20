"""The analysis pipeline map of Settings > Analysis > Global: the analyses, the order they run in and what each
needs (assets/images/analysis_pipeline.png, drawn at 2x; its block rectangles in analysis_pipeline.json).
Clicking a block opens that analysis's settings tab."""

from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtWidgets

from core.resource_paths import resource_path

IMAGE = "assets/images/analysis_pipeline.png"
REGIONS = "assets/images/analysis_pipeline.json"
SHOWN_WIDTH = 680
# block -> the settings tab it opens (the audio decode / HPSS has no settings)
TARGETS = {
    "Waveform": "View",              # the waveform's bands are view settings
    "Beat grid": "Beat",
    "Phrase": "Phrase",
    "CUE points": "CUE",
    "JumpCUE": "CUE",
    "Key": "Key",
}
HINT = "The order the analyses run in and what each needs. Click an analysis to open its settings."


class PipelineMap(QtWidgets.QLabel):
    """The pipeline image; ``tabRequested`` names the settings tab of the block clicked."""

    tabRequested = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.setMouseTracking(True)
        self.setToolTip(HINT)
        self._blocks: dict[str, QtCore.QRectF] = {}
        self._scale = 1.0
        self._hovered: str | None = None
        pixmap = QtGui.QPixmap(str(resource_path(IMAGE)))
        if pixmap.isNull():
            return
        shown = pixmap.scaledToWidth(2 * SHOWN_WIDTH, QtCore.Qt.SmoothTransformation)
        shown.setDevicePixelRatio(2.0)
        self.setPixmap(shown)
        self.setFixedSize(shown.deviceIndependentSize().toSize())
        try:
            meta = json.loads(resource_path(REGIONS).read_text(encoding="utf-8"))
            self._scale = SHOWN_WIDTH / float(meta["size"][0])
            self._blocks = {name: QtCore.QRectF(*rect) for name, rect in meta["blocks"].items()}
        except (OSError, ValueError, KeyError, TypeError):
            self._blocks = {}

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
                QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), f"{name}: open the {target} settings", self)
            else:
                QtWidgets.QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            target = TARGETS.get(self.block_at(event.position()) or "")
            if target:
                self.tabRequested.emit(target)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hovered = None
        self.unsetCursor()
        super().leaveEvent(event)
