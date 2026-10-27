from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget


def groups_page(*groups: QWidget) -> QScrollArea:
    """A page of groups, top to bottom, scrolling (a settings page's sub-page)."""
    widget = QWidget()
    lay = QVBoxLayout(widget)
    for group in groups:
        lay.addWidget(group)
    lay.addStretch(1)
    return scroll_page(widget)


def scroll_page(page: QWidget) -> QScrollArea:
    """``page`` in a vertical scroll area (a settings page taller than the dialog
    scrolls instead of stretching it), on the window background like the page."""
    area = QScrollArea()
    area.setObjectName("settingsPageScroll")
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.viewport().setObjectName("settingsPageViewport")
    area.viewport().setBackgroundRole(QtGui.QPalette.Window)
    area.viewport().setAutoFillBackground(True)
    page.setObjectName(page.objectName() or "settingsPage")
    page.setBackgroundRole(QtGui.QPalette.Window)
    page.setAutoFillBackground(True)
    area.setWidget(page)
    area.setStyleSheet(
        "QScrollArea#settingsPageScroll, QWidget#settingsPageViewport { "
        "background-color: palette(window); }"
    )
    return area
