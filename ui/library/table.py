"""The track table: its rows painted fast, dragged and dropped."""
from __future__ import annotations

import json
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
import os
from core.library import jobs
from core.library.service import LibraryService
from ui.library.common import _PLAYLIST_TRACKS_MIME, _SORT_ROLE


class FastDelegate(QtWidgets.QStyledItemDelegate):
    """Cell delegate that caches rendered text as pixmaps.

    On this platform a QTableView scroll cannot blit and repaints the whole
    viewport each step (~19ms for a screenful of CJK text), which competes with
    the 60fps playback rendering on the GUI thread and drops frames. Caching each
    cell's rendered text as a pixmap turns the repaint into a bitblt (~7ms),
    keeping it within the frame budget. Keyed by (text, width, height, selected).
    """

    _CACHE_LIMIT = 4000

    def __init__(self, parent=None, *, art_column: int = -1, library: Optional[LibraryService] = None):
        """``library``: the thumbnails' owner (read on its thread, painted when they come)."""
        super().__init__(parent)
        self._cache: dict = {}
        self._art_column = int(art_column)
        self._library = library
        self._art: dict = {}  # (uid, width, dpr) -> QPixmap or None (no thumbnail; or asked, not come yet)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._art.clear()

    def _art_pixmap(self, uid: str, width: int, dpr: float):
        """The track's thumbnail scaled to ``width`` (logical px), or None (none, or still being read)."""
        key = (uid, width, dpr)
        if key not in self._art:
            if len(self._art) > self._CACHE_LIMIT:
                self._art.clear()
            self._art[key] = None
            if uid and self._library is not None:
                self._library.read(jobs.thumbnail, uid, max(1, round(width * dpr)),
                                   done=lambda image: self._art_came(key, image))
        return self._art[key]

    def _art_came(self, key: tuple, image) -> None:
        if image is None or key not in self._art:
            return
        pm = QtGui.QPixmap.fromImage(image)
        pm.setDevicePixelRatio(key[2])
        self._art[key] = pm
        view = self.parent()
        if isinstance(view, QtWidgets.QAbstractItemView):
            view.viewport().update()

    def paint(self, painter, option, index):
        rect = option.rect
        selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        if selected:
            painter.fillRect(rect, option.palette.highlight())
        else:
            background = index.data(QtCore.Qt.BackgroundRole)   # a row's own (Variants: its song's)
            if isinstance(background, QtGui.QBrush):
                painter.fillRect(rect, background)
        if index.column() == self._art_column:
            # the top of the art, fitted to the column's width
            dpr = painter.device().devicePixelRatio() or 1.0
            pm = self._art_pixmap(str(index.data(QtCore.Qt.UserRole) or ""), rect.width(), dpr)
            if pm is not None:
                painter.save()
                painter.setClipRect(rect)
                painter.drawPixmap(rect.topLeft(), pm)
                painter.restore()
            return
        text = index.data(QtCore.Qt.DisplayRole)
        if not text:
            return
        text = str(text)
        foreground = index.data(QtCore.Qt.ForegroundRole)
        color = foreground.color() if isinstance(foreground, QtGui.QBrush) else None
        key = (text, rect.width(), rect.height(), selected, color.rgba() if color is not None else None)
        pm = self._cache.get(key)
        if pm is None:
            if len(self._cache) > self._CACHE_LIMIT:
                self._cache.clear()
            dpr = painter.device().devicePixelRatio() or 1.0
            pm = QtGui.QPixmap(max(1, round(rect.width() * dpr)), max(1, round(rect.height() * dpr)))
            pm.setDevicePixelRatio(dpr)
            pm.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(pm)
            p.setFont(option.font)
            if color is not None:
                p.setPen(color)
            else:
                p.setPen(option.palette.highlightedText().color() if selected else option.palette.text().color())
            text_rect = QtCore.QRect(0, 0, rect.width(), rect.height()).adjusted(4, 0, -4, 0)
            p.drawText(text_rect, QtCore.Qt.AlignVCenter | QtCore.Qt.TextSingleLine, text)
            p.end()
            self._cache[key] = pm
        painter.drawPixmap(rect.topLeft(), pm)


# QTableWidget-based implementation
class _TrackTable(QtWidgets.QTableWidget):
    """Library table whose rows drag out as their audio files (dropped on the
    deck, the window loads the track); files dropped on it from elsewhere go into the library (analyzed, not
    loaded: filesDropped)."""

    playlistRowsDropped = QtCore.Signal(object, object)
    filesDropped = QtCore.Signal(object)   # paths dropped from elsewhere (a file manager): analyzed, not loaded

    def __init__(self, path_column: int, index_column: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path_column = int(path_column)
        self._index_column = int(index_column)
        self._playlist_reorder_enabled = False
        self._drop_line_y: Optional[int] = None
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.setDropIndicatorShown(True)

    def setPlaylistReorderEnabled(self, enabled: bool) -> None:
        self._playlist_reorder_enabled = bool(enabled)

    def mimeTypes(self):
        return ["text/uri-list", _PLAYLIST_TRACKS_MIME]

    def _entry_position(self, row: int) -> Optional[int]:
        """The playlist position of the table row (a playlist entry's identity)."""
        item = self.item(row, self._index_column)
        value = item.data(_SORT_ROLE) if item is not None else None
        return int(value) if value is not None else None

    def mimeData(self, items):
        rows = sorted({item.row() for item in items if not self.isRowHidden(item.row())})
        data = QtCore.QMimeData()
        urls = []
        positions = []
        for row in rows:
            item = self.item(row, self._path_column)
            if item is not None and item.text():
                urls.append(QtCore.QUrl.fromLocalFile(item.text()))
                positions.append(self._entry_position(row))
        if urls:
            data.setUrls(urls)
        if self._playlist_reorder_enabled and positions:
            # the entries by position: a track may be in the playlist more than once
            data.setData(
                _PLAYLIST_TRACKS_MIME,
                QtCore.QByteArray(json.dumps(positions).encode("utf-8")),
            )
        return data

    def startDrag(self, supported_actions):
        if self._playlist_reorder_enabled:
            self.horizontalHeader().setSortIndicator(
                self._index_column, QtCore.Qt.AscendingOrder
            )
            self.sortItems(self._index_column, QtCore.Qt.AscendingOrder)
        data = self.mimeData(self.selectedItems())
        if not data.hasUrls():
            return
        # Do not use QAbstractItemView.startDrag here. When an internal drop is
        # accepted as a move, its default cleanup removes the source rows even
        # though the DB-backed table has already been rebuilt by our handler.
        drag = QtGui.QDrag(self)
        drag.setMimeData(data)
        drag.exec(
            QtCore.Qt.CopyAction | QtCore.Qt.MoveAction,
            QtCore.Qt.CopyAction,
        )

    def _drop_boundary(self, pos: QtCore.QPoint) -> tuple[int, int]:
        index = self.indexAt(pos)
        if index.isValid():
            rect = self.visualRect(index)
            after = pos.y() >= rect.center().y()
            return index.row() + int(after), rect.bottom() + 1 if after else rect.top()
        shown = [row for row in range(self.rowCount()) if not self.isRowHidden(row)]
        if pos.y() <= 0 or not shown:
            return 0, 0
        last = self.model().index(shown[-1], 0)
        return self.rowCount(), self.visualRect(last).bottom() + 1

    def _set_drop_line(self, y: Optional[int]) -> None:
        if self._drop_line_y != y:
            self._drop_line_y = y
            self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_line_y is None:
            return
        painter = QtGui.QPainter(self.viewport())
        pen = QtGui.QPen(self.palette().highlight().color(), 3)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        y = max(1, min(self.viewport().height() - 2, self._drop_line_y))
        painter.drawLine(4, y, self.viewport().width() - 5, y)

    def dragEnterEvent(self, event):
        if (
            self._playlist_reorder_enabled
            and event.source() is self
            and event.mimeData().hasFormat(_PLAYLIST_TRACKS_MIME)
        ):
            self._set_drop_line(self._drop_boundary(event.position().toPoint())[1])
            event.setDropAction(QtCore.Qt.MoveAction)
            event.accept()
            return
        if self._external_files(event):
            event.acceptProposedAction()
            return
        self._set_drop_line(None)
        event.ignore()

    @staticmethod
    def _external_files(event) -> bool:
        """Files dragged from outside the library (its own rows carry their files too: those not taken)."""
        return event.mimeData().hasUrls() and not isinstance(event.source(), QtWidgets.QAbstractItemView)

    def dragLeaveEvent(self, event):
        self._set_drop_line(None)
        event.accept()

    def dragMoveEvent(self, event):
        if (
            self._playlist_reorder_enabled
            and event.source() is self
            and event.mimeData().hasFormat(_PLAYLIST_TRACKS_MIME)
        ):
            self._set_drop_line(self._drop_boundary(event.position().toPoint())[1])
            event.setDropAction(QtCore.Qt.MoveAction)
            event.accept()
            return
        if self._external_files(event):
            event.acceptProposedAction()
            return
        self._set_drop_line(None)
        event.ignore()

    def dropEvent(self, event):
        if self._external_files(event):
            self._set_drop_line(None)
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            self.filesDropped.emit([p for p in paths if os.path.isfile(p)])
            event.acceptProposedAction()
            return
        if not (
            self._playlist_reorder_enabled
            and event.source() is self
            and event.mimeData().hasFormat(_PLAYLIST_TRACKS_MIME)
        ):
            self._set_drop_line(None)
            event.ignore()
            return
        try:
            moved = [int(position) for position in json.loads(
                bytes(event.mimeData().data(_PLAYLIST_TRACKS_MIME)).decode("utf-8")
            )]
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            self._set_drop_line(None)
            event.ignore()
            return
        pos = event.position().toPoint()
        boundary, _line_y = self._drop_boundary(pos)
        before_position = None
        for row in range(boundary, self.rowCount()):
            position = self._entry_position(row)
            if position is not None and position not in moved and not self.isRowHidden(row):
                before_position = position
                break
        self._set_drop_line(None)
        self.playlistRowsDropped.emit(moved, before_position)
        event.setDropAction(QtCore.Qt.MoveAction)
        event.accept()
