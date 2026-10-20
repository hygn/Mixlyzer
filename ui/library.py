from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Optional
from PySide6 import QtCore, QtGui, QtWidgets
import os
from utils.labels import idx_to_labels
from utils.phrases import PHRASE_LABELS, phrase_display_name
from core import reading
from core.library_handler import TrackRow, LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.event_bus import EventBus
from core.config import config
from core.model import DataModel
from ui.edit_song import EditSongDialog
from ui.export import ExportTrackDialog
from ui.theme import current_theme
from ui.edit_panel.key_picker import KeyPickButton
from core.thumbnails import delete_thumbnail, load_thumbnail


# Data schema
@dataclass
class Track:
    track_id: str
    title: str
    artist: str
    album: str = ""
    bpm: Optional[float] = None
    key: Optional[int] = None
    duration_sec: Optional[float] = None
    rating: int = 0
    added: Optional[QtCore.QDateTime] = None
    path: str = ""
    comment: str = ""
    uid: str = ""
    playlist_index: Optional[int] = None
    analyzed: bool = True
    # search keys of the title / artist / album readings (core.reading): field -> one per reading
    reading_keys: dict = field(default_factory=dict)
    _disp_playlist_index: str = field(init=False, default="")

    _disp_bpm: str = field(init=False, default="--")
    _disp_key_camelot: str = field(init=False, default="--")
    _disp_duration: str = field(init=False, default="00:00")
    _disp_rating: str = field(init=False, default="☆☆☆☆☆")
    _disp_added: str = field(init=False, default="")

    def __post_init__(self):
        # BPM
        self._disp_bpm = f"{self.bpm:.2f}" if isinstance(self.bpm, (int, float)) else "--"
        # Key
        try:
            if isinstance(self.key, int) and 0 <= self.key <= 23:
                self._disp_key_camelot = idx_to_labels(self.key)[0]
        except Exception:
            pass
        # Duration
        if self.duration_sec and self.duration_sec > 0:
            s = int(round(self.duration_sec))
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            self._disp_duration = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        # Rating
        r = max(0, min(5, int(self.rating or 0)))
        self._disp_rating = "★"*r + "☆"*(5-r)
        # Added
        if self.added:
            self._disp_added = self.added.toString("yyyy-MM-dd HH:mm")
        if self.playlist_index is not None:
            self._disp_playlist_index = str(self.playlist_index + 1)

def row_to_track(row: TrackRow, readings: Optional[dict] = None) -> Track:
    """``readings``: the track's readings by field (LibraryDB.all_readings), for the search."""
    added_qt = QtCore.QDateTime.fromSecsSinceEpoch(int(row.added_ts)) if row.added_ts else None
    return Track(
        track_id=row.path, title=row.title or "", artist=row.artist or "",
        album=row.album or "", bpm=row.bpm, key=row.key, duration_sec=row.duration,
        rating=int(row.rating or 0), added=added_qt, path=row.path, comment=row.comment or "",
        uid=row.uid or "",
        reading_keys={f: [reading.keys(r) for r in (readings or {}).get(f, [])] for f in READ_FIELDS},
    )


# the columns found by their reading too (core.reading): the text search's columns -> Track fields
READ_FIELDS = ("title", "artist", "album")
READ_COLUMNS = {"Title": "title", "Artist": "artist", "Album": "album"}


_SOURCE_KIND_ROLE = QtCore.Qt.UserRole
_SOURCE_VALUE_ROLE = QtCore.Qt.UserRole + 1
_DIRECTORY_LOADED_ROLE = QtCore.Qt.UserRole + 2
_SORT_ROLE = QtCore.Qt.UserRole + 10
_ANALYZED_ROLE = QtCore.Qt.UserRole + 11
_PLAYLIST_MIME = "application/x-mixlyzer-playlist"
_PLAYLIST_TRACKS_MIME = "application/x-mixlyzer-playlist-tracks"
# The filters hide the rows they leave out (rows are made only when the source or the library
# changes: about 60 ms a thousand rows); a typed query is applied once typing pauses.
SEARCH_DEBOUNCE_MS = 120
# Library updates in a row (an analysis finished, Edit Track saved, a re-read) are applied once.
LIBRARY_UPDATE_DELAY_MS = 100
# A library update changing up to this many tracks changes their rows; more rebuild the table.
INCREMENTAL_ROWS_MAX = 300


class _SortItem(QtWidgets.QTableWidgetItem):
    def __lt__(self, other):
        left = self.data(_SORT_ROLE)
        right = other.data(_SORT_ROLE)
        if left is not None and right is not None:
            return left < right
        return self.text().casefold() < other.text().casefold()
class FastDelegate(QtWidgets.QStyledItemDelegate):
    """Cell delegate that caches rendered text as pixmaps.

    On this platform a QTableView scroll cannot blit and repaints the whole
    viewport each step (~19ms for a screenful of CJK text), which competes with
    the 60fps playback rendering on the GUI thread and drops frames. Caching each
    cell's rendered text as a pixmap turns the repaint into a bitblt (~7ms),
    keeping it within the frame budget. Keyed by (text, width, height, selected).
    """

    _CACHE_LIMIT = 4000

    def __init__(self, parent=None, *, art_column: int = -1, libpath: str = ""):
        super().__init__(parent)
        self._cache: dict = {}
        self._art_column = int(art_column)
        self._libpath = libpath
        self._art: dict = {}  # (uid, width, dpr) -> QPixmap or None (no thumbnail)

    def clear_cache(self) -> None:
        self._cache.clear()
        self._art.clear()

    def _art_pixmap(self, uid: str, width: int, dpr: float):
        """The track's thumbnail scaled to ``width`` (logical px), or None."""
        key = (uid, width, dpr)
        if key not in self._art:
            if len(self._art) > self._CACHE_LIMIT:
                self._art.clear()
            image = load_thumbnail(self._libpath, uid) if uid else None
            pm = None
            if image is not None:
                pm = QtGui.QPixmap.fromImage(image.scaledToWidth(
                    max(1, round(width * dpr)), QtCore.Qt.SmoothTransformation))
                pm.setDevicePixelRatio(dpr)
            self._art[key] = pm
        return self._art[key]

    def paint(self, painter, option, index):
        rect = option.rect
        selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        if selected:
            painter.fillRect(rect, option.palette.highlight())
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
            tr = QtCore.QRect(0, 0, rect.width(), rect.height()).adjusted(4, 0, -4, 0)
            p.drawText(tr, QtCore.Qt.AlignVCenter | QtCore.Qt.TextSingleLine, text)
            p.end()
            self._cache[key] = pm
        painter.drawPixmap(rect.topLeft(), pm)
# QTableWidget-based implementation
class _TrackTable(QtWidgets.QTableWidget):
    """Library table whose rows drag out as their audio files (dropped on the
    deck, the window loads the track)."""

    playlistRowsDropped = QtCore.Signal(object, object)

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
        self._set_drop_line(None)
        event.ignore()

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
        self._set_drop_line(None)
        event.ignore()

    def dropEvent(self, event):
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


class _LibrarySourceTree(QtWidgets.QTreeWidget):
    playlistFilesDropped = QtCore.Signal(str, object)
    playlistOrderDropped = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.setDropIndicatorShown(True)
        self.setMinimumWidth(190)
        self.setMaximumWidth(360)
        self._drop_line_y: Optional[int] = None
        self._drop_target_rect: Optional[QtCore.QRect] = None

    def mimeTypes(self):
        return ["text/uri-list", _PLAYLIST_MIME]

    def mimeData(self, items):
        data = QtCore.QMimeData()
        playlists = [
            item for item in items
            if item.data(0, _SOURCE_KIND_ROLE) == "playlist"
        ]
        if len(playlists) == 1:
            playlist_uid = str(playlists[0].data(0, _SOURCE_VALUE_ROLE))
            data.setData(
                _PLAYLIST_MIME,
                QtCore.QByteArray(playlist_uid.encode("ascii")),
            )
        return data

    def startDrag(self, supported_actions):
        data = self.mimeData(self.selectedItems())
        if not data.hasFormat(_PLAYLIST_MIME):
            return
        # Custom move: the database owns the order, so Qt must not remove the
        # source QTreeWidgetItem after the drop completes.
        drag = QtGui.QDrag(self)
        drag.setMimeData(data)
        drag.exec(QtCore.Qt.MoveAction, QtCore.Qt.MoveAction)

    def _playlist_insertion_at(
        self, pos: QtCore.QPoint
    ) -> tuple[bool, Optional[str], Optional[int]]:
        item = self.itemAt(pos)
        kind = item.data(0, _SOURCE_KIND_ROLE) if item is not None else ""
        if kind == "playlist":
            rect = self.visualItemRect(item)
            if pos.y() < rect.center().y():
                return True, str(item.data(0, _SOURCE_VALUE_ROLE)), rect.top()
            parent = item.parent()
            next_index = parent.indexOfChild(item) + 1 if parent is not None else -1
            next_item = (
                parent.child(next_index)
                if parent is not None and next_index < parent.childCount()
                else None
            )
            before_uid = (
                str(next_item.data(0, _SOURCE_VALUE_ROLE))
                if next_item is not None else None
            )
            return True, before_uid, rect.bottom() + 1
        if kind == "playlists":
            if item.childCount():
                y = self.visualItemRect(item.child(item.childCount() - 1)).bottom() + 1
            else:
                y = self.visualItemRect(item).bottom() + 1
            return True, None, y
        return False, None, None

    def _set_drop_visual(
        self,
        *,
        line_y: Optional[int] = None,
        target_rect: Optional[QtCore.QRect] = None,
    ) -> None:
        if self._drop_line_y != line_y or self._drop_target_rect != target_rect:
            self._drop_line_y = line_y
            self._drop_target_rect = target_rect
            self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_line_y is None and self._drop_target_rect is None:
            return
        painter = QtGui.QPainter(self.viewport())
        color = self.palette().highlight().color()
        if self._drop_target_rect is not None:
            painter.setPen(QtGui.QPen(color, 2))
            fill = QtGui.QColor(color)
            fill.setAlpha(35)
            painter.fillRect(self._drop_target_rect, fill)
            painter.drawRect(self._drop_target_rect.adjusted(1, 1, -2, -2))
        if self._drop_line_y is not None:
            pen = QtGui.QPen(color, 3)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(pen)
            y = max(1, min(self.viewport().height() - 2, self._drop_line_y))
            painter.drawLine(8, y, self.viewport().width() - 5, y)

    def _playlist_at(self, pos) -> Optional[str]:
        item = self.itemAt(pos)
        if item is None or item.data(0, _SOURCE_KIND_ROLE) != "playlist":
            return None
        return str(item.data(0, _SOURCE_VALUE_ROLE) or "") or None

    def dragEnterEvent(self, event):
        if event.source() is self and event.mimeData().hasFormat(_PLAYLIST_MIME):
            event.setDropAction(QtCore.Qt.MoveAction)
            event.accept()
        elif event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.source() is self and event.mimeData().hasFormat(_PLAYLIST_MIME):
            valid, _before_uid, line_y = self._playlist_insertion_at(
                event.position().toPoint()
            )
            if valid:
                self._set_drop_visual(line_y=line_y)
                event.setDropAction(QtCore.Qt.MoveAction)
                event.accept()
            else:
                self._set_drop_visual()
                event.ignore()
        elif event.mimeData().hasUrls() and self._playlist_at(event.position().toPoint()) is not None:
            item = self.itemAt(event.position().toPoint())
            self._set_drop_visual(target_rect=self.visualItemRect(item))
            event.acceptProposedAction()
        else:
            self._set_drop_visual()
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_drop_visual()
        event.accept()

    def dropEvent(self, event):
        if event.source() is self and event.mimeData().hasFormat(_PLAYLIST_MIME):
            try:
                moved_uid = bytes(event.mimeData().data(_PLAYLIST_MIME)).decode("ascii")
            except UnicodeDecodeError:
                event.ignore()
                return
            valid, before_uid, _line_y = self._playlist_insertion_at(
                event.position().toPoint()
            )
            if not valid:
                self._set_drop_visual()
                event.ignore()
                return
            self._set_drop_visual()
            self.playlistOrderDropped.emit(moved_uid, before_uid)
            event.setDropAction(QtCore.Qt.MoveAction)
            event.accept()
            return
        playlist_uid = self._playlist_at(event.position().toPoint())
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if playlist_uid is None or not paths:
            self._set_drop_visual()
            event.ignore()
            return
        self._set_drop_visual()
        self.playlistFilesDropped.emit(playlist_uid, paths)
        event.acceptProposedAction()


class _FillRowsLayout(QtWidgets.QLayout):
    """Lays its widgets out in rows that always span the full width.

    Items go left to right and wrap when the next one no longer fits; each row's
    spare width is shared out in proportion to the items' preferred widths, so
    the filter cards follow the window width instead of leaving a gap.
    """

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _rows(self, items: list, width: int) -> list[list]:
        """Split ``items`` (in order) into the fewest rows that fit ``width``,
        choosing among those the split whose rows are the most even, so a small
        card never ends up alone on a row while another row is crowded."""
        gap = self.spacing()
        hints = [item.sizeHint().width() for item in items]
        n = len(hints)
        best = None
        for mask in range(1 << max(0, n - 1)):  # bit i: break after item i
            rows, start = [], 0
            for i in range(n):
                if i == n - 1 or mask >> i & 1:
                    rows.append((start, i + 1))
                    start = i + 1
            natural = [sum(hints[a:b]) + gap * (b - a - 1) for a, b in rows]
            if any(w > width and b - a > 1 for w, (a, b) in zip(natural, rows)):
                continue  # only a single oversized card may overflow its row
            score = (len(rows), max(natural) - min(natural), max(natural))
            if best is None or score < best[0]:
                best = (score, rows)
        return [items[a:b] for a, b in best[1]] if best else [items]

    def _widths(self, row: list, width: int) -> list[float]:
        """Grow each card from its preferred width in proportion to it, up to its
        maximum (its inputs stop growing at a sensible size); a row that cannot
        use all of ``width`` keeps the rest at its right end."""
        gap = self.spacing()
        widths = [float(min(item.sizeHint().width(), width)) for item in row]
        limits = [float(max(w, min(self._max_width(item), width))) for item, w in zip(row, widths)]
        spare = width - sum(widths) - gap * (len(row) - 1)
        while spare > 0.5:
            open_ = [i for i in range(len(row)) if widths[i] < limits[i] - 0.5]
            if not open_:
                break
            total = sum(row[i].sizeHint().width() for i in open_) or 1
            used = 0.0
            for i in open_:
                add = min(spare * row[i].sizeHint().width() / total, limits[i] - widths[i])
                widths[i] += add
                used += add
            spare -= used
            if used < 0.5:
                break
        return widths

    @staticmethod
    def _max_width(item: QtWidgets.QLayoutItem) -> int:
        # A growable widget reports no maximum of its own; its layout knows how
        # far the inputs inside can grow before the card would only gain gaps.
        widget = item.widget()
        if widget is not None and widget.layout() is not None:
            return min(widget.maximumWidth(), widget.layout().maximumSize().width())
        return item.maximumSize().width()

    def _arrange(self, rect: QtCore.QRect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        gap = self.spacing()
        items = [item for item in self._items if not item.isEmpty()]
        if not items:
            return 0
        y = area.y()
        for row in self._rows(items, area.width()):
            height = max(item.sizeHint().height() for item in row)
            if apply:
                x = float(area.x())
                for item, w in zip(row, self._widths(row, area.width())):
                    left = round(x)
                    item.setGeometry(QtCore.QRect(left, y, round(x + w) - left, height))
                    x += w + gap
            y += height + gap
        return y - gap - area.y() + m.top() + m.bottom()


class _MatchToggle(QtWidgets.QAbstractButton):
    """AND | OR as one switch: both words always show, the active half is lit,
    and a click anywhere on it flips the mode (checked = OR)."""

    LABELS = ("AND", "OR")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.TabFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        font = QtGui.QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        self.setFont(font)
        self.setAttribute(QtCore.Qt.WA_Hover)
        self.toggled.connect(self._update_tooltip)
        self._update_tooltip()

    def match_all(self) -> bool:
        return not self.isChecked()

    def _update_tooltip(self) -> None:
        self.setToolTip(
            ("AND: require every search condition" if self.match_all()
             else "OR: accept any search condition") + " (click to switch)")

    def sizeHint(self) -> QtCore.QSize:
        fm = QtGui.QFontMetrics(self.font())
        half = max(fm.horizontalAdvance(t) for t in self.LABELS) + 20
        return QtCore.QSize(half * 2, fm.height() + 10)

    def paintEvent(self, _event) -> None:
        theme = current_theme()
        enabled = self.isEnabled()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        frame = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        border = theme.separator if not enabled else (
            theme.border_strong if self.underMouse() or self.hasFocus() else theme.border)
        p.setPen(QtGui.QPen(QtGui.QColor(border), 1))
        p.setBrush(QtGui.QColor(theme.raised if enabled else theme.window))
        p.drawRoundedRect(frame, theme.radius, theme.radius)

        half = frame.width() / 2
        active = 1 if self.isChecked() else 0
        lit = QtCore.QRectF(frame.left() + active * half, frame.top(), half, frame.height())
        p.setPen(QtGui.QPen(QtGui.QColor(theme.accent if enabled else theme.separator), 1))
        p.setBrush(QtGui.QColor(theme.accent_pressed if enabled else theme.raised))
        p.drawRoundedRect(lit, theme.radius, theme.radius)

        p.setFont(self.font())
        for i, text in enumerate(self.LABELS):
            if not enabled:
                color = theme.text_disabled
            elif i == active:
                color = theme.on_accent
            else:
                color = theme.text_dim
            p.setPen(QtGui.QColor(color))
            cell = QtCore.QRectF(frame.left() + i * half, frame.top(), half, frame.height())
            p.drawText(cell, QtCore.Qt.AlignCenter, text)
        p.end()


class LibraryWidget(QtWidgets.QWidget):
    ART_HEADER = ""  # the album art column (the top of the art)
    ART_WIDTH = 56
    HEADERS = [ART_HEADER, "#", "Title", "Artist", "Album", "BPM", "Key", "Duration", "Rating", "Added", "Path", "Comment"]
    SEARCH_MODES = ["All"] + HEADERS[1:]
    AUDIO_EXTENSIONS = frozenset({
        ".aac", ".aif", ".aiff", ".alac", ".flac", ".m4a", ".mp3",
        ".mp4", ".ogg", ".opus", ".wav", ".wma",
    })

    def _chip(self, text: str, tooltip: str, *, small: bool = False) -> QtWidgets.QToolButton:
        """A checkable toggle chip: the filter toggles."""
        font = QtGui.QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        button = QtWidgets.QToolButton()
        button.setText(text)
        button.setFont(font)
        button.setCheckable(True)
        button.setFocusPolicy(QtCore.Qt.TabFocus)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        button.setMinimumWidth(QtGui.QFontMetrics(font).horizontalAdvance(text) + (12 if small else 20))
        button.setStyleSheet("QToolButton { padding: 0 %dpx; }" % (4 if small else 8))
        button.setToolTip(tooltip)
        return button

    def _card(self, title: str, widgets) -> QtWidgets.QFrame:
        """One active filter's editors in a framed group with a small caption."""
        theme = current_theme()
        card = QtWidgets.QFrame()
        card.setObjectName("searchCard")
        card.setStyleSheet(
            f"QFrame#searchCard {{ background: {theme.raised}; border: 1px solid {theme.border};"
            f" border-radius: {theme.radius}px; }}")
        card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        row = QtWidgets.QHBoxLayout(card)
        row.setContentsMargins(8, 3, 4, 3)
        row.setSpacing(4)
        caption = QtWidgets.QLabel(title.upper())
        font = QtGui.QFont(caption.font())
        font.setPointSize(8)
        font.setBold(True)
        caption.setFont(font)
        caption.setStyleSheet(f"color: {theme.text_dim}; background: transparent;")
        caption.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        row.addWidget(caption)
        row.addSpacing(2)
        for widget in widgets:
            row.addWidget(widget)
        return card

    def __init__(self, bus: EventBus, cfg: config, model: DataModel | None = None, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.cfg = cfg
        self.model = model
        self._external_sync_enabled = bool(getattr(cfg, "externalsyncconfig", None) and cfg.externalsyncconfig.enabled)
        self._all_tracks: List[Track] = []
        self._library_tracks: List[Track] = []
        self._source_kind = "library"
        self._source_value = None
        # file identity -> the playlists (uids) it goes into once analyzed, once per drop
        self._pending_playlist_paths: dict[str, list[str]] = {}
        self._edit_dlg: EditSongDialog | None = None
        self._export_dlg: ExportTrackDialog | None = None

        self.source_tree = _LibrarySourceTree()
        self.source_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        # Search UI. Row 1: text search, one toggle chip per structured
        # filter and AND / OR. Row 2 holds only the enabled filters' editors,
        # so an idle search bar is a single line and a filter is one click.
        self.e_search = QtWidgets.QLineEdit(placeholderText="Search title, artist, key ...")
        self.e_search.setToolTip(
            "<p>Titles, artists and albums are also found by how they read, in any script: "
            "あのね by anone, ano ne or 아노네 (the reading is set in Edit Track).</p>")
        self.e_search.setClearButtonEnabled(True)
        self.cb_column = QtWidgets.QComboBox()
        self.cb_column.addItems(self.SEARCH_MODES)
        self.cb_column.setToolTip("Column the text search looks in")
        self.cb_column.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cb_fuzzy = self._chip(
            "Fuzzy", "<p>Also find a similar pronunciation (from six letters of the reading on: about one "
            "sound in five off). Off: the reading must match, up to its spelling.</p>")
        self.cb_fuzzy.setChecked(True)

        def grow(widget: QtWidgets.QWidget, width: int) -> None:
            """An input that grows with its card from ``width`` to a capped size,
            so spare row width spreads over the cards instead of one huge field."""
            widget.setMinimumWidth(width)
            widget.setMaximumWidth(round(width * 1.8))
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        def compact_spin(spin: QtWidgets.QDoubleSpinBox, lo: float, hi: float, value: float,
                         suffix: str = "") -> QtWidgets.QDoubleSpinBox:
            spin.setRange(lo, hi)
            spin.setDecimals(2)
            spin.setValue(value)
            spin.setSuffix(suffix)
            # At least as wide as the widest value it can show (the global font is large).
            grow(spin, spin.fontMetrics().horizontalAdvance(f"{hi:.2f}{suffix}") + 26)
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            spin.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return spin

        def compact_combo(combo: QtWidgets.QComboBox, width: int = 104) -> None:
            grow(combo, width)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)

        def dim_label(text: str) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet(f"color: {current_theme().text_dim}; background: transparent;")
            label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
            return label

        # Filter toggles (row 1)
        self.cb_phrase_filter = self._chip("Phrase", "Tracks with a phrase of this label (optionally at a BPM / key)")
        self.cb_transition_bpm = self._chip("BPM \u2192", "Tracks whose BPM changes from one tempo to another")
        self.cb_transition_key = self._chip("Key \u2192", "Tracks whose key changes from one key to another")
        self.match_toggle = _MatchToggle()
        self.btn_clear_filters = QtWidgets.QToolButton()
        self.btn_clear_filters.setText("Clear")
        self.btn_clear_filters.setToolTip("Remove every search condition (text and filters)")

        # Phrase filter: the label always applies; BPM and key refine it.
        self.cmb_phrase_label = QtWidgets.QComboBox()
        for label in PHRASE_LABELS:
            self.cmb_phrase_label.addItem(phrase_display_name(label), label)
        compact_combo(self.cmb_phrase_label, 112)
        self.cb_phrase_bpm = self._chip("BPM", "Also require this BPM inside the phrase", small=True)
        self.sp_phrase_bpm = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 400.0, 125.0)
        self.lbl_phrase_bpm_tol = dim_label("\u00b1")
        self.sp_phrase_bpm_tolerance = compact_spin(
            QtWidgets.QDoubleSpinBox(), 0.0, 100.0, 2.0, "%")
        self.sp_phrase_bpm_tolerance.setToolTip("Phrase BPM tolerance")
        self.cb_phrase_key = self._chip("Key", "Also require this key inside the phrase", small=True)
        self.cmb_phrase_key = KeyPickButton()
        self.cmb_phrase_key.setToolTip("Key inside the selected phrase (Camelot wheel)")
        self.cb_phrase_key_compatible = self._chip(
            "Compat", "Also match harmonically compatible keys (\u00b11 on the Camelot wheel, relative key)",
            small=True)

        # BPM transition
        self.sp_from_bpm = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 400.0, 125.0)
        self.lbl_bpm_arrow = dim_label("\u2192")
        self.sp_to_bpm = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 400.0, 129.0)
        self.lbl_bpm_tol = dim_label("\u00b1")
        self.sp_bpm_tolerance = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 100.0, 2.0, "%")
        self.sp_bpm_tolerance.setToolTip("BPM transition tolerance")
        self.cb_bpm_first_segment = self._chip(
            "1st", "Require the transition to start at the first BPM segment", small=True)

        # Key transition
        self.cmb_from_key = KeyPickButton()
        self.cmb_from_key.setToolTip("Key before the change (Camelot wheel)")
        self.lbl_arrow = dim_label("\u2192")
        self.cmb_to_key = KeyPickButton()
        self.cmb_to_key.setToolTip("Key after the change (Camelot wheel)")
        for picker in (self.cmb_phrase_key, self.cmb_from_key, self.cmb_to_key):
            grow(picker, 84)
            # as tall as the text inputs beside it
            picker.setFixedHeight(self.sp_from_bpm.sizeHint().height())
        self.cb_key_first_segment = self._chip(
            "1st", "Require the transition to start at the first key segment", small=True)
        self.cb_key_compatible = self._chip(
            "Compat", "Also match harmonically compatible keys at both ends "
            "(\u00b11 on the Camelot wheel, relative key)", small=True)
        self.cb_key_compatible.setChecked(True)

        # Shared by both transitions
        self.lbl_min_sec = dim_label("\u2265")
        self.sp_transition_min_duration = compact_spin(
            QtWidgets.QDoubleSpinBox(), 0.0, 600.0, 4.0, "s")
        self.sp_transition_min_duration.setToolTip(
            "Minimum duration of both sides of each enabled transition")

        self.card_phrase = self._card("Phrase", (
            self.cmb_phrase_label, self.cb_phrase_bpm, self.sp_phrase_bpm,
            self.lbl_phrase_bpm_tol, self.sp_phrase_bpm_tolerance,
            self.cb_phrase_key, self.cmb_phrase_key, self.cb_phrase_key_compatible))
        self.card_bpm = self._card("BPM", (
            self.sp_from_bpm, self.lbl_bpm_arrow, self.sp_to_bpm,
            self.lbl_bpm_tol, self.sp_bpm_tolerance, self.cb_bpm_first_segment))
        self.card_key = self._card("Key", (
            self.cmb_from_key, self.lbl_arrow, self.cmb_to_key,
            self.cb_key_compatible, self.cb_key_first_segment))
        self.card_min_sec = self._card("Min seg", (self.lbl_min_sec, self.sp_transition_min_duration))
        self.card_min_sec.setToolTip("Minimum duration of both sides of each enabled transition")

        # Table
        self.table = _TrackTable(
            self.HEADERS.index("Path"), self.HEADERS.index("#")
        )
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setDefaultSectionSize(140)
        # compact rows without grid lines; the header stays plain when a row is selected
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.table.verticalHeader().setDefaultSectionSize(self.table.fontMetrics().height() + 8)
        self.table.setItemDelegate(FastDelegate(
            self.table, art_column=self.HEADERS.index(self.ART_HEADER), libpath=self.cfg.libconfig.libpath))
        art_column = self.HEADERS.index(self.ART_HEADER)
        self.table.horizontalHeader().setSectionResizeMode(art_column, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(art_column, self.ART_WIDTH)
        self.table.doubleClicked.connect(self._on_activate)

        # Layout
        def separator() -> QtWidgets.QFrame:
            line = QtWidgets.QFrame()
            line.setFixedWidth(1)
            line.setStyleSheet(f"background: {current_theme().separator};")
            return line

        row_h = self.e_search.sizeHint().height()
        for widget in (self.cb_column, self.cb_fuzzy, self.cb_phrase_filter, self.cb_transition_bpm,
                       self.cb_transition_key, self.match_toggle,
                       self.btn_clear_filters):
            widget.setFixedHeight(row_h)

        primary = QtWidgets.QHBoxLayout()
        primary.setContentsMargins(0, 0, 0, 0)
        primary.setSpacing(4)
        self.e_search.setMinimumWidth(220)
        primary.addWidget(self.e_search, 1)
        primary.addWidget(self.cb_column)
        primary.addWidget(self.cb_fuzzy)
        primary.addSpacing(4)
        primary.addWidget(separator())
        primary.addSpacing(4)
        for chip in (self.cb_phrase_filter, self.cb_transition_bpm, self.cb_transition_key):
            primary.addWidget(chip)
        primary.addSpacing(4)
        primary.addWidget(separator())
        primary.addSpacing(4)
        primary.addWidget(self.match_toggle)
        primary.addWidget(self.btn_clear_filters)

        self.filter_bar = QtWidgets.QWidget()
        filters = _FillRowsLayout(self.filter_bar)
        for card in (self.card_phrase, self.card_bpm, self.card_key, self.card_min_sec):
            filters.addWidget(card)

        content = QtWidgets.QWidget()
        content_lay = QtWidgets.QVBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(6)
        content_lay.addLayout(primary)
        content_lay.addWidget(self.filter_bar)
        content_lay.addWidget(self.table, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.source_tree)
        splitter.addWidget(content)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 1200])

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)

        # Signals: a filter change is applied once they pause (SEARCH_DEBOUNCE_MS)
        self._search_timer = QtCore.QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(SEARCH_DEBOUNCE_MS)
        self._search_timer.timeout.connect(self._apply_filter)
        self.e_search.textChanged.connect(self._on_search_changed)
        self.e_search.textChanged.connect(self._sync_filter_ui)
        self.cb_column.currentIndexChanged.connect(self._on_search_changed)
        self.cb_fuzzy.toggled.connect(self._on_search_changed)
        self.cb_phrase_filter.toggled.connect(self._sync_filter_ui)
        self.cb_phrase_filter.toggled.connect(self._on_search_changed)
        self.cmb_phrase_label.currentIndexChanged.connect(self._on_search_changed)
        self.cb_phrase_bpm.toggled.connect(self._sync_filter_ui)
        self.cb_phrase_bpm.toggled.connect(self._on_search_changed)
        self.sp_phrase_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_phrase_bpm_tolerance.valueChanged.connect(self._on_search_changed)
        self.cb_phrase_key.toggled.connect(self._sync_filter_ui)
        self.cb_phrase_key.toggled.connect(self._on_search_changed)
        self.cmb_phrase_key.keyChanged.connect(self._on_search_changed)
        self.cb_phrase_key_compatible.toggled.connect(self._on_search_changed)
        self.cb_key_compatible.toggled.connect(self._on_search_changed)
        self.cb_transition_bpm.toggled.connect(self._sync_filter_ui)
        self.cb_transition_bpm.toggled.connect(self._on_search_changed)
        self.cb_transition_key.toggled.connect(self._sync_filter_ui)
        self.cb_transition_key.toggled.connect(self._on_search_changed)
        self.sp_from_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_to_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_bpm_tolerance.valueChanged.connect(self._on_search_changed)
        self.cb_bpm_first_segment.toggled.connect(self._on_search_changed)
        self.cmb_from_key.keyChanged.connect(self._on_search_changed)
        self.cmb_to_key.keyChanged.connect(self._on_search_changed)
        self.sp_transition_min_duration.valueChanged.connect(self._on_search_changed)
        self.cb_key_first_segment.toggled.connect(self._on_search_changed)
        self.match_toggle.toggled.connect(lambda _on: self._on_search_changed())
        self.btn_clear_filters.clicked.connect(self._clear_filters)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.source_tree.currentItemChanged.connect(self._on_source_changed)
        self.source_tree.itemExpanded.connect(self._on_source_expanded)
        self.source_tree.customContextMenuRequested.connect(self._on_source_context_menu)
        self.source_tree.playlistFilesDropped.connect(self._add_paths_to_playlist)
        self.source_tree.playlistOrderDropped.connect(self._drop_playlist_order)
        self.table.playlistRowsDropped.connect(self._drop_playlist_tracks)

        # The loaded track is shown in green.
        self._loaded_uid: Optional[str] = None
        self._marked_uid: Optional[str] = None   # the uid whose rows are green
        self.bus.sig_properties_loaded.connect(self._on_properties_loaded)

        # the table's rows are this source's tracks (a library update changes them in place)
        self._table_source: Optional[tuple] = None
        self._pending_library_rows = None
        self._library_timer = QtCore.QTimer(self)
        self._library_timer.setSingleShot(True)
        self._library_timer.setInterval(LIBRARY_UPDATE_DELAY_MS)
        self._library_timer.timeout.connect(self._apply_library_update)

        # Initial load
        self.bus.sig_lib_updated.connect(self._on_library_updated)
        self.bus.sig_external_sync_enabled.connect(self._on_external_sync_enabled)
        self._rebuild_source_tree()
        self.reload_from_db()
        self._sync_filter_ui()

    # Data loading
    def reload_from_db(self):
        l = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        l.connect()
        lib_full = l.list_all()
        l.close()
        self.reload(lib_full)

    def reload(self, data):
        """Refresh analyzed library data, then redraw the selected source."""
        delegate = self.table.itemDelegate()
        if isinstance(delegate, FastDelegate):
            delegate.clear_cache()  # thumbnails may have been replaced
        self._library_tracks = self._tracks(data)
        self._refresh_current_source()
        self.table.viewport().update()

    def _tracks(self, rows) -> List[Track]:
        """Library rows as Tracks, with their readings for the search."""
        db = self._db()
        try:
            readings = db.all_readings()
        finally:
            db.close()
        return [row_to_track(row, readings.get(row.uid)) if isinstance(row, TrackRow) else row for row in rows]

    @staticmethod
    def _file_identity(path: object) -> str:
        text = os.path.expanduser(os.path.expandvars(str(path or "")))
        return os.path.normcase(os.path.realpath(os.path.abspath(text)))

    def _db(self) -> LibraryDB:
        db = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        db.connect()
        return db

    def _local_music_roots(self) -> list[str]:
        configured = [
            str(path).strip()
            for path in self.cfg.libconfig.local_music_directories
            if str(path).strip()
        ]
        if not configured:
            os_music = QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.MusicLocation
            )
            configured = [os_music or os.path.join(os.path.expanduser("~"), "Music")]
        roots = []
        identities = set()
        for configured_path in configured:
            root_path = os.path.abspath(
                os.path.expandvars(os.path.expanduser(configured_path))
            )
            identity = self._file_identity(root_path)
            if identity in identities:
                continue
            identities.add(identity)
            roots.append(root_path)
        return roots

    def _rebuild_source_tree(self) -> None:
        previous = (self._source_kind, self._source_value)
        tree = self.source_tree
        tree.blockSignals(True)
        tree.clear()

        library_item = QtWidgets.QTreeWidgetItem(["Library (All)"])
        library_item.setData(0, _SOURCE_KIND_ROLE, "library")
        tree.addTopLevelItem(library_item)

        playlists_item = QtWidgets.QTreeWidgetItem(["Playlists"])
        playlists_item.setData(0, _SOURCE_KIND_ROLE, "playlists")
        tree.addTopLevelItem(playlists_item)
        db = self._db()
        try:
            playlists = db.list_playlists()
        finally:
            db.close()
        for playlist in playlists:
            item = QtWidgets.QTreeWidgetItem([playlist.name])
            item.setData(0, _SOURCE_KIND_ROLE, "playlist")
            item.setData(0, _SOURCE_VALUE_ROLE, playlist.uid)
            playlists_item.addChild(item)
        playlists_item.setExpanded(True)

        local_item = QtWidgets.QTreeWidgetItem(["Local Folders"])
        local_item.setData(0, _SOURCE_KIND_ROLE, "local_folders")
        tree.addTopLevelItem(local_item)
        for root_path in self._local_music_roots():
            root_label = Path(root_path).name or root_path
            root_item = QtWidgets.QTreeWidgetItem([root_label])
            root_item.setToolTip(0, root_path)
            root_item.setData(0, _SOURCE_KIND_ROLE, "directory")
            root_item.setData(0, _SOURCE_VALUE_ROLE, root_path)
            local_item.addChild(root_item)
            self._populate_directory_children(root_item)
            root_item.setExpanded(False)
        local_item.setExpanded(True)

        selected = library_item
        iterator = QtWidgets.QTreeWidgetItemIterator(tree)
        while iterator.value():
            candidate = iterator.value()
            if (
                candidate.data(0, _SOURCE_KIND_ROLE),
                candidate.data(0, _SOURCE_VALUE_ROLE),
            ) == previous:
                selected = candidate
                break
            iterator += 1
        tree.blockSignals(False)
        tree.setCurrentItem(selected)
        self._source_kind = str(selected.data(0, _SOURCE_KIND_ROLE) or "library")
        self._source_value = selected.data(0, _SOURCE_VALUE_ROLE)

    def _populate_directory_children(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if item.data(0, _SOURCE_KIND_ROLE) != "directory":
            return
        if item.data(0, _DIRECTORY_LOADED_ROLE):
            return
        item.takeChildren()
        path = str(item.data(0, _SOURCE_VALUE_ROLE) or "")
        directories = []
        try:
            with os.scandir(path) as entries:
                directories = sorted(
                    (entry for entry in entries if entry.is_dir(follow_symlinks=False)),
                    key=lambda entry: entry.name.casefold(),
                )
        except OSError:
            directories = []
        for entry in directories:
            child = QtWidgets.QTreeWidgetItem([entry.name])
            child.setToolTip(0, entry.path)
            child.setData(0, _SOURCE_KIND_ROLE, "directory")
            child.setData(0, _SOURCE_VALUE_ROLE, entry.path)
            child.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ShowIndicator)
            item.addChild(child)
        item.setData(0, _DIRECTORY_LOADED_ROLE, True)

    def _on_source_expanded(self, item: QtWidgets.QTreeWidgetItem) -> None:
        self._populate_directory_children(item)

    def _on_source_changed(self, current, _previous=None) -> None:
        if current is None:
            return
        self._source_kind = str(current.data(0, _SOURCE_KIND_ROLE) or "")
        self._source_value = current.data(0, _SOURCE_VALUE_ROLE)
        self._refresh_current_source(reset_sort=True)

    def _refresh_current_source(self, *, reset_sort: bool = False) -> None:
        self.table.setPlaylistReorderEnabled(self._source_kind == "playlist")
        if self._source_kind == "library":
            tracks = list(self._library_tracks)
            default_column = "Added"
            default_order = QtCore.Qt.SortOrder.DescendingOrder
        elif self._source_kind == "playlist":
            db = self._db()
            try:
                rows = db.list_playlist_tracks(str(self._source_value))
                readings = db.all_readings()
            finally:
                db.close()
            tracks = []
            for position, row in rows:
                track = row_to_track(row, readings.get(row.uid))
                track.playlist_index = position
                track._disp_playlist_index = str(position + 1)
                tracks.append(track)
            default_column = "#"
            default_order = QtCore.Qt.SortOrder.AscendingOrder
        elif self._source_kind == "directory":
            tracks = self._scan_directory(str(self._source_value or ""))
            default_column = "Title"
            default_order = QtCore.Qt.SortOrder.AscendingOrder
        else:
            tracks = []
            default_column = "Title"
            default_order = QtCore.Qt.SortOrder.AscendingOrder

        source = (self._source_kind, self._source_value)
        if (self._source_kind == "library" and not reset_sort and self._table_source == source
                and self._update_library_rows(tracks)):
            self._apply_filter()
            return
        self.table.setUpdatesEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(0)
        self._all_tracks = tracks
        if reset_sort:
            self.table.horizontalHeader().setSortIndicator(
                self.HEADERS.index(default_column), default_order
            )
        self._populate_table(tracks)   # sorted by the sort indicator
        self._table_source = source
        self._apply_filter()
        self.table.setUpdatesEnabled(True)

    def _update_library_rows(self, tracks: List[Track]) -> bool:
        """The library shown, a few of its tracks added, changed or removed: their rows changed in
        place (each moved to its sorted place as it is set); False (nothing done) for more than
        INCREMENTAL_ROWS_MAX of them."""
        old = {t.uid: t for t in self._all_tracks}
        new = {t.uid: t for t in tracks}
        if "" in old or "" in new or len(old) != len(self._all_tracks) or len(new) != len(tracks):
            return False
        removed = old.keys() - new.keys()
        changed = [t for t in tracks if t.uid not in old or t != old[t.uid]]
        if len(removed) + len(changed) > INCREMENTAL_ROWS_MAX:
            return False
        if not removed and not changed:
            self._all_tracks = tracks
            return True
        # a row's art item moves with it (sorting): its row() is where the track is now
        art = self.HEADERS.index(self.ART_HEADER)
        items = {}
        for r in range(self.table.rowCount()):
            item = self.table.item(r, art)
            if item is not None:
                items[str(item.data(QtCore.Qt.UserRole) or "")] = item
        if any(uid not in items for uid in removed) or any(t.uid in old and t.uid not in items for t in changed):
            return False
        self._all_tracks = tracks
        for r in sorted((items[uid].row() for uid in removed), reverse=True):
            self.table.removeRow(r)
        for track in changed:
            if track.uid in old:
                self._fill_row(items[track.uid].row(), track)
            else:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self._fill_row(r, track)
        return True

    def _scan_directory(self, directory: str) -> List[Track]:
        analyzed = {self._file_identity(track.path): track for track in self._library_tracks}
        paths = []
        try:
            with os.scandir(directory) as entries:
                paths = sorted(
                    (
                        entry.path for entry in entries
                        if entry.is_file(follow_symlinks=False)
                        and Path(entry.name).suffix.lower() in self.AUDIO_EXTENSIONS
                    ),
                    key=lambda path: Path(path).name.casefold(),
                )
        except OSError:
            return []
        result = []
        for path in paths:
            known = analyzed.get(self._file_identity(path))
            if known is not None:
                result.append(known)
            else:
                result.append(Track(
                    track_id=path,
                    title=Path(path).stem,
                    artist="",
                    path=path,
                    analyzed=False,
                    reading_keys={"title": [reading.keys(r) for r, _src in reading.readings(
                        Path(path).stem, self.cfg.searchconfig.db_languages,
                        han_run=self.cfg.searchconfig.han_run())]},
                ))
        return result

    def _on_library_updated(self, data) -> None:
        """Library updates in a row are applied once, the last (LIBRARY_UPDATE_DELAY_MS)."""
        self._pending_library_rows = list(data or [])
        self._library_timer.start()

    def _apply_library_update(self) -> None:
        data, self._pending_library_rows = self._pending_library_rows, None
        if data is None:
            return
        self._library_tracks = self._tracks(data)
        resolved: dict[str, list[str]] = {}
        for track in self._library_tracks:
            playlist_uids = self._pending_playlist_paths.pop(
                self._file_identity(track.path), []
            )
            for playlist_uid in playlist_uids:
                resolved.setdefault(playlist_uid, []).append(track.uid)
        if resolved:
            db = self._db()
            try:
                for playlist_uid, uids in resolved.items():
                    db.add_playlist_tracks(playlist_uid, uids)
            finally:
                db.close()
        self._refresh_current_source()

    def set_config(self, cfg: config) -> None:
        old_library = self.cfg.libconfig.libpath
        old_local = list(self.cfg.libconfig.local_music_directories)
        self.cfg = cfg
        delegate = self.table.itemDelegate()
        if isinstance(delegate, FastDelegate):
            delegate._libpath = cfg.libconfig.libpath
            delegate.clear_cache()
        if old_local != list(cfg.libconfig.local_music_directories):
            self._source_kind, self._source_value = "library", None
        if old_library != cfg.libconfig.libpath:
            self._pending_playlist_paths.clear()
        self._rebuild_source_tree()
        self.reload_from_db()

    def set_search_config(self, cfg: config) -> None:
        """The search settings changed: the query is read again (stored readings change with a re-read)."""
        self.cfg = cfg
        self._apply_filter()

    def _populate_table(self, tracks: List[Track]):
        header = self.table.horizontalHeader()
        sorting_enabled = self.table.isSortingEnabled()
        sort_section = header.sortIndicatorSection() if header else -1
        sort_order = header.sortIndicatorOrder() if header else QtCore.Qt.SortOrder.AscendingOrder
        if sort_section < 0:
            sort_section = self.HEADERS.index("Added")
            sort_order = QtCore.Qt.SortOrder.DescendingOrder
        if sorting_enabled:
            self.table.setSortingEnabled(False)

        self.table.setRowCount(len(tracks))
        for r, t in enumerate(tracks):
            self._fill_row(r, t)
        if sorting_enabled:
            self.table.setSortingEnabled(True)
            self.table.sortItems(sort_section, sort_order)
        self._marked_uid = self._loaded_uid

    def _fill_row(self, r: int, t: Track) -> None:
        """Row ``r`` made for track ``t``, coloured (green loaded, dim not analyzed). The sort
        column is set last: with sorting on, the row then moves to its place once."""
        values = [
            "", t._disp_playlist_index,
            t.title, t.artist, t.album, t._disp_bpm, t._disp_key_camelot,
            t._disp_duration, t._disp_rating, t._disp_added, t.path, t.comment
        ]
        bpm_col = self.HEADERS.index("BPM")
        art_col = self.HEADERS.index(self.ART_HEADER)
        index_col = self.HEADERS.index("#")
        sort_col = self.table.horizontalHeader().sortIndicatorSection() if self.table.isSortingEnabled() else -1
        brush = None
        if self._loaded_uid is not None and t.uid == self._loaded_uid:
            brush = QtGui.QBrush(QtGui.QColor(current_theme().success))
        elif not t.analyzed:
            brush = QtGui.QBrush(QtGui.QColor(current_theme().text_dim))
        for c in sorted(range(len(values)), key=lambda c: c == sort_col):
            item = _SortItem(str(values[c]))
            if c == bpm_col:  # for BPM sorting
                item.setData(_SORT_ROLE, float(t.bpm or -1))
            elif c == art_col:  # the thumbnail's track
                item.setData(QtCore.Qt.UserRole, t.uid)
            elif c == index_col:
                item.setData(_SORT_ROLE, int(t.playlist_index or 0))
            item.setData(_ANALYZED_ROLE, bool(t.analyzed))
            if brush is not None:
                item.setForeground(brush)
            self.table.setItem(r, c, item)

    def _row_uid(self, row: int) -> str:
        """The track uid of a table row ("" for a file not analyzed)."""
        item = self.table.item(row, self.HEADERS.index(self.ART_HEADER))
        return str(item.data(QtCore.Qt.UserRole) or "") if item is not None else ""

    def _on_properties_loaded(self) -> None:
        properties = self.model.properties if self.model is not None else {}
        uid = properties.get("uid") if isinstance(properties, dict) else None
        self._loaded_uid = str(uid or "").strip() or None
        self._mark_loaded_track()

    def _mark_loaded_track(self) -> None:
        """Green text on the loaded track's rows, the default again on the ones green before (the
        rows are coloured when made)."""
        if self._marked_uid == self._loaded_uid:
            return
        green = QtGui.QBrush(QtGui.QColor(current_theme().success))
        dim = QtGui.QBrush(QtGui.QColor(current_theme().text_dim))
        art = self.HEADERS.index(self.ART_HEADER)
        uids = {uid for uid in (self._marked_uid, self._loaded_uid) if uid}
        for r in range(self.table.rowCount()):
            item = self.table.item(r, art)
            if item is None or item.data(QtCore.Qt.UserRole) not in uids:
                continue
            loaded = item.data(QtCore.Qt.UserRole) == self._loaded_uid
            for c in range(self.table.columnCount()):
                cell = self.table.item(r, c)
                if cell is None:
                    continue
                if loaded:
                    cell.setForeground(green)
                elif not bool(cell.data(_ANALYZED_ROLE)):
                    cell.setForeground(dim)
                else:
                    cell.setData(QtCore.Qt.ForegroundRole, None)
        self._marked_uid = self._loaded_uid

    # Filter
    def _on_search_changed(self, *_args):
        """A filter changed: applied once the changes pause (typing, a held spin box arrow)."""
        self._search_timer.start()

    def _apply_filter(self) -> None:
        """Hide the rows the search and the filters leave out (the rows stay made)."""
        self._search_timer.stop()
        text = self.e_search.text().strip().lower()
        col = self.cb_column.currentText()
        field_map = {
            "#": "_disp_playlist_index",
            "Title": "title",
            "Artist": "artist",
            "Album": "album",
            "Comment": "comment",
            "Key": "_disp_key_camelot",
            "BPM": "_disp_bpm",
            "Duration": "_disp_duration",
            "Rating": "_disp_rating",
            "Added": "_disp_added",
            "Path": "path",
        }
        conditions: list[set[str]] = []
        if text:
            text_paths: set[str] = set()
            cols = self.HEADERS if col == "All" else [col]
            read_fields = [READ_COLUMNS[name] for name in cols if name in READ_COLUMNS]
            s = self.cfg.searchconfig
            # as typed (not lower-cased): Shift picks the 2-set keyboard's double consonants
            query = reading.query_keys(self.e_search.text().strip(), s.query_languages,
                                       hangul_korean=s.query_hangul_korean, hangul_foreign=s.query_hangul_foreign,
                                       keyboard=s.query_keyboard_layout, han_run=s.han_run()) if read_fields else []
            for t in self._all_tracks:
                for name in cols:
                    attr = field_map.get(name)
                    if not attr:
                        continue
                    val = getattr(t, attr, "")
                    if val and text in str(val).lower():
                        text_paths.add(t.path)
                        break
            if read_fields:
                # by sound: the query read like the readings, normalized, exactly or nearly
                entries = [(t.path, k) for t in self._all_tracks for f in read_fields
                           for k in t.reading_keys.get(f, [])]
                text_paths.update(entries[i][0] for i in reading.find(
                    query, [k for _path, k in entries], similar=self.cb_fuzzy.isChecked()))
            conditions.append(text_paths)

        conditions.extend(self._search_structured_paths())
        if not conditions:
            allowed = None
        elif self.match_toggle.match_all():
            allowed = set.intersection(*conditions)
        else:
            allowed = set.union(*conditions)
        path_col = self.HEADERS.index("Path")
        for r in range(self.table.rowCount()):
            item = self.table.item(r, path_col)
            hidden = allowed is not None and (item is None or item.text() not in allowed)
            if self.table.isRowHidden(r) != hidden:
                self.table.setRowHidden(r, hidden)
        # a hidden row is not acted on: out of the selection
        selection = self.table.selectionModel()
        for index in selection.selectedRows():
            if self.table.isRowHidden(index.row()):
                selection.select(index, QtCore.QItemSelectionModel.Deselect | QtCore.QItemSelectionModel.Rows)

    def _sync_filter_ui(self):
        is_phrase = self.cb_phrase_filter.isChecked()
        is_bpm = self.cb_transition_bpm.isChecked()
        is_key = self.cb_transition_key.isChecked()
        self.card_phrase.setVisible(is_phrase)
        self.card_bpm.setVisible(is_bpm)
        self.card_key.setVisible(is_key)
        self.card_min_sec.setVisible(is_bpm or is_key)
        self.filter_bar.setVisible(is_phrase or is_bpm or is_key)

        phrase_bpm = self.cb_phrase_bpm.isChecked()
        for widget in (self.sp_phrase_bpm, self.lbl_phrase_bpm_tol, self.sp_phrase_bpm_tolerance):
            widget.setEnabled(phrase_bpm)
        phrase_key = self.cb_phrase_key.isChecked()
        self.cmb_phrase_key.setEnabled(phrase_key)
        self.cb_phrase_key_compatible.setEnabled(phrase_key)

        # AND / OR is offered as soon as a filter is on: the text search is the
        # other condition (typed already or about to be).
        has_filter = is_phrase or is_bpm or is_key
        self.match_toggle.setEnabled(has_filter)

    def _clear_filters(self) -> None:
        """Remove every condition: the text and all filter toggles (values stay)."""
        widgets = (
            self.e_search, self.cb_phrase_filter, self.cb_transition_bpm, self.cb_transition_key,
            self.cb_phrase_bpm, self.cb_phrase_key, self.cb_bpm_first_segment,
            self.cb_key_first_segment,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.e_search.clear()
        for chip in widgets[1:]:
            chip.setChecked(False)
        for widget in widgets:
            widget.blockSignals(False)
        self._sync_filter_ui()
        self._apply_filter()

    def _search_structured_paths(self) -> list[set[str]]:
        """Return one path set per enabled phrase/transition condition."""
        if not any((
            self.cb_phrase_filter.isChecked(),
            self.cb_transition_bpm.isChecked(),
            self.cb_transition_key.isChecked(),
        )):
            return []
        db = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            conditions: list[set[str]] = []
            match_all = self.match_toggle.match_all()
            if self.cb_phrase_filter.isChecked():
                bpm_min = bpm_max = None
                if self.cb_phrase_bpm.isChecked():
                    center = float(self.sp_phrase_bpm.value())
                    scale = max(0.0, float(self.sp_phrase_bpm_tolerance.value())) / 100.0
                    # At 0%, still match values that round to the BPM shown by
                    # this two-decimal editor (analysis values retain more
                    # floating-point precision in the database).
                    delta = max(center * scale, 0.5 * 10 ** -self.sp_phrase_bpm.decimals())
                    bpm_min = center - delta
                    bpm_max = center + delta
                key_value = (
                    self.cmb_phrase_key.key()
                    if self.cb_phrase_key.isChecked()
                    else None
                )
                rows = db.search_phrase_sections(
                    str(self.cmb_phrase_label.currentData()),
                    bpm_min=bpm_min,
                    bpm_max=bpm_max,
                    key_value=key_value,
                    compatible_keys=self.cb_phrase_key_compatible.isChecked(),
                    match_all=match_all,
                )
                conditions.append({str(row.path) for row in rows})

            min_duration = float(self.sp_transition_min_duration.value())
            if self.cb_transition_bpm.isChecked():
                center_from = float(self.sp_from_bpm.value())
                center_to = float(self.sp_to_bpm.value())
                tol_pct = max(0.0, float(self.sp_bpm_tolerance.value()))
                scale = tol_pct / 100.0
                from_delta = max(
                    center_from * scale, 0.5 * 10 ** -self.sp_from_bpm.decimals())
                to_delta = max(
                    center_to * scale, 0.5 * 10 ** -self.sp_to_bpm.decimals())
                bpm_rows = db.search_bpm_transitions(
                    center_from - from_delta,
                    center_from + from_delta,
                    center_to - to_delta,
                    center_to + to_delta,
                    min_duration_sec=min_duration,
                    require_first_segment=self.cb_bpm_first_segment.isChecked(),
                )
                conditions.append({str(row.path) for row in bpm_rows})
            if self.cb_transition_key.isChecked():
                key_rows = db.search_harmonic_key_transitions(
                    self.cmb_from_key.key(),
                    self.cmb_to_key.key(),
                    min_duration_sec=min_duration,
                    require_first_segment=self.cb_key_first_segment.isChecked(),
                    compatible=self.cb_key_compatible.isChecked(),
                )
                conditions.append({str(row.path) for row in key_rows})
            return conditions
        finally:
            db.close()

    def _on_activate(self, idx: QtCore.QModelIndex):
        if self._external_sync_enabled:
            QtWidgets.QMessageBox.information(
                self,
                "External Sync",
                "Track loading is disabled while External Sync is enabled.",
            )
            return
        r = idx.row()
        if not (0 <= r < self.table.rowCount()):
            return
        path_col = self.HEADERS.index("Path")
        item = self.table.item(r, path_col)
        if not item:
            return
        path = item.text()
        if path:
            self.bus.sig_request_load_track.emit(path)

    def _on_external_sync_enabled(self, enabled: bool) -> None:
        self._external_sync_enabled = bool(enabled)

    def _select_source(self, kind: str, value=None) -> None:
        iterator = QtWidgets.QTreeWidgetItemIterator(self.source_tree)
        while iterator.value():
            item = iterator.value()
            if (
                item.data(0, _SOURCE_KIND_ROLE) == kind
                and item.data(0, _SOURCE_VALUE_ROLE) == value
            ):
                self.source_tree.setCurrentItem(item)
                return
            iterator += 1

    def _on_source_context_menu(self, pos) -> None:
        item = self.source_tree.itemAt(pos)
        kind = item.data(0, _SOURCE_KIND_ROLE) if item is not None else ""
        menu = QtWidgets.QMenu(self)
        actions = {}
        if kind in ("playlists", "playlist"):
            actions["new"] = menu.addAction("New Playlist")
        if kind == "playlist":
            menu.addSeparator()
            actions["rename"] = menu.addAction("Rename")
            actions["up"] = menu.addAction("Move Up")
            actions["down"] = menu.addAction("Move Down")
            actions["delete"] = menu.addAction("Delete Playlist")
        elif kind in ("directory", "local_folders"):
            actions["refresh"] = menu.addAction("Refresh Folders")
        if menu.isEmpty():
            return
        chosen = menu.exec_(self.source_tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == actions.get("new"):
            self._create_playlist()
        elif chosen == actions.get("rename"):
            self._rename_playlist(item)
        elif chosen == actions.get("delete"):
            self._delete_playlist(item)
        elif chosen == actions.get("up"):
            self._move_playlist(item, -1)
        elif chosen == actions.get("down"):
            self._move_playlist(item, 1)
        elif chosen == actions.get("refresh"):
            self._rebuild_source_tree()
            self._refresh_current_source()

    def _create_playlist(self) -> None:
        name, accepted = QtWidgets.QInputDialog.getText(self, "New Playlist", "Name")
        if not accepted or not name.strip():
            return
        db = self._db()
        try:
            playlist = db.create_playlist(name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "New Playlist", str(exc))
            return
        finally:
            db.close()
        self._source_kind, self._source_value = "playlist", playlist.uid
        self._rebuild_source_tree()
        self._select_source("playlist", playlist.uid)

    def _rename_playlist(self, item: QtWidgets.QTreeWidgetItem) -> None:
        playlist_uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        name, accepted = QtWidgets.QInputDialog.getText(
            self, "Rename Playlist", "Name", text=item.text(0)
        )
        if not accepted or not name.strip():
            return
        db = self._db()
        try:
            db.rename_playlist(playlist_uid, name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Playlist", str(exc))
            return
        finally:
            db.close()
        self._rebuild_source_tree()
        self._select_source("playlist", playlist_uid)

    def _delete_playlist(self, item: QtWidgets.QTreeWidgetItem) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete Playlist",
            f'Delete playlist "{item.text(0)}"?\nTracks remain in the Library.',
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        playlist_uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        db = self._db()
        try:
            db.delete_playlist(playlist_uid)
        finally:
            db.close()
        self._source_kind, self._source_value = "library", None
        self._rebuild_source_tree()
        self._select_source("library", None)

    def _move_playlist(self, item: QtWidgets.QTreeWidgetItem, offset: int) -> None:
        playlist_uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        db = self._db()
        try:
            db.move_playlist(playlist_uid, offset)
        finally:
            db.close()
        self._rebuild_source_tree()
        self._select_source("playlist", playlist_uid)

    def _drop_playlist_order(self, playlist_uid: str, before_playlist_uid) -> None:
        db = self._db()
        try:
            changed = db.move_playlist_before(playlist_uid, before_playlist_uid)
        finally:
            db.close()
        if changed:
            QtCore.QTimer.singleShot(
                0, lambda playlist_uid=playlist_uid: self._finish_playlist_drop(playlist_uid)
            )

    def _finish_playlist_drop(self, playlist_uid: str) -> None:
        self._rebuild_source_tree()
        self._select_source("playlist", playlist_uid)

    def _drop_playlist_tracks(self, positions, before_position) -> None:
        if self._source_kind != "playlist":
            return
        db = self._db()
        try:
            changed = db.move_playlist_tracks_before(
                str(self._source_value), positions, before_position
            )
        finally:
            db.close()
        if changed:
            QtCore.QTimer.singleShot(
                0, lambda: self._refresh_current_source(reset_sort=True)
            )

    def _selected_rows(self) -> list[int]:
        """The rows selected and shown (Ctrl+A selects hidden rows too: never acted on)."""
        return sorted(index.row() for index in self.table.selectionModel().selectedRows()
                      if not self.table.isRowHidden(index.row()))

    def _selected_paths(self) -> list[str]:
        path_column = self.HEADERS.index("Path")
        result = []
        for row in self._selected_rows():
            item = self.table.item(row, path_column)
            if item is not None and item.text():
                result.append(item.text())
        return result

    def _selected_uids(self) -> list[str]:
        """The analyzed tracks selected, each once."""
        return list(dict.fromkeys(uid for uid in map(self._row_uid, self._selected_rows()) if uid))

    def _add_paths_to_playlist(self, playlist_uid: str, paths) -> None:
        requested = [str(path) for path in paths if str(path)]
        if not requested:
            return
        analyzed_uids = []
        pending_paths = []
        db = self._db()
        try:
            for path in requested:
                row = db.find_file(path)
                if row is not None:
                    analyzed_uids.append(row.uid)
                elif os.path.isfile(path):
                    pending_paths.append(path)
            if analyzed_uids:
                db.add_playlist_tracks(playlist_uid, analyzed_uids)
        finally:
            db.close()
        for path in pending_paths:
            self._pending_playlist_paths.setdefault(
                self._file_identity(path), []
            ).append(playlist_uid)
            self.bus.sig_request_load_track.emit(path)
        if self._source_kind == "playlist" and str(self._source_value) == playlist_uid:
            self._refresh_current_source()

    def _on_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        selected_paths = self._selected_paths()
        analyzed_by_path = {track.path: track.analyzed for track in self._all_tracks}
        all_analyzed = bool(selected_paths) and all(
            analyzed_by_path.get(path, False) for path in selected_paths
        )
        act_reveal = menu.addAction("Reveal in Explorer/Finder")
        act_remove = menu.addAction("Remove from Library")
        act_edit = menu.addAction("Edit Track")
        act_reanalyze = menu.addAction(
            "Reanalyze Track" if all_analyzed else "Analyze Track"
        )
        act_export = menu.addAction("Export Track")
        act_remove_playlist = None
        if self._source_kind == "playlist":
            act_remove_playlist = menu.addAction("Remove from Playlist")
        add_menu = menu.addMenu("Add to Playlist")
        playlist_actions = {}
        db = self._db()
        try:
            playlists = db.list_playlists()
        finally:
            db.close()
        for playlist in playlists:
            playlist_actions[add_menu.addAction(playlist.name)] = playlist.uid
        if not playlists:
            add_menu.setEnabled(False)
        act_remove.setEnabled(all_analyzed)
        act_edit.setEnabled(all_analyzed)
        act_export.setEnabled(all_analyzed)
        act = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if act == act_remove:
            self._on_remove_selected()
        elif act == act_reveal:
            self._reveal_selected()
        elif act == act_reanalyze:
            self._reanalyze_selected()
        elif act == act_edit:
            self._edit_selected()
        elif act == act_export:
            self._export_selected()
        elif act == act_remove_playlist:
            self._remove_selected_from_playlist()
        elif act in playlist_actions:
            self._add_paths_to_playlist(playlist_actions[act], selected_paths)

    def _remove_selected_from_playlist(self) -> None:
        if self._source_kind != "playlist":
            return
        positions = [self.table._entry_position(row) for row in self._selected_rows()]
        positions = [position for position in positions if position is not None]
        if not positions:
            return
        db = self._db()
        try:
            db.remove_playlist_tracks(str(self._source_value), positions)
        finally:
            db.close()
        self._refresh_current_source(reset_sort=True)

    def _on_remove_selected(self):
        uids = self._selected_uids()
        if not uids:
            return
        l = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        l.connect()
        l.delete_tracks(uids)
        lib_full = l.list_all()
        l.close()
        n = FeatureNPZStore(base_dir=self.cfg.libconfig.libpath, compressed=True)
        for uid in uids:
            n.delete(uid)
            delete_thumbnail(self.cfg.libconfig.libpath, uid)
        self.bus.sig_lib_updated.emit(lib_full)
        for uid in uids:
            track_uid = str(uid or "").strip()
            if track_uid:
                self.bus.sig_rekordbox_sync_track_requested.emit(track_uid)

    # Explorer
    def _reveal_selected(self):
        import subprocess, platform
        for row in self._selected_rows():
            path = self.table.item(row, self.HEADERS.index("Path")).text()
            if not path:
                continue
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                folder = os.path.dirname(path)
                subprocess.Popen(["xdg-open", folder])

    def _export_selected(self):
        uids = self._selected_uids()
        if not uids:
            return

        db = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        db.connect()
        track_row = db.get_by_uid(uids[0])
        db.close()
        if track_row is None:
            QtWidgets.QMessageBox.warning(self, "Export Track", "Track metadata could not be found in the library.")
            return

        if self._export_dlg is None:
            self._export_dlg = ExportTrackDialog(self.cfg.libconfig.libpath, self, model=self.model)
        self._export_dlg.set_track(track_row)
        self._export_dlg.show()
        self._export_dlg.raise_()
        self._export_dlg.activateWindow()

    # Reanalyze
    def _reanalyze_selected(self):
        analyzed_by_path = {track.path: track.analyzed for track in self._all_tracks}
        for row in self._selected_rows():
            path = self.table.item(row, self.HEADERS.index("Path")).text()
            if path:
                if analyzed_by_path.get(path, False):
                    self.bus.sig_reanalyze_requested.emit(path)
                else:
                    self.bus.sig_request_load_track.emit(path)

    # Edit
    def _edit_selected(self):
        uids = self._selected_uids()
        if not uids:
            return
        # First selected track
        l = LibraryDB(os.path.join(self.cfg.libconfig.libpath, "library.db"))
        l.connect()
        row = l.get_by_uid(uids[0])
        l.close()
        if not row:
            return
        if self._edit_dlg is None:
            self._edit_dlg = EditSongDialog(bus=self.bus, libpath=self.cfg.libconfig.libpath,
                                            search_settings=lambda: self.cfg.searchconfig)
        self._edit_dlg.set_track(row)
        self._edit_dlg.show()
