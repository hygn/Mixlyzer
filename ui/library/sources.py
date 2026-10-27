"""The sources panel's part: the library, Variants, playlists and folders listed, the one shown read."""
from __future__ import annotations

import copy
import html
from pathlib import Path
from typing import List, Optional
from PySide6 import QtCore, QtGui, QtWidgets
import os
from core.library.records import FOLDER_NODE, PLAYLIST_NODE
from core.search import reading
from ui.edit_panel.icons import folder_icon
from ui.theme import current_theme
from analyzer.search import variants as variant_fp
from core.search import variants as variant_relations
from system.i18n import tr
from ui.library.common import (Track, _DIRECTORY_LOADED_ROLE, _PLAYLIST_MIME, _SOURCE_KIND_ROLE,
    _SOURCE_VALUE_ROLE, row_to_track)


# a source's kind in the tree -> its node's kind in the playlists' tree (core.library.records)
_NODE_OF = {"playlist": PLAYLIST_NODE, "playlist_folder": FOLDER_NODE}


def _folder_of(item: Optional[QtWidgets.QTreeWidgetItem]) -> Optional[str]:
    """The playlist folder an item is in (None: the top)."""
    parent = item.parent() if item is not None else None
    if parent is not None and parent.data(0, _SOURCE_KIND_ROLE) == "playlist_folder":
        return str(parent.data(0, _SOURCE_VALUE_ROLE))
    return None


def _node_of(item: Optional[QtWidgets.QTreeWidgetItem]) -> Optional[tuple[str, str]]:
    """(kind, uid) of a playlist or folder item, else None."""
    kind = _NODE_OF.get(item.data(0, _SOURCE_KIND_ROLE)) if item is not None else None
    return (kind, str(item.data(0, _SOURCE_VALUE_ROLE))) if kind else None


class _LibrarySourceTree(QtWidgets.QTreeWidget):
    playlistFilesDropped = QtCore.Signal(str, object)
    # a playlist or folder dropped: (kind, uid), the folder it goes into (None: the top), the child it goes before
    # ((kind, uid); None: at the folder's end)
    playlistNodeDropped = QtCore.Signal(object, object, object)
    focusLeft = QtCore.Signal()

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
        self._drop_line: Optional[tuple[int, int]] = None   # (y, x it starts at: the level the drop goes into)
        self._drop_target_rect: Optional[QtCore.QRect] = None
        # moving with the keys only highlights a source; the library shows it once picked (the load key)
        self.keyboard_move = False

    def keyPressEvent(self, event):
        self.keyboard_move = True
        try:
            super().keyPressEvent(event)
        finally:
            self.keyboard_move = False

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.focusLeft.emit()

    def mimeTypes(self):
        return ["text/uri-list", _PLAYLIST_MIME]

    def mimeData(self, items):
        data = QtCore.QMimeData()
        nodes = [node for node in map(_node_of, items) if node is not None]
        if len(nodes) == 1:
            kind, uid = nodes[0]
            data.setData(_PLAYLIST_MIME, QtCore.QByteArray(f"{kind}:{uid}".encode("ascii")))
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

    def _playlist_insertion_at(self, pos: QtCore.QPoint, moving: tuple[str, str]):
        """Where a playlist or folder dropped at ``pos`` goes: (valid, folder, before, line (y, x), folder rect). Over
        a playlist: before it (its upper half) or after it; over a folder: before it (its top quarter), into it at its
        end, or under it (its bottom quarter: first in it when it is open with children, as the line there shows,
        else after it); over Playlists: at the top's end. The line starts at the level the drop goes into. A folder
        never into itself or a folder inside it."""
        item = self.itemAt(pos)
        kind = item.data(0, _SOURCE_KIND_ROLE) if item is not None else ""
        if kind == "playlists":
            last = item
            while last.childCount() and last.isExpanded():   # under the last row shown (an open folder's last)
                last = last.child(last.childCount() - 1)
            line = (self.visualItemRect(last).bottom() + 1, self.visualItemRect(item).left() + self.indentation())
            return True, None, None, line, None
        node = _node_of(item)
        if node is None:
            return False, None, None, None, None
        if moving[0] == FOLDER_NODE:          # not into the folder itself or one inside it
            up = item
            while up is not None:
                if _node_of(up) == moving:
                    return False, None, None, None, None
                up = up.parent()
        rect = self.visualItemRect(item)
        y = pos.y() - rect.top()
        folder = _folder_of(item)
        if kind == "playlist_folder" and rect.height() // 4 <= y < rect.height() - rect.height() // 4:
            return True, node[1], None, None, rect
        if y < rect.height() // 2:
            return True, folder, node, (rect.top(), rect.left()), None
        if kind == "playlist_folder" and item.isExpanded() and item.childCount():
            first = item.child(0)       # the line between the folder and its first child: first in it
            return True, node[1], _node_of(first), (rect.bottom() + 1, self.visualItemRect(first).left()), None
        parent = item.parent()
        following = parent.child(parent.indexOfChild(item) + 1) if parent is not None else None
        return True, folder, _node_of(following), (rect.bottom() + 1, rect.left()), None

    @staticmethod
    def _moving(event) -> Optional[tuple[str, str]]:
        """The playlist or folder a drag from this tree carries."""
        try:
            kind, _sep, uid = bytes(event.mimeData().data(_PLAYLIST_MIME)).decode("ascii").partition(":")
        except UnicodeDecodeError:
            return None
        return (kind, uid) if kind in (PLAYLIST_NODE, FOLDER_NODE) and uid else None

    def _set_drop_visual(
        self,
        *,
        line: Optional[tuple[int, int]] = None,
        target_rect: Optional[QtCore.QRect] = None,
    ) -> None:
        if self._drop_line != line or self._drop_target_rect != target_rect:
            self._drop_line = line
            self._drop_target_rect = target_rect
            self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_line is None and self._drop_target_rect is None:
            return
        painter = QtGui.QPainter(self.viewport())
        color = self.palette().highlight().color()
        if self._drop_target_rect is not None:
            painter.setPen(QtGui.QPen(color, 2))
            fill = QtGui.QColor(color)
            fill.setAlpha(35)
            painter.fillRect(self._drop_target_rect, fill)
            painter.drawRect(self._drop_target_rect.adjusted(1, 1, -2, -2))
        if self._drop_line is not None:
            pen = QtGui.QPen(color, 3)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(pen)
            y, x = self._drop_line
            y = max(1, min(self.viewport().height() - 2, y))
            painter.drawLine(max(8, x), y, self.viewport().width() - 5, y)

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
        moving = self._moving(event) if event.source() is self else None
        if moving is not None:
            valid, _folder, _before, line, into = self._playlist_insertion_at(event.position().toPoint(), moving)
            if valid:
                self._set_drop_visual(line=line, target_rect=into)
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
        moving = self._moving(event) if event.source() is self else None
        if moving is not None:
            valid, folder, before, _line, _into = self._playlist_insertion_at(event.position().toPoint(), moving)
            self._set_drop_visual()
            if not valid:
                event.ignore()
                return
            self.playlistNodeDropped.emit(moving, folder, before)
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


class SourcesMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _build_sources(self):
        """The sources panel's tree."""
        self._closed_folders: set[str] = set()     # the playlist folders the user closed (this run)
        self.source_tree = _LibrarySourceTree()
        self.source_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def _connect_sources(self):
        """The sources panel's signals."""
        self.source_tree.currentItemChanged.connect(self._on_source_changed)
        self.source_tree.itemExpanded.connect(self._on_source_expanded)
        self.source_tree.itemCollapsed.connect(self._on_source_collapsed)
        self.source_tree.customContextMenuRequested.connect(self._on_source_context_menu)
        self.source_tree.focusLeft.connect(self._mark_shown_source)
        self.source_tree.playlistFilesDropped.connect(self._add_paths_to_playlist)
        self.source_tree.playlistNodeDropped.connect(self._drop_playlist_node)

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

        library_item = QtWidgets.QTreeWidgetItem([tr("Library (All)")])
        library_item.setData(0, _SOURCE_KIND_ROLE, "library")
        tree.addTopLevelItem(library_item)

        variants_item = QtWidgets.QTreeWidgetItem([tr("Variants")])
        variants_item.setData(0, _SOURCE_KIND_ROLE, "variants")
        variants_item.setToolTip(0, tr(
            "<p>The library's tracks that are versions of one song. # is the song.</p>"))
        tree.addTopLevelItem(variants_item)
        loaded_item = QtWidgets.QTreeWidgetItem([tr("Loaded Track")])
        loaded_item.setData(0, _SOURCE_KIND_ROLE, "variants_loaded")
        loaded_item.setToolTip(0, tr("<p>The versions of the track loaded on the deck.</p>"))
        variants_item.addChild(loaded_item)
        variants_item.setExpanded(True)

        playlists_item = QtWidgets.QTreeWidgetItem([tr("Playlists")])
        playlists_item.setData(0, _SOURCE_KIND_ROLE, "playlists")
        tree.addTopLevelItem(playlists_item)
        self._shown_playlists = self._playlists_shape()
        children = self.library.snapshot.playlist_children()

        def add(parent: QtWidgets.QTreeWidgetItem, folder_uid) -> None:
            """A folder's children in their order, its folders open unless the user closed them."""
            for node, row in children.get(folder_uid, ()):
                item = QtWidgets.QTreeWidgetItem([row.name])
                item.setData(0, _SOURCE_VALUE_ROLE, row.uid)
                parent.addChild(item)
                if node == FOLDER_NODE:
                    item.setData(0, _SOURCE_KIND_ROLE, "playlist_folder")
                    add(item, row.uid)
                    opened = row.uid not in self._closed_folders
                    item.setExpanded(opened)
                    item.setIcon(0, folder_icon(opened and item.childCount() > 0))
                else:
                    item.setData(0, _SOURCE_KIND_ROLE, "playlist")
        add(playlists_item, None)
        playlists_item.setExpanded(True)

        local_item = QtWidgets.QTreeWidgetItem([tr("Local Folders")])
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

    def _playlists_shape(self) -> tuple:
        """The playlists' tree as the sources panel shows it (a change rebuilds it)."""
        snap = self.library.snapshot
        return (tuple((p.uid, p.name, p.folder_uid, p.sort_index) for p in snap.playlists),
                tuple((f.uid, f.name, f.parent_uid, f.sort_index) for f in snap.folders))

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
        if item.data(0, _SOURCE_KIND_ROLE) == "playlist_folder":
            self._closed_folders.discard(str(item.data(0, _SOURCE_VALUE_ROLE)))
            item.setIcon(0, folder_icon(item.childCount() > 0))
        self._populate_directory_children(item)

    def _on_source_collapsed(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if item.data(0, _SOURCE_KIND_ROLE) == "playlist_folder":
            self._closed_folders.add(str(item.data(0, _SOURCE_VALUE_ROLE)))
            item.setIcon(0, folder_icon(False))

    def _on_source_changed(self, current, _previous=None) -> None:
        if current is None or self.source_tree.keyboard_move:
            return
        if self._music_mode:
            self._exit_music_mode()
        self._source_kind = str(current.data(0, _SOURCE_KIND_ROLE) or "")
        self._source_value = current.data(0, _SOURCE_VALUE_ROLE)
        self._refresh_current_source(reset_sort=True)

    def _refresh_current_source(self, *, reset_sort: bool = False) -> None:
        self.table.setPlaylistReorderEnabled(self._source_kind == "playlist" and not self._music_mode)
        if self._music_mode:
            tracks = self._music_tracks()
            default_column = "#"
            default_order = QtCore.Qt.SortOrder.AscendingOrder
        elif self._source_kind == "library":
            tracks = list(self._library_tracks)
            default_column = "Added"
            default_order = QtCore.Qt.SortOrder.DescendingOrder
        elif self._source_kind == "playlist":
            snap = self.library.snapshot
            tracks = []
            for position, row in snap.playlist_tracks(str(self._source_value)):
                track = row_to_track(row, snap.readings.get(row.uid), snap.tieups.get(row.uid))
                track.playlist_index = position
                track._disp_playlist_index = str(position + 1)
                tracks.append(track)
            default_column = "#"
            default_order = QtCore.Qt.SortOrder.AscendingOrder
        elif self._source_kind in ("variants", "variants_loaded"):
            tracks = self._variant_tracks(self._loaded_uid or "" if self._source_kind == "variants_loaded" else None)
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

        source = ("music", None) if self._music_mode else (self._source_kind, self._source_value)
        if (source[0] == "library" and not reset_sort and self._table_source == source
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
        if self._music_mode:
            self._music_tooltips()
        elif self._source_kind in ("variants", "variants_loaded"):
            self._variant_marks()
        self._apply_filter()
        self.table.setUpdatesEnabled(True)

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

    def _toggle_sources(self) -> None:
        """The focus to the sources panel (playlists, local folders: Up / Down to browse), or back to the tracks."""
        if self.source_tree.hasFocus():
            self.focus_tracks()
        else:
            self._mark_shown_source()
            self.source_tree.setFocus(QtCore.Qt.ShortcutFocusReason)

    def _mark_shown_source(self) -> None:
        """The sources panel's highlight back on the source the library shows (keys moved it elsewhere)."""
        tree = self.source_tree
        current = tree.currentItem()
        if current is not None and (current.data(0, _SOURCE_KIND_ROLE), current.data(0, _SOURCE_VALUE_ROLE)) == (
                self._source_kind, self._source_value):
            return
        iterator = QtWidgets.QTreeWidgetItemIterator(tree)
        while iterator.value():
            item = iterator.value()
            if (item.data(0, _SOURCE_KIND_ROLE), item.data(0, _SOURCE_VALUE_ROLE)) == (
                    self._source_kind, self._source_value):
                tree.blockSignals(True)
                tree.setCurrentItem(item)
                tree.blockSignals(False)
                return
            iterator += 1

    def _variant_tracks(self, of: str | None = None) -> List[Track]:
        """The library's tracks with variants (core.search.variants), # their song: the songs with the newest track first,
        each song's longest track first; with ``of``, that track's song only (none for "": no track loaded)."""
        by_uid = {t.uid: t for t in self._library_tracks if t.uid}
        found = list(self.library.snapshot.variant_pairs)
        self._variant_pairs = {}
        for p in found:
            if p.a in by_uid and p.b in by_uid:
                self._variant_pairs.setdefault(p.a, []).append(p)
                self._variant_pairs.setdefault(p.b, []).append(p)

        def added(uid: str) -> int:
            t = by_uid[uid]
            return t.added.toSecsSinceEpoch() if t.added else 0

        groups = sorted(variant_relations.groups(found, set(by_uid)), key=lambda g: -max(added(u) for u in g))
        if of is not None:
            groups = [g for g in groups if of in g]
        self._variant_song = {}
        tracks = []
        for song, group in enumerate(groups, start=1):
            group.sort(key=lambda u: (-(by_uid[u].duration_sec or 0.0), by_uid[u].title))
            for k, uid in enumerate(group):
                track = copy.copy(by_uid[uid])
                track.playlist_index = song * 1000 + k
                track._disp_playlist_index = str(song)
                tracks.append(track)
                self._variant_song[uid] = song
        return tracks

    def _variant_relation(self, uid: str, p: variant_relations.Pair) -> str:
        """How the other track of ``p`` differs from ``uid``'s, in words ("" for nothing to say): the same mix, and
        its length. Another vocal / the instrumental are not told apart (the vocal range's level misreads quiet
        vocals)."""
        cover_self, cover_other = (p.cover_b, p.cover_a) if p.b == uid else (p.cover_a, p.cover_b)
        length = variant_fp.length(cover_other, cover_self)
        if p.kind == variant_fp.SAME_MIX_KIND:
            text = tr("the same mix")
            if length == "edit":
                return tr("{relation}, a shorter edit").format(relation=text)
            if length == "longer":
                return tr("{relation}, a longer version").format(relation=text)
            if length == "partly":
                return tr("{relation}, some sections differ").format(relation=text)
            return text
        if length == "edit":
            return tr("a shorter edit")
        if length == "longer":
            return tr("a longer version")
        if length == "partly":
            return tr("some sections differ")
        return ""

    def _variant_marks(self) -> None:
        """Each song's rows on one background (the songs alternating), each row's tooltip its variants."""
        theme = current_theme()
        shades = (QtGui.QBrush(QtGui.QColor(theme.base)), QtGui.QBrush(QtGui.QColor(theme.alt_base)))
        titles = {t.uid: t.title or os.path.basename(t.path) for t in self._all_tracks}
        for r in range(self.table.rowCount()):
            uid = self._row_uid(r)
            song = self._variant_song.get(uid)
            if song is None:
                continue
            lines = []
            for p in self._variant_pairs.get(uid, ()):
                title = titles.get(p.b if p.a == uid else p.a, "")
                relation = self._variant_relation(uid, p)
                lines.append(html.escape(tr("{title}: {relation}").format(title=title, relation=relation)
                                         if relation else title))
            tip = "<p>" + html.escape(tr("Variants of this track:")) + "<br>" + "<br>".join(lines) + "</p>"
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item is not None:
                    item.setToolTip(tip)
                    item.setBackground(shades[song % 2])

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
        # a new playlist or folder goes into the folder clicked, or beside the playlist clicked
        folder = str(item.data(0, _SOURCE_VALUE_ROLE)) if kind == "playlist_folder" else _folder_of(item)
        if kind in ("playlists", "playlist", "playlist_folder"):
            actions["new"] = menu.addAction(tr("New Playlist"))
            actions["new_folder"] = menu.addAction(tr("New Folder"))
        if kind in ("playlist", "playlist_folder"):
            menu.addSeparator()
            actions["rename"] = menu.addAction(tr("Rename"))
            actions["up"] = menu.addAction(tr("Move Up"))
            actions["down"] = menu.addAction(tr("Move Down"))
            actions["delete"] = menu.addAction(tr("Delete Playlist") if kind == "playlist" else tr("Delete Folder"))
        elif kind in ("directory", "local_folders"):
            actions["refresh"] = menu.addAction(tr("Refresh Folders"))
        if menu.isEmpty():
            return
        chosen = menu.exec_(self.source_tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == actions.get("new"):
            self._create_playlist(folder)
        elif chosen == actions.get("new_folder"):
            self._create_folder(folder)
        elif chosen == actions.get("rename"):
            if kind == "playlist":
                self._rename_playlist(item)
            else:
                self._rename_folder(item)
        elif chosen == actions.get("delete"):
            if kind == "playlist":
                self._delete_playlist(item)
            else:
                self._delete_folder(item)
        elif chosen == actions.get("up"):
            self._move_playlist(item, -1)
        elif chosen == actions.get("down"):
            self._move_playlist(item, 1)
        elif chosen == actions.get("refresh"):
            self._rebuild_source_tree()
            self._refresh_current_source()
