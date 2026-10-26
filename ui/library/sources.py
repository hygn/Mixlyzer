"""The sources panel's part: the library, Variants, playlists and folders listed, the one shown read."""
from __future__ import annotations

import copy
import html
from pathlib import Path
from typing import List, Optional
from PySide6 import QtCore, QtGui, QtWidgets
import os
from core.search import reading
from ui.theme import current_theme
from analyzer.search import variants as variant_fp
from core.search import variants as variant_relations
from system.i18n import tr
from ui.library.common import (Track, _DIRECTORY_LOADED_ROLE, _PLAYLIST_MIME, _SOURCE_KIND_ROLE,
    _SOURCE_VALUE_ROLE, row_to_track)


class _LibrarySourceTree(QtWidgets.QTreeWidget):
    playlistFilesDropped = QtCore.Signal(str, object)
    playlistOrderDropped = QtCore.Signal(str, object)
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
        self._drop_line_y: Optional[int] = None
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


class SourcesMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _build_sources(self):
        """The sources panel's tree."""
        self.source_tree = _LibrarySourceTree()
        self.source_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

    def _connect_sources(self):
        """The sources panel's signals."""
        self.source_tree.currentItemChanged.connect(self._on_source_changed)
        self.source_tree.itemExpanded.connect(self._on_source_expanded)
        self.source_tree.customContextMenuRequested.connect(self._on_source_context_menu)
        self.source_tree.focusLeft.connect(self._mark_shown_source)
        self.source_tree.playlistFilesDropped.connect(self._add_paths_to_playlist)
        self.source_tree.playlistOrderDropped.connect(self._drop_playlist_order)

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
            "<p>The library's tracks that are versions of one song: the same audio, another vocal, the instrumental, "
            "an edit (a TV size). # is the song; a row's tooltip says how it differs from the others.</p>"))
        tree.addTopLevelItem(variants_item)
        loaded_item = QtWidgets.QTreeWidgetItem([tr("Loaded Track")])
        loaded_item.setData(0, _SOURCE_KIND_ROLE, "variants_loaded")
        loaded_item.setToolTip(0, tr("<p>The versions of the track loaded on the deck, following each track "
                                     "loaded.</p>"))
        variants_item.addChild(loaded_item)
        variants_item.setExpanded(True)

        playlists_item = QtWidgets.QTreeWidgetItem([tr("Playlists")])
        playlists_item.setData(0, _SOURCE_KIND_ROLE, "playlists")
        tree.addTopLevelItem(playlists_item)
        playlists = self.library.snapshot.playlists
        self._shown_playlists = tuple((p.uid, p.name) for p in playlists)
        for playlist in playlists:
            item = QtWidgets.QTreeWidgetItem([playlist.name])
            item.setData(0, _SOURCE_KIND_ROLE, "playlist")
            item.setData(0, _SOURCE_VALUE_ROLE, playlist.uid)
            playlists_item.addChild(item)
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
        if kind in ("playlists", "playlist"):
            actions["new"] = menu.addAction(tr("New Playlist"))
        if kind == "playlist":
            menu.addSeparator()
            actions["rename"] = menu.addAction(tr("Rename"))
            actions["up"] = menu.addAction(tr("Move Up"))
            actions["down"] = menu.addAction(tr("Move Down"))
            actions["delete"] = menu.addAction(tr("Delete Playlist"))
        elif kind in ("directory", "local_folders"):
            actions["refresh"] = menu.addAction(tr("Refresh Folders"))
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
