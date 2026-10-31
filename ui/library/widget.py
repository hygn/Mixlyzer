"""LibraryWidget: the library's table, its sources and search, built from its parts (the mixins beside it)."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
from PySide6 import QtCore, QtGui, QtWidgets
import os
from core.library import jobs
from core.library.records import TrackRow
from core.library.service import LibraryService
from core.workflows.event_bus import EventBus
from core.settings.config import config
from core.track.model import DataModel
from ui.theme import current_theme
from system.i18n import tr, tr_mark
from system.os_support import path_key
from ui.library.actions import ActionsMixin
from ui.library.common import (INCREMENTAL_ROWS_MAX, LIBRARY_UPDATE_DELAY_MS, Track, 
    _ANALYZED_ROLE, _SORT_ROLE, _SortItem, key_order, row_to_track)
from ui.library.mix import MixMixin
from ui.library.music import MusicMixin
from ui.library.playlists import PlaylistsMixin
from ui.library.search import SearchMixin
from ui.library.sources import LOADED_SOURCES, SourcesMixin
from ui.library.table import FastDelegate, _TrackTable
from ui.library.track_panel import TrackPanelMixin

if TYPE_CHECKING:   # the window imports the tie-up editor's parts (ui.library.tabs): imported where it opens
    from ui.windows.tieup_works import WorksWindow


class LibraryWidget(SourcesMixin, SearchMixin, MusicMixin, MixMixin, PlaylistsMixin, TrackPanelMixin, ActionsMixin,
                    QtWidgets.QWidget):
    ART_HEADER = ""  # the album art column (the top of the art)
    ART_WIDTH = 56
    # the columns by their English names (the code's keys); shown as HEADER_NAMES (BPM, Key: as they are)
    HEADERS = [ART_HEADER, "#", "Title", "Artist", "Album", "Tie-up", "BPM", "Key", "Duration", "Rating", "Added",
               "Path", "Comment"]
    SEARCH_MODES = ["All"] + HEADERS[1:]
    HEADER_NAMES = {"All": tr_mark("All"), "Title": tr_mark("Title"), "Artist": tr_mark("Artist"),
                    "Album": tr_mark("Album"), "Tie-up": tr_mark("Tie-up"), "Duration": tr_mark("Duration"),
                    "Rating": tr_mark("Rating"),
                    "Added": tr_mark("Added"), "Path": tr_mark("Path"), "Comment": tr_mark("Comment")}
    AUDIO_EXTENSIONS = frozenset({
        ".aac", ".aif", ".aiff", ".alac", ".flac", ".m4a", ".mp3",
        ".mp4", ".ogg", ".opus", ".wav", ".wma",
    })
    @classmethod
    def _header_name(cls, header: str) -> str:
        """A column's name as shown: translated, BPM and Key as they are."""
        return tr(cls.HEADER_NAMES[header]) if header in cls.HEADER_NAMES else header
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
    def __init__(self, bus: EventBus, cfg: config, library: LibraryService, model: DataModel | None = None,
                 parent=None):
        """``library``: the library's reads (its snapshot) and writes, on its own thread."""
        super().__init__(parent)
        self.bus = bus
        self.cfg = cfg
        self.library = library
        self.model = model
        self._external_sync_enabled = bool(getattr(cfg, "externalsyncconfig", None) and cfg.externalsyncconfig.enabled)
        self._all_tracks: List[Track] = []
        self._library_tracks: List[Track] = []
        self._source_kind = "library"
        self._source_value = None
        # Variants shown: uid -> its relations (core.search.variants.Pair), uid -> its song's number
        self._variant_pairs: dict[str, list] = {}
        self._variant_song: dict[str, int] = {}
        self._works_window: Optional[WorksWindow] = None     # made when first opened (open_works)
        # file identity -> the playlists (uids) it goes into once analyzed, once per drop
        self._pending_playlist_paths: dict[str, list[str]] = {}

        self._build_sources()
        self._build_music()
        primary = self._build_search()
        primary.addWidget(self._build_mix())     # the Mix Candidates' criteria, at the row's end while shown
        self._build_table()

        content = QtWidgets.QWidget()
        content_lay = QtWidgets.QVBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(6)
        content_lay.addLayout(primary)
        content_lay.addWidget(self.filter_bar)
        content_lay.addLayout(self._build_track_panel(), 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.source_tree)
        splitter.addWidget(content)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 1200])

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)

        self._connect_search()
        self._connect_sources()

        # The loaded track is shown in green.
        self._loaded_uid: Optional[str] = None
        self._marked_uid: Optional[str] = None   # the uid whose rows are green
        if self.model is not None:
            self.model.sig_properties_changed.connect(self._on_properties_loaded)
        # Index Now found new Variants: read again and shown
        self.bus.sig_music_index_finished.connect(lambda *_args: self.library.refresh())

        # the table's rows are this source's tracks (a library update changes them in place)
        self._table_source: Optional[tuple] = None
        # the phrase / transition filters' search (on the library thread): asked, and the last answer
        self._structured_asked: Optional[tuple] = None
        self._structured_found: Optional[tuple] = None
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

    def _build_table(self):
        """The track table, its columns, painting and actions."""
        # Table
        self.table = _TrackTable(
            self.HEADERS.index("Path"), self.HEADERS.index("#")
        )
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels([self._header_name(h) for h in self.HEADERS])
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
            self.table, art_column=self.HEADERS.index(self.ART_HEADER), library=self.library))
        art_column = self.HEADERS.index(self.ART_HEADER)
        self.table.horizontalHeader().setSectionResizeMode(art_column, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(art_column, self.ART_WIDTH)
        self.table.doubleClicked.connect(self._on_activate)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.table.playlistRowsDropped.connect(self._drop_playlist_tracks)

        # files dropped on the tracks: into the library (analyzed), not loaded (the deck's drop loads)
        self.table.filesDropped.connect(lambda paths: [self.bus.sig_request_analyze.emit(p) for p in paths])
    # Data loading
    def reload_from_db(self):
        self.reload(self.library.snapshot.tracks)
    def reload(self, data):
        """Refresh analyzed library data, then redraw the selected source."""
        delegate = self.table.itemDelegate()
        if isinstance(delegate, FastDelegate):
            delegate.clear_cache()  # thumbnails may have been replaced
        self._library_tracks = self._tracks(data)
        self._refresh_current_source()
        self.table.viewport().update()
    def _tracks(self, rows) -> List[Track]:
        """Library rows as Tracks, with their readings and tie-ups for the search (the library's snapshot)."""
        snap = self.library.snapshot
        return [row_to_track(row, snap.readings.get(row.uid), snap.tieups.get(row.uid), snap.initials.get(row.uid),
                             snap.sort_keys.get(row.uid)) if isinstance(row, TrackRow)
                else row for row in rows]
    @staticmethod
    def _file_identity(path: object) -> str:
        text = os.path.expanduser(os.path.expandvars(str(path or "")))
        return path_key(os.path.realpath(os.path.abspath(text)))
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
    def _on_library_updated(self, data) -> None:
        """Library updates in a row are applied once, the last (LIBRARY_UPDATE_DELAY_MS)."""
        self._pending_library_rows = list(data or [])
        self._library_timer.start()
    def _apply_library_update(self) -> None:
        data, self._pending_library_rows = self._pending_library_rows, None
        if data is None:
            return
        self._library_tracks = self._tracks(data)
        self._structured_found = None      # the phrase / transition searches asked again
        self._mix_forget()                 # and the mix candidates
        resolved: dict[str, list[str]] = {}
        for track in self._library_tracks:
            playlist_uids = self._pending_playlist_paths.pop(
                self._file_identity(track.path), []
            )
            for playlist_uid in playlist_uids:
                resolved.setdefault(playlist_uid, []).append(track.uid)
        if resolved:     # files analyzed for a playlist: into it now (the library updated again once written)
            self.library.write(jobs.add_to_playlists, resolved)
        if self._playlists_shape() != self._shown_playlists:
            self._rebuild_source_tree()      # another library, or playlists changed elsewhere
        self._refresh_current_source()
    def set_config(self, cfg: config) -> None:
        old_library = self.cfg.libconfig.libpath
        old_local = list(self.cfg.libconfig.local_music_directories)
        self.cfg = cfg
        delegate = self.table.itemDelegate()
        if isinstance(delegate, FastDelegate):
            delegate.clear_cache()      # another library's thumbnails
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
            t.title, t.artist, t.album, t.tieup, t._disp_bpm, t._disp_key,
            t._disp_duration, t._disp_rating, t._disp_added, t.path, t.comment
        ]
        bpm_col = self.HEADERS.index("BPM")
        art_col = self.HEADERS.index(self.ART_HEADER)
        index_col = self.HEADERS.index("#")
        key_col = self.HEADERS.index("Key")
        name_cols = {self.HEADERS.index(h): f for h, f in (("Title", "title"), ("Artist", "artist"), ("Album", "album"),
                                                           ("Tie-up", "tieup"))}
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
            elif c == key_col:  # in the Alphanumeric order (as text, 10A came before 2A)
                item.setData(_SORT_ROLE, key_order(t.key))
            elif c in name_cols and t.sort_keys.get(name_cols[c]) is not None:   # by reading (Sort Names)
                item.setData(_SORT_ROLE, t.sort_keys[name_cols[c]])
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
        loaded = str(uid or "").strip() or None
        if loaded != self._loaded_uid and self._source_kind in LOADED_SOURCES and not self._music_mode:
            self._loaded_uid = loaded
            self._refresh_current_source(reset_sort=True)   # the new track's candidates, related tracks or versions
        self._loaded_uid = loaded
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
    def _on_activate(self, idx: QtCore.QModelIndex):
        if self._external_sync_enabled:
            QtWidgets.QMessageBox.information(
                self,
                tr("External Sync"),
                tr("Track loading is disabled while External Sync is enabled."),
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
        match = next((m for m in self._music_matches if m.uid == self._row_uid(r)), None) if self._music_mode else None
        if path and match is not None:
            self._deck_seek.load(path, max(0.0, match.start_sec - 1.0))
        elif path:
            self.bus.sig_request_load_track.emit(path)
    def _on_external_sync_enabled(self, enabled: bool) -> None:
        self._external_sync_enabled = bool(enabled)
    # The track panel: Edit / Index / Tie-ups / Export / Sync tabs on a thin vertical tab bar right of the track list, under the search (as
    # tall as the list). Clicking the open tab folds it to the tab bar; another tab unfolds it. Unfolded, it shows
    # the selected track (the first analyzed one), following the selection once it rests for PANEL_FOLLOW_MS (keys
    # held down do not load every track passed); a tab's changes not applied (Edit, Search Index) are asked about
    # first.
    EDIT_TAB, INDEX_TAB, TIEUPS_TAB, EXPORT_TAB, SYNC_TAB = 0, 1, 2, 3, 4
    TRACK_PANEL_WIDTH = 380
    TRACK_PANEL_MIN_WIDTH = 300
    PANEL_FOLLOW_MS = 120
    # Keyboard shortcuts (ui.shortcuts): on the table's selection, wherever the focus is
    def shortcut_handlers(self) -> dict:
        return {
            "library.previous": lambda: self._step_selection(-1),
            "library.next": lambda: self._step_selection(1),
            "library.load": self._load_selected,
            "library.browse_sources": self._toggle_sources,
            "library.search": self._focus_search,
            "library.music_search": self._toggle_mic,
            "library.locate_loaded": self._select_loaded_track,
            "library.edit_track": self._edit_selected,
            "library.search_index_track": lambda: self._open_tab_selected(self.INDEX_TAB),
            "library.tieups_track": lambda: self._open_tab_selected(self.TIEUPS_TAB),
            "library.rebuild_fingerprints": self._rebuild_fingerprints_selected,
            "library.find_tieups_on_web": self._find_tieups_selected,
            "library.tieup_works": self.open_works,
            "library.export_track": self._export_selected,
            "library.sync_track": lambda: self._open_tab_selected(self.SYNC_TAB),
            "library.write_tags": self._write_tags_selected,
            "library.rekordbox_sync_track": self._rekordbox_sync_selected,
            "library.export_tempo_map": self._export_tempo_map_selected,
            "library.render_audio": self._render_audio_selected,
            "library.reveal": self._reveal_selected,
            "library.new_playlist": lambda: self._create_playlist(self._shown_folder()),
            "library.new_playlist_folder": lambda: self._create_folder(self._shown_folder()),
            "library.reanalyze": self._reanalyze_selected,
            "library.remove_from_library": self._on_remove_selected,
            "library.remove_from_playlist": self._remove_selected_from_playlist,
        }
    def _select_row(self, row: int) -> None:
        self.table.setCurrentCell(row, max(self.table.currentColumn(), 0))
        self.table.scrollToItem(self.table.item(row, 0))
    def _step_selection(self, step: int) -> None:
        """Select the shown row before (``step`` -1) / after the current one; without one, the last / first."""
        n = self.table.rowCount()
        current = self.table.currentRow()
        if current < 0 or self.table.isRowHidden(current):
            rows = range(n) if step > 0 else range(n - 1, -1, -1)
        else:
            rows = range(current + 1, n) if step > 0 else range(current - 1, -1, -1)
        row = next((r for r in rows if not self.table.isRowHidden(r)), None)
        if row is not None:
            self._select_row(row)
    def _load_selected(self) -> None:
        """Load the current row when it is selected, else the first selected. In the sources panel, the
        highlighted source instead: shown in the library, the focus on its tracks."""
        if self.source_tree.hasFocus():
            self._on_source_changed(self.source_tree.currentItem())
            self.focus_tracks()
            return
        rows = self._selected_rows()
        if rows:
            row = self.table.currentRow() if self.table.currentRow() in rows else rows[0]
            self._on_activate(self.table.model().index(row, 0))
    def focus_tracks(self) -> None:
        self.table.setFocus(QtCore.Qt.ShortcutFocusReason)
    def _select_loaded_track(self) -> str | None:
        if not self._loaded_uid:
            return None
        row = next((r for r in range(self.table.rowCount())
                    if not self.table.isRowHidden(r) and self._row_uid(r) == self._loaded_uid), None)
        if row is None:
            return tr("The loaded track is not in the list shown.")
        self._select_row(row)
        return None
