"""The playlists' part: playlists and their folders made, renamed, deleted, moved (in a folder's order or into
another folder), tracks added and removed."""
from __future__ import annotations

from PySide6 import QtWidgets
import os
from core.library import jobs
from core.library.records import FOLDER_NODE, PLAYLIST_NODE
from system.i18n import tr
from ui.library.common import LIBRARY_SOURCE, Source, SourceKind, _SOURCE_VALUE_ROLE, kind_of


class PlaylistsMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _create_playlist(self, folder_uid: str | None = None) -> None:
        """A playlist at the end of ``folder_uid`` (None: the top)."""
        name, accepted = QtWidgets.QInputDialog.getText(self, tr("New Playlist"), tr("Name"))
        if not accepted or not name.strip():
            return
        self.library.write(jobs.create_playlist, name, folder_uid,
                           done=lambda playlist: self._show_playlist(playlist.uid),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("New Playlist"), str(exc)))

    def _create_folder(self, parent_uid: str | None = None) -> None:
        """A playlist folder at the end of ``parent_uid`` (None: the top)."""
        name, accepted = QtWidgets.QInputDialog.getText(self, tr("New Folder"), tr("Name"))
        if not accepted or not name.strip():
            return
        self.library.write(jobs.create_playlist_folder, name, parent_uid,
                           done=lambda folder: self._show_source(Source(SourceKind.PLAYLIST_FOLDER, folder.uid)),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("New Folder"), str(exc)))

    def _shown_folder(self) -> str | None:
        """The playlist folder chosen in the sources panel, or the chosen playlist's (None: neither, or at the top):
        where a new playlist or folder goes."""
        source = self._source
        if source.kind is SourceKind.PLAYLIST_FOLDER:
            return source.value
        if source.kind is SourceKind.PLAYLIST:
            return next((p.folder_uid for p in self.library.snapshot.playlists if p.uid == source.value), None)
        return None

    def _show_playlist(self, playlist_uid: str) -> None:
        """A playlist written (the library's snapshot has it): the sources again, it selected."""
        self._show_source(Source(SourceKind.PLAYLIST, playlist_uid))

    def _show_source(self, source: Source) -> None:
        """A playlist or folder written: the sources again, it selected."""
        self._source = source
        self._rebuild_source_tree()
        self._select_source(source)

    def _rename_playlist(self, item: QtWidgets.QTreeWidgetItem) -> None:
        playlist_uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        name, accepted = QtWidgets.QInputDialog.getText(
            self, tr("Rename Playlist"), tr("Name"), text=item.text(0)
        )
        if not accepted or not name.strip():
            return
        self.library.write(jobs.rename_playlist, playlist_uid, name, done=lambda _r: self._show_playlist(playlist_uid),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("Rename Playlist"), str(exc)))

    def _rename_folder(self, item: QtWidgets.QTreeWidgetItem) -> None:
        folder_uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        name, accepted = QtWidgets.QInputDialog.getText(self, tr("Rename Folder"), tr("Name"), text=item.text(0))
        if not accepted or not name.strip():
            return
        self.library.write(jobs.rename_playlist_folder, folder_uid, name,
                           done=lambda _r: self._show_source(Source(SourceKind.PLAYLIST_FOLDER, folder_uid)),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("Rename Folder"), str(exc)))

    def _delete_playlist(self, item: QtWidgets.QTreeWidgetItem) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            tr("Delete Playlist"),
            tr('Delete playlist "{name}"?\nTracks remain in the Library.').format(name=item.text(0)),
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.library.write(jobs.delete_playlist, str(item.data(0, _SOURCE_VALUE_ROLE)), done=self._show_library)

    def _delete_folder(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """The folder deleted with every folder and playlist in it, asked first (No by default)."""
        answer = QtWidgets.QMessageBox.question(
            self, tr("Delete Folder"),
            tr('Delete folder "{name}" with every playlist and folder in it?\nTracks remain in the Library.').format(
                name=item.text(0)),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.library.write(jobs.delete_playlist_folder, str(item.data(0, _SOURCE_VALUE_ROLE)),
                           done=self._show_library)

    def _show_library(self, _result=None) -> None:
        """What was shown deleted: the whole library shown."""
        self._show_source(LIBRARY_SOURCE)

    def _move_playlist(self, item: QtWidgets.QTreeWidgetItem, offset: int) -> None:
        """A playlist or folder moved ``offset`` places among its folder's children."""
        source = Source(kind_of(item), str(item.data(0, _SOURCE_VALUE_ROLE)))
        node = FOLDER_NODE if source.kind is SourceKind.PLAYLIST_FOLDER else PLAYLIST_NODE
        self.library.write(jobs.move_playlist_node_by, node, source.value, offset,
                           done=lambda changed: changed and self._show_source(source))

    def _drop_playlist_node(self, moving, folder_uid, before) -> None:
        """A playlist or folder dropped: into ``folder_uid`` (None: the top), before ``before`` (None: at its end);
        shown once written (the drop's own handling done by then: the tree is rebuilt after it)."""
        node, uid = moving
        source = Source(SourceKind.PLAYLIST_FOLDER if node == FOLDER_NODE else SourceKind.PLAYLIST, uid)
        self.library.write(jobs.move_playlist_node, node, uid, folder_uid, before,
                           done=lambda changed: changed and self._show_source(source),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("Playlists"), str(exc)))

    def _drop_playlist_tracks(self, positions, before_position) -> None:
        if not self._playlist_shown():
            return
        shown = self._shown_source()

        def moved(changed) -> None:
            if changed and self._shown_source() == shown:
                self._refresh_current_source(reset_sort=True)

        self.library.write(jobs.move_playlist_entries, shown.value, positions, before_position, done=moved)

    def _add_paths_to_playlist(self, playlist_uid: str, paths) -> None:
        requested = [str(path) for path in paths if str(path)]
        if not requested:
            return
        analyzed_uids = []
        pending_paths = []
        snap = self.library.snapshot
        for path in requested:
            row = snap.find_file(path)
            if row is not None:
                analyzed_uids.append(row.uid)
            elif os.path.isfile(path):
                pending_paths.append(path)

        def added(_r) -> None:
            if self._shown_source() == Source(SourceKind.PLAYLIST, playlist_uid):
                self._refresh_current_source()

        if analyzed_uids:
            self.library.write(jobs.add_to_playlists, {playlist_uid: analyzed_uids}, done=added)
        for path in pending_paths:
            self._pending_playlist_paths.setdefault(
                self._file_identity(path), []
            ).append(playlist_uid)
            self.bus.sig_request_analyze.emit(path)

    def _playlist_shown(self) -> bool:
        """Whether the table shows a playlist's entries, each row's # its place in it: what the playlist's actions
        (remove, reorder, a drop into it) work on. Search by Music's results over a playlist are the library's tracks,
        their # a rank."""
        return self._shown_source().kind is SourceKind.PLAYLIST

    def _files_dropped_on_tracks(self, paths) -> None:
        """Files dropped on the track list: into the playlist it shows, as on that playlist in the sources panel; else
        into the library (analyzed, not loaded: the deck's drop loads)."""
        if self._playlist_shown():
            self._add_paths_to_playlist(self._shown_source().value, paths)
            return
        for path in paths:
            self.bus.sig_request_analyze.emit(path)

    def _remove_selected_from_playlist(self) -> None:
        if not self._playlist_shown():
            return
        positions = [self.table._entry_position(row) for row in self._selected_rows()]
        positions = [position for position in positions if position is not None]
        if not positions:
            return
        shown = self._shown_source()

        def removed(_r) -> None:
            if self._shown_source() == shown:
                self._refresh_current_source(reset_sort=True)

        self.library.write(jobs.remove_playlist_entries, shown.value, positions, done=removed)
