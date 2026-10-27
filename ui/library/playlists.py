"""The playlists' part: playlists and their folders made, renamed, deleted, moved (in a folder's order or into
another folder), tracks added and removed."""
from __future__ import annotations

from PySide6 import QtWidgets
import os
from core.library import jobs
from core.library.records import FOLDER_NODE, PLAYLIST_NODE
from system.i18n import tr
from ui.library.common import _SOURCE_KIND_ROLE, _SOURCE_VALUE_ROLE


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
                           done=lambda folder: self._show_source("playlist_folder", folder.uid),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("New Folder"), str(exc)))

    def _shown_folder(self) -> str | None:
        """The playlist folder the library shows, or the shown playlist's (None: neither, or at the top)."""
        if self._source_kind == "playlist_folder":
            return str(self._source_value)
        if self._source_kind == "playlist":
            return next((p.folder_uid for p in self.library.snapshot.playlists if p.uid == self._source_value), None)
        return None

    def _show_playlist(self, playlist_uid: str) -> None:
        """A playlist written (the library's snapshot has it): the sources again, it selected."""
        self._show_source("playlist", playlist_uid)

    def _show_source(self, kind: str, value) -> None:
        """A playlist or folder written: the sources again, it selected."""
        self._source_kind, self._source_value = kind, value
        self._rebuild_source_tree()
        self._select_source(kind, value)

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
                           done=lambda _r: self._show_source("playlist_folder", folder_uid),
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
        self._source_kind, self._source_value = "library", None
        self._rebuild_source_tree()
        self._select_source("library", None)

    def _move_playlist(self, item: QtWidgets.QTreeWidgetItem, offset: int) -> None:
        """A playlist or folder moved ``offset`` places among its folder's children."""
        kind = str(item.data(0, _SOURCE_KIND_ROLE))
        node = FOLDER_NODE if kind == "playlist_folder" else PLAYLIST_NODE
        uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        self.library.write(jobs.move_playlist_node_by, node, uid, offset,
                           done=lambda changed: changed and self._show_source(kind, uid))

    def _drop_playlist_node(self, moving, folder_uid, before) -> None:
        """A playlist or folder dropped: into ``folder_uid`` (None: the top), before ``before`` (None: at its end);
        shown once written (the drop's own handling done by then: the tree is rebuilt after it)."""
        node, uid = moving
        kind = "playlist_folder" if node == FOLDER_NODE else "playlist"
        self.library.write(jobs.move_playlist_node, node, uid, folder_uid, before,
                           done=lambda changed: changed and self._show_source(kind, uid),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("Playlists"), str(exc)))

    def _drop_playlist_tracks(self, positions, before_position) -> None:
        if self._source_kind != "playlist":
            return
        playlist_uid = str(self._source_value)

        def moved(changed) -> None:
            if changed and self._source_kind == "playlist" and str(self._source_value) == playlist_uid:
                self._refresh_current_source(reset_sort=True)

        self.library.write(jobs.move_playlist_entries, playlist_uid, positions, before_position, done=moved)

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
            if self._source_kind == "playlist" and str(self._source_value) == playlist_uid:
                self._refresh_current_source()

        if analyzed_uids:
            self.library.write(jobs.add_to_playlists, {playlist_uid: analyzed_uids}, done=added)
        for path in pending_paths:
            self._pending_playlist_paths.setdefault(
                self._file_identity(path), []
            ).append(playlist_uid)
            self.bus.sig_request_analyze.emit(path)

    def _remove_selected_from_playlist(self) -> None:
        if self._source_kind != "playlist":
            return
        positions = [self.table._entry_position(row) for row in self._selected_rows()]
        positions = [position for position in positions if position is not None]
        if not positions:
            return
        playlist_uid = str(self._source_value)

        def removed(_r) -> None:
            if self._source_kind == "playlist" and str(self._source_value) == playlist_uid:
                self._refresh_current_source(reset_sort=True)

        self.library.write(jobs.remove_playlist_entries, playlist_uid, positions, done=removed)
