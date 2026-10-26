"""The playlists' part: made, renamed, deleted, reordered, tracks added and removed."""
from __future__ import annotations

from PySide6 import QtWidgets
import os
from core.library import jobs
from system.i18n import tr
from ui.library.common import _SOURCE_VALUE_ROLE


class PlaylistsMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _create_playlist(self) -> None:
        name, accepted = QtWidgets.QInputDialog.getText(self, tr("New Playlist"), tr("Name"))
        if not accepted or not name.strip():
            return
        self.library.write(jobs.create_playlist, name, done=lambda playlist: self._show_playlist(playlist.uid),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("New Playlist"), str(exc)))

    def _show_playlist(self, playlist_uid: str) -> None:
        """A playlist written (the library's snapshot has it): the sources again, it selected."""
        self._source_kind, self._source_value = "playlist", playlist_uid
        self._rebuild_source_tree()
        self._select_source("playlist", playlist_uid)

    def _rename_playlist(self, item: QtWidgets.QTreeWidgetItem) -> None:
        playlist_uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        name, accepted = QtWidgets.QInputDialog.getText(
            self, tr("Rename Playlist"), tr("Name"), text=item.text(0)
        )
        if not accepted or not name.strip():
            return
        self.library.write(jobs.rename_playlist, playlist_uid, name, done=lambda _r: self._show_playlist(playlist_uid),
                           failed=lambda exc: QtWidgets.QMessageBox.warning(self, tr("Rename Playlist"), str(exc)))

    def _delete_playlist(self, item: QtWidgets.QTreeWidgetItem) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            tr("Delete Playlist"),
            tr('Delete playlist "{name}"?\nTracks remain in the Library.').format(name=item.text(0)),
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        playlist_uid = str(item.data(0, _SOURCE_VALUE_ROLE))

        def deleted(_r) -> None:
            self._source_kind, self._source_value = "library", None
            self._rebuild_source_tree()
            self._select_source("library", None)

        self.library.write(jobs.delete_playlist, playlist_uid, done=deleted)

    def _move_playlist(self, item: QtWidgets.QTreeWidgetItem, offset: int) -> None:
        playlist_uid = str(item.data(0, _SOURCE_VALUE_ROLE))
        self.library.write(jobs.move_playlist, playlist_uid, offset, done=lambda _r: self._show_playlist(playlist_uid))

    def _drop_playlist_order(self, playlist_uid: str, before_playlist_uid) -> None:
        # shown once written (the drop's own handling done by then: the tree is rebuilt after it)
        self.library.write(jobs.move_playlist_before, playlist_uid, before_playlist_uid,
                           done=lambda changed: changed and self._show_playlist(playlist_uid))

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
