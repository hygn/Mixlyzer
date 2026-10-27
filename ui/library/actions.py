"""The tracks' actions: the context menu, remove, reveal, export, write tags, reanalyze, rebuild fingerprints, find
tie-ups; the selection they act on."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
import os
from system import os_support
from core.library import jobs
from core.settings.config import load_cfg
from ui.tieup_works import WorksWindow
from ui.reanalyze_dialog import ReanalyzeDialog
from ui.edit_panel.flash import flash
from system.i18n import tr, tr_n


class ActionsMixin:
    """This module's part of LibraryWidget (mixed into it)."""

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

    def _on_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        selected_paths = self._selected_paths()
        analyzed_by_path = {track.path: track.analyzed for track in self._all_tracks}
        all_analyzed = bool(selected_paths) and all(
            analyzed_by_path.get(path, False) for path in selected_paths
        )
        act_reveal = menu.addAction(tr("Show in Explorer") if os_support.IS_WINDOWS else
                                    tr("Show in Finder") if os_support.IS_MACOS else tr("Show in File Manager"))
        act_remove = menu.addAction(tr("Remove from Library"))
        act_edit = menu.addAction(tr("Edit Track"))
        act_reanalyze = menu.addAction(
            tr("Reanalyze Track") if all_analyzed else tr("Analyze Track")
        )
        act_export = menu.addAction(tr("Export Track"))
        act_remove_playlist = None
        if self._source_kind == "playlist":
            act_remove_playlist = menu.addAction(tr("Remove from Playlist"))
        add_menu = menu.addMenu(tr("Add to Playlist"))
        playlist_actions = {}
        playlists = self.library.snapshot.playlists
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

    def _on_remove_selected(self):
        uids = self._selected_uids()
        if not uids or not self._confirm_remove(uids):
            return

        def removed(_r) -> None:
            for uid in uids:
                self.bus.sig_rekordbox_sync_track_requested.emit(uid)
            if self._loaded_uid in uids:    # the deck's track left the library: off the deck (nowhere to save it)
                self.bus.sig_unload_track_requested.emit()

        # the tracks, their features and thumbnails, and their place in the indexes made from them
        self.library.write(jobs.remove_tracks, list(uids), done=removed)

    def _confirm_remove(self, uids: list[str]) -> bool:
        """Asked before tracks leave the library (their analysis and edits deleted; No by default)."""
        if len(uids) == 1:
            title = next((t.title for t in self._all_tracks if t.uid == uids[0]), "") or uids[0]
            # a long title cut, so the message box keeps its width
            title = QtGui.QFontMetrics(self.font()).elidedText(title, QtCore.Qt.ElideMiddle, 420)
            text = tr('Remove "{title}" from the library?\nIts analysis and edits are deleted; the audio file '
                      'stays.').format(title=title)
        else:
            text = tr_n("Remove {count} track from the library?\nIts analysis and edits are deleted; the audio "
                        "file stays.",
                        "Remove {count} tracks from the library?\nTheir analysis and edits are deleted; the audio "
                        "files stay.", len(uids)).format(count=len(uids))
        answer = QtWidgets.QMessageBox.question(
            self, tr("Remove from Library"), text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return answer == QtWidgets.QMessageBox.Yes

    # the system's file manager
    def _reveal_selected(self):
        for row in self._selected_rows():
            path = self.table.item(row, self.HEADERS.index("Path")).text()
            if path:
                os_support.reveal_file(path)

    def _export_selected(self):
        if self._selected_uids():
            self.open_track_panel(self.EXPORT_TAB)

    def _write_tags_selected(self):
        """The Export tab with the selected track, its checked fields written into the file's tags."""
        if self._selected_uids():
            self.open_track_panel(self.EXPORT_TAB, focus=False)
            flash(self.export_panel.btn_write_tags)
            self.export_panel.write_tags()

    # Reanalyze
    def _reanalyze_selected(self):
        """The selected tracks analyzed again: the analyzed ones once the reanalysis dialog is accepted, with
        the analysis settings set there for this run only (the saved ones stay); the others analyzed."""
        by_path = {track.path: track for track in self._all_tracks}
        paths = [p for p in (self.table.item(r, self.HEADERS.index("Path")).text() for r in self._selected_rows()) if p]
        analyzed = [p for p in paths if by_path.get(p) is not None and by_path[p].analyzed]
        analysis = None
        if analyzed:
            saved = load_cfg().analysisconfig
            dlg = ReanalyzeDialog(saved, [by_path[p].title or os.path.basename(p) for p in analyzed], self)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return
            analysis = {k: v for k, v in dlg.analysis().items() if getattr(saved, k) != v} or None
        for path in paths:
            if path in analyzed:
                self.bus.sig_reanalyze_requested.emit(path, analysis)
            else:
                self.bus.sig_request_analyze.emit(path)

    def _rebuild_fingerprints_selected(self) -> None:
        """The Search Index tab with the selected track, its audio fingerprints rebuilt."""
        if self._selected_uids():
            self.open_track_panel(self.INDEX_TAB, focus=False)
            flash(self.index_panel.btn_rebuild)
            self.index_panel.rebuild()

    def _find_tieups_selected(self) -> None:
        """The Tie-ups tab with the selected track, looked up on the web."""
        if self._selected_uids():
            self.open_track_panel(self.TIEUPS_TAB, focus=False)
            flash(self.tieup_panel.box.btn_web)
            self.tieup_panel.box.btn_web.click()

    def open_works(self) -> None:
        """The tie-ups' works window (non-modal; one)."""
        if self._works_window is None:
            self._works_window = WorksWindow(self.bus, lambda: self.cfg, self.library, self.window())
        self._works_window.show_works()
