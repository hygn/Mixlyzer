"""The track panel's part: its tabs (Edit, Index, Tie-ups, Export, Sync), folded and unfolded, following the
selection."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
from ui.library.tabs.edit_song import EditTrackPanel
from ui.library.tabs.export import ExportTrackPanel
from ui.library.tabs.sync import SyncTrackPanel
from ui.library.tabs.tieup_editor import TieupPanel
from ui.library.tabs.track_search_index import SearchIndexPanel
from ui.theme import current_theme
from system.i18n import tr, tr_ctx


class TrackPanelMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _build_track_panel(self) -> QtWidgets.QHBoxLayout:
        self.edit_panel = EditTrackPanel(bus=self.bus, library=self.library)
        self.index_panel = SearchIndexPanel(bus=self.bus, library=self.library,
                                            search_settings=lambda: self.cfg.searchconfig)
        self.tieup_panel = TieupPanel(bus=self.bus, library=self.library,
                                      settings=lambda: self.cfg.searchconfig,
                                      to_versions=lambda: bool(self.cfg.libconfig.tieup_to_versions))
        self.export_panel = ExportTrackPanel(self.library, model=self.model)
        self.sync_panel = SyncTrackPanel(
            self.library, model=self.model, bus=self.bus,
            tieup_language=lambda: self.cfg.libconfig.export_tieup_language,
            unapplied_edits=lambda uid: self.edit_panel.track_uid() == uid and self.edit_panel.has_changes())
        self.panel_stack = QtWidgets.QStackedWidget()
        for panel in (self.edit_panel, self.index_panel, self.tieup_panel, self.export_panel, self.sync_panel):
            self.panel_stack.addWidget(panel)
        self.panel_stack.setMinimumWidth(self.TRACK_PANEL_MIN_WIDTH)
        self.panel_stack.hide()
        self.panel_tabs = QtWidgets.QTabBar()
        self.panel_tabs.setShape(QtWidgets.QTabBar.RoundedEast)
        self.panel_tabs.setDrawBase(False)
        # thin: the app's tab padding (its sides) is this bar's thickness; the open tab marked on the panel's side
        theme = current_theme()
        tab_font = QtGui.QFont(self.panel_tabs.font())
        tab_font.setPointSize(9)
        self.panel_tabs.setFont(tab_font)
        self.panel_tabs.setStyleSheet(
            "QTabBar::tab { padding: 10px 2px; border: none; border-left: 2px solid transparent; }"
            f"QTabBar::tab:selected {{ border-left-color: {theme.accent}; }}")
        self.panel_tabs.addTab(tr_ctx("track panel tab", "Edit"))
        self.panel_tabs.addTab(tr_ctx("track panel tab", "Index"))
        self.panel_tabs.addTab(tr_ctx("track panel tab", "Tie-ups"))
        self.panel_tabs.addTab(tr_ctx("track panel tab", "Export"))
        self.panel_tabs.addTab(tr_ctx("track panel tab", "Sync"))
        for i in range(self.panel_tabs.count()):
            self.panel_tabs.setTabToolTip(i, tr("The selected track; click the open tab to fold the panel"))
        self.panel_tabs.tabBarClicked.connect(self._on_panel_tab_clicked)
        self._panel_follow = QtCore.QTimer(self)
        self._panel_follow.setSingleShot(True)
        self._panel_follow.setInterval(self.PANEL_FOLLOW_MS)
        self._panel_follow.timeout.connect(self._show_selection_in_panel)
        self.table.itemSelectionChanged.connect(
            lambda: self._panel_follow.start() if self.panel_stack.isVisible() else None)
        self._panel_width = self.TRACK_PANEL_WIDTH
        self.panel_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.panel_split.addWidget(self.table)
        self.panel_split.addWidget(self.panel_stack)
        self.panel_split.setCollapsible(0, False)
        self.panel_split.setCollapsible(1, False)
        self.panel_split.setStretchFactor(0, 1)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.panel_split, 1)
        row.addWidget(self.panel_tabs, 0, QtCore.Qt.AlignTop)
        return row

    def _on_panel_tab_clicked(self, index: int) -> None:
        if index < 0:
            return
        if self.panel_stack.isVisible() and index == self.panel_stack.currentIndex():
            self._fold_track_panel()
        else:
            self.open_track_panel(index, focus=False)

    def _fold_track_panel(self) -> None:
        self._panel_width = self.panel_split.sizes()[1] or self._panel_width
        if self.track_panel_has_focus():
            self.focus_tracks()
        self.panel_stack.hide()

    def open_track_panel(self, index: int, *, focus: bool = True) -> None:
        """Unfold the panel at tab ``index`` with the selected track (the first analyzed one) and, with
        ``focus``, its first field focused (the context menu's Edit / Export, their shortcuts)."""
        self.panel_tabs.setCurrentIndex(index)
        self.panel_stack.setCurrentIndex(index)
        if not self.panel_stack.isVisible():
            self.panel_stack.show()
            total = sum(self.panel_split.sizes())
            width = min(self._panel_width, max(self.TRACK_PANEL_MIN_WIDTH, total // 2))
            self.panel_split.setSizes([max(1, total - width), width])
        self._show_selection_in_panel(reload=True)
        if focus:
            target = {self.EDIT_TAB: self.edit_panel.ed_title, self.INDEX_TAB: self.index_panel.tbl_readings,
                      self.TIEUPS_TAB: self.tieup_panel.box.table, self.SYNC_TAB: self.sync_panel.btn_write_tags,
                      }.get(index, self.export_panel.spn_tempo)
            if not target.isEnabled():
                target = self.export_panel.dest_edit if index == self.EXPORT_TAB else self.table
            target.setFocus(QtCore.Qt.OtherFocusReason)
            if isinstance(target, QtWidgets.QLineEdit):
                target.selectAll()

    def _show_selection_in_panel(self, reload: bool = False) -> None:
        """The open tab shows the selected track (the first analyzed one), unless it shows it already (``reload``:
        read again, a tab just opened: another tab may have changed it, unless it holds changes not applied); a
        tab's changes not applied (Edit, Search Index) are asked about first (applied, or dropped)."""
        if not self.panel_stack.isVisible():
            return
        uids = self._selected_uids()
        panel = self.panel_stack.currentWidget()
        unapplied = getattr(panel, "has_changes", lambda: False)()
        if not uids or (uids[0] == panel.track_uid() and (not reload or unapplied)):
            return
        if unapplied:
            title = next((t.title for t in self._all_tracks if t.uid == panel.track_uid()), "") or panel.track_uid()
            title = QtGui.QFontMetrics(self.font()).elidedText(title or "", QtCore.Qt.ElideMiddle, 420)
            answer = QtWidgets.QMessageBox.question(
                self, self.panel_tabs.tabText(self.panel_stack.currentIndex()),
                tr('Apply the changes to "{title}"? (No: they are dropped)').format(title=title),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
            if answer == QtWidgets.QMessageBox.Yes and not panel.apply():
                return  # not saved (said why): the panel keeps them
        row = self.library.snapshot.track(uids[0])
        if row is not None:
            panel.set_track(row)

    def track_panel_has_focus(self) -> bool:
        focus = QtWidgets.QApplication.focusWidget()
        return focus is not None and self.panel_stack.isAncestorOf(focus)

    # Edit
    def _edit_selected(self):
        self._open_tab_selected(self.EDIT_TAB)

    def _open_tab_selected(self, index: int) -> None:
        if self._selected_uids():
            self.open_track_panel(index)
