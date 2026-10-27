import os

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QFileDialog,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QWidget,
)

from ui.config.dialogs import LibrarySyncProgressDialog, RekordboxSyncProgressDialog
from ui.config.scroll import groups_page
from system.i18n import tr


def build_library_tab(dialog) -> None:
    """Settings > Library: its pages (General, Database, Rekordbox XML, Advanced)."""
    dialog.tab_lib = QWidget()
    dialog.ed_libpath = QtWidgets.QLineEdit()
    dialog.list_local_music_directories = QListWidget()
    dialog.list_local_music_directories.setSelectionMode(
        QtWidgets.QAbstractItemView.ExtendedSelection
    )
    dialog.list_local_music_directories.setMinimumHeight(100)
    dialog.btn_add_local_music_directory = QPushButton(tr("Add Folder…"))
    dialog.btn_remove_local_music_directory = QPushButton(tr("Remove"))
    dialog.lbl_local_music_directory_note = QLabel(
        tr("When this list is empty, the operating system's default Music folder is used.")
    )
    dialog.lbl_local_music_directory_note.setWordWrap(True)
    dialog.cb_write_log = QCheckBox(tr("Write log to file"))
    dialog.ed_logpath = QLineEdit()
    dialog.cb_rekordbox_sync = QCheckBox(tr("Sync with Rekordbox XML"))
    dialog.ed_rekordbox_xml_path = QLineEdit()
    dialog.sp_rekordbox_ramp_tolerance_ms = QDoubleSpinBox()
    dialog.sp_rekordbox_ramp_tolerance_ms.setRange(0.0, 20.0)
    dialog.sp_rekordbox_ramp_tolerance_ms.setDecimals(1)
    dialog.sp_rekordbox_ramp_tolerance_ms.setSingleStep(0.5)
    dialog.sp_rekordbox_ramp_tolerance_ms.setSuffix(" ms")
    dialog.sp_rekordbox_ramp_tolerance_ms.setToolTip(tr(
        "<p>Rekordbox has no tempo ramps: a ramp is written as steps of steady tempo. Smaller makes the steps "
        "finer.</p>"
    ))
    dialog.cb_rekordbox_comment_tieups = QCheckBox(tr("Tie-ups after the comment"))
    dialog.cb_rekordbox_comment_tieups.setToolTip(tr(
        "<p>Each track's comment in the XML with its song's tie-ups after it, as [Title OP; Other ED]; the "
        "library's comment stays as it is.</p>"))
    dialog.btn_rekordbox_sync_now = QPushButton(tr("Sync Now"))
    dialog.lbl_log_note = QLabel(tr("Logging settings apply on next app launch."))
    dialog.lbl_rekordbox_note = QLabel(tr("Use Sync Now to rebuild the whole XML explicitly."))
    dialog.lbl_log_note.setWordWrap(True)
    dialog.lbl_rekordbox_note.setWordWrap(True)
    dialog.ed_libpath.setToolTip(
        tr("Folder holding library.db and the per-track feature files (.npz).")
    )
    dialog.list_local_music_directories.setToolTip(
        tr("Root folders shown under Local Folders in the library browser.")
    )
    dialog.btn_add_local_music_directory.setToolTip(tr("Add a local music root folder."))
    dialog.btn_remove_local_music_directory.setToolTip(tr("Remove the selected root folders."))
    dialog.cb_write_log.setToolTip(tr("Write console output to a log file."))
    dialog.ed_logpath.setToolTip(tr("Log file written when 'Write log to file' is on."))
    dialog.cb_rekordbox_sync.setToolTip(
        tr("Keep a Rekordbox XML file updated with the library's beat grids, keys and cues.")
    )
    dialog.ed_rekordbox_xml_path.setToolTip(
        tr("Rekordbox XML file to write. Import it in Rekordbox to use the analysis.")
    )
    dialog.btn_rekordbox_sync_now.setToolTip(
        tr("Save these settings and rebuild the whole XML from the library now.")
    )

    grp_library = QGroupBox(tr("Library"))
    f_library = QFormLayout(grp_library)
    f_library.addRow(tr("Library Path"), dialog.ed_libpath)
    local_box = QWidget()
    local_layout = QtWidgets.QVBoxLayout(local_box)
    local_layout.setContentsMargins(0, 0, 0, 0)
    local_layout.addWidget(dialog.list_local_music_directories)
    local_buttons = QtWidgets.QHBoxLayout()
    local_buttons.addWidget(dialog.btn_add_local_music_directory)
    local_buttons.addWidget(dialog.btn_remove_local_music_directory)
    local_buttons.addStretch(1)
    local_layout.addLayout(local_buttons)
    local_layout.addWidget(dialog.lbl_local_music_directory_note)
    f_library.addRow(tr("Local Music Folders"), local_box)
    dialog.cb_tieup_to_versions = QCheckBox(tr("Add a tie-up to the track's versions too"))
    dialog.cb_tieup_to_versions.setToolTip(tr(
        "<p>A tie-up added in the Edit tab is added to the track's other versions as well.</p>"))
    f_library.addRow(dialog.cb_tieup_to_versions)
    dialog.cb_tieup_web_search = QCheckBox(tr("Look up a new track's tie-ups on Wikipedia"))
    dialog.cb_tieup_web_search.setToolTip(tr(
        "<p>After a new track's analysis, its tie-ups are looked up on Japanese Wikipedia and Wikidata and the ones "
        "found for certain are added. Its title and artist are sent there.</p>"))
    f_library.addRow(dialog.cb_tieup_web_search)

    # library.db and the stored analyses ({uid}.npz) kept in agreement (core.library.jobs.save_edits / sync_index)
    dialog.cb_sync_stored_analysis = QCheckBox(tr("Update the library database when the editor saves"))
    dialog.cb_sync_stored_analysis.setToolTip(tr(
        "<p>The editor's save also writes the track's BPM, key and search tables into the library database.</p>"))
    dialog.btn_library_sync_now = QPushButton(tr("Sync Now"))
    dialog.btn_library_sync_now.setToolTip(tr(
        "<p>Every track's BPM, key and search tables in the library database made again from its stored analysis. A "
        "BPM or key typed in the Edit Track tab is replaced too. Titles, artists, ratings and comments are not "
        "touched.</p>"))
    grp_database = QGroupBox(tr("Library Database"))
    f_database = QFormLayout(grp_database)
    f_database.addRow(dialog.cb_sync_stored_analysis)
    f_database.addRow("", dialog.btn_library_sync_now)

    # the library packed into one file and restored from one (core.workflows.library_backup)
    dialog.cb_backup_feature_cache = QCheckBox(tr("Include the feature caches"))
    dialog.cb_backup_feature_cache.setToolTip(tr(
        "<p>A backup also packs the analyses' feature caches (Reoptimize's training data). Without them a backup is "
        "smaller.</p>"))
    dialog.btn_backup_library = QPushButton(tr("Back Up Library…"))
    dialog.btn_backup_library.setToolTip(tr(
        "<p>The library packed into one file: its database, every track's stored analysis, the thumbnails, the "
        "search indexes, the settings, your shortcuts, MIDI mapping and retrained models. The music files "
        "themselves are not packed.</p>"))
    dialog.btn_restore_library = QPushButton(tr("Restore Library…"))
    dialog.btn_restore_library.setToolTip(tr(
        "<p>A backup unpacked into a new library folder, then Mixlyzer restarts on it. The library in use now is "
        "kept as it is.</p>"))
    backup_buttons = QWidget()
    backup_row = QtWidgets.QHBoxLayout(backup_buttons)
    backup_row.setContentsMargins(0, 0, 0, 0)
    backup_row.addWidget(dialog.btn_backup_library)
    backup_row.addWidget(dialog.btn_restore_library)
    backup_row.addStretch(1)
    grp_backup = QGroupBox(tr("Backup"))
    f_backup = QFormLayout(grp_backup)
    f_backup.addRow(dialog.cb_backup_feature_cache)
    f_backup.addRow(backup_buttons)

    grp_rekordbox = QGroupBox(tr("Rekordbox XML"))
    f_rekordbox = QFormLayout(grp_rekordbox)
    f_rekordbox.addRow(dialog.cb_rekordbox_sync)
    f_rekordbox.addRow(tr("XML Path"), dialog.ed_rekordbox_xml_path)
    f_rekordbox.addRow(tr("Tempo ramp tolerance"), dialog.sp_rekordbox_ramp_tolerance_ms)
    f_rekordbox.addRow(dialog.cb_rekordbox_comment_tieups)
    f_rekordbox.addRow("", dialog.btn_rekordbox_sync_now)
    f_rekordbox.addRow(dialog.lbl_rekordbox_note)

    grp_advanced = QGroupBox(tr("Advanced"))
    f_advanced = QFormLayout(grp_advanced)
    f_advanced.addRow(dialog.cb_write_log)
    f_advanced.addRow(tr("Log Path"), dialog.ed_logpath)
    f_advanced.addRow(dialog.lbl_log_note)

    dialog.library_tabs = QTabWidget(dialog.tab_lib)
    dialog.library_tabs.addTab(groups_page(grp_library), tr("General"))
    dialog.library_tabs.addTab(groups_page(grp_database, grp_backup), tr("Database"))
    dialog.library_tabs.addTab(groups_page(grp_rekordbox), tr("Rekordbox XML"))
    dialog.library_tabs.addTab(groups_page(grp_advanced), tr("Advanced"))
    root = QtWidgets.QVBoxLayout(dialog.tab_lib)
    root.addWidget(dialog.library_tabs)

    dialog.btn_rekordbox_sync_now.clicked.connect(dialog._sync_rekordbox_now)
    dialog.btn_library_sync_now.clicked.connect(dialog._sync_library_now)
    dialog.btn_backup_library.clicked.connect(dialog._backup_library)
    dialog.btn_restore_library.clicked.connect(dialog._restore_library)
    dialog.btn_add_local_music_directory.clicked.connect(dialog._browse_local_music_directory)
    dialog.btn_remove_local_music_directory.clicked.connect(
        dialog._remove_local_music_directories
    )
    dialog.tabs.addTab(dialog.tab_lib, tr("Library"))


class LibrarySettingsMixin:
    def _browse_local_music_directory(self) -> None:
        current = self.list_local_music_directories.currentItem()
        if current is not None:
            initial = current.text()
        else:
            initial = QtCore.QStandardPaths.writableLocation(
                QtCore.QStandardPaths.MusicLocation
            ) or str(os.path.join(os.path.expanduser("~"), "Music"))
        directory = QFileDialog.getExistingDirectory(
            self, tr("Add Local Music Folder"), os.path.expanduser(initial)
        )
        if directory:
            identity = os.path.normcase(os.path.realpath(os.path.abspath(directory)))
            existing = {
                os.path.normcase(os.path.realpath(os.path.abspath(
                    self.list_local_music_directories.item(row).text()
                )))
                for row in range(self.list_local_music_directories.count())
            }
            if identity not in existing:
                item = QtWidgets.QListWidgetItem(directory)
                item.setToolTip(directory)
                self.list_local_music_directories.addItem(item)

    def _remove_local_music_directories(self) -> None:
        for item in self.list_local_music_directories.selectedItems():
            self.list_local_music_directories.takeItem(
                self.list_local_music_directories.row(item)
            )

    def _set_local_music_directories(self, directories) -> None:
        self.list_local_music_directories.clear()
        for directory in directories or []:
            text = str(directory).strip()
            if not text:
                continue
            item = QtWidgets.QListWidgetItem(text)
            item.setToolTip(text)
            self.list_local_music_directories.addItem(item)

    def _get_local_music_directories(self) -> list[str]:
        return [
            self.list_local_music_directories.item(row).text().strip()
            for row in range(self.list_local_music_directories.count())
            if self.list_local_music_directories.item(row).text().strip()
        ]

    def _initialize_library_settings(self, bus) -> None:
        self._rekordbox_sync_dialog: RekordboxSyncProgressDialog | None = None
        self._rekordbox_progress_request = None
        self._accept_next_library_sync = False
        self._library_sync_dialog: LibrarySyncProgressDialog | None = None
        if bus is None:
            self.btn_library_sync_now.setEnabled(False)
            self.btn_backup_library.setEnabled(False)
            self.btn_restore_library.setEnabled(False)
            return
        bus.sig_library_sync_progress.connect(self._on_library_sync_progress)
        bus.sig_library_sync_finished.connect(self._on_library_sync_finished)
        bus.sig_library_sync_failed.connect(self._on_library_sync_failed)
        bus.sig_rekordbox_sync_started.connect(self._on_rekordbox_sync_started)
        bus.sig_rekordbox_sync_progress.connect(self._on_rekordbox_sync_progress)
        bus.sig_rekordbox_sync_finished.connect(self._on_rekordbox_sync_finished)
        bus.sig_rekordbox_sync_failed.connect(self._on_rekordbox_sync_failed)

    def _sync_library_now(self) -> None:
        """Sync Now: library.db made to agree with the stored analyses (core.workflows.library_sync), shown as it
        runs; the settings on the page play no part in it, so nothing is saved first. Asked first (No by default):
        it replaces every BPM and key typed in the Edit Track tab too."""
        if self._bus is None:
            return
        answer = QMessageBox.warning(
            self, tr("Sync Library Database"),
            tr("Every track's BPM, key and search tables in the library database will be made again from its "
               "stored analysis. A BPM or key typed in the Edit Track tab is replaced too, and this cannot be "
               "undone.\n\nSync now?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        if self._library_sync_dialog is not None:
            self._library_sync_dialog.deleteLater()
        self._library_sync_dialog = LibrarySyncProgressDialog(self)
        self._library_sync_dialog.show()
        self.btn_library_sync_now.setEnabled(False)
        self._bus.sig_library_sync_requested.emit()

    def _backup_library(self) -> None:
        """Back Up Library…: the window asks where and writes it (the checkbox as it is now, saved or not)."""
        if self._bus is not None:
            self._bus.sig_library_backup_requested.emit(bool(self.cb_backup_feature_cache.isChecked()))

    def _restore_library(self) -> None:
        if self._bus is not None:
            self._bus.sig_library_restore_requested.emit()

    @QtCore.Slot(int, int)
    def _on_library_sync_progress(self, done: int, total: int) -> None:
        if self._library_sync_dialog is not None:
            self._library_sync_dialog.set_progress(done, total)

    @QtCore.Slot(object)
    def _on_library_sync_finished(self, result) -> None:
        self.btn_library_sync_now.setEnabled(True)
        if self._library_sync_dialog is not None:
            self._library_sync_dialog.set_finished(result)

    @QtCore.Slot(str)
    def _on_library_sync_failed(self, message: str) -> None:
        self.btn_library_sync_now.setEnabled(True)
        if self._library_sync_dialog is not None:
            self._library_sync_dialog.set_failed(message)

    def _sync_rekordbox_now(self) -> None:
        if self._bus is None:
            return
        cfg = self.get_config()
        if not cfg.libconfig.rekordbox_sync_enabled:
            QMessageBox.warning(
                self, tr("Sync Rekordbox XML"), tr("Enable Sync with Rekordbox XML first.")
            )
            return
        if not cfg.libconfig.rekordbox_xml_path.strip():
            QMessageBox.warning(
                self, tr("Sync Rekordbox XML"), tr("Rekordbox XML Path is empty.")
            )
            return
        previous_libcfg = getattr(self._current_cfg, "libconfig", None)
        sync_config_changed = (
            previous_libcfg is None
            or bool(previous_libcfg.rekordbox_sync_enabled)
            != bool(cfg.libconfig.rekordbox_sync_enabled)
            or str(previous_libcfg.rekordbox_xml_path).strip()
            != str(cfg.libconfig.rekordbox_xml_path).strip()
            or float(previous_libcfg.rekordbox_ramp_tolerance_ms)
            != float(cfg.libconfig.rekordbox_ramp_tolerance_ms)
            or bool(previous_libcfg.rekordbox_comment_tieups)
            != bool(cfg.libconfig.rekordbox_comment_tieups)
        )
        if self._rekordbox_sync_dialog is not None:
            self._rekordbox_sync_dialog.deleteLater()
        self._rekordbox_sync_dialog = RekordboxSyncProgressDialog(self)
        self._rekordbox_sync_dialog.show()
        self.btn_rekordbox_sync_now.setEnabled(False)
        self._rekordbox_progress_request = None
        self._accept_next_library_sync = sync_config_changed
        self.saveJsonRequested.emit(cfg)
        if not sync_config_changed:
            self._bus.sig_rekordbox_sync_requested.emit(
                {"full_rebuild": True, "show_progress": True}
            )

    @QtCore.Slot(object)
    def _on_rekordbox_sync_started(self, request) -> None:
        show_progress = bool(getattr(request, "show_progress", False))
        accepts_saved_config_sync = (
            self._accept_next_library_sync
            and getattr(request, "mode", "") == "library"
        )
        if not show_progress and not accepts_saved_config_sync:
            return
        self._accept_next_library_sync = False
        self._rekordbox_progress_request = request
        dialog = self._rekordbox_sync_dialog
        if dialog is not None:
            dialog.set_progress(0, tr("Rekordbox XML worker started"))
            dialog.show()
            dialog.raise_()

    @QtCore.Slot(object, int, str)
    def _on_rekordbox_sync_progress(
        self, request, value: int, message: str
    ) -> None:
        if request is not self._rekordbox_progress_request:
            return
        dialog = self._rekordbox_sync_dialog
        if dialog is not None:
            dialog.set_progress(value, message)

    @QtCore.Slot(object, object)
    def _on_rekordbox_sync_finished(self, request, result) -> None:
        if request is not self._rekordbox_progress_request:
            return
        self.btn_rekordbox_sync_now.setEnabled(True)
        dialog = self._rekordbox_sync_dialog
        if dialog is not None:
            dialog.set_finished(result)
        self._rekordbox_progress_request = None

    @QtCore.Slot(object, str)
    def _on_rekordbox_sync_failed(self, request, message: str) -> None:
        if request is not self._rekordbox_progress_request:
            return
        self.btn_rekordbox_sync_now.setEnabled(True)
        dialog = self._rekordbox_sync_dialog
        if dialog is not None:
            dialog.set_failed(message)
        self._rekordbox_progress_request = None
