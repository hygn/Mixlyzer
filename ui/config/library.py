from PySide6 import QtCore
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ui.config.dialogs import RekordboxSyncProgressDialog


def build_library_tab(dialog) -> None:
    dialog.tab_lib = QWidget()
    form = QFormLayout(dialog.tab_lib)
    dialog.ed_libpath = QLineEdit()
    dialog.cb_write_log = QCheckBox("Write log to file")
    dialog.ed_logpath = QLineEdit()
    dialog.cb_rekordbox_sync = QCheckBox("Sync with Rekordbox XML")
    dialog.ed_rekordbox_xml_path = QLineEdit()
    dialog.btn_rekordbox_sync_now = QPushButton("Sync Now")
    dialog.lbl_log_note = QLabel("Logging settings apply on next app launch.")
    dialog.lbl_rekordbox_note = QLabel("Use Sync Now to rebuild the whole XML explicitly.")
    dialog.lbl_log_note.setWordWrap(True)
    dialog.lbl_rekordbox_note.setWordWrap(True)
    dialog.ed_libpath.setToolTip(
        "Folder holding library.db and the per-track feature files (.npz).\n"
        "Relative paths are resolved from the app folder."
    )
    dialog.cb_write_log.setToolTip("Write console output to a log file.")
    dialog.ed_logpath.setToolTip("Log file written when 'Write log to file' is on.")
    dialog.cb_rekordbox_sync.setToolTip(
        "Keep a Rekordbox XML file updated with the library's beat grids, keys and cues."
    )
    dialog.ed_rekordbox_xml_path.setToolTip(
        "Rekordbox XML file to write. Import it in Rekordbox to use the analysis."
    )
    dialog.btn_rekordbox_sync_now.setToolTip(
        "Save these settings and rebuild the whole XML from the library now."
    )

    grp_library = QGroupBox("Library")
    f_library = QFormLayout(grp_library)
    f_library.addRow("Library Path", dialog.ed_libpath)

    grp_rekordbox = QGroupBox("Rekordbox XML")
    f_rekordbox = QFormLayout(grp_rekordbox)
    f_rekordbox.addRow(dialog.cb_rekordbox_sync)
    f_rekordbox.addRow("XML Path", dialog.ed_rekordbox_xml_path)
    f_rekordbox.addRow("", dialog.btn_rekordbox_sync_now)
    f_rekordbox.addRow(dialog.lbl_rekordbox_note)

    grp_advanced = QGroupBox("Advanced")
    f_advanced = QFormLayout(grp_advanced)
    f_advanced.addRow(dialog.cb_write_log)
    f_advanced.addRow("Log Path", dialog.ed_logpath)
    f_advanced.addRow(dialog.lbl_log_note)

    form.addRow(grp_library)
    form.addRow(grp_rekordbox)
    form.addRow(grp_advanced)

    dialog.btn_rekordbox_sync_now.clicked.connect(dialog._sync_rekordbox_now)
    dialog.tabs.addTab(dialog.tab_lib, "Library")


class LibrarySettingsMixin:
    def _initialize_library_settings(self, bus) -> None:
        self._rekordbox_sync_dialog: RekordboxSyncProgressDialog | None = None
        self._rekordbox_progress_request = None
        self._accept_next_library_sync = False
        if bus is None:
            return
        bus.sig_rekordbox_sync_started.connect(self._on_rekordbox_sync_started)
        bus.sig_rekordbox_sync_progress.connect(self._on_rekordbox_sync_progress)
        bus.sig_rekordbox_sync_finished.connect(self._on_rekordbox_sync_finished)
        bus.sig_rekordbox_sync_failed.connect(self._on_rekordbox_sync_failed)

    def _sync_rekordbox_now(self) -> None:
        if self._bus is None:
            return
        cfg = self.get_config()
        if not cfg.libconfig.rekordbox_sync_enabled:
            QMessageBox.warning(
                self, "Sync Rekordbox XML", "Enable Sync with Rekordbox XML first."
            )
            return
        if not cfg.libconfig.rekordbox_xml_path.strip():
            QMessageBox.warning(
                self, "Sync Rekordbox XML", "Rekordbox XML Path is empty."
            )
            return
        previous_libcfg = getattr(self._current_cfg, "libconfig", None)
        sync_config_changed = (
            previous_libcfg is None
            or bool(previous_libcfg.rekordbox_sync_enabled)
            != bool(cfg.libconfig.rekordbox_sync_enabled)
            or str(previous_libcfg.rekordbox_xml_path).strip()
            != str(cfg.libconfig.rekordbox_xml_path).strip()
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
            dialog.set_progress(0, "Rekordbox XML worker started")
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
