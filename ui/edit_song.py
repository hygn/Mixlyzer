from __future__ import annotations
from typing import Optional
from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QLabel, QDialogButtonBox,
    QGroupBox, QComboBox
)

from core.event_bus import EventBus
from core.library_handler import TrackRow, LibraryDB
from core.config import config
from utils.labels import CAMELOT_LABELS, CLASSICAL_LABELS


class EditSongDialog(QDialog):
    """Dialog to edit TrackRow metadata.

    - Non-modal; intended to be reused like SettingsDialog/WorkersDialog.
    - Only mutable metadata is editable (title, artist, album, bpm, key, rating, comment).
    - Immutable fields (added_ts, file_mtime, file_size, path, uid, duration) are shown read-only.

    Usage:
        dlg = EditSongDialog(bus, libpath)
        dlg.set_track(track_row)
        dlg.show()

    If `libpath` is provided, Apply/OK will persist to the library.db under that folder
    and emit bus.sig_lib_updated with the refreshed list. Otherwise, it emits
    `sig_track_saved(TrackRow)` for the caller to handle persistence.
    """

    sig_track_saved = Signal(object)  # TrackRow

    def __init__(self, bus: Optional[EventBus] = None, libpath: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Edit Track Metadata")
        self.setModal(False)
        self.resize(500, 300)
        self._bus = bus
        self._libpath = libpath
        self._row: Optional[TrackRow] = None

        root = QVBoxLayout(self)

        # Editable fields
        grp_edit = QGroupBox("Editable Fields")
        form_edit = QFormLayout(grp_edit)
        self.ed_title = QLineEdit()
        self.ed_artist = QLineEdit()
        self.ed_album = QLineEdit()
        self.sp_bpm = QDoubleSpinBox(); self.sp_bpm.setRange(0.0, 1000.0); self.sp_bpm.setDecimals(3); self.sp_bpm.setSingleStep(0.1)
        # Human-readable key: show both Camelot and Classical labels
        self.cmb_key = QComboBox(); self._populate_key_combo()
        self.sp_rating = QSpinBox(); self.sp_rating.setRange(0, 5)
        self.ed_comment = QLineEdit()

        form_edit.addRow("Title", self.ed_title)
        form_edit.addRow("Artist", self.ed_artist)
        form_edit.addRow("Album", self.ed_album)
        form_edit.addRow("BPM", self.sp_bpm)
        form_edit.addRow("Key", self.cmb_key)
        form_edit.addRow("Rating (0-5)", self.sp_rating)
        form_edit.addRow("Comment", self.ed_comment)

        # Read-only fields
        grp_ro = QGroupBox("Read-only Fields")
        form_ro = QFormLayout(grp_ro)
        self.ro_path = QLineEdit(); self._make_ro(self.ro_path)
        self.ro_uid = QLineEdit(); self._make_ro(self.ro_uid)
        self.ro_duration = QLineEdit(); self._make_ro(self.ro_duration)
        self.ro_added = QLineEdit(); self._make_ro(self.ro_added)
        self.ro_mtime = QLineEdit(); self._make_ro(self.ro_mtime)
        self.ro_size = QLineEdit(); self._make_ro(self.ro_size)

        form_ro.addRow("Path", self.ro_path)
        form_ro.addRow("UID", self.ro_uid)
        form_ro.addRow("Duration (sec)", self.ro_duration)
        form_ro.addRow("Added (epoch)", self.ro_added)
        form_ro.addRow("File mtime", self.ro_mtime)
        form_ro.addRow("File size (bytes)", self.ro_size)

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)

        root.addWidget(grp_edit)
        root.addWidget(grp_ro)
        root.addWidget(btn_box)

    # Public API
    def set_track(self, row: TrackRow) -> None:
        """Populate dialog with an existing TrackRow."""
        self._row = row
        # editable
        self.ed_title.setText(row.title or "")
        self.ed_artist.setText(row.artist or "")
        self.ed_album.setText(row.album or "")
        if isinstance(row.bpm, (int, float)):
            self.sp_bpm.setValue(float(row.bpm))
        else:
            self.sp_bpm.clear()  # shows 0; user may set
        if isinstance(row.key, int):
            self._set_key_index(int(row.key))
        else:
            self.cmb_key.setCurrentIndex(0)  # Unknown
        self.sp_rating.setValue(int(row.rating or 0))
        self.ed_comment.setText(row.comment or "")

        # read-only
        self.ro_path.setText(row.path or "")
        self.ro_uid.setText(row.uid or "")
        self.ro_duration.setText("" if row.duration is None else f"{float(row.duration):.3f}")
        self.ro_added.setText(str(int(row.added_ts or 0)))
        self.ro_mtime.setText("" if row.file_mtime is None else str(float(row.file_mtime)))
        self.ro_size.setText(str(int(row.file_size or 0)))

    def get_updated_row(self) -> Optional[TrackRow]:
        if self._row is None:
            return None
        # Create a copy with mutable fields updated
        row = TrackRow(**asdict(self._row))
        row.title = self.ed_title.text().strip()
        row.artist = self.ed_artist.text().strip()
        row.album = self.ed_album.text().strip()
        # BPM: treat 0 as None to avoid writing meaningless zero
        bpm_val = float(self.sp_bpm.value())
        row.bpm = None if bpm_val == 0.0 else bpm_val
        kd = self.cmb_key.currentData()
        row.key = int(kd) if kd is not None else None
        row.rating = int(self.sp_rating.value())
        row.comment = self.ed_comment.text().strip()
        return row

    # Internal
    def _make_ro(self, w: QLineEdit) -> None:
        w.setReadOnly(True)
        w.setStyleSheet("QLineEdit[readOnly=\"true\"] { background: #2a2a2a; color: #bbb; }")

    def _persist_if_configured(self, updated: TrackRow) -> None:
        if not self._libpath:
            # No persistence configured; emit signal for external handling
            self.sig_track_saved.emit(updated)
            return
        try:
            import os
            db = LibraryDB(os.path.join(self._libpath, "library.db"))
            db.connect()
            db.upsert(updated)
            db.conn.commit()
            lib_full = db.list_all()
            db.close()
            if self._bus is not None:
                self._bus.sig_lib_updated.emit(lib_full)
        except Exception as e:
            # For now, surface the error by re-raising; caller can catch via Qt
            raise

    def _on_apply(self):
        updated = self.get_updated_row()
        if updated is None:
            return
        self._persist_if_configured(updated)

    def _on_ok(self):
        updated = self.get_updated_row()
        if updated is None:
            self.reject()
            return
        self._persist_if_configured(updated)
        self.accept()

    # Key helpers
    def _populate_key_combo(self) -> None:
        # First item: Unknown/None
        self.cmb_key.addItem("--", None)
        for i in range(24):
            camel = CAMELOT_LABELS[i]
            classical = CLASSICAL_LABELS[i]
            text = f"{camel} / {classical}"
            self.cmb_key.addItem(text, i)

    def _set_key_index(self, idx: int) -> None:
        idx = int(idx) % 24
        # items: 0 is Unknown, so select idx+1
        self.cmb_key.setCurrentIndex(idx + 1)
