from __future__ import annotations
import os
from typing import Callable, Optional
from dataclasses import asdict, replace

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QLabel, QDialogButtonBox,
    QGroupBox, QComboBox, QPushButton, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)

from core import reading
from core.event_bus import EventBus
from core.library_handler import READING_FIELDS, USER_SOURCE, TrackRow, LibraryDB, field_readings
from ui.theme import current_theme
from core.config import config, default_cfg, searchconfig
from utils.labels import ALPHANUMERIC_LABELS, CLASSICAL_LABELS
from core.i18n import tr


class EditSongDialog(QDialog):
    """Dialog to edit TrackRow metadata.

    - Non-modal; intended to be reused like SettingsDialog/WorkersDialog.
    - Only mutable metadata is editable (title, artist, album, bpm, key, rating, comment)
      and the file path (a moved or renamed file; the uid keeps the track's analysis).
    - Immutable fields (added_ts, file_mtime, file_size, uid, duration) are shown read-only.

    Usage:
        dlg = EditSongDialog(bus, libpath)
        dlg.set_track(track_row)
        dlg.show()

    If `libpath` is provided, Apply/OK will persist to the library.db under that folder
    and emit bus.sig_lib_updated with the refreshed list. Otherwise, it emits
    `sig_track_saved(TrackRow)` for the caller to handle persistence.
    """

    sig_track_saved = Signal(object)  # TrackRow

    def __init__(self, bus: Optional[EventBus] = None, libpath: Optional[str] = None,
                 search_settings: Optional[Callable[[], searchconfig]] = None):
        """``search_settings``: the search settings (read when needed: the stored readings'
        languages; the defaults when None)."""
        super().__init__()
        self.setWindowTitle(tr("Edit Track Metadata"))
        self.setModal(False)
        self.resize(500, 300)
        self._bus = bus
        self._libpath = libpath
        self._search_settings = search_settings or (lambda: default_cfg().searchconfig)
        self._row: Optional[TrackRow] = None

        root = QVBoxLayout(self)

        # Editable fields
        grp_edit = QGroupBox(tr("Editable Fields"))
        form_edit = QFormLayout(grp_edit)
        self.ed_title = QLineEdit()
        self.ed_artist = QLineEdit()
        self.ed_album = QLineEdit()
        self.sp_bpm = QDoubleSpinBox(); self.sp_bpm.setRange(0.0, 1000.0); self.sp_bpm.setDecimals(3); self.sp_bpm.setSingleStep(0.1)
        # Human-readable key: show both Alphanumeric and Classical labels
        self.cmb_key = QComboBox(); self._populate_key_combo()
        self.sp_rating = QSpinBox(); self.sp_rating.setRange(0, 5)
        self.ed_comment = QLineEdit()
        self.ed_path = QLineEdit()
        self.ed_path.setToolTip(tr("The track's audio file. Set it when the file was moved or renamed;\n"
                                   "the analysis, playlists and CUEs stay with the track."))
        btn_browse = QPushButton(tr("Browse..."))
        btn_browse.clicked.connect(self._browse_path)
        path_row = QWidget(); path_lay = QHBoxLayout(path_row); path_lay.setContentsMargins(0, 0, 0, 0)
        path_lay.addWidget(self.ed_path, 1); path_lay.addWidget(btn_browse)

        # The readings (core.reading): how title / artist / album read in Latin letters, up to
        # READING_SLOTS each (Han: one per language) -- what the search compares. The ones typed here
        # are the user's (kept); the others (dim) are read from the text in the stored languages,
        # after them, but those emptied or typed over here (removed, kept so). A column per field,
        # a row per slot.
        self.tbl_readings = QTableWidget(reading.READING_SLOTS, len(READING_FIELDS))
        names = {"title": tr("Title"), "artist": tr("Artist"), "album": tr("Album")}
        self.tbl_readings.setHorizontalHeaderLabels([names.get(f, f.capitalize()) for f in READING_FIELDS])
        self.tbl_readings.setVerticalHeaderLabels([str(i + 1) for i in range(reading.READING_SLOTS)])
        self.tbl_readings.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_readings.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.tbl_readings.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.tbl_readings.setToolTip(
            tr("<p>How the title, artist and album read in Latin letters: the search finds the track by them "
               "in any script (あのね by anone or 아노네). Dim: read from the text in Settings > Search's "
               "languages. Type in a cell to add a reading of yours; empty a cell (Delete) or type over it to "
               "remove its reading for good, and type it again to bring it back.</p>"))
        self.tbl_readings.setFixedHeight(
            self.tbl_readings.horizontalHeader().sizeHint().height()
            + self.tbl_readings.verticalHeader().defaultSectionSize() * reading.READING_SLOTS + 4)
        self.tbl_readings.itemChanged.connect(self._on_reading_changed)
        self.tbl_readings.installEventFilter(self)                   # Delete / Backspace empty the selected cells
        self._reading_user = {f: [] for f in READING_FIELDS}         # the readings the user added
        self._reading_removed = {f: [] for f in READING_FIELDS}      # the automatic ones the user removed
        self._reading_touched = {f: False for f in READING_FIELDS}   # changed here (saved on apply)
        self._sources = {"title": self.ed_title, "artist": self.ed_artist, "album": self.ed_album}
        for f, edit in self._sources.items():
            edit.textChanged.connect(lambda _text, f=f: self._refresh_auto_readings(f))

        form_edit.addRow(tr("Path"), path_row)
        form_edit.addRow(tr("Title"), self.ed_title)
        form_edit.addRow(tr("Artist"), self.ed_artist)
        form_edit.addRow(tr("Album"), self.ed_album)
        form_edit.addRow(tr("Readings"), self.tbl_readings)
        form_edit.addRow("BPM", self.sp_bpm)
        form_edit.addRow(tr("Key"), self.cmb_key)
        form_edit.addRow(tr("Rating (0-5)"), self.sp_rating)
        form_edit.addRow(tr("Comment"), self.ed_comment)

        # Read-only fields
        grp_ro = QGroupBox(tr("Read-only Fields"))
        form_ro = QFormLayout(grp_ro)
        self.ro_uid = QLineEdit(); self._make_ro(self.ro_uid)
        self.ro_duration = QLineEdit(); self._make_ro(self.ro_duration)
        self.ro_added = QLineEdit(); self._make_ro(self.ro_added)
        self.ro_mtime = QLineEdit(); self._make_ro(self.ro_mtime)
        self.ro_size = QLineEdit(); self._make_ro(self.ro_size)

        form_ro.addRow("UID", self.ro_uid)
        form_ro.addRow(tr("Duration (sec)"), self.ro_duration)
        form_ro.addRow(tr("Added (epoch)"), self.ro_added)
        form_ro.addRow(tr("File mtime"), self.ro_mtime)
        form_ro.addRow(tr("File size (bytes)"), self.ro_size)

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
        self.ed_path.setText(row.path or "")
        stored = {f: [] for f in READING_FIELDS}
        removed = {f: [] for f in READING_FIELDS}
        if self._libpath:
            db = LibraryDB(os.path.join(self._libpath, "library.db"))
            db.connect()
            try:
                stored = db.readings(row.uid)
                removed = db.removed_readings(row.uid)
            finally:
                db.close()
        for f in READING_FIELDS:
            self._reading_touched[f] = False
            self._reading_user[f] = [text for text, src in stored.get(f, []) if src == USER_SOURCE]
            self._reading_removed[f] = list(removed.get(f, []))
            self._fill_readings(f, stored.get(f, []))

        # read-only
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
        row.path = self.ed_path.text().strip()
        return row

    def user_readings(self) -> dict:
        """The fields whose readings were changed here: field -> (the readings the user added, the
        automatic ones they removed)."""
        return {f: (list(self._reading_user[f]), list(self._reading_removed[f]))
                for f in READING_FIELDS if self._reading_touched[f]}

    # Readings
    _SOURCE_ROLE = Qt.UserRole        # a cell's source (USER_SOURCE: the user's)
    _SHOWN_ROLE = Qt.UserRole + 1     # its text as shown, before an edit

    def _fill_readings(self, field: str, readings: list) -> None:
        """Show (reading, source) pairs in the field's column: dim when read from the text."""
        c = READING_FIELDS.index(field)
        dim = QBrush(QColor(current_theme().text_dim))
        self.tbl_readings.blockSignals(True)
        for r in range(reading.READING_SLOTS):
            text, source = readings[r] if r < len(readings) else ("", "")
            item = QTableWidgetItem(text)
            item.setData(self._SOURCE_ROLE, source)
            item.setData(self._SHOWN_ROLE, text)
            item.setToolTip(f"{text}  ({tr(reading.SOURCES.get(source, source))})" if text else "")
            if source != USER_SOURCE:
                item.setForeground(dim)
            self.tbl_readings.setItem(r, c, item)
        self.tbl_readings.blockSignals(False)

    def _automatic_readings(self, field: str) -> list:
        s = self._search_settings()
        return reading.readings(self._sources[field].text(), s.db_languages, han_run=s.han_run())

    def _refresh_auto_readings(self, field: str) -> None:
        """The user's readings, then the ones read from the text (following it as it is typed) but
        the removed ones."""
        if self._row is None:
            return
        self._fill_readings(field, field_readings(
            self._reading_user[field], self._automatic_readings(field), self._reading_removed[field]))

    def _edit_reading(self, field: str, shown: str, source: str, text: str) -> None:
        """A cell's reading changed from ``shown`` (from ``source``) to ``text`` ("": emptied)."""
        user, removed = list(self._reading_user[field]), list(self._reading_removed[field])
        if source == USER_SOURCE and shown in user:
            i = user.index(shown)
            if text:
                user[i] = text
            else:
                del user[i]
            if shown in [r for r, _src in self._automatic_readings(field)]:
                removed.append(shown)           # also read from the text: removed too
        else:
            if shown:
                removed.append(shown)           # an automatic reading emptied or typed over
            if text:
                user.append(text)
        self._reading_user[field] = list(dict.fromkeys(user))
        self._reading_removed[field] = [r for r in dict.fromkeys(removed) if r not in self._reading_user[field]]
        self._reading_touched[field] = True

    def _on_reading_changed(self, item: QTableWidgetItem) -> None:
        field = READING_FIELDS[item.column()]
        shown = str(item.data(self._SHOWN_ROLE) or "")
        text = item.text().strip()
        if text != shown:
            self._edit_reading(field, shown, str(item.data(self._SOURCE_ROLE) or ""), text)
        self._refresh_auto_readings(field)

    def eventFilter(self, obj, event) -> bool:
        if (obj is self.tbl_readings and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Delete, Qt.Key_Backspace)):
            cells = [(READING_FIELDS[i.column()], str(i.data(self._SHOWN_ROLE) or ""), str(i.data(self._SOURCE_ROLE) or ""))
                     for i in self.tbl_readings.selectedItems()]
            for field, shown, source in cells:
                if shown:
                    self._edit_reading(field, shown, source, "")
            for field in dict.fromkeys(field for field, _shown, _source in cells):
                self._refresh_auto_readings(field)
            return True
        return super().eventFilter(obj, event)

    # Internal
    def _make_ro(self, w: QLineEdit) -> None:
        w.setReadOnly(True)
        w.setStyleSheet("QLineEdit[readOnly=\"true\"] { background: #2a2a2a; color: #bbb; }")

    def _browse_path(self) -> None:
        current = self.ed_path.text().strip()
        folder = os.path.dirname(current)
        start = current if os.path.isfile(current) else (folder if os.path.isdir(folder) else "")
        path, _ = QFileDialog.getOpenFileName(self, tr("Track File"), start)
        if path:
            self.ed_path.setText(os.path.normpath(path))

    def _persist_if_configured(self, updated: TrackRow) -> bool:
        """Save the row; False (with a message) when the new path cannot be used."""
        path_changed = self._row is not None and updated.path != self._row.path
        if path_changed and not os.path.isfile(updated.path):
            QMessageBox.warning(self, tr("Edit Track"), tr("The file does not exist:\n{path}").format(path=updated.path))
            return False
        if not self._libpath:
            # No persistence configured; emit signal for external handling
            self.sig_track_saved.emit(updated)
            return True
        s = self._search_settings()
        db = LibraryDB(os.path.join(self._libpath, "library.db"), reading_languages=s.db_languages,
                       han_run=s.han_run())
        db.connect()
        try:
            # the path first (checked: one file, one track; the file's size and mtime follow it),
            # then the metadata and the readings set here
            if path_changed:
                db.set_path(updated.uid, updated.path)
            db.upsert(replace(updated, file_mtime=None, file_size=None))
            for field, (added, removed) in self.user_readings().items():
                db.set_user_readings(updated.uid, field, added, removed)
            db.conn.commit()
            saved = db.get_by_uid(updated.uid)
            lib_full = db.list_all()
        except ValueError as exc:
            db.conn.rollback()
            QMessageBox.warning(self, tr("Edit Track"), str(exc))
            return False
        finally:
            db.close()
        if saved is not None:
            self.set_track(saved)
        if self._bus is not None:
            self._bus.sig_lib_updated.emit(lib_full)
            track_uid = str(updated.uid or "").strip()
            if track_uid:
                self._bus.sig_rekordbox_sync_track_requested.emit(track_uid)
        return True

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
        if self._persist_if_configured(updated):
            self.accept()

    # Key helpers
    def _populate_key_combo(self) -> None:
        # First item: Unknown/None
        self.cmb_key.addItem("--", None)
        for i in range(24):
            alphanumeric = ALPHANUMERIC_LABELS[i]
            classical = CLASSICAL_LABELS[i]
            text = f"{alphanumeric} / {classical}"
            self.cmb_key.addItem(text, i)

    def _set_key_index(self, idx: int) -> None:
        idx = int(idx) % 24
        # items: 0 is Unknown, so select idx+1
        self.cmb_key.setCurrentIndex(idx + 1)
