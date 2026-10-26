from __future__ import annotations
import os
from typing import Optional
from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDoubleSpinBox, QDialogButtonBox,
    QGroupBox, QComboBox, QPushButton, QFileDialog, QMessageBox,
    QPlainTextEdit, QToolButton
)

from core.workflows.event_bus import EventBus
from core.library import jobs
from core.library.records import TrackRow
from core.library.service import LibraryService
from ui.config.scroll import scroll_page
from ui.theme import current_theme
from analyzer.music.key_names import ALPHANUMERIC_LABELS, CLASSICAL_LABELS
from system.i18n import tr, tr_n

COMMENT_LINES = 6   # the comment field's height in lines


class RatingStars(QWidget):
    """Five star buttons: a click sets the rating to its star; the set star clicked again clears it (0)."""

    valueChanged = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._value = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._stars: list[QToolButton] = []
        for n in range(1, 6):
            star = QToolButton()
            star.setAutoRaise(True)
            star.setCursor(Qt.PointingHandCursor)
            font = star.font()
            font.setPointSizeF(font.pointSizeF() * 1.4)
            star.setFont(font)
            star.setToolTip(tr_n("{count} star (click it again to clear the rating)",
                                 "{count} stars (click the last one again to clear the rating)", n).format(count=n))
            star.clicked.connect(lambda _checked=False, n=n: self._clicked(n))
            layout.addWidget(star)
            self._stars.append(star)
        layout.addStretch(1)
        self._refresh()

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        value = max(0, min(5, int(value or 0)))
        if value != self._value:
            self._value = value
            self._refresh()
            self.valueChanged.emit(value)

    def _clicked(self, n: int) -> None:
        self.setValue(0 if n == self._value else n)

    def _refresh(self) -> None:
        theme = current_theme()
        for n, star in enumerate(self._stars, start=1):
            lit = n <= self._value
            star.setText("★" if lit else "☆")
            star.setStyleSheet(f"QToolButton {{ color: {theme.accent if lit else theme.text_dim}; padding: 0 1px; }}")


class EditTrackPanel(QWidget):
    """The library panel's Edit tab: a track's metadata (TrackRow). Its readings and audio fingerprints are the
    Search Index tab's (ui.track_search_index), its tie-ups the Tie-ups tab's (ui.tieup_editor.TieupPanel).

    - Only mutable metadata is editable (title, artist, album, bpm, key, rating, comment)
      and the file path (a moved or renamed file; the uid keeps the track's analysis).
    - Immutable fields (added_ts, file_mtime, file_size, uid, duration) are shown read-only.
    - The fields scroll; Apply (save) and Reset (the stored values again) stay under them.

    If `libpath` is provided, Apply persists to the library.db under that folder
    and emits bus.sig_lib_updated with the refreshed list. Otherwise, it emits
    `sig_track_saved(TrackRow)` for the caller to handle persistence.
    """

    sig_track_saved = Signal(object)  # TrackRow

    def __init__(self, bus: Optional[EventBus] = None, library: Optional[LibraryService] = None):
        """``library``: where the changes are written (core.library.service); None: only sig_track_saved."""
        super().__init__()
        self._bus = bus
        self._library = library
        self._row: Optional[TrackRow] = None

        body = QWidget()
        content = QVBoxLayout(body)

        # Editable fields
        grp_edit = QGroupBox(tr("Editable Fields"))
        form_edit = QFormLayout(grp_edit)
        self.ed_title = QLineEdit()
        self.ed_artist = QLineEdit()
        self.ed_album = QLineEdit()
        self.sp_bpm = QDoubleSpinBox(); self.sp_bpm.setRange(0.0, 1000.0); self.sp_bpm.setDecimals(3); self.sp_bpm.setSingleStep(0.1)
        # Human-readable key: show both Alphanumeric and Classical labels
        self.cmb_key = QComboBox(); self._populate_key_combo()
        self.rating_stars = RatingStars()
        self.ed_comment = QPlainTextEdit()
        self.ed_comment.setTabChangesFocus(True)   # Tab goes on to the next field, as in the others
        margins = 2 * (self.ed_comment.frameWidth() + int(self.ed_comment.document().documentMargin()))
        self.ed_comment.setFixedHeight(QFontMetrics(self.ed_comment.font()).lineSpacing() * COMMENT_LINES + margins)
        self.ed_path = QLineEdit()
        self.ed_path.setToolTip(tr("The track's audio file. Set it when the file was moved or renamed;\n"
                                   "the analysis, playlists and CUEs stay with the track."))
        btn_browse = QPushButton(tr("Browse..."))
        btn_browse.clicked.connect(self._browse_path)
        path_row = QWidget(); path_lay = QHBoxLayout(path_row); path_lay.setContentsMargins(0, 0, 0, 0)
        path_lay.addWidget(self.ed_path, 1); path_lay.addWidget(btn_browse)

        form_edit.addRow(tr("Path"), path_row)
        form_edit.addRow(tr("Title"), self.ed_title)
        form_edit.addRow(tr("Artist"), self.ed_artist)
        form_edit.addRow(tr("Album"), self.ed_album)
        form_edit.addRow("BPM", self.sp_bpm)
        form_edit.addRow(tr("Key"), self.cmb_key)
        form_edit.addRow(tr("Rating"), self.rating_stars)
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

        # Buttons: Apply saves, Reset shows the stored values again
        btn_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Reset)
        btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        btn_box.button(QDialogButtonBox.Reset).clicked.connect(self._on_reset)

        content.addWidget(grp_edit)
        content.addWidget(grp_ro)
        content.addStretch(1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll_page(body), 1)
        root.addWidget(btn_box)

    # Public API
    def track_uid(self) -> Optional[str]:
        return self._row.uid if self._row is not None else None

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
        self.rating_stars.setValue(int(row.rating or 0))
        self.ed_comment.setPlainText(row.comment or "")
        self.ed_path.setText(row.path or "")

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
        row.rating = int(self.rating_stars.value())
        row.comment = self.ed_comment.toPlainText().strip()
        row.path = self.ed_path.text().strip()
        return row

    def has_changes(self) -> bool:
        """Whether the fields differ from the track's stored values (Apply not pressed yet)."""
        if self._row is None:
            return False
        updated = self.get_updated_row()

        def norm(name: str, value):
            if name == "bpm":
                return round(float(value), 3) if value else None
            if name == "rating":
                return int(value or 0)
            if name == "key":
                return value
            return str(value or "").strip()

        fields = ("title", "artist", "album", "bpm", "key", "rating", "comment", "path")
        return any(norm(f, getattr(updated, f)) != norm(f, getattr(self._row, f)) for f in fields)

    def apply(self) -> bool:
        """Save the fields (as Apply); False when they cannot be (said in a message)."""
        updated = self.get_updated_row()
        return updated is None or self._persist_if_configured(updated)

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
        if self._library is None:
            # No persistence configured; emit signal for external handling
            self.sig_track_saved.emit(updated)
            return True

        def written(saved: Optional[TrackRow]) -> None:
            # shown as stored, unless another track is shown or the fields were changed again meanwhile
            if saved is not None and self._row is not None and self._row.uid == saved.uid and not self.has_changes():
                self.set_track(saved)
            if self._bus is not None and updated.uid:
                self._bus.sig_rekordbox_sync_track_requested.emit(str(updated.uid))

        def refused(exc: Exception) -> None:
            # the path another track's (one file, one track): said, the fields kept as changes not applied
            if self._row is not None and self._row.uid == updated.uid:
                self._row = before
            QMessageBox.warning(self, tr("Edit Track"), str(exc))

        # written by the library's owner (the path first, then the metadata: jobs.save_track); meanwhile the fields
        # count as stored (has_changes: none)
        before, self._row = self._row, updated
        self._library.write(jobs.save_track, updated, path_changed, done=written, failed=refused)
        return True

    def _on_apply(self):
        updated = self.get_updated_row()
        if updated is None:
            return
        self._persist_if_configured(updated)

    def _on_reset(self):
        if self._row is not None:
            self.set_track(self._row)

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
