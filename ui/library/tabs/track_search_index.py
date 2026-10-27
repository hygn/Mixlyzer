"""The track panel's Search Index tab: what the library's searches find the track by. Its readings (core.search.reading: how
its title, artist and album read in Latin letters; the user's added or the automatic ones removed, saved by Apply), and
its audio fingerprints (the Variants' and Search by Music's), rebuilt on demand: the track dropped from both indexes,
then Index Now indexes it again (core.workflows.music_index: its audio decoded in a worker process)."""
from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QDialogButtonBox, QFormLayout, QGroupBox, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from core.search import reading
from core.settings.config import default_cfg, searchconfig
from core.workflows.event_bus import EventBus
from system.i18n import tr
from core.library import jobs
from core.library.records import READING_FIELDS, USER_SOURCE, TrackRow, field_readings
from core.library.service import LibraryService
from ui.config.scroll import scroll_page
from ui.theme import current_theme


class SearchIndexPanel(QWidget):
    """A track's readings and audio fingerprints (the module)."""

    def __init__(self, bus: Optional[EventBus] = None, library: Optional[LibraryService] = None,
                 search_settings: Optional[Callable[[], searchconfig]] = None):
        """``library``: where the readings are read and written (core.library.service)."""
        super().__init__()
        self._bus = bus
        self._library = library
        self._search_settings = search_settings or (lambda: default_cfg().searchconfig)
        self._row: Optional[TrackRow] = None
        self._rebuilding: Optional[str] = None    # the track dropped, waiting for Index Now
        self._asked_again = False                 # Index Now asked a second time (it was running another run)
        self._checking = False                    # Index Now done: whether the track was indexed being read

        body = QWidget()
        content = QVBoxLayout(body)

        # The readings (core.search.reading): how title / artist / album read in Latin letters, up to READING_SLOTS each
        # (Han: one per language) -- what the search compares. The ones typed here are the user's (kept); the others
        # (dim) are read from the stored text in the stored languages, after them, but those emptied or typed over
        # here (removed, kept so). A column per field, a row per slot.
        grp_readings = QGroupBox(tr("Readings"))
        readings_layout = QVBoxLayout(grp_readings)
        self.tbl_readings = QTableWidget(reading.READING_SLOTS, len(READING_FIELDS))
        names = {"title": tr("Title"), "artist": tr("Artist"), "album": tr("Album")}
        self.tbl_readings.setHorizontalHeaderLabels([names.get(f, f.capitalize()) for f in READING_FIELDS])
        self.tbl_readings.setVerticalHeaderLabels([str(i + 1) for i in range(reading.READING_SLOTS)])
        self.tbl_readings.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_readings.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.tbl_readings.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.tbl_readings.setToolTip(
            tr("<p>How the title, artist and album read in Latin letters: the search finds the track by them. Type "
               "in a cell to add a reading; empty it to remove one.</p>"))
        self.tbl_readings.setFixedHeight(
            self.tbl_readings.horizontalHeader().sizeHint().height()
            + self.tbl_readings.verticalHeader().defaultSectionSize() * reading.READING_SLOTS + 4)
        self.tbl_readings.itemChanged.connect(self._on_reading_changed)
        self.tbl_readings.installEventFilter(self)                   # Delete / Backspace empty the selected cells
        self._reading_user = {f: [] for f in READING_FIELDS}         # the readings the user added
        self._reading_removed = {f: [] for f in READING_FIELDS}      # the automatic ones the user removed
        self._reading_touched = {f: False for f in READING_FIELDS}   # changed here (saved on apply)
        readings_layout.addWidget(self.tbl_readings)
        btn_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Reset)
        btn_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        btn_box.button(QDialogButtonBox.Reset).clicked.connect(self._on_reset)
        readings_layout.addWidget(btn_box)

        # The audio fingerprints: the Variants' (from the stored analysis) and Search by Music's (from the audio)
        grp_audio = QGroupBox(tr("Audio Fingerprints"))
        audio_form = QFormLayout(grp_audio)
        self.lbl_variants = QLabel()
        self.lbl_music = QLabel()
        self.btn_rebuild = QPushButton(tr("Rebuild Audio Fingerprints"))
        self.btn_rebuild.setToolTip(
            tr("<p>The track's Variants fingerprint and its Search by Music index made again.</p>"))
        self.btn_rebuild.clicked.connect(self.rebuild)
        self.lbl_rebuild = QLabel()
        self.lbl_rebuild.setWordWrap(True)
        audio_form.addRow(tr("Variants"), self.lbl_variants)
        audio_form.addRow(tr("Search by Music"), self.lbl_music)
        audio_form.addRow("", self.btn_rebuild)
        audio_form.addRow(self.lbl_rebuild)
        if bus is not None:
            bus.sig_music_index_progress.connect(self._on_index_progress)
            bus.sig_music_index_finished.connect(self._on_index_finished)
            bus.sig_music_index_failed.connect(self._on_index_failed)

        content.addWidget(grp_readings)
        content.addWidget(grp_audio)
        content.addStretch(1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll_page(body), 1)
        self.set_track(None)

    # Public API
    def track_uid(self) -> Optional[str]:
        return self._row.uid if self._row is not None else None

    def set_track(self, row: Optional[TrackRow]) -> None:
        """The track's readings, read on the library thread (shown when they come, while it is still shown)."""
        self._row = row
        if row is not None and self._library is not None:
            uid = row.uid
            self._library.read(jobs.readings, uid, done=lambda found: self._show_readings(uid, *found, came=True))
        self._show_readings(row.uid if row is not None else None, {}, {})

    def _show_readings(self, uid: Optional[str], stored: dict, removed: dict, *, came: bool = False) -> None:
        """``came``: the stored readings, come from the library (not shown over readings typed meanwhile)."""
        if (self._row.uid if self._row is not None else None) != uid or (came and self.has_changes()):
            return
        for f in READING_FIELDS:
            self._reading_touched[f] = False
            self._reading_user[f] = [text for text, src in stored.get(f, []) if src == USER_SOURCE]
            self._reading_removed[f] = list(removed.get(f, []))
            self._fill_readings(f, stored.get(f, []))
        self.tbl_readings.setEnabled(uid is not None)
        if not came:
            self._show_index_state()

    def has_changes(self) -> bool:
        """Whether readings were changed here and not applied."""
        return self._row is not None and any(self._reading_touched.values())

    def apply(self) -> bool:
        """The readings changed here saved (the others of the track kept)."""
        if self._row is None or not self.has_changes() or self._library is None:
            return True
        uid, changes = self._row.uid, self.user_readings()

        def written(saved: Optional[TrackRow]) -> None:
            if saved is not None and self.track_uid() == uid and not self.has_changes():
                self.set_track(saved)

        # written by the library's owner; meanwhile nothing counts as not applied
        for f in READING_FIELDS:
            self._reading_touched[f] = False
        self._library.write(jobs.set_user_readings, uid, changes, done=written)
        return True

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
        return reading.readings(str(getattr(self._row, field, "") or ""), s.db_languages, han_run=s.han_run())

    def _refresh_auto_readings(self, field: str) -> None:
        """The user's readings, then the ones read from the stored text but the removed ones."""
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

    def _on_reset(self) -> None:
        if self._row is not None:
            self.set_track(self._row)

    # Audio fingerprints
    def _show_index_state(self) -> None:
        """Whether the track is in each index, read by the library's owner (_on_state)."""
        uid = self.track_uid()
        self.lbl_variants.setText("")
        self.lbl_music.setText("")
        self.btn_rebuild.setEnabled(uid is not None and self._rebuilding is None and self._bus is not None)
        if self._rebuilding is None:
            self.lbl_rebuild.setText("")
        if uid is not None:
            self._read_state(uid)

    def _read_state(self, uid: str) -> None:
        if self._library is not None:
            self._library.read(jobs.index_state, uid, done=lambda found: self._on_state(uid, *found),
                               failed=lambda _exc: self._on_state(uid, False, False))

    def _on_state(self, uid: str, variants: bool, music: bool) -> None:
        if uid == self._rebuilding and self._checking:      # Index Now done: was the track indexed?
            self._checking = False
            if not (variants and music) and not self._asked_again:
                # Index Now was running another run when asked (it takes one at a time): asked again, for this track
                self._asked_again = True
                self._bus.sig_music_index_requested.emit(None)
                return
            self._rebuilding = None
            self.lbl_rebuild.setText(tr("Rebuilt.") if variants and music
                                     else tr("Not rebuilt: Index Now could not index the track."))
            self.btn_rebuild.setEnabled(uid == self.track_uid())
        if uid != self.track_uid():
            return
        self.lbl_variants.setText(tr("Fingerprinted") if variants else tr("Not indexed"))
        self.lbl_music.setText(tr("Indexed") if music else tr("Not indexed"))

    def rebuild(self) -> None:
        """The track out of both indexes (by the library's owner), then Index Now asked for (_on_dropped)."""
        uid = self.track_uid()
        if uid is None or self._rebuilding is not None or self._bus is None or self._library is None:
            return
        if self._row is not None and not os.path.isfile(self._row.path or ""):
            self.lbl_rebuild.setText(tr("The audio file is not there: the fingerprints cannot be made again."))
            return
        self._rebuilding, self._asked_again = uid, False
        self.btn_rebuild.setEnabled(False)
        self.lbl_rebuild.setText(tr("Taking the track out of the indexes…"))
        self._library.write(jobs.drop_from_indexes, [uid], done=lambda _r: self._on_dropped(uid, ""),
                            failed=lambda exc: self._on_dropped(uid, str(exc)))

    def _on_dropped(self, uid: str, error: str) -> None:
        if error:
            self._rebuilding = None
            self.lbl_rebuild.setText(tr("Not rebuilt: {error}").format(error=error))
            self._show_index_state()
            return
        self.lbl_rebuild.setText(tr("Indexing…"))
        if uid == self.track_uid():
            self._show_index_state()
        self._bus.sig_music_index_requested.emit(None)

    def _on_index_progress(self, _run, percent: int, _text: str) -> None:
        if self._rebuilding is not None:
            self.lbl_rebuild.setText(tr("Indexing… {percent}%").format(percent=int(percent)))

    def _on_index_finished(self, _run, _summary) -> None:
        uid = self._rebuilding
        if uid is None:
            if self._row is not None:
                self._show_index_state()
            return
        self._checking = True       # whether it was indexed: read by the library's owner (_on_state)
        self._read_state(uid)

    def _on_index_failed(self, _run, error: str) -> None:
        if self._rebuilding is None:
            return
        self._rebuilding = None
        self.lbl_rebuild.setText(tr("Not rebuilt: {error}").format(error=error))
        self._show_index_state()
