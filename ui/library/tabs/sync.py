"""The track panel's Sync tab: the track's analysis and stored values taken to other programs, in place: its audio
file's tags (Write Tags, Restore from Backup), the Rekordbox XML's entry for it and rekordbox's phrases."""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from core.tieups import domain as tieup
from system.resource_paths import resource_path
from core.analysis_library import jobs as analysis_jobs
from core.analysis_library.stored import StoredAnalysis
from core.library import jobs
from core.library.records import TrackRow
from core.library.service import LibraryService
from analyzer.audio.decoder import get_samplerate
from core.settings import store as settings_store
from core.tags import backup as tag_backup
from core.tags.writer import (SERATO, TEXT_FIELDS, TagWriteError, bpm_text, planned_writes, read_tags, tag_format,
                              write_tags)
from core.track.model import DataModel
from third_party import rekordbox_phrases, serato
from ui.config.scroll import scroll_page
from ui.library.tabs.export import model_holds, track_source
from ui.widgets.elided_label import ElidedLabel
from analyzer.music.key_names import idx_to_labels
from ui.theme import current_theme
from system.i18n import tr, tr_n


class SyncTrackPanel(QtWidgets.QWidget):
    """The library panel's Sync tab: the track's stored values written into its audio file's tags
    (``core.tags.writer``: the fields checked, the file's other tags and its audio kept), after a confirmation
    showing each change (``TagWriteDialog``) with the backup it asks for made first (core.tags.backup: none, the tag
    block, the whole file); a backup put back (Restore from Backup); the track written into the Rekordbox XML again
    (Sync to Rekordbox XML: the request an edit's save sends, core.workflows.rekordbox_sync's track sync, its
    outcome shown here)."""

    _sig_tags_read = QtCore.Signal(str, object, object)   # the track's uid, the planned writes (None: unread), the write
    _sig_tags_done = QtCore.Signal(str, str, str)   # the track's uid, the error ("": written), the backup made ("")
    _sig_restored = QtCore.Signal(str, str)          # the track's uid, the error ("": put back)
    _sig_phrases_planned = QtCore.Signal(str, object, str)   # the track's uid, the plan (None: not made), the error
    _sig_phrases_written = QtCore.Signal(str, str, str)      # the track's uid, the backup made, the error ("": written)
    _sig_rb_backups = QtCore.Signal(str, object, object, str)   # the uid, its analysis folder, its backups, the error
    _sig_rb_restored = QtCore.Signal(str, object, str)          # the uid, the names put back, the error ("": put back)

    def __init__(self, library: LibraryService, parent: Optional[QtWidgets.QWidget] = None, *,
                 model: Optional[DataModel] = None, bus=None,
                 unapplied_edits: Optional[Callable[[str], bool]] = None,
                 tieup_language: Optional[Callable[[], str]] = None) -> None:
        """``library``: the library's owner (core.library.service: the stored values, the analysis);
        ``unapplied_edits(uid)``: whether the Edit tab has changes to that track not applied yet;
        ``tieup_language()``: the language of the tie-ups written after the comment (Settings > Library)."""
        super().__init__(parent)
        self._library = library
        self._tieup_language = tieup_language
        self._model: Optional[DataModel] = model
        self._bus = bus
        self._unapplied_edits = unapplied_edits
        self._tags_writing: Optional[str] = None   # the uid whose tags are being written
        self._rekordbox_asked: set[str] = set()     # the uids this tab asked to sync, their outcome still to show
        self._sig_tags_read.connect(self._confirm_tags)
        self._sig_tags_done.connect(self._on_tags_done)
        self._sig_restored.connect(self._on_restored)
        self._sig_phrases_planned.connect(self._confirm_phrases)
        self._sig_phrases_written.connect(self._on_phrases_written)
        self._sig_rb_backups.connect(self._choose_rekordbox_backup)
        self._sig_rb_restored.connect(self._on_rekordbox_restored)
        self._phrases_writing: Optional[str] = None   # the uid whose rekordbox phrases are being written
        self._phrase_wave = (None, 0.0)                # the overview waveform the preview shows, and its length (s)
        self._track: Optional[TrackRow] = None
        self._source: Optional[Path] = None

        self.lbl_title = ElidedLabel("-")
        self.lbl_artist = ElidedLabel("-")
        self.lbl_source = ElidedLabel("-", mode=QtCore.Qt.ElideMiddle)
        track_form = QtWidgets.QFormLayout()
        track_form.addRow(tr("Title"), self.lbl_title)
        track_form.addRow(tr("Artist"), self.lbl_artist)
        track_form.addRow(tr("Source"), self.lbl_source)

        tags = QtWidgets.QGroupBox(tr("File Tags"))
        tags.setToolTip(tr(
            "<p>Write the library's values into the audio file's own tags: only the fields checked change; the "
            "file's other tags, its artwork and its audio stay as they are.</p>"
        ))
        self.tag_fields: dict[str, QtWidgets.QCheckBox] = {}
        self._field_names: dict[str, str] = {}
        grid = QtWidgets.QGridLayout()
        for i, (field, label, checked, tip) in enumerate((
            ("title", tr("Title"), True, ""),
            ("artist", tr("Artist"), True, ""),
            ("album", tr("Album"), True, ""),
            ("comment", tr("Comment"), True, ""),
            ("bpm", "BPM", False, tr("MP4 / M4A files hold a whole BPM only.")),
            ("key", "Key", False, tr("Written as a musical key (Am, F#).")),
        )):
            box = QtWidgets.QCheckBox(label)
            box.setChecked(checked)
            if tip:
                box.setToolTip(tip)
            grid.addWidget(box, i // 2, i % 2)
            self.tag_fields[field] = box
            self._field_names[field] = label
        self.cb_comment_tieups = QtWidgets.QCheckBox(tr("Tie-ups after the comment"))
        self.cb_comment_tieups.setToolTip(tr(
            "<p>The comment written with the song's tie-ups after it, as [Title OP; Other ED]; the library's "
            "comment stays as it is.</p>"))
        self.cb_comment_tieups.setEnabled(False)
        self.tag_fields["comment"].toggled.connect(lambda _on: self._update_tag_controls())
        grid.addWidget(self.cb_comment_tieups, (len(self.tag_fields) + 1) // 2, 0, 1, 2)
        self.cb_serato = QtWidgets.QCheckBox(tr("Serato beatgrid and hot cues"))
        self.cb_serato.setToolTip(tr(
            "<p>Serato DJ's beatgrid and hot cues, in the file's tags where Serato reads them: the grid from the "
            "first downbeat (a ramp a marker a beat), the hot cues made of the JumpCUEs, the CUE points or both, as "
            "Settings > Library > Serato says. The file's Serato loops and track colour are kept.</p>"))
        grid.addWidget(self.cb_serato, (len(self.tag_fields) + 1) // 2 + 1, 0, 1, 2)
        self._field_names.update({"serato_beatgrid": tr("Serato beatgrid"), "serato_markers": tr("Serato hot cues"),
                                  "serato_markers_old": tr("Serato's old hot cues (A-E)"),
                                  "serato_tempo_map": tr("Serato's tempo map")})
        self.tag_note = QtWidgets.QLabel("")
        self.tag_note.setWordWrap(True)
        self.btn_write_tags = QtWidgets.QPushButton(tr("Write Tags"))
        self.btn_write_tags.setToolTip(tr("Write the checked fields' stored values into the file's tags."))
        self.btn_write_tags.clicked.connect(lambda: self.write_tags())
        self.btn_restore_tags = QtWidgets.QPushButton(tr("Restore from Backup..."))
        self.btn_restore_tags.setToolTip(tr(
            "<p>Put the track's file back as a backup made before writing its tags holds it: a tag backup "
            "(.mxtags) or a copy of the whole file.</p>"))
        self.btn_restore_tags.clicked.connect(self.restore_tags)
        # one above the other, the panel's width each (side by side they widened a narrow panel)
        tag_btns = QtWidgets.QVBoxLayout()
        tag_btns.addWidget(self.btn_write_tags)
        tag_btns.addWidget(self.btn_restore_tags)
        tags_layout = QtWidgets.QVBoxLayout(tags)
        tags_layout.addLayout(grid)
        tags_layout.addWidget(self.tag_note)
        tags_layout.addLayout(tag_btns)

        # what is synced checked (kept in deck.json as last checked), then one button
        rekordbox = QtWidgets.QGroupBox("Rekordbox")
        self.cb_rekordbox_xml = QtWidgets.QCheckBox("Rekordbox XML")
        self.cb_rekordbox_xml.setChecked(True)
        self.cb_rekordbox_xml.setToolTip(tr(
            "<p>The track written into the Rekordbox XML again, as saving an edit does: its beatgrid, CUE points, "
            "JumpCUEs and values as stored now. The XML's file and its options are in Settings > Library.</p>"))
        self.cb_rekordbox_phrases = QtWidgets.QCheckBox(tr("Phrases"))
        self.cb_rekordbox_phrases.setToolTip(tr(
            "<p>The track's phrases written into rekordbox's own analysis of it (the XML has no phrases), on the "
            "beatgrid rekordbox holds: import the XML's beatgrid first. rekordbox must be closed; a preview shows "
            "what is written, and its analysis file is copied first. The phrases' mood and names: Settings > "
            "Library > Rekordbox.</p>"))
        self.rekordbox_note = QtWidgets.QLabel("")
        self.rekordbox_note.setWordWrap(True)
        self.phrase_note = QtWidgets.QLabel("")
        self.phrase_note.setWordWrap(True)
        self.phrase_note.hide()
        self.btn_rekordbox_sync = QtWidgets.QPushButton(tr("Sync"))
        self.btn_rekordbox_sync.setToolTip(tr("Sync what is checked: the XML at once, the phrases after a preview."))
        self.btn_rekordbox_sync.clicked.connect(self.sync_to_rekordbox)
        for box in (self.cb_rekordbox_xml, self.cb_rekordbox_phrases):
            box.toggled.connect(lambda _on: self._update_rekordbox_controls())
        rekordbox_layout = QtWidgets.QVBoxLayout(rekordbox)
        rekordbox_layout.addWidget(self.cb_rekordbox_xml)
        rekordbox_layout.addWidget(self.cb_rekordbox_phrases)
        rekordbox_layout.addWidget(self.btn_rekordbox_sync)
        self.btn_rekordbox_restore = QtWidgets.QPushButton(tr("Restore from Backup..."))
        self.btn_rekordbox_restore.setToolTip(tr(
            "<p>Put the track's rekordbox analysis back as a backup made before writing its phrases holds it (the "
            "track's backups, newest first). rekordbox must be closed.</p>"))
        self.btn_rekordbox_restore.clicked.connect(self.restore_rekordbox)
        rekordbox_layout.addWidget(self.btn_rekordbox_restore)
        rekordbox_layout.addWidget(self.rekordbox_note)
        rekordbox_layout.addWidget(self.phrase_note)
        if bus is not None:
            bus.sig_rekordbox_sync_finished.connect(self._on_rekordbox_finished)
            bus.sig_rekordbox_sync_failed.connect(self._on_rekordbox_failed)

        # the fields scroll (as the Edit tab's): the panel never holds the library taller than a small window
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.addLayout(track_form)
        layout.addWidget(tags)
        layout.addWidget(rekordbox)
        layout.addStretch(1)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll_page(body), 1)
        self._show_choices()
        for box in (*self.tag_fields.values(), self.cb_comment_tieups, self.cb_serato, self.cb_rekordbox_xml,
                    self.cb_rekordbox_phrases):
            box.toggled.connect(lambda _on: self._keep_choices())

    # what is checked and the backups asked for, as last made (deck.json: deckconfig's sync_ / _backup values)
    def _show_choices(self) -> None:
        deck = settings_store.current().deckconfig
        checked = set(getattr(deck, "sync_tag_fields", ()) or ())
        for field, box in self.tag_fields.items():
            box.setChecked(field in checked)
        self.cb_comment_tieups.setChecked("comment_tieups" in checked)
        self.cb_serato.setChecked("serato" in checked)
        self.cb_rekordbox_xml.setChecked(bool(getattr(deck, "sync_rekordbox_xml", True)))
        self.cb_rekordbox_phrases.setChecked(bool(getattr(deck, "sync_rekordbox_phrases", False)))

    def _keep_choices(self) -> None:
        deck = settings_store.current().deckconfig
        deck.sync_tag_fields = ([f for f, box in self.tag_fields.items() if box.isChecked()]
                                + ["comment_tieups"] * self.cb_comment_tieups.isChecked()
                                + ["serato"] * self.cb_serato.isChecked())
        deck.sync_rekordbox_xml = self.cb_rekordbox_xml.isChecked()
        deck.sync_rekordbox_phrases = self.cb_rekordbox_phrases.isChecked()
        settings_store.deck_changed()

    @staticmethod
    def _keep_backup(name: str, kind: str) -> None:
        setattr(settings_store.current().deckconfig, name, kind)
        settings_store.deck_changed()

    def track_uid(self) -> Optional[str]:
        return self._track.uid if self._track is not None else None

    def set_track(self, track: TrackRow) -> None:
        self._track = track
        self.lbl_title.setText(track.title or "-")
        self.lbl_artist.setText(track.artist or "-")
        self._source = track_source(track, self._model)
        self.lbl_source.setText(str(self._source) if self._source else (track.path or "-"))
        self._set_tag_note("")
        self._update_tag_controls()
        self._set_rekordbox_note("")
        self._update_rekordbox_controls()
        self._set_phrase_note("")
        self._update_rekordbox_controls()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._update_rekordbox_controls()     # the settings may have changed while the tab was hidden

    def _model_holds_track(self) -> bool:
        return model_holds(self._model, self._track)

    # Rekordbox XML
    @staticmethod
    def _rekordbox_ready() -> bool:
        lib = settings_store.current().libconfig
        return bool(getattr(lib, "rekordbox_sync_enabled", False)
                    and str(getattr(lib, "rekordbox_xml_path", "") or "").strip())

    def _xml_ready(self) -> bool:
        """The XML can be synced for this track (the setting on, with a file)."""
        return self._rekordbox_ready() and self._track is not None and bool(self._track.uid) and self._bus is not None

    def _update_rekordbox_controls(self) -> None:
        ready = self._rekordbox_ready()
        self.cb_rekordbox_xml.setEnabled(ready)
        phrases = (self.cb_rekordbox_phrases.isChecked() and self._source is not None and self._source.is_file()
                   and self._phrases_writing is None)
        self.btn_rekordbox_sync.setEnabled((self.cb_rekordbox_xml.isChecked() and self._xml_ready()) or phrases)
        self.btn_rekordbox_restore.setEnabled(self._source is not None and self._source.is_file()
                                              and self._phrases_writing is None)
        off = tr("Sync with Rekordbox XML is off, or has no file: Settings > Library > Rekordbox.")
        if not ready:
            self._set_rekordbox_note(off)
        elif self.rekordbox_note.text() == off:
            self._set_rekordbox_note("")

    def sync_to_rekordbox(self) -> None:
        """What is checked synced: the XML (sync_rekordbox) and the phrases (write_phrases: after its preview)."""
        if self.cb_rekordbox_xml.isChecked() and self._xml_ready():
            self.sync_rekordbox()
        if self.cb_rekordbox_phrases.isChecked():
            self.write_phrases()

    def _set_rekordbox_note(self, text: str, *, error: bool = False) -> None:
        self.rekordbox_note.setText(text)
        theme = current_theme()
        self.rekordbox_note.setStyleSheet(f"color: {theme.danger if error else theme.text};")
        self.rekordbox_note.setVisible(bool(text))

    def sync_rekordbox(self) -> None:
        """The track written into the Rekordbox XML again: the request an edit's save sends (the bus's
        sig_rekordbox_sync_track_requested); its outcome shown under the button."""
        if self._track is None or not self._track.uid or self._bus is None:
            return
        if not self._rekordbox_ready():
            self._update_rekordbox_controls()
            return
        uid = self._track.uid
        if self._unapplied_edits is not None and self._unapplied_edits(uid):
            answer = QtWidgets.QMessageBox.question(
                self, tr("Sync to Rekordbox XML"),
                tr("The Edit tab has changes to this track that are not applied: the stored values are "
                   "written.\nSync anyway?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self._rekordbox_asked.add(uid)
        self._set_rekordbox_note(tr("Writing the track into the Rekordbox XML..."))
        self._bus.sig_rekordbox_sync_track_requested.emit(uid)

    def _asked_sync(self, request) -> Optional[str]:
        """The uid of a track sync this tab asked for (taken off the list), else None."""
        uid = str(getattr(request, "uid", "") or "")
        if getattr(request, "mode", "") != "track" or uid not in self._rekordbox_asked:
            return None
        self._rekordbox_asked.discard(uid)
        return uid

    def _on_rekordbox_finished(self, request, result) -> None:
        uid = self._asked_sync(request)
        if uid is not None and self._track is not None and self._track.uid == uid:
            self._set_rekordbox_note(tr("Written into {path}.").format(
                path=breakable_path(str(getattr(result, "output_path", "") or request.xml_path))))

    def _on_rekordbox_failed(self, request, message: str) -> None:
        uid = self._asked_sync(request)
        if uid is not None and self._track is not None and self._track.uid == uid:
            self._set_rekordbox_note(tr("The Rekordbox XML could not be written: {error}").format(error=message),
                                     error=True)

    # rekordbox's phrases (third_party.rekordbox_phrases): one track, from this button only
    def _set_phrase_note(self, text: str, *, error: bool = False) -> None:
        self.phrase_note.setText(text)
        theme = current_theme()
        self.phrase_note.setStyleSheet(f"color: {theme.danger if error else theme.text};")
        self.phrase_note.setVisible(bool(text))

    def write_phrases(self, analysis: Optional[StoredAnalysis] = None) -> None:
        """The track's phrases (the deck's when it holds the track, else the stored ones, read now) written into
        rekordbox's analysis of it, in the mood and names Settings > Library > Rekordbox says: what changes read on
        a thread, previewed (PhraseWriteDialog: over the track's overview waveform), then written there."""
        if self._track is None or not self._track.uid or self._source is None or self._phrases_writing:
            return
        uid = self._track.uid
        if self._model_holds_track():
            phrases, wave = list(self._model.phrases), self._model.analysis.wave
            duration = float(self._model.analysis.duration_sec or 0.0)
        elif analysis is None:
            def came(read) -> None:
                if self._track is None or self._track.uid != uid:
                    return
                if read is None:
                    self._set_phrase_note(tr("The track's analysis could not be read."), error=True)
                    return
                self.write_phrases(read)

            self._library.read(analysis_jobs.analysis, uid, done=came)
            return
        else:
            phrases, wave, duration = list(analysis.phrases), analysis.wave, float(analysis.duration_sec or 0.0)
        lib = settings_store.current().libconfig
        mood = rekordbox_phrases.MOODS.get(str(getattr(lib, "rekordbox_phrase_mood", "mid")),
                                           rekordbox_phrases.MOOD_MID)
        mapping = rekordbox_phrases.phrase_map(mood, str(getattr(lib, "rekordbox_phrase_map", "")))
        self._phrases_writing = uid
        self._phrase_wave = (wave, duration)
        self._update_rekordbox_controls()
        self._set_phrase_note(tr("Reading rekordbox's analysis of the track..."))
        threading.Thread(target=self._plan_phrases_job, args=(uid, str(self._source), phrases, mood, mapping),
                         daemon=True).start()

    def _plan_phrases_job(self, uid: str, path: str, phrases: list, mood: int, mapping: dict) -> None:
        try:
            self._sig_phrases_planned.emit(uid, rekordbox_phrases.plan(path, phrases, mood, mapping), "")
        except rekordbox_phrases.RekordboxError as exc:
            self._sig_phrases_planned.emit(uid, None, str(exc))
        except OSError as exc:
            self._sig_phrases_planned.emit(uid, None, tr("rekordbox's analysis file could not be read: {error}").format(
                error=exc.strerror or exc))

    def _confirm_phrases(self, uid: str, plan, error: str) -> None:
        if error or plan is None:
            self._phrases_writing = None
            self._update_rekordbox_controls()
            self._set_phrase_note(error, error=True)
            return
        notes = [_phrase_loss(loss) for loss in plan.losses]
        if plan.off_grid > OFF_GRID_SEC:
            notes.insert(0, tr("The phrases are {ms} ms off rekordbox's beats at most: its beatgrid is not the "
                               "track's. Import the XML's beatgrid in rekordbox first, then write the phrases again."
                               ).format(ms=int(round(plan.off_grid * 1000))))
        wave, duration = getattr(self, "_phrase_wave", (None, 0.0))
        dialog = PhraseWriteDialog(self._source.name if self._source else "", plan, wave, duration, notes,
                                   str(getattr(settings_store.current().deckconfig, "phrase_backup", "ext")), self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            self._phrases_writing = None
            self._update_rekordbox_controls()
            self._set_phrase_note("")
            return
        self._keep_backup("phrase_backup", dialog.backup())
        self._set_phrase_note(tr("Writing the phrases..."))
        threading.Thread(target=self._write_phrases_job, args=(uid, plan, dialog.backup()), daemon=True).start()

    def _write_phrases_job(self, uid: str, plan, backup: str = "ext") -> None:
        try:
            made = rekordbox_phrases.write(plan, backup)
            self._sig_phrases_written.emit(uid, str(made) if made is not None else "", "")
        except rekordbox_phrases.RekordboxError as exc:
            self._sig_phrases_written.emit(uid, "", str(exc))
        except OSError as exc:
            self._sig_phrases_written.emit(uid, "", tr("rekordbox's analysis file could not be written: {error}").format(
                error=exc.strerror or exc))

    def _on_phrases_written(self, uid: str, backup: str, error: str) -> None:
        self._phrases_writing = None
        self._update_rekordbox_controls()
        if self._track is not None and self._track.uid == uid:
            note = error or (tr("The phrases were written into rekordbox's analysis.\nIts backup: {path}").format(
                path=breakable_path(backup)) if backup else tr("The phrases were written into rekordbox's analysis."))
            self._set_phrase_note(note, error=bool(error))
        elif error:
            QtWidgets.QMessageBox.warning(self, tr("Write Phrases to rekordbox"), error)

    def restore_rekordbox(self) -> None:
        """One of the track's rekordbox backups (third_party.rekordbox_phrases.backups_of) chosen and put back after a
        confirmation; its analysis folder found and the files written on threads of their own."""
        if self._track is None or self._source is None or self._phrases_writing:
            return
        uid, path = self._track.uid, str(self._source)
        self._phrases_writing = uid
        self._update_rekordbox_controls()
        self._set_phrase_note(tr("Reading rekordbox's analysis of the track..."))

        def find() -> None:
            try:
                folder = rekordbox_phrases.analysis_folder(path)
                self._sig_rb_backups.emit(uid, folder, rekordbox_phrases.backups_of(folder), "")
            except rekordbox_phrases.RekordboxError as exc:
                self._sig_rb_backups.emit(uid, None, [], str(exc))
            except OSError as exc:
                self._sig_rb_backups.emit(uid, None, [], str(exc.strerror or exc))

        threading.Thread(target=find, daemon=True).start()

    def _choose_rekordbox_backup(self, uid: str, folder, backups: list, error: str) -> None:
        def done(note: str = "", failed: bool = False) -> None:
            self._phrases_writing = None
            self._update_rekordbox_controls()
            self._set_phrase_note(note, error=failed)

        if error or folder is None:
            done(error, True)
            return
        if not backups:
            done(tr("There is no backup of the track's rekordbox analysis."))
            return
        names = [_rekordbox_backup_name(b) for b in backups]
        chosen, ok = QtWidgets.QInputDialog.getItem(self, tr("Restore from Backup"),
                                                    tr("The backup to put back:"), names, 0, False)
        if not ok:
            done()
            return
        backup = backups[names.index(chosen)]
        answer = QtWidgets.QMessageBox.question(
            self, tr("Restore from Backup"),
            tr("Put the track's rekordbox analysis back as the backup of {when} holds it?").format(when=chosen),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            done()
            return
        self._set_phrase_note(tr("Putting the files back..."))

        def put() -> None:
            try:
                self._sig_rb_restored.emit(uid, rekordbox_phrases.restore(backup, folder), "")
            except rekordbox_phrases.RekordboxError as exc:
                self._sig_rb_restored.emit(uid, [], str(exc))
            except OSError as exc:
                self._sig_rb_restored.emit(uid, [], tr("rekordbox's analysis file could not be written: {error}")
                                           .format(error=exc.strerror or exc))

        threading.Thread(target=put, daemon=True).start()

    def _on_rekordbox_restored(self, uid: str, names: list, error: str) -> None:
        self._phrases_writing = None
        self._update_rekordbox_controls()
        note = error or tr("Put back as the backup holds it: {files}").format(files=", ".join(names))
        if self._track is not None and self._track.uid == uid:
            self._set_phrase_note(note, error=bool(error))
        elif error:
            QtWidgets.QMessageBox.warning(self, tr("Restore from Backup"), error)

    # File tags
    def _tag_format(self) -> Optional[str]:
        return tag_format(self._source) if self._source is not None and self._source.is_file() else None

    def _update_tag_controls(self) -> None:
        fmt = self._tag_format()
        for box in (*self.tag_fields.values(), self.cb_serato):
            box.setEnabled(fmt is not None)
        self.cb_comment_tieups.setEnabled(fmt is not None and self.tag_fields["comment"].isChecked())
        self.btn_write_tags.setEnabled(fmt is not None and self._tags_writing is None)
        self.btn_restore_tags.setEnabled(self._source is not None and self._source.is_file()
                                         and self._tags_writing is None)
        if self._source is not None and self._source.is_file() and fmt is None:
            self._set_tag_note(tr("Tags cannot be written to this file's format."))

    def _set_tag_note(self, text: str, *, error: bool = False) -> None:
        self.tag_note.setText(text)
        # the colour set either way (a style sheet emptied after the error's kept its red)
        theme = current_theme()
        self.tag_note.setStyleSheet(f"color: {theme.danger if error else theme.text};")

    def write_tags(self, analysis: Optional[StoredAnalysis] = None) -> None:
        """The checked fields' stored values (read from the library now) written into the track's file, in a
        thread; the result shown under the fields. Serato's marks: the deck's when it holds the track, else the
        stored analysis read now (``analysis``: the one just read for this write; never one kept from an earlier
        write, which an edit saved since would make stale)."""
        if self._track is None or not self._track.uid or self._tags_writing is not None:
            return
        if self._tag_format() is None:
            self._update_tag_controls()
            return
        fields = [field for field, box in self.tag_fields.items() if box.isChecked()]
        with_serato = self.cb_serato.isChecked()
        if not fields and not with_serato:
            self._set_tag_note(tr("Check the fields to write."), error=True)
            return
        uid = self._track.uid
        marks = self._serato_marks(analysis) if with_serato else None
        if with_serato and marks is None:      # the stored analysis still to read: asked again when it comes
            def came(read) -> None:
                if self._track is None or self._track.uid != uid:
                    return
                if read is None:
                    self._set_tag_note(tr("The track's analysis could not be read."), error=True)
                    return
                self.write_tags(read)

            self._library.read(analysis_jobs.analysis, uid, done=came)
            return
        if self._unapplied_edits is not None and self._unapplied_edits(uid):
            answer = QtWidgets.QMessageBox.question(
                self, tr("Write Tags"),
                tr("The Edit tab has changes to this track that are not applied: the stored values are "
                   "written.\nWrite the tags anyway?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        with_tieups = self.cb_comment_tieups.isChecked()
        self._tags_writing = uid
        self._update_tag_controls()
        self._set_tag_note(tr("Reading the file's tags..."))
        # the stored values read by the library's owner, the file's tags on a thread of its own, the changes
        # confirmed, then written there
        lang = self._tieup_language() if self._tieup_language is not None else ""
        self._library.read(jobs.tag_values, uid, with_tieups, lang,
                           done=lambda found: self._write_stored(uid, fields, *found, marks=marks))

    def _serato_marks(self, a: Optional[StoredAnalysis]) -> Optional[tuple]:
        """What Serato's tags are made of (the beats, the tempo rows, the JumpCUE groups, the CUE points): the deck's
        marks when it holds the track (as edited), else ``a``'s (the stored analysis); None: ``a`` still to read."""
        if self._model_holds_track():
            m = self._model
            return m.beats, m.tempo_segments, list(m.jump_cues), list(m.cue_points)
        return None if a is None else (a.beats, a.tempo_segments, list(a.jump_cues), list(a.cue_points))

    def _write_stored(self, uid: str, fields: list, row: Optional[TrackRow], suffix: str,
                      marks: Optional[tuple] = None) -> None:
        if row is None:
            self._tags_writing = None
            self._update_tag_controls()
            self._set_tag_note(tr("The track is no longer in the library."), error=True)
            return
        stored = {
            "title": row.title or "",
            "artist": row.artist or "",
            "album": row.album or "",
            "comment": tieup.with_suffix(row.comment or "", suffix),
            "bpm": row.bpm,
            "key": idx_to_labels(row.key)[1] if row.key is not None else "",
        }
        values = {field: stored[field] for field in fields}
        write = (str(self._source), row.path, values, marks, settings_store.current().libconfig)
        threading.Thread(target=self._read_tags_job, args=(uid, write), daemon=True).start()

    def _read_tags_job(self, uid: str, write: tuple) -> None:
        """A worker thread: where the tags go and what those places hold now (``planned_writes``, None: the file
        not read), for the confirmation. The fields' sources are the library's (where the analysis read them); a
        track without them (one added before they were kept, or given another file) has them found now as the
        analysis would read the file, kept with the write. The library's part asked of its owner
        (LibraryService.call)."""
        from core.tags.reader import extract_tags, read_tag_sources
        file_path, stored_path, values, marks, lib = write
        sources, kept, losses = {}, {}, []
        try:
            sources = self._library.call(jobs.tag_sources, uid)
            missing = [f for f in TEXT_FIELDS if f in values and f not in sources]
            if missing:
                found = read_tag_sources(file_path, *extract_tags(file_path))
                if found:
                    kept = {f: found[f] for f in missing}
                    sources = {**sources, **kept}
            if marks is not None:
                losses = self._serato_values(file_path, marks, values, lib)
            plan = planned_writes(file_path, values, sources)
        except (TagWriteError, OSError):
            plan = None
        self._sig_tags_read.emit(uid, plan, (file_path, stored_path, values, sources, kept, losses))

    @staticmethod
    def _serato_values(file_path: str, marks: tuple, values: dict, lib=None) -> list[str]:
        """Serato's fields put in ``values`` from ``marks`` (the beats, the tempo rows, the JumpCUE groups, the CUE
        points) as ``lib`` (the libconfig: its serato_ options; None: the defaults) says, in Serato's time of the
        file (third_party.serato.time_offset): the beatgrid, the hot cues over the file's own Markers2 (its loops
        and colour kept; its hot cues in the slots left kept as the options say: "kept" among the losses then),
        the old Markers_ of the same hot cues A-E (its loops kept; a FLAC's emptied when it has one); returns what
        Serato loses (third_party.serato's losses)."""
        beats, rows, groups, cue_points = marks
        source = getattr(lib, "serato_hot_cues", "jumpcues")
        keep, bpm_lock = getattr(lib, "serato_keep_cues", True), getattr(lib, "serato_bpm_lock", True)
        now = read_tags(file_path)
        existing = now.get("serato_markers", "").encode("latin-1")
        offset = serato.time_offset(file_path, get_samplerate(file_path))
        markers, losses = serato.grid_markers(beats, rows)
        markers = serato.shifted(markers, offset)
        if markers:     # the grid both ways: the Tempo Map newer Serato versions read first, the BeatGrid
            values["serato_beatgrid"] = serato.build_beatgrid(markers)
            values["serato_tempo_map"] = serato.build_tempo_map(markers)
        cues, cue_losses = serato.hot_cues(groups, cue_points, source, existing, keep, offset)
        values["serato_markers"] = serato.build_markers2(cues, existing, bpm_lock)
        kept = [c for c in cues if c.kept]
        if kept:
            cue_losses = cue_losses + ["kept:" + "".join("ABCDEFGH"[c.slot] for c in kept)]
        old = now.get("serato_markers_old", "").encode("latin-1")
        kind = {"mp4": "mp4", "id3": "id3", "wav": "id3", "aiff": "id3"}.get(tag_format(file_path) or "")
        if kind is not None:    # the old hot cues (A-E) older Serato versions read: as Serato writes them
            values["serato_markers_old"] = serato.build_markers_old(cues, kind, old, values["serato_markers"])
        elif old:               # FLAC: Serato keeps no Markers_ there; one found is emptied (read before Markers2)
            values["serato_markers_old"] = ""
        return losses + cue_losses

    def _confirm_tags(self, uid: str, plan: Optional[dict], write: tuple) -> None:
        """The changes shown (``plan``: core.tags.writer.planned_writes, None: the file not read), the backup asked
        for; written when confirmed. A field changes where a place it is written to does not hold it yet; each
        place shown with its value (two or more: each by its name), and a warning where the format's own place is
        written beside the one the library read (a store's M4A with a blank ©cmt and its comment in desc).
        Each place is checked to be written (unchecked: left as it is). Nothing changing: said, not written."""
        file_path, stored_path, values, sources, kept, losses = write
        changes = []
        for field, value in values.items():
            new = bpm_text(value) if field == "bpm" else str(value or "")
            places = (plan or {}).get(field)
            if plan is not None and all(now == after for _s, _name, now, after, _kind in places or ()):
                continue
            shown = (lambda vs: " / ".join(v.strip() for v in vs if v.strip())) if field not in SERATO else \
                (lambda vs: " / ".join(_serato_shown(field, v) for v in vs))
            olds = ([(spelling, name, shown(now)) for spelling, name, now, _after, _kind in places]
                    if places else [("", "", "?")])
            if field in SERATO:
                raw = value.decode("latin-1") if isinstance(value, bytes) else str(value or "")
                new = _serato_shown(field, raw) if raw else ""
            warning = "\n".join(_serato_losses(field, losses))
            own = next((p for p in places or () if p[4] == "own"), None)
            if own is not None and any(p[4] == "source" for p in places) and own[2] != own[3]:
                there = " / ".join(v.strip() for v in own[2] if v.strip())
                warning = tr("Also written to {place}, the field DJ software (rekordbox ...) reads. It holds "
                             "now: {value}").format(place=own[1], value=there or tr("(empty)"))
            if field == "serato_markers_old" and not new:
                warning = tr("Serato reads this before the hot cues written here: it is emptied (its loops and "
                             "names are in the hot cues' tag too).")
            elif field == "serato_markers_old":
                warning = tr("Older Serato versions read the hot cues from here: the first five (A-E), without "
                             "names; the file's old loops kept.")
            changes.append((field, self._field_names.get(field, field), olds, new, warning))
        if not changes:
            self._tags_writing = None
            self._update_tag_controls()
            notes = [tr("The file's tags already hold these values.")]
            notes += [_kept_note(loss[len("kept:"):]) for loss in losses if loss.startswith("kept:")]
            self._set_tag_note("\n".join(notes))
            return
        dialog = TagWriteDialog(Path(file_path).name, changes,
                                str(getattr(settings_store.current().deckconfig, "tag_backup", "tags")), self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            self._tags_writing = None
            self._update_tag_controls()
            self._set_tag_note("")
            return
        self._keep_backup("tag_backup", dialog.backup())
        self._set_tag_note(tr("Writing the tags..."))
        places = dialog.places() if plan is not None else None
        threading.Thread(target=self._write_tags_job,
                         args=(uid, file_path, stored_path, values, sources, kept, dialog.backup(), places),
                         daemon=True).start()

    def _write_tags_job(self, uid: str, file_path: str, stored_path: str, values: dict, sources: dict,
                        kept: dict, backup: str, places: Optional[dict] = None) -> None:
        """A worker thread (no widgets here): the backup asked for made (``backup``: "" none, "tags" the tag block
        just before the write, "file" the whole file), then the tags written as the confirmation showed
        (``sources``: where the analysis read the fields; ``kept`` those found now, stored with the write;
        ``places``: the places checked there, None: where ``write_tags`` puts them), the file's new size and time
        stored. A write that fails leaves no backup behind. The library's part asked of its owner
        (LibraryService.call)."""
        library = self._library
        made: list[Path] = []
        try:
            if kept:
                library.call(jobs.set_tag_sources, uid, kept, write=True)
            if backup == "file":
                made.append(tag_backup.backup_copy(file_path))
            about = (lambda p, segments: made.append(tag_backup.tag_block_backup(p, segments))) \
                if backup == "tags" else None
            write_tags(file_path, values, sources, about_to_write=about, places=places)
            library.call(jobs.touch_file_stats, stored_path, write=True)
            error = ""
        except TagWriteError as exc:
            error = str(exc)
        except PermissionError:
            error = tr("Another program has the file open (a DJ program with the track loaded?): the tags were not "
                       "written. Close it there, then write again.")
        except OSError as exc:
            error = tr("The file could not be written: {error}").format(error=exc.strerror or exc)
        if error:
            for path in made:
                path.unlink(missing_ok=True)
        self._sig_tags_done.emit(uid, error, str(made[0]) if made and not error else "")

    def _on_tags_done(self, uid: str, error: str, backup: str) -> None:
        self._tags_writing = None
        if self._track is not None and self._track.uid == uid:
            note = error or (tr("The tags were written to the file.\nIts backup: {path}").format(
                path=breakable_path(backup)) if backup else tr("The tags were written to the file."))
            self._set_tag_note(note, error=bool(error))
            self.tag_note.setToolTip(backup if backup and not error else "")
        elif error:
            QtWidgets.QMessageBox.warning(self, tr("Write Tags"), error)
        self._update_tag_controls()
        if not error and self._bus is not None:
            self._bus.sig_rekordbox_sync_track_requested.emit(uid)   # the file's size in the XML

    def restore_tags(self) -> None:
        """One of the file's backups (core.tags.backup.backups_of: the track's file's, newest first) chosen, and the
        file put back as it holds it after a confirmation, on a thread of its own."""
        if self._track is None or self._source is None or not self._source.is_file() or self._tags_writing:
            return
        try:
            backups = tag_backup.backups_of(self._source)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, tr("Restore from Backup"), str(exc.strerror or exc))
            return
        if not backups:
            self._set_tag_note(tr("There is no backup of this file's tags."))
            return
        names = [_tag_backup_name(b) for b in backups]
        chosen, ok = QtWidgets.QInputDialog.getItem(self, tr("Restore from Backup"), tr("The backup to put back:"),
                                                    names, 0, False)
        if not ok:
            return
        path = str(backups[names.index(chosen)])
        question = tr("Put '{file}' back as the backup of {when} holds it?").format(file=self._source.name,
                                                                                    when=chosen)
        answer = QtWidgets.QMessageBox.question(self, tr("Restore from Backup"), question,
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        uid = self._track.uid
        self._tags_writing = uid
        self._update_tag_controls()
        self._set_tag_note(tr("Putting the file back..."))
        threading.Thread(target=self._restore_job, args=(uid, path, str(self._source), self._track.path),
                         daemon=True).start()

    def _restore_job(self, uid: str, backup: str, file_path: str, stored_path: str) -> None:
        try:
            tag_backup.restore(backup, file_path)
            self._library.call(jobs.touch_file_stats, stored_path, write=True)
            error = ""
        except tag_backup.BackupError as exc:
            error = str(exc)
        except PermissionError:
            error = tr("Another program has the file open (a DJ program with the track loaded?): it was not put "
                       "back. Close it there, then restore again.")
        except OSError as exc:
            error = tr("The file could not be written: {error}").format(error=exc.strerror or exc)
        self._sig_restored.emit(uid, error)

    def _on_restored(self, uid: str, error: str) -> None:
        self._tags_writing = None
        if self._track is not None and self._track.uid == uid:
            self._set_tag_note(error or tr("The file was put back as the backup holds it."), error=bool(error))
        elif error:
            QtWidgets.QMessageBox.warning(self, tr("Restore from Backup"), error)
        self._update_tag_controls()
        if not error and self._bus is not None:
            self._bus.sig_rekordbox_sync_track_requested.emit(uid)


def _serato_shown(field: str, text: str) -> str:
    """A Serato field's value (its payload as Latin-1 text) as the confirmation shows it."""
    raw = text.encode("latin-1", "replace")
    if field == "serato_beatgrid":
        markers = serato.parse_beatgrid(raw)
        if not markers:
            return tr("(not read)")
        return tr_n("{count} grid marker from {start}, {bpm} BPM at the end",
                    "{count} grid markers from {start}, {bpm} BPM at the end", len(markers)).format(
            count=len(markers), start=_mmss(markers[0][0]), bpm=f"{markers[-1][1]:.2f}")
    if field == "serato_markers":
        cues = serato.parse_cues(raw)
        text = "; ".join(f"{'ABCDEFGH'[c.slot] if c.slot < 8 else c.slot + 1} {_mmss(c.time_sec)} {c.name}".strip()
                         for c in cues) or tr("no hot cues")
        return text + (" · " + tr("BPM locked") if serato.bpm_locked(raw) else "")
    if field == "serato_tempo_map":
        anchor, segments = serato.parse_tempo_map(raw)
        if not segments:
            return tr("(not read)")
        return tr_n("{count} tempo segment from {start}, {bpm} BPM at the end",
                    "{count} tempo segments from {start}, {bpm} BPM at the end", len(segments)).format(
            count=len(segments), start=_mmss(anchor), bpm=f"{segments[-1][1]:.2f}")
    if field == "serato_markers_old" and raw:
        cues = serato.parse_markers_old(raw)
        return "; ".join(f"{'ABCDE'[c.slot]} {_mmss(c.time_sec)}" for c in cues) or tr("no hot cues")
    return ""


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def _serato_losses(field: str, losses: list[str]) -> list[str]:
    """What Serato cannot hold of the field's marks (third_party.serato's losses), said."""
    out = []
    if field == "serato_beatgrid" and "bars" in losses:
        out.append(tr("Serato counts 4-beat bars from the first downbeat: the bars of another length here (a 3-beat "
                      "bar, 7/4) are not kept."))
    if field == "serato_markers":
        if "slots" in losses:
            out.append(tr("Serato has 8 hot cues: the JumpCUEs past H are left out."))
        if "cue_points" in losses:
            out.append(tr("Serato has no memory cues: the CUE points are left out (Settings > Library > Serato can "
                          "make them hot cues)."))
        if "full" in losses:
            out.append(tr("Serato has 8 hot cues: the CUE points with no slot left are left out."))
        out += [_kept_note(loss[len("kept:"):]) for loss in losses if loss.startswith("kept:")]
    return out


def _made_when(day: str, stamp: str) -> str:
    """A backup's day folder (YYYYMMDD) and time (HHMMSS) as a date and time ("" when they are not those)."""
    if len(day) == 8 and len(stamp) >= 6 and (day + stamp[:6]).isdigit():
        return f"{day[:4]}-{day[4:6]}-{day[6:8]} {stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}"
    return ""


def _rekordbox_backup_name(backup: Path) -> str:
    """A rekordbox backup (rekordbox_phrases.backups_of's) as the restore lists it: when it was made, and what it is."""
    parts = backup.name.split(" ")
    when = _made_when(backup.parent.name, parts[1] if len(parts) > 1 else "") or backup.name
    what = tr("The track's analysis folder") if backup.is_dir() else tr("The analysis file (.EXT)")
    return f"{when} · {what}"


def _tag_backup_name(backup: Path) -> str:
    """A tag backup (core.tags.backup.backups_of's) as the restore lists it: when it was made, and what it holds."""
    when = _made_when(backup.parent.name, backup.name.rsplit(" ", 1)[-1]) or backup.name
    what = tr("The tags only") if backup.suffix.lower() == tag_backup.SUFFIX else tr("The whole file")
    return f"{when} · {what}"


OFF_GRID_SEC = 0.06     # a phrase this far from rekordbox's beat: its beatgrid is not the track's (a decoder's offset
                        # is ~0.02-0.05 s)


def _phrase_loss(loss: str) -> str:
    """What rekordbox's phrases cannot hold (third_party.rekordbox_phrases' losses), said."""
    if loss.startswith("continued:"):
        return tr("{label} goes on with the phrase before it (Settings > Library > Rekordbox).").format(
            label=loss[len("continued:"):])
    return tr("The mood has no more verses: the verse runs past its last verse are its last verse.")


class PhraseStrip(QtWidgets.QWidget):
    """A track's overview waveform (``wave``: an RGB image, columns over ``duration`` s) with phrases under it: each
    (name, start s) to the next one's start, the last to ``end`` (s), coloured as the phrase labels are."""

    WAVE_HEIGHT, BAR_HEIGHT = 56, 24
    _COLORS = {"Intro": "INTRO", "Verse": "VERSE", "Up": "VERSE", "Down": "BRIDGE", "Bridge": "BRIDGE",
               "Chorus": "CHORUS", "Outro": "OUTRO"}

    def __init__(self, wave: Optional[QtGui.QImage], duration: float, phrases: list[tuple[str, float]], end: float,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._wave, self._duration, self._phrases, self._end = wave, max(float(duration), 1e-6), phrases, end
        self.setMinimumHeight(self.WAVE_HEIGHT + self.BAR_HEIGHT + 2)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def _color(self, name: str) -> QtGui.QColor:
        from core.marks.phrases import phrase_color
        label = next((label for prefix, label in self._COLORS.items() if name.startswith(prefix)), "OUTRO")
        return QtGui.QColor(*phrase_color(label))

    def paintEvent(self, _event) -> None:
        theme = current_theme()
        painter = QtGui.QPainter(self)
        width = self.width()
        wave_rect = QtCore.QRect(0, 0, width, self.WAVE_HEIGHT)
        painter.fillRect(wave_rect, QtGui.QColor(theme.base))
        if self._wave is not None:
            painter.drawImage(wave_rect, self._wave)
        bar_top = self.WAVE_HEIGHT + 2
        painter.fillRect(QtCore.QRect(0, bar_top, width, self.BAR_HEIGHT), QtGui.QColor(theme.base))
        x = lambda t: int(round(width * min(max(t, 0.0), self._duration) / self._duration))   # noqa: E731
        ends = [start for _name, start in self._phrases[1:]] + [self._end]
        for (name, start), stop in zip(self._phrases, ends):
            rect = QtCore.QRect(x(start), bar_top, max(1, x(stop) - x(start)), self.BAR_HEIGHT)
            painter.fillRect(rect, self._color(name))
            painter.setPen(QtGui.QColor(theme.base))
            painter.drawLine(rect.topLeft(), rect.bottomLeft())
            text = painter.fontMetrics().elidedText(name, QtCore.Qt.ElideRight, rect.width() - 4)
            painter.setPen(QtGui.QColor("#ffffff"))
            painter.drawText(rect.adjusted(3, 0, -1, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, text)
        if not self._phrases:           # rekordbox has none (no phrase analysis yet)
            painter.setPen(QtGui.QColor(theme.text_dim))
            painter.drawText(QtCore.QRect(6, bar_top, width - 12, self.BAR_HEIGHT),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, tr("(none)"))
        painter.end()


def _wave_image(wave, duration: float) -> Optional[QtGui.QImage]:
    """The overview waveform as the overview draws it (Settings > View's style and bands), as an image."""
    if wave is None or duration <= 0:
        return None
    from views.render.wave import band_ranges, render_wave_overview
    view = settings_store.current().viewconfig
    columns = render_wave_overview(wave, band_ranges(view), view.waveform_style, PhraseStrip.WAVE_HEIGHT,
                                   duration_sec=duration, columns=1200)
    pixels = np.ascontiguousarray(columns.transpose(1, 0, 2)[::-1])     # rows top down, bars rising from the bottom
    image = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QtGui.QImage.Format_RGB888)
    return image.copy()


class PhraseWriteDialog(QtWidgets.QDialog):
    """Write Phrases to rekordbox's preview: the waveform with rekordbox's phrases now, an arrow, the waveform with
    the phrases written (third_party.rekordbox_phrases.PhrasePlan), the notes (what rekordbox cannot hold, a
    beatgrid that is not the track's) and the backup made first (none, the .EXT, the track's analysis folder);
    written when accepted."""

    def __init__(self, file_name: str, plan, wave, duration: float, notes: list[str], backup: str = "ext",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Write Phrases to rekordbox"))
        theme = current_theme()
        image = _wave_image(wave, duration)
        span = duration if image is not None else plan.length
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(tr("rekordbox's phrases of '{file}' change:").format(file=file_name))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        arrow = QtGui.QPixmap(str(resource_path("assets/icons/arrow_down.png")))
        for title, rows, end in ((tr("rekordbox now"), plan.now, plan.now_end), (None, None, None),
                                 (tr("Written"), plan.after, plan.after_end)):
            if title is None:
                mark = QtWidgets.QLabel()
                mark.setPixmap(arrow.scaled(20, 20, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                mark.setAlignment(QtCore.Qt.AlignHCenter)
                layout.addWidget(mark)
                continue
            heading = QtWidgets.QLabel(title)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            layout.addWidget(heading)
            strip = PhraseStrip(image, span, rows, end)
            # each phrase with its start, as a list (the strip's names are cut where a phrase is short)
            strip.setToolTip("<p>" + ", ".join(f"{name} {_mmss(t)[:-2]}" for name, t in rows) + "</p>"
                             if rows else tr("(none)"))
            layout.addWidget(strip)
        for text in notes:
            note = QtWidgets.QLabel(text)
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {theme.warning};")
            layout.addWidget(note)
        box = QtWidgets.QGroupBox(tr("Backup before writing"))
        column = QtWidgets.QVBoxLayout(box)
        self._choices: dict[str, QtWidgets.QRadioButton] = {}
        for kind, text, tip in (
            ("", tr("None"), ""),
            ("ext", tr("The analysis file (.EXT)"), tr("The one file the phrases are written into.")),
            ("folder", tr("The track's analysis folder"),
             tr("All of rekordbox's analysis files of the track (.DAT, .EXT, .2EX, .3EX)."))):
            radio = QtWidgets.QRadioButton(text)
            radio.setToolTip(tip)
            radio.setChecked(kind == backup)
            column.addWidget(radio)
            self._choices[kind] = radio
        if not any(r.isChecked() for r in self._choices.values()):
            self._choices["ext"].setChecked(True)
        folder = rekordbox_phrases.backup_folder()
        where = QtWidgets.QLabel(tr("Kept in {folder}; copy it back to undo the write.").format(
            folder=breakable_path(str(folder))))
        where.setWordWrap(True)
        where.setToolTip(str(folder))
        column.addWidget(where)
        layout.addWidget(box)
        layout.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        buttons.addButton(tr("Write Phrases"), QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(760, 520)

    def backup(self) -> str:
        """The backup asked for (rekordbox_phrases.BACKUP_KINDS)."""
        return next(kind for kind, radio in self._choices.items() if radio.isChecked())


def _kept_note(slots: str) -> str:
    """The hot cues the file holds kept in the slots the marks leave (``slots``: their letters), said: those
    written before by Mixlyzer among them."""
    return tr("The file's hot cues {slots} stay: Settings > Library > Serato keeps the file's hot cues in the slots "
              "left, those written here before too (a JumpCUE removed since). Turn that off to replace them all."
              ).format(slots=", ".join(slots))


def breakable_path(path: str) -> str:
    """A path a word-wrapped label can break after each separator (a zero-width space there: a path has no space
    of its own, so a long one would widen its window)."""
    return path.replace("\\", "\\​").replace("/", "/​")


class TagWriteDialog(QtWidgets.QDialog):
    """Write Tags' confirmation: each field that changes, its places in the file, each checked to be written (by
    its name, with its value now), above the value written with an arrow between ("?" when the file could not be
    read; whole, wrapped), a warning where the format's own place is written beside the library's, and the backup
    made first (none, the tag block, the whole file). Nothing checked: no write."""

    BACKUPS = ("", "tags", "file")

    def __init__(self, file_name: str, changes: list[tuple[str, str, list[tuple[str, str, str]], str, str]],
                 backup: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """``changes``: (the field, its name shown, the places written: (the place as write_tags takes it, its name,
        its value now; "" place: not known), the value written, a warning: "" none)."""
        super().__init__(parent)
        self.changes = changes
        self._place_boxes: list[tuple[str, str, QtWidgets.QCheckBox]] = []   # (field, place, its box)
        self.setWindowTitle(tr("Write Tags"))
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(tr("These tags of '{file}' change:").format(file=file_name))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        theme = current_theme()
        arrow = QtGui.QPixmap(str(resource_path("assets/icons/arrow_down.png")))
        changes_body = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(changes_body)
        stack.setContentsMargins(0, 0, 4, 0)
        for field, name, old, new, warning in changes:
            title = QtWidgets.QLabel(name)
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            stack.addWidget(title)
            if warning:
                note = QtWidgets.QLabel(warning)
                note.setWordWrap(True)
                note.setTextFormat(QtCore.Qt.PlainText)
                note.setStyleSheet(f"color: {theme.warning};")
                stack.addWidget(note)
            # the places written, each checked by its name; the arrow; the value written
            boxes = [(place, place_name, text) for place, place_name, text in old] + [("", "", None), ("", "", new)]
            for place, place_name, text in boxes:
                if text is None:
                    mark = QtWidgets.QLabel()
                    mark.setPixmap(arrow.scaled(20, 20, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                    mark.setAlignment(QtCore.Qt.AlignHCenter)
                    stack.addWidget(mark)
                    continue
                if place:
                    check = QtWidgets.QCheckBox(place_name)
                    check.setChecked(True)
                    check.setToolTip(tr("Write the value here (unchecked: this place stays as it is)."))
                    check.toggled.connect(self._update_write)
                    stack.addWidget(check)
                    self._place_boxes.append((field, place, check))
                value = QtWidgets.QLabel(text if text else tr("(empty)"))
                value.setWordWrap(True)
                value.setTextFormat(QtCore.Qt.PlainText)
                value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                value.setStyleSheet(f"QLabel {{ background: {theme.base}; border: 1px solid {theme.border}; "
                                    f"border-radius: 4px; padding: 6px; color: {theme.text if text else theme.text_dim}; }}")
                stack.addWidget(value)
            stack.addSpacing(10)
        stack.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(changes_body)
        layout.addWidget(scroll, 1)
        box = QtWidgets.QGroupBox(tr("Backup before writing"))
        row = QtWidgets.QVBoxLayout(box)
        self._choices = {}
        for kind, text, tip in (
            ("", tr("None"), ""),
            ("tags", tr("The tags only"),
             tr("The bytes the write replaces (a few KB, more with a cover picture): the file is put back exactly, "
                "as long as its audio stays as it is.")),
            ("file", tr("The whole file"), tr("A copy of the whole file, as large as it is."))):
            radio = QtWidgets.QRadioButton(text)
            radio.setToolTip(tip)
            radio.setChecked(kind == backup)
            row.addWidget(radio)
            self._choices[kind] = radio
        if not any(r.isChecked() for r in self._choices.values()):
            self._choices["tags"].setChecked(True)
        where = QtWidgets.QLabel(tr("Kept in {folder}; Restore from Backup puts one back.").format(
            folder=breakable_path(str(tag_backup.folder()))))
        where.setWordWrap(True)
        where.setToolTip(str(tag_backup.folder()))
        row.addWidget(where)
        layout.addWidget(box)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self._write = buttons.addButton(tr("Write Tags"), QtWidgets.QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(600, 680)

    def _update_write(self) -> None:
        self._write.setEnabled(not self._place_boxes or any(box.isChecked() for _f, _p, box in self._place_boxes))

    def places(self) -> dict[str, list[str]]:
        """The places checked, by field (a field with none: not written)."""
        out: dict[str, list[str]] = {}
        for field, place, box in self._place_boxes:
            if box.isChecked():
                out.setdefault(field, []).append(place)
        return out

    def backup(self) -> str:
        """The backup asked for: "" none, "tags" the tag block, "file" the whole file."""
        return next(kind for kind, radio in self._choices.items() if radio.isChecked())
