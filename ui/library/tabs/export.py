from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import soundfile as sf
from core.tieups import domain as tieup
from system import os_support
from system.os_support import path_key
from system.atomic_io import atomic_output_path
from system.resource_paths import resource_path
from core.analysis_library import jobs as analysis_jobs
from core.analysis_library.stored import StoredAnalysis
from core.library import jobs
from core.library.records import TrackRow
from core.library.service import LibraryService
from analyzer.audio.decoder import decode_to_memmap, get_samplerate
from core.tags import backup as tag_backup
from core.tags.writer import TEXT_FIELDS, TagWriteError, bpm_text, planned_writes, tag_format, write_tags
from core.track.model import DataModel
from third_party.midi_tempo_map import TempoMapError, midi_tempo_map
from third_party.rekordbox import sanitize_filename
from ui.config.scroll import scroll_page
from ui.widgets.elided_label import ElidedLabel
from analyzer.music.marks import JumpCue, JumpCueGroup
from ui.render.jumprender import jump_renderer
from analyzer.music.key_names import idx_to_labels
from ui.theme import current_theme
from system.i18n import tr


class ExportTrackPanel(QtWidgets.QWidget):
    """The library panel's Export tab: a track's audio rendered with a JumpCUE jump baked in (WAV), one direction
    of one JumpCUE pair (prefers the in-memory DataModel for the loaded track, else the track's NPZ); its beatgrid
    as a DAW's tempo map (a MIDI file: third_party.midi_tempo_map, the marks taken the same way); and the
    track's stored values written into its audio file's tags (``core.tags.writer``: the fields checked,
    the file's other tags and its audio kept), after a confirmation showing each change (``TagWriteDialog``) with
    the backup it asks for made first (core.tags.backup: none, the tag block, the whole file); a backup put back
    (Restore from Backup)."""

    _sig_tags_read = QtCore.Signal(str, object, object)   # the track's uid, the planned writes (None: unread), the write
    _sig_tags_done = QtCore.Signal(str, str, str)   # the track's uid, the error ("": written), the backup made ("")
    _sig_restored = QtCore.Signal(str, str)          # the track's uid, the error ("": put back)
    _backup = "tags"     # the backup the last confirmation asked for (the session's: the next one's choice)

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
        self._sig_tags_read.connect(self._confirm_tags)
        self._sig_tags_done.connect(self._on_tags_done)
        self._sig_restored.connect(self._on_restored)
        self._track: Optional[TrackRow] = None
        self._source: Optional[Path] = None
        self._jump_cues: Optional[list[JumpCueGroup]] = None   # the track's JumpCUE groups (None: not read yet)
        self._analysis: Optional[StoredAnalysis] = None        # the track's stored analysis (None: not read yet)
        self._jump_pairs: list[tuple] = []     # (JumpCue, JumpCue): every two cues of a group

        self.lbl_title = ElidedLabel("-")
        self.lbl_artist = ElidedLabel("-")
        self.lbl_source = ElidedLabel("-", mode=QtCore.Qt.ElideMiddle)

        self.dest_edit = QtWidgets.QLineEdit()
        self.dest_browse = QtWidgets.QPushButton(tr("Browse..."))
        self.dest_browse.clicked.connect(self._on_browse)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #ff8080;")

        dest_layout = QtWidgets.QHBoxLayout()
        dest_layout.addWidget(self.dest_edit, 1)
        dest_layout.addWidget(self.dest_browse)

        self.cmb_jump_pair = QtWidgets.QComboBox()
        self.cmb_jump_pair.setEnabled(False)
        self.cmb_jump_pair.setPlaceholderText(tr("No JumpCUE directions"))
        self.cmb_jump_pair.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_jump_pair.setMinimumContentsLength(8)

        track_form = QtWidgets.QFormLayout()
        track_form.addRow(tr("Title"), self.lbl_title)
        track_form.addRow(tr("Artist"), self.lbl_artist)
        track_form.addRow(tr("Source"), self.lbl_source)
        track_form.addRow(tr("Destination Folder"), dest_layout)

        jump = QtWidgets.QGroupBox(tr("Prerender JumpCUE Audio"))
        jump.setToolTip(tr("Write WAV files with JumpCUE skips baked in."))
        form = QtWidgets.QFormLayout()
        form.addRow("JumpCUE", self.cmb_jump_pair)
        self.btn_export = QtWidgets.QPushButton(tr("Export"))
        self.btn_export.clicked.connect(self._on_export)
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_export)
        jump_layout = QtWidgets.QVBoxLayout(jump)
        jump_layout.addLayout(form)
        jump_layout.addLayout(btns)

        tempo_map = QtWidgets.QGroupBox(tr("DAW Tempo Map"))
        tempo_map.setToolTip(tr(
            "<p>A MIDI file of the track's beatgrid for a DAW: its tempo (ramps too), bars and key changes. Put the "
            "audio at the project's start and import the file's tempo map there.</p>"))
        self.btn_tempo_map = QtWidgets.QPushButton(tr("Export MIDI"))
        self.btn_tempo_map.clicked.connect(self.export_tempo_map)
        tempo_btns = QtWidgets.QHBoxLayout()
        tempo_btns.addStretch(1)
        tempo_btns.addWidget(self.btn_tempo_map)
        QtWidgets.QVBoxLayout(tempo_map).addLayout(tempo_btns)

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
        self.tag_note = QtWidgets.QLabel("")
        self.tag_note.setWordWrap(True)
        self.btn_write_tags = QtWidgets.QPushButton(tr("Write Tags"))
        self.btn_write_tags.setToolTip(tr("Write the checked fields' stored values into the file's tags."))
        self.btn_write_tags.clicked.connect(self.write_tags)
        self.btn_restore_tags = QtWidgets.QPushButton(tr("Restore from Backup..."))
        self.btn_restore_tags.setToolTip(tr(
            "<p>Put the track's file back as a backup made before writing its tags holds it: a tag backup "
            "(.mxtags) or a copy of the whole file.</p>"))
        self.btn_restore_tags.clicked.connect(self.restore_tags)
        tag_btns = QtWidgets.QHBoxLayout()
        tag_btns.addWidget(self.btn_restore_tags)
        tag_btns.addStretch(1)
        tag_btns.addWidget(self.btn_write_tags)
        tags_layout = QtWidgets.QVBoxLayout(tags)
        tags_layout.addLayout(grid)
        tags_layout.addWidget(self.tag_note)
        tags_layout.addLayout(tag_btns)

        # the fields scroll (as the Edit tab's): the panel never holds the library taller than a small window
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.addLayout(track_form)
        layout.addWidget(jump)
        layout.addWidget(tempo_map)
        layout.addWidget(tags)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll_page(body), 1)

    def track_uid(self) -> Optional[str]:
        return self._track.uid if self._track is not None else None

    def set_track(self, track: TrackRow) -> None:
        self._track = track
        self._jump_cues = None
        self._analysis = None
        self._jump_pairs = []
        self.cmb_jump_pair.clear()
        self.cmb_jump_pair.setEnabled(False)
        self.lbl_title.setText(track.title or "-")
        self.lbl_artist.setText(track.artist or "-")

        resolved = self._resolve_path(track.path) or self._resolve_path(getattr(self._model, "properties", {}).get("path") if self._model else None)
        self._source = resolved if resolved else None
        self.lbl_source.setText(str(resolved) if resolved else (track.path or "-"))
        self.status_label.clear()

        if resolved and resolved.exists():
            self.dest_edit.setText(str(resolved.parent))
        elif resolved:
            self.dest_edit.setText(str(resolved.parent))
            self.status_label.setText(tr("Source file not found at the specified path."))
        else:
            self.dest_edit.clear()
            self.status_label.setText(tr("Source path is invalid."))
        self._set_tag_note("")
        self._update_tag_controls()
        self._populate_jump_pairs()
        self._read_jump_cues()

    # File tags
    def _tag_format(self) -> Optional[str]:
        return tag_format(self._source) if self._source is not None and self._source.is_file() else None

    def _update_tag_controls(self) -> None:
        fmt = self._tag_format()
        for box in self.tag_fields.values():
            box.setEnabled(fmt is not None)
        self.cb_comment_tieups.setEnabled(fmt is not None and self.tag_fields["comment"].isChecked())
        self.btn_write_tags.setEnabled(fmt is not None and self._tags_writing is None)
        self.btn_restore_tags.setEnabled(self._source is not None and self._source.is_file()
                                         and self._tags_writing is None)
        if self._source is not None and self._source.is_file() and fmt is None:
            self._set_tag_note(tr("Tags cannot be written to this file's format."))

    def _set_tag_note(self, text: str, *, error: bool = False) -> None:
        self.tag_note.setText(text)
        self.tag_note.setStyleSheet(f"color: {current_theme().danger};" if error else "")

    def write_tags(self) -> None:
        """The checked fields' stored values (read from the library now) written into the track's file, in a
        thread; the result shown under the fields."""
        if self._track is None or not self._track.uid or self._tags_writing is not None:
            return
        if self._tag_format() is None:
            self._update_tag_controls()
            return
        fields = [field for field, box in self.tag_fields.items() if box.isChecked()]
        if not fields:
            self._set_tag_note(tr("Check the fields to write."), error=True)
            return
        uid = self._track.uid
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
                           done=lambda found: self._write_stored(uid, fields, *found))

    def _write_stored(self, uid: str, fields: list, row: Optional[TrackRow], suffix: str) -> None:
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
        write = (str(self._source), row.path, values)
        threading.Thread(target=self._read_tags_job, args=(uid, write), daemon=True).start()

    def _read_tags_job(self, uid: str, write: tuple) -> None:
        """A worker thread: where the tags go and what those places hold now (``planned_writes``, None: the file
        not read), for the confirmation. The fields' sources are the library's (where the analysis read them); a
        track without them (one added before they were kept, or given another file) has them found now as the
        analysis would read the file, kept with the write. The library's part asked of its owner
        (LibraryService.call)."""
        from core.tags.reader import extract_tags, read_tag_sources
        file_path, stored_path, values = write
        sources, kept = {}, {}
        try:
            sources = self._library.call(jobs.tag_sources, uid)
            missing = [f for f in TEXT_FIELDS if f in values and f not in sources]
            if missing:
                found = read_tag_sources(file_path, *extract_tags(file_path))
                if found:
                    kept = {f: found[f] for f in missing}
                    sources = {**sources, **kept}
            plan = planned_writes(file_path, values, sources)
        except (TagWriteError, OSError):
            plan = None
        self._sig_tags_read.emit(uid, plan, (file_path, stored_path, values, sources, kept))

    def _confirm_tags(self, uid: str, plan: Optional[dict], write: tuple) -> None:
        """The changes shown (``plan``: core.tags.writer.planned_writes, None: the file not read), the backup asked
        for; written when confirmed. A field changes where a place it is written to does not hold it yet; each
        place shown with its value (two or more: each by its name), and a warning where the format's own place is
        written beside the one the library read (a store's M4A with a blank ©cmt and its comment in desc).
        Each place is checked to be written (unchecked: left as it is). Nothing changing: said, not written."""
        file_path, stored_path, values, sources, kept = write
        changes = []
        for field, value in values.items():
            new = bpm_text(value) if field == "bpm" else str(value or "")
            places = (plan or {}).get(field)
            if plan is not None and all(now == after for _s, _name, now, after, _kind in places or ()):
                continue
            olds = ([(spelling, name, " / ".join(v.strip() for v in now if v.strip()))
                     for spelling, name, now, _after, _kind in places] if places else [("", "", "?")])
            warning = ""
            own = next((p for p in places or () if p[4] == "own"), None)
            if own is not None and any(p[4] == "source" for p in places) and own[2] != own[3]:
                there = " / ".join(v.strip() for v in own[2] if v.strip())
                warning = tr("Also written to {place}, the field DJ software (rekordbox ...) reads. It holds "
                             "now: {value}").format(place=own[1], value=there or tr("(empty)"))
            changes.append((field, self._field_names.get(field, field), olds, new, warning))
        if not changes:
            self._tags_writing = None
            self._update_tag_controls()
            self._set_tag_note(tr("The file's tags already hold these values."))
            return
        dialog = TagWriteDialog(Path(file_path).name, changes, type(self)._backup, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            self._tags_writing = None
            self._update_tag_controls()
            self._set_tag_note("")
            return
        type(self)._backup = dialog.backup()
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
        """A backup chosen (core.tags.backup's folder) and the track's file put back as it holds it, after a
        confirmation, on a thread of its own."""
        if self._track is None or self._source is None or not self._source.is_file() or self._tags_writing:
            return
        start = tag_backup.folder()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Restore from Backup"), str(start if start.is_dir() else self._source.parent),
            tr("Tag backups (*.mxtags);;All files (*)"))
        if not path:
            return
        try:
            name = tag_backup.backed_up_name(path)
        except (tag_backup.BackupError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, tr("Restore from Backup"), str(exc))
            return
        question = tr("Put '{file}' back as the backup '{backup}' holds it?").format(
            file=self._source.name, backup=Path(path).name)
        if name and name != self._source.name:
            question += "\n" + tr("The backup was made of '{name}'.").format(name=name)
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

    def _resolve_path(self, raw: str | None) -> Optional[Path]:
        if not raw:
            return None
        candidate: Optional[str]
        if raw.startswith("file://"):
            parsed = urlparse(raw)
            path_part = unquote(parsed.path or "")
            if parsed.netloc and parsed.netloc not in ("", "localhost"):
                path_part = f"/{parsed.netloc}{path_part}"
            if os_support.IS_WINDOWS and path_part.startswith("/"):
                path_part = path_part.lstrip("/")
            candidate = path_part
        else:
            candidate = unquote(raw)
        try:
            return Path(candidate).expanduser()
        except (TypeError, ValueError):
            return None

    def _on_browse(self) -> None:
        current = self.dest_edit.text().strip()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            tr("Select Destination Folder"),
            current or (str(self._source.parent) if self._source else os.getcwd()),
        )
        if directory:
            self.dest_edit.setText(directory)

    def _destination(self) -> Optional[Path]:
        """The destination folder (made if need be), or None (said)."""
        if self._track is None:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("No track is selected."))
            return None
        dest_text = self.dest_edit.text().strip()
        if not dest_text:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("Please select a destination folder."))
            return None
        dest_dir = Path(dest_text).expanduser()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, tr("Export Track"), tr("Cannot create destination folder:\n{error}").format(error=exc))
            return None
        return dest_dir

    def _overwrite(self, path: Path) -> bool:
        """``path`` may be written: it does not exist, or the user said to overwrite it."""
        if not path.exists():
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            tr("Export Track"),
            tr("'{name}' already exists.\nOverwrite?").format(name=path.name),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def export_tempo_map(self) -> None:
        """The track's beatgrid as a MIDI file (third_party.midi_tempo_map): the deck's marks when it holds the
        track (as edited), else the stored ones."""
        dest_dir = self._destination()
        if dest_dir is None:
            return
        if self._model_holds_track():
            m = self._model
            marks = (m.beats, m.tempo_segments, m.key_segments, m.duration_sec)
        elif self._analysis is not None:
            a = self._analysis
            marks = (a.beats, a.tempo_segments, a.key_segments, a.duration_sec)
        else:        # its stored analysis still being read (set_track asked for it): written when it comes
            uid = self._track.uid

            def came(analysis) -> None:
                if self._track is None or self._track.uid != uid:
                    return
                if analysis is None:
                    QtWidgets.QMessageBox.warning(self, tr("Export Track"),
                                                  tr("The track's analysis could not be read."))
                    return
                self._analysis = analysis
                self.export_tempo_map()

            self._library.read(analysis_jobs.analysis, uid, done=came)
            return
        title = self._track.title or (self._source.stem if self._source else "")
        try:
            data = midi_tempo_map(*marks[:3], title=title, duration_sec=float(marks[3] or 0.0))
        except TempoMapError:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("The track has no beatgrid to write."))
            return
        out_path = dest_dir / f"{sanitize_filename(title or 'track')}_tempo_map.mid"
        if not self._overwrite(out_path):
            return
        try:
            with atomic_output_path(out_path) as temporary:
                temporary.write_bytes(data)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, tr("Export Track"), tr("The file could not be written: {error}").format(error=exc))
            return
        QtWidgets.QMessageBox.information(self, tr("Export Track"), tr("Export completed:\n{files}").format(
            files=str(out_path)))

    def _on_export(self) -> None:
        dest_dir = self._destination()
        if dest_dir is None:
            return

        if self._load_jump_cues() is None:
            return

        if not self._jump_pairs:
            self._populate_jump_pairs()
        selection = self._selected_jump_pair()
        if selection is None:
            QtWidgets.QMessageBox.warning(
                self,
                tr("Export Track"),
                tr("Select a JumpCUE direction to render."),
            )
            return
        src, dst = selection
        exported = self._write_jumpcue_audio(dest_dir, [(src, dst)])
        if exported is None:
            return

        lines = "\n".join(f"{label}: {path}" for label, path in exported)
        QtWidgets.QMessageBox.information(self, tr("Export Track"), tr("Export completed:\n{files}").format(files=lines))

    def _load_jump_cues(self) -> Optional[list[JumpCueGroup]]:
        """The track's JumpCUE groups: the deck's (as edited) when it holds the track, else the stored ones as read
        on the library thread (set_track asks for them; None until they come)."""
        if self._jump_cues is not None:
            return self._jump_cues
        if self._track is None or not self._track.uid:
            return None
        # Prefer in-memory model data when it matches the requested track.
        if self._model_holds_track():
            self._jump_cues = list(self._model.jump_cues)
            return self._jump_cues
        return None

    def _model_holds_track(self) -> bool:
        """The deck holds this track (its marks as edited are the model's)."""
        if self._model is None or self._track is None:
            return False
        model_path = (getattr(self._model, "properties", {}) or {}).get("path")
        return bool(model_path and self._track.path and path_key(model_path) == path_key(self._track.path))

    def _read_jump_cues(self) -> None:
        """The track's stored analysis asked of the library's owner (not the deck's track); the JumpCUE pairs
        listed when it comes."""
        if self._track is None or not self._track.uid or self._load_jump_cues() is not None:
            return
        uid = self._track.uid

        def came(analysis) -> None:
            if self._track is None or self._track.uid != uid:
                return
            if analysis is None:
                self.status_label.setText(tr("The track's analysis could not be read."))
                return
            self._analysis = analysis
            self._jump_cues = list(analysis.jump_cues)
            self._populate_jump_pairs()

        self._library.read(analysis_jobs.analysis, uid, done=came)

    def _populate_jump_pairs(self) -> None:
        self.cmb_jump_pair.clear()
        self._jump_pairs = []
        groups = self._load_jump_cues()
        if not groups:
            self.cmb_jump_pair.setEnabled(False)
            return
        # every two cues of a group, each way (a one-cue group has no jump)
        pairs = [(group.cues[i], group.cues[j]) for group in groups
                 for i in range(len(group.cues)) for j in range(i + 1, len(group.cues))]
        self._jump_pairs = pairs
        for idx, (a, b) in enumerate(pairs):
            fwd, bwd = a.label.strip(), b.label.strip()
            self.cmb_jump_pair.addItem(f"{idx + 1}: {fwd}->{bwd}", (a, b))
            self.cmb_jump_pair.addItem(f"{idx + 1}: {bwd}->{fwd}", (b, a))
        self.cmb_jump_pair.setEnabled(bool(pairs))

    def _selected_jump_pair(self) -> Optional[tuple[JumpCue, JumpCue]]:
        """The jump picked: (the cue jumped from, the cue jumped to)."""
        if self.cmb_jump_pair.count() <= 0:
            return None
        idx = int(self.cmb_jump_pair.currentIndex())
        if idx < 0:
            return None
        data = self.cmb_jump_pair.itemData(idx)
        if not isinstance(data, tuple) or len(data) != 2 or not all(isinstance(c, JumpCue) for c in data):
            return None
        return data

    def _write_jumpcue_audio(
        self,
        dest_dir: Path,
        jumps: list[tuple[JumpCue, JumpCue]],
    ) -> Optional[list[tuple[str, Path]]]:
        assert self._track is not None

        audio_path = self._source or self._resolve_path(self._track.path)
        assert audio_path is not None
        sr_probe = get_samplerate(audio_path.as_posix())
        sr = sr_probe or 44100
        pcm_arr = decode_to_memmap(audio_path.as_posix(), sr, 2)
        pcm_arr = np.asarray(pcm_arr, dtype=np.float32)
        if pcm_arr.shape[1] == 1:
            pcm_arr = np.repeat(pcm_arr, 2, axis=1)

        total_samples = int(pcm_arr.shape[0])
        max_time = (total_samples - 1) / sr if total_samples > 0 else 0.0
        exports: list[tuple[str, Path]] = []
        title_base = sanitize_filename(self._track.title or audio_path.stem)

        for src, dst in jumps:
            src_label = src.label.strip()
            dst_label = dst.label.strip()

            start = float(np.clip(float(src.point), 0.0, max_time))
            dest = float(np.clip(float(dst.point), 0.0, max_time))

            rendered_channels = []
            for ch_idx in range(pcm_arr.shape[1]):
                rendered, _ = jump_renderer(
                    np.array(pcm_arr[:, ch_idx], copy=True),
                    sr,
                    start,
                    dest,
                    128,
                )
                rendered_channels.append(rendered)
            rendered_pcm = np.stack(rendered_channels, axis=1)
            rendered_pcm = np.clip(rendered_pcm, -1.0, 1.0).astype(np.float32, copy=False)

            safe_label = sanitize_filename(f"{src_label}_to_{dst_label}")
            filename = f"{title_base}_jump_{safe_label}.wav"
            out_path = dest_dir / filename
            if not self._overwrite(out_path):
                continue

            try:
                sf.write(out_path, rendered_pcm, sr)
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self, tr("Export Track"), tr("Failed to write JumpCUE audio:\n{error}").format(error=exc)
                )
                return None

            exports.append((f"JumpCUE {src_label}->{dst_label}", out_path))

        return exports if exports else None


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
