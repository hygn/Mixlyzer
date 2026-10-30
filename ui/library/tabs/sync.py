"""The track panel's Sync tab: the track's analysis and stored values taken to other programs, in place: its audio
file's tags (Write Tags, Restore from Backup) and the Rekordbox XML's entry for it."""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Callable, Optional

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
from third_party import serato
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
        self._rekordbox_asked: set[str] = set()     # the uids this tab asked to sync, their outcome still to show
        self._sig_tags_read.connect(self._confirm_tags)
        self._sig_tags_done.connect(self._on_tags_done)
        self._sig_restored.connect(self._on_restored)
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
                                  "serato_markers_old": tr("Serato's old cue data"),
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
        tag_btns = QtWidgets.QHBoxLayout()
        tag_btns.addWidget(self.btn_restore_tags)
        tag_btns.addStretch(1)
        tag_btns.addWidget(self.btn_write_tags)
        tags_layout = QtWidgets.QVBoxLayout(tags)
        tags_layout.addLayout(grid)
        tags_layout.addWidget(self.tag_note)
        tags_layout.addLayout(tag_btns)

        rekordbox = QtWidgets.QGroupBox(tr("Rekordbox XML"))
        rekordbox.setToolTip(tr(
            "<p>The track written into the Rekordbox XML again, as saving an edit does: its beatgrid, CUE points, "
            "JumpCUEs and values as stored now. The XML's file and its options are in Settings > Library.</p>"))
        self.rekordbox_note = QtWidgets.QLabel("")
        self.rekordbox_note.setWordWrap(True)
        self.btn_rekordbox_sync = QtWidgets.QPushButton(tr("Sync to Rekordbox XML"))
        self.btn_rekordbox_sync.setToolTip(tr("Write this track into the Rekordbox XML again."))
        self.btn_rekordbox_sync.clicked.connect(self.sync_rekordbox)
        rekordbox_btns = QtWidgets.QHBoxLayout()
        rekordbox_btns.addStretch(1)
        rekordbox_btns.addWidget(self.btn_rekordbox_sync)
        rekordbox_layout = QtWidgets.QVBoxLayout(rekordbox)
        rekordbox_layout.addWidget(self.rekordbox_note)
        rekordbox_layout.addLayout(rekordbox_btns)
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

    def _update_rekordbox_controls(self) -> None:
        ready = self._rekordbox_ready()
        self.btn_rekordbox_sync.setEnabled(ready and self._track is not None and bool(self._track.uid)
                                           and self._bus is not None)
        off = tr("Sync with Rekordbox XML is off, or has no file: Settings > Library > Rekordbox XML.")
        if not ready:
            self._set_rekordbox_note(off)
        elif self.rekordbox_note.text() == off:
            self._set_rekordbox_note("")

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
        its old Markers_ emptied when it has one; returns what Serato loses (third_party.serato's losses)."""
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
        if now.get("serato_markers_old"):
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
            if field == "serato_markers_old":
                warning = tr("Serato reads this before the hot cues written here: it is emptied (its loops and "
                             "names are in the hot cues' tag too).")
            changes.append((field, self._field_names.get(field, field), olds, new, warning))
        if not changes:
            self._tags_writing = None
            self._update_tag_controls()
            notes = [tr("The file's tags already hold these values.")]
            notes += [_kept_note(loss[len("kept:"):]) for loss in losses if loss.startswith("kept:")]
            self._set_tag_note("\n".join(notes))
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
    return tr("Serato's old cue data") if text else ""


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
