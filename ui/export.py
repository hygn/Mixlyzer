from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import numpy as np
from PySide6 import QtCore, QtWidgets
import soundfile as sf
from core import os_support
from core.analysis_lib_handler import FeatureNPZStore
from core.library_handler import LibraryDB, TrackRow
from core.audio.decoder import decode_to_memmap, get_samplerate
from core.audio.tag_writer import TEXT_FIELDS, TagWriteError, tag_format, write_tags
from core.model import DataModel
from third_party.rekordbox import sanitize_filename
from ui.elided_label import ElidedLabel
from utils.jump_cues import extract_jump_cue_pairs
from utils.jumprender import jump_renderer
from utils.labels import idx_to_labels
from ui.theme import current_theme
from core.i18n import tr


class ExportTrackPanel(QtWidgets.QWidget):
    """The library panel's Export tab: a track's audio rendered with a JumpCUE jump baked in (WAV), one direction
    of one JumpCUE pair (prefers the in-memory DataModel for the loaded track, else the track's NPZ); and the
    track's stored values written into its audio file's tags (``core.audio.tag_writer``: the fields checked,
    the file's other tags and its audio kept)."""

    _sig_tags_done = QtCore.Signal(str, str)   # the track's uid, the error ("": written)

    def __init__(self, lib_dir: str, parent: Optional[QtWidgets.QWidget] = None, *, model: Optional[DataModel] = None,
                 bus=None, unapplied_edits: Optional[Callable[[str], bool]] = None) -> None:
        """``unapplied_edits(uid)``: whether the Edit tab has changes to that track not applied yet."""
        super().__init__(parent)
        self._lib_dir = lib_dir
        self._model: Optional[DataModel] = model
        self._bus = bus
        self._unapplied_edits = unapplied_edits
        self._tags_writing: Optional[str] = None   # the uid whose tags are being written
        self._sig_tags_done.connect(self._on_tags_done)
        self._track: Optional[TrackRow] = None
        self._source: Optional[Path] = None
        self._features: Optional[dict[str, np.ndarray]] = None
        self._jump_pairs: list[dict] = []

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

        jump = QtWidgets.QGroupBox(tr("Prerender JumpCUE Audio"))
        jump.setToolTip(tr("Write WAV files with JumpCUE skips baked in."))
        form = QtWidgets.QFormLayout()
        form.addRow(tr("Destination Folder"), dest_layout)
        form.addRow("JumpCUE", self.cmb_jump_pair)
        self.btn_export = QtWidgets.QPushButton(tr("Export"))
        self.btn_export.clicked.connect(self._on_export)
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_export)
        jump_layout = QtWidgets.QVBoxLayout(jump)
        jump_layout.addLayout(form)
        jump_layout.addLayout(btns)

        tags = QtWidgets.QGroupBox(tr("File Tags"))
        tags.setToolTip(tr(
            "<p>Write the library's values into the audio file's own tags: only the fields checked change; the "
            "file's other tags, its artwork and its audio stay as they are. FLAC, MP3, AAC, WAV, AIFF and MP4 / "
            "M4A files.</p>"
        ))
        self.tag_fields: dict[str, QtWidgets.QCheckBox] = {}
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
        self.tag_note = QtWidgets.QLabel("")
        self.tag_note.setWordWrap(True)
        self.btn_write_tags = QtWidgets.QPushButton(tr("Write Tags"))
        self.btn_write_tags.setToolTip(tr("Write the checked fields' stored values into the file's tags."))
        self.btn_write_tags.clicked.connect(self.write_tags)
        tag_btns = QtWidgets.QHBoxLayout()
        tag_btns.addStretch(1)
        tag_btns.addWidget(self.btn_write_tags)
        tags_layout = QtWidgets.QVBoxLayout(tags)
        tags_layout.addLayout(grid)
        tags_layout.addWidget(self.tag_note)
        tags_layout.addLayout(tag_btns)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(track_form)
        layout.addWidget(jump)
        layout.addWidget(tags)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def track_uid(self) -> Optional[str]:
        return self._track.uid if self._track is not None else None

    def set_track(self, track: TrackRow) -> None:
        self._track = track
        self._features = None
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

    # File tags
    def _tag_format(self) -> Optional[str]:
        return tag_format(self._source) if self._source is not None and self._source.is_file() else None

    def _update_tag_controls(self) -> None:
        fmt = self._tag_format()
        for box in self.tag_fields.values():
            box.setEnabled(fmt is not None)
        self.btn_write_tags.setEnabled(fmt is not None and self._tags_writing is None)
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
        db = LibraryDB(os.path.join(self._lib_dir, "library.db"))
        db.connect()
        try:
            row = db.get_by_uid(uid)
        finally:
            db.close()
        if row is None:
            self._set_tag_note(tr("The track is no longer in the library."), error=True)
            return
        stored = {
            "title": row.title or "",
            "artist": row.artist or "",
            "album": row.album or "",
            "comment": row.comment or "",
            "bpm": row.bpm,
            "key": idx_to_labels(row.key)[1] if row.key is not None else "",
        }
        values = {field: stored[field] for field in fields}
        self._tags_writing = uid
        self.btn_write_tags.setEnabled(False)
        self._set_tag_note(tr("Writing the tags..."))
        threading.Thread(target=self._write_tags_job, args=(uid, str(self._source), row.path, values),
                         daemon=True).start()

    def _write_tags_job(self, uid: str, file_path: str, stored_path: str, values: dict) -> None:
        """A worker thread (no widgets here): the tags written where the analysis read the fields from (their
        sources in the library; a track without them, as one added before they were kept or given another
        file: found now as the analysis would read the file, and kept), the file's new size and time stored."""
        from analyzer_core.global_analyzer import extract_tags, read_tag_sources
        try:
            db = LibraryDB(os.path.join(self._lib_dir, "library.db"))
            db.connect()
            try:
                sources = db.tag_sources(uid)
                missing = [f for f in TEXT_FIELDS if f in values and f not in sources]
                if missing:
                    found = read_tag_sources(file_path, *extract_tags(file_path))
                    if found:
                        db.set_tag_sources(uid, {f: found[f] for f in missing})
                        sources.update({f: found[f] for f in missing})
                write_tags(file_path, values, sources)
                db.touch_file_stats(stored_path)
            finally:
                db.close()
            error = ""
        except TagWriteError as exc:
            error = str(exc)
        except OSError as exc:
            error = tr("The file could not be written: {error}").format(error=exc.strerror or exc)
        self._sig_tags_done.emit(uid, error)

    def _on_tags_done(self, uid: str, error: str) -> None:
        self._tags_writing = None
        if self._track is not None and self._track.uid == uid:
            self._set_tag_note(error or tr("The tags were written to the file."), error=bool(error))
        elif error:
            QtWidgets.QMessageBox.warning(self, tr("Write Tags"), error)
        self._update_tag_controls()
        if not error and self._bus is not None:
            self._bus.sig_rekordbox_sync_track_requested.emit(uid)   # the file's size in the XML

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

    def _on_export(self) -> None:
        if self._track is None:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("No track is selected."))
            return

        dest_text = self.dest_edit.text().strip()
        if not dest_text:
            QtWidgets.QMessageBox.warning(self, tr("Export Track"), tr("Please select a destination folder."))
            return
        dest_dir = Path(dest_text).expanduser()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, tr("Export Track"), tr("Cannot create destination folder:\n{error}").format(error=exc))
            return

        features = self._load_features()
        if features is None:
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
        pair, directions = selection
        exported = self._write_jumpcue_audio(dest_dir, features, pair, [directions])
        if exported is None:
            return

        lines = "\n".join(f"{label}: {path}" for label, path in exported)
        QtWidgets.QMessageBox.information(self, tr("Export Track"), tr("Export completed:\n{files}").format(files=lines))

    def _load_features(self) -> Optional[dict[str, np.ndarray]]:
        if self._features is not None:
            return self._features
        if self._track is None or not self._track.uid:
            return None
        # Prefer in-memory model data when it matches the requested track.
        if self._model is not None:
            props = getattr(self._model, "properties", {}) or {}
            model_path = props.get("path")
            if model_path and self._track.path and os.path.normcase(os.path.abspath(model_path)) == os.path.normcase(os.path.abspath(self._track.path)):
                self._features = getattr(self._model, "features", None)
                if self._features:
                    return self._features
        store = FeatureNPZStore(base_dir=self._lib_dir, compressed=True)
        try:
            self._features = store.load(self._track.uid)
        except (OSError, ValueError):
            self.status_label.setText(tr("The track's analysis could not be read."))
            return None
        return self._features

    def _populate_jump_pairs(self) -> None:
        self.cmb_jump_pair.clear()
        self._jump_pairs = []
        features = self._load_features()
        if not features:
            self.cmb_jump_pair.setEnabled(False)
            return
        # One-cue groups are pairs of a cue with itself; there is no jump in them.
        pairs = [
            pair for pair in extract_jump_cue_pairs(features)
            if pair.get("forward", {}).get("point") != pair.get("backward", {}).get("point")
        ]
        self._jump_pairs = pairs
        for idx, pair in enumerate(pairs):
            fwd = str(pair.get("forward", {}).get("label", "")).strip()
            bwd = str(pair.get("backward", {}).get("label", "")).strip()
            self.cmb_jump_pair.addItem(f"{idx + 1}: {fwd}->{bwd}", (pair, ("forward", "backward")))
            self.cmb_jump_pair.addItem(f"{idx + 1}: {bwd}->{fwd}", (pair, ("backward", "forward")))
        self.cmb_jump_pair.setEnabled(bool(pairs))

    def _selected_jump_pair(self) -> Optional[tuple[dict, tuple[str, str]]]:
        if self.cmb_jump_pair.count() <= 0:
            return None
        idx = int(self.cmb_jump_pair.currentIndex())
        if idx < 0:
            return None
        data = self.cmb_jump_pair.itemData(idx)
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        pair, direction = data
        if not isinstance(pair, dict) or not isinstance(direction, tuple) or len(direction) != 2:
            return None
        return pair, direction

    def _write_jumpcue_audio(
        self,
        dest_dir: Path,
        features: dict[str, np.ndarray],
        pair: dict,
        directions: list[tuple[str, str]],
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

        for src_role, dst_role in directions:
            src = pair.get(src_role, {})
            dst = pair.get(dst_role, {})
            src_label = str(src.get("label", "")).strip()
            dst_label = str(dst.get("label", "")).strip()

            start = float(np.clip(float(src.get("point")), 0.0, max_time))
            dest = float(np.clip(float(dst.get("point")), 0.0, max_time))

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
            if out_path.exists():
                reply = QtWidgets.QMessageBox.question(
                    self,
                    tr("Export Track"),
                    tr("'{name}' already exists.\nOverwrite?").format(name=out_path.name),
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
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
