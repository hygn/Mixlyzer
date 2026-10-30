from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import numpy as np
from PySide6 import QtCore, QtWidgets
import soundfile as sf
from system import os_support
from system.os_support import path_key
from system.atomic_io import atomic_output_path
from core.analysis_library import jobs as analysis_jobs
from core.analysis_library.stored import StoredAnalysis
from core.library.records import TrackRow
from core.library.service import LibraryService
from analyzer.audio.decoder import decode_to_memmap, get_samplerate
from core.track.model import DataModel
from third_party.midi_tempo_map import TempoMapError, midi_tempo_map
from third_party.rekordbox import sanitize_filename
from ui.config.scroll import scroll_page
from ui.widgets.elided_label import ElidedLabel
from analyzer.music.marks import JumpCue, JumpCueGroup
from ui.render.jumprender import jump_renderer
from system.i18n import tr


class ExportTrackPanel(QtWidgets.QWidget):
    """The library panel's Export tab: a track's audio rendered with a JumpCUE jump baked in (WAV), one direction
    of one JumpCUE pair (prefers the in-memory DataModel for the loaded track, else the track's NPZ); its beatgrid
    as a DAW's tempo map (a MIDI file: third_party.midi_tempo_map, the marks taken the same way). Its file's tags
    are the Sync tab's (ui.library.tabs.sync)."""

    def __init__(self, library: LibraryService, parent: Optional[QtWidgets.QWidget] = None, *,
                 model: Optional[DataModel] = None) -> None:
        """``library``: the library's owner (core.library.service: the analysis)."""
        super().__init__(parent)
        self._library = library
        self._model: Optional[DataModel] = model
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

        # the fields scroll (as the Edit tab's): the panel never holds the library taller than a small window
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.addLayout(track_form)
        layout.addWidget(jump)
        layout.addWidget(tempo_map)
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

        resolved = track_source(track, self._model)
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
        self._populate_jump_pairs()
        self._read_jump_cues()

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
        return model_holds(self._model, self._track)

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

        audio_path = self._source or resolve_track_path(self._track.path)
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


def resolve_track_path(raw: str | None) -> Optional[Path]:
    """A stored path (a file:// URL too) as a local path; None: none or not a path."""
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


def track_source(track: TrackRow, model: Optional[DataModel]) -> Optional[Path]:
    """The track's audio file: its stored path, else the deck's track's."""
    return resolve_track_path(track.path) or resolve_track_path(
        getattr(model, "properties", {}).get("path") if model else None)


def model_holds(model: Optional[DataModel], track: Optional[TrackRow]) -> bool:
    """The deck holds ``track`` (its marks as edited are the model's)."""
    if model is None or track is None:
        return False
    model_path = (getattr(model, "properties", {}) or {}).get("path")
    return bool(model_path and track.path and path_key(model_path) == path_key(track.path))
