from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtWidgets
import soundfile as sf
from core.analysis_lib_handler import FeatureNPZStore
from core.library_handler import TrackRow
from core.audio.decoder import decode_to_memmap, get_samplerate
from core.model import DataModel
from third_party.rekordbox import build_rekordbox_xml, sanitize_filename
from utils.jump_cues import extract_jump_cue_pairs
from utils.jumprender import jump_renderer


class ExportTrackDialog(QtWidgets.QDialog):
    def __init__(self, lib_dir: str, parent: Optional[QtWidgets.QWidget] = None, *, model: Optional[DataModel] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Track")
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._lib_dir = lib_dir
        self._model: Optional[DataModel] = model
        self._track: Optional[TrackRow] = None
        self._source: Optional[Path] = None
        self._features: Optional[dict[str, np.ndarray]] = None
        self._jump_pairs: list[dict] = []

        self.lbl_title = QtWidgets.QLabel("-")
        self.lbl_artist = QtWidgets.QLabel("-")
        self.lbl_album = QtWidgets.QLabel("-")
        self.lbl_source = QtWidgets.QLabel("-")
        self.lbl_source.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.dest_edit = QtWidgets.QLineEdit()
        self.dest_browse = QtWidgets.QPushButton("Browse...")
        self.dest_browse.clicked.connect(self._on_browse)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #ff8080;")

        dest_layout = QtWidgets.QHBoxLayout()
        dest_layout.addWidget(self.dest_edit, 1)
        dest_layout.addWidget(self.dest_browse)

        form = QtWidgets.QFormLayout()
        form.addRow("Title", self.lbl_title)
        form.addRow("Artist", self.lbl_artist)
        form.addRow("Album", self.lbl_album)
        form.addRow("Source", self.lbl_source)
        form.addRow("Destination Folder", dest_layout)

        self.chk_rekordbox = QtWidgets.QCheckBox("Rekordbox XML")
        self.chk_waveform = QtWidgets.QCheckBox("Waveform Image")
        self.chk_rendered_jump = QtWidgets.QCheckBox("Prerender JumpCUE Audio")
        self.chk_rendered_jump.setToolTip("Write WAV files with JumpCUE skips baked in.")
        self.chk_rekordbox.setChecked(True)
        self.chk_waveform.setChecked(False)
        self.chk_rendered_jump.setChecked(False)
        self.cmb_jump_pair = QtWidgets.QComboBox()
        self.cmb_jump_pair.setEnabled(False)
        self.cmb_jump_pair.setPlaceholderText("No JumpCUE pairs")
        self.chk_jump_dir_bwd_fwd = QtWidgets.QCheckBox("B->A (forward)")
        self.chk_jump_dir_fwd_bwd = QtWidgets.QCheckBox("A->B (return)")
        self.chk_jump_dir_bwd_fwd.setChecked(True)
        self.chk_jump_dir_fwd_bwd.setChecked(False)
        jump_dir_layout = QtWidgets.QHBoxLayout()
        jump_dir_layout.addWidget(self.chk_jump_dir_bwd_fwd)
        jump_dir_layout.addWidget(self.chk_jump_dir_fwd_bwd)
        jump_dir_layout.addStretch(1)
        export_opts = QtWidgets.QHBoxLayout()
        export_opts.addWidget(self.chk_rekordbox)
        export_opts.addWidget(self.chk_waveform)
        export_opts.addWidget(self.chk_rendered_jump)
        export_opts.addStretch(1)
        form.addRow("Include", export_opts)
        form.addRow("JumpCUE Pair", self.cmb_jump_pair)
        form.addRow("JumpCUE Direction", jump_dir_layout)

        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_cancel.clicked.connect(self.reject)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_cancel)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        layout.addLayout(btns)

        self.resize(440, 260)

    def set_track(self, track: TrackRow) -> None:
        self._track = track
        self._features = None
        self._jump_pairs = []
        self.cmb_jump_pair.clear()
        self.cmb_jump_pair.setEnabled(False)
        self.lbl_title.setText(track.title or "-")
        self.lbl_artist.setText(track.artist or "-")
        self.lbl_album.setText(track.album or "-")

        resolved = self._resolve_path(track.path) or self._resolve_path(getattr(self._model, "properties", {}).get("path") if self._model else None)
        self._source = resolved if resolved else None
        self.lbl_source.setText(str(resolved) if resolved else (track.path or "-"))
        self.status_label.clear()

        if resolved and resolved.exists():
            self.dest_edit.setText(str(resolved.parent))
        elif resolved:
            self.dest_edit.setText(str(resolved.parent))
            self.status_label.setText("Source file not found at the specified path.")
        else:
            self.dest_edit.clear()
            self.status_label.setText("Source path is invalid.")
        self._populate_jump_pairs()

    def _resolve_path(self, raw: str | None) -> Optional[Path]:
        if not raw:
            return None
        candidate: Optional[str]
        if raw.startswith("file://"):
            parsed = urlparse(raw)
            path_part = unquote(parsed.path or "")
            if parsed.netloc and parsed.netloc not in ("", "localhost"):
                path_part = f"/{parsed.netloc}{path_part}"
            if os.name == "nt" and path_part.startswith("/"):
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
            "Select Destination Folder",
            current or (str(self._source.parent) if self._source else os.getcwd()),
        )
        if directory:
            self.dest_edit.setText(directory)

    def _on_export(self) -> None:
        if self._track is None:
            QtWidgets.QMessageBox.warning(self, "Export Track", "No track is selected.")
            return

        export_xml = self.chk_rekordbox.isChecked()
        export_wave = self.chk_waveform.isChecked()
        export_jump = self.chk_rendered_jump.isChecked()
        if not (export_xml or export_wave or export_jump):
            QtWidgets.QMessageBox.warning(
                self,
                "Export Track",
                "Select at least one export target (Rekordbox XML, Waveform Image, or JumpCUE audio).",
            )
            return

        dest_text = self.dest_edit.text().strip()
        if not dest_text:
            QtWidgets.QMessageBox.warning(self, "Export Track", "Please select a destination folder.")
            return
        dest_dir = Path(dest_text).expanduser()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export Track", f"Cannot create destination folder:\n{exc}")
            return

        features = self._load_features()
        if features is None:
            return

        exported: list[tuple[str, Path]] = []
        if export_xml:
            audio_for_xml = self._source or self._resolve_path(self._track.path)
            xml_path = self._write_xml(dest_dir, features, audio_for_xml)
            if xml_path is None:
                return
            exported.append(("Rekordbox XML", xml_path))

        if export_wave:
            image_path = self._write_waveform_image(dest_dir, features)
            if image_path is None:
                return
            exported.append(("Waveform Image", image_path))
        
        if export_jump:
            if not self._jump_pairs:
                self._populate_jump_pairs()
            pair = self._selected_jump_pair()
            if pair is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Export Track",
                    "Select a JumpCUE pair to render.",
                )
                return
            directions: list[tuple[str, str]] = []
            if self.chk_jump_dir_bwd_fwd.isChecked():
                directions.append(("backward", "forward"))
            if self.chk_jump_dir_fwd_bwd.isChecked():
                directions.append(("forward", "backward"))
            if not directions:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Export Track",
                    "Select at least one JumpCUE direction (B→A or A→B).",
                )
                return
            jump_exports = self._write_jumpcue_audio(dest_dir, features, pair, directions)
            if jump_exports is None:
                return
            exported.extend(jump_exports)

        lines = "\n".join(f"{label}: {path}" for label, path in exported)
        QtWidgets.QMessageBox.information(self, "Export Track", f"Export completed:\n{lines}")
        self.accept()

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
        self._features = store.load(self._track.uid)
        return self._features

    def _populate_jump_pairs(self) -> None:
        self.cmb_jump_pair.clear()
        self._jump_pairs = []
        features = self._load_features()
        if not features:
            self.cmb_jump_pair.setEnabled(False)
            return
        pairs = extract_jump_cue_pairs(features)
        self._jump_pairs = pairs
        for idx, pair in enumerate(pairs):
            fwd = str(pair.get("forward", {}).get("label", "")).strip()
            bwd = str(pair.get("backward", {}).get("label", "")).strip()
            self.cmb_jump_pair.addItem(f"{idx + 1}: {bwd}->{fwd}")
        self.cmb_jump_pair.setEnabled(bool(pairs))

    def _selected_jump_pair(self) -> Optional[dict]:
        if not self._jump_pairs:
            return None
        idx = int(self.cmb_jump_pair.currentIndex())
        if idx < 0 or idx >= len(self._jump_pairs):
            return None
        return self._jump_pairs[idx]

    def _write_xml(
        self,
        dest_dir: Path,
        features: dict[str, np.ndarray],
        audio_path: Optional[Path],
    ) -> Optional[Path]:
        assert self._track is not None
        xml_text, file_name = build_rekordbox_xml(
            self._track,
            features,
            audio_path=audio_path,
            resolve_path=self._resolve_path,
        )
        xml_path = dest_dir / file_name
        if xml_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Export Track",
                f"'{xml_path.name}' already exists.\nOverwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return None
        try:
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_text)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export Track", f"Failed to write XML file:\n{exc}")
            return None
        return xml_path

    def _write_waveform_image(
        self,
        dest_dir: Path,
        features: dict[str, np.ndarray],
    ) -> Optional[Path]:
        assert self._track is not None
        wave = features.get("wave_img_np")
        if wave is None:
            wave = features.get("wave_img_np_preview")
        if wave is None:
            QtWidgets.QMessageBox.warning(
                self, "Export Track", "Waveform image is not available for this track."
            )
            return None

        arr = np.asarray(wave)
        if arr.size == 0:
            QtWidgets.QMessageBox.warning(
                self, "Export Track", "Waveform data is empty and cannot be exported."
            )
            return None

        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            QtWidgets.QMessageBox.warning(
                self, "Export Track", "Waveform data has an unsupported shape."
            )
            return None

        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        # Stored arrays are (time, height, channels); transpose for image saving.
        arr = np.transpose(arr, (1, 0, 2))
        arr = np.ascontiguousarray(arr)

        mode = "RGB" if arr.shape[2] == 3 else "RGBA"
        image = Image.fromarray(arr, mode=mode)

        title = self._track.title or (self._source.stem if self._source else "track")
        filename = f"{sanitize_filename(title)}_waveform.png"
        image_path = dest_dir / filename
        if image_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Export Track",
                f"'{image_path.name}' already exists.\nOverwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return None
        try:
            image.save(image_path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export Track", f"Failed to save waveform image:\n{exc}")
            return None
        return image_path

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
                    "Export Track",
                    f"'{out_path.name}' already exists.\nOverwrite?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    continue

            try:
                sf.write(out_path, rendered_pcm, sr)
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self, "Export Track", f"Failed to write JumpCUE audio:\n{exc}"
                )
                return None

            exports.append((f"JumpCUE {src_label}->{dst_label}", out_path))

        return exports if exports else None
