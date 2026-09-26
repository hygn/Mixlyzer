from datetime import datetime
from pathlib import Path
import shutil

from PySide6 import QtCore
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.resource_paths import project_root
from core.workflows.optimizer import OptimizerWorkflow
from ui.config.dialogs import (
    BeatParameterOptimizeDialog,
    ParameterOptimizeProgressDialog,
)


def build_analysis_tab(dialog) -> None:
    dialog.tab_analysis = QWidget()
    root = QVBoxLayout(dialog.tab_analysis)
    analysis_tabs = QTabWidget(dialog.tab_analysis)

    # Global
    tab_global = QWidget(); f_global = QFormLayout(tab_global)
    dialog.sp_analysis_samp_rate = QSpinBox(); dialog.sp_analysis_samp_rate.setRange(8000, 384000); dialog.sp_analysis_samp_rate.setSingleStep(1000)
    dialog.cb_use_hpss = QCheckBox("Use HPSS")
    dialog.sp_analysis_samp_rate.setToolTip(
        "Audio is resampled to this rate before analysis.\n"
        "Higher is slower and rarely more accurate."
    )
    dialog.cb_use_hpss.setToolTip(
        "Split the audio into harmonic and percussive parts (HPSS) so beat analysis\n"
        "uses the percussive part and key analysis the harmonic part. Slower."
    )
    grp_global = QGroupBox("General")
    f_global_general = QFormLayout(grp_global)
    f_global_general.addRow("Sample rate (SPS)", dialog.sp_analysis_samp_rate)
    f_global_general.addRow(dialog.cb_use_hpss)
    f_global.addRow(grp_global)
    

    # Beat
    tab_beat = QWidget(); f_beat = QFormLayout(tab_beat)
    dialog.cb_bpm_dynamic = QCheckBox("Use Dynamic Analysis")
    dialog.cb_bpm_adaptive_win = QCheckBox("Use Adaptive Window for Dynamic Analysis")
    dialog.cmb_onset_source = QComboBox()
    dialog.cmb_onset_source.addItem("librosa", "librosa")
    dialog.cmb_onset_source.addItem("optimized", "optimized")
    dialog.cmb_onset_source.setToolTip(
        "Onset (ODF) used to detect BPM and the beat grid.\n"
        "librosa: librosa onset strength.\n"
        "optimized: learned linear model over librosa onset and the frame features\n"
        "shared with downbeat analysis (trained on the library's beat grids)."
    )
    dialog.ed_onset_parameter_path = QLineEdit()
    dialog.ed_onset_parameter_path.setPlaceholderText("relative or absolute .json path")
    dialog.ed_onset_feature_cache_path = QLineEdit()
    dialog.cb_beat_phase_correction = QCheckBox("Beat Phase Correction")
    dialog.cb_beat_phase_correction.setToolTip(
        "Shift the tracked beat grid by 0, 1/4, 1/2 or 3/4 beat to the phase a learned\n"
        "model scores highest over the track, from how the local pulse of each onset\n"
        "source (drums, harmony, melody) lines up with each candidate."
    )
    dialog.ed_beat_phase_parameter_path = QLineEdit()
    dialog.ed_beat_phase_parameter_path.setPlaceholderText("relative or absolute .json path")
    dialog.ed_beat_phase_feature_cache_path = QLineEdit()
    dialog.cb_dynamic_downbeat = QCheckBox("Dynamic Downbeat Detection")
    dialog.cb_dynamic_downbeat.setToolTip(
        "Checked: detect per-section downbeat changes (Dynamic).\n"
        "Unchecked: one global downbeat for the whole track (Global)."
    )
    dialog.ed_downbeat_parameter_path = QLineEdit()
    dialog.ed_downbeat_parameter_path.setPlaceholderText("relative or absolute .json path")
    dialog.ed_downbeat_feature_cache_path = QLineEdit()
    dialog.btn_reoptimize_beat_parameters = QPushButton("Reoptimize Beat Parameters")
    dialog.btn_reoptimize_beat_parameters.setToolTip(
        "Retrain the onset, beat phase and / or downbeat models on the library's beat grids."
    )
    dialog.btn_reoptimize_beat_parameters.clicked.connect(dialog._on_reoptimize_beat_parameters)
    dialog.sp_bpm_hop = QSpinBox(); dialog.sp_bpm_hop.setRange(16, 512); dialog.sp_bpm_hop.setSingleStep(32)
    dialog.sp_bpm_win = QSpinBox(); dialog.sp_bpm_win.setRange(1000, 60000); dialog.sp_bpm_win.setSingleStep(64)
    dialog.sp_bpm_min = QSpinBox(); dialog.sp_bpm_min.setRange(60,  400)
    dialog.sp_bpm_max = QSpinBox(); dialog.sp_bpm_max.setRange(100, 800)
    dialog.sp_beatgrid_offset = QDoubleSpinBox(); dialog.sp_beatgrid_offset.setRange(-10000.0, 10000.0); dialog.sp_beatgrid_offset.setDecimals(3); dialog.sp_beatgrid_offset.setSingleStep(1.0)
    dialog.cb_bpm_dynamic.setToolTip(
        "Detect tempo changes within a track instead of one BPM for the whole track."
    )
    dialog.cb_bpm_adaptive_win.setToolTip(
        "Also try 2x, 4x and whole-track analysis windows for dynamic analysis\n"
        "and keep the best fit."
    )
    onset_path_tip = "Learned onset model (.json) used when Onset source is 'optimized'."
    beat_phase_path_tip = "Learned beat phase model (.json) used by Beat Phase Correction."
    downbeat_path_tip = "Learned downbeat model (.json)."
    cache_tip = (
        "Folder for features cached while reoptimizing, so later runs are faster."
    )
    dialog.ed_onset_parameter_path.setToolTip(onset_path_tip)
    dialog.ed_onset_feature_cache_path.setToolTip(cache_tip)
    dialog.ed_beat_phase_parameter_path.setToolTip(beat_phase_path_tip)
    dialog.ed_beat_phase_feature_cache_path.setToolTip(cache_tip)
    dialog.ed_downbeat_parameter_path.setToolTip(downbeat_path_tip)
    dialog.ed_downbeat_feature_cache_path.setToolTip(cache_tip)
    dialog.sp_bpm_hop.setToolTip(
        "Step between onset frames in samples. Smaller is finer and slower.\n"
        "Also the frame grid of the learned onset, beat phase and downbeat features."
    )
    dialog.lbl_bpm_hop_warning = QLabel()
    dialog.lbl_bpm_hop_warning.setWordWrap(True)
    dialog.lbl_bpm_hop_warning.setStyleSheet("color: #d8a441;")
    dialog.lbl_bpm_hop_warning.setVisible(False)
    dialog.sp_bpm_win.setToolTip(
        "Length of the analysis window used to estimate tempo."
    )
    dialog.sp_bpm_min.setToolTip("Lowest BPM the tempo search considers.")
    dialog.sp_bpm_max.setToolTip("Highest BPM the tempo search considers.")
    dialog.sp_beatgrid_offset.setToolTip(
        "Constant shift applied to every detected beat (positive = later)."
    )
    grp_detection = QGroupBox("Detection")
    f_detection = QFormLayout(grp_detection)
    f_detection.addRow(dialog.cb_bpm_dynamic)
    f_detection.addRow(dialog.cb_bpm_adaptive_win)
    f_detection.addRow("Onset source", dialog.cmb_onset_source)
    f_detection.addRow(dialog.cb_beat_phase_correction)
    f_detection.addRow(dialog.cb_dynamic_downbeat)

    grp_bpm = QGroupBox("BPM & Beatgrid")
    f_bpm = QFormLayout(grp_bpm)
    f_bpm.addRow("Hop length (samples)", dialog.sp_bpm_hop)
    f_bpm.addRow("", dialog.lbl_bpm_hop_warning)
    f_bpm.addRow("Window (ms)", dialog.sp_bpm_win)
    f_bpm.addRow("BPM min", dialog.sp_bpm_min)
    f_bpm.addRow("BPM max", dialog.sp_bpm_max)
    f_bpm.addRow("Beatgrid offset (ms)", dialog.sp_beatgrid_offset)

    grp_beat_advanced = QGroupBox("Advanced")
    f_beat_advanced = QFormLayout(grp_beat_advanced)
    f_beat_advanced.addRow("Onset parameters (JSON)", dialog.ed_onset_parameter_path)
    f_beat_advanced.addRow("Onset feature cache", dialog.ed_onset_feature_cache_path)
    f_beat_advanced.addRow("Beat phase parameters (JSON)", dialog.ed_beat_phase_parameter_path)
    f_beat_advanced.addRow("Beat phase feature cache", dialog.ed_beat_phase_feature_cache_path)
    f_beat_advanced.addRow("Downbeat parameters (JSON)", dialog.ed_downbeat_parameter_path)
    f_beat_advanced.addRow("Downbeat feature cache", dialog.ed_downbeat_feature_cache_path)
    f_beat_advanced.addRow("", dialog.btn_reoptimize_beat_parameters)

    f_beat.addRow(grp_detection)
    f_beat.addRow(grp_bpm)
    f_beat.addRow(grp_beat_advanced)
    dialog.sp_bpm_hop.valueChanged.connect(dialog._update_bpm_hop_warning)
    dialog.cmb_onset_source.currentIndexChanged.connect(dialog._update_bpm_hop_warning)
    dialog.ed_onset_parameter_path.textChanged.connect(dialog._update_bpm_hop_warning)

    # Key (Chroma analysis)
    tab_key = QWidget(); f_key = QFormLayout(tab_key)
    dialog.cmb_chroma_method = QComboBox(); dialog.cmb_chroma_method.addItems(["cqt", "cens"])
    dialog.sp_chroma_hop_length = QSpinBox(); dialog.sp_chroma_hop_length.setRange(32, 8192); dialog.sp_chroma_hop_length.setSingleStep(32)
    dialog.sp_cqt_bins_per_oct = QSpinBox(); dialog.sp_cqt_bins_per_oct.setRange(1, 96)
    dialog.sp_cqt_octaves = QSpinBox(); dialog.sp_cqt_octaves.setRange(1, 10)
    dialog.cmb_chroma_method.setToolTip(
        "cqt: constant-Q chroma.\n"
        "cens: smoothed, energy-normalized chroma (more robust to dynamics and timbre)."
    )
    dialog.sp_chroma_hop_length.setToolTip("Step between chroma frames in samples.")
    dialog.sp_cqt_bins_per_oct.setToolTip(
        "Frequency bins per octave of the constant-Q transform (multiple of 12)."
    )
    dialog.sp_cqt_octaves.setToolTip("Number of octaves the constant-Q transform covers.")
    grp_chroma = QGroupBox("Chroma")
    f_chroma = QFormLayout(grp_chroma)
    f_chroma.addRow("Method", dialog.cmb_chroma_method)
    f_chroma.addRow("Hop length (samples)", dialog.sp_chroma_hop_length)
    f_chroma.addRow("CQT bins per octave", dialog.sp_cqt_bins_per_oct)
    f_chroma.addRow("CQT octaves", dialog.sp_cqt_octaves)
    f_key.addRow(grp_chroma)

    # Advanced (Viterbi transition probabilities)
    grp_adv = QGroupBox("Advanced")
    f_adv = QFormLayout(grp_adv)
    dialog.sp_min_offset = QDoubleSpinBox(); dialog.sp_min_offset.setDecimals(3); dialog.sp_min_offset.setRange(0.0, 1e6)
    dialog.sp_min_offset.setSingleStep(0.0001)
    dialog.sp_pitch_self = QDoubleSpinBox(); dialog.sp_pitch_self.setRange(-1e9, 1e9); dialog.sp_pitch_self.setDecimals(6); dialog.sp_pitch_self.setSingleStep(0.0001)
    dialog.sp_pitch_semitone = QDoubleSpinBox(); dialog.sp_pitch_semitone.setRange(-1e9, 1e9); dialog.sp_pitch_semitone.setDecimals(6); dialog.sp_pitch_semitone.setSingleStep(0.0001)
    dialog.sp_pitch_fifth = QDoubleSpinBox(); dialog.sp_pitch_fifth.setRange(-1e9, 1e9); dialog.sp_pitch_fifth.setDecimals(6); dialog.sp_pitch_fifth.setSingleStep(0.0001)
    dialog.sp_pitch_others = QDoubleSpinBox(); dialog.sp_pitch_others.setRange(-1e9, 1e9); dialog.sp_pitch_others.setDecimals(6); dialog.sp_pitch_others.setSingleStep(0.0001)
    dialog.sp_min_offset.setToolTip(
        "Weight added to out-of-scale notes in the key templates.\n"
        "Higher makes key matching less strict."
    )
    pitch_tip = (
        "Relative probability of the key moving by this interval between beats\n"
        "(Viterbi transition). The four values are normalized together."
    )
    dialog.sp_pitch_self.setToolTip(
        "Relative probability of the key staying the same between beats.\n"
        "Higher gives fewer key changes. The four values are normalized together."
    )
    dialog.sp_pitch_semitone.setToolTip(pitch_tip)
    dialog.sp_pitch_fifth.setToolTip(pitch_tip)
    dialog.sp_pitch_others.setToolTip(
        "Relative probability of the key moving to any other key between beats.\n"
        "The four values are normalized together."
    )
    f_adv.addRow("Min offset", dialog.sp_min_offset)
    f_adv.addRow("Pitch: self", dialog.sp_pitch_self)
    f_adv.addRow("Pitch: semitone", dialog.sp_pitch_semitone)
    f_adv.addRow("Pitch: fifth", dialog.sp_pitch_fifth)
    f_adv.addRow("Pitch: others", dialog.sp_pitch_others)
    f_key.addRow(grp_adv)

    # Waveform (Envelope bands)
    tab_wave = QWidget(); f_wave = QFormLayout(tab_wave)
    def make_band_row(parent_form, label: str):
        box = QWidget(); h = QHBoxLayout(box); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        lo = QDoubleSpinBox(); hi = QDoubleSpinBox()
        box.setToolTip(f"Frequency range (Hz) drawn as the {label.lower()} band of the waveform.")
        lo.setToolTip(f"Lower edge of the {label.lower()} band (Hz).")
        hi.setToolTip(f"Upper edge of the {label.lower()} band (Hz).")
        for sp in (lo, hi):
            sp.setDecimals(3); sp.setRange(0.0, 1e6); sp.setSingleStep(1.0); sp.setMinimumWidth(100)
        h.addWidget(QLabel("lo")); h.addWidget(lo)
        h.addWidget(QLabel("hi")); h.addWidget(hi)
        parent_form.addRow(label, box)
        return lo, hi
    dialog.sp_env_frame_ms = QSpinBox(); dialog.sp_env_frame_ms.setRange(1, 10)
    dialog.sp_env_order = QSpinBox(); dialog.sp_env_order.setRange(1, 12)
    dialog.sp_env_frame_ms.setToolTip(
        "Time resolution of the waveform bars. Smaller is more detailed."
    )
    dialog.sp_env_order.setToolTip(
        "Order of the band filters. Higher separates the bands more sharply."
    )
    grp_wave_general = QGroupBox("General")
    f_wave_general = QFormLayout(grp_wave_general)
    f_wave_general.addRow("Frame (ms)", dialog.sp_env_frame_ms)
    f_wave_general.addRow("Filter order", dialog.sp_env_order)
    grp_wave_bands = QGroupBox("Envelope Bands")
    f_wave_bands = QFormLayout(grp_wave_bands)
    dialog.sp_env_lo_lo,  dialog.sp_env_lo_hi  = make_band_row(f_wave_bands, "Low")
    dialog.sp_env_mid_lo, dialog.sp_env_mid_hi = make_band_row(f_wave_bands, "Mid")
    dialog.sp_env_hi_lo,  dialog.sp_env_hi_hi  = make_band_row(f_wave_bands, "High")
    f_wave.addRow(grp_wave_general)
    f_wave.addRow(grp_wave_bands)

    # Phrase
    tab_phrase = QWidget(); f_phrase = QFormLayout(tab_phrase)
    dialog.cb_phrase_analysis_enabled = QCheckBox("Use Phrase Analysis")
    dialog.ed_phrase_parameter_path = QLineEdit()
    dialog.ed_phrase_parameter_path.setPlaceholderText("relative or absolute .npz path")
    dialog.ed_phrase_feature_cache_path = QLineEdit()
    dialog.btn_reoptimize_phrase_parameter = QPushButton("Reoptimize Phrase Parameter")
    dialog.btn_reoptimize_phrase_parameter.clicked.connect(dialog._on_reoptimize_phrase_parameter)
    dialog.cb_phrase_analysis_enabled.setToolTip(
        "Detect phrase boundaries and labels during track analysis."
    )
    dialog.ed_phrase_parameter_path.setToolTip("Learned phrase model (.npz).")
    dialog.ed_phrase_feature_cache_path.setToolTip(cache_tip)
    dialog.btn_reoptimize_phrase_parameter.setToolTip(
        "Retrain the phrase model on the phrases stored in the library."
    )
    grp_phrase_general = QGroupBox("General")
    f_phrase_general = QFormLayout(grp_phrase_general)
    f_phrase_general.addRow(dialog.cb_phrase_analysis_enabled)
    grp_phrase_advanced = QGroupBox("Advanced")
    f_phrase_advanced = QFormLayout(grp_phrase_advanced)
    f_phrase_advanced.addRow("Parameters (NPZ)", dialog.ed_phrase_parameter_path)
    f_phrase_advanced.addRow("Feature cache", dialog.ed_phrase_feature_cache_path)
    f_phrase_advanced.addRow("", dialog.btn_reoptimize_phrase_parameter)
    f_phrase.addRow(grp_phrase_general)
    f_phrase.addRow(grp_phrase_advanced)

    # Assemble
    analysis_tabs.addTab(tab_global, "Global")
    analysis_tabs.addTab(tab_beat, "Beat")
    analysis_tabs.addTab(tab_key, "Key")
    analysis_tabs.addTab(tab_phrase, "Phrase")
    analysis_tabs.addTab(tab_wave, "Waveform")

    root.addWidget(analysis_tabs)
    dialog.tabs.addTab(dialog.tab_analysis, "Analysis")


class AnalysisSettingsMixin:
    def _project_path(self, text: str) -> Path:
        path = Path(str(text or "").strip())
        if not path.is_absolute():
            path = project_root() / path
        return path

    def _update_bpm_hop_warning(self, *_args) -> None:
        """Warn when the hop differs from the one the learned onset model was optimized at."""
        from analyzer_core.beat.learned_onset import onset_model_frame_hop_length

        text = ""
        if self.cmb_onset_source.currentData() == "optimized":
            hop = int(self.sp_bpm_hop.value())
            try:
                model_hop = onset_model_frame_hop_length(
                    self._project_path(self.ed_onset_parameter_path.text())
                )
            except Exception:
                model_hop = None
            if model_hop is not None and model_hop != hop:
                text = f"Does not match the learned onset model (optimized at hop {model_hop})."
        self.lbl_bpm_hop_warning.setText(text)
        self.lbl_bpm_hop_warning.setVisible(bool(text))

    def _backup_parameter(self, parameter_path: Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = project_root() / "backups"
        backup_path = backup_dir / (
            f"{parameter_path.stem}.bak_{stamp}{parameter_path.suffix}"
        )
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parameter_path, backup_path)
        return backup_path

    def _confirm_parameter_overwrite(
        self, parameter_path: Path, title: str, parameter_name: str
    ) -> bool:
        if not parameter_path.exists():
            return True
        answer = QMessageBox.question(
            self,
            title,
            (
                f"A new {parameter_name} parameter file will be created at the "
                "configured path.\nBack up the existing file before overwriting it?"
            ),
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Yes:
            try:
                self._backup_parameter(parameter_path)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    title,
                    f"Failed to back up {parameter_name} parameter:\n{exc}",
                )
                return False
        return True

    def _start_parameter_optimization(
        self,
        *,
        key: str,
        title: str,
        request_values: dict,
        finished_slot,
        failed_slot,
    ) -> None:
        dialog = ParameterOptimizeProgressDialog(title, self)
        worker = OptimizerWorkflow(
            key,
            [{"optimizer_name": key, "display_name": title, "request_values": request_values}],
            parent=self,
        )
        worker.featureProgress.connect(dialog.set_feature_progress)
        worker.optimizeProgress.connect(dialog.set_optimize_progress)
        worker.finished.connect(dialog.set_finished)
        worker.failed.connect(dialog.set_failed)
        worker.finished.connect(finished_slot)
        worker.failed.connect(failed_slot)
        setattr(self, f"_{key}_optimize_worker", worker)
        setattr(self, f"_{key}_optimize_dialog", dialog)
        worker.finished.connect(lambda _result, k=key, w=worker: self._clear_optimizer_worker(k, w))
        worker.failed.connect(lambda _message, k=key, w=worker: self._clear_optimizer_worker(k, w))
        self._set_parameter_optimization_enabled(False)
        dialog.show()
        worker.start()

    def _clear_optimizer_worker(self, key: str, worker: OptimizerWorkflow) -> None:
        if getattr(self, f"_{key}_optimize_worker", None) is worker:
            setattr(self, f"_{key}_optimize_worker", None)
        worker.deleteLater()

    def _parameter_optimization_is_running(self) -> bool:
        return any(
            getattr(self, f"_{key}_optimize_worker", None) is not None
            for key in ("beat", "phrase")
        )

    def _set_parameter_optimization_enabled(self, enabled: bool) -> None:
        self.btn_reoptimize_beat_parameters.setEnabled(enabled)
        self.btn_reoptimize_phrase_parameter.setEnabled(enabled)

    @staticmethod
    def _show_optimization_result(
        parent, title: str, summary: str, skipped_tracks
    ) -> None:
        skipped = tuple(skipped_tracks or ())
        message = QMessageBox(QMessageBox.Information, title, summary, parent=parent)
        if skipped:
            message.setInformativeText(
                f"Optimization completed after ignoring {len(skipped)} invalid track(s). "
                "Open Details to see the songs and reasons."
            )
            details: list[str] = []
            for index, track in enumerate(skipped, start=1):
                details.append(
                    f"{index}.\n{track.display_name}\n"
                    f"UUID: {track.uid}\n"
                    f"Reason: {track.reason}"
                )
            message.setDetailedText("\n\n".join(details))
        message.exec()

    def _beat_parameter_paths(self) -> dict[str, tuple[str, str]]:
        return {
            "onset": (self.ed_onset_parameter_path.text().strip(), self.ed_onset_feature_cache_path.text().strip()),
            "beat_phase": (
                self.ed_beat_phase_parameter_path.text().strip(),
                self.ed_beat_phase_feature_cache_path.text().strip(),
            ),
            "downbeat": (
                self.ed_downbeat_parameter_path.text().strip(),
                self.ed_downbeat_feature_cache_path.text().strip(),
            ),
        }

    def _on_reoptimize_beat_parameters(self) -> None:
        if self._parameter_optimization_is_running():
            return
        title = "Reoptimize Beat Parameters"
        from optimizer import beat_phase_parameter_optimizer, downbeat_parameter_optimizer, onset_parameter_optimizer

        dialog = BeatParameterOptimizeDialog(
            {
                "onset": onset_parameter_optimizer.DEFAULT_L2_STRENGTH,
                "beat_phase": beat_phase_parameter_optimizer.DEFAULT_L2_STRENGTH,
                "downbeat": downbeat_parameter_optimizer.DEFAULT_L2_STRENGTH,
            },
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        jobs = dialog.jobs()
        paths = self._beat_parameter_paths()
        queue = []
        for key, name, options in jobs:
            parameter_text, cache_text = paths[key]
            if not parameter_text or not cache_text:
                QMessageBox.warning(self, title, f"{name}: parameter path or feature cache path is empty.")
                return
            parameter_path = self._project_path(parameter_text)
            request_values = {
                "library_dir": self._project_path(self.ed_libpath.text().strip()),
                "cache_dir": self._project_path(cache_text),
                "output_path": parameter_path,
                "use_hpss": bool(self.cb_use_hpss.isChecked()),
                "rebuild_cache": False,
                **options,
            }
            if key == "onset":
                request_values["beatgrid_offset_msec"] = float(self.sp_beatgrid_offset.value())
                request_values["frame_hop_length"] = int(self.sp_bpm_hop.value())
            queue.append((key, name, options, request_values))
        if dialog.backup_existing():
            for _key, name, _options, request_values in queue:
                parameter_path = Path(request_values["output_path"])
                if not parameter_path.exists():
                    continue
                try:
                    self._backup_parameter(parameter_path)
                except Exception as exc:
                    QMessageBox.critical(self, title, f"Failed to back up the {name} parameter:\n{exc}")
                    return
        self._start_beat_optimization_queue(title, queue)

    def _start_beat_optimization_queue(self, title: str, queue: list) -> None:
        dialog = ParameterOptimizeProgressDialog(title, self)
        self._beat_optimize_dialog = dialog
        self._beat_queue_title = title
        self._beat_jobs = list(queue)
        self._beat_current_index = 0
        worker = OptimizerWorkflow(
            "beat",
            [
                {
                    "optimizer_name": key,
                    "display_name": name,
                    "request_values": request_values,
                }
                for key, name, _options, request_values in self._beat_jobs
            ],
            parent=self,
        )
        worker.featureProgress.connect(dialog.set_feature_progress)
        worker.optimizeProgress.connect(dialog.set_optimize_progress)
        worker.stageStarted.connect(self._on_beat_optimization_stage_started)
        worker.finished.connect(self._on_beat_optimization_finished)
        worker.failed.connect(self._on_beat_optimization_failed)
        worker.finished.connect(lambda _result, w=worker: self._clear_optimizer_worker("beat", w))
        worker.failed.connect(lambda _message, w=worker: self._clear_optimizer_worker("beat", w))
        self._beat_optimize_worker = worker
        self._set_parameter_optimization_enabled(False)
        dialog.show()
        worker.start()

    @QtCore.Slot(int, int, str, str)
    def _on_beat_optimization_stage_started(
        self, position: int, total: int, _key: str, name: str
    ) -> None:
        self._beat_current_index = position
        dialog = self._beat_optimize_dialog
        dialog.setWindowTitle(f"{self._beat_queue_title} ({position}/{total}: {name})")
        dialog.set_feature_progress(0, f"{name}: feature build pending")
        dialog.set_optimize_progress(0, f"{name}: optimize pending")
        dialog.btn_close.setEnabled(False)

    @QtCore.Slot(object)
    def _on_beat_optimization_finished(self, results) -> None:
        result_items = results if isinstance(results, tuple) else (results,)
        self._beat_queue_results = [
            (key, name, options, result)
            for (key, name, options, _request_values), result in zip(
                self._beat_jobs, result_items, strict=True
            )
        ]
        self._finish_beat_optimization_queue()

    @QtCore.Slot(str)
    def _on_beat_optimization_failed(self, message: str) -> None:
        current = max(1, self._beat_current_index)
        name = self._beat_jobs[current - 1][1]
        done = ", ".join(item[1] for item in self._beat_jobs[: current - 1]) or "none"
        remaining = [item[1] for item in self._beat_jobs[current:]]
        self._set_parameter_optimization_enabled(True)
        dialog = self._beat_optimize_dialog
        dialog.set_failed(message)
        QMessageBox.critical(
            dialog,
            self._beat_queue_title,
            f"{name} optimization failed:\n{message}\n\n"
            f"Finished before the failure: {done}\n"
            f"Not run: {', '.join(remaining) or 'none'}",
        )

    def _finish_beat_optimization_queue(self) -> None:
        self._set_parameter_optimization_enabled(True)
        self._update_bpm_hop_warning()
        dialog = self._beat_optimize_dialog
        dialog.bar_feature.setValue(100)
        dialog.bar_optimize.setValue(100)
        dialog.lbl_optimize.setText("Done")
        dialog.btn_close.setEnabled(True)
        dialog.setWindowTitle(self._beat_queue_title)
        blocks, skipped = [], []
        for key, name, options, result in self._beat_queue_results:
            blocks.append(self._beat_optimization_summary(key, name, options, result))
            skipped.extend(result.skipped_tracks)
        self._show_optimization_result(dialog, self._beat_queue_title, "\n\n".join(blocks), skipped)

    @staticmethod
    def _beat_optimization_summary(key: str, name: str, options: dict, result) -> str:
        kind = "CV" if result.cross_validated else "Training"
        if not result.cross_validated:
            validation = "no cross-validation"
        elif options.get("l2_sweep"):
            validation = f"{options['cv_folds']}-fold CV, L2 swept"
        else:
            validation = f"{options['cv_folds']}-fold CV, L2 fixed"
        head = f"{name} parameter updated:\n{result.output_path}\nL2 {result.selected_l2_strength:g} ({validation})\n"
        if key == "onset":
            return head + (
                f"Tracks: {result.track_count}, frames: {result.frame_count}\n"
                f"{kind} grid top-1: {result.grid_top1_accuracy:.1%}, "
                f"beat AP: {result.beat_average_precision:.1%}, loss: {result.cross_entropy:.4f}"
            )
        if key == "beat_phase":
            return head + (
                f"Tracks: {result.track_count}, beats: {result.beat_count}\n"
                f"{kind} track phase: {result.track_top1_accuracy:.1%}, "
                f"beat top-1: {result.beat_top1_accuracy:.1%}, cross-entropy: {result.cross_entropy:.4f}"
            )
        return head + (
            f"Tracks: {result.track_count}, bars: {result.bar_count}\n"
            f"{kind} top-1: {result.top1_accuracy:.1%}, cross-entropy: {result.cross_entropy:.4f}"
        )

    def _on_reoptimize_phrase_parameter(self) -> None:
        if self._parameter_optimization_is_running():
            return
        parameter_text = self.ed_phrase_parameter_path.text().strip()
        cache_text = self.ed_phrase_feature_cache_path.text().strip()
        if not parameter_text:
            QMessageBox.warning(
                self,
                "Reoptimize Phrase Parameter",
                "Phrase Parameter Path (npz) is empty.",
            )
            return
        if not cache_text:
            QMessageBox.warning(
                self,
                "Reoptimize Phrase Parameter",
                "Phrase Feature Cache Path (directory) is empty.",
            )
            return

        parameter_path = self._project_path(parameter_text)
        cache_dir = self._project_path(cache_text)
        library_dir = self._project_path(self.ed_libpath.text().strip())

        if not self._confirm_parameter_overwrite(
            parameter_path, "Reoptimize Phrase Parameter", "phrase"
        ):
            return

        self._start_parameter_optimization(
            key="phrase",
            title="Reoptimize Phrase Parameter",
            request_values={
                "library_dir": library_dir,
                "cache_dir": cache_dir,
                "output_path": parameter_path,
                "rebuild_cache": False,
                "seed": 0,
                "cv_folds": 0,
            },
            finished_slot=self._on_phrase_optimize_finished,
            failed_slot=self._on_phrase_optimize_failed,
        )

    @QtCore.Slot(object)
    def _on_phrase_optimize_finished(self, result) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_phrase_optimize_dialog", self)
        kind = "CV" if result.cross_validated else "Training"
        self._show_optimization_result(
            dialog,
            "Reoptimize Phrase Parameter",
            (
                f"Phrase parameter updated:\n{result.output_path}\n\n"
                f"Tracks: {result.track_count}\n"
                f"{kind} boundary AP: {result.boundary_average_precision:.1%}\n"
                f"{kind} boundary F1 @ 0.70: {result.boundary_f1:.1%}\n"
                f"{kind} label accuracy: {result.label_accuracy:.1%}\n"
                f"{kind} label macro-F1: {result.label_macro_f1:.1%}"
                + (
                    ""
                    if result.cross_validated
                    else "\n\nInterpretation: these in-sample scores only confirm how well "
                    "the model fits its training data. They do not estimate "
                    "performance on unseen tracks."
                )
            ),
            result.skipped_tracks,
        )

    @QtCore.Slot(str)
    def _on_phrase_optimize_failed(self, message: str) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_phrase_optimize_dialog", self)
        QMessageBox.critical(
            dialog,
            "Reoptimize Phrase Parameter",
            f"Phrase parameter optimization failed:\n{message}",
        )
