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
    QGridLayout,
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

from core.config import cuepointconfig, cuepointrule
from core.resource_paths import project_root
from core.workflows.optimizer import OptimizerWorkflow
from ui.config.dialogs import (
    BeatParameterOptimizeDialog,
    ParameterOptimizeProgressDialog,
)
from ui.config.scroll import scroll_page


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
    dialog.sp_bpm_min = QSpinBox(); dialog.sp_bpm_min.setRange(60,  400)
    dialog.sp_bpm_min.setToolTip(
        "The beat tracker reads every tempo in the octave from here (BPM min .. 2 x BPM min);\n"
        "a song outside it is read at half or double tempo."
    )
    dialog.lbl_bpm_range = QLabel()
    dialog.sp_bpm_min.valueChanged.connect(
        lambda value: dialog.lbl_bpm_range.setText(f"Reads {int(value)} - {2 * int(value)} BPM")
    )
    dialog.cb_bpm_dynamic = QCheckBox("Dynamic Analysis")
    dialog.cb_bpm_dynamic.setToolTip(
        "Checked: tempo changes within a track (sections, ramps).\n"
        "Unchecked: one steady BPM for the whole track."
    )
    dialog.cb_dynamic_downbeat = QCheckBox("Dynamic Downbeat")
    dialog.cb_dynamic_downbeat.setToolTip(
        "Checked: odd bars and meter changes within a track.\n"
        "Unchecked: one bar length and downbeat phase for the whole track.\n"
        "Always on with Dynamic Analysis."
    )

    def _follow_dynamic(on):
        # a dynamic grid always has dynamic bars; static analysis may have either
        if on:
            dialog._static_dynamic_downbeat = dialog.cb_dynamic_downbeat.isChecked()
            dialog.cb_dynamic_downbeat.setChecked(True)
        else:
            dialog.cb_dynamic_downbeat.setChecked(getattr(dialog, "_static_dynamic_downbeat", True))
        dialog.cb_dynamic_downbeat.setEnabled(not on)
    dialog.cb_bpm_dynamic.toggled.connect(_follow_dynamic)
    dialog.ed_refine_onset_parameter_path = QLineEdit()
    dialog.ed_refine_onset_parameter_path.setPlaceholderText("relative or absolute .npz path")
    dialog.ed_refine_onset_parameter_path.setToolTip(
        "Gradient-boosted onset model (.npz) that places the tracker's tempo changes."
    )
    cache_tip = (
        "Folder for features cached while reoptimizing, so later runs are faster."
    )
    dialog.ed_onset_feature_cache_path = QLineEdit()
    dialog.ed_onset_feature_cache_path.setToolTip(cache_tip)
    dialog.ed_beat_tcn_parameter_path = QLineEdit()
    dialog.ed_beat_tcn_parameter_path.setPlaceholderText("relative or absolute .npz path")
    dialog.ed_beat_tcn_parameter_path.setToolTip(
        "The frame TCN (.npz, its trainings) that hears the beats and downbeats."
    )
    dialog.ed_beat_critic_parameter_path = QLineEdit()
    dialog.ed_beat_critic_parameter_path.setPlaceholderText("relative or absolute .npz path")
    dialog.ed_beat_critic_parameter_path.setToolTip(
        "The metronome critic (.npz, its fold models) that judges 3:2 / 4:3 readings."
    )
    dialog.ed_beat_decoder_parameter_path = QLineEdit()
    dialog.ed_beat_decoder_parameter_path.setPlaceholderText("relative or absolute .json path")
    dialog.ed_beat_decoder_parameter_path.setToolTip(
        "The decoder's costs (.json): a new section, a same-tempo half-beat reset, an odd bar."
    )
    dialog.ed_downbeat_ctx_parameter_path = QLineEdit()
    dialog.ed_downbeat_ctx_parameter_path.setPlaceholderText("relative or absolute .npz path")
    dialog.ed_downbeat_ctx_parameter_path.setToolTip(
        "Gradient-boosted downbeat model (.npz) the tracker's bars read."
    )
    dialog.ed_beat_feature_cache_path = QLineEdit()
    dialog.ed_beat_feature_cache_path.setToolTip(
        "Folder for the frame features the beat models are retrained on (the tracker's own frames)."
    )
    dialog.btn_reoptimize_beat_parameters = QPushButton("Reoptimize Beat Models...")
    dialog.btn_reoptimize_beat_parameters.setToolTip(
        "Retrain the beat analysis's models (activation network, metronome critic, decoder costs, downbeat and "
        "tempo-change onset models) on the beat grids in your library."
    )
    dialog.btn_reoptimize_beat_parameters.clicked.connect(dialog._on_reoptimize_beat_parameters)

    grp_bpm = QGroupBox("BPM")
    f_bpm = QFormLayout(grp_bpm)
    f_bpm.addRow("BPM min", dialog.sp_bpm_min)
    f_bpm.addRow("", dialog.lbl_bpm_range)
    f_bpm.addRow(dialog.cb_bpm_dynamic)
    f_bpm.addRow(dialog.cb_dynamic_downbeat)

    grp_beat_advanced = QGroupBox("Advanced")
    f_beat_advanced = QFormLayout(grp_beat_advanced)
    f_beat_advanced.addRow("Tempo-change onset (NPZ)", dialog.ed_refine_onset_parameter_path)
    f_beat_advanced.addRow("Onset feature cache", dialog.ed_onset_feature_cache_path)
    f_beat_advanced.addRow("Beat TCN (NPZ)", dialog.ed_beat_tcn_parameter_path)
    f_beat_advanced.addRow("Metronome critic (NPZ)", dialog.ed_beat_critic_parameter_path)
    f_beat_advanced.addRow("Downbeat model (NPZ)", dialog.ed_downbeat_ctx_parameter_path)
    f_beat_advanced.addRow("Decoder costs (JSON)", dialog.ed_beat_decoder_parameter_path)
    f_beat_advanced.addRow("Beat feature cache", dialog.ed_beat_feature_cache_path)
    f_beat_advanced.addRow("", dialog.btn_reoptimize_beat_parameters)

    f_beat.addRow(grp_bpm)
    f_beat.addRow(grp_beat_advanced)

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

    # Waveform bands (stored overview levels; the main view waveform uses them too)
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
    grp_wave_bands = QGroupBox("Bands")
    grp_wave_bands.setToolTip(
        "Low / mid / high bands of the waveform. The overview's band levels are\n"
        "stored by the analysis: a change applies to tracks analyzed after it."
    )
    f_wave_bands = QFormLayout(grp_wave_bands)
    dialog.sp_env_lo_lo,  dialog.sp_env_lo_hi  = make_band_row(f_wave_bands, "Low")
    dialog.sp_env_mid_lo, dialog.sp_env_mid_hi = make_band_row(f_wave_bands, "Mid")
    dialog.sp_env_hi_lo,  dialog.sp_env_hi_hi  = make_band_row(f_wave_bands, "High")
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

    # CUE: AutoCUE, the CUE point rules and the JumpCUE model
    tab_cue = QWidget(); f_cue = QFormLayout(tab_cue)
    dialog.cmb_autocue = QComboBox()
    dialog.cmb_autocue.addItem("CUE point", "cue_point")
    dialog.cmb_autocue.addItem("JumpCUE", "jumpcue")
    dialog.cmb_autocue.addItem("Off", "off")
    dialog.cmb_autocue.setToolTip(
        "What the analysis puts at the automatic start position:\n"
        "the lead-in end, an early first downbeat, or 0:00.\n"
        "CUE point: a CUE point (memory cue).\n"
        "JumpCUE: a JumpCUE (hot cue) of its own.\n"
        "Off: nothing."
    )
    grp_autocue = QGroupBox("AutoCUE")
    f_autocue = QFormLayout(grp_autocue)
    f_autocue.addRow("Automatic start", dialog.cmb_autocue)
    dialog.ed_jumpcue_parameter_path = QLineEdit()
    dialog.ed_jumpcue_parameter_path.setPlaceholderText("relative or absolute .json path")
    dialog.ed_jumpcue_feature_cache_path = QLineEdit()
    dialog.btn_reoptimize_jumpcue_parameter = QPushButton("Reoptimize JumpCUE Parameter")
    dialog.btn_reoptimize_jumpcue_parameter.clicked.connect(dialog._on_reoptimize_jumpcue_parameter)
    dialog.ed_jumpcue_parameter_path.setToolTip("Learned JumpCUE jump point model (.json).")
    dialog.ed_jumpcue_feature_cache_path.setToolTip(cache_tip)
    dialog.btn_reoptimize_jumpcue_parameter.setToolTip(
        "Retrain the jump point model on the JumpCUEs stored in the library. Where a "
        "track's JumpCUEs differ from its automatic analysis, the automatic choice is "
        "learned as the one to avoid."
    )
    grp_jumpcue_advanced = QGroupBox("JumpCUE Advanced")
    f_jumpcue_advanced = QFormLayout(grp_jumpcue_advanced)
    f_jumpcue_advanced.addRow("Parameters (JSON)", dialog.ed_jumpcue_parameter_path)
    f_jumpcue_advanced.addRow("Feature cache", dialog.ed_jumpcue_feature_cache_path)
    f_jumpcue_advanced.addRow("", dialog.btn_reoptimize_jumpcue_parameter)
    f_cue.addRow(grp_autocue)
    f_cue.addRow(_build_cue_point_rules(dialog))
    f_cue.addRow(grp_jumpcue_advanced)

    # Assemble
    analysis_tabs.addTab(scroll_page(tab_global), "Global")
    analysis_tabs.addTab(scroll_page(tab_beat), "Beat")
    analysis_tabs.addTab(scroll_page(tab_key), "Key")
    analysis_tabs.addTab(scroll_page(tab_phrase), "Phrase")
    analysis_tabs.addTab(scroll_page(tab_cue), "CUE")
    analysis_tabs.addTab(scroll_page(tab_wave), "Waveform")

    root.addWidget(analysis_tabs)
    dialog.tabs.addTab(dialog.tab_analysis, "Analysis")


def _build_cue_point_rules(dialog) -> QGroupBox:
    """Per phrase label: a CUE point at its start / end, and whether also between
    two phrases of that label (else only next to another label)."""
    grp = QGroupBox("CUE points (set by the analysis from the phrases)")
    grid = QGridLayout(grp)
    grid.setHorizontalSpacing(16)
    dialog.cue_rule_boxes = {}
    for row, name in enumerate(f.name for f in cuepointconfig.__dataclass_fields__.values()):
        start = QCheckBox("At start")
        end = QCheckBox("At end")
        same = QCheckBox("Also between two of them")
        label = name.upper()
        start.setToolTip(f"A CUE point where a {label} phrase starts")
        end.setToolTip(f"A CUE point where a {label} phrase ends")
        same.setToolTip(
            f"Also where two {label} phrases meet; otherwise only where a {label} phrase "
            "meets a phrase of another label"
        )

        def _sync(_=None, start=start, end=end, same=same):
            same.setEnabled(start.isChecked() or end.isChecked())

        start.toggled.connect(_sync)
        end.toggled.connect(_sync)
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(start, row, 1)
        grid.addWidget(end, row, 2)
        grid.addWidget(same, row, 3)
        dialog.cue_rule_boxes[name] = (start, end, same)
    grid.setColumnStretch(4, 1)
    note = QLabel("Applies to tracks analyzed (or reanalyzed) from now on.")
    note.setEnabled(False)
    grid.addWidget(note, len(dialog.cue_rule_boxes), 0, 1, 5)
    return grp


class AnalysisSettingsMixin:
    def _set_autocue(self, mode: str) -> None:
        self.cmb_autocue.setCurrentIndex(self.cmb_autocue.findData(mode))

    def _get_autocue(self) -> str:
        return str(self.cmb_autocue.currentData())

    def _set_cue_point_rules(self, rules: cuepointconfig) -> None:
        for name, (start, end, same) in self.cue_rule_boxes.items():
            rule = getattr(rules, name)
            start.setChecked(bool(rule.at_start))
            end.setChecked(bool(rule.at_end))
            same.setChecked(bool(rule.same_label_boundaries))
            same.setEnabled(start.isChecked() or end.isChecked())

    def _get_cue_point_rules(self) -> cuepointconfig:
        return cuepointconfig(**{
            name: cuepointrule(at_start=start.isChecked(), at_end=end.isChecked(),
                               same_label_boundaries=same.isChecked())
            for name, (start, end, same) in self.cue_rule_boxes.items()
        })

    def _project_path(self, text: str) -> Path:
        path = Path(str(text or "").strip())
        if not path.is_absolute():
            path = project_root() / path
        return path

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
            for key in ("beat", "phrase", "jumpcue")
        )

    def _set_parameter_optimization_enabled(self, enabled: bool) -> None:
        self.btn_reoptimize_beat_parameters.setEnabled(enabled)
        self.btn_reoptimize_phrase_parameter.setEnabled(enabled)
        self.btn_reoptimize_jumpcue_parameter.setEnabled(enabled)

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

    def _on_reoptimize_beat_parameters(self) -> None:
        if self._parameter_optimization_is_running():
            return
        title = "Reoptimize Beat Models"
        from analyzer_core.beat.frame_features import FRAME_HOP_LENGTH

        paths = {
            "refine_onset": (self.ed_refine_onset_parameter_path, self.ed_onset_feature_cache_path),
            "downbeat_ctx": (self.ed_downbeat_ctx_parameter_path, self.ed_beat_feature_cache_path),
            "beat_tcn": (self.ed_beat_tcn_parameter_path, self.ed_beat_feature_cache_path),
            "beat_critic": (self.ed_beat_critic_parameter_path, self.ed_beat_feature_cache_path),
            "beat_decoder": (self.ed_beat_decoder_parameter_path, self.ed_beat_feature_cache_path),
        }
        library = Path(self._project_path(self.ed_libpath.text().strip()))
        try:
            track_count = sum(1 for _ in library.glob("*.npz"))
        except OSError:
            track_count = None
        dialog = BeatParameterOptimizeDialog(
            {}, self, track_count=track_count,
            model_paths={key: self._project_path(field.text().strip()) for key, (field, _cache) in paths.items()},
        )
        if dialog.exec() != QDialog.Accepted:
            return
        queue = []
        for key, name, options in dialog.jobs():
            parameter_text = paths[key][0].text().strip()
            cache_text = paths[key][1].text().strip()
            if not parameter_text or not cache_text:
                QMessageBox.warning(self, title, f"{name}: parameter path or feature cache path is empty.")
                return
            request_values = {
                "library_dir": self._project_path(self.ed_libpath.text().strip()),
                "cache_dir": self._project_path(cache_text),
                "output_path": self._project_path(parameter_text),
                "rebuild_cache": False,
                **options,
            }
            # the analysis's model files in use: the end-to-end check runs the whole analysis with them, the
            # retrained model swapped in
            request_values["bpm_min"] = float(self.sp_bpm_min.value())
            request_values["current_models"] = {
                "downbeat_ctx": self._project_path(self.ed_downbeat_ctx_parameter_path.text().strip()),
                "tcn": self._project_path(self.ed_beat_tcn_parameter_path.text().strip()),
                "critic": self._project_path(self.ed_beat_critic_parameter_path.text().strip()),
                "decoder": self._project_path(self.ed_beat_decoder_parameter_path.text().strip()),
                "refine_onset": self._project_path(self.ed_refine_onset_parameter_path.text().strip()),
            }
            if key == "refine_onset":
                request_values.update(use_hpss=bool(self.cb_use_hpss.isChecked()), frame_hop_length=int(FRAME_HOP_LENGTH),
                                      frames_cache_dir=self._project_path(self.ed_beat_feature_cache_path.text().strip()))
            if key == "beat_decoder":
                # the costs are fitted for the models in use (after any of them is retrained)
                models = request_values.pop("current_models")
                request_values.update(tcn_path=models["tcn"], critic_path=models["critic"],
                                      downbeat_ctx_path=models["downbeat_ctx"], refine_onset_path=models["refine_onset"])
            if key == "beat_critic":
                # its hard examples are the decoder's readings with the beat model in use
                request_values["tcn_path"] = request_values["current_models"]["tcn"]
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
        """The end-to-end report: the whole analysis on songs the model was not trained on, against the grids."""
        lines = []
        saved = getattr(result, "saved", True)
        grid = getattr(result, "grid", None)
        if saved:
            lines.append(f"{name}: saved to {result.output_path}")
        else:
            lines.append(f"{name}: NOT saved -- the end-to-end check found the grids worse; the model in use is kept.")
        if key == "beat_decoder":
            a, b = result.knobs_before, result.knobs_after
            lines.append(f"Costs: section {a['section']:g} -> {b['section']:g}, half-beat reset {a['half_reset']:g} -> "
                         f"{b['half_reset']:g}, odd bar {a['odd_bar']:g} -> {b['odd_bar']:g}")
        if grid is None:
            lines.append("No end-to-end check (cross-validation is off): saved without a comparison.")
            return chr(10).join(lines)
        where = "held-out songs" if result.cross_validated else "songs"
        lines.append(f"End-to-end check on {grid.songs} {where} (the whole analysis, current vs new):")
        lines.append(f"  beats off by more than 25 ms: {grid.beats_before_sec / 60:.2f} -> {grid.beats_after_sec / 60:.2f} min")
        lines.append(f"  bars whose downbeat is off by more than 70 ms: {grid.bars_before_sec / 60:.2f} -> "
                     f"{grid.bars_after_sec / 60:.2f} min")

        def listed(label, items):
            if not items:
                return
            shown = ", ".join(f"{c.title} ({c.change:+.1f} s)" for c in items[:10])
            more = f" and {len(items) - 10} more" if len(items) > 10 else ""
            lines.append(f"  {label} ({len(items)}): {shown}{more}")
        listed("better", grid.better)
        listed("worse", grid.worse)
        if not grid.better and not grid.worse:
            lines.append("  no song changed by more than a second")
        return chr(10).join(lines)

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
                f"{kind} boundary F1 (exact beat): {result.boundary_f1:.1%}\n"
                f"{kind} key-section change F1: {result.key_change_f1:.1%}\n"
                f"{kind} label-change boundary F1: {result.label_change_f1:.1%}\n"
                f"{kind} beat label accuracy: {result.label_accuracy:.1%}\n"
                f"{kind} lead-in accuracy: {result.lead_in_accuracy:.1%}"
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

    def _on_reoptimize_jumpcue_parameter(self) -> None:
        if self._parameter_optimization_is_running():
            return
        title = "Reoptimize JumpCUE Parameter"
        parameter_text = self.ed_jumpcue_parameter_path.text().strip()
        cache_text = self.ed_jumpcue_feature_cache_path.text().strip()
        if not parameter_text or not cache_text:
            QMessageBox.warning(self, title, "JumpCUE parameter path and feature cache path are required.")
            return
        parameter_path = self._project_path(parameter_text)
        if not self._confirm_parameter_overwrite(parameter_path, title, "JumpCUE"):
            return
        self._start_parameter_optimization(
            key="jumpcue",
            title=title,
            request_values={
                "library_dir": self._project_path(self.ed_libpath.text().strip()),
                "cache_dir": self._project_path(cache_text),
                "output_path": parameter_path,
                "phrase_parameter_path": self._project_path(self.ed_phrase_parameter_path.text().strip()),
            },
            finished_slot=self._on_jumpcue_optimize_finished,
            failed_slot=self._on_jumpcue_optimize_failed,
        )

    @QtCore.Slot(object)
    def _on_jumpcue_optimize_finished(self, result) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_jumpcue_optimize_dialog", self)
        kind = "CV" if result.cross_validated else "Training"
        corrections = (
            f"{kind} saved choice above the automatic one: {result.user_above_automatic:.1%} "
            f"({result.preference_pairs} corrected groups)"
            if result.preference_pairs else
            "No corrected groups yet: the library's JumpCUEs all match their automatic "
            "analysis, or were analyzed before it was kept."
        )
        self._show_optimization_result(
            dialog,
            "Reoptimize JumpCUE Parameter",
            (
                f"JumpCUE parameter updated:\n{result.output_path}\n\n"
                f"Tracks: {result.track_count}, groups: {result.group_count}\n"
                f"{kind} rank of the saved point among the candidates (AUC): {result.label_auc:.3f}\n"
                f"{kind} chosen points on a downbeat: {result.pick_downbeat:.1%}\n"
                f"{corrections}"
            ),
            result.skipped_tracks,
        )

    @QtCore.Slot(str)
    def _on_jumpcue_optimize_failed(self, message: str) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_jumpcue_optimize_dialog", self)
        QMessageBox.critical(
            dialog,
            "Reoptimize JumpCUE Parameter",
            f"JumpCUE parameter optimization failed:\n{message}",
        )
