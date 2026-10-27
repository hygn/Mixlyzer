from datetime import datetime
import os
from pathlib import Path
import shutil

from PySide6 import QtCore
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.settings.config import cuepointconfig, cuepointrule, default_cfg
from system.i18n import tr, tr_n
from system.resource_paths import project_root
from core.workflows.optimizer import OptimizerWorkflow
from core.workers.training_models import words
from ui.config.dialogs import (
    BeatParameterOptimizeDialog,
    KeyModelOptimizeDialog,
    ParameterOptimizeProgressDialog,
)
from ui.config.pipeline_map import PipelineMap
from ui.config.scroll import scroll_page


def _path_hints() -> tuple[str, str, str]:
    """The path fields' hints: a .npz model, a .json model, a feature cache folder."""
    return (tr("relative or absolute .npz path"), tr("relative or absolute .json path"),
            tr("Folder for features cached while reoptimizing, so later runs are faster."))


def _global_page(dialog) -> QWidget:
    """Global: the analyses' order and dependencies (a click opens one's tab)."""
    tab_global = QWidget(); f_global = QFormLayout(tab_global)
    grp_global = QGroupBox(tr("General"))
    f_global_general = QFormLayout(grp_global)
    dialog.sp_max_parallel_analyses = QSpinBox()
    # up to the logical cores, never below the default (a smaller computer keeps it unless the user lowers it)
    dialog.sp_max_parallel_analyses.setRange(
        1, max(os.cpu_count() or 1, default_cfg().analysisconfig.max_parallel_analyses))
    dialog.sp_max_parallel_analyses.setToolTip(tr("How many tracks are analyzed at once."))
    f_global_general.addRow(tr("Full analyses at once"), dialog.sp_max_parallel_analyses)
    # the analyses, their order and what each needs; a click opens the analysis's settings tab
    dialog.lbl_pipeline_title = QLabel(tr("Analysis order and dependencies"))
    dialog.pipeline_map = PipelineMap()
    f_global_general.addRow(dialog.lbl_pipeline_title)
    f_global_general.addRow(dialog.pipeline_map)
    f_global.addRow(grp_global)
    return tab_global


def _beat_page(dialog) -> QWidget:
    """Beat: the BPM octave, dynamic analysis / downbeats; Advanced: the beat models and their caches, Reoptimize Beat Models."""
    npz_hint, json_hint, cache_tip = _path_hints()
    tab_beat = QWidget(); f_beat = QFormLayout(tab_beat)
    dialog.sp_bpm_min = QSpinBox(); dialog.sp_bpm_min.setRange(60,  400)
    dialog.sp_bpm_min.setToolTip(tr(
        "The beat tracker reads tempos from here to twice this;\n"
        "a song outside it is read at half or double tempo."
    ))
    dialog.lbl_bpm_range = QLabel()
    dialog.sp_bpm_min.valueChanged.connect(
        lambda value: dialog.lbl_bpm_range.setText(tr("Reads {low} - {high} BPM").format(low=int(value), high=2 * int(value)))
    )
    dialog.cb_bpm_dynamic = QCheckBox(tr("Dynamic Analysis"))
    dialog.cb_bpm_dynamic.setToolTip(tr(
        "Checked: tempo changes within a track (sections, ramps).\n"
        "Unchecked: one steady BPM for the whole track."
    ))
    dialog.cb_dynamic_downbeat = QCheckBox(tr("Dynamic Downbeat"))
    dialog.cb_dynamic_downbeat.setToolTip(tr(
        "Checked: odd bars and meter changes within a track.\n"
        "Unchecked: one bar length and downbeat phase for the whole track."
    ))

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
    dialog.ed_refine_onset_parameter_path.setPlaceholderText(npz_hint)
    dialog.ed_refine_onset_parameter_path.setToolTip(tr(
        "The model (.npz) that places the tracker's tempo changes."
    ))
    dialog.ed_onset_feature_cache_path = QLineEdit()
    dialog.ed_onset_feature_cache_path.setToolTip(cache_tip)
    dialog.ed_beat_tcn_parameter_path = QLineEdit()
    dialog.ed_beat_tcn_parameter_path.setPlaceholderText(npz_hint)
    dialog.ed_beat_tcn_parameter_path.setToolTip(tr(
        "The model (.npz) that hears the beats and downbeats."
    ))
    dialog.ed_beat_critic_parameter_path = QLineEdit()
    dialog.ed_beat_critic_parameter_path.setPlaceholderText(npz_hint)
    dialog.ed_beat_critic_parameter_path.setToolTip(tr(
        "The model (.npz) that judges 3:2 / 4:3 readings."
    ))
    dialog.ed_beat_decoder_parameter_path = QLineEdit()
    dialog.ed_beat_decoder_parameter_path.setPlaceholderText(json_hint)
    dialog.ed_beat_decoder_parameter_path.setToolTip(tr(
        "The decoder's costs (.json)."
    ))
    dialog.ed_downbeat_ctx_parameter_path = QLineEdit()
    dialog.ed_downbeat_ctx_parameter_path.setPlaceholderText(npz_hint)
    dialog.ed_downbeat_ctx_parameter_path.setToolTip(tr(
        "The downbeat model (.npz) the tracker's bars read."
    ))
    dialog.ed_beat_feature_cache_path = QLineEdit()
    dialog.ed_beat_feature_cache_path.setToolTip(tr(
        "Folder for the frame features the beat models are retrained on."
    ))
    dialog.btn_reoptimize_beat_parameters = QPushButton(tr("Reoptimize Beat Models..."))
    dialog.btn_reoptimize_beat_parameters.setToolTip(tr("Retrain the beat analysis's models on your library's beat grids."))
    dialog.btn_reoptimize_beat_parameters.clicked.connect(dialog._on_reoptimize_beat_parameters)

    grp_bpm = QGroupBox("BPM")
    f_bpm = QFormLayout(grp_bpm)
    f_bpm.addRow(tr("BPM min"), dialog.sp_bpm_min)
    f_bpm.addRow("", dialog.lbl_bpm_range)
    f_bpm.addRow(dialog.cb_bpm_dynamic)
    f_bpm.addRow(dialog.cb_dynamic_downbeat)

    grp_beat_advanced = QGroupBox(tr("Advanced"))
    f_beat_advanced = QFormLayout(grp_beat_advanced)
    f_beat_advanced.addRow(tr("Tempo-change onset (NPZ)"), dialog.ed_refine_onset_parameter_path)
    f_beat_advanced.addRow(tr("Onset feature cache"), dialog.ed_onset_feature_cache_path)
    f_beat_advanced.addRow(tr("Beat TCN (NPZ)"), dialog.ed_beat_tcn_parameter_path)
    f_beat_advanced.addRow(tr("Metronome critic (NPZ)"), dialog.ed_beat_critic_parameter_path)
    f_beat_advanced.addRow(tr("Downbeat model (NPZ)"), dialog.ed_downbeat_ctx_parameter_path)
    f_beat_advanced.addRow(tr("Decoder costs (JSON)"), dialog.ed_beat_decoder_parameter_path)
    f_beat_advanced.addRow(tr("Beat feature cache"), dialog.ed_beat_feature_cache_path)
    f_beat_advanced.addRow("", dialog.btn_reoptimize_beat_parameters)

    f_beat.addRow(grp_bpm)
    f_beat.addRow(grp_beat_advanced)
    return tab_beat


def _key_page(dialog) -> QWidget:
    """Key: dynamic analysis; Advanced: the key model, its cache, Reoptimize Key Model."""
    npz_hint, _json_hint, cache_tip = _path_hints()
    tab_key = QWidget(); f_key = QFormLayout(tab_key)
    dialog.cb_key_dynamic = QCheckBox(tr("Dynamic Analysis"))
    dialog.cb_key_dynamic.setToolTip(tr(
        "Key changes within a track. Off: one key for the whole track, the key of its intro, outro and interludes."
    ))
    grp_key_general = QGroupBox(tr("General"))
    f_key_general = QFormLayout(grp_key_general)
    f_key_general.addRow(dialog.cb_key_dynamic)
    f_key.addRow(grp_key_general)
    dialog.ed_key_parameter_path = QLineEdit()
    dialog.ed_key_parameter_path.setPlaceholderText(npz_hint)
    dialog.ed_key_parameter_path.setToolTip(tr(
        "The key model (.npz). A missing file means the shipped model."
    ))
    dialog.ed_key_feature_cache_path = QLineEdit()
    dialog.ed_key_feature_cache_path.setToolTip(cache_tip)
    dialog.btn_reoptimize_key_model = QPushButton(tr("Reoptimize Key Model..."))
    dialog.btn_reoptimize_key_model.setToolTip(tr("Retrain the key model on the keys you checked in your library."))
    dialog.btn_reoptimize_key_model.clicked.connect(dialog._on_reoptimize_key_model)
    grp_key_advanced = QGroupBox(tr("Advanced"))
    f_key_advanced = QFormLayout(grp_key_advanced)
    f_key_advanced.addRow(tr("Key model (NPZ)"), dialog.ed_key_parameter_path)
    f_key_advanced.addRow(tr("Key feature cache"), dialog.ed_key_feature_cache_path)
    f_key_advanced.addRow("", dialog.btn_reoptimize_key_model)
    f_key.addRow(grp_key_advanced)
    return tab_key


def _phrase_page(dialog) -> QWidget:
    """Phrase: Advanced: the phrase model, its cache, Reoptimize Phrase Parameter."""
    npz_hint, _json_hint, cache_tip = _path_hints()
    tab_phrase = QWidget(); f_phrase = QFormLayout(tab_phrase)
    dialog.ed_phrase_parameter_path = QLineEdit()
    dialog.ed_phrase_parameter_path.setPlaceholderText(npz_hint)
    dialog.ed_phrase_feature_cache_path = QLineEdit()
    dialog.btn_reoptimize_phrase_parameter = QPushButton(tr("Reoptimize Phrase Parameter"))
    dialog.btn_reoptimize_phrase_parameter.clicked.connect(dialog._on_reoptimize_phrase_parameter)
    dialog.ed_phrase_parameter_path.setToolTip(tr("Learned phrase model (.npz)."))
    dialog.ed_phrase_feature_cache_path.setToolTip(cache_tip)
    dialog.btn_reoptimize_phrase_parameter.setToolTip(tr(
        "Retrain the phrase model on the phrases stored in the library."
    ))
    grp_phrase_advanced = QGroupBox(tr("Advanced"))
    f_phrase_advanced = QFormLayout(grp_phrase_advanced)
    f_phrase_advanced.addRow(tr("Parameters (NPZ)"), dialog.ed_phrase_parameter_path)
    f_phrase_advanced.addRow(tr("Feature cache"), dialog.ed_phrase_feature_cache_path)
    f_phrase_advanced.addRow("", dialog.btn_reoptimize_phrase_parameter)
    f_phrase.addRow(grp_phrase_advanced)
    return tab_phrase


def _cue_page(dialog) -> QWidget:
    """CUE: AutoCUE, the CUE point rules, the JumpCUE model, its cache, Reoptimize JumpCUE Parameter."""
    _npz_hint, json_hint, cache_tip = _path_hints()
    tab_cue = QWidget(); f_cue = QFormLayout(tab_cue)
    dialog.cmb_autocue = QComboBox()
    dialog.cmb_autocue.addItem(tr("CUE point"), "cue_point")
    dialog.cmb_autocue.addItem("JumpCUE", "jumpcue")
    dialog.cmb_autocue.addItem(tr("Off"), "off")
    dialog.cmb_autocue.setToolTip(tr(
        "The mark the analysis puts at the automatic start position (the lead-in end, the first downbeat or 0:00)."
    ))
    grp_autocue = QGroupBox("AutoCUE")
    f_autocue = QFormLayout(grp_autocue)
    f_autocue.addRow(tr("Automatic start"), dialog.cmb_autocue)
    dialog.ed_jumpcue_parameter_path = QLineEdit()
    dialog.ed_jumpcue_parameter_path.setPlaceholderText(json_hint)
    dialog.ed_jumpcue_feature_cache_path = QLineEdit()
    dialog.btn_reoptimize_jumpcue_parameter = QPushButton(tr("Reoptimize JumpCUE Parameter"))
    dialog.btn_reoptimize_jumpcue_parameter.clicked.connect(dialog._on_reoptimize_jumpcue_parameter)
    dialog.ed_jumpcue_parameter_path.setToolTip(tr("Learned JumpCUE jump point model (.json)."))
    dialog.ed_jumpcue_feature_cache_path.setToolTip(cache_tip)
    dialog.btn_reoptimize_jumpcue_parameter.setToolTip(tr(
        "Retrain the jump point model on the JumpCUEs stored in the library."
    ))
    grp_jumpcue_advanced = QGroupBox(tr("JumpCUE Advanced"))
    f_jumpcue_advanced = QFormLayout(grp_jumpcue_advanced)
    f_jumpcue_advanced.addRow(tr("Parameters (JSON)"), dialog.ed_jumpcue_parameter_path)
    f_jumpcue_advanced.addRow(tr("Feature cache"), dialog.ed_jumpcue_feature_cache_path)
    f_jumpcue_advanced.addRow("", dialog.btn_reoptimize_jumpcue_parameter)
    f_cue.addRow(grp_autocue)
    f_cue.addRow(_build_cue_point_rules(dialog))
    f_cue.addRow(grp_jumpcue_advanced)
    return tab_cue


def build_analysis_tab(dialog) -> None:
    """Settings > Analysis: its pages (Global, Beat, Key, Phrase, CUE)."""
    dialog.tab_analysis = QWidget()
    root = QVBoxLayout(dialog.tab_analysis)
    analysis_tabs = QTabWidget(dialog.tab_analysis)
    tab_global, tab_beat, tab_key = _global_page(dialog), _beat_page(dialog), _key_page(dialog)
    tab_phrase, tab_cue = _phrase_page(dialog), _cue_page(dialog)

    # Assemble
    analysis_tabs.addTab(scroll_page(tab_global), tr("Global"))
    analysis_tabs.addTab(scroll_page(tab_beat), tr("Beat"))
    analysis_tabs.addTab(scroll_page(tab_key), tr("Key"))
    analysis_tabs.addTab(scroll_page(tab_phrase), tr("Phrase"))
    analysis_tabs.addTab(scroll_page(tab_cue), tr("CUE"))

    def open_tab(name: str) -> None:
        """An Analysis tab, or a top-level settings tab (the waveform's bands are in View), by its title
        (translated as the tab's)."""
        for tabs in (analysis_tabs, dialog.tabs):
            for index in range(tabs.count()):
                if tabs.tabText(index) == name:
                    tabs.setCurrentIndex(index)
                    return

    dialog.pipeline_map.tabRequested.connect(open_tab)
    root.addWidget(analysis_tabs)
    dialog.tabs.addTab(dialog.tab_analysis, tr("Analysis"))



def _build_cue_point_rules(dialog) -> QGroupBox:
    """Per phrase label: a CUE point at its start / end, and whether also between
    two phrases of that label (else only next to another label)."""
    grp = QGroupBox(tr("CUE points (set by the analysis from the phrases)"))
    grid = QGridLayout(grp)
    grid.setHorizontalSpacing(16)
    dialog.cue_rule_boxes = {}
    for row, name in enumerate(f.name for f in cuepointconfig.__dataclass_fields__.values()):
        start = QCheckBox(tr("At start"))
        end = QCheckBox(tr("At end"))
        same = QCheckBox(tr("Also between two of them"))
        label = name.upper()
        start.setToolTip(tr("A CUE point where a {label} phrase starts").format(label=label))
        end.setToolTip(tr("A CUE point where a {label} phrase ends").format(label=label))
        same.setToolTip(tr(
            "Also where two {label} phrases meet; otherwise only where a {label} phrase "
            "meets a phrase of another label"
        ).format(label=label))

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
    note = QLabel(tr("Applies to tracks analyzed (or reanalyzed) from now on."))
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
            tr(
                "A new {name} parameter file will be created at the "
                "configured path.\nBack up the existing file before overwriting it?"
            ).format(name=parameter_name),
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
                    tr("Failed to back up {name} parameter:\n{error}").format(name=parameter_name, error=exc),
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
            self._library,
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
            for key in ("beat", "phrase", "jumpcue", "key_model")
        )

    def _set_parameter_optimization_enabled(self, enabled: bool) -> None:
        self.btn_reoptimize_beat_parameters.setEnabled(enabled)
        self.btn_reoptimize_phrase_parameter.setEnabled(enabled)
        self.btn_reoptimize_jumpcue_parameter.setEnabled(enabled)
        self.btn_reoptimize_key_model.setEnabled(enabled)

    @staticmethod
    def _show_optimization_result(
        parent, title: str, summary: str, skipped_tracks
    ) -> None:
        skipped = tuple(skipped_tracks or ())
        message = QMessageBox(QMessageBox.Information, title, summary, parent=parent)
        if skipped:
            message.setInformativeText(tr_n(
                "Optimization completed after ignoring {count} invalid track. "
                "Open Details to see the songs and reasons.",
                "Optimization completed after ignoring {count} invalid tracks. "
                "Open Details to see the songs and reasons.", len(skipped)).format(count=len(skipped)))
            details: list[str] = []
            for index, track in enumerate(skipped, start=1):
                details.append(
                    f"{index}.\n{words(track.display_name)}\n"
                    f"UUID: {track.uid}\n"
                    + tr("Reason: {reason}").format(reason=words(track.reason))
                )
            message.setDetailedText("\n\n".join(details))
        message.exec()

    def _library_track_count(self) -> int | None:
        """The library's tracks, as its owner's snapshot holds them (the dialogs' warnings on too few songs)."""
        library = getattr(self, "_library", None)
        return len(library.snapshot.tracks) if library is not None else None

    def _on_reoptimize_beat_parameters(self) -> None:
        if self._parameter_optimization_is_running():
            return
        title = tr("Reoptimize Beat Models")
        from analyzer.beat.frame_features import FRAME_HOP_LENGTH

        paths = {
            "refine_onset": (self.ed_refine_onset_parameter_path, self.ed_onset_feature_cache_path),
            "downbeat_ctx": (self.ed_downbeat_ctx_parameter_path, self.ed_beat_feature_cache_path),
            "beat_tcn": (self.ed_beat_tcn_parameter_path, self.ed_beat_feature_cache_path),
            "beat_critic": (self.ed_beat_critic_parameter_path, self.ed_beat_feature_cache_path),
            "beat_decoder": (self.ed_beat_decoder_parameter_path, self.ed_beat_feature_cache_path),
        }
        track_count = self._library_track_count()
        dialog = BeatParameterOptimizeDialog(
            self, track_count=track_count,
            model_paths={key: self._project_path(field.text().strip()) for key, (field, _cache) in paths.items()},
        )
        if dialog.exec() != QDialog.Accepted:
            return
        queue = []
        for key, name, options in dialog.jobs():
            parameter_text = paths[key][0].text().strip()
            cache_text = paths[key][1].text().strip()
            if not parameter_text or not cache_text:
                QMessageBox.warning(self, title, tr("{name}: parameter path or feature cache path is empty.").format(name=name))
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
                request_values.update(frame_hop_length=int(FRAME_HOP_LENGTH),
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
                    QMessageBox.critical(self, title, tr("Failed to back up the {name} parameter:\n{error}").format(name=name, error=exc))
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
            self._library,
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
        dialog.set_feature_progress(0, tr("{name}: feature build pending").format(name=name))
        dialog.set_optimize_progress(0, tr("{name}: optimize pending").format(name=name))
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
        done = ", ".join(item[1] for item in self._beat_jobs[: current - 1]) or tr("none")
        remaining = [item[1] for item in self._beat_jobs[current:]]
        self._set_parameter_optimization_enabled(True)
        dialog = self._beat_optimize_dialog
        dialog.set_failed(message)
        QMessageBox.critical(
            dialog,
            self._beat_queue_title,
            tr("{name} optimization failed:\n{error}\n\n"
               "Finished before the failure: {done}\n"
               "Not run: {remaining}").format(
                name=name, error=message, done=done, remaining=", ".join(remaining) or tr("none")),
        )

    def _finish_beat_optimization_queue(self) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = self._beat_optimize_dialog
        dialog.bar_feature.setValue(100)
        dialog.bar_optimize.setValue(100)
        dialog.lbl_optimize.setText(tr("Done"))
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
            lines.append(tr("{name}: saved to {path}").format(name=name, path=result.output_path))
        else:
            lines.append(tr("{name}: NOT saved -- the end-to-end check found the grids worse; the model in use is "
                            "kept.").format(name=name))
        if key == "beat_decoder":
            a, b = result.knobs_before, result.knobs_after
            lines.append(tr("Costs: section {section_before} -> {section_after}, half-beat reset {reset_before} -> "
                            "{reset_after}, odd bar {odd_before} -> {odd_after}").format(
                section_before=f"{a['section']:g}", section_after=f"{b['section']:g}",
                reset_before=f"{a['half_reset']:g}", reset_after=f"{b['half_reset']:g}",
                odd_before=f"{a['odd_bar']:g}", odd_after=f"{b['odd_bar']:g}"))
        if grid is None:
            lines.append(tr("No end-to-end check (cross-validation is off): saved without a comparison."))
            return chr(10).join(lines)
        if result.cross_validated:
            lines.append(tr("End-to-end check on {count} held-out songs (the whole analysis, current vs new):")
                         .format(count=grid.songs))
        else:
            lines.append(tr("End-to-end check on {count} songs (the whole analysis, current vs new):")
                         .format(count=grid.songs))
        lines.append("  " + tr("beats off by more than 25 ms: {before} -> {after} min").format(
            before=f"{grid.beats_before_sec / 60:.2f}", after=f"{grid.beats_after_sec / 60:.2f}"))
        lines.append("  " + tr("bars whose downbeat is off by more than 70 ms: {before} -> {after} min").format(
            before=f"{grid.bars_before_sec / 60:.2f}", after=f"{grid.bars_after_sec / 60:.2f}"))

        def listed(label, items):
            if not items:
                return
            shown = ", ".join(f"{c.title} ({c.change:+.1f} s)" for c in items[:10])
            more = tr(" and {count} more").format(count=len(items) - 10) if len(items) > 10 else ""
            lines.append(f"  {label} ({len(items)}): {shown}{more}")
        listed(tr("better"), grid.better)
        listed(tr("worse"), grid.worse)
        if not grid.better and not grid.worse:
            lines.append("  " + tr("no song changed by more than a second"))
        return chr(10).join(lines)

    def _on_reoptimize_phrase_parameter(self) -> None:
        if self._parameter_optimization_is_running():
            return
        parameter_text = self.ed_phrase_parameter_path.text().strip()
        cache_text = self.ed_phrase_feature_cache_path.text().strip()
        title = tr("Reoptimize Phrase Parameter")
        if not parameter_text:
            QMessageBox.warning(
                self,
                title,
                tr("Phrase Parameter Path (npz) is empty."),
            )
            return
        if not cache_text:
            QMessageBox.warning(
                self,
                title,
                tr("Phrase Feature Cache Path (directory) is empty."),
            )
            return

        parameter_path = self._project_path(parameter_text)
        cache_dir = self._project_path(cache_text)
        library_dir = self._project_path(self.ed_libpath.text().strip())

        if not self._confirm_parameter_overwrite(
            parameter_path, title, tr("phrase")
        ):
            return

        self._start_parameter_optimization(
            key="phrase",
            title=title,
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
        kind = tr("CV") if result.cross_validated else tr("Training")
        lines = [
            tr("Phrase parameter updated:\n{path}").format(path=result.output_path),
            "",
            tr("Tracks: {count}").format(count=result.track_count),
            tr("{kind} boundary F1 (exact beat): {value}").format(kind=kind, value=f"{result.boundary_f1:.1%}"),
            tr("{kind} key-section change F1: {value}").format(kind=kind, value=f"{result.key_change_f1:.1%}"),
            tr("{kind} label-change boundary F1: {value}").format(kind=kind, value=f"{result.label_change_f1:.1%}"),
            tr("{kind} beat label accuracy: {value}").format(kind=kind, value=f"{result.label_accuracy:.1%}"),
            tr("{kind} lead-in accuracy: {value}").format(kind=kind, value=f"{result.lead_in_accuracy:.1%}"),
        ]
        if not result.cross_validated:
            lines += ["", tr("Interpretation: these in-sample scores only confirm how well "
                             "the model fits its training data. They do not estimate "
                             "performance on unseen tracks.")]
        self._show_optimization_result(
            dialog,
            tr("Reoptimize Phrase Parameter"),
            "\n".join(lines),
            result.skipped_tracks,
        )

    @QtCore.Slot(str)
    def _on_phrase_optimize_failed(self, message: str) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_phrase_optimize_dialog", self)
        QMessageBox.critical(
            dialog,
            tr("Reoptimize Phrase Parameter"),
            tr("Phrase parameter optimization failed:\n{error}").format(error=message),
        )

    def _on_reoptimize_jumpcue_parameter(self) -> None:
        if self._parameter_optimization_is_running():
            return
        title = tr("Reoptimize JumpCUE Parameter")
        parameter_text = self.ed_jumpcue_parameter_path.text().strip()
        cache_text = self.ed_jumpcue_feature_cache_path.text().strip()
        if not parameter_text or not cache_text:
            QMessageBox.warning(self, title, tr("JumpCUE parameter path and feature cache path are required."))
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
        kind = tr("CV") if result.cross_validated else tr("Training")
        corrections = (
            tr("{kind} saved choice above the automatic one: {value} ({count} corrected groups)").format(
                kind=kind, value=f"{result.user_above_automatic:.1%}", count=result.preference_pairs)
            if result.preference_pairs else
            tr("No corrected groups yet: the library's JumpCUEs all match their automatic "
               "analysis, or were analyzed before it was kept.")
        )
        lines = [
            tr("JumpCUE parameter updated:\n{path}").format(path=result.output_path),
            "",
            tr("Tracks: {tracks}, groups: {groups}").format(tracks=result.track_count, groups=result.group_count),
            tr("{kind} rank of the saved point among the candidates (AUC): {value}").format(
                kind=kind, value=f"{result.label_auc:.3f}"),
            tr("{kind} chosen points on a downbeat: {value}").format(kind=kind, value=f"{result.pick_downbeat:.1%}"),
            corrections,
        ]
        self._show_optimization_result(
            dialog,
            tr("Reoptimize JumpCUE Parameter"),
            "\n".join(lines),
            result.skipped_tracks,
        )

    @QtCore.Slot(str)
    def _on_jumpcue_optimize_failed(self, message: str) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_jumpcue_optimize_dialog", self)
        QMessageBox.critical(
            dialog,
            tr("Reoptimize JumpCUE Parameter"),
            tr("JumpCUE parameter optimization failed:\n{error}").format(error=message),
        )

    def _on_reoptimize_key_model(self) -> None:
        if self._parameter_optimization_is_running():
            return
        title = tr("Reoptimize Key Model")
        parameter_text = self.ed_key_parameter_path.text().strip()
        cache_text = self.ed_key_feature_cache_path.text().strip()
        if not parameter_text or not cache_text:
            QMessageBox.warning(self, title, tr("Key model path and feature cache path are required."))
            return
        parameter_path = self._project_path(parameter_text)
        dialog = KeyModelOptimizeDialog(self, track_count=self._library_track_count(), model_path=str(parameter_path))
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.backup_existing() and parameter_path.exists():
            try:
                self._backup_parameter(parameter_path)
            except Exception as exc:
                QMessageBox.critical(self, title, tr("Failed to back up the key model:\n{error}").format(error=exc))
                return
        self._start_parameter_optimization(
            key="key_model",
            title=title,
            request_values={
                "library_dir": str(self._project_path(self.ed_libpath.text().strip())),
                "cache_dir": str(self._project_path(cache_text)),
                "output_path": str(parameter_path),
                **dialog.options(),
            },
            finished_slot=self._on_key_model_optimize_finished,
            failed_slot=self._on_key_model_optimize_failed,
        )

    @QtCore.Slot(object)
    def _on_key_model_optimize_finished(self, result) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_key_model_optimize_dialog", self)
        lines = [
            tr("Key model saved to {path}").format(path=result.output_path) if result.saved else
            tr("Key model NOT saved: the held-out songs' keys got worse; the model in use is kept."),
            "",
            tr("Training songs: {count} ({edited} with keys changed from the automatic result, "
               "{legacy} analyzed before it was kept)").format(
                count=result.track_count, edited=result.edited_count, legacy=result.legacy_count),
            tr("Key analysis on {count} held-out songs, current -> new:").format(count=result.eval_songs)
            if result.cross_validated else
            tr("Key analysis on the {count} training songs, current -> new:").format(count=result.eval_songs),
            "  " + tr("beats in the right key (Alphanumeric number): {before} -> {after}").format(
                before=f"{result.accuracy_before:.1%}", after=f"{result.accuracy_after:.1%}"),
            "  " + tr("beats off by a fifth (Alphanumeric neighbor): {before} -> {after}").format(
                before=f"{result.fifth_before:.1%}", after=f"{result.fifth_after:.1%}"),
        ]
        if not result.cross_validated:
            lines.append(tr("Without cross-validation these are the training songs: they show the fit, not unseen "
                            "songs."))
        self._show_optimization_result(dialog, tr("Reoptimize Key Model"), "\n".join(lines), result.skipped_tracks)

    @QtCore.Slot(str)
    def _on_key_model_optimize_failed(self, message: str) -> None:
        self._set_parameter_optimization_enabled(True)
        dialog = getattr(self, "_key_model_optimize_dialog", self)
        QMessageBox.critical(dialog, tr("Reoptimize Key Model"),
                             tr("Key model optimization failed:\n{error}").format(error=message))
