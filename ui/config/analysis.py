import os

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.settings.config import cuepointconfig, cuepointrule, default_cfg
from system.i18n import tr
from ui.config.pipeline_map import PipelineMap
from ui.config.scroll import scroll_page


def _path_hints() -> tuple[str, str]:
    """The path fields' hints: a .npz model, a .json model."""
    return tr("relative or absolute .npz path"), tr("relative or absolute .json path")


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
    """Beat: the BPM octave, dynamic analysis / downbeats; Advanced: the beat models."""
    npz_hint, json_hint = _path_hints()
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

    grp_bpm = QGroupBox("BPM")
    f_bpm = QFormLayout(grp_bpm)
    f_bpm.addRow(tr("BPM min"), dialog.sp_bpm_min)
    f_bpm.addRow("", dialog.lbl_bpm_range)
    f_bpm.addRow(dialog.cb_bpm_dynamic)
    f_bpm.addRow(dialog.cb_dynamic_downbeat)

    grp_beat_advanced = QGroupBox(tr("Advanced"))
    f_beat_advanced = QFormLayout(grp_beat_advanced)
    f_beat_advanced.addRow(tr("Tempo-change onset (NPZ)"), dialog.ed_refine_onset_parameter_path)
    f_beat_advanced.addRow(tr("Beat TCN (NPZ)"), dialog.ed_beat_tcn_parameter_path)
    f_beat_advanced.addRow(tr("Metronome critic (NPZ)"), dialog.ed_beat_critic_parameter_path)
    f_beat_advanced.addRow(tr("Downbeat model (NPZ)"), dialog.ed_downbeat_ctx_parameter_path)
    f_beat_advanced.addRow(tr("Decoder costs (JSON)"), dialog.ed_beat_decoder_parameter_path)

    f_beat.addRow(grp_bpm)
    f_beat.addRow(grp_beat_advanced)
    return tab_beat


def _key_page(dialog) -> QWidget:
    """Key: dynamic analysis; Advanced: the key model."""
    npz_hint, _json_hint = _path_hints()
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
    grp_key_advanced = QGroupBox(tr("Advanced"))
    f_key_advanced = QFormLayout(grp_key_advanced)
    f_key_advanced.addRow(tr("Key model (NPZ)"), dialog.ed_key_parameter_path)
    f_key.addRow(grp_key_advanced)
    return tab_key


def _phrase_page(dialog) -> QWidget:
    """Phrase: Advanced: the phrase model."""
    npz_hint, _json_hint = _path_hints()
    tab_phrase = QWidget(); f_phrase = QFormLayout(tab_phrase)
    dialog.ed_phrase_parameter_path = QLineEdit()
    dialog.ed_phrase_parameter_path.setPlaceholderText(npz_hint)
    dialog.ed_phrase_parameter_path.setToolTip(tr("Learned phrase model (.npz)."))
    grp_phrase_advanced = QGroupBox(tr("Advanced"))
    f_phrase_advanced = QFormLayout(grp_phrase_advanced)
    f_phrase_advanced.addRow(tr("Parameters (NPZ)"), dialog.ed_phrase_parameter_path)
    f_phrase.addRow(grp_phrase_advanced)
    return tab_phrase


def _cue_page(dialog) -> QWidget:
    """CUE: AutoCUE, the CUE point rules, the JumpCUE model."""
    _npz_hint, json_hint = _path_hints()
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
    dialog.ed_jumpcue_parameter_path.setToolTip(tr("Learned JumpCUE jump point model (.json)."))
    grp_jumpcue_advanced = QGroupBox(tr("JumpCUE Advanced"))
    f_jumpcue_advanced = QFormLayout(grp_jumpcue_advanced)
    f_jumpcue_advanced.addRow(tr("Parameters (JSON)"), dialog.ed_jumpcue_parameter_path)
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
