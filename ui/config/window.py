from dataclasses import asdict, is_dataclass
from typing import Tuple

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
)

from core.config import (
    analysisconfig,
    config,
    externalsyncconfig,
    libconfig,
    memorydeckconfig,
    memoryvalueconfig,
    playbackconfig,
    viewconfig,
)
from core.event_bus import EventBus
from ui.config.analysis import AnalysisSettingsMixin, build_analysis_tab
from ui.config.external_sync import (
    ExternalSyncSettingsMixin,
    build_external_sync_tab,
)
from ui.config.library import LibrarySettingsMixin, build_library_tab
from ui.config.playback import PlaybackSettingsMixin, build_playback_tab
from ui.config.search import SearchSettingsMixin, build_search_tab
from ui.config.view import ViewSettingsMixin, build_view_tab


class SettingsDialog(
    LibrarySettingsMixin,
    SearchSettingsMixin,
    ViewSettingsMixin,
    PlaybackSettingsMixin,
    AnalysisSettingsMixin,
    ExternalSyncSettingsMixin,
    QDialog,
):
    DIALOG_WIDTH = 780
    FORM_LABEL_WIDTH = 260
    COMPACT_FORM_LABEL_WIDTH = 220

    def __init__(self, bus: EventBus):
        super().__init__()
        self.saveJsonRequested = bus.sig_setting_saveJsonRequested
        self._bus = bus
        self.setWindowTitle("Settings")
        self.setModal(False)

        self.tabs = QTabWidget(self)
        build_library_tab(self)
        build_search_tab(self)
        build_view_tab(self)
        build_playback_tab(self)
        build_analysis_tab(self)
        build_external_sync_tab(self)
        self._stabilize_layout_widths()

        self.btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        self.btn_box.accepted.connect(self._on_ok)
        self.btn_box.rejected.connect(self._on_cancel)
        self.btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self.btn_box.button(QDialogButtonBox.Ok).setToolTip("Save the settings and close.")
        self.btn_box.button(QDialogButtonBox.Cancel).setToolTip(
            "Close without saving unapplied changes."
        )
        self.btn_box.button(QDialogButtonBox.Apply).setToolTip(
            "Save the settings and keep the window open."
        )

        root = QVBoxLayout(self)
        root.addWidget(self.tabs)
        root.addWidget(self.btn_box)

        # Keep page-specific size hints from changing the top-level window width.
        # The height remains resizable for the External Sync memory editor.
        self.setFixedWidth(self.DIALOG_WIDTH)
        self.resize(self.DIALOG_WIDTH, 640)

        self._current_cfg = None
        self._initialize_library_settings(bus)
        self._initialize_search_settings(bus)
        self._initialize_view_settings(bus)
        self._initialize_playback_settings(bus)

    def _stabilize_layout_widths(self) -> None:
        """Give every settings form the same label/field geometry."""
        for form in self.findChildren(QFormLayout):
            parent = form.parentWidget()
            compact = False
            while parent is not None and parent is not self:
                if isinstance(parent, QGroupBox):
                    compact = True
                    break
                parent = parent.parentWidget()
            label_width = (
                self.COMPACT_FORM_LABEL_WIDTH if compact else self.FORM_LABEL_WIDTH
            )
            form.setRowWrapPolicy(QFormLayout.DontWrapRows)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QFormLayout.LabelRole)
                if label_item is not None and label_item.widget() is not None:
                    label_item.widget().setMinimumWidth(label_width)

                field_item = form.itemAt(row, QFormLayout.FieldRole)
                if field_item is None or field_item.widget() is None:
                    continue
                field = field_item.widget()
                # Hovering the row label shows the field's tooltip too.
                if label_item is not None and label_item.widget() is not None:
                    label = label_item.widget()
                    if not label.toolTip() and field.toolTip():
                        label.setToolTip(field.toolTip())
                policy = field.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Expanding)
                field.setSizePolicy(policy)

        # Spin boxes and combo boxes default to a compact fixed width. Let them
        # occupy the same field column as line edits instead.
        for control_type in (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox):
            for control in self.findChildren(control_type):
                policy = control.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Expanding)
                control.setSizePolicy(policy)

    # Public API
    def set_config(self, cfg: config):
        assert is_dataclass(cfg), "cfg must be a dataclass 'config'"
        self._current_cfg = cfg

        # lib
        self.ed_libpath.setText(cfg.libconfig.libpath)
        self._set_local_music_directories(cfg.libconfig.local_music_directories)
        self.cb_write_log.setChecked(bool(cfg.libconfig.write_log))
        self.ed_logpath.setText(cfg.libconfig.logpath)
        self.cb_rekordbox_sync.setChecked(bool(cfg.libconfig.rekordbox_sync_enabled))
        self.ed_rekordbox_xml_path.setText(cfg.libconfig.rekordbox_xml_path)

        # search
        self._set_search_config(cfg.searchconfig)

        # view
        v = cfg.viewconfig
        self.cb_waveform.setChecked(bool(v.display_waveform))
        self.cb_beatgrid.setChecked(bool(v.display_beatgrid))
        self.cb_keystrip.setChecked(bool(v.display_keystrip))
        self.cb_JumpCUE.setChecked(bool(v.display_JumpCUE))
        self.cb_phrase.setChecked(bool(getattr(v, "display_phrase", True)))
        self.cb_use_output_volume_as_peak_meter_input.setChecked(
            bool(v.use_output_volume_as_peak_meter_input)
        )
        self.cb_reduce_fps_when_occluded.setChecked(bool(v.reduce_fps_when_occluded))
        self.sp_fps.setValue(int(v.fps))
        self.ed_record_img_path.setText(v.record_img_path)
        self._set_phrase_label_names(v.phrase_labels)
        self.cmb_waveform_style.setCurrentIndex(self.cmb_waveform_style.findData(v.waveform_style))
        self._set_band(self.sp_env_lo_lo,  self.sp_env_lo_hi,  v.env_lo)
        self._set_band(self.sp_env_mid_lo, self.sp_env_mid_hi, v.env_mid)
        self._set_band(self.sp_env_hi_lo,  self.sp_env_hi_hi,  v.env_hi)
        self.sp_mainview_wave_frame_ms_short.setValue(float(v.mainview_wave_frame_ms_short))
        self.sp_mainview_wave_frame_ms_long.setValue(float(v.mainview_wave_frame_ms_long))
        self.sp_mainview_wave_filter_order.setValue(int(v.mainview_wave_filter_order))

        p = cfg.playbackconfig
        self.cb_metronome.setChecked(bool(p.enable_metronome))
        self.ed_metronome_wav_path.setText(p.metronome_wav_path)
        self.sp_metronome_offset_msec.setValue(float(getattr(p, "metronome_offset_msec", 0.0)))
        self.sp_volume_trim_dbfs.setValue(float(p.volume_trim_dbfs))
        self.sp_default_volume_percent.setValue(int(p.default_volume_percent))
        self.cb_use_timestretch.setChecked(bool(getattr(p, "use_timestretch", False)))
        self.sp_click_down_volume.setValue(int(p.metronome_downbeat_volume_percent))
        self.sp_click_down_pitch.setValue(float(p.metronome_downbeat_pitch_semitones))
        self.sp_click_beat_volume.setValue(int(p.metronome_beat_volume_percent))
        self.sp_click_beat_pitch.setValue(float(p.metronome_beat_pitch_semitones))
        self.cb_metronome_ducking.setChecked(bool(p.metronome_ducking))
        self.cb_soft_clip.setChecked(bool(p.soft_clip))

        # analysis
        a = cfg.analysisconfig
        self.sp_bpm_min.setValue(int(a.bpm_min))
        self.lbl_bpm_range.setText(f"Reads {int(a.bpm_min)} - {2 * int(a.bpm_min)} BPM")
        dynamic = bool(getattr(a, "bpm_dynamic", True))
        self.cb_bpm_dynamic.setChecked(dynamic)
        # the static analysis's own choice, kept while the box shows the dynamic analysis's (always on)
        self._static_dynamic_downbeat = bool(getattr(a, "dynamic_downbeat", True))
        self.cb_dynamic_downbeat.setChecked(dynamic or self._static_dynamic_downbeat)
        self.cb_dynamic_downbeat.setEnabled(not dynamic)
        self.ed_refine_onset_parameter_path.setText(str(getattr(a, "refine_onset_parameter_path", "") or ""))
        self.ed_onset_feature_cache_path.setText(
            str(getattr(a, "onset_feature_cache_path", "") or "")
        )
        self.ed_beat_tcn_parameter_path.setText(str(getattr(a, "beat_tcn_parameter_path", "") or ""))
        self.ed_beat_critic_parameter_path.setText(str(getattr(a, "beat_critic_parameter_path", "") or ""))
        self.ed_beat_decoder_parameter_path.setText(str(getattr(a, "beat_decoder_parameter_path", "") or ""))
        self.ed_downbeat_ctx_parameter_path.setText(str(getattr(a, "downbeat_ctx_parameter_path", "") or ""))
        self.ed_beat_feature_cache_path.setText(str(getattr(a, "beat_feature_cache_path", "") or ""))
        self.cb_phrase_analysis_enabled.setChecked(
            bool(getattr(a, "phrase_analysis_enabled", True))
        )
        self.ed_phrase_parameter_path.setText(
            str(
                getattr(
                    a,
                    "phrase_parameter_path",
                    "",
                )
                or ""
            )
        )
        self.ed_phrase_feature_cache_path.setText(
            str(getattr(a, "phrase_feature_cache_path", "") or "")
        )
        self.ed_jumpcue_parameter_path.setText(str(getattr(a, "jumpcue_parameter_path", "") or ""))
        self.ed_jumpcue_feature_cache_path.setText(str(getattr(a, "jumpcue_feature_cache_path", "") or ""))
        self._set_cue_point_rules(cfg.cuepointconfig)
        self._set_autocue(a.autocue)

        # key
        self.cb_key_dynamic.setChecked(bool(getattr(a, "key_dynamic", True)))
        self.ed_key_parameter_path.setText(str(getattr(a, "key_parameter_path", "") or ""))
        self.ed_key_feature_cache_path.setText(str(getattr(a, "key_feature_cache_path", "") or ""))

        # external sync
        x = cfg.externalsyncconfig
        self.cb_external_sync_enabled.setChecked(bool(x.enabled))
        self.cmb_external_sync_mode.setCurrentIndex(0 if x.mode == "time" else 1)
        self.cmb_total_sample_count_source.setCurrentIndex(
            0 if x.total_sample_count_source == "reference_sample_rate" else 1
        )
        self.sp_reference_sample_rate.setValue(int(x.reference_sample_rate))
        self._set_selected_memory_process(str(x.memory_process_name), int(x.memory_process_pid))
        self._set_memory_deck(self.deck1_specs, x.memory_deck1)
        self._set_memory_deck(self.deck2_specs, x.memory_deck2)
        self._sync_external_sync_mode_ui()

    def _set_band(self, sp_lo: QDoubleSpinBox, sp_hi: QDoubleSpinBox, band: Tuple[float, float]):
        sp_lo.setValue(float(band[0])); sp_hi.setValue(float(band[1]))

    def _set_memory_value(self, widgets: dict, spec: memoryvalueconfig) -> None:
        widgets["offsets"].setText(str(spec.offsets))
        widgets["value_type"].setCurrentText(str(spec.value_type))
        if "length" in widgets:
            widgets["length"].setValue(int(spec.length))
        if "encoding" in widgets:
            widgets["encoding"].setText(str(spec.encoding))
        if "bit_pos" in widgets:
            widgets["bit_pos"].setValue(int(spec.bit_pos))
        if "multiplier" in widgets:
            widgets["multiplier"].setValue(float(spec.multiplier))

    def _get_memory_value(self, widgets: dict) -> memoryvalueconfig:
        return memoryvalueconfig(
            offsets=widgets["offsets"].text().strip(),
            value_type=widgets["value_type"].currentText(),
            length=int(widgets["length"].value()) if "length" in widgets else 0,
            encoding=widgets["encoding"].text().strip() if "encoding" in widgets else "utf-8",
            bit_pos=int(widgets["bit_pos"].value()) if "bit_pos" in widgets else 0,
            multiplier=float(widgets["multiplier"].value()) if "multiplier" in widgets else 1.0,
        )

    def _set_memory_deck(self, widgets: dict, deck_cfg: memorydeckconfig) -> None:
        self._set_memory_value(widgets["time"], deck_cfg.time)
        self._set_memory_value(widgets["sample_index"], deck_cfg.sample_index)
        self._set_memory_value(widgets["path"], deck_cfg.path)
        self._set_memory_value(widgets["active"], deck_cfg.active)
        self._set_memory_value(widgets["loaded"], deck_cfg.loaded)

    def _get_memory_deck(self, widgets: dict) -> memorydeckconfig:
        return memorydeckconfig(
            time=self._get_memory_value(widgets["time"]),
            sample_index=self._get_memory_value(widgets["sample_index"]),
            path=self._get_memory_value(widgets["path"]),
            active=self._get_memory_value(widgets["active"]),
            loaded=self._get_memory_value(widgets["loaded"]),
        )

    def get_config(self):
        return config(
            analysisconfig=analysisconfig(
                bpm_min=int(self.sp_bpm_min.value()),
                bpm_dynamic=bool(self.cb_bpm_dynamic.isChecked()),
                dynamic_downbeat=bool(
                    self.cb_dynamic_downbeat.isChecked() if not self.cb_bpm_dynamic.isChecked()
                    else getattr(self, "_static_dynamic_downbeat", True)
                ),
                refine_onset_parameter_path=self.ed_refine_onset_parameter_path.text().strip(),
                onset_feature_cache_path=self.ed_onset_feature_cache_path.text().strip(),
                beat_tcn_parameter_path=self.ed_beat_tcn_parameter_path.text().strip(),
                beat_critic_parameter_path=self.ed_beat_critic_parameter_path.text().strip(),
                beat_decoder_parameter_path=self.ed_beat_decoder_parameter_path.text().strip(),
                downbeat_ctx_parameter_path=self.ed_downbeat_ctx_parameter_path.text().strip(),
                beat_feature_cache_path=self.ed_beat_feature_cache_path.text().strip(),
                phrase_analysis_enabled=bool(self.cb_phrase_analysis_enabled.isChecked()),
                phrase_parameter_path=self.ed_phrase_parameter_path.text().strip(),
                phrase_feature_cache_path=self.ed_phrase_feature_cache_path.text().strip(),
                jumpcue_parameter_path=self.ed_jumpcue_parameter_path.text().strip(),
                jumpcue_feature_cache_path=self.ed_jumpcue_feature_cache_path.text().strip(),
                key_dynamic=bool(self.cb_key_dynamic.isChecked()),
                key_parameter_path=self.ed_key_parameter_path.text().strip(),
                key_feature_cache_path=self.ed_key_feature_cache_path.text().strip(),
                autocue=self._get_autocue(),
            ),
            libconfig=libconfig(
                libpath=self.ed_libpath.text().strip(),
                local_music_directories=self._get_local_music_directories(),
                write_log=bool(self.cb_write_log.isChecked()),
                logpath=self.ed_logpath.text().strip(),
                rekordbox_sync_enabled=bool(self.cb_rekordbox_sync.isChecked()),
                rekordbox_xml_path=self.ed_rekordbox_xml_path.text().strip(),
            ),
            viewconfig=viewconfig(
                display_waveform=bool(self.cb_waveform.isChecked()),
                display_beatgrid=bool(self.cb_beatgrid.isChecked()),
                display_keystrip=bool(self.cb_keystrip.isChecked()),
                display_JumpCUE=bool(self.cb_JumpCUE.isChecked()),
                display_phrase=bool(self.cb_phrase.isChecked()),
                waveform_style=str(self.cmb_waveform_style.currentData()),
                env_lo=(float(self.sp_env_lo_lo.value()), float(self.sp_env_lo_hi.value())),
                env_mid=(float(self.sp_env_mid_lo.value()), float(self.sp_env_mid_hi.value())),
                env_hi=(float(self.sp_env_hi_lo.value()), float(self.sp_env_hi_hi.value())),
                mainview_wave_frame_ms_short=round(float(self.sp_mainview_wave_frame_ms_short.value()), 1),
                mainview_wave_frame_ms_long=round(float(self.sp_mainview_wave_frame_ms_long.value()), 1),
                mainview_wave_filter_order=int(self.sp_mainview_wave_filter_order.value()),
                use_output_volume_as_peak_meter_input=bool(
                    self.cb_use_output_volume_as_peak_meter_input.isChecked()
                ),
                fps=int(self.sp_fps.value()),
                reduce_fps_when_occluded=bool(self.cb_reduce_fps_when_occluded.isChecked()),
                record_img_path=self.ed_record_img_path.text().strip(),
                phrase_labels=self._get_phrase_label_names(),
            ),
            playbackconfig=playbackconfig(
                enable_metronome=bool(self.cb_metronome.isChecked()),
                metronome_wav_path=self.ed_metronome_wav_path.text().strip(),
                metronome_offset_msec=float(self.sp_metronome_offset_msec.value()),
                volume_trim_dbfs=float(self.sp_volume_trim_dbfs.value()),
                default_volume_percent=int(self.sp_default_volume_percent.value()),
                use_timestretch=bool(self.cb_use_timestretch.isChecked()),
                metronome_downbeat_volume_percent=int(self.sp_click_down_volume.value()),
                metronome_downbeat_pitch_semitones=float(self.sp_click_down_pitch.value()),
                metronome_beat_volume_percent=int(self.sp_click_beat_volume.value()),
                metronome_beat_pitch_semitones=float(self.sp_click_beat_pitch.value()),
                metronome_ducking=bool(self.cb_metronome_ducking.isChecked()),
                soft_clip=bool(self.cb_soft_clip.isChecked()),
            ),
            externalsyncconfig=externalsyncconfig(
                enabled=bool(self.cb_external_sync_enabled.isChecked()),
                mode=("time" if self.cmb_external_sync_mode.currentIndex() == 0 else "sample_index"),
                total_sample_count_source=(
                    "reference_sample_rate"
                    if self.cmb_total_sample_count_source.currentIndex() == 0
                    else "file"
                ),
                reference_sample_rate=int(self.sp_reference_sample_rate.value()),
                memory_process_name=self._memory_process_name(),
                memory_process_pid=self._memory_process_pid(),
                memory_deck1=self._get_memory_deck(self.deck1_specs),
                memory_deck2=self._get_memory_deck(self.deck2_specs),
            ),
            cuepointconfig=self._get_cue_point_rules(),
            searchconfig=self._get_search_config(),
        )

    def get_dict(self) -> dict:
        return asdict(self.get_config())

    # Buttons
    def _on_apply(self):
        cfg = self.get_config()
        # Save and Reload Requenst
        self.saveJsonRequested.emit(cfg)

    def _on_ok(self):
        cfg = self.get_config()
        # Save and Reload Requenst
        self.saveJsonRequested.emit(cfg)
        self.accept()

    def _on_cancel(self):
        self.reject()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._reset_playback_monitor()
