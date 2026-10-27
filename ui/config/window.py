import copy
from dataclasses import is_dataclass
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

from core.settings.config import (
    analysisconfig,
    config,
    deckconfig,
    default_cfg,
    externalsyncconfig,
    libconfig,
    playbackconfig,
    viewconfig,
)
from audioengine.controls import booth_eq_db_to_knob, level_db_to_knob
from core.workflows.event_bus import EventBus
from system.i18n import tr
from ui.config.analysis import AnalysisSettingsMixin, build_analysis_tab
from ui.config.external_sync import (
    ExternalSyncSettingsMixin,
    build_external_sync_tab,
)
from ui.config.library import LibrarySettingsMixin, build_library_tab
from ui.config.midi import MidiSettingsMixin, build_midi_tab
from ui.config.playback import PlaybackSettingsMixin, build_playback_tab
from ui.config.search import SearchSettingsMixin, build_search_tab
from ui.config.shortcuts import ShortcutSettingsMixin, build_shortcuts_tab
from ui.config.view import ViewSettingsMixin, build_view_tab


class SettingsDialog(
    LibrarySettingsMixin,
    SearchSettingsMixin,
    ViewSettingsMixin,
    PlaybackSettingsMixin,
    AnalysisSettingsMixin,
    ExternalSyncSettingsMixin,
    ShortcutSettingsMixin,
    MidiSettingsMixin,
    QDialog,
):
    DIALOG_WIDTH = 980
    FORM_LABEL_WIDTH = 260
    COMPACT_FORM_LABEL_WIDTH = 220

    def __init__(self, bus: EventBus):
        super().__init__()
        self.saveJsonRequested = bus.sig_setting_saveJsonRequested
        self._bus = bus
        self.setWindowTitle(tr("Settings"))
        self.setModal(False)

        self.tabs = QTabWidget(self)
        build_view_tab(self)
        build_playback_tab(self)
        build_search_tab(self)
        build_library_tab(self)
        build_analysis_tab(self)
        build_shortcuts_tab(self)
        build_midi_tab(self)
        build_external_sync_tab(self)
        self._stabilize_layout_widths()

        self.btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        self.btn_box.accepted.connect(self._on_ok)
        self.btn_box.rejected.connect(self._on_cancel)
        self.btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self.btn_box.button(QDialogButtonBox.Ok).setToolTip(tr("Save the settings and close."))
        self.btn_box.button(QDialogButtonBox.Cancel).setToolTip(
            tr("Close without saving unapplied changes.")
        )
        self.btn_box.button(QDialogButtonBox.Apply).setToolTip(
            tr("Save the settings and keep the window open.")
        )

        root = QVBoxLayout(self)
        root.addWidget(self.tabs)
        root.addWidget(self.btn_box)

        # Keep page-specific size hints from changing the top-level window width.
        # The height remains resizable (the pages scroll).
        self.setFixedWidth(self.DIALOG_WIDTH)
        self.resize(self.DIALOG_WIDTH, 860)

        self._current_cfg = None
        self._cue_mix_shown = None   # the headphones' CUE / MASTER mix as last shown (_show_cue_mix)
        self._library = None      # the library's owner (set_library)
        self._initialize_library_settings(bus)
        self._initialize_search_settings(bus)
        self._initialize_view_settings(bus)
        self._initialize_playback_settings(bus)
        self.audio_tabs.currentChanged.connect(lambda _i: self._refresh_overview())
        self.tabs.currentChanged.connect(lambda _i: self._refresh_overview())
        self._initialize_shortcut_settings()
        self._initialize_midi_settings(bus)

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
    def set_library(self, library) -> None:
        """The library's owner (core.library.service): what the pages show of the library is asked of it."""
        self._library = library

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
        self.sp_rekordbox_ramp_tolerance_ms.setValue(float(cfg.libconfig.rekordbox_ramp_tolerance_ms))
        self.cb_rekordbox_comment_tieups.setChecked(bool(cfg.libconfig.rekordbox_comment_tieups))
        self.cb_tieup_to_versions.setChecked(bool(cfg.libconfig.tieup_to_versions))
        self.cb_tieup_web_search.setChecked(bool(cfg.libconfig.tieup_web_search))
        self.cb_sync_stored_analysis.setChecked(bool(cfg.libconfig.sync_stored_analysis))
        self.cb_backup_feature_cache.setChecked(bool(cfg.libconfig.backup_feature_cache))

        # search
        self._set_search_config(cfg.searchconfig)

        # view
        v = cfg.viewconfig
        self._set_language(v.language)
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

        p, deck = cfg.playbackconfig, cfg.deckconfig
        self.ed_metronome_wav_path.setText(deck.metronome_wav_path)
        self.sp_metronome_offset_msec.setValue(float(deck.metronome_offset_msec))
        self.sp_default_volume_percent.setValue(int(p.default_volume_percent))
        self.sp_click_down_volume.setValue(int(deck.metronome_downbeat_volume_percent))
        self.sp_click_down_pitch.setValue(float(deck.metronome_downbeat_pitch_semitones))
        self.sp_click_beat_volume.setValue(int(deck.metronome_beat_volume_percent))
        self.sp_click_beat_pitch.setValue(float(deck.metronome_beat_pitch_semitones))
        self.cb_metronome_ducking.setChecked(bool(deck.metronome_ducking))
        self.cb_soft_clip.setChecked(bool(p.soft_clip))
        self.cb_audio_engine.setCurrentIndex(max(0, self.cb_audio_engine.findData(p.audio_engine)))
        self.sp_native_buffer_ms.setValue(int(p.native_buffer_ms))
        self.sp_python_buffer_ms.setValue(int(p.python_buffer_ms))
        self.cb_native_api.setCurrentIndex(max(0, self.cb_native_api.findData(p.native_audio_api)))
        self.cb_native_output_mode.setCurrentIndex(1 if p.native_exclusive else 0)
        self.cb_native_rate.setCurrentIndex(max(0, self.cb_native_rate.findData(int(p.native_exclusive_rate))))
        self.cb_native_priority.setCurrentIndex(max(0, self.cb_native_priority.findData(p.native_priority)))
        self.cb_native_power.setCurrentIndex(1 if p.native_power_throttling else 0)
        self._output_device = p.output_device
        self._load_output_devices(p.output_device)
        self.grp_output_headphones.set_config(p.output_headphones)
        self.grp_output_headphones.cb_master_mix.setChecked(bool(p.headphones_master_mix))
        self._show_cue_mix(int(deck.headphones_cue_mix))
        self.grp_output_headphones.sp_cue_mix.setEnabled(bool(p.headphones_master_mix))
        self.grp_output_booth.set_config(p.output_booth)
        self._set_audio_mix_config(p)
        self._load_output_routes(p.output_main)
        self._set_midi_config(p)

        # analysis
        a = cfg.analysisconfig
        self.sp_max_parallel_analyses.setValue(int(a.max_parallel_analyses))
        self.sp_bpm_min.setValue(int(a.bpm_min))
        self.lbl_bpm_range.setText(tr("Reads {low} - {high} BPM").format(low=int(a.bpm_min), high=2 * int(a.bpm_min)))
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
        self.ed_external_sync_host.setText(str(x.osc_host or ""))
        self.sp_external_sync_port.setValue(int(x.osc_port))
        self._sync_external_sync_address_ui()

    def _kept_deck(self) -> deckconfig:
        """deck.json's settings as last kept (the main screen's, a controller's: the dialog shows only the
        metronome's sound and the headphones' CUE / MASTER mix of them)."""
        kept = getattr(self._current_cfg, "deckconfig", None)
        return copy.deepcopy(kept) if kept is not None else default_cfg().deckconfig

    def _set_band(self, sp_lo: QDoubleSpinBox, sp_hi: QDoubleSpinBox, band: Tuple[float, float]):
        sp_lo.setValue(float(band[0])); sp_hi.setValue(float(band[1]))

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
                phrase_parameter_path=self.ed_phrase_parameter_path.text().strip(),
                phrase_feature_cache_path=self.ed_phrase_feature_cache_path.text().strip(),
                jumpcue_parameter_path=self.ed_jumpcue_parameter_path.text().strip(),
                jumpcue_feature_cache_path=self.ed_jumpcue_feature_cache_path.text().strip(),
                key_dynamic=bool(self.cb_key_dynamic.isChecked()),
                key_parameter_path=self.ed_key_parameter_path.text().strip(),
                key_feature_cache_path=self.ed_key_feature_cache_path.text().strip(),
                autocue=self._get_autocue(),
                max_parallel_analyses=int(self.sp_max_parallel_analyses.value()),
            ),
            libconfig=libconfig(
                libpath=self.ed_libpath.text().strip(),
                local_music_directories=self._get_local_music_directories(),
                write_log=bool(self.cb_write_log.isChecked()),
                logpath=self.ed_logpath.text().strip(),
                rekordbox_sync_enabled=bool(self.cb_rekordbox_sync.isChecked()),
                rekordbox_xml_path=self.ed_rekordbox_xml_path.text().strip(),
                rekordbox_ramp_tolerance_ms=float(self.sp_rekordbox_ramp_tolerance_ms.value()),
                rekordbox_comment_tieups=bool(self.cb_rekordbox_comment_tieups.isChecked()),
                tieup_to_versions=bool(self.cb_tieup_to_versions.isChecked()),
                tieup_web_search=bool(self.cb_tieup_web_search.isChecked()),
                sync_stored_analysis=bool(self.cb_sync_stored_analysis.isChecked()),
                backup_feature_cache=bool(self.cb_backup_feature_cache.isChecked()),
            ),
            viewconfig=viewconfig(
                language=self._get_language(),
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
                default_volume_percent=int(self.sp_default_volume_percent.value()),
                soft_clip=bool(self.cb_soft_clip.isChecked()),
                audio_engine=str(self.cb_audio_engine.currentData() or "native"),
                native_buffer_ms=int(self.sp_native_buffer_ms.value()),
                python_buffer_ms=int(self.sp_python_buffer_ms.value()),
                output_device=str(self.cb_output_device.currentData() or ""),
                native_audio_api=str(self.cb_native_api.currentData() or "cpal"),
                native_exclusive=bool(self.cb_native_output_mode.currentData()),
                native_exclusive_rate=int(self.cb_native_rate.currentData() or 0),
                native_priority=str(self.cb_native_priority.currentData() or "high"),
                native_power_throttling=bool(self.cb_native_power.currentData()),
                midi_input=self._midi_input(),
                midi_jog_ticks=int(self.sp_midi_jog_ticks.value()),
                output_main=str(self.cb_output_main.currentData() or "1-2"),
                output_headphones=self.grp_output_headphones.get_config(),
                output_booth=self.grp_output_booth.get_config(),
                headphones_master_mix=bool(self.grp_output_headphones.cb_master_mix.isChecked()),
                booth_level=level_db_to_knob(self.sp_booth_level.value()),
                booth_eq_hi=booth_eq_db_to_knob(self.sp_booth_eq_hi.value()),
                booth_eq_low=booth_eq_db_to_knob(self.sp_booth_eq_low.value()),
                headphones_split=bool(self.cb_headphones_split.isChecked()),
            ),
            externalsyncconfig=externalsyncconfig(
                enabled=bool(self.cb_external_sync_enabled.isChecked()),
                osc_host=self._external_sync_host(),
                osc_port=int(self.sp_external_sync_port.value()),
            ),
            cuepointconfig=self._get_cue_point_rules(),
            searchconfig=self._get_search_config(),
            deckconfig=self._get_deck_config(),
        )

    def _get_deck_config(self) -> deckconfig:
        """deck.json's settings: as kept, with the ones this dialog edits (the metronome's sound, the headphones'
        CUE / MASTER mix)."""
        deck = self._kept_deck()
        deck.metronome_wav_path = self.ed_metronome_wav_path.text().strip()
        deck.metronome_offset_msec = float(self.sp_metronome_offset_msec.value())
        deck.metronome_downbeat_volume_percent = int(self.sp_click_down_volume.value())
        deck.metronome_downbeat_pitch_semitones = float(self.sp_click_down_pitch.value())
        deck.metronome_beat_volume_percent = int(self.sp_click_beat_volume.value())
        deck.metronome_beat_pitch_semitones = float(self.sp_click_beat_pitch.value())
        deck.metronome_ducking = bool(self.cb_metronome_ducking.isChecked())
        # the CUE / MASTER mix a controller also moves: written only when changed here, else the one kept stays
        cue_mix = int(self.grp_output_headphones.sp_cue_mix.value())
        if cue_mix != self._cue_mix_shown:
            deck.headphones_cue_mix = cue_mix
        return deck

    def _show_cue_mix(self, value: int) -> None:
        """The headphones' CUE / MASTER mix as kept (deck.json: this dialog's and a controller's)."""
        self._cue_mix_shown = int(value)
        self.grp_output_headphones.sp_cue_mix.setValue(int(value))

    # Buttons
    def _on_apply(self):
        cfg = self.get_config()
        self._save_shortcut_settings()
        self._save_midi_settings()
        # Save and Reload Requenst
        self.saveJsonRequested.emit(cfg)

    def _on_ok(self):
        cfg = self.get_config()
        self._save_shortcut_settings()
        self._save_midi_settings()
        # Save and Reload Requenst
        self.saveJsonRequested.emit(cfg)
        self.accept()

    def _on_cancel(self):
        self._stop_midi_learn()
        self.reject()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._reset_playback_monitor()
        # the devices connected now (one plugged in since)
        self._load_output_devices(str(self.cb_output_device.currentData() or getattr(self, "_output_device", "")))
        self._load_output_routes(str(self.cb_output_main.currentData() or "1-2"))
        self._load_shortcut_settings()  # the edits not kept are dropped
        self._load_midi_settings()
        if int(self.grp_output_headphones.sp_cue_mix.value()) == self._cue_mix_shown:   # not changed here
            self._show_cue_mix(int(self._kept_deck().headphones_cue_mix))       # a controller's since
        self._refresh_overview()
