from typing import Tuple
from collections import deque
from dataclasses import is_dataclass, asdict
import csv
import io
import json
from pathlib import Path
import subprocess
from PySide6.QtCore import Qt, Signal
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLineEdit, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QDialogButtonBox, QGroupBox, QPushButton,
    QScrollArea)
from core.config import (
    config, libconfig, viewconfig, analysisconfig, keyconfig, externalsyncconfig,
    memorydeckconfig, memoryvalueconfig,
)
from core.event_bus import EventBus

class SettingsDialog(QDialog):

    def __init__(self, bus:EventBus=None):
        super().__init__()
        self.saveJsonRequested = bus.sig_setting_saveJsonRequested
        self._bus = bus
        self._refresh_samples = deque(maxlen=32)
        self.setWindowTitle("Settings")
        self.setModal(False)

        self.tabs = QTabWidget(self)
        self._make_tab_library()
        self._make_tab_view()
        self._make_tab_analysis()
        self._make_tab_key()
        self._make_tab_external_sync()

        self.btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )

        self.btn_box.accepted.connect(self._on_ok)
        self.btn_box.rejected.connect(self._on_cancel)
        self.btn_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)

        root = QVBoxLayout(self)
        root.addWidget(self.tabs)
        root.addWidget(self.btn_box)

        self._current_cfg = None 
        if self._bus is not None:
            self._bus.sig_ui_draw_interval.connect(self._on_ui_draw_interval)

    # Tabs
    def _make_tab_library(self):
        self.tab_lib = QWidget()
        f = QFormLayout(self.tab_lib)
        self.ed_libpath = QLineEdit()
        self.cb_write_log = QCheckBox("Write log to file")
        self.ed_logpath = QLineEdit()
        self.lbl_log_note = QLabel("Logging settings apply on next app launch.")
        self.lbl_log_note.setWordWrap(True)
        f.addRow("Library Path", self.ed_libpath)
        f.addRow(self.cb_write_log)
        f.addRow("Log Path", self.ed_logpath)
        f.addRow(self.lbl_log_note)
        self.tabs.addTab(self.tab_lib, "Library")

    def _make_tab_view(self):
        self.tab_view = QWidget()
        f = QFormLayout(self.tab_view)

        self.cb_waveform = QCheckBox("Display waveform")
        self.cb_beatgrid = QCheckBox("Display beatgrid")
        self.cb_keystrip = QCheckBox("Display keystrip")
        self.cb_JumpCUE = QCheckBox("Display JumpCUE")
        self.cb_metronome = QCheckBox("Enable metronome")
        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(1, 240)
        self.sp_fps.setSingleStep(5)

        self.ed_record_img_path = QLineEdit()
        self.ed_metronome_wav_path = QLineEdit()
        self.lbl_fps = QLabel("Waiting for FPS...")
        self.lbl_fps.setToolTip("Measured from actual pyqtgraph plot repaint intervals.")

        f.addRow(self.cb_waveform)
        f.addRow(self.cb_beatgrid)
        f.addRow(self.cb_keystrip)
        f.addRow(self.cb_JumpCUE)
        f.addRow(self.cb_metronome)
        f.addRow("FPS", self.sp_fps)
        f.addRow("Record image path", self.ed_record_img_path)
        f.addRow("Metronome WAV path", self.ed_metronome_wav_path)
        f.addRow("Actual FPS", self.lbl_fps)

        self.tabs.addTab(self.tab_view, "View")

    def _make_tab_analysis(self):
        self.tab_analysis = QWidget()
        root = QVBoxLayout(self.tab_analysis)
        analysis_tabs = QTabWidget(self.tab_analysis)

        # Global
        tab_global = QWidget(); f_global = QFormLayout(tab_global)
        self.sp_analysis_samp_rate = QSpinBox(); self.sp_analysis_samp_rate.setRange(8000, 384000); self.sp_analysis_samp_rate.setSingleStep(1000)
        self.cb_use_hpss = QCheckBox("Use HPSS")
        f_global.addRow("Analysis sample rate (SPS)", self.sp_analysis_samp_rate)
        f_global.addRow(self.cb_use_hpss)
        

        # Beat
        tab_beat = QWidget(); f_beat = QFormLayout(tab_beat)
        self.cb_bpm_dynamic = QCheckBox("Use Dynamic Analysis")
        self.cb_bpm_adaptive_win = QCheckBox("Use Adaptive Window for Dynamic Analysis")
        self.sp_bpm_hop = QSpinBox(); self.sp_bpm_hop.setRange(16, 512); self.sp_bpm_hop.setSingleStep(32)
        self.sp_bpm_win = QSpinBox(); self.sp_bpm_win.setRange(1000, 60000); self.sp_bpm_win.setSingleStep(64)
        self.sp_bpm_min = QSpinBox(); self.sp_bpm_min.setRange(60,  400)
        self.sp_bpm_max = QSpinBox(); self.sp_bpm_max.setRange(100, 800)
        self.sp_beatgrid_offset = QDoubleSpinBox(); self.sp_beatgrid_offset.setRange(-10000.0, 10000.0); self.sp_beatgrid_offset.setDecimals(3); self.sp_beatgrid_offset.setSingleStep(1.0)
        f_beat.addRow(self.cb_bpm_dynamic)
        f_beat.addRow(self.cb_bpm_adaptive_win)
        f_beat.addRow("BPM hop length (samp)", self.sp_bpm_hop)
        f_beat.addRow("BPM Autocorrelation win_length (ms)", self.sp_bpm_win)
        f_beat.addRow("BPM min", self.sp_bpm_min)
        f_beat.addRow("BPM max", self.sp_bpm_max)
        f_beat.addRow("Beatgrid offset (msec)", self.sp_beatgrid_offset)

        # Key (Chroma analysis)
        tab_key = QWidget(); f_key = QFormLayout(tab_key)
        self.cmb_chroma_method = QComboBox(); self.cmb_chroma_method.addItems(["cqt", "cens"])
        self.sp_chroma_hop_length = QSpinBox(); self.sp_chroma_hop_length.setRange(32, 8192); self.sp_chroma_hop_length.setSingleStep(32)
        self.sp_cqt_bins_per_oct = QSpinBox(); self.sp_cqt_bins_per_oct.setRange(1, 96)
        self.sp_cqt_octaves = QSpinBox(); self.sp_cqt_octaves.setRange(1, 10)
        f_key.addRow("Chroma method", self.cmb_chroma_method)
        f_key.addRow("Chroma hop length (samp)", self.sp_chroma_hop_length)
        f_key.addRow("CQT bins per octave", self.sp_cqt_bins_per_oct)
        f_key.addRow("CQT octaves", self.sp_cqt_octaves)

        # Advanced (Viterbi transition probabilities)
        grp_adv = QGroupBox("Advanced")
        f_adv = QFormLayout(grp_adv)
        self.sp_min_offset = QDoubleSpinBox(); self.sp_min_offset.setDecimals(3); self.sp_min_offset.setRange(0.0, 1e6)
        self.sp_pitch_self = QDoubleSpinBox(); self.sp_pitch_self.setRange(-1e9, 1e9); self.sp_pitch_self.setDecimals(6)
        self.sp_pitch_semitone = QDoubleSpinBox(); self.sp_pitch_semitone.setRange(-1e9, 1e9); self.sp_pitch_semitone.setDecimals(6)
        self.sp_pitch_fifth = QDoubleSpinBox(); self.sp_pitch_fifth.setRange(-1e9, 1e9); self.sp_pitch_fifth.setDecimals(6)
        self.sp_pitch_others = QDoubleSpinBox(); self.sp_pitch_others.setRange(-1e9, 1e9); self.sp_pitch_others.setDecimals(6)
        f_adv.addRow("Min offset", self.sp_min_offset)
        f_adv.addRow("Pitch: self", self.sp_pitch_self)
        f_adv.addRow("Pitch: semitone", self.sp_pitch_semitone)
        f_adv.addRow("Pitch: fifth", self.sp_pitch_fifth)
        f_adv.addRow("Pitch: others", self.sp_pitch_others)
        f_key.addRow(grp_adv)

        # Waveform (Envelope bands)
        tab_wave = QWidget(); f_wave = QFormLayout(tab_wave)
        def make_band_row(parent_form, label: str):
            box = QWidget(); h = QHBoxLayout(box); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
            lo = QDoubleSpinBox(); hi = QDoubleSpinBox()
            for sp in (lo, hi):
                sp.setDecimals(3); sp.setRange(0.0, 1e6); sp.setSingleStep(1.0); sp.setMinimumWidth(100)
            h.addWidget(QLabel("lo")); h.addWidget(lo)
            h.addWidget(QLabel("hi")); h.addWidget(hi)
            parent_form.addRow(label, box)
            return lo, hi
        self.sp_env_frame_ms = QSpinBox(); self.sp_env_frame_ms.setRange(1, 10)
        self.sp_env_lo_lo,  self.sp_env_lo_hi  = make_band_row(f_wave, "Env band (lo)")
        self.sp_env_mid_lo, self.sp_env_mid_hi = make_band_row(f_wave, "Env band (mid)")
        self.sp_env_hi_lo,  self.sp_env_hi_hi  = make_band_row(f_wave, "Env band (hi)")
        self.sp_env_order = QSpinBox(); self.sp_env_order.setRange(1, 12)
        f_wave.addRow("Env frame (ms)", self.sp_env_frame_ms)
        f_wave.addRow("Env order", self.sp_env_order)

        # Assemble
        analysis_tabs.addTab(tab_global, "Global")
        analysis_tabs.addTab(tab_beat, "Beat")
        analysis_tabs.addTab(tab_key, "Key")
        analysis_tabs.addTab(tab_wave, "Waveform")

        root.addWidget(analysis_tabs)
        self.tabs.addTab(self.tab_analysis, "Analysis")

    def _make_tab_key(self):
        # Deprecated: moved under Analysis -> Key (Advanced)
        pass

    def _make_tab_external_sync(self):
        self.tab_external_sync = QWidget()
        root = QVBoxLayout(self.tab_external_sync)
        f = QFormLayout()

        self.cb_external_sync_enabled = QCheckBox("Enable external sync")
        self.cmb_external_sync_mode = QComboBox()
        self.cmb_external_sync_mode.addItems(["Time Sync", "Sample Index Sync"])

        self.cmb_memory_process = QComboBox()
        self.cmb_memory_process.setEditable(True)
        self.btn_memory_process_refresh = QPushButton("Refresh")
        self.deck1_specs = self._make_memory_deck_group("Deck 1")
        self.deck2_specs = self._make_memory_deck_group("Deck 2")

        self.grp_memory = QGroupBox("Memory")
        f_memory = QFormLayout(self.grp_memory)
        process_row = QWidget()
        process_row_layout = QHBoxLayout(process_row)
        process_row_layout.setContentsMargins(0, 0, 0, 0)
        process_row_layout.setSpacing(6)
        process_row_layout.addWidget(self.cmb_memory_process, 1)
        process_row_layout.addWidget(self.btn_memory_process_refresh, 0)
        f_memory.addRow("Process", process_row)
        self.lbl_memory_process_warning = QLabel(
            "Warning: attaching memory sync to an unrelated process can trigger "
            "anti-cheat or antivirus false positives. Only target software you trust "
            "and explicitly intend to sync with."
        )
        self.lbl_memory_process_warning.setWordWrap(True)
        self.lbl_memory_process_warning.setStyleSheet("color: #d8a441;")
        f_memory.addRow(self.lbl_memory_process_warning)
        self.lbl_memory_process_blocked = QLabel("")
        self.lbl_memory_process_blocked.setWordWrap(True)
        self.lbl_memory_process_blocked.setStyleSheet("color: #d86f41;")
        f_memory.addRow(self.lbl_memory_process_blocked)

        self.lbl_external_sync_note = QLabel(
            "When enabled, local track loading and transport playback are restricted "
            "so the external software remains the source of truth."
        )
        self.lbl_external_sync_note.setWordWrap(True)

        f.addRow(self.cb_external_sync_enabled)
        f.addRow("Mode", self.cmb_external_sync_mode)
        f.addRow(self.lbl_external_sync_note)
        root.addLayout(f)
        self.memory_scroll = QScrollArea()
        self.memory_scroll.setWidgetResizable(True)
        self.memory_scroll.setMinimumHeight(340)
        self.memory_scroll_contents = QWidget()
        memory_scroll_layout = QVBoxLayout(self.memory_scroll_contents)
        memory_scroll_layout.setContentsMargins(0, 0, 0, 0)
        memory_scroll_layout.setSpacing(8)
        memory_scroll_layout.addWidget(self.grp_memory)
        memory_scroll_layout.addWidget(self.deck1_specs["group"])
        memory_scroll_layout.addWidget(self.deck2_specs["group"])
        memory_scroll_layout.addStretch(1)
        self.memory_scroll.setWidget(self.memory_scroll_contents)
        root.addWidget(self.memory_scroll)
        root.addStretch(1)

        self.cmb_external_sync_mode.currentIndexChanged.connect(self._sync_external_sync_mode_ui)
        self.btn_memory_process_refresh.clicked.connect(self._refresh_memory_processes)
        self._refresh_memory_processes()
        self._sync_external_sync_mode_ui()

        self.tabs.addTab(self.tab_external_sync, "External Sync")

    # Public API
    def set_config(self, cfg: config):
        assert is_dataclass(cfg), "cfg must be a dataclass 'config'"
        self._current_cfg = cfg

        # lib
        self.ed_libpath.setText(cfg.libconfig.libpath)
        self.cb_write_log.setChecked(bool(cfg.libconfig.write_log))
        self.ed_logpath.setText(cfg.libconfig.logpath)

        # view
        v = cfg.viewconfig
        self.cb_waveform.setChecked(bool(v.display_waveform))
        self.cb_beatgrid.setChecked(bool(v.display_beatgrid))
        self.cb_keystrip.setChecked(bool(v.display_keystrip))
        self.cb_JumpCUE.setChecked(bool(v.display_JumpCUE))
        self.cb_metronome.setChecked(bool(v.enable_metronome))
        self.sp_fps.setValue(int(v.fps))
        self.ed_record_img_path.setText(v.record_img_path)
        self.ed_metronome_wav_path.setText(v.metronome_wav_path)

        # analysis
        a = cfg.analysisconfig
        self.sp_analysis_samp_rate.setValue(int(a.analysis_samp_rate))
        self.cmb_chroma_method.setCurrentIndex(0 if a.chroma_method == "cqt" else 1)
        self.sp_chroma_hop_length.setValue(int(a.chroma_hop_length))
        self.sp_cqt_bins_per_oct.setValue(int(a.chroma_cqt_bins_per_octave))
        self.sp_cqt_octaves.setValue(int(a.chroma_cqt_octaves))
        self.cb_use_hpss.setChecked(bool(a.use_hpss))
        self.sp_bpm_hop.setValue(int(a.bpm_hop_length))
        self.sp_bpm_win.setValue(int(a.bpm_win_length))
        self.sp_bpm_min.setValue(int(a.bpm_min))
        self.sp_bpm_max.setValue(int(a.bpm_max))
        self.cb_bpm_dynamic.setChecked(bool(a.bpm_dynamic))
        self.cb_bpm_adaptive_win.setChecked(bool(a.bpm_adaptive_window))
        self.sp_beatgrid_offset.setValue(float(a.beatgrid_offset_msec))
        self.sp_env_frame_ms.setValue(int(a.env_frame_ms))
        self._set_band(self.sp_env_lo_lo,  self.sp_env_lo_hi,  a.env_lo)
        self._set_band(self.sp_env_mid_lo, self.sp_env_mid_hi, a.env_mid)
        self._set_band(self.sp_env_hi_lo,  self.sp_env_hi_hi,  a.env_hi)
        self.sp_env_order.setValue(int(a.env_order))

        # key
        k = cfg.keyconfig
        self.sp_min_offset.setValue(float(k.min_offset))
        self.sp_pitch_self.setValue(float(k.pitch_self))
        self.sp_pitch_semitone.setValue(float(k.pitch_semitone))
        self.sp_pitch_fifth.setValue(float(k.pitch_fifth))
        self.sp_pitch_others.setValue(float(k.pitch_others))

        # external sync
        x = cfg.externalsyncconfig
        self.cb_external_sync_enabled.setChecked(bool(x.enabled))
        self.cmb_external_sync_mode.setCurrentIndex(0 if x.mode == "time" else 1)
        self._set_selected_memory_process(str(x.memory_process_name), int(x.memory_process_pid))
        self._set_memory_deck(self.deck1_specs, x.memory_deck1)
        self._set_memory_deck(self.deck2_specs, x.memory_deck2)
        self._sync_external_sync_mode_ui()

    def _set_band(self, sp_lo: QDoubleSpinBox, sp_hi: QDoubleSpinBox, band: Tuple[float, float]):
        sp_lo.setValue(float(band[0])); sp_hi.setValue(float(band[1]))

    def _make_memory_value_group(
        self,
        title: str,
        *,
        include_length: bool = False,
        include_encoding: bool = False,
        include_bit_pos: bool = False,
        include_multiplier: bool = False,
    ) -> dict:
        group = QGroupBox(title)
        form = QFormLayout(group)
        ed_offsets = QLineEdit()
        ed_offsets.setPlaceholderText("00000000,00,00,00")
        cmb_type = QComboBox()
        cmb_type.addItems(["float", "bool", "str", "int"])
        widgets = {
            "group": group,
            "offsets": ed_offsets,
            "value_type": cmb_type,
        }
        form.addRow("Offset chain", ed_offsets)
        form.addRow("Type", cmb_type)
        if include_length:
            sp_length = QSpinBox()
            sp_length.setRange(0, 1_000_000)
            widgets["length"] = sp_length
            form.addRow("Length", sp_length)
        if include_encoding:
            ed_encoding = QLineEdit()
            widgets["encoding"] = ed_encoding
            form.addRow("Encoding", ed_encoding)
        if include_bit_pos:
            sp_bit_pos = QSpinBox()
            sp_bit_pos.setRange(0, 63)
            widgets["bit_pos"] = sp_bit_pos
            form.addRow("Bit position", sp_bit_pos)
        if include_multiplier:
            sp_multiplier = QDoubleSpinBox()
            sp_multiplier.setDecimals(8)
            sp_multiplier.setRange(-1_000_000.0, 1_000_000.0)
            sp_multiplier.setSingleStep(0.01)
            widgets["multiplier"] = sp_multiplier
            form.addRow("Multiplier", sp_multiplier)
        return widgets

    def _make_memory_deck_group(self, title: str) -> dict:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        time_spec = self._make_memory_value_group("Time", include_multiplier=True)
        sample_index_spec = self._make_memory_value_group("Current Sample Index")
        path_spec = self._make_memory_value_group("Path", include_length=True, include_encoding=True)
        active_spec = self._make_memory_value_group("Active", include_bit_pos=True)
        loaded_spec = self._make_memory_value_group("Loaded", include_bit_pos=True)
        layout.addWidget(time_spec["group"])
        layout.addWidget(sample_index_spec["group"])
        layout.addWidget(path_spec["group"])
        layout.addWidget(active_spec["group"])
        layout.addWidget(loaded_spec["group"])
        return {
            "group": group,
            "time": time_spec,
            "sample_index": sample_index_spec,
            "path": path_spec,
            "active": active_spec,
            "loaded": loaded_spec,
        }

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
                analysis_samp_rate=int(self.sp_analysis_samp_rate.value()),
                chroma_method=("cqt" if self.cmb_chroma_method.currentIndex() == 0 else "cens"),
                chroma_hop_length=int(self.sp_chroma_hop_length.value()),
                chroma_cqt_bins_per_octave=int(self.sp_cqt_bins_per_oct.value()),
                chroma_cqt_octaves=int(self.sp_cqt_octaves.value()),
                use_hpss=bool(self.cb_use_hpss.isChecked()),
                bpm_hop_length=int(self.sp_bpm_hop.value()),
                bpm_win_length=int(self.sp_bpm_win.value()),
                bpm_min=int(self.sp_bpm_min.value()),
                bpm_max=int(self.sp_bpm_max.value()),
                bpm_dynamic=bool(self.cb_bpm_dynamic.isChecked()),
                bpm_adaptive_window=bool(self.cb_bpm_adaptive_win.isChecked()),
                beatgrid_offset_msec=float(self.sp_beatgrid_offset.value()),
                env_frame_ms=int(self.sp_env_frame_ms.value()),
                env_lo=(float(self.sp_env_lo_lo.value()), float(self.sp_env_lo_hi.value())),
                env_mid=(float(self.sp_env_mid_lo.value()), float(self.sp_env_mid_hi.value())),
                env_hi=(float(self.sp_env_hi_lo.value()), float(self.sp_env_hi_hi.value())),
                env_order=int(self.sp_env_order.value()),
            ),
            keyconfig=keyconfig(
                min_offset=float(self.sp_min_offset.value()),
                pitch_self=float(self.sp_pitch_self.value()),
                pitch_semitone=float(self.sp_pitch_semitone.value()),
                pitch_fifth=float(self.sp_pitch_fifth.value()),
                pitch_others=float(self.sp_pitch_others.value()),
            ),
            libconfig=libconfig(
                libpath=self.ed_libpath.text().strip(),
                write_log=bool(self.cb_write_log.isChecked()),
                logpath=self.ed_logpath.text().strip(),
            ),
            viewconfig=viewconfig(
                display_waveform=bool(self.cb_waveform.isChecked()),
                display_beatgrid=bool(self.cb_beatgrid.isChecked()),
                display_keystrip=bool(self.cb_keystrip.isChecked()),
                display_JumpCUE=bool(self.cb_JumpCUE.isChecked()),
                fps=int(self.sp_fps.value()),
                enable_metronome=bool(self.cb_metronome.isChecked()),
                record_img_path=self.ed_record_img_path.text().strip(),
                metronome_wav_path=self.ed_metronome_wav_path.text().strip(),
            ),
            externalsyncconfig=externalsyncconfig(
                enabled=bool(self.cb_external_sync_enabled.isChecked()),
                mode=("time" if self.cmb_external_sync_mode.currentIndex() == 0 else "sample_index"),
                memory_process_name=self._memory_process_name(),
                memory_process_pid=self._memory_process_pid(),
                memory_deck1=self._get_memory_deck(self.deck1_specs),
                memory_deck2=self._get_memory_deck(self.deck2_specs),
            ),
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

    def _on_ui_draw_interval(self, dt_ms: float):
        if 1.0 <= dt_ms <= 1000.0:
            self._refresh_samples.append(float(dt_ms))
            avg_ms = sum(self._refresh_samples) / len(self._refresh_samples)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            self.lbl_fps.setText(f"{avg_ms:.1f} ms ({fps:.1f} FPS)")

    def _sync_external_sync_mode_ui(self):
        is_time_sync = self.cmb_external_sync_mode.currentIndex() == 0
        self.memory_scroll.setVisible(True)
        for deck_widgets in (self.deck1_specs, self.deck2_specs):
            deck_widgets["time"]["group"].setVisible(is_time_sync)
            deck_widgets["sample_index"]["group"].setVisible(not is_time_sync)

    def _refresh_memory_processes(self):
        current_name = self._memory_process_name()
        current_pid = self._memory_process_pid()
        self.cmb_memory_process.blockSignals(True)
        self.cmb_memory_process.clear()
        processes = self._list_running_processes()
        processes.sort(
            key=lambda proc: (
                1 if self._is_denied_process_name(str(proc.get("name") or "")) else 0,
                str(proc.get("name") or "").lower(),
                int(proc.get("pid") or 0),
            )
        )
        model = self.cmb_memory_process.model()
        for proc in processes:
            denied = self._is_denied_process_name(str(proc.get("name") or ""))
            label = f'{proc["name"]} (PID {proc["pid"]})'
            if denied:
                label = f"[Blocked] {label}"
            self.cmb_memory_process.addItem(label, proc)
            row = self.cmb_memory_process.count() - 1
            item = model.item(row) if hasattr(model, "item") else None
            if item is not None and denied:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
                item.setForeground(QtGui.QColor("#888888"))
        self.cmb_memory_process.blockSignals(False)
        self._set_selected_memory_process(current_name, current_pid)

    def _set_selected_memory_process(self, process_name: str, process_pid: int):
        target_name = (process_name or "").strip().lower()
        if self._is_denied_process_name(target_name):
            self.cmb_memory_process.setCurrentIndex(-1)
            self.cmb_memory_process.setEditText("")
            self.lbl_memory_process_blocked.setText(
                f"Blocked by denylist: {process_name.strip()}"
            )
            return
        target_pid = int(process_pid or 0)
        for idx in range(self.cmb_memory_process.count()):
            data = self.cmb_memory_process.itemData(idx)
            if not isinstance(data, dict):
                continue
            item_name = str(data.get("name") or "").strip().lower()
            item_pid = int(data.get("pid") or 0)
            if target_pid and item_pid == target_pid:
                self.cmb_memory_process.setCurrentIndex(idx)
                self.lbl_memory_process_blocked.setText("")
                return
            if target_name and item_name == target_name:
                self.cmb_memory_process.setCurrentIndex(idx)
                self.lbl_memory_process_blocked.setText("")
                return
        text = process_name.strip() if process_name else ""
        if target_pid:
            text = f"{text} (PID {target_pid})".strip()
        if self._is_denied_process_name(text):
            self.cmb_memory_process.setCurrentIndex(-1)
            self.cmb_memory_process.setEditText("")
            self.lbl_memory_process_blocked.setText(
                f"Blocked by denylist: {process_name.strip()}"
            )
            return
        self.cmb_memory_process.setEditText(text)
        self.lbl_memory_process_blocked.setText("")

    def _memory_process_name(self) -> str:
        data = self.cmb_memory_process.currentData()
        if isinstance(data, dict):
            name = str(data.get("name") or "").strip()
            return "" if self._is_denied_process_name(name) else name
        text = self.cmb_memory_process.currentText().strip()
        name = text.split(" (PID ", 1)[0].strip()
        if self._is_denied_process_name(name):
            self.lbl_memory_process_blocked.setText(f"Blocked by denylist: {name}")
            return ""
        return name

    def _memory_process_pid(self) -> int:
        data = self.cmb_memory_process.currentData()
        if isinstance(data, dict):
            try:
                return int(data.get("pid") or 0)
            except Exception:
                return 0
        text = self.cmb_memory_process.currentText().strip()
        if " (PID " in text and text.endswith(")"):
            try:
                return int(text.rsplit(" (PID ", 1)[1][:-1])
            except Exception:
                return 0
        return 0

    def _list_running_processes(self) -> list[dict]:
        try:
            out = subprocess.check_output(
                ["tasklist", "/FO", "CSV", "/NH"],
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            return []
        rows = []
        reader = csv.reader(io.StringIO(out))
        for row in reader:
            if len(row) < 2:
                continue
            name = str(row[0]).strip()
            pid_text = str(row[1]).strip()
            try:
                pid = int(pid_text)
            except Exception:
                continue
            if not name:
                continue
            rows.append({"name": name, "pid": pid})
        rows.sort(key=lambda item: (item["name"].lower(), item["pid"]))
        return rows

    def _is_denied_process_name(self, name: str) -> bool:
        target = self._normalize_process_name(name)
        if not target:
            return False
        deny = self._load_process_denylist()
        return target in deny

    def _load_process_denylist(self) -> set[str]:
        cache = getattr(self, "_process_denylist_cache", None)
        if isinstance(cache, set):
            return cache
        denylist_path = Path("process_denylist.json")
        names: set[str] = set()
        try:
            payload = json.loads(denylist_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for item in payload.get("blocked_process_names", []):
            text = self._normalize_process_name(item)
            if text:
                names.add(text)
        self._process_denylist_cache = names
        return names

    def _normalize_process_name(self, name: str) -> str:
        text = str(name or "").strip().lower()
        if text.endswith(".exe"):
            text = text[:-4]
        return text


