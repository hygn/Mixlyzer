# settings_ui_autosave.py
from typing import Tuple
from dataclasses import is_dataclass, asdict
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QLineEdit, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QDialogButtonBox, QGroupBox)
from core.config import config, libconfig, viewconfig, analysisconfig, keyconfig
from core.event_bus import EventBus

class SettingsDialog(QDialog):

    def __init__(self, bus:EventBus=None):
        super().__init__()
        self.saveJsonRequested = bus.sig_setting_saveJsonRequested
        self.setWindowTitle("Settings")
        self.setModal(False)

        self.tabs = QTabWidget(self)
        self._make_tab_library()
        self._make_tab_view()
        self._make_tab_analysis()
        self._make_tab_key()

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

    # Tabs
    def _make_tab_library(self):
        self.tab_lib = QWidget()
        f = QFormLayout(self.tab_lib)
        self.ed_libpath = QLineEdit()
        f.addRow("Library Path", self.ed_libpath)
        self.tabs.addTab(self.tab_lib, "Library")

    def _make_tab_view(self):
        self.tab_view = QWidget()
        f = QFormLayout(self.tab_view)

        self.cb_waveform = QCheckBox("Display waveform")
        self.cb_beatgrid = QCheckBox("Display beatgrid")
        self.cb_keystrip = QCheckBox("Display keystrip")
        self.cb_JumpCUE = QCheckBox("Display JumpCUE")
        self.cb_metronome = QCheckBox("Enable metronome")

        self.ed_record_img_path = QLineEdit()
        self.ed_metronome_wav_path = QLineEdit()

        f.addRow(self.cb_waveform)
        f.addRow(self.cb_beatgrid)
        f.addRow(self.cb_keystrip)
        f.addRow(self.cb_JumpCUE)
        f.addRow(self.cb_metronome)
        f.addRow("Record image path", self.ed_record_img_path)
        f.addRow("Metronome WAV path", self.ed_metronome_wav_path)

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
        f_beat.addRow(self.cb_bpm_dynamic)
        f_beat.addRow(self.cb_bpm_adaptive_win)
        f_beat.addRow("BPM hop length (samp)", self.sp_bpm_hop)
        f_beat.addRow("BPM Autocorrelation win_length (ms)", self.sp_bpm_win)
        f_beat.addRow("BPM min", self.sp_bpm_min)
        f_beat.addRow("BPM max", self.sp_bpm_max)

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

    # Public API
    def set_config(self, cfg: config):
        assert is_dataclass(cfg), "cfg must be a dataclass 'config'"
        self._current_cfg = cfg

        # lib
        self.ed_libpath.setText(cfg.libconfig.libpath)

        # view
        v = cfg.viewconfig
        self.cb_waveform.setChecked(bool(v.display_waveform))
        self.cb_beatgrid.setChecked(bool(v.display_beatgrid))
        self.cb_keystrip.setChecked(bool(v.display_keystrip))
        self.cb_JumpCUE.setChecked(bool(v.display_JumpCUE))
        self.cb_metronome.setChecked(bool(v.enable_metronome))
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
        self.cb_bpm_dynamic.setChecked(int(a.bpm_dynamic))
        self.cb_bpm_adaptive_win.setChecked(int(a.bpm_adaptive_window))
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

    def _set_band(self, sp_lo: QDoubleSpinBox, sp_hi: QDoubleSpinBox, band: Tuple[float, float]):
        sp_lo.setValue(float(band[0])); sp_hi.setValue(float(band[1]))

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
            ),
            viewconfig=viewconfig(
                display_waveform=bool(self.cb_waveform.isChecked()),
                display_beatgrid=bool(self.cb_beatgrid.isChecked()),
                display_keystrip=bool(self.cb_keystrip.isChecked()),
                display_JumpCUE=bool(self.cb_JumpCUE.isChecked()),
                enable_metronome=bool(self.cb_metronome.isChecked()),
                record_img_path=self.ed_record_img_path.text().strip(),
                metronome_wav_path=self.ed_metronome_wav_path.text().strip(),
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


