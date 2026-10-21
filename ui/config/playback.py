from collections import deque
import math
import time

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from audioengine.native.client import NativeFeeder
from audioengine.python.feeder import PCMFeeder
from ui.track_info_panel import HorizontalBufferMeter
from ui.config.scroll import scroll_page
from core.i18n import tr


def build_playback_tab(dialog) -> None:
    dialog.tab_playback = QWidget()
    form = QFormLayout(dialog.tab_playback)

    dialog.cb_metronome = QCheckBox(tr("Enable metronome"))
    dialog.ed_metronome_wav_path = QLineEdit()
    dialog.sp_metronome_offset_msec = QDoubleSpinBox()
    dialog.sp_metronome_offset_msec.setRange(-500.0, 500.0)
    dialog.sp_metronome_offset_msec.setDecimals(1)
    dialog.sp_metronome_offset_msec.setSingleStep(1.0)
    dialog.sp_metronome_offset_msec.setToolTip(
        tr("Shift metronome clicks relative to the beat (positive = earlier)")
    )
    dialog.sp_volume_trim_dbfs = QDoubleSpinBox()
    dialog.sp_volume_trim_dbfs.setRange(-60.0, 0.0)
    dialog.sp_volume_trim_dbfs.setDecimals(2)
    dialog.sp_volume_trim_dbfs.setSingleStep(0.5)
    dialog.sp_default_volume_percent = QSpinBox()
    dialog.sp_default_volume_percent.setRange(0, 100)
    dialog.sp_default_volume_percent.setSingleStep(1)
    dialog.cb_use_timestretch = QCheckBox(tr("Use timestretch"))
    dialog.cb_use_timestretch.setToolTip(
        tr("Tempo changes keep the pitch (time stretch) instead of varispeed")
    )

    dialog.sp_click_down_volume = QSpinBox()
    dialog.sp_click_down_volume.setRange(0, 100)
    dialog.sp_click_down_pitch = QDoubleSpinBox()
    dialog.sp_click_down_pitch.setRange(-24.0, 24.0)
    dialog.sp_click_down_pitch.setSingleStep(1.0)
    dialog.sp_click_down_pitch.setDecimals(1)
    dialog.sp_click_beat_volume = QSpinBox()
    dialog.sp_click_beat_volume.setRange(0, 100)
    dialog.sp_click_beat_pitch = QDoubleSpinBox()
    dialog.sp_click_beat_pitch.setRange(-24.0, 24.0)
    dialog.sp_click_beat_pitch.setSingleStep(1.0)
    dialog.sp_click_beat_pitch.setDecimals(1)
    dialog.cb_metronome_ducking = QCheckBox(
        tr("Duck music under metronome clicks")
    )
    dialog.cb_metronome_ducking.setToolTip(
        tr("Lowers the music by up to 6 dB, following each click")
    )
    dialog.cb_soft_clip = QCheckBox(tr("Soft clip output (tanh)"))
    dialog.cb_soft_clip.setToolTip(
        tr("Output above -1.9 dBFS is rounded off by tanh up to 0 dBFS "
           "instead of hard clipping")
    )

    # the audio engine (applied at the next start), the output device, each engine's buffer (ms): the buffer of
    # the engine playing is shown with the rest, the other one's under Advanced (set_audio_engine)
    dialog.cb_audio_engine = QComboBox()
    dialog.cb_audio_engine.addItem(tr("Native first"), "native")
    dialog.cb_audio_engine.addItem(tr("Python"), "python")
    dialog.cb_output_device = QComboBox()
    dialog.cb_output_device.setToolTip(tr(
        "<p>The device the player plays on, as its audio engine names it. Changed while the app runs, the sound "
        "keeps the format (sample rate, channels) of the device it started with, converted by the system.</p>"
    ))
    dialog.sp_native_buffer_ms = QSpinBox()
    dialog.sp_native_buffer_ms.setRange(*NativeFeeder.BUFFER_MS_RANGE)
    dialog.sp_native_buffer_ms.setToolTip(tr(
        "<p>The native engine's audio queued ahead of the output device. Smaller answers sooner (play, pause, seek, "
        "CUE calls, volume, tempo); too small drops out when the computer is busy. The device's own latency comes "
        "after it (shown under the meter).</p>"
    ))
    dialog.sp_python_buffer_ms = QSpinBox()
    dialog.sp_python_buffer_ms.setRange(*PCMFeeder.BUFFER_MS_RANGE)
    dialog.sp_python_buffer_ms.setToolTip(tr(
        "<p>The Python engine's audio queued in the output device ahead of what is heard. Smaller answers sooner "
        "(play, pause, seek, CUE calls, volume, tempo); too small drops out when the computer is busy (with the time "
        "stretch, below about 80 ms).</p>"
    ))
    for spin in (dialog.sp_native_buffer_ms, dialog.sp_python_buffer_ms):
        spin.setSingleStep(5)
        spin.setSuffix(" ms")
    # where each buffer setting goes (set_audio_engine): with the rest, or under Advanced
    dialog.buffer_slot = QWidget()
    QHBoxLayout(dialog.buffer_slot).setContentsMargins(0, 0, 0, 0)
    dialog.other_buffer_slot = QWidget()
    QHBoxLayout(dialog.other_buffer_slot).setContentsMargins(0, 0, 0, 0)
    dialog.lbl_other_buffer = QLabel()

    dialog.cb_metronome.setToolTip(tr("Play a click on every beat of the beat grid."))
    dialog.ed_metronome_wav_path.setToolTip(tr("WAV file used as the metronome click."))
    dialog.sp_volume_trim_dbfs.setToolTip(
        tr("Fixed gain applied to the output before the volume slider (headroom).")
    )
    dialog.sp_default_volume_percent.setToolTip(tr("Volume slider position at startup."))
    dialog.sp_click_down_volume.setToolTip(tr("Click volume on the first beat of each bar."))
    dialog.sp_click_down_pitch.setToolTip(
        tr("Pitch shift of the click on the first beat of each bar.")
    )
    dialog.sp_click_beat_volume.setToolTip(tr("Click volume on the other beats."))
    dialog.sp_click_beat_pitch.setToolTip(tr("Pitch shift of the click on the other beats."))

    dialog.buffer_meter = HorizontalBufferMeter(hold_marker=False)
    dialog.buffer_low_meter = HorizontalBufferMeter(hold_marker=False)
    dialog.buffer_meter.setToolTip(
        tr("Average audio queued in the output buffer (smoothed).")
    )
    dialog.buffer_low_meter.setToolTip(
        tr("Buffer level the output stayed above 99% of the time in the recent window.")
    )
    dialog.lbl_buffer = QLabel(tr("Waiting for playback..."))
    dialog.lbl_buffer.setToolTip(
        tr("Audio queued in the output device buffer while playing, 0 ms = dropout.")
    )
    # two short lines that wrap further if need be: the readout never widens the page
    dialog.lbl_buffer.setWordWrap(True)
    dialog._buffer_ema_ms = None
    dialog._buffer_window = deque()
    dialog._buffer_dropout_sec = 0.0

    grp_general = QGroupBox(tr("General"))
    f_general = QFormLayout(grp_general)
    f_general.addRow(tr("Audio engine"), dialog.cb_audio_engine)
    f_general.addRow(tr("Output device"), dialog.cb_output_device)
    f_general.addRow(tr("Trim"), dialog.sp_volume_trim_dbfs)
    f_general.addRow(tr("Default volume (%)"), dialog.sp_default_volume_percent)
    f_general.addRow(dialog.cb_soft_clip)
    f_general.addRow(tr("Buffer size"), dialog.buffer_slot)
    f_general.addRow(tr("Output buffer"), dialog.buffer_meter)
    f_general.addRow(
        tr("1% low ({seconds} s)").format(seconds=f"{dialog.BUFFER_WINDOW_SEC:.0f}"), dialog.buffer_low_meter
    )
    f_general.addRow("", dialog.lbl_buffer)
    dialog.btn_playback_advanced = QToolButton()
    dialog.btn_playback_advanced.setText(tr("Advanced"))
    dialog.btn_playback_advanced.setCheckable(True)
    dialog.btn_playback_advanced.setArrowType(Qt.RightArrow)
    dialog.btn_playback_advanced.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    dialog.btn_playback_advanced.setAutoRaise(True)
    dialog.btn_playback_advanced.setToolTip(tr("Show or hide the settings of the audio engine not playing."))
    f_general.addRow(dialog.btn_playback_advanced)
    dialog.playback_advanced = QWidget()
    f_advanced = QFormLayout(dialog.playback_advanced)
    f_advanced.setContentsMargins(16, 0, 0, 0)
    f_advanced.addRow(dialog.lbl_other_buffer, dialog.other_buffer_slot)
    dialog.playback_advanced.setVisible(False)
    f_general.addRow(dialog.playback_advanced)
    dialog.btn_playback_advanced.toggled.connect(dialog._toggle_playback_advanced)
    dialog.set_audio_engine(native=False, devices=None)

    grp_timestretch = QGroupBox(tr("Time Stretch"))
    f_timestretch = QFormLayout(grp_timestretch)
    f_timestretch.addRow(dialog.cb_use_timestretch)

    grp_metronome = QGroupBox(tr("Metronome"))
    metronome_layout = QVBoxLayout(grp_metronome)
    metronome_layout.addWidget(dialog.cb_metronome)
    dialog.btn_metronome_details = QToolButton()
    dialog.btn_metronome_details.setText(tr("Details"))
    dialog.btn_metronome_details.setCheckable(True)
    dialog.btn_metronome_details.setArrowType(Qt.RightArrow)
    dialog.btn_metronome_details.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    dialog.btn_metronome_details.setAutoRaise(True)
    dialog.btn_metronome_details.setToolTip(tr("Show or hide the metronome options."))
    metronome_layout.addWidget(dialog.btn_metronome_details)

    dialog.metronome_details = QWidget()
    f_metronome_details = QFormLayout(dialog.metronome_details)
    f_metronome_details.setContentsMargins(16, 0, 0, 0)
    f_metronome_details.addRow(tr("WAV path"), dialog.ed_metronome_wav_path)
    f_metronome_details.addRow(tr("Offset (ms)"), dialog.sp_metronome_offset_msec)
    f_metronome_details.addRow(
        tr("Downbeat volume (%)"), dialog.sp_click_down_volume
    )
    f_metronome_details.addRow(
        tr("Downbeat pitch (semitones)"), dialog.sp_click_down_pitch
    )
    f_metronome_details.addRow(tr("Beat volume (%)"), dialog.sp_click_beat_volume)
    f_metronome_details.addRow(
        tr("Beat pitch (semitones)"), dialog.sp_click_beat_pitch
    )
    f_metronome_details.addRow(dialog.cb_metronome_ducking)
    dialog.metronome_details.setVisible(False)
    metronome_layout.addWidget(dialog.metronome_details)
    dialog.btn_metronome_details.toggled.connect(
        dialog._toggle_metronome_details
    )

    form.addRow(grp_general)
    form.addRow(grp_timestretch)
    form.addRow(grp_metronome)
    dialog.tabs.addTab(scroll_page(dialog.tab_playback), tr("Playback"))


class PlaybackSettingsMixin:
    BUFFER_WINDOW_SEC = 10.0
    BUFFER_EMA_TAU_SEC = 0.3

    def _initialize_playback_settings(self, bus) -> None:
        if bus is not None:
            bus.sig_output_buffer.connect(self._on_output_buffer)

    def set_audio_engine(self, native: bool, devices) -> None:
        """The engine that plays (``native``: its buffer is shown with the rest, the other one's under Advanced)
        and ``devices()`` -> (the output device in use, the output devices) of it (None: not known yet)."""
        self._native_playing = bool(native)
        self._output_devices_provider = devices
        shown, other = ((self.sp_native_buffer_ms, self.sp_python_buffer_ms) if native
                        else (self.sp_python_buffer_ms, self.sp_native_buffer_ms))
        for slot in (self.buffer_slot, self.other_buffer_slot):
            for spin in (self.sp_native_buffer_ms, self.sp_python_buffer_ms):
                slot.layout().removeWidget(spin)
        self.buffer_slot.layout().addWidget(shown)
        self.other_buffer_slot.layout().addWidget(other)
        self.lbl_other_buffer.setText(tr("Buffer size, Python engine") if native
                                      else tr("Buffer size, native engine"))
        playing = tr("Native") if native else tr("Python")
        self.cb_audio_engine.setToolTip(tr(
            "<p>Native: the audio engine in a process of its own (no dropouts when the computer is busy, smaller "
            "buffers); the Python engine where it is not built or cannot start. Python: the Python engine only. "
            "Applied at the next start.</p><p>Playing now: {engine}.</p>"
        ).format(engine=playing))

    def _toggle_playback_advanced(self, shown: bool) -> None:
        self.btn_playback_advanced.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)
        self.playback_advanced.setVisible(shown)
        self.tab_playback.updateGeometry()

    def _load_output_devices(self, wanted: str) -> None:
        """The output devices of the engine playing, ``wanted`` ("": the system's default) picked; one not
        connected stays listed as such."""
        combo = self.cb_output_device
        combo.clear()
        combo.addItem(tr("System default"), "")
        provider = getattr(self, "_output_devices_provider", None)
        names = provider()[1] if provider is not None else []
        for name in names:
            combo.addItem(name, name)
        if wanted and wanted not in names:
            combo.addItem(tr("{name} (not connected)").format(name=wanted), wanted)
        combo.setCurrentIndex(max(0, combo.findData(wanted)))

    def _reset_playback_monitor(self) -> None:
        self._buffer_ema_ms = None
        self._buffer_window.clear()
        self._buffer_dropout_sec = 0.0

    def _toggle_metronome_details(self, shown: bool) -> None:
        self.btn_metronome_details.setArrowType(
            Qt.DownArrow if shown else Qt.RightArrow
        )
        self.metronome_details.setVisible(shown)
        self.tab_playback.updateGeometry()

    def _on_output_buffer(self, queued_ms: float, stats: object, capacity_ms: float) -> None:
        if not self.isVisible() or capacity_ms <= 0.0 or not isinstance(stats, dict):
            return
        seconds = float(stats["seconds"])
        hist = np.asarray(stats["hist"], dtype=np.float64)
        now = time.monotonic()
        # EMA over time of the time-weighted mean level.
        mean_ms = float(stats["mean_ms"])
        if self._buffer_ema_ms is None:
            self._buffer_ema_ms = mean_ms
        else:
            alpha = 1.0 - math.exp(-seconds / self.BUFFER_EMA_TAU_SEC)
            self._buffer_ema_ms += alpha * (mean_ms - self._buffer_ema_ms)
        # the time the device played silence: told by the native engine (its buffer's lowest level is touched
        # without a gap: the device reads it in whole periods), else the time below 1 ms
        self._buffer_dropout_sec += float(stats.get("dropout_sec", hist[0]))
        self._buffer_window.append((now, hist))
        while self._buffer_window and now - self._buffer_window[0][0] > self.BUFFER_WINDOW_SEC:
            self._buffer_window.popleft()

        size = max(len(h) for _, h in self._buffer_window)
        total = np.zeros(size, dtype=np.float64)
        for _, h in self._buffer_window:
            total[:len(h)] += h
        cumulative = np.cumsum(total)
        low_1pct_ms = float(np.searchsorted(cumulative, 0.01 * cumulative[-1]))
        lowest_ms = float(np.flatnonzero(total)[0]) if cumulative[-1] > 0 else 0.0

        self.buffer_meter.set_level(self._buffer_ema_ms / capacity_ms)
        self.buffer_low_meter.set_level(low_1pct_ms / capacity_ms)
        values = dict(ema=f"{self._buffer_ema_ms:.0f}", capacity=f"{capacity_ms:.0f}", low=f"{low_1pct_ms:.0f}",
                      lowest=f"{lowest_ms:.0f}", seconds=f"{self.BUFFER_WINDOW_SEC:.0f}",
                      dropout=f"{self._buffer_dropout_sec * 1000.0:.0f}")
        if "device_ms" in stats:
            self.lbl_buffer.setText(tr(
                "EMA {ema} / {capacity} ms   1% low {low} ms   lowest {lowest} ms ({seconds} s)\n"
                "Dropout since opened {dropout} ms   Device latency {device} ms (fixed)"
            ).format(device=f"{float(stats['device_ms']):.0f}", **values))
        else:
            self.lbl_buffer.setText(tr(
                "EMA {ema} / {capacity} ms   1% low {low} ms   lowest {lowest} ms ({seconds} s)\n"
                "Dropout since opened {dropout} ms"
            ).format(**values))
