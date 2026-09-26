from collections import deque
import math
import time

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.track_info_panel import HorizontalBufferMeter


def build_playback_tab(dialog) -> None:
    dialog.tab_playback = QWidget()
    form = QFormLayout(dialog.tab_playback)

    dialog.cb_metronome = QCheckBox("Enable metronome")
    dialog.ed_metronome_wav_path = QLineEdit()
    dialog.sp_metronome_offset_msec = QDoubleSpinBox()
    dialog.sp_metronome_offset_msec.setRange(-500.0, 500.0)
    dialog.sp_metronome_offset_msec.setDecimals(1)
    dialog.sp_metronome_offset_msec.setSingleStep(1.0)
    dialog.sp_metronome_offset_msec.setToolTip(
        "Shift metronome clicks relative to the beat (positive = earlier)"
    )
    dialog.sp_volume_trim_dbfs = QDoubleSpinBox()
    dialog.sp_volume_trim_dbfs.setRange(-60.0, 0.0)
    dialog.sp_volume_trim_dbfs.setDecimals(2)
    dialog.sp_volume_trim_dbfs.setSingleStep(0.5)
    dialog.sp_default_volume_percent = QSpinBox()
    dialog.sp_default_volume_percent.setRange(0, 100)
    dialog.sp_default_volume_percent.setSingleStep(1)
    dialog.cb_use_timestretch = QCheckBox("Use timestretch")
    dialog.cb_use_timestretch.setToolTip(
        "Tempo changes keep the pitch (time stretch) instead of varispeed"
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
        "Duck music under metronome clicks"
    )
    dialog.cb_metronome_ducking.setToolTip(
        "Lowers the music by up to 6 dB, following each click"
    )
    dialog.cb_soft_clip = QCheckBox("Soft clip output (tanh)")
    dialog.cb_soft_clip.setToolTip(
        "Output above -1.9 dBFS is rounded off by tanh up to 0 dBFS "
        "instead of hard clipping"
    )

    dialog.cb_metronome.setToolTip("Play a click on every beat of the beat grid.")
    dialog.ed_metronome_wav_path.setToolTip("WAV file used as the metronome click.")
    dialog.sp_volume_trim_dbfs.setToolTip(
        "Fixed gain applied to the output before the volume slider (headroom)."
    )
    dialog.sp_default_volume_percent.setToolTip("Volume slider position at startup.")
    dialog.sp_click_down_volume.setToolTip("Click volume on the first beat of each bar.")
    dialog.sp_click_down_pitch.setToolTip(
        "Pitch shift of the click on the first beat of each bar."
    )
    dialog.sp_click_beat_volume.setToolTip("Click volume on the other beats.")
    dialog.sp_click_beat_pitch.setToolTip("Pitch shift of the click on the other beats.")

    dialog.buffer_meter = HorizontalBufferMeter(hold_marker=False)
    dialog.buffer_low_meter = HorizontalBufferMeter(hold_marker=False)
    dialog.buffer_meter.setToolTip(
        "Average audio queued in the output buffer (smoothed)."
    )
    dialog.buffer_low_meter.setToolTip(
        "Buffer level the output stayed above 99% of the time in the recent window."
    )
    dialog.lbl_buffer = QLabel("Waiting for playback...")
    dialog.lbl_buffer.setToolTip(
        "Audio queued in the output device buffer while playing, 0 ms = dropout."
    )
    dialog._buffer_ema_ms = None
    dialog._buffer_window = deque()
    dialog._buffer_dropout_sec = 0.0

    grp_general = QGroupBox("General")
    f_general = QFormLayout(grp_general)
    f_general.addRow("Trim", dialog.sp_volume_trim_dbfs)
    f_general.addRow("Default volume (%)", dialog.sp_default_volume_percent)
    f_general.addRow(dialog.cb_soft_clip)
    f_general.addRow("Output buffer", dialog.buffer_meter)
    f_general.addRow(
        f"1% low ({dialog.BUFFER_WINDOW_SEC:.0f} s)", dialog.buffer_low_meter
    )
    f_general.addRow("", dialog.lbl_buffer)

    grp_timestretch = QGroupBox("Time Stretch")
    f_timestretch = QFormLayout(grp_timestretch)
    f_timestretch.addRow(dialog.cb_use_timestretch)

    grp_metronome = QGroupBox("Metronome")
    metronome_layout = QVBoxLayout(grp_metronome)
    metronome_layout.addWidget(dialog.cb_metronome)
    dialog.btn_metronome_details = QToolButton()
    dialog.btn_metronome_details.setText("Details")
    dialog.btn_metronome_details.setCheckable(True)
    dialog.btn_metronome_details.setArrowType(Qt.RightArrow)
    dialog.btn_metronome_details.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    dialog.btn_metronome_details.setAutoRaise(True)
    dialog.btn_metronome_details.setToolTip("Show or hide the metronome options.")
    metronome_layout.addWidget(dialog.btn_metronome_details)

    dialog.metronome_details = QWidget()
    f_metronome_details = QFormLayout(dialog.metronome_details)
    f_metronome_details.setContentsMargins(16, 0, 0, 0)
    f_metronome_details.addRow("WAV path", dialog.ed_metronome_wav_path)
    f_metronome_details.addRow("Offset (ms)", dialog.sp_metronome_offset_msec)
    f_metronome_details.addRow(
        "Downbeat volume (%)", dialog.sp_click_down_volume
    )
    f_metronome_details.addRow(
        "Downbeat pitch (semitones)", dialog.sp_click_down_pitch
    )
    f_metronome_details.addRow("Beat volume (%)", dialog.sp_click_beat_volume)
    f_metronome_details.addRow(
        "Beat pitch (semitones)", dialog.sp_click_beat_pitch
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
    dialog.tabs.addTab(dialog.tab_playback, "Playback")


class PlaybackSettingsMixin:
    BUFFER_WINDOW_SEC = 10.0
    BUFFER_EMA_TAU_SEC = 0.3

    def _initialize_playback_settings(self, bus) -> None:
        if bus is not None:
            bus.sig_output_buffer.connect(self._on_output_buffer)

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
        self._buffer_dropout_sec += float(hist[0])  # time below 1 ms
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
        self.lbl_buffer.setText(
            f"EMA {self._buffer_ema_ms:.0f} / {capacity_ms:.0f} ms   "
            f"1% low {low_1pct_ms:.0f} ms   lowest {lowest_ms:.0f} ms "
            f"({self.BUFFER_WINDOW_SEC:.0f} s)   "
            f"dropout since opened {self._buffer_dropout_sec * 1000.0:.0f} ms"
        )
