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
from ui.buffer_histogram import BufferHistogram
from ui.track_info_panel import HorizontalBufferMeter
from ui.config.scroll import scroll_page
from core import os_support
from core.i18n import tr, tr_ctx, tr_mark

# the native engine's threads as it got them (client.THREAD_STATES), shown under the meter
THREAD_STATE_NAMES = {
    "api": tr_mark("the audio API's own"),
    "normal": tr_mark("normal"),
    "raised": tr_mark("raised"),
    "raised_rtkit": tr_mark("raised through rtkit"),
    "realtime": tr_mark("real-time"),
    "realtime_rtkit": tr_mark("real-time through rtkit"),
    "refused": tr_mark("not permitted to raise"),
}


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
    # the native engine's way to the device: the audio APIs this system has, WASAPI's mode and exclusive rate
    dialog.cb_native_api = QComboBox()
    if os_support.IS_WINDOWS:
        dialog.cb_native_api.addItem("WASAPI", "wasapi")
        dialog.cb_native_api.addItem("WASAPI (cpal)", "cpal")
    elif os_support.IS_LINUX:
        dialog.cb_native_api.addItem("ALSA", "cpal")
        dialog.cb_native_api.addItem("JACK", "jack")
    else:
        dialog.cb_native_api.addItem("CoreAudio", "cpal")
    dialog.cb_native_api.setToolTip(tr(
        "<p>The audio API the native engine plays through. WASAPI: its own, with a low-latency shared stream "
        "or exclusive mode. WASAPI (cpal): the usual shared stream (more latency), for a device the other one "
        "cannot open. JACK (Linux): through the JACK or PipeWire server.</p>"
    ))
    dialog.cb_native_output_mode = QComboBox()
    dialog.cb_native_output_mode.addItem(tr("Shared"), False)
    dialog.cb_native_output_mode.addItem(tr("Exclusive"), True)
    dialog.cb_native_output_mode.setToolTip(tr(
        "<p>Shared: with other apps, at the smallest period the driver allows. Exclusive: the device to Mixlyzer "
        "alone, the shortest latency; other apps cannot play on it meanwhile. A mode the device cannot open falls "
        "back to the usual shared stream.</p>"
    ))
    dialog.cb_native_rate = QComboBox()
    dialog.cb_native_rate.addItem(tr("The device's"), 0)
    for rate in NativeFeeder.EXCLUSIVE_RATES:
        dialog.cb_native_rate.addItem(f"{rate / 1000:g} kHz", rate)
    dialog.cb_native_rate.setToolTip(tr(
        "<p>The sample rate the device runs at in exclusive mode (the tracks are decoded for it). Applied from the "
        "next track loaded; a rate the device does not take falls back to the usual shared stream.</p>"
    ))
    # the engine's own share of the computer: Windows its process's class, Linux / macOS its threads' (sched.rs)
    dialog.cb_native_priority = QComboBox()
    dialog.cb_native_priority.addItem(tr_ctx("process priority", "Normal"), "normal")
    dialog.cb_native_priority.addItem(tr_ctx("process priority", "Above normal"), "above_normal")
    dialog.cb_native_priority.addItem(tr_ctx("process priority", "High"), "high")
    if os_support.IS_WINDOWS:
        priority_tip = tr(
            "<p>The priority class of the native engine's process. High: its threads run ahead of the analysis and "
            "other apps, so it wakes on time when the computer is busy (fewer dropouts at small buffers). Applied at "
            "once.</p>")
    elif os_support.IS_LINUX:
        priority_tip = tr(
            "<p>The priority of the native engine's threads (its own and the ALSA output's; JACK's is the "
            "server's). High: real-time, else nice -10; Above normal: nice -5. What the user may not set (without "
            "the audio group's limits) is asked of rtkit. Applied at once; what was got is shown under the "
            "meter.</p>")
    else:
        priority_tip = tr(
            "<p>The priority of the native engine's thread. High: real-time (a time constraint, as CoreAudio's own "
            "output thread); Above normal: QoS user-interactive; Normal: QoS default. Applied at once.</p>")
    dialog.cb_native_priority.setToolTip(priority_tip)
    dialog.cb_native_power = QComboBox()
    dialog.cb_native_power.addItem(tr("Off"), False)
    dialog.cb_native_power.addItem(tr("The system's choice"), True)
    if os_support.IS_WINDOWS:
        power_tip = tr(
            "<p>Whether Windows may power-throttle the native engine's process (EcoQoS: efficiency cores, lower "
            "clocks, coarser timers) to save power. Off: it runs at full speed with the 1 ms timer, so it wakes on "
            "time. Applied at once.</p>")
    elif os_support.IS_LINUX:
        power_tip = tr(
            "<p>Whether Linux may save power on the native engine's threads. Off: a 1 ns timer slack, and uclamp "
            "keeping the CPU clock up while they run (Linux 5.3 and later, with the schedutil governor). Applied "
            "at once.</p>")
    else:
        power_tip = tr(
            "<p>Whether macOS may save power on the native engine (App Nap, timer coalescing). Off: a "
            "latency-critical activity keeps both off (the computer may still sleep when idle). Applied at "
            "once.</p>")
    dialog.cb_native_power.setToolTip(power_tip)
    for combo in (dialog.cb_native_api, dialog.cb_native_output_mode):
        combo.currentIndexChanged.connect(dialog._follow_native_api)
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
    # where each engine's settings go (set_audio_engine): with the rest, or under Advanced
    dialog.buffer_slot = QWidget()
    QHBoxLayout(dialog.buffer_slot).setContentsMargins(0, 0, 0, 0)
    dialog.other_buffer_slot = QWidget()
    QHBoxLayout(dialog.other_buffer_slot).setContentsMargins(0, 0, 0, 0)
    dialog.lbl_other_buffer = QLabel()
    # the native engine's own rows: shown with the rest when it plays, else under Advanced
    dialog._native_rows = []
    for text, widget in ((tr("Audio API"), dialog.cb_native_api), (tr("Output mode"), dialog.cb_native_output_mode),
                         (tr("Sample rate (exclusive)"), dialog.cb_native_rate),
                         (tr("Process priority") if os_support.IS_WINDOWS else tr("Thread priority"),
                          dialog.cb_native_priority),
                         (tr("Power throttling"), dialog.cb_native_power)):
        slots = []
        for _place in range(2):
            slot = QWidget()
            QHBoxLayout(slot).setContentsMargins(0, 0, 0, 0)
            slots.append((QLabel(text), slot))
        dialog._native_rows.append((widget, slots[0], slots[1]))
    dialog.lbl_native_advanced = QLabel(tr("Native engine"))

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
    dialog.buffer_histogram = BufferHistogram()
    dialog.buffer_histogram.setToolTip(tr(
        "<p>The time the output buffer spent at each level over the last {seconds} s (log scale): the left end is "
        "a dropout. Dashed: the buffer size; yellow: the 1% low.</p>"
    ).format(seconds=f"{dialog.BUFFER_WINDOW_SEC:.0f}"))
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
    for _widget, (label, slot), _other in dialog._native_rows:
        f_general.addRow(label, slot)
    f_general.addRow(tr("Trim"), dialog.sp_volume_trim_dbfs)
    f_general.addRow(tr("Default volume (%)"), dialog.sp_default_volume_percent)
    f_general.addRow(dialog.cb_soft_clip)
    f_general.addRow(tr("Buffer size"), dialog.buffer_slot)
    f_general.addRow(tr("Output buffer"), dialog.buffer_meter)
    f_general.addRow(
        tr("1% low ({seconds} s)").format(seconds=f"{dialog.BUFFER_WINDOW_SEC:.0f}"), dialog.buffer_low_meter
    )
    f_general.addRow(tr("Histogram"), dialog.buffer_histogram)
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
    f_advanced.addRow(dialog.lbl_native_advanced)
    for _widget, _shown, (label, slot) in dialog._native_rows:
        f_advanced.addRow(label, slot)
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
        # the native engine's own rows: with the rest when it plays, else under Advanced (with its buffer)
        for widget, (label, slot), (other_label, other_slot) in self._native_rows:
            for s in (slot, other_slot):
                s.layout().removeWidget(widget)
            (slot if native else other_slot).layout().addWidget(widget)
            for w, shown in ((label, native), (slot, native), (other_label, not native), (other_slot, not native)):
                w.setVisible(shown)
        self.lbl_native_advanced.setVisible(not native)
        self._follow_native_api()
        self.lbl_other_buffer.setText(tr("Buffer size, Python engine") if native
                                      else tr("Buffer size, native engine"))
        playing = tr("Native") if native else tr("Python")
        tip = tr(
            "<p>Native: the audio engine in a process of its own (no dropouts when the computer is busy, smaller "
            "buffers); the Python engine where it is not built or cannot start. Python: the Python engine only. "
            "Applied at the next start.</p><p>Playing now: {engine}.</p>"
        ).format(engine=playing)
        if not os_support.IS_WINDOWS:
            # developed and measured on Windows; on Linux and macOS it is built and checked, not yet measured
            tip += tr("<p>The native engine on Linux and macOS is experimental.</p>")
        self.cb_audio_engine.setToolTip(tip)

    def _follow_native_api(self, *_args) -> None:
        """WASAPI's own mode only with its API, the exclusive rate only in exclusive mode; both hidden where the
        system has neither (only Windows has). The priority and power rows are every system's."""
        wasapi = self.cb_native_api.currentData() == "wasapi"
        exclusive = wasapi and bool(self.cb_native_output_mode.currentData())
        self.cb_native_output_mode.setEnabled(wasapi)
        self.cb_native_rate.setEnabled(exclusive)
        for widget, (label, slot), (other_label, other_slot) in self._native_rows:
            if widget in (self.cb_native_output_mode, self.cb_native_rate) and not os_support.IS_WINDOWS:
                for w in (label, slot, other_label, other_slot):
                    w.setVisible(False)

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
        if stats.get("output_open") is False:
            self.lbl_buffer.setText(tr(
                "The output device could not be opened (another app may hold it in exclusive mode); it is tried "
                "again every second."))
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
        # the histogram's bucket (ms): 1 from the Python engine, 0.1 (wider for big buffers) from the native one;
        # the window restarts where it changes
        bucket = float(stats.get("bucket_ms", 1.0))
        if bucket != getattr(self, "_buffer_bucket_ms", bucket):
            self._buffer_window.clear()
        self._buffer_bucket_ms = bucket
        self._buffer_window.append((now, hist))
        while self._buffer_window and now - self._buffer_window[0][0] > self.BUFFER_WINDOW_SEC:
            self._buffer_window.popleft()

        size = max(len(h) for _, h in self._buffer_window)
        total = np.zeros(size, dtype=np.float64)
        for _, h in self._buffer_window:
            total[:len(h)] += h
        cumulative = np.cumsum(total)
        target = 0.01 * cumulative[-1]
        k = int(np.searchsorted(cumulative, target))
        if cumulative[-1] > 0 and k < total.size and total[k] > 0:
            # within its bucket, by the share of the time there below the 1 %
            below = cumulative[k - 1] if k > 0 else 0.0
            low_1pct_ms = (k + (target - below) / total[k]) * bucket
        else:
            low_1pct_ms = k * bucket
        lowest_ms = float(np.flatnonzero(total)[0]) * bucket if cumulative[-1] > 0 else 0.0

        self.buffer_meter.set_level(self._buffer_ema_ms / capacity_ms)
        self.buffer_low_meter.set_level(low_1pct_ms / capacity_ms)
        self.buffer_histogram.set_data(total, bucket, capacity_ms, low_1pct_ms, micro="device_ms" in stats)
        values = dict(ema=f"{self._buffer_ema_ms:.0f}", capacity=f"{capacity_ms:.0f}", low=f"{low_1pct_ms:.0f}",
                      lowest=f"{lowest_ms:.0f}", seconds=f"{self.BUFFER_WINDOW_SEC:.0f}",
                      dropout=f"{self._buffer_dropout_sec * 1000.0:.0f}")
        if "device_ms" in stats:
            # the native engine: its buffer in µs (its histogram is 0.1 ms fine; the lowest level is its own,
            # exact, over the same 10 s)
            exact_low = float(stats.get("lowest_ms", -1.0))
            us = dict(ema=f"{self._buffer_ema_ms * 1000:,.0f}", capacity=f"{capacity_ms * 1000:,.0f}",
                      low=f"{low_1pct_ms * 1000:,.0f}",
                      lowest=f"{(exact_low if exact_low >= 0.0 else lowest_ms) * 1000:,.0f}",
                      seconds=values["seconds"], dropout=values["dropout"])
            text = tr(
                "EMA {ema} / {capacity} µs   1% low {low} µs   lowest {lowest} µs ({seconds} s)\n"
                "Dropout since opened {dropout} ms   Device latency {device} ms"
            ).format(device=f"{float(stats['device_ms']):.1f}", **us)
            # the output thread woken late (the device may have played a gap with audio in the ring)
            if int(stats.get("late", 0)) > 0:
                text += "\n" + tr("Late output wake-ups: {count} (longest gap {gap} ms)").format(
                    count=int(stats["late"]), gap=f"{float(stats.get('max_gap_ms', 0.0)):.1f}")
            # the threads' scheduling as the engine got it (Settings' priority), and power throttling not off
            text += "\n" + tr("Engine thread: {engine}   Output thread: {output}").format(
                engine=tr(THREAD_STATE_NAMES.get(stats.get("engine_thread"), THREAD_STATE_NAMES["api"])),
                output=tr(THREAD_STATE_NAMES.get(stats.get("output_thread"), THREAD_STATE_NAMES["api"])))
            if stats.get("power") == "partly":
                text += "\n" + tr("Power throttling could be turned off only in part.")
            elif stats.get("power") == "refused":
                text += "\n" + tr("Power throttling could not be turned off.")
            # the native engine's measured responses: a command sent -> its sound heard (the last of each)
            response = stats.get("response_ms") or {}
            if any(v > 0.0 for v in response.values()):
                shown = {k: (f"{v:.0f}" if v > 0.0 else "-") for k, v in response.items()}
                text += "\n" + tr("Response measured: play {play} ms   pause {pause} ms   seek {seek} ms").format(**shown)
            self.lbl_buffer.setText(text)
        else:
            self.lbl_buffer.setText(tr(
                "EMA {ema} / {capacity} ms   1% low {low} ms   lowest {lowest} ms ({seconds} s)\n"
                "Dropout since opened {dropout} ms"
            ).format(**values))
