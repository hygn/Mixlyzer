"""Settings > Audio's output: its buffer monitor, its last stage's choice, and what its groups share (the audio APIs,
the devices and channel pairs listed, the mode it opened as)."""
from collections import deque
import math
import time

import numpy as np

from PySide6.QtWidgets import QButtonGroup, QComboBox, QFormLayout, QHBoxLayout, QLabel, QRadioButton, QWidget

from audioengine.controls import LIMITER_DELAY_SEC
from system import os_support
from system.i18n import tr, tr_mark
from ui.widgets.buffer_histogram import BufferHistogram
from ui.deck.track_info_panel import HorizontalBufferMeter

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


def add_native_api_items(combo: QComboBox) -> None:
    """The audio APIs this system has, for the native engine (client.API_CODES)."""
    if os_support.IS_WINDOWS:
        combo.addItem("WASAPI", "wasapi")
        combo.addItem("WASAPI (cpal)", "cpal")
    elif os_support.IS_LINUX:
        combo.addItem("ALSA", "cpal")
        combo.addItem("JACK", "jack")
    else:
        combo.addItem("CoreAudio", "cpal")


def native_api_tip() -> str:
    return tr(
        "<p>The audio API the native engine plays through. WASAPI: its own, with a low-latency shared stream "
        "or exclusive mode. WASAPI (cpal): the usual shared stream (more latency), for a device the other one "
        "cannot open. JACK (Linux): through the JACK or PipeWire server.</p>"
    )


def output_mode_tip() -> str:
    return tr(
        "<p>Shared: with other apps, at the smallest period the driver allows. Exclusive: the device to Mixlyzer "
        "alone, the shortest latency; other apps cannot play on it meanwhile. A mode the device cannot open falls "
        "back to the usual shared stream.</p>"
    )


def opened_as_text(stats: dict, api: str, exclusive: bool, api_names: dict) -> str:
    """A line when the output did not open as asked (its API or mode: one that does not open falls back to cpal's
    shared stream), "" when it did."""
    opened_api = stats.get("output_api")
    if opened_api is None or (opened_api == api and bool(stats.get("exclusive")) == bool(exclusive or False)):
        return ""
    mode = tr("Exclusive") if stats.get("exclusive") else tr("Shared")
    return tr("Opened as {api}, {mode} (the output asked for could not be opened on this device)").format(
        api=api_names.get(opened_api, opened_api), mode=mode)


def api_names(combo: QComboBox) -> dict:
    """The audio APIs' names as the combo shows them, by code."""
    return {combo.itemData(i): combo.itemText(i) for i in range(combo.count())}


def load_channel_pairs(combo: QComboBox, channels: int, wanted: str) -> None:
    """The channel pairs of a device of ``channels`` channels, ``wanted`` ("3-4") picked; one the device has not
    stays listed as such."""
    pairs = [f"{c}-{c + 1}" for c in range(1, max(2, channels), 2)]
    combo.blockSignals(True)
    combo.clear()
    for pair in pairs:
        combo.addItem(tr("Channels {pair}").format(pair=pair), pair)
    if wanted and wanted not in pairs:
        combo.addItem(tr("Channels {pair} (not on this device)").format(pair=wanted), wanted)
    combo.setCurrentIndex(max(0, combo.findData(wanted)))
    combo.blockSignals(False)


def load_devices(combo: QComboBox, names: list[str], wanted: str) -> None:
    """The devices listed, ``wanted`` ("": the system's default) picked; one not connected stays listed as such."""
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(tr("System default"), "")
    for name in names:
        combo.addItem(name, name)
    if wanted and wanted not in names:
        combo.addItem(tr("{name} (not connected)").format(name=wanted), wanted)
    combo.setCurrentIndex(max(0, combo.findData(wanted)))
    combo.blockSignals(False)


class BufferMonitor:
    """An output's buffer as it played: its mean level (smoothed), the level it stayed above 99 % of the time over
    the last WINDOW_SEC, the time at each level (a histogram), and a readout under them; rows of a form."""

    WINDOW_SEC = 10.0
    EMA_TAU_SEC = 0.3

    def __init__(self) -> None:
        self.meter = HorizontalBufferMeter(hold_marker=False)
        self.low_meter = HorizontalBufferMeter(hold_marker=False)
        self.histogram = BufferHistogram()
        self.histogram.setToolTip(tr(
            "<p>The time the output buffer spent at each level over the last {seconds} s (log scale): the left end is "
            "a dropout. Dashed: the buffer size; yellow: the 1% low.</p>"
        ).format(seconds=f"{self.WINDOW_SEC:.0f}"))
        self.meter.setToolTip(tr("Average audio queued in the output buffer."))
        self.low_meter.setToolTip(tr("Buffer level the output stayed above 99% of the time in the recent window."))
        self.label = QLabel(tr("Waiting for playback..."))
        self.label.setToolTip(tr("Audio queued in the output device buffer while playing, 0 ms = dropout."))
        # short lines that wrap further if need be: the readout never widens the page
        self.label.setWordWrap(True)
        self._rows: list = []
        self.reset()

    def add_rows(self, form: QFormLayout) -> None:
        rows = ((tr("Output buffer"), self.meter),
                (tr("1% low ({seconds} s)").format(seconds=f"{self.WINDOW_SEC:.0f}"), self.low_meter),
                (tr("Histogram"), self.histogram),
                ("", self.label))
        for text, widget in rows:
            label = QLabel(text)
            form.addRow(label, widget)
            self._rows.append((label, widget))

    def set_visible(self, shown: bool) -> None:
        for label, widget in self._rows:
            label.setVisible(shown)
            widget.setVisible(shown)

    def reset(self) -> None:
        self._ema_ms = None
        self._window: deque = deque()
        self._dropout_sec = 0.0
        self._bucket_ms = None

    def observe(self, stats: dict, capacity_ms: float, micro: bool) -> dict:
        """The observations since the last call (client's "hist", "seconds", "mean_ms", "dropout_sec",
        "bucket_ms"): the meters and the histogram updated; returns the readout's values (ms): ema, low, lowest,
        dropout."""
        seconds = float(stats["seconds"])
        hist = np.asarray(stats["hist"], dtype=np.float64)
        now = time.monotonic()
        # EMA over time of the time-weighted mean level
        mean_ms = float(stats["mean_ms"])
        if self._ema_ms is None:
            self._ema_ms = mean_ms
        else:
            alpha = 1.0 - math.exp(-seconds / self.EMA_TAU_SEC)
            self._ema_ms += alpha * (mean_ms - self._ema_ms)
        # the time the device played silence: told by the native engine (its buffer's lowest level is touched
        # without a gap: the device reads it in whole periods), else the time below 1 ms
        self._dropout_sec += float(stats.get("dropout_sec", hist[0] if hist.size else 0.0))
        # the histogram's bucket (ms): 1 from the Python engine, 0.1 (wider for big buffers) from the native one;
        # the window restarts where it changes
        bucket = float(stats.get("bucket_ms", 1.0))
        if self._bucket_ms is not None and bucket != self._bucket_ms:
            self._window.clear()
        self._bucket_ms = bucket
        self._window.append((now, hist))
        while self._window and now - self._window[0][0] > self.WINDOW_SEC:
            self._window.popleft()
        size = max(len(h) for _, h in self._window)
        total = np.zeros(size, dtype=np.float64)
        for _, h in self._window:
            total[:len(h)] += h
        cumulative = np.cumsum(total)
        target = 0.01 * cumulative[-1] if cumulative.size else 0.0
        k = int(np.searchsorted(cumulative, target))
        if cumulative.size and cumulative[-1] > 0 and k < total.size and total[k] > 0:
            # within its bucket, by the share of the time there below the 1 %
            below = cumulative[k - 1] if k > 0 else 0.0
            low_ms = (k + (target - below) / total[k]) * bucket
        else:
            low_ms = k * bucket
        lowest_ms = float(np.flatnonzero(total)[0]) * bucket if cumulative.size and cumulative[-1] > 0 else 0.0
        capacity = max(capacity_ms, 1e-6)
        self.meter.set_level(self._ema_ms / capacity)
        self.low_meter.set_level(low_ms / capacity)
        self.histogram.set_data(total, bucket, capacity_ms, low_ms, micro=micro)
        return {"ema": self._ema_ms, "low": low_ms, "lowest": lowest_ms, "dropout": self._dropout_sec * 1000.0}


def native_buffer_text(values: dict, capacity_ms: float, lowest_ms: float, device_ms: float) -> str:
    """The native engine's buffer readout's first lines (µs: its histogram is 0.1 ms fine; ``lowest_ms`` its own,
    exact, over the same window, -1: not known)."""
    us = dict(ema=f"{values['ema'] * 1000:,.0f}", capacity=f"{capacity_ms * 1000:,.0f}",
              low=f"{values['low'] * 1000:,.0f}",
              lowest=f"{(lowest_ms if lowest_ms >= 0.0 else values['lowest']) * 1000:,.0f}",
              seconds=f"{BufferMonitor.WINDOW_SEC:.0f}", dropout=f"{values['dropout']:.0f}")
    return tr(
        "EMA {ema} / {capacity} µs   1% low {low} µs   lowest {lowest} µs ({seconds} s)\n"
        "Dropout since opened {dropout} ms   Device latency {device} ms"
    ).format(device=f"{device_ms:.1f}", **us)


class ClipChoice(QWidget):
    """An output's last stage (audioengine.controls.CLIPS) as one line of radio buttons, the limiter's delay said
    beside them while it is picked."""

    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, QRadioButton] = {}
        for value, name in (("hard", tr("Hard clip")), ("tanh", tr("Soft clip (tanh)")), ("limiter", tr("Limiter"))):
            button = QRadioButton(name)
            self._group.addButton(button)
            self._buttons[value] = button
            row.addWidget(button)
        self.lbl_delay = QLabel(tr("+{ms:g} ms latency").format(ms=LIMITER_DELAY_SEC * 1000.0))
        self.lbl_delay.setToolTip(tr("<p>The limiter looks this far ahead: this output is heard that much later. "
                                     "On the master the playhead counts it.</p>"))
        row.addWidget(self.lbl_delay)
        row.addStretch(1)
        tip = tr(
            "<p>What keeps this output at 0 dBFS at most, so the system's mixer never has to limit it on its way to "
            "the device.</p><p>Hard clip: nothing changes under 0 dBFS; an over is cut flat (heard as distortion). "
            "Soft clip (tanh): an over is rounded off above -1.9 dBFS; the top 2 dB are a little compressed. "
            "Limiter: the level is brought down smoothly just before an over and back up over 100 ms; nothing "
            "changes under 0 dBFS; the output is heard {ms:g} ms later (its look ahead).</p>"
            "<p>The Python engine plays the master alone, and the soft clip in place of the limiter.</p>").format(
            ms=LIMITER_DELAY_SEC * 1000.0)
        for w in (self, *self._buttons.values()):
            w.setToolTip(tip)
        self._group.buttonToggled.connect(self._follow)
        self.set_value("tanh")

    def set_value(self, value: str) -> None:
        self._buttons.get(value, self._buttons["tanh"]).setChecked(True)

    def value(self) -> str:
        return next((v for v, b in self._buttons.items() if b.isChecked()), "tanh")

    def _follow(self, *_args) -> None:
        self.lbl_delay.setVisible(self.value() == "limiter")
