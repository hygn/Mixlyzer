"""Settings > Playback's outputs: an output's buffer monitor, and the group of a bus beyond the master (the
headphones, the booth) with where it plays, its device's options, its buffer and its monitor."""
from collections import deque
import math
import time

import numpy as np

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QLabel, QSpinBox

from audioengine.native.client import NativeFeeder
from core import os_support
from core.config import outputconfig
from core.i18n import tr, tr_mark
from ui.buffer_histogram import BufferHistogram
from ui.live_label import LiveLabel
from ui.track_info_panel import HorizontalBufferMeter

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


def load_channel_pairs(combo: QComboBox, channels: int, wanted: str, off: bool = False) -> None:
    """The channel pairs of a device of ``channels`` channels, ``wanted`` ("3-4") picked; one the device has not
    stays listed as such; ``off``: a None item first."""
    pairs = [f"{c}-{c + 1}" for c in range(1, max(2, channels), 2)]
    combo.blockSignals(True)
    combo.clear()
    if off:
        combo.addItem(tr("None"), "")
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
        self.meter.setToolTip(tr("Average audio queued in the output buffer (smoothed)."))
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


class AuxOutputGroup(QGroupBox):
    """A bus of the native engine's beyond the master (the headphones, the booth): where it plays (off, channels
    of the main output's device, or a device of its own: its device, audio API, mode, channels and buffer), and,
    on a device of its own, its buffer monitor (the bridge from the main device's mix, src/bridge.rs)."""

    def __init__(self, title: str, tip: str, list_devices, main_channels, cue_options: bool = False) -> None:
        """``list_devices(api)`` -> (names, channels by name, "" the system default's) of the native engine (None:
        not known); ``main_channels()`` -> the main output's device's channels; ``cue_options``: the headphones'
        (the master mixed in or not, the CUE / MASTER mix)."""
        super().__init__(title)
        self._list_devices = list_devices
        self._main_channels = main_channels
        self._native = True
        self._device_channels: dict[str, int] = {}
        self._wanted_device = ""
        self._wanted_channels = ""
        self.cb_route = QComboBox()
        self.cb_route.addItem(tr("Off"), "off")
        self.cb_route.addItem(tr("Channels of the main output's device"), "main_device")
        self.cb_route.addItem(tr("A device of its own"), "device")
        self.cb_route.setToolTip(tip + tr(
            "<p>Channels of the main output's device: a controller's or an interface's other output, played with "
            "the master. A device of its own: another device (its clock followed, the master's sound kept: "
            "resampled by the clocks' drift, a few hundred ppm at most).</p>"))
        self.cb_device = QComboBox()
        self.cb_device.setToolTip(tr("<p>The device this output plays on, as the native engine names it.</p>"))
        self.cb_api = QComboBox()
        add_native_api_items(self.cb_api)
        self.cb_api.setToolTip(native_api_tip())
        self.cb_mode = QComboBox()
        self.cb_mode.addItem(tr("Shared"), False)
        self.cb_mode.addItem(tr("Exclusive"), True)
        self.cb_mode.setToolTip(output_mode_tip())
        self.cb_channels = QComboBox()
        self.cb_channels.setToolTip(tr("<p>The device's channels this output plays on.</p>"))
        self.cb_rate = QComboBox()
        self.cb_rate.addItem(tr("The device's"), 0)
        for rate in NativeFeeder.EXCLUSIVE_RATES:
            self.cb_rate.addItem(f"{rate / 1000:g} kHz", rate)
        self.cb_rate.setToolTip(tr(
            "<p>The rate this output's device runs at; the main output's sound is converted to it (a band-limited "
            "interpolator, with the clocks' drift). The device's: its own format (WASAPI's shared mix format), the "
            "lowest latency. Another rate: in exclusive mode the device runs at it; shared, the system converts it "
            "(WASAPI (cpal)'s usual stream, more latency).</p>"))
        self.sp_buffer = QSpinBox()
        self.sp_buffer.setRange(*NativeFeeder.BUFFER_MS_RANGE)
        self.sp_buffer.setSingleStep(5)
        self.sp_buffer.setSuffix(" ms")
        self.sp_buffer.setToolTip(tr(
            "<p>The audio kept ahead of this output's device, from the main output's mix: the least it holds "
            "whatever the two devices' periods (it holds up to one main device period more as they fall). Smaller "
            "is heard sooner; too small drops out when the computer is busy. Its device's own latency comes after "
            "it.</p>"))
        # its clock against the main device's, as the engine follows it (the ratio it is resampled at)
        self.lbl_drift = LiveLabel("-")
        self.lbl_drift.setToolTip(tr(
            "<p>This device's clock against the main output's: how much faster (+) or slower (-) it runs, in parts "
            "per million. The output is resampled by it (a band-limited interpolator), so it neither runs dry nor "
            "fills up, and nothing is repeated or dropped. A few tens to a few hundred ppm is usual between two "
            "devices.</p>"))
        # the headphones: the master mixed into them, or not (a controller mixing it in itself)
        self.cb_master_mix = QCheckBox(tr("Mix the master in"))
        self.cb_master_mix.setToolTip(tr(
            "<p>The headphones carry the channels whose CUE is on (before their faders) and, with this on, the "
            "master, mixed by the CUE / MASTER mix (a controller's knob moves it). Off for a controller that mixes "
            "the master into its headphones itself (the DDJ-400's CUE / MASTER knob, in analog): the headphones "
            "output then carries the CUE alone, silent while no CUE is on.</p>"))
        self.sp_cue_mix = QSpinBox()
        self.sp_cue_mix.setRange(0, 100)
        self.sp_cue_mix.setSuffix(" %")
        self.sp_cue_mix.setToolTip(tr(
            "<p>The master's share in the headphones: 0 % the CUE alone, 100 % the master alone. A controller's CUE "
            "/ MASTER knob moves it as it is turned.</p>"))
        self.cb_master_mix.toggled.connect(self.sp_cue_mix.setEnabled)
        self._cue_options = cue_options
        self.lbl_note = QLabel(tr("The native engine's: applied when it plays."))
        self.lbl_note.setWordWrap(True)
        self.monitor = BufferMonitor()
        form = QFormLayout(self)
        form.addRow(self.lbl_note)
        self._rows = {}
        rows = [("route", tr("Output"), self.cb_route),
                ("device", tr("Output device"), self.cb_device),
                ("api", tr("Audio API"), self.cb_api),
                ("mode", tr("Output mode"), self.cb_mode),
                ("rate", tr("Sample rate"), self.cb_rate),
                ("channels", tr("Channels"), self.cb_channels),
                ("buffer", tr("Buffer size"), self.sp_buffer)]
        if cue_options:
            rows += [("master_mix", tr("Master in the headphones"), self.cb_master_mix),
                     ("cue_mix", tr("CUE / MASTER"), self.sp_cue_mix)]
        for key, text, widget in rows:
            label = QLabel(text)
            form.addRow(label, widget)
            self._rows[key] = (label, widget)
        self.lbl_drift_title = QLabel(tr("Clock drift"))
        form.addRow(self.lbl_drift_title, self.lbl_drift)
        self.monitor.add_rows(form)
        self.cb_route.currentIndexChanged.connect(self._follow_route)
        self.cb_api.currentIndexChanged.connect(self._follow_api)
        self.cb_device.currentIndexChanged.connect(self._follow_device)
        self.cb_channels.currentIndexChanged.connect(self._follow_channels)
        self._follow_route()

    # the settings
    def set_config(self, o: outputconfig) -> None:
        self._wanted_device = o.device
        self._wanted_channels = o.channels
        for combo, value in ((self.cb_route, o.route), (self.cb_api, o.audio_api)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)
        self.cb_mode.setCurrentIndex(1 if o.exclusive else 0)
        self.cb_rate.setCurrentIndex(max(0, self.cb_rate.findData(int(o.rate))))
        self.sp_buffer.setValue(int(o.buffer_ms) or 20)
        self._follow_route()

    def get_config(self) -> outputconfig:
        return outputconfig(
            route=str(self.cb_route.currentData() or "off"),
            device=str(self.cb_device.currentData() or "") if self.cb_device.count() else self._wanted_device,
            channels=self._wanted_channels or "1-2",
            audio_api=str(self.cb_api.currentData() or "cpal"),
            exclusive=bool(self.cb_mode.currentData()),
            buffer_ms=int(self.sp_buffer.value()),
            rate=int(self.cb_rate.currentData() or 0),
        )

    def set_native(self, native: bool) -> None:
        """Whether the native engine plays (the Python one has none of these outputs)."""
        self._native = bool(native)
        self.lbl_note.setVisible(not native)
        self._follow_route()

    def refresh(self) -> None:
        """The devices (of the API picked) and the channel pairs as they are now."""
        if self.cb_route.currentData() == "device":
            listed = self._list_devices(str(self.cb_api.currentData() or "cpal"))
            names, channels = listed if listed is not None else ([], {})
            self._device_channels = channels
            load_devices(self.cb_device, names, self._wanted_device)
        self._load_channels()

    # following the choices
    def _follow_route(self, *_args) -> None:
        route = self.cb_route.currentData()
        own = route == "device"
        for key, (label, widget) in self._rows.items():
            shown = (key == "route" or (key in ("channels", "master_mix", "cue_mix") and route != "off")
                     or (own and key != "route"))
            if key == "mode" and not os_support.IS_WINDOWS:
                shown = False
            label.setVisible(shown)
            widget.setVisible(shown)
        self.monitor.set_visible(own and self._native)
        self.lbl_drift_title.setVisible(own and self._native)
        self.lbl_drift.setVisible(own and self._native)
        self.refresh()
        self._follow_api()

    def _follow_api(self, *_args) -> None:
        wasapi = self.cb_api.currentData() == "wasapi"
        self.cb_mode.setEnabled(wasapi)
        if self.sender() is self.cb_api:
            self.refresh()

    def _follow_device(self, *_args) -> None:
        self._wanted_device = str(self.cb_device.currentData() or "")
        self._load_channels()

    def _follow_channels(self, *_args) -> None:
        self._wanted_channels = str(self.cb_channels.currentData() or self._wanted_channels)

    def _load_channels(self) -> None:
        wanted = self._wanted_channels or "1-2"
        if self.cb_route.currentData() == "device":
            channels = self._device_channels.get(self._wanted_device, self._device_channels.get("", 2))
        else:
            channels = self._main_channels()
        load_channel_pairs(self.cb_channels, int(channels or 2), wanted)

    # the monitor
    def on_stats(self, d) -> None:
        """This output's part of the buffer stats (client: take_buffer_stats' "outputs"), None: it has none."""
        if not self.isVisible() or self.cb_route.currentData() != "device":
            return
        if d is None:
            self.monitor.label.setText(tr("Waiting for playback..."))
            self.lbl_drift.setText("-")
            return
        if d.get("state") == "opening":
            self.monitor.label.setText(tr("Opening the output device..."))
            return
        if not d.get("open"):
            self.monitor.label.setText(tr(
                "The output device could not be opened (another app may hold it in exclusive mode); it is tried "
                "again every second."))
            return
        if d.get("observed"):
            self.lbl_drift.setText(tr("{ppm} ppm (resampled to follow it)").format(
                ppm=f"{float(d.get('drift_ppm', 0.0)):+.1f}"))
        observed = d.get("observed")
        if not observed:
            return
        capacity_ms = max(float(d.get("capacity_ms", 0.0)), 1e-3)
        observed = dict(observed, bucket_ms=d.get("bucket_ms", 0.1))
        values = self.monitor.observe(observed, capacity_ms, micro=True)
        text = native_buffer_text(values, capacity_ms, float(d.get("lowest_ms", -1.0)), float(d.get("device_ms", 0.0)))
        text += "\n" + tr("Output thread: {output}").format(
            output=tr(THREAD_STATE_NAMES.get(d.get("output_thread"), THREAD_STATE_NAMES["api"])))
        if int(d.get("late", 0)) > 0:
            text += "\n" + tr("Late output wake-ups: {count} (longest gap {gap} ms)").format(
                count=int(d["late"]), gap=f"{float(d.get('max_gap_ms', 0.0)):.1f}")
        if d.get("device"):
            text += "\n" + tr("Playing on {device} at {rate} kHz").format(
                device=d["device"], rate=f"{int(d.get('rate', 0)) / 1000:g}")
        opened = opened_as_text(d, str(self.cb_api.currentData() or "cpal"),
                                self.cb_api.currentData() == "wasapi" and bool(self.cb_mode.currentData()),
                                api_names(self.cb_api))
        if opened:
            text += "\n" + opened
        # outputs on one device share its stream, at the first one's (the headphones') rate
        asked = int(self.cb_rate.currentData() or 0)
        if asked and int(d.get("rate", 0)) and asked != int(d["rate"]):
            text += "\n" + tr("Its device's stream is shared with another output, at that output's rate.")
        self.monitor.label.setText(text)
