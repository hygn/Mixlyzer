"""Settings > Audio: the audio engine and its outputs in pages: Overview (the signal path, each part's state and
buffer: ui/config/audio_overview.py), Engine, Main output, Headphones, Booth, Mixer, Metronome."""
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
    QPushButton,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from audioengine.native.client import BOOTH_EQ_DB, LEVEL_DB_MIN, NativeFeeder, booth_eq_knob_to_db, level_knob_to_db
from audioengine.python.feeder import PCMFeeder
from ui.config.output_group import (
    THREAD_STATE_NAMES,
    AuxOutputGroup,
    BufferMonitor,
    add_native_api_items,
    api_names,
    load_channel_pairs,
    load_devices,
    native_api_tip,
    native_buffer_text,
    opened_as_text,
    output_mode_tip,
)
from ui.config.audio_overview import AudioOverview
from ui.config.scroll import scroll_page
from core import os_support
from core.i18n import tr, tr_ctx
from utils.volume import TRIM_DB_RANGE


def build_playback_tab(dialog) -> None:
    dialog.tab_playback = QWidget()

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
    dialog.sp_volume_trim_dbfs.setRange(*TRIM_DB_RANGE)   # the player's TRIM knob's
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

    # the audio engine (applied at the next start)
    dialog.cb_audio_engine = QComboBox()
    dialog.cb_audio_engine.addItem(tr("Native first"), "native")
    dialog.cb_audio_engine.addItem(tr("Python"), "python")
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

    # the main output: its device, the native engine's way to it (the audio APIs this system has, WASAPI's mode
    # and exclusive rate), the master's channels, each engine's buffer (ms): the one playing with the rest, the
    # other one's under Advanced (set_audio_engine)
    dialog.cb_output_device = QComboBox()
    dialog.cb_output_device.setToolTip(tr(
        "<p>The device the player plays on, as its audio engine names it. Changed while the app runs, the sound "
        "keeps the format (sample rate, channels) of the device it started with, converted by the system.</p>"
    ))
    dialog.cb_native_api = QComboBox()
    add_native_api_items(dialog.cb_native_api)
    dialog.cb_native_api.setToolTip(native_api_tip())
    dialog.cb_native_output_mode = QComboBox()
    dialog.cb_native_output_mode.addItem(tr("Shared"), False)
    dialog.cb_native_output_mode.addItem(tr("Exclusive"), True)
    dialog.cb_native_output_mode.setToolTip(output_mode_tip())
    dialog.cb_native_rate = QComboBox()
    dialog.cb_native_rate.addItem(tr("The device's"), 0)
    for rate in NativeFeeder.EXCLUSIVE_RATES:
        dialog.cb_native_rate.addItem(f"{rate / 1000:g} kHz", rate)
    dialog.cb_native_rate.setToolTip(tr(
        "<p>The sample rate the device runs at in exclusive mode (the tracks are decoded for it). Applied from the "
        "next track loaded; a rate the device does not take falls back to the usual shared stream.</p>"
    ))
    dialog.cb_native_api.currentIndexChanged.connect(dialog._follow_native_api)
    dialog.cb_native_output_mode.currentIndexChanged.connect(dialog._follow_native_api)
    dialog.cb_output_device.currentIndexChanged.connect(dialog._follow_main_device)
    dialog.cb_output_main = QComboBox()
    dialog.cb_output_main.setToolTip(tr("<p>The device's channels the master plays on.</p>"))
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
    # where each engine's buffer goes (set_audio_engine): with the rest, or under Advanced
    dialog.buffer_slot = QWidget()
    QHBoxLayout(dialog.buffer_slot).setContentsMargins(0, 0, 0, 0)
    dialog.other_buffer_slot = QWidget()
    QHBoxLayout(dialog.other_buffer_slot).setContentsMargins(0, 0, 0, 0)
    dialog.lbl_other_buffer = QLabel()
    dialog.main_monitor = BufferMonitor()

    dialog.cb_metronome.setToolTip(tr("Play a click on every beat of the beat grid."))
    dialog.ed_metronome_wav_path.setToolTip(tr("WAV file used as the metronome click."))
    dialog.sp_volume_trim_dbfs.setToolTip(
        tr("The level before the volume fader, as the player's TRIM knob sets it (-24 to +9 dB).")
    )
    dialog.sp_default_volume_percent.setToolTip(tr("Volume slider position at startup."))
    dialog.sp_click_down_volume.setToolTip(tr("Click volume on the first beat of each bar."))
    dialog.sp_click_down_pitch.setToolTip(
        tr("Pitch shift of the click on the first beat of each bar.")
    )
    dialog.sp_click_beat_volume.setToolTip(tr("Click volume on the other beats."))
    dialog.sp_click_beat_pitch.setToolTip(tr("Pitch shift of the click on the other beats."))

    grp_mixer = QGroupBox(tr("Channel"))
    f_mixer = QFormLayout(grp_mixer)
    f_mixer.addRow(tr("Trim"), dialog.sp_volume_trim_dbfs)
    f_mixer.addRow(tr("Default volume (%)"), dialog.sp_default_volume_percent)
    lbl_mixer_note = QLabel(tr("The EQ, the channel fader's curve, the headphones CUE and the master and headphones "
                               "levels are on the main screen, between the player and the overview."))
    lbl_mixer_note.setWordWrap(True)
    f_mixer.addRow(lbl_mixer_note)

    grp_master = QGroupBox(tr("Master"))
    f_master = QFormLayout(grp_master)
    f_master.addRow(dialog.cb_soft_clip)

    # the booth's level and EQ (a DJ mixer's BOOTH knobs: set for the room), the headphones' MONO SPLIT
    def db_spin(low: float, high: float, tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" dB")
        spin.setToolTip(tip)
        return spin

    dialog.sp_booth_level = db_spin(LEVEL_DB_MIN, round(level_knob_to_db(1.0), 1), tr(
        "<p>The booth output's level against the master ({low} dB: silent).</p>").format(low=f"{LEVEL_DB_MIN:.0f}"))
    dialog.sp_booth_eq_hi = db_spin(*BOOTH_EQ_DB, tr("<p>The booth's high shelf (-12 to +6 dB).</p>"))
    dialog.sp_booth_eq_low = db_spin(*BOOTH_EQ_DB, tr("<p>The booth's low shelf (-12 to +6 dB).</p>"))
    grp_booth_mix = QGroupBox(tr("Booth level and EQ"))
    f_booth_mix = QFormLayout(grp_booth_mix)
    f_booth_mix.addRow(tr("Level"), dialog.sp_booth_level)
    f_booth_mix.addRow(tr("EQ HI"), dialog.sp_booth_eq_hi)
    f_booth_mix.addRow(tr("EQ LOW"), dialog.sp_booth_eq_low)
    dialog.cb_headphones_split = QCheckBox(tr("MONO SPLIT"))
    dialog.cb_headphones_split.setToolTip(tr(
        "<p>The headphones in mono, split: the CUE in the left ear, the master in the right (a DJ mixer's MONO "
        "SPLIT / STEREO switch). Off: both in stereo, mixed by the CUE / MASTER mix.</p>"))
    grp_hp_mix = QGroupBox(tr("Headphones mix"))
    f_hp_mix = QFormLayout(grp_hp_mix)
    f_hp_mix.addRow(dialog.cb_headphones_split)

    grp_main = QGroupBox(tr("Main output"))
    f_main = QFormLayout(grp_main)
    f_main.addRow(tr("Output device"), dialog.cb_output_device)
    # the native engine's own rows (hidden while the Python engine plays: it has none of them)
    dialog._native_rows = []
    for text, widget in ((tr("Audio API"), dialog.cb_native_api), (tr("Output mode"), dialog.cb_native_output_mode),
                         (tr("Sample rate (exclusive)"), dialog.cb_native_rate),
                         (tr("Channels"), dialog.cb_output_main)):
        label = QLabel(text)
        f_main.addRow(label, widget)
        dialog._native_rows.append((label, widget))
    f_main.addRow(tr("Buffer size"), dialog.buffer_slot)
    dialog.main_monitor.add_rows(f_main)

    grp_engine = QGroupBox(tr("Engine"))
    f_engine = QFormLayout(grp_engine)
    f_engine.addRow(tr("Audio engine"), dialog.cb_audio_engine)
    f_engine.addRow(tr("Process priority") if os_support.IS_WINDOWS else tr("Thread priority"),
                    dialog.cb_native_priority)
    f_engine.addRow(tr("Power throttling"), dialog.cb_native_power)
    # the last resort: the native engine started again (a device or the engine stuck)
    dialog.btn_restart_engine = QPushButton(tr("Restart Audio Engine"))
    dialog.btn_restart_engine.setToolTip(tr(
        "<p>Stops the native audio engine and starts it again on the same outputs, with the track, its place and "
        "the settings as they were. The last resort when an output or the engine stays stuck (a controller "
        "unplugged and plugged in again).</p>"))
    dialog.lbl_restart_engine = QLabel()
    restart_row = QHBoxLayout()
    restart_row.addWidget(dialog.btn_restart_engine)
    restart_row.addWidget(dialog.lbl_restart_engine, 1)
    f_engine.addRow(restart_row)
    dialog.btn_restart_engine.clicked.connect(dialog._restart_engine)
    dialog.btn_playback_advanced = QToolButton()
    dialog.btn_playback_advanced.setText(tr("Advanced"))
    dialog.btn_playback_advanced.setCheckable(True)
    dialog.btn_playback_advanced.setArrowType(Qt.RightArrow)
    dialog.btn_playback_advanced.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    dialog.btn_playback_advanced.setAutoRaise(True)
    dialog.btn_playback_advanced.setToolTip(tr("Show or hide the settings of the audio engine not playing."))
    f_engine.addRow(dialog.btn_playback_advanced)
    dialog.playback_advanced = QWidget()
    f_advanced = QFormLayout(dialog.playback_advanced)
    f_advanced.setContentsMargins(16, 0, 0, 0)
    f_advanced.addRow(dialog.lbl_other_buffer, dialog.other_buffer_slot)
    dialog.playback_advanced.setVisible(False)
    f_engine.addRow(dialog.playback_advanced)
    dialog.btn_playback_advanced.toggled.connect(dialog._toggle_playback_advanced)

    # the headphones and the booth: off, channels of the main device, or a device of their own
    dialog.grp_output_headphones = AuxOutputGroup(
        tr("Headphones output"),
        tr("<p>The headphones (a controller's or an interface's second output): the decks whose CUE is on, before "
           "their faders, mixed with the master (a controller's CUE / MASTER knob).</p>"),
        dialog._list_native_devices, dialog._main_output_channels, cue_options=True)
    dialog.grp_output_booth = AuxOutputGroup(
        tr("Booth output"), tr("<p>The booth monitors: the master again.</p>"),
        dialog._list_native_devices, dialog._main_output_channels)
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

    # the pages
    def page(*groups) -> QWidget:
        widget = QWidget()
        lay = QVBoxLayout(widget)
        for group in groups:
            lay.addWidget(group)
        lay.addStretch(1)
        return scroll_page(widget)

    dialog.audio_overview = AudioOverview()
    dialog.audio_tabs = QTabWidget(dialog.tab_playback)
    dialog.audio_tabs.addTab(scroll_page(dialog.audio_overview), tr_ctx("settings page", "Overview"))
    dialog.audio_tabs.addTab(page(grp_engine, grp_timestretch), tr("Engine"))
    dialog.audio_tabs.addTab(page(grp_main, grp_master), tr("Main output"))
    dialog.audio_tabs.addTab(page(dialog.grp_output_headphones, grp_hp_mix), tr("Headphones"))
    dialog.audio_tabs.addTab(page(dialog.grp_output_booth, grp_booth_mix), tr("Booth"))
    dialog.audio_tabs.addTab(page(grp_mixer), tr("Mixer"))
    dialog.audio_tabs.addTab(page(grp_metronome), tr("Metronome"))
    root = QVBoxLayout(dialog.tab_playback)
    root.addWidget(dialog.audio_tabs)
    dialog.tabs.addTab(dialog.tab_playback, tr("Audio"))


class PlaybackSettingsMixin:
    def _initialize_playback_settings(self, bus) -> None:
        self._deck_mix: dict = {}
        self._midi_state = 0
        self._last_stats = None
        self._bus = bus
        if bus is not None:
            bus.sig_output_buffer.connect(self._on_output_buffer)
            bus.sig_deck_state.connect(self._on_deck_report)
            bus.sig_engine_restarted.connect(self._on_engine_restarted)

    def _restart_engine(self) -> None:
        if self._bus is None:
            return
        self.btn_restart_engine.setEnabled(False)
        self.lbl_restart_engine.setText(tr("Restarting..."))
        self._bus.sig_engine_restart_requested.emit()

    def _on_engine_restarted(self, ok: bool) -> None:
        self.btn_restart_engine.setEnabled(getattr(self, "_native_playing", False))
        self.lbl_restart_engine.setText(tr("Restarted.") if ok else
                                        tr("The engine did not start; it is tried again every 2 seconds."))

    def _on_deck_report(self, report) -> None:
        """The deck's mixer values and MIDI input state, for the overview."""
        from audioengine.native.client import MIX_KEYS
        if report.mix:
            self._deck_mix = {k: report.mix_value(k) for k in MIX_KEYS}
        self._midi_state = int(report.midi_state)
        self._deck_info = {"playing": report.playing, "factor": report.factor, "volume": report.volume,
                           "knobs": report.knobs, "channel_cue": report.channel_cue, "peak": report.peak,
                           "color": report.color}

    def _set_audio_mix_config(self, p) -> None:
        """The booth's level and EQ, the headphones' MONO SPLIT (playbackconfig)."""
        self.sp_booth_level.setValue(round(level_knob_to_db(float(p.booth_level)), 1))
        self.sp_booth_eq_hi.setValue(round(booth_eq_knob_to_db(float(p.booth_eq_hi)), 1))
        self.sp_booth_eq_low.setValue(round(booth_eq_knob_to_db(float(p.booth_eq_low)), 1))
        self.cb_headphones_split.setChecked(bool(p.headphones_split))

    def _overview_context(self) -> dict:
        return {
            "native": getattr(self, "_native_playing", False),
            "timestretch": self.cb_use_timestretch.isChecked(),
            "main_device": getattr(self, "_main_device_in_use", "") or self.cb_output_device.currentText(),
            "api_names": api_names(self.cb_native_api),
            "routes": {"headphones": self.grp_output_headphones.get_config(),
                       "booth": self.grp_output_booth.get_config()},
            "midi": (self._midi_input(), self._midi_state),
            "mix": self._deck_mix,
            "deck": getattr(self, "_deck_info", {}),
            "soft_clip": self.cb_soft_clip.isChecked(),
        }

    def _refresh_overview(self) -> None:
        if self.isVisible() and self.audio_tabs.currentIndex() == 0 and self.tabs.currentWidget() is self.tab_playback:
            stats, capacity = self._last_stats or (None, 0.0)
            self.audio_overview.update_view(stats, capacity, self._overview_context())

    def set_audio_engine(self, native: bool, devices, channels=None) -> None:
        """The engine that plays (``native``: its buffer is shown with the rest, the other one's under Advanced;
        the Python engine plays deck 1 on one output, without the native engine's rows and outputs) and
        ``devices(api=None)`` -> (the output device in use, the output devices, their channels by name) of it
        (None: not known yet); ``channels()`` -> its main output device's channels (when not listed)."""
        self._native_playing = bool(native)
        self.btn_restart_engine.setEnabled(bool(native))
        self._output_devices_provider = devices
        self._output_channels_provider = channels
        shown, other = ((self.sp_native_buffer_ms, self.sp_python_buffer_ms) if native
                        else (self.sp_python_buffer_ms, self.sp_native_buffer_ms))
        for slot in (self.buffer_slot, self.other_buffer_slot):
            for spin in (self.sp_native_buffer_ms, self.sp_python_buffer_ms):
                slot.layout().removeWidget(spin)
        self.buffer_slot.layout().addWidget(shown)
        self.other_buffer_slot.layout().addWidget(other)
        for label, widget in self._native_rows:
            label.setVisible(native)
            widget.setVisible(native)
        for group in (self.grp_output_headphones, self.grp_output_booth):
            group.set_native(native)
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
        system has neither (only Windows has). Another API lists its own devices."""
        wasapi = self.cb_native_api.currentData() == "wasapi"
        exclusive = wasapi and bool(self.cb_native_output_mode.currentData())
        self.cb_native_output_mode.setEnabled(wasapi)
        self.cb_native_rate.setEnabled(exclusive)
        for label, widget in self._native_rows:
            if widget in (self.cb_native_output_mode, self.cb_native_rate) and not os_support.IS_WINDOWS:
                label.setVisible(False)
                widget.setVisible(False)
        if self.sender() is self.cb_native_api and getattr(self, "_native_playing", False):
            self._load_output_devices(str(self.cb_output_device.currentData() or ""))

    def _toggle_playback_advanced(self, shown: bool) -> None:
        self.btn_playback_advanced.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)
        self.playback_advanced.setVisible(shown)
        self.tab_playback.updateGeometry()

    def _list_native_devices(self, api: str):
        """The native engine's devices of ``api``: (names, channels by name, "" the system default's), None while
        it does not play."""
        provider = getattr(self, "_output_devices_provider", None)
        if provider is None or not getattr(self, "_native_playing", False):
            return None
        _current, names, channels = provider(api)
        return names, channels

    def _load_output_devices(self, wanted: str) -> None:
        """The output devices of the engine playing (the native one's: of the API picked), ``wanted`` ("": the
        system's default) picked; one not connected stays listed as such."""
        provider = getattr(self, "_output_devices_provider", None)
        native = getattr(self, "_native_playing", False)
        listed = provider(str(self.cb_native_api.currentData() or "cpal") if native else None) \
            if provider is not None else ("", [], {})
        current, names, self._main_device_channels = listed
        self._main_device_in_use = current
        load_devices(self.cb_output_device, names, wanted)

    def _main_output_channels(self) -> int:
        """The main output's device's channels: as listed, else the engine's own count."""
        wanted = str(self.cb_output_device.currentData() or "")
        listed = getattr(self, "_main_device_channels", {}) or {}
        if wanted in listed:
            return int(listed[wanted])
        provider = getattr(self, "_output_channels_provider", None)
        return int(provider()) if provider is not None else 2

    def _follow_main_device(self, *_args) -> None:
        """Another main device: its channel pairs (and the headphones' and the booth's on it)."""
        self._load_output_routes(str(self.cb_output_main.currentData() or "1-2"))

    def _load_output_routes(self, main: str) -> None:
        """The main device's channel pairs, the master's picked (one the device has not stays listed as such),
        and the headphones' and the booth's lists as they are now."""
        load_channel_pairs(self.cb_output_main, self._main_output_channels(), main or "1-2")
        for group in (self.grp_output_headphones, self.grp_output_booth):
            group.refresh()

    def _reset_playback_monitor(self) -> None:
        self.main_monitor.reset()
        self.audio_overview.reset()
        self._last_stats = None
        for group in (self.grp_output_headphones, self.grp_output_booth):
            group.monitor.reset()

    def _toggle_metronome_details(self, shown: bool) -> None:
        self.btn_metronome_details.setArrowType(
            Qt.DownArrow if shown else Qt.RightArrow
        )
        self.metronome_details.setVisible(shown)
        self.tab_playback.updateGeometry()

    def _on_output_buffer(self, queued_ms: float, stats: object, capacity_ms: float) -> None:
        if not self.isVisible() or not isinstance(stats, dict):
            return
        self._last_stats = (stats, capacity_ms)
        self._refresh_overview()
        outputs = stats.get("outputs") or {}
        self.grp_output_headphones.on_stats(outputs.get("headphones"))
        self.grp_output_booth.on_stats(outputs.get("booth"))
        label = self.main_monitor.label
        if stats.get("output_open") is False:
            label.setText(tr(
                "The output device could not be opened (another app may hold it in exclusive mode); it is tried "
                "again every second."))
            return
        if "hist" not in stats or capacity_ms <= 0.0:
            return
        native = "device_ms" in stats
        values = self.main_monitor.observe(stats, capacity_ms, micro=native)
        if native:
            text = native_buffer_text(values, capacity_ms, float(stats.get("lowest_ms", -1.0)),
                                      float(stats["device_ms"]))
            # the API or mode opened, where it is not the one asked for (falls back to cpal's shared stream)
            api = str(self.cb_native_api.currentData() or "cpal")
            opened = opened_as_text(stats, api, api == "wasapi" and bool(self.cb_native_output_mode.currentData()),
                                    api_names(self.cb_native_api))
            if opened:
                text += "\n" + opened
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
        else:
            text = tr(
                "EMA {ema} / {capacity} ms   1% low {low} ms   lowest {lowest} ms ({seconds} s)\n"
                "Dropout since opened {dropout} ms"
            ).format(ema=f"{values['ema']:.0f}", capacity=f"{capacity_ms:.0f}", low=f"{values['low']:.0f}",
                     lowest=f"{values['lowest']:.0f}", seconds=f"{BufferMonitor.WINDOW_SEC:.0f}",
                     dropout=f"{values['dropout']:.0f}")
        label.setText(text)
