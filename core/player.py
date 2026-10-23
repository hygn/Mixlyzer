from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtMultimedia

from audioengine.native.client import (
    MIDI_EVENT_GUI, MIDI_EVENT_HOTCUE_SET, NO_CHANNEL, DeckReport, NativeClicks, NativeEngine, NativeFeeder,
)
from audioengine.python.clicks import ClickTrack
from audioengine.python.deck import PythonDeck
from audioengine.python.feeder import PCMFeeder
from core import os_support
from core.audio.decoder import decode_to_memmap
from core.event_bus import EventBus
from utils.volume import trim_knob_to_db


class _Latest:
    """A frequent report from the audio thread to the GUI (the playhead time, the output peak):
    the newest value and one queued notification at a time. The GUI reads the newest when it
    gets to it, so reports never pile up in its queue while it is busy (the stale ones are
    skipped, as frames are)."""

    def __init__(self, notify) -> None:
        self._notify = notify
        self._value = None
        self._pending = False

    def publish(self, value) -> None:
        """Audio thread."""
        self._value = value
        if not self._pending:
            self._pending = True
            self._notify.emit()

    def take(self):
        """GUI thread: the newest value. Cleared before it is read, so a value published from
        then on notifies again (none is lost)."""
        self._pending = False
        return self._value


class _AudioWorker(QtCore.QObject):
    """
    Owns the feeder; lives in the player's audio thread. The feeder is the native engine's
    (audioengine/native: playback in a process of its own, this thread only relays commands and
    reads its state) or, where that is not built or cannot start, the Python engine's PCMFeeder
    on a QAudioSink (audioengine/python), fed from this thread.

    Every command from PlayerController arrives through run_command(). Decoding runs on a
    separate thread so the sink keeps being fed while a track loads; transport commands
    that arrive meanwhile are replayed, in order, once the track is ready. The playhead time
    and the output peak reach the GUI through _Latest (newest only, never piling up).

    The deck (``deck``: the native engine's, src/deck.rs, or PythonDeck over PCMFeeder) takes the
    requests (actions, tempo, volume, JUMP, CALL) and reports what it is (DeckReport: playing,
    tempo, volume, the cue point, the lamps, the armed JumpCUE), which is all the GUI shows: this
    thread keeps no transport state of its own. The native engine reads a MIDI controller itself;
    the messages bound to the GUI's actions (and all of them while learning) come out here.
    """

    time_ready = QtCore.Signal()   # latest_time has a new value
    duration_changed = QtCore.Signal(float)
    playback_status = QtCore.Signal(bool)
    transport_enabled = QtCore.Signal(bool)
    track_loaded = QtCore.Signal(object, int)  # pcm [N, ch] float32, sample rate
    peak_ready = QtCore.Signal()   # latest_peak has a new value (dBFS)
    deck_ready = QtCore.Signal()   # latest_deck has a new DeckReport
    midi_events = QtCore.Signal(object)   # [(status, data1, data2, flags, action)] the GUI wants
    output_buffer = QtCore.Signal(float, object, float)  # queued ms, observations (see PCMFeeder.take_buffer_stats) or None, capacity ms
    engine_restarted = QtCore.Signal(bool)   # restart_engine: the new engine started
    _decoded = QtCore.Signal(int, object, str)  # token, pcm or None, error message

    PEAK_FLOOR_DBFS = -24.0
    # the output buffer's observations reported this often (they add up in between): a
    # monitor, not worth a report every frame
    BUFFER_REPORT_SEC = 0.25

    _COMMANDS = frozenset({
        "load", "unload", "play", "pause", "stop", "seek",
        "scrub_begin", "scrub_update", "scrub_end",
        "scratch_begin", "scratch_update", "scratch_end",
        "deck_action", "set_tempo", "set_tempo_range", "set_volume", "set_volume_trim", "set_marks",
        "jump_to", "call_at", "set_peak_meter_gain",
        "set_refresh_fps", "set_timestretch", "set_soft_clip", "set_buffer_ms", "set_output_device",
        "set_output_mode", "set_exclusive_rate", "set_process",
        "tone_hold", "midi_open", "midi_map", "midi_options", "set_midi_learn", "set_routing", "set_headphones",
        "set_knob", "set_mixer", "set_channel_cue", "restart_engine",
    })
    _WAIT_FOR_TRACK = frozenset({
        "play", "pause", "stop", "seek",
        "scrub_begin", "scrub_update", "scrub_end",
        "scratch_begin", "scratch_update", "scratch_end",
        "deck_action", "jump_to", "call_at",
    })

    def __init__(self, *, volume_position: float = 1.0, volume_trim_dbfs: float = 0.0, tempo_range: int = 100,
                 peak_meter_gain_linear: float = 1.0, buffer_ms: float = 80.0,
                 engine: Optional[NativeEngine] = None, output_device: str = "") -> None:
        super().__init__()
        self._device_name = output_device   # the output device asked for ("": the system's default)
        self._output_route = ("wasapi", False)   # the native engine's API and exclusive mode (as PlayerController started it)
        self._fmt: Optional[QtMultimedia.QAudioFormat] = None   # the Python engine's sink format (the session's)
        self._engine = engine   # the native engine, started by PlayerController (None: the Python engine)
        self.audio: Optional[QtMultimedia.QAudioSink] = None
        self.feeder: Optional[PCMFeeder | NativeFeeder] = None
        self.deck: Optional[PythonDeck | NativeFeeder] = None
        self.rate = 0
        self.ch = 0

        self._path: Optional[str] = None
        self._track_ready = False
        self._is_playing = False   # as the deck last reported
        self._duration_ms = 0
        self._last_emit_ms: float = -1
        self._last_peak: Optional[float] = None
        self._buffer_reported = 0.0
        self.latest_time = _Latest(self.time_ready)
        self.latest_peak = _Latest(self.peak_ready)
        self.latest_deck = _Latest(self.deck_ready)
        self._deck_key: Optional[tuple] = None
        self._marks: Optional[tuple] = None   # the track's marks (set_marks), sent again once a track is decoded
        self._midi_learn = False

        self._decode_token = 0
        self._decoding = False
        self._deferred: list[tuple[str, tuple]] = []

        self._volume_position = max(0.0, min(1.0, float(volume_position)))
        self._volume_trim_dbfs = float(volume_trim_dbfs)
        self._tempo_range = int(tempo_range)
        self._peak_meter_gain_linear = max(0.0, min(1.0, float(peak_meter_gain_linear)))
        self._buffer_ms = float(buffer_ms)  # the output buffer

        self._frame = QtCore.QTimer(self)
        self._frame.setTimerType(QtCore.Qt.PreciseTimer)
        self._frame.setInterval(16)
        self._frame.timeout.connect(self._tick)
        self._decoded.connect(self._on_decoded, QtCore.Qt.QueuedConnection)

    @QtCore.Slot()
    def initialize(self) -> None:
        engine = self._engine
        if engine is not None:
            self.rate, self.ch = engine.rate, engine.channels
            self.feeder = NativeFeeder(engine, self)
            self.deck = self.feeder
            route = getattr(self, "_native_route", None)
            if route is not None:
                self.feeder._device, self.feeder._api, self.feeder._exclusive, self.feeder._exclusive_rate = route
            self._init_feeder()
            return
        # the Python engine: this thread feeds the output device, served ahead of busy threads (an analysis
        # on every core); where this is not available the thread's Qt priority (PlayerController) is all
        os_support.register_audio_thread()
        out_dev = qt_output_device(self._device_name)
        fmt = out_dev.preferredFormat()
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.Float)
        self.rate = fmt.sampleRate()
        self.ch = fmt.channelCount()
        self._fmt = fmt

        self.audio = QtMultimedia.QAudioSink(out_dev, fmt, self)

        self.feeder = PCMFeeder(self.audio, self.rate, self.ch, self)
        self.deck = PythonDeck(self.feeder)
        self._init_feeder()

    def _init_feeder(self) -> None:
        self.feeder.set_buffer_ms(self._buffer_ms)   # the sink's size, set where the feeder opens it
        self.deck.set_volume_trim(self._volume_trim_dbfs)
        self.deck.set_volume(self._volume_position)
        self.deck.set_tempo_range(self._tempo_range)
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_meter_gain(self._peak_meter_gain_linear)   # a controller's level meter, from the start
        # at the track's end either engine plays on (silence): the record dragged or the playhead moved back plays
        # again from there
        # While playing, every transport change (seek, scrub, jump) is rendered by the feeder
        # into the running sink: reset()/start() cycles during scrubbing crash Qt 6.10's WASAPI
        # backend. The feeder suspends the sink only when playback is idle.
        self.feeder.open()
        self._frame.start()

    @QtCore.Slot(str, object)
    def run_command(self, name: str, args: tuple) -> None:
        if name not in self._COMMANDS or self.feeder is None:
            return
        if self._decoding and name in self._WAIT_FOR_TRACK:
            self._deferred.append((name, args))
            return
        getattr(self, name)(*args)

    # Track loading
    def load(self, path: str) -> None:
        self.feeder.pause()
        self.feeder.set_track(None)
        self.deck.new_track()
        self._path = path
        self._track_ready = False
        self._duration_ms = 0
        self._last_emit_ms = -1
        self._deferred.clear()
        self.duration_changed.emit(0.0)
        self.transport_enabled.emit(False)

        self._decode_token += 1
        self._decoding = True
        # the native engine at the session format asked for (exclusive mode's rate: from the next track on)
        apply = getattr(self.feeder, "apply_session_format", None)
        if apply is not None and apply():
            self.rate, self.ch = self.feeder.rate, self.feeder.ch
        token, rate, ch = self._decode_token, self.rate, self.ch
        adopt = getattr(self.feeder, "adopt", None)   # the native engine reads it from a shared block

        def decode() -> None:
            try:
                pcm, error = decode_to_memmap(path, rate, ch), ""
                if adopt is not None:
                    pcm = adopt(pcm)
            except Exception as exc:
                pcm, error = None, str(exc)
            try:
                self._decoded.emit(token, pcm, error)
            except RuntimeError:
                pass  # worker already destroyed (app shutting down)

        threading.Thread(target=decode, name="PlayerDecode", daemon=True).start()

    def unload(self) -> None:
        """No track: silent, nothing to play."""
        self.feeder.pause()
        self.feeder.set_track(None)
        self.deck.new_track()
        self._path = None
        self._track_ready = False
        self._decode_token += 1  # drop a decode still running
        self._decoding = False
        self._duration_ms = 0
        self._last_emit_ms = -1
        self._deferred.clear()
        self.duration_changed.emit(0.0)
        self.transport_enabled.emit(False)

    @QtCore.Slot(int, object, str)
    def _on_decoded(self, token: int, pcm: object, error: str) -> None:
        if token != self._decode_token or self.feeder is None:
            return  # a newer track was requested meanwhile
        self._decoding = False
        try:
            if pcm is None:
                raise RuntimeError(error)
            pcm_array = pcm if pcm.dtype == np.float32 and pcm.flags.c_contiguous else np.ascontiguousarray(pcm, np.float32)
            self.feeder.set_track(pcm_array)
        except Exception as exc:
            print(f"[Player] Decode error: {exc}")
            self._deferred.clear()
            self.transport_enabled.emit(True)
            return

        self._track_ready = True
        self._duration_ms = int(round((pcm_array.shape[0] * 1000.0) / max(1, self.rate)))
        # the deck starts the track anew (the native engine on its load): its marks again
        self.deck.new_track()
        if self._marks is not None:
            self.deck.set_marks(*self._marks)
        self.track_loaded.emit(pcm_array, self.rate)
        self.duration_changed.emit(self._duration_ms / 1000.0)
        self.transport_enabled.emit(True)

        deferred, self._deferred = self._deferred, []
        for name, args in deferred:
            self.run_command(name, args)

    # Transport: requests; whether it plays and where comes back in the deck's report and the playhead
    @QtCore.Slot()
    def play(self) -> None:
        if self._path and self._track_ready:
            self.feeder.play()

    @QtCore.Slot()
    def pause(self) -> None:
        self.feeder.pause()

    def stop(self) -> None:
        self.feeder.pause()
        self.feeder.seek(0)

    def seek(self, sec: float) -> None:
        if not self._track_ready:
            return
        self.feeder.seek(int(round(max(0.0, sec) * self.rate)))

    # The deck (src/deck.rs / PythonDeck)
    def deck_action(self, code: int, pressed: bool) -> None:
        if self._track_ready:
            self.deck.deck_action(int(code), bool(pressed))

    def set_tempo(self, factor: float) -> None:
        self.deck.set_tempo(float(factor))

    def set_tempo_range(self, percent: int) -> None:
        self.deck.set_tempo_range(int(percent))

    def set_volume(self, position: float) -> None:
        self.deck.set_volume(float(position))

    def set_volume_trim(self, db: float) -> None:
        self.deck.set_volume_trim(float(db))

    def set_marks(self, marks: tuple) -> None:
        """(beats s, downbeat flags, CUE points s, JumpCUEs A-Z s): kept for the next track decoded too (they
        may come before it)."""
        self._marks = marks
        if self._track_ready:
            self.deck.set_marks(*marks)

    def jump_to(self, sec: float) -> None:
        if self._track_ready:
            self.deck.jump_to(float(sec))

    def call_at(self, sec: float) -> None:
        if self._track_ready:
            self.deck.call_at(float(sec))

    # The MIDI controller: the native engine's alone
    def midi_open(self, port: str) -> None:
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.midi_open(port)

    def midi_map(self, bindings: list) -> None:
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.midi_map(bindings)

    def midi_options(self, jog_ticks, vinyl) -> None:
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.midi_options(jog_ticks, vinyl)

    def set_midi_learn(self, learning: bool) -> None:
        """Every message passed on (Settings > MIDI's learn), not only those bound to the GUI's actions."""
        self._midi_learn = bool(learning)

    def set_routing(self, main: str, headphones: dict, booth: dict) -> None:
        """The native engine's outputs: the master on the main device's channel pair ``main`` ("1-2"), the
        headphones and the booth as outputconfig's fields say (off, channels of the main device, or a device of
        their own); the Python engine plays deck 1 on its device's first two channels and has none of them."""
        if not isinstance(self.feeder, NativeFeeder):
            return
        none = (NO_CHANNEL, NO_CHANNEL)
        on_main = [channel_pair(o.get("channels", "")) or none if o.get("route") == "main_device" else none
                   for o in (headphones, booth)]
        self.feeder.set_routing(channel_pair(main) or (0, 1), on_main[0], on_main[1])
        for k, o in enumerate((headphones, booth)):
            if o.get("route") == "device":
                self.feeder.set_aux_output(k, str(o.get("device") or ""), str(o.get("audio_api") or "wasapi"),
                                           bool(o.get("exclusive")), float(o.get("buffer_ms") or 20),
                                           channel_pair(o.get("channels", "")) or (0, 1), int(o.get("rate") or 0))
            else:
                self.feeder.set_aux_output(k, None)

    def restart_engine(self) -> None:
        """The native engine started again by hand, the track and the settings as they were."""
        if isinstance(self.feeder, NativeFeeder):
            self.engine_restarted.emit(self.feeder.restart_engine())

    def set_headphones(self, master: bool, mix: float, level: float) -> None:
        """The native engine's headphones: the master mixed in (by ``mix``, 0 CUE .. 1 MASTER) or not, their level
        (a knob's position)."""
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_headphones(master, mix, level)

    def set_channel_cue(self, on: bool) -> None:
        """The deck's channel's headphones CUE (the native engine's mixer)."""
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_channel_cue(bool(on))

    def set_knob(self, which: int, position: float) -> None:
        """The deck's channel knob: TRIM (0, the trim on either engine) or EQ LOW / MID / HIGH, COLOR (1 - 4: the
        native engine's mixer; the Python engine has none)."""
        if which == 0:
            self.deck.set_volume_trim(trim_knob_to_db(position))
        elif isinstance(self.feeder, NativeFeeder):
            self.feeder.set_knob(int(which), float(position))

    def set_mixer(self, key: str, value: float) -> None:
        """The native engine's mixer setting ``key`` (client.MIX_KEYS; the Python engine has no mixer)."""
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_mixer(str(key), float(value))

    def scrub_begin(self) -> None:
        self.feeder.set_scrubbing(True)

    def scrub_update(self, sec: float) -> None:
        if not self._track_ready:
            return
        self.feeder.seek(int(round(sec * self.rate)))

    def scrub_end(self, _sec: float) -> None:
        self.feeder.set_scrubbing(False)

    # Scratch: no scratch sound yet, the deck is silent while held
    def scratch_begin(self, sec_per_px: float = 0.0) -> None:
        self.feeder.set_scratching(True, sec_per_px * self.rate)

    def scratch_update(self, sec: float, at: float) -> None:
        if not self._track_ready:
            return
        self.feeder.scratch_to(sec * self.rate, at)
        self._emit_time(self._track_now_ms())

    def scratch_end(self, _sec: float) -> None:
        self.feeder.set_scratching(False)

    # Settings
    def set_timestretch(self, enabled: bool) -> None:
        self.feeder.set_timestretch(bool(enabled))

    def set_soft_clip(self, enabled: bool) -> None:
        self.feeder.set_soft_clip(bool(enabled))

    def set_buffer_ms(self, ms: float) -> None:
        self._buffer_ms = float(ms)
        self.feeder.set_buffer_ms(self._buffer_ms)

    def set_output_mode(self, api: str, exclusive: bool) -> None:
        if isinstance(self.feeder, NativeFeeder) and (api, exclusive) != self._output_route:
            self._output_route = (api, exclusive)
            self.feeder.set_output(api, exclusive)

    def set_exclusive_rate(self, rate: int) -> None:
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_exclusive_rate(rate)

    def set_process(self, priority: str, power_throttling: bool) -> None:
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_process(priority, power_throttling)

    def set_output_device(self, name: str) -> None:
        """Play on the output device ``name`` ("": the system's default) from now, at the session's format."""
        if name == self._device_name:
            return
        self._device_name = name
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_output_device(name)
            return
        self.feeder.set_sink(QtMultimedia.QAudioSink(qt_output_device(name), self._fmt, self))

    def tone_hold(self, notes: tuple) -> None:
        self.feeder.tones.hold(notes)
        if notes:
            self.feeder.wake()

    def set_peak_meter_gain(self, value: float) -> None:
        self._peak_meter_gain_linear = max(0.0, min(1.0, float(value)))
        if isinstance(self.feeder, NativeFeeder):
            self.feeder.set_meter_gain(self._peak_meter_gain_linear)   # a controller's level meter: the same reading
        self._emit_peak()

    def set_refresh_fps(self, fps: int) -> None:
        fps = max(1, min(240, int(fps)))
        self._frame.setInterval(max(1, int(round(1000.0 / fps))))

    @QtCore.Slot()
    def shutdown(self) -> None:
        self._decode_token += 1  # drop any decode still running
        self._decoding = False
        self._deferred.clear()
        if self.feeder is None:
            return
        self.stop()
        self.midi_open("")   # the controller's lamps out
        self._track_ready = False
        self._publish_peak(self.PEAK_FLOOR_DBFS)
        self.transport_enabled.emit(False)
        self._frame.stop()
        self.feeder.close()

    # Periodic time / peak reporting
    def _track_now_ms(self) -> float:
        return (self.feeder.playhead_frame() * 1000.0) / max(1, self.rate)

    def _emit_time(self, ms: float, force: bool = False) -> None:
        """The playhead time for the GUI, when it moved (``force``: also when not)."""
        if force or ms != self._last_emit_ms:
            self._last_emit_ms = ms
            self.latest_time.publish(ms / 1000.0)

    def _publish_peak(self, dbfs: float) -> None:
        """The output peak for the meter, when it changed (silence: reported once)."""
        if dbfs != self._last_peak:
            self._last_peak = dbfs
            self.latest_peak.publish(dbfs)

    def _emit_peak(self) -> None:
        if self._peak_meter_gain_linear <= 1e-9:
            self._publish_peak(self.PEAK_FLOOR_DBFS)
            return
        adjusted = self.feeder.heard_peak_dbfs() + 20.0 * math.log10(self._peak_meter_gain_linear)
        self._publish_peak(max(self.PEAK_FLOOR_DBFS, min(0.0, adjusted)))

    @QtCore.Slot()
    def _tick(self) -> None:
        poll = getattr(self.feeder, "poll", None)   # the native engine's state, read once a frame
        if poll is not None:
            poll()
        self.deck.tick()
        self._report_deck()
        cur = (self.feeder.display_frame() * 1000.0) / max(1, self.rate)
        if self._duration_ms and cur >= self._duration_ms:
            cur = self._duration_ms
        self._emit_time(cur)
        self._emit_peak()
        now = time.monotonic()
        if now - self._buffer_reported >= self.BUFFER_REPORT_SEC:
            self._buffer_reported = now
            queued, hist, capacity = self.feeder.take_buffer_stats()
            ms = 1000.0 / max(1, self.rate)
            self.output_buffer.emit(queued * ms, hist, capacity * ms)

    def _report_deck(self) -> None:
        """The deck's report to the GUI when it changed (the play state also as playback_status), and the MIDI
        messages it wants."""
        report: Optional[DeckReport] = self.deck.deck_report()
        if report is not None:
            if report.playing != self._is_playing:
                self._is_playing = report.playing
                self.playback_status.emit(report.playing)
            key = report.key()
            if key != self._deck_key:
                self._deck_key = key
                self.latest_deck.publish(report)
        if isinstance(self.feeder, NativeFeeder):
            events = self.feeder.midi_events()
            if not self._midi_learn:
                events = [e for e in events if e[3] & (MIDI_EVENT_GUI | MIDI_EVENT_HOTCUE_SET)]
            if events:
                self.midi_events.emit(events)


def channel_pair(text: str) -> tuple[int, int] | None:
    """A setting's channel pair ("3-4", 1-based) as the engine's (2, 3); "" or malformed: None."""
    try:
        left, right = (int(v) - 1 for v in str(text or "").split("-"))
    except ValueError:
        return None
    return (left, right) if 0 <= left < 64 and 0 <= right < 64 else None


def qt_output_device(name: str) -> QtMultimedia.QAudioDevice:
    """Qt's output device described ``name`` ("" or one not found: the system's default)."""
    if name:
        for dev in QtMultimedia.QMediaDevices.audioOutputs():
            if dev.description() == name:
                return dev
    return QtMultimedia.QMediaDevices.defaultAudioOutput()


class PlayerController(QtCore.QObject):
    """
    Main-thread facade of the player: forwards commands to the audio-thread worker and
    relays its state to the event bus. External sync mode disables local playback (and the
    MIDI controller). The deck's report reaches the bus as sig_deck_state (and the tempo, when it
    changed, as sig_tempo_factor_changed); the GUI shows it rather than what it asked for.
    """

    _command = QtCore.Signal(str, object)  # worker method name, args tuple
    SCRUB_INTERVAL_MS = 30

    def __init__(
        self,
        bus: EventBus,
        model=None,
        *,
        volume_position: float = 1.0,
        volume_trim_dbfs: float = 0.0,
        tempo_range: int = 100,
        default_peak_meter_gain_linear: float = 1.0,
        engine: str = "native",
        native_buffer_ms: float = 20.0,
        python_buffer_ms: float = 80.0,
        output_device: str = "",
        native_api: str = "wasapi",
        native_exclusive: bool = False,
        native_exclusive_rate: int = 0,
    ):
        super().__init__()
        self.bus = bus
        self.model = model
        self._path: Optional[str] = None
        self._external_sync_enabled = False
        self._midi_port = ""   # the MIDI input asked for (closed while External Sync follows another deck)
        self._factor: Optional[float] = None   # the tempo last reported

        # the audio engine (Settings > Audio > Engine): the native one (audioengine/native), started here so the app
        # knows which plays, or where it is not built or does not start the Python one (audioengine/python);
        # "python": the Python one only
        native_engine = (NativeEngine.start(output_device, native_api, native_exclusive,
                                            native_exclusive_rate if native_exclusive else 0)
                         if engine == "native" else None)
        self.native = native_engine is not None
        self._engine = native_engine
        self._audio_thread = QtCore.QThread(self)
        self._audio_thread.setObjectName("AudioThread")
        self._audio_worker = _AudioWorker(
            volume_position=volume_position,
            volume_trim_dbfs=volume_trim_dbfs,
            tempo_range=tempo_range,
            peak_meter_gain_linear=default_peak_meter_gain_linear,
            buffer_ms=native_buffer_ms if self.native else python_buffer_ms,
            engine=native_engine,
            output_device=output_device,
        )
        self._audio_worker._output_route = (native_api, bool(native_exclusive))
        feeder_route = (output_device, native_api, bool(native_exclusive), int(native_exclusive_rate))
        self._audio_worker._native_route = feeder_route
        self._audio_worker.moveToThread(self._audio_thread)
        self._command.connect(self._audio_worker.run_command, QtCore.Qt.QueuedConnection)

        w = self._audio_worker
        w.time_ready.connect(self._relay_time, QtCore.Qt.QueuedConnection)
        w.duration_changed.connect(bus.sig_duration_changed)
        w.playback_status.connect(bus.sig_playback_status)
        w.transport_enabled.connect(bus.sig_transport_enabled)
        w.peak_ready.connect(self._relay_peak, QtCore.Qt.QueuedConnection)
        w.output_buffer.connect(bus.sig_output_buffer)
        w.engine_restarted.connect(bus.sig_engine_restarted)
        w.deck_ready.connect(self._relay_deck, QtCore.Qt.QueuedConnection)
        w.midi_events.connect(bus.sig_midi_events)
        w.track_loaded.connect(self._on_track_loaded, QtCore.Qt.QueuedConnection)

        self._audio_thread.started.connect(w.initialize)
        self._audio_thread.finished.connect(w.deleteLater)
        # time-critical: the feed ticks must not wait behind busy threads (an analysis on every core), or a
        # small output buffer runs dry
        self._audio_thread.start(QtCore.QThread.Priority.TimeCriticalPriority)

        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

        bus.sig_seek_requested.connect(self.seek)
        # a scrub's positions reach the audio at most every SCRUB_INTERVAL_MS (the newest; the views follow the
        # drag itself): each one is a seek, crossfaded, which a mouse moving at its own rate turns into a rasp
        self._scrub_pending: Optional[float] = None
        self._scrub_sent: Optional[float] = None   # the position last sent
        self._scrub_timer = QtCore.QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(self.SCRUB_INTERVAL_MS)
        self._scrub_timer.timeout.connect(self._send_scrub)
        bus.sig_scrub_begin.connect(self._on_scrub_begin)
        bus.sig_scrub_update.connect(self._on_scrub_update)
        bus.sig_scrub_end.connect(self._on_scrub_end)
        bus.sig_scratch_begin.connect(lambda sec_per_px: self._send_local("scratch_begin", float(sec_per_px)))
        bus.sig_scratch_update.connect(lambda sec, at: self._send_local("scratch_update", float(sec), float(at)))
        bus.sig_scratch_end.connect(lambda sec: self._send_local("scratch_end", float(sec)))
        bus.sig_peak_meter_gain_changed.connect(lambda v: self._send("set_peak_meter_gain", float(v)))
        bus.sig_tone_hold.connect(lambda notes: self._send("tone_hold", tuple(notes or ())))
        # the deck's requests (what it then is: sig_deck_state)
        bus.sig_deck_action.connect(lambda code, pressed: self._send_local("deck_action", int(code), bool(pressed)))
        bus.sig_tempo_requested.connect(lambda f: self._send("set_tempo", float(f)))
        bus.sig_tempo_range_requested.connect(lambda r: self._send("set_tempo_range", int(r)))
        bus.sig_volume_requested.connect(lambda v: self._send("set_volume", float(v)))
        bus.sig_knob_requested.connect(lambda which, v: self._send("set_knob", int(which), float(v)))
        bus.sig_mixer_requested.connect(lambda key, v: self._send("set_mixer", str(key), float(v)))
        bus.sig_jog_mode_requested.connect(self.set_jog_mode)
        bus.sig_channel_cue_requested.connect(lambda on: self._send("set_channel_cue", bool(on)))
        bus.sig_engine_restart_requested.connect(lambda: self._send("restart_engine"))
        bus.sig_jump_to_requested.connect(lambda sec: self._send_local("jump_to", float(sec)))
        bus.sig_call_requested.connect(lambda sec: self._send_local("call_at", float(sec)))
        bus.sig_deck_marks.connect(lambda marks: self._send("set_marks", marks))
        bus.sig_external_sync_enabled.connect(self._on_external_sync_enabled)

    def _on_scrub_begin(self) -> None:
        self._scrub_pending = None
        self._scrub_sent = None
        self._send("scrub_begin")

    def _on_scrub_update(self, sec: float) -> None:
        self._scrub_pending = float(sec)
        if not self._scrub_timer.isActive():
            self._send_scrub()

    def _send_scrub(self) -> None:
        """The newest scrub position (if any came since the last), then a pause of SCRUB_INTERVAL_MS."""
        if self._scrub_pending is None:
            return
        self._send("scrub_update", self._scrub_pending)
        self._scrub_sent = self._scrub_pending
        self._scrub_pending = None
        self._scrub_timer.start()

    def _on_scrub_end(self, sec: float) -> None:
        """Let go: where, unless that is the position last sent (held still there, it plays on from it: a seek
        back to it would start it again)."""
        self._scrub_timer.stop()
        self._scrub_pending = None
        last = self._scrub_sent
        if last is None or abs(float(sec) - last) > 1e-6:
            self._send("scrub_update", float(sec))
        self._send("scrub_end", float(sec))

    @QtCore.Slot()
    def _relay_time(self) -> None:
        sec = self._audio_worker.latest_time.take()
        if sec is not None:
            self.bus.sig_time_changed.emit(float(sec))

    @QtCore.Slot()
    def _relay_deck(self) -> None:
        report = self._audio_worker.latest_deck.take()
        if report is None:
            return
        self.bus.sig_deck_state.emit(report)
        if report.factor != self._factor:
            self._factor = report.factor
            self.bus.sig_tempo_factor_changed.emit(report.factor)

    @QtCore.Slot()
    def _relay_peak(self) -> None:
        dbfs = self._audio_worker.latest_peak.take()
        if dbfs is not None:
            self.bus.sig_output_peak_dbfs.emit(float(dbfs))

    def _send(self, name: str, *args) -> None:
        self._command.emit(name, args)

    def _send_local(self, name: str, *args) -> None:
        """A local deck command; external sync follows the external deck instead."""
        if not self._external_sync_enabled:
            self._send(name, *args)

    # Public API
    def set_source(self, path: str) -> None:
        self._path = path
        self._send("load", path)

    def get_source(self) -> Optional[str]:
        return self._path

    def unload(self) -> None:
        self._path = None
        self._send("unload")

    def audio_thread(self) -> QtCore.QThread:
        return self._audio_thread

    def click_mixer(self) -> Optional[ClickTrack | NativeClicks]:
        """The metronome click mixer. Only touch it from the audio thread."""
        feeder = self._audio_worker.feeder
        return feeder.clicks if feeder is not None else None

    def play(self) -> None:
        if not self._path:
            return
        if self._external_sync_enabled:
            self.bus.sig_playback_status.emit(False)
            return
        self._send("play")

    def pause(self) -> None:
        if self._external_sync_enabled:
            self.bus.sig_playback_status.emit(False)
            return
        self._send("pause")

    def stop(self) -> None:
        if self._external_sync_enabled:
            self.bus.sig_playback_status.emit(False)
            self.bus.sig_time_changed.emit(0.0)
            return
        self._send("stop")

    def seek(self, sec: float) -> None:
        if self._external_sync_enabled:
            self.bus.sig_time_changed.emit(max(0.0, float(sec)))
            return
        self._send("seek", float(sec))

    def set_volume_trim(self, db: float) -> None:
        """The volume fader's trim (dB; Settings > Audio > Mixer)."""
        self._send("set_volume_trim", float(db))

    def set_jog_mode(self, vinyl: bool) -> None:
        """The jog wheel's mode: VINYL (its top holds the record) or CDJ (a bend only); the native engine's."""
        self._send("midi_options", None, bool(vinyl))

    # The MIDI controller: the native engine reads it (never this process); none on the Python engine
    def midi_configure(self, port: str, bindings: list, jog_ticks: int) -> None:
        """Read MIDI input ``port`` ("": none) with ``bindings`` ((status, data1, 14-bit, action code) each)."""
        if not self.native:
            return
        self._midi_port = str(port or "")
        self._send("midi_options", int(jog_ticks), None)
        self._send("midi_map", list(bindings))
        self._send("midi_open", "" if self._external_sync_enabled else self._midi_port)

    def set_midi_learn(self, learning: bool) -> None:
        """Every MIDI message to sig_midi_events (Settings > MIDI's learn), not only the GUI's actions'."""
        self._send("set_midi_learn", bool(learning))

    def midi_ports(self) -> list[str]:
        """The MIDI input ports as the native engine finds them ([] on the Python engine)."""
        feeder = self._audio_worker.feeder
        return feeder.midi_ports() if isinstance(feeder, NativeFeeder) else []

    def set_timestretch(self, enabled: bool) -> None:
        """Pitch-preserving time stretch instead of varispeed for tempo changes."""
        self._send("set_timestretch", bool(enabled))

    def set_soft_clip(self, enabled: bool) -> None:
        """tanh soft clip of the output (music + metronome clicks)."""
        self._send("set_soft_clip", bool(enabled))

    def set_buffer_ms(self, native_ms: float, python_ms: float) -> None:
        """The output buffer (ms) of each engine: the one that plays takes its own."""
        self._send("set_buffer_ms", float(native_ms if self.native else python_ms))

    def set_refresh_fps(self, fps: int) -> None:
        self._send("set_refresh_fps", int(fps))

    def set_output_device(self, name: str) -> None:
        """The output device by name ("": the system's default)."""
        self._send("set_output_device", str(name or ""))

    def set_output_mode(self, api: str, exclusive: bool) -> None:
        """The native engine's audio API and WASAPI's exclusive mode; the Python engine has neither."""
        self._send("set_output_mode", str(api or "wasapi"), bool(exclusive))

    def set_exclusive_rate(self, rate: int) -> None:
        """The native engine's rate in exclusive mode (0: the device's), from the next track loaded on."""
        self._send("set_exclusive_rate", int(rate))

    def set_process(self, priority: str, power_throttling: bool) -> None:
        """The native engine's priority and power throttling (sched.rs: per system); the Python engine plays in
        the app's own process and has neither."""
        self._send("set_process", str(priority or "high"), bool(power_throttling))

    def set_routing(self, main: str, headphones, booth) -> None:
        """The native engine's outputs: the master on the main device's channel pair ``main`` ("1-2"), the
        headphones and the booth (config.outputconfig: off, channels of the main device, or a device of their
        own)."""
        self._send("set_routing", str(main or "1-2"), asdict(headphones), asdict(booth))

    def set_headphones(self, master: bool, mix: float, level: float) -> None:
        """The native engine's headphones: the CUE'd channels, and the master mixed in by ``mix`` (0 CUE .. 1
        MASTER) with ``master`` (off for a device that mixes it in itself), at ``level`` (a knob's position)."""
        self._send("set_headphones", bool(master), float(mix), float(level))

    def output_channels(self) -> int:
        """The output device's channels (the native engine's; the Python engine: 2)."""
        feeder = self._audio_worker.feeder
        return feeder.output_channels() if isinstance(feeder, NativeFeeder) else 2

    def output_devices(self, api: Optional[str] = None) -> tuple[str, list[str], dict[str, int]]:
        """(the output device in use, the output devices, their channels by name ("": the system's default's)) as
        the engine that plays names them (they differ by engine on some systems); the native engine's of ``api``
        (None: its main output's)."""
        feeder = self._audio_worker.feeder
        if isinstance(feeder, NativeFeeder):
            return feeder.output_devices(api)   # the engine it holds now (started again for another rate)
        # the Python engine: the device it was asked for (QAudioSink does not say which it opened)
        current = qt_output_device(self._audio_worker._device_name).description()
        devices = QtMultimedia.QMediaDevices.audioOutputs()
        channels = {d.description(): d.maximumChannelCount() for d in devices}
        return current, [d.description() for d in devices], channels

    def shutdown(self) -> None:
        if self._audio_thread.isRunning():
            try:
                QtCore.QMetaObject.invokeMethod(
                    self._audio_worker,
                    "shutdown",
                    QtCore.Qt.BlockingQueuedConnection,
                )
            except Exception:
                pass
            self._audio_thread.quit()
            self._audio_thread.wait(2000)

    @QtCore.Slot(bool)
    def _on_external_sync_enabled(self, enabled: bool) -> None:
        self._external_sync_enabled = bool(enabled)
        if self.native:
            self._send("midi_open", "" if self._external_sync_enabled else self._midi_port)
        if self._external_sync_enabled:
            self._send("pause")
            self.bus.sig_playback_status.emit(False)
            self.bus.sig_transport_enabled.emit(False)
        elif self._path:
            self._send("load", self._path)

    @QtCore.Slot(object, int)
    def _on_track_loaded(self, pcm: object, rate: int) -> None:
        if self.model is not None:
            self.model.set_predecoded_audio(pcm, rate)
        if self._external_sync_enabled:
            self.bus.sig_transport_enabled.emit(False)
            self.bus.sig_playback_status.emit(False)
