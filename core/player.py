from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtMultimedia

from core.audio.clicks import ClickTrack
from core.audio.decoder import decode_to_memmap, warm_ffmpeg_decoder
from core.audio.feeder import PCMFeeder
from core.event_bus import EventBus


class _AudioWorker(QtCore.QObject):
    """
    Owns the audio sink and feeder; lives in the player's audio thread.

    Every command from PlayerController arrives through run_command(). Decoding runs on a
    separate thread so the sink keeps being fed while a track loads; transport commands
    that arrive meanwhile are replayed, in order, once the track is ready.
    """

    time_changed = QtCore.Signal(float)
    duration_changed = QtCore.Signal(float)
    playback_status = QtCore.Signal(bool)
    transport_enabled = QtCore.Signal(bool)
    track_loaded = QtCore.Signal(object, int)  # pcm [N, ch] float32, sample rate
    output_peak_dbfs = QtCore.Signal(float)
    _decoded = QtCore.Signal(int, object, str)  # token, pcm or None, error message

    BUFFER_MS = 80
    PEAK_FLOOR_DBFS = -24.0

    _COMMANDS = frozenset({
        "load", "play", "pause", "stop", "seek",
        "scrub_begin", "scrub_update", "scrub_end", "arm_jump", "disarm_jump",
        "set_tempo_mode", "set_tempo_factor", "set_volume", "set_peak_meter_gain",
        "set_refresh_fps",
    })
    _WAIT_FOR_TRACK = frozenset({
        "play", "pause", "stop", "seek",
        "scrub_begin", "scrub_update", "scrub_end", "arm_jump", "disarm_jump",
    })

    def __init__(self, *, volume_linear: float = 1.0, peak_meter_gain_linear: float = 1.0) -> None:
        super().__init__()
        self.audio: Optional[QtMultimedia.QAudioSink] = None
        self.feeder: Optional[PCMFeeder] = None
        self.rate = 0
        self.ch = 0

        self._path: Optional[str] = None
        self._track_ready = False
        self._is_playing = False
        self._duration_ms = 0
        self._last_emit_ms: float = -1

        self._decode_token = 0
        self._decoding = False
        self._deferred: list[tuple[str, tuple]] = []

        self._volume_linear = max(0.0, min(1.0, float(volume_linear)))
        self._peak_meter_gain_linear = max(0.0, min(1.0, float(peak_meter_gain_linear)))

        self._frame = QtCore.QTimer(self)
        self._frame.setTimerType(QtCore.Qt.PreciseTimer)
        self._frame.setInterval(16)
        self._frame.timeout.connect(self._tick)
        self._decoded.connect(self._on_decoded, QtCore.Qt.QueuedConnection)

    @QtCore.Slot()
    def initialize(self) -> None:
        out_dev = QtMultimedia.QMediaDevices.defaultAudioOutput()
        fmt = out_dev.preferredFormat()
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.Float)
        self.rate = fmt.sampleRate()
        self.ch = fmt.channelCount()

        self.audio = QtMultimedia.QAudioSink(out_dev, fmt, self)
        bpf = self.ch * 4
        self.audio.setBufferSize(max(bpf, int(self.rate * (self.BUFFER_MS / 1000.0) * bpf)))

        self.feeder = PCMFeeder(self.audio, self.rate, self.ch, self)
        self.feeder.set_music_gain(self._volume_linear)
        self.feeder.finished.connect(self.pause)
        # While playing, every transport change (seek, scrub, jump) is rendered by the feeder
        # into the running sink: reset()/start() cycles during scrubbing crash Qt 6.10's WASAPI
        # backend. The feeder stops the sink only when playback is idle.
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
        self.pause()
        self.feeder.set_track(None)
        self._path = path
        self._track_ready = False
        self._duration_ms = 0
        self._last_emit_ms = -1
        self._deferred.clear()
        self.playback_status.emit(False)
        self.duration_changed.emit(0.0)
        self.time_changed.emit(0.0)
        self.output_peak_dbfs.emit(self.PEAK_FLOOR_DBFS)
        self.transport_enabled.emit(False)

        self._decode_token += 1
        self._decoding = True
        token, rate, ch = self._decode_token, self.rate, self.ch

        def decode() -> None:
            try:
                pcm, error = decode_to_memmap(path, rate, ch), ""
            except Exception as exc:
                pcm, error = None, str(exc)
            try:
                self._decoded.emit(token, pcm, error)
            except RuntimeError:
                pass  # worker already destroyed (app shutting down)

        threading.Thread(target=decode, name="PlayerDecode", daemon=True).start()

    @QtCore.Slot(int, object, str)
    def _on_decoded(self, token: int, pcm: object, error: str) -> None:
        if token != self._decode_token or self.feeder is None:
            return  # a newer track was requested meanwhile
        self._decoding = False
        try:
            if pcm is None:
                raise RuntimeError(error)
            pcm_array = np.asarray(pcm, dtype=np.float32, order="C")
            self.feeder.set_track(pcm_array)
        except Exception as exc:
            print(f"[Player] Decode error: {exc}")
            self._deferred.clear()
            self.playback_status.emit(False)
            self.output_peak_dbfs.emit(self.PEAK_FLOOR_DBFS)
            self.transport_enabled.emit(True)
            return

        self._track_ready = True
        self._duration_ms = int(round((pcm_array.shape[0] * 1000.0) / max(1, self.rate)))
        self.track_loaded.emit(pcm_array, self.rate)
        self.duration_changed.emit(self._duration_ms / 1000.0)
        self.time_changed.emit(0.0)
        self.playback_status.emit(False)
        self.transport_enabled.emit(True)
        self.output_peak_dbfs.emit(self.PEAK_FLOOR_DBFS)

        deferred, self._deferred = self._deferred, []
        for name, args in deferred:
            self.run_command(name, args)

    # Transport
    @QtCore.Slot()
    def play(self) -> None:
        if not self._path or self._is_playing or not self._track_ready:
            return
        if not self.feeder.play():
            return
        self._is_playing = True
        self.playback_status.emit(True)

    @QtCore.Slot()
    def pause(self) -> None:
        if not self._is_playing:
            return
        self.feeder.pause()
        self._is_playing = False
        self.playback_status.emit(False)
        self._emit_time(self._track_now_ms())

    def stop(self) -> None:
        self.feeder.pause()
        self.feeder.seek(0)
        self._is_playing = False
        self._last_emit_ms = -1
        self.playback_status.emit(False)
        self._emit_time(0)

    def seek(self, sec: float) -> None:
        if not self._track_ready:
            return
        self.feeder.seek(int(round(max(0.0, sec) * self.rate)))
        if not self._is_playing:
            self.time_changed.emit(sec)

    def scrub_begin(self) -> None:
        self.feeder.set_scrubbing(True)

    def scrub_update(self, sec: float) -> None:
        if not self._track_ready:
            return
        self.feeder.seek(int(round(sec * self.rate)))
        if not self._is_playing:
            self.time_changed.emit(sec)

    def scrub_end(self, _sec: float) -> None:
        self.feeder.set_scrubbing(False)

    def arm_jump(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        jump_start = None
        jump_dest = None
        source = payload.get("source")
        target = payload.get("target")
        if isinstance(source, dict) and isinstance(target, dict):
            jump_start = source.get("point", source.get("start"))
            jump_dest = target.get("point", target.get("start"))
        else:
            label = payload["label"]
            cue = payload["cue"]
            if cue["forward"]["label"] == label:
                jump_start = cue["forward"]["point"]
                jump_dest = cue["backward"]["point"]
            elif cue["backward"]["label"] == label:
                jump_start = cue["backward"]["point"]
                jump_dest = cue["forward"]["point"]
        self.feeder.arm_jump(jump_start, jump_dest)

    def disarm_jump(self) -> None:
        self.feeder.disarm_jump()

    # Settings
    def set_tempo_mode(self, mode: str) -> None:
        self.feeder.set_mode(mode)

    def set_tempo_factor(self, factor: float) -> None:
        self.feeder.set_factor(float(factor))

    def set_volume(self, value: float) -> None:
        # Applied to the music inside the feeder so metronome clicks keep their own level.
        self._volume_linear = max(0.0, min(1.0, float(value)))
        self.feeder.set_music_gain(self._volume_linear)
        self._emit_peak()

    def set_peak_meter_gain(self, value: float) -> None:
        self._peak_meter_gain_linear = max(0.0, min(1.0, float(value)))
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
        self._track_ready = False
        self.output_peak_dbfs.emit(self.PEAK_FLOOR_DBFS)
        self.transport_enabled.emit(False)
        self._frame.stop()
        self.feeder.close()

    # Periodic time / peak reporting
    def _track_now_ms(self) -> float:
        return (self.feeder.playhead_frame() * 1000.0) / max(1, self.rate)

    def _emit_time(self, ms: float) -> None:
        if ms != self._last_emit_ms:
            self._last_emit_ms = ms
            self.time_changed.emit(ms / 1000.0)

    def _emit_peak(self) -> None:
        if self._peak_meter_gain_linear <= 1e-9:
            self.output_peak_dbfs.emit(self.PEAK_FLOOR_DBFS)
            return
        adjusted = self.feeder.heard_peak_dbfs() + 20.0 * math.log10(self._peak_meter_gain_linear)
        self.output_peak_dbfs.emit(max(self.PEAK_FLOOR_DBFS, min(0.0, adjusted)))

    @QtCore.Slot()
    def _tick(self) -> None:
        cur = self._track_now_ms()
        if self._duration_ms and cur >= self._duration_ms:
            cur = self._duration_ms
        self._emit_time(cur)
        self._emit_peak()


class PlayerController(QtCore.QObject):
    """
    Main-thread facade of the player: forwards commands to the audio-thread worker and
    relays its state to the event bus. External sync mode disables local playback.
    """

    _command = QtCore.Signal(str, object)  # worker method name, args tuple

    def __init__(
        self,
        bus: EventBus,
        model=None,
        *,
        default_volume_linear: float = 1.0,
        default_peak_meter_gain_linear: float = 1.0,
    ):
        super().__init__()
        self.bus = bus
        self.model = model
        self._path: Optional[str] = None
        self._external_sync_enabled = False

        self._audio_thread = QtCore.QThread(self)
        self._audio_thread.setObjectName("AudioThread")
        self._audio_worker = _AudioWorker(
            volume_linear=default_volume_linear,
            peak_meter_gain_linear=default_peak_meter_gain_linear,
        )
        self._audio_worker.moveToThread(self._audio_thread)
        self._command.connect(self._audio_worker.run_command, QtCore.Qt.QueuedConnection)

        w = self._audio_worker
        w.time_changed.connect(bus.sig_time_changed)
        w.duration_changed.connect(bus.sig_duration_changed)
        w.playback_status.connect(bus.sig_playback_status)
        w.transport_enabled.connect(bus.sig_transport_enabled)
        w.output_peak_dbfs.connect(bus.sig_output_peak_dbfs)
        w.track_loaded.connect(self._on_track_loaded, QtCore.Qt.QueuedConnection)

        self._audio_thread.started.connect(w.initialize)
        self._audio_thread.finished.connect(w.deleteLater)
        self._audio_thread.start()
        threading.Thread(target=warm_ffmpeg_decoder, name="FFmpegWarmup", daemon=True).start()

        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

        bus.sig_seek_requested.connect(self.seek)
        bus.sig_scrub_begin.connect(lambda: self._send("scrub_begin"))
        bus.sig_scrub_update.connect(lambda sec: self._send("scrub_update", float(sec)))
        bus.sig_scrub_end.connect(lambda sec: self._send("scrub_end", float(sec)))
        bus.sig_volume_changed.connect(lambda v: self._send("set_volume", float(v)))
        bus.sig_peak_meter_gain_changed.connect(lambda v: self._send("set_peak_meter_gain", float(v)))
        bus.sig_jump_arm.connect(lambda payload: self._send("arm_jump", payload))
        bus.sig_jump_disarm.connect(lambda: self._send("disarm_jump"))
        bus.sig_tempo_mode_changed.connect(self.set_tempo_mode)
        bus.sig_tempo_factor_changed.connect(self.set_tempo_factor)
        bus.sig_external_sync_enabled.connect(self._on_external_sync_enabled)

    def _send(self, name: str, *args) -> None:
        self._command.emit(name, args)

    # Public API
    def set_source(self, path: str) -> None:
        self._path = path
        self._send("load", path)

    def get_source(self) -> Optional[str]:
        return self._path

    def audio_thread(self) -> QtCore.QThread:
        return self._audio_thread

    def click_mixer(self) -> Optional[ClickTrack]:
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

    def set_tempo_mode(self, mode: str) -> None:
        self._send("set_tempo_mode", mode)

    def set_tempo_factor(self, factor: float) -> None:
        self._send("set_tempo_factor", float(factor))

    def set_refresh_fps(self, fps: int) -> None:
        self._send("set_refresh_fps", int(fps))

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
