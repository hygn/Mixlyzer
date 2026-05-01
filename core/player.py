from __future__ import annotations

import math
from typing import Optional

import numpy as np
import threading
from PySide6 import QtCore, QtMultimedia, QtGui

from core.audio.decoder import decode_to_memmap, warm_ffmpeg_decoder
from core.audio.feeder import PCMFeeder
from core.event_bus import EventBus
from pymediainfo import MediaInfo
import base64


def _thread_tag(obj: QtCore.QObject | None = None) -> str:
    qt_thread = QtCore.QThread.currentThread()
    qt_name = qt_thread.objectName() if qt_thread is not None else ""
    if not qt_name:
        qt_name = f"qt@{id(qt_thread):x}" if qt_thread is not None else "qt@none"
    affinity = ""
    if obj is not None:
        try:
            obj_thread = obj.thread()
            obj_name = obj_thread.objectName() if obj_thread is not None else ""
            if not obj_name:
                obj_name = f"qt@{id(obj_thread):x}" if obj_thread is not None else "qt@none"
            affinity = f" affinity={obj_name}"
        except Exception:
            affinity = ""
    return f"py={threading.get_ident()} cur={qt_name}{affinity}"


class _AudioWorker(QtCore.QObject):
    time_changed = QtCore.Signal(float)
    duration_changed = QtCore.Signal(float)
    playback_status = QtCore.Signal(bool)
    transport_enabled = QtCore.Signal(bool)
    predecoded_buffer_ready = QtCore.Signal(object, int)
    output_peak_dbfs = QtCore.Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.fmt: Optional[QtMultimedia.QAudioFormat] = None
        self.audio: Optional[QtMultimedia.QAudioSink] = None
        self.rate: int = 0
        self.ch: int = 0
        self._feeder: Optional[PCMFeeder] = None

        self._path: Optional[str] = None
        self._is_playing = False
        self._last_emit_ms = -1
        self._duration_ms = 0
        self._buffer_ready = False
        self._desired_buffer_ms: int = 80

        self._scrubbing = False
        self._saved_buffer_ms = None
        self._saved_chunk_frames = None
        self._decode_token = 0
        self._volume_linear = 10.0 ** (-6.0 / 20.0)
        self._last_peak_pre_volume_dbfs = -24.0

        self._frame = QtCore.QTimer(self)
        self._frame.setTimerType(QtCore.Qt.PreciseTimer)
        self._frame.setInterval(16)
        self._frame.timeout.connect(self._tick_time)

    @QtCore.Slot()
    def initialize(self) -> None:
        out_dev = QtMultimedia.QMediaDevices.defaultAudioOutput()
        fmt = out_dev.preferredFormat()
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.Float)
        self.fmt = fmt
        self.audio = QtMultimedia.QAudioSink(out_dev, fmt, self)
        self.audio.setVolume(self._volume_linear)

        self.rate = fmt.sampleRate()
        self.ch = fmt.channelCount()
        self._feeder = PCMFeeder(self.audio, self.rate, self.ch, self)
        self._feeder.finished.connect(self._on_feeder_finished)
        self._feeder.peak_level.connect(self._on_peak_level_pre_volume)
        self._warm_backend()

    @QtCore.Slot(str)
    def prepare_source(self, path: str) -> None:
        self.pause()
        self._path = path
        self._buffer_ready = False
        self._duration_ms = 0
        self._last_emit_ms = -1
        self.playback_status.emit(False)
        self.duration_changed.emit(0.0)
        self.time_changed.emit(0.0)
        self.output_peak_dbfs.emit(-24.0)
        self.transport_enabled.emit(False)
        self._decode_token += 1
        try:
            pcm = decode_to_memmap(path, self.rate, self.ch)
            pcm_array = np.array(pcm, dtype=np.float32, order="C", copy=False)
            duration_ms = int(round((pcm_array.shape[0] * 1000.0) / max(1, self.rate)))
            self.set_predecoded_buffer(pcm_array, duration_ms)
        except Exception as exc:
            print(f"[Player] Decode error: {exc}")
            self.playback_status.emit(False)
            self.output_peak_dbfs.emit(-24.0)
            self.transport_enabled.emit(True)

    @QtCore.Slot(object, int)
    def set_predecoded_buffer(self, pcm: object, duration_ms: int) -> None:
        if self._feeder is None:
            return
        try:
            pcm_array = np.asarray(pcm, dtype=np.float32, order="C")
            self._feeder.set_predecoded_buffer(pcm_array)
        except Exception as exc:
            print(f"[Player] Failed to load decoded buffer: {exc}")
            self._buffer_ready = False
            self.output_peak_dbfs.emit(-24.0)
            self.transport_enabled.emit(True)
            return

        self._buffer_ready = True
        self._duration_ms = int(duration_ms)
        self.predecoded_buffer_ready.emit(pcm_array, self._duration_ms)
        self.duration_changed.emit(self._duration_ms / 1000.0)
        self.time_changed.emit(0.0)
        self.playback_status.emit(False)
        self.transport_enabled.emit(True)
        self.output_peak_dbfs.emit(-24.0)

    @QtCore.Slot()
    def play(self) -> None:
        if not self._path or self._is_playing or not self._buffer_ready or self._feeder is None:
            return
        self._apply_buffer_ms()
        self._feeder.start()
        self._is_playing = True
        self._frame.start()
        self.playback_status.emit(True)

    @QtCore.Slot()
    def pause(self) -> None:
        if self.audio is None or self._feeder is None:
            return
        if not self._is_playing:
            try:
                self.audio.reset()
            except Exception:
                pass
            self._frame.stop()
            return

        self._apply_buffer_ms()
        cur_abs_frame = self._feeder.input_playhead_abs_frame()
        cur_ms = int(round((cur_abs_frame * 1000.0) / max(1, self.rate)))

        try:
            self.audio.reset()
        except Exception:
            pass
        self._frame.stop()
        self._feeder.stop()

        self._is_playing = False
        self.playback_status.emit(False)
        self.output_peak_dbfs.emit(-24.0)

        self._last_emit_ms = -1
        self._emit_time(cur_ms)

    @QtCore.Slot()
    def stop(self) -> None:
        if self.audio is None or self._feeder is None:
            return
        self._apply_buffer_ms()
        self._frame.stop()
        try:
            self.audio.reset()
        except Exception:
            pass
        try:
            self._feeder.stop()
        except Exception:
            pass

        try:
            self._feeder.seek_frames(0)
            self._feeder.reset_counters()
        except Exception:
            pass

        self._is_playing = False
        self._last_emit_ms = -1
        self.playback_status.emit(False)
        self.output_peak_dbfs.emit(-24.0)
        self._emit_time(0)

    @QtCore.Slot(float)
    def seek(self, sec: float) -> None:
        if not self._buffer_ready or self._feeder is None:
            return
        frames = int(round(max(0.0, sec) * self.rate))
        self._feeder.seek_frames(frames)
        if self._is_playing:
            if self.audio is not None:
                try:
                    self.audio.reset()
                except Exception:
                    pass
            self._feeder.reset_counters()
            self._feeder.start()
        else:
            self.time_changed.emit(sec)

    @QtCore.Slot(str)
    def set_tempo_mode(self, mode: str) -> None:
        if self._feeder is None:
            return
        m = (mode or "speed").lower()
        self._feeder.set_mode(m if m in ("none", "speed") else "speed")

    @QtCore.Slot(float)
    def set_tempo_factor(self, factor: float) -> None:
        if self._feeder is None:
            return
        self._feeder.set_factor(float(factor))

    @QtCore.Slot(float)
    def set_volume(self, value: float) -> None:
        self._volume_linear = max(0.0, min(1.0, float(value)))
        if self.audio is None:
            self._emit_adjusted_peak()
            return
        self.audio.setVolume(self._volume_linear)
        self._emit_adjusted_peak()

    @QtCore.Slot(int)
    def set_refresh_fps(self, fps: int) -> None:
        fps = max(1, min(240, int(fps)))
        interval_ms = max(1, int(round(1000.0 / fps)))
        self._frame.setInterval(interval_ms)
        if self._feeder is not None:
            self._feeder.set_peak_emit_interval_ms(interval_ms)

    @QtCore.Slot()
    def scrub_begin(self) -> None:
        if self._scrubbing or self.audio is None or self._feeder is None:
            return
        self._scrubbing = True
        try:
            self.audio.reset()
        except Exception:
            pass
        self._feeder.reset_counters()

        if self._saved_buffer_ms is None:
            self._saved_buffer_ms = self._desired_buffer_ms
        if self._saved_chunk_frames is None:
            self._saved_chunk_frames = self._feeder._chunk_frames

        if self._is_playing:
            self._feeder.start()

    @QtCore.Slot(float)
    def scrub_update(self, sec: float) -> None:
        if not self._buffer_ready or self._feeder is None:
            return
        frames = int(round(sec * self.rate))
        self._feeder.seek_frames(frames)
        if not self._is_playing:
            self.time_changed.emit(sec)

    @QtCore.Slot(float)
    def scrub_end(self, _sec: float) -> None:
        if not self._is_playing or self._feeder is None:
            return
        if self._saved_buffer_ms is not None:
            self._desired_buffer_ms = self._saved_buffer_ms
            self._saved_buffer_ms = None
        if self._saved_chunk_frames is not None:
            self._feeder._chunk_frames = self._saved_chunk_frames
            self._saved_chunk_frames = None
        self._scrubbing = False

    @QtCore.Slot(object)
    def arm_jump(self, payload: object) -> None:
        if self._feeder is None or not isinstance(payload, dict):
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
        self._feeder.arm_jump(jump_start, jump_dest)

    @QtCore.Slot()
    def disarm_jump(self) -> None:
        if self._feeder is not None:
            self._feeder.disarm_jump()

    @QtCore.Slot()
    def shutdown(self) -> None:
        self.stop()
        self._buffer_ready = False
        self.output_peak_dbfs.emit(-24.0)
        self.transport_enabled.emit(False)
        if self.audio is not None:
            try:
                self.audio.stop()
            except Exception:
                pass

    def _apply_buffer_ms(self) -> None:
        if self.audio is None:
            return
        bpf = self.ch * 4
        buf_bytes = int(self.rate * (self._desired_buffer_ms / 1000.0) * bpf)
        try:
            self.audio.setBufferSize(max(bpf, buf_bytes))
        except Exception:
            pass

    def _warm_backend(self) -> None:
        """
        Force Qt Multimedia to open the output path during app startup so the
        first real track load does not pay the cold-start penalty.
        """
        if self.audio is None:
            return
        bpf = max(1, self.ch * 4)
        warm_frames = max(1, min(1024, self.rate // 20 if self.rate > 0 else 1024))
        silence = np.zeros((warm_frames, self.ch), dtype=np.float32)
        dev = None
        try:
            self._apply_buffer_ms()
            dev = self.audio.start()
            if dev is not None:
                dev.write(silence.astype("<f4", copy=False).tobytes())
        except Exception:
            pass
        finally:
            try:
                self.audio.reset()
            except Exception:
                pass

    def _track_now_ms(self) -> float:
        if self._feeder is None:
            return 0.0
        frame = (
            self._feeder.input_playhead_abs_frame()
            if self._is_playing
            else self._feeder.current_base_frame()
        )
        return (frame * 1000.0) / max(1, self.rate)

    def _emit_time(self, ms: int) -> None:
        if ms != self._last_emit_ms:
            self._last_emit_ms = ms
            self.time_changed.emit(ms / 1000.0)

    @QtCore.Slot()
    def _tick_time(self) -> None:
        cur = self._track_now_ms()
        if self._duration_ms and cur >= self._duration_ms:
            cur = self._duration_ms
            self._emit_time(cur)
            self.pause()
            return
        self._emit_time(cur)

    @QtCore.Slot(int)
    def _on_feeder_finished(self, _code: int) -> None:
        self.pause()

    @QtCore.Slot(float)
    def _on_peak_level_pre_volume(self, dbfs: float) -> None:
        self._last_peak_pre_volume_dbfs = float(dbfs)
        self._emit_adjusted_peak()

    def _emit_adjusted_peak(self) -> None:
        floor_dbfs = -24.0
        if self._volume_linear <= 1e-9:
            self.output_peak_dbfs.emit(floor_dbfs)
            return
        adjusted = float(self._last_peak_pre_volume_dbfs) + (20.0 * math.log10(self._volume_linear))
        adjusted = max(floor_dbfs, min(0.0, adjusted))
        self.output_peak_dbfs.emit(adjusted)


class PlayerController(QtCore.QObject):
    """
    Audio controller that pre-decodes tracks and feeds them to QAudioSink.
    """

    _cmd_prepare_source = QtCore.Signal(str)
    _cmd_play = QtCore.Signal()
    _cmd_pause = QtCore.Signal()
    _cmd_stop = QtCore.Signal()
    _cmd_seek = QtCore.Signal(float)
    _cmd_set_tempo_mode = QtCore.Signal(str)
    _cmd_set_tempo_factor = QtCore.Signal(float)
    _cmd_set_volume = QtCore.Signal(float)
    _cmd_set_refresh_fps = QtCore.Signal(int)
    _cmd_scrub_begin = QtCore.Signal()
    _cmd_scrub_update = QtCore.Signal(float)
    _cmd_scrub_end = QtCore.Signal(float)
    _cmd_arm_jump = QtCore.Signal(object)
    _cmd_disarm_jump = QtCore.Signal()
    _cmd_shutdown = QtCore.Signal()

    def __init__(self, bus: EventBus, model=None):
        super().__init__()
        self.bus = bus
        self.model = model
        out_dev = QtMultimedia.QMediaDevices.defaultAudioOutput()
        fmt = out_dev.preferredFormat()
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.Float)
        self.rate = fmt.sampleRate()
        self.ch = fmt.channelCount()

        self._path: Optional[str] = None
        self._external_sync_enabled = False

        self._audio_thread = QtCore.QThread(self)
        self._audio_thread.setObjectName("AudioThread")
        self._audio_worker = _AudioWorker()
        self._audio_worker.moveToThread(self._audio_thread)

        self._cmd_prepare_source.connect(self._audio_worker.prepare_source, QtCore.Qt.QueuedConnection)
        self._cmd_play.connect(self._audio_worker.play, QtCore.Qt.QueuedConnection)
        self._cmd_pause.connect(self._audio_worker.pause, QtCore.Qt.QueuedConnection)
        self._cmd_stop.connect(self._audio_worker.stop, QtCore.Qt.QueuedConnection)
        self._cmd_seek.connect(self._audio_worker.seek, QtCore.Qt.QueuedConnection)
        self._cmd_set_tempo_mode.connect(self._audio_worker.set_tempo_mode, QtCore.Qt.QueuedConnection)
        self._cmd_set_tempo_factor.connect(self._audio_worker.set_tempo_factor, QtCore.Qt.QueuedConnection)
        self._cmd_set_volume.connect(self._audio_worker.set_volume, QtCore.Qt.QueuedConnection)
        self._cmd_set_refresh_fps.connect(self._audio_worker.set_refresh_fps, QtCore.Qt.QueuedConnection)
        self._cmd_scrub_begin.connect(self._audio_worker.scrub_begin, QtCore.Qt.QueuedConnection)
        self._cmd_scrub_update.connect(self._audio_worker.scrub_update, QtCore.Qt.QueuedConnection)
        self._cmd_scrub_end.connect(self._audio_worker.scrub_end, QtCore.Qt.QueuedConnection)
        self._cmd_arm_jump.connect(self._audio_worker.arm_jump, QtCore.Qt.QueuedConnection)
        self._cmd_disarm_jump.connect(self._audio_worker.disarm_jump, QtCore.Qt.QueuedConnection)
        self._cmd_shutdown.connect(self._audio_worker.shutdown, QtCore.Qt.QueuedConnection)

        self._audio_worker.time_changed.connect(self.bus.sig_time_changed)
        self._audio_worker.duration_changed.connect(self.bus.sig_duration_changed)
        self._audio_worker.playback_status.connect(self.bus.sig_playback_status)
        self._audio_worker.transport_enabled.connect(self.bus.sig_transport_enabled)
        self._audio_worker.output_peak_dbfs.connect(self.bus.sig_output_peak_dbfs)
        self._audio_worker.predecoded_buffer_ready.connect(self._on_predecoded_buffer_ready, QtCore.Qt.QueuedConnection)

        self._audio_thread.started.connect(self._audio_worker.initialize)
        self._audio_thread.finished.connect(self._audio_worker.deleteLater)
        self._audio_thread.start()
        threading.Thread(
            target=warm_ffmpeg_decoder,
            name="FFmpegWarmup",
            daemon=True,
        ).start()

        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

        # bus connections
        bus.sig_scrub_begin.connect(self._cmd_scrub_begin.emit)
        bus.sig_scrub_update.connect(self._cmd_scrub_update.emit)
        bus.sig_scrub_end.connect(self._cmd_scrub_end.emit)
        bus.sig_seek_requested.connect(self.seek)
        bus.sig_volume_changed.connect(self._cmd_set_volume.emit)
        bus.sig_jump_arm.connect(self._cmd_arm_jump.emit)
        bus.sig_jump_disarm.connect(self._cmd_disarm_jump.emit)

        try:
            bus.sig_tempo_mode_changed.connect(self.set_tempo_mode)
            bus.sig_tempo_factor_changed.connect(self.set_tempo_factor)
        except Exception:
            pass
        try:
            bus.sig_external_sync_enabled.connect(self._on_external_sync_enabled)
        except Exception:
            pass

    # public API
    def set_source(self, path: str):
        self._path = path
        self._cmd_prepare_source.emit(path)

    def get_source(self) -> Optional[str]:
        return self._path

    def audio_thread(self) -> QtCore.QThread:
        return self._audio_thread

    def connect_audio_time(self, slot) -> None:
        self._audio_worker.time_changed.connect(slot, QtCore.Qt.DirectConnection)

    def play(self):
        if not self._path:
            return
        if self._external_sync_enabled:
            self.bus.sig_playback_status.emit(False)
            return
        self._cmd_play.emit()

    def set_refresh_fps(self, fps: int):
        self._cmd_set_refresh_fps.emit(int(fps))

    def pause(self):
        if self._external_sync_enabled:
            self.bus.sig_playback_status.emit(False)
            return
        self._cmd_pause.emit()

    def stop(self):
        if self._external_sync_enabled:
            self.bus.sig_playback_status.emit(False)
            self.bus.sig_time_changed.emit(0.0)
            return
        self._cmd_stop.emit()

    def seek(self, sec: float):
        if self._external_sync_enabled:
            self.bus.sig_time_changed.emit(max(0.0, float(sec)))
            return
        self._cmd_seek.emit(sec)

    # tempo control
    def set_tempo_mode(self, mode: str):
        self._cmd_set_tempo_mode.emit(mode)

    def set_tempo_factor(self, factor: float):
        self._cmd_set_tempo_factor.emit(float(factor))

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
            self._cmd_pause.emit()
            self.bus.sig_playback_status.emit(False)
            self.bus.sig_transport_enabled.emit(False)
        elif self._path:
            self._cmd_prepare_source.emit(self._path)

    @QtCore.Slot(object, int)
    def _on_predecoded_buffer_ready(self, pcm: object, duration_ms: int) -> None:
        if self.model is None:
            return
        self.model.set_predecoded_audio(pcm, self.rate)
        if self._external_sync_enabled:
            self.bus.sig_transport_enabled.emit(False)
            self.bus.sig_playback_status.emit(False)

    # artwork
    def getAlbumArt(self, path: Optional[str] = None) -> Optional[QtGui.QImage]:
        track_path = path or self._path
        if not track_path:
            return None
        try:
            mi = MediaInfo.parse(track_path, cover_data=True)
            if not mi:
                return None
            data: Optional[bytes] = None
            for tr in mi.tracks:
                blob = getattr(tr, "cover_data", None)
                if not blob:
                    continue
                if isinstance(blob, (bytes, bytearray)):
                    data = bytes(blob)
                elif isinstance(blob, list):
                    try:
                        data = bytes(blob)
                    except Exception:
                        data = None
                elif isinstance(blob, str):
                    try:
                        data = base64.b64decode(blob, validate=False)
                    except Exception:
                        data = None
                if data:
                    break
            if data:
                image = QtGui.QImage.fromData(data)
                return image if not image.isNull() else None
        except Exception:
            pass
        return None
