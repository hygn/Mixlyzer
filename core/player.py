from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtMultimedia, QtGui

from core.audio.decoder import decode_to_memmap
from core.audio.feeder import PCMFeeder
from core.event_bus import EventBus
from pymediainfo import MediaInfo
import base64


class _DecodeWorker(QtCore.QObject):
    finished = QtCore.Signal(int, str, object, int)
    error = QtCore.Signal(int, str)

    def __init__(self, token: int, path: str, rate: int, ch: int):
        print("[Player] Decoder init")
        super().__init__()
        self._token = token
        self._path = path
        self._rate = rate
        self._ch = ch

    @QtCore.Slot()
    def run(self):
        try:
            pcm = decode_to_memmap(self._path, self._rate, self._ch)
            arr = np.array(pcm, dtype=np.float32, order="C", copy=False)
            duration_ms = int(round((arr.shape[0] * 1000.0) / max(1, self._rate)))
            self.finished.emit(self._token, self._path, arr, duration_ms)
            print("[Player] Decoder finished")
        except Exception as exc:
            self.error.emit(self._token, str(exc))
            print("[Player] Decoder error")


class PlayerController(QtCore.QObject):
    """
    Audio controller that pre-decodes tracks and feeds them to QAudioSink.
    """

    def __init__(self, bus: EventBus):
        super().__init__()
        self.bus = bus

        out_dev = QtMultimedia.QMediaDevices.defaultAudioOutput()
        fmt = out_dev.preferredFormat()
        fmt.setSampleFormat(QtMultimedia.QAudioFormat.Float)
        self.fmt = fmt
        self.audio = QtMultimedia.QAudioSink(out_dev, fmt, self)
        self.audio.setVolume(0.5)

        self.rate = fmt.sampleRate()
        self.ch = fmt.channelCount()

        self._feeder = PCMFeeder(self.audio, self.rate, self.ch, self)

        self._path: Optional[str] = None
        self._is_playing = False
        self._last_emit_ms = -1
        self._duration_ms = 0
        self._buffer_ready = False
        self._decode_token = 0
        self._decode_jobs: Dict[int, Tuple[QtCore.QThread, _DecodeWorker]] = {}

        self._frame = QtCore.QTimer(self)
        self._frame.setTimerType(QtCore.Qt.PreciseTimer)
        self._frame.setInterval(16)
        self._frame.timeout.connect(self._tick_time)

        self._bps = 4
        self._desired_buffer_ms: int = 80

        # scrub state
        self._scrubbing = False
        self._saved_buffer_ms = None
        self._saved_chunk_frames = None

        # bus connections
        bus.sig_scrub_begin.connect(self._on_scrub_begin)
        bus.sig_scrub_update.connect(self._on_scrub_update)
        bus.sig_scrub_end.connect(self._on_scrub_end)
        bus.sig_seek_requested.connect(self.seek)
        bus.sig_volume_changed.connect(
            lambda v: self.audio.setVolume(max(0.0, min(1.0, v)))
        )
        bus.sig_jump_arm.connect(self._on_jump_arm)
        bus.sig_jump_disarm.connect(lambda: self._feeder.disarm_jump())

        try:
            bus.sig_tempo_mode_changed.connect(self.set_tempo_mode)
            bus.sig_tempo_factor_changed.connect(self.set_tempo_factor)
        except Exception:
            pass

    def _apply_buffer_ms(self):
        """Reconfigure QAudioSink buffer size if needed."""
        bpf = self.ch * 4  # float32 bytes per frame
        buf_bytes = int(self.rate * (self._desired_buffer_ms / 1000.0) * bpf)
        try:
            self.audio.setBufferSize(max(bpf, buf_bytes))
        except Exception:
            pass

    # public API
    def set_source(self, path: str):
        self.pause()
        self._path = path
        self._buffer_ready = False
        self._duration_ms = 0
        self._last_emit_ms = -1
        self.bus.sig_playback_status.emit(False)
        self.bus.sig_duration_changed.emit(0.0)
        self.bus.sig_time_changed.emit(0.0)
        try:
            self.bus.sig_transport_enabled.emit(False)
        except Exception:
            pass

        self._decode_token += 1
        token = self._decode_token
        self._start_decode_job(token, path)

    def get_source(self) -> Optional[str]:
        return self._path

    def play(self):
        if not self._path or self._is_playing or not self._buffer_ready:
            return
        self._apply_buffer_ms()
        self._feeder.start()
        self._is_playing = True
        self._frame.start()
        self.bus.sig_playback_status.emit(True)

    def pause(self):
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
        self.bus.sig_playback_status.emit(False)

        self._last_emit_ms = -1
        self._emit_time(cur_ms)

    def stop(self):
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
        self.bus.sig_playback_status.emit(False)

        self._emit_time(0)

    def seek(self, sec: float):
        if not self._buffer_ready:
            return
        frames = int(round(max(0.0, sec) * self.rate))
        self._feeder.seek_frames(frames)
        if self._is_playing:
            try:
                self.audio.reset()
            except Exception:
                pass
            self._feeder.reset_counters()
            self._feeder.start()
        else:
            self.bus.sig_time_changed.emit(sec)

    # tempo control
    def set_tempo_mode(self, mode: str):
        m = (mode or "speed").lower()
        self._feeder.set_mode(m if m in ("none", "speed") else "speed")

    def set_tempo_factor(self, factor: float):
        self._feeder.set_factor(float(factor))

    # timeline helpers
    def _track_now_ms(self) -> float:
        frame = (
            self._feeder.input_playhead_abs_frame()
            if self._is_playing
            else self._feeder.current_base_frame()
        )
        return (frame * 1000.0) / max(1, self.rate)

    def _emit_time(self, ms: int):
        if ms != self._last_emit_ms:
            self._last_emit_ms = ms
            self.bus.sig_time_changed.emit(ms / 1000.0)

    def _tick_time(self):
        cur = self._track_now_ms()
        if self._duration_ms and cur >= self._duration_ms:
            cur = self._duration_ms
            self._emit_time(cur)
            self.pause()
            return
        self._emit_time(cur)
    
    def _on_jump_arm(self, payload):
        label = payload["label"]
        cue = payload["cue"]
        jump_start = None
        jump_dest = None
        if cue["forward"]["label"] == label: 
            jump_start = cue["forward"]["point"]
            jump_dest = cue["backward"]["point"]
        elif cue["backward"]["label"] == label: 
            jump_start = cue["backward"]["point"]
            jump_dest = cue["forward"]["point"]
        self._feeder.arm_jump(jump_start, jump_dest)

    # scrub handling
    def _on_scrub_begin(self):
        if self._scrubbing:
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

    def _on_scrub_update(self, sec: float):
        if not self._buffer_ready:
            return
        frames = int(round(sec * self.rate))
        self._feeder.seek_frames(frames)
        if not self._is_playing:
            self.bus.sig_time_changed.emit(sec)

    def _on_scrub_end(self, _sec: float):
        if not self._is_playing:
            return
        if self._saved_buffer_ms is not None:
            self._desired_buffer_ms = self._saved_buffer_ms
            self._saved_buffer_ms = None
        if self._saved_chunk_frames is not None:
            self._feeder._chunk_frames = self._saved_chunk_frames
            self._saved_chunk_frames = None
        self._scrubbing = False

    # async decode helpers
    def _start_decode_job(self, token: int, path: str):
        thread = QtCore.QThread(self)
        worker = _DecodeWorker(token, path, self.rate, self.ch)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_decode_ready)
        worker.error.connect(self._on_decode_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._decode_jobs[token] = (thread, worker)
        thread.start()

    @QtCore.Slot(int, str, object, int)
    def _on_decode_ready(self, token: int, path: str, pcm: object, duration_ms: int):
        print("[Player] Buffer ready")
        self._cleanup_decode_job(token)
        if token != self._decode_token:
            return

        try:
            pcm_array = np.asarray(pcm, dtype=np.float32, order="C")
            self._feeder.set_predecoded_buffer(pcm_array)
        except Exception as exc:
            print(f"[Player] Failed to load decoded buffer: {exc}")
            self._buffer_ready = False
            return

        self._buffer_ready = True
        self._duration_ms = duration_ms
        self.bus.sig_duration_changed.emit(self._duration_ms / 1000.0)
        self.bus.sig_time_changed.emit(0.0)
        self.bus.sig_playback_status.emit(False)
        try:
            self.bus.sig_transport_enabled.emit(True)
        except Exception:
            pass

    @QtCore.Slot(int, str)
    def _on_decode_error(self, token: int, message: str):
        self._cleanup_decode_job(token)
        if token != self._decode_token:
            return
        self._buffer_ready = False
        print(f"[Player] Decode error: {message}")
        self.bus.sig_playback_status.emit(False)
        try:
            self.bus.sig_transport_enabled.emit(True)
        except Exception:
            pass

    def _cleanup_decode_job(self, token: int):
        entry = self._decode_jobs.pop(token, None)
        if not entry:
            return
        thread, worker = entry
        try:
            if thread.isRunning():
                thread.quit()
        except Exception:
            pass
        try:
            worker.deleteLater()
        except Exception:
            pass

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
