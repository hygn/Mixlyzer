# audio/feeder.py
from __future__ import annotations
from typing import Optional, Deque, Tuple
from collections import deque

import numpy as np
from PySide6 import QtCore, QtMultimedia

from core.audio.dsp import SpeedResampler  # provides set_factor(), process()


class PCMFeeder(QtCore.QObject):
    """
    Predecode-only PCM feeder with precise time estimation.

    - Time/frame-based timeline API:
        * input_playhead_abs_frame() : current playhead as absolute input frames (float)
        * current_base_frame()       : base frame of current stream (float)
    - Keeps playhead consistent after speed changes via recent (out_frames, in_frames) mapping.
    - Modes: "speed" (5-tap Lagrange interp) / "none" (pass-through)
    """

    finished = QtCore.Signal(int)  # playback finished signal (code 0)
    peak_level = QtCore.Signal(float)  # peak dBFS
    playhead_time = QtCore.Signal(float)  # precise playback time in sec

    def __init__(self, audio: QtMultimedia.QAudioSink, rate: int, channels: int,
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.audio = audio
        self.rate = int(rate)
        self.ch = int(channels)

        # Device handle
        self.dev: Optional[QtCore.QIODevice] = None

        # Decoded buffer
        self._pcm: Optional[np.ndarray] = None   # [N, ch] float32
        self._len: int = 0
        self._read_idx: int = 0                  # next frame index to consume/output

        # DSP
        self._mode = "speed"
        self._factor = 1.0
        self._speed = SpeedResampler(self.ch)

        # Timer/output control
        self._flush_timer: Optional[QtCore.QTimer] = None
        self._chunk_frames = 2048                # max frames to output per flush
        # Time/progress tracking (input-based)
        self._abs_base_frames: int = 0           # base frame since last seek
        self._total_in_consumed: int = 0         # total consumed input frames

        # Output/input mapping (for delay correction)
        # Elements: (out_frames:int, in_frames:int)
        self._out_map: Deque[Tuple[int, int]] = deque()
        self._out_map_out_total: int = 0         # total out_frames remaining in deque

        # Prevent duplicate finish handling and track jump state
        self._sent_finished = False
        self._jump_start = None
        self._jump_dest = None
        self._feeder_prev_t = 0
        self._last_written_frames = 0
        self._peak_emit_interval_ms = 16
        self._peak_floor_dbfs = -24.0
        self._last_peak_emit_ms = -10**9
        self._time_emit_interval_ms = 5
        self._last_time_emit_ms = -10**9

    # Configuration API
    def set_predecoded_buffer(self, pcm: np.ndarray):
        """Set decoded PCM ([N, ch] float32)."""
        assert pcm.ndim == 2 and pcm.shape[1] == self.ch, "pcm shape must be [N, ch]"
        self._pcm = np.asarray(pcm, dtype=np.float32, order="C")
        self._len = int(self._pcm.shape[0])
        self._read_idx = 0

        # Reset base counters
        self._abs_base_frames = 0
        self._total_in_consumed = 0
        self._speed.pos = 0.0

        # Reset mapping
        self._out_map.clear()
        self._out_map_out_total = 0
        self._sent_finished = False

        self._jump_start = None
        self._jump_dest = None
        self._feeder_prev_t = 0
        self._last_written_frames = 0
        self._last_peak_emit_ms = -10**9
        self._last_time_emit_ms = -10**9

    def set_mode(self, mode: str):
        m = (mode or "speed").lower()
        self._mode = m if m in ("none", "speed") else "speed"

    def set_factor(self, f: float):
        f = float(max(0.25, min(4.0, f)))
        self._factor = f
        self._speed.set_factor(f)

    def set_peak_emit_interval_ms(self, interval_ms: int) -> None:
        self._peak_emit_interval_ms = max(1, int(interval_ms))

    def is_empty(self) -> bool:
        return (self._pcm is None) or (self._read_idx >= self._len)

    def start(self):
        """Start audio output (create timer)."""
        self.stop()
        self.dev = self.audio.start()
        self._flush_timer = QtCore.QTimer(self)
        self._flush_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._flush_timer.setInterval(1)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start()
        self._sent_finished = False
        self._last_peak_emit_ms = -10**9
        self._last_time_emit_ms = -10**9

    def stop(self):
        """Stop audio/timer; leave time tracking/mapping intact."""
        if self._flush_timer:
            try:
                self._flush_timer.timeout.disconnect(self._flush)
            except Exception:
                pass
            self._flush_timer.stop()
            self._flush_timer.deleteLater()
            self._flush_timer = None

        try:
            self.audio.reset()  # clear buffer
        except Exception:
            pass

        self.dev = None
        self._speed.pos = 0.0
        self._sent_finished = False
        self.peak_level.emit(float(self._peak_floor_dbfs))
        self.playhead_time.emit(float(self.current_base_frame() / max(1, self.rate)))

    def seek_frames(self, f_idx: int):
        """Jump to input-frame index (without resetting the device)."""
        f_idx = int(max(0, min(f_idx, self._len)))
        self._abs_base_frames = f_idx
        self._read_idx = f_idx
        self._total_in_consumed = 0
        self._speed.pos = 0.0
        self.playhead_time.emit(float(self._abs_base_frames / max(1, self.rate)))

    def reset_counters(self):
        """Reset base counters to clear delay-compensation mapping."""
        self._total_in_consumed = 0
        self._speed.pos = 0.0
        self._out_map.clear()
        self._out_map_out_total = 0

    # Time API
    def _queued_out_frames_float(self) -> float:
        """Estimate queued 'output' frames in the device as float."""
        if not self.audio:
            return 0.0
        bpf = self.ch * 4  # float32
        try:
            buf_sz = max(bpf, int(self.audio.bufferSize()))
            bytes_free = max(0, int(self.audio.bytesFree()))
            queued_bytes = max(0, buf_sz - bytes_free)
            return float(queued_bytes) / float(bpf)
        except Exception:
            return 0.0

    def _queued_input_frames_float(self) -> float:
        """
        Estimate queued frames in *input* frame units.
        In speed mode, queued device frames correspond to more/less source
        frames depending on the playback factor.
        """
        q_out = self._queued_out_frames_float()
        if q_out <= 0.0:
            return 0.0
        if self._mode == "speed":
            return q_out * float(max(0.25, min(4.0, self._factor)))
        return q_out

    def input_playhead_frames(self) -> float:
        """
        Return current position (frames, float) with delay compensated in *input* terms.
        = (total input consumed + interpolation fraction) - (input delay from queued output frames)
        """
        total_in_f = float(self._total_in_consumed) + float(getattr(self._speed, "pos", 0.0))

        q_out = self._queued_out_frames_float()
        if q_out <= 1e-9 or not self._out_map:
            return max(0.0, total_in_f)

        need = q_out
        in_delay = 0.0
        for o, i in reversed(self._out_map):
            if need <= 1e-9:
                break
            take = min(need, float(o))
            if o > 0:
                in_delay += float(i) * (take / float(o))
            need -= take

        return max(0.0, total_in_f - in_delay)

    def input_playhead_abs_frame(self) -> float:
        """Return absolute input-frame playhead (float) during playback."""
        return float(self._abs_base_frames) + self.input_playhead_frames()

    def current_base_frame(self) -> float:
        """Return base frame of the playing track (float)."""
        return float(self._abs_base_frames)
    
    def arm_jump(self, start, dest):
        self._jump_start = start
        self._jump_dest = dest
    
    def disarm_jump(self):
        self._jump_start = None
        self._jump_dest = None
        frame = self.input_playhead_abs_frame()
        self._feeder_prev_t = frame / self.rate

    # Playback loop
    @QtCore.Slot()
    def _flush(self):
        if self._jump_start != None and self._jump_dest != None:
            frame = self.input_playhead_abs_frame()
            t = frame / self.rate
            if self._feeder_prev_t < self._jump_start and t > self._jump_start:
                # Keep device running; skip only what is already queued to avoid stutter
                queued_in = self._queued_input_frames_float()
                safety = min(int(self._chunk_frames * max(1.0, self._factor)), int(round(queued_in)))
                target_frame = int(round(self._jump_dest * self.rate + safety))
                self.seek_frames(target_frame)
                self.reset_counters()  # clear delay map so playhead math matches new base
                self._last_written_frames = 0
                self._feeder_prev_t = target_frame / self.rate
                frame = self.input_playhead_abs_frame()
                t = frame / self.rate
            self._feeder_prev_t = t

        if not (self.dev and self.audio and self._pcm is not None):
            return

        # Estimate how many output frames the device needs
        bpf = self.ch * 4
        try:
            frames_free = max(0, int(self.audio.bytesFree())) // bpf
        except Exception:
            frames_free = 0
        if frames_free <= 0:
            self._emit_playhead_time()
            return

        max_out_frames = min(frames_free, self._chunk_frames)

        # If source drained, emit finished once
        if self._read_idx >= self._len:
            self._emit_playhead_time()
            if not self._sent_finished:
                self._sent_finished = True
                self.finished.emit(0)
            return

        # Input block to process this loop (slightly generous)
        # In speed mode, IO ratio depends on factor so leave headroom.
        take_in = min(self._len - self._read_idx, int(max_out_frames * self._factor) + 8)
        in_block = self._pcm[self._read_idx : self._read_idx + take_in]

        # Convert
        if self._mode == "speed":
            out, consumed = self._speed.process(in_block, max_out_frames)
            produced = int(out.shape[0])
            self._last_written_frames = produced
        else:  # 'none' pass-through (ignore factor)
            take = min(max_out_frames, in_block.shape[0])
            out = in_block[:take]
            consumed = int(take)
            produced = int(take)
            self._last_written_frames = produced

        if produced <= 0:
            self._emit_playhead_time()
            return
        

        # Progress bookkeeping
        self._read_idx += int(consumed)          # input-based cursor
        self._total_in_consumed += int(consumed) # cumulative input consumption

        # Write to audio device
        try:
            self.dev.write(out.astype("<f4", copy=False).tobytes())
        except Exception:
            # Ignore device errors and retry next loop
            return

        self._emit_peak_level(out)
        self._emit_playhead_time()

        # Output/input mapping metadata (for delay correction)
        self._out_map.append((int(produced), int(consumed)))
        self._out_map_out_total += int(produced)

        # Memory/queue hygiene: keep at most 2x audio buffer size (bufferSize/bpf)
        try:
            dev_cap_out = max(1, int(self.audio.bufferSize()) // bpf)
        except Exception:
            dev_cap_out = max(1, self._chunk_frames * 2)

        keep_out = max(1, dev_cap_out * 2)
        while self._out_map and self._out_map_out_total > keep_out:
            o, _i = self._out_map.popleft()
            self._out_map_out_total -= int(o)

    def _emit_peak_level(self, out: np.ndarray) -> None:
        if out.size == 0:
            return
        now_ms = int(QtCore.QDateTime.currentMSecsSinceEpoch())
        if (now_ms - self._last_peak_emit_ms) < int(self._peak_emit_interval_ms):
            return
        peak = float(np.max(np.abs(out)))
        peak = min(max(peak, 1e-12), 1.0)
        dbfs = 20.0 * float(np.log10(peak))
        dbfs = max(self._peak_floor_dbfs, min(0.0, dbfs))
        self._last_peak_emit_ms = now_ms
        self.peak_level.emit(dbfs)

    def _emit_playhead_time(self, *, force: bool = False) -> None:
        now_ms = int(QtCore.QDateTime.currentMSecsSinceEpoch())
        if (not force) and ((now_ms - self._last_time_emit_ms) < int(self._time_emit_interval_ms)):
            return
        self._last_time_emit_ms = now_ms
        cur_sec = float(self.input_playhead_abs_frame() / max(1, self.rate))
        self.playhead_time.emit(cur_sec)
