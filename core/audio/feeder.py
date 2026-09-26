# audio/feeder.py
from __future__ import annotations
import math
import time
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtMultimedia

from core.audio.clicks import ClickTrack
from core.audio.dsp import SpeedResampler, fade_ramp, make_speed_renderer, soft_clip
from core.audio.timeline import OutputTimeline


class PCMFeeder(QtCore.QObject):
    """
    Renders a predecoded track into a QAudioSink.

    While playing, the sink is never reset: seeks, scrubbing, jumps and tempo changes are
    rendered into the running stream (reset()/start() cycles during scrubbing crash Qt
    6.10's WASAPI stream thread). Only pause() resets it once, to stop at the frame being
    heard (mid-waveform) instead of after the queued audio. Once playback stops and the
    last audible frame has been heard, the feed timer is stopped and the sink suspended, so an
    idle player costs no CPU; play() resumes them. Every rendered block is recorded on an
    OutputTimeline, so the playhead is the input position actually being heard, even while
    audio rendered before a tempo change, seek or jump is still queued in the device.

    Modes: "speed" (5-tap Lagrange varispeed, or the pitch-preserving time stretch after
    set_timestretch(True); while scrubbing, the time stretch plays the original at unity)
    / "none" (pass-through, ignores factor).
    Metronome clicks (`clicks`) are mixed into the same stream; music volume is applied here
    too, so the click level stays independent of it.
    """

    finished = QtCore.Signal()  # the last frame of the track has been heard

    PEAK_FLOOR_DBFS = -24.0
    SOFT_CLIP_KNEE = 0.8

    def __init__(self, audio: QtMultimedia.QAudioSink, rate: int, channels: int,
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.audio = audio
        self.rate = int(rate)
        self.ch = int(channels)
        self._bpf = self.ch * 4  # float32

        self.dev: Optional[QtCore.QIODevice] = None
        self._flush_timer: Optional[QtCore.QTimer] = None
        self._chunk_frames = 2048                          # max frames rendered per block
        self._idle_queue_frames = max(1, self.rate // 50)  # ~20 ms of silence while paused
        self._fade_frames = max(1, self.rate // 250)       # ~4 ms declick fades/crossfades
        # Silence played after a pause before the sink is suspended, so the device's own
        # buffer holds only silence when it resumes.
        self._device_tail_frames = max(1, self.rate // 20)  # 50 ms
        self._restart_at = 0.0
        # Idle stop: the sink is suspended once output frame `_audible_until` (the end of
        # the pause fade-out) has been heard while not playing.
        self._device_stopped = False
        self._audible_until = 0
        # Buffer monitor, continuous in time while playing: the queued level is read just
        # before and after every write; in between, the device drains it in real time
        # (down to the next reading), so stalls in this thread are covered too. Time spent
        # at each level is accumulated in 1 ms buckets (seconds per bucket).
        self._obs_time: Optional[float] = None
        self._obs_frames = 0
        self._obs_hist: Optional[np.ndarray] = None
        self._obs_level_ms_seconds = 0.0  # integral of level (ms) over time (s)
        self._obs_seconds = 0.0

        self._timeline = OutputTimeline(self._bpf)
        self.clicks = ClickTrack(self.rate, self.ch)
        self._resampler = make_speed_renderer(self.ch, self.rate)
        self._unity_while_scrubbing = False

        # Track
        self._pcm: Optional[np.ndarray] = None   # [N, ch] float32
        self._len: int = 0

        # Render head (input frames)
        self._pos: float = 0.0
        self._playing = False
        self._scrubbing = False
        self._seeked_while_paused = False
        self._mode = "speed"
        self._factor = 1.0
        self._music_gain = 1.0
        self._soft_clip = False
        self._jump_start: Optional[float] = None
        self._jump_dest: Optional[float] = None

        # End of track
        self._end_out_frame: Optional[int] = None  # output frame where the track ran out
        self._sent_finished = False

    # Lifecycle
    def open(self) -> None:
        """Start the sink and the feed timer; they pause again as soon as the player is idle."""
        if self._flush_timer is None:
            self._flush_timer = QtCore.QTimer(self)
            self._flush_timer.setTimerType(QtCore.Qt.PreciseTimer)
            self._flush_timer.setInterval(1)
            self._flush_timer.timeout.connect(self._flush)
        self._start_sink()
        self._device_stopped = False
        self._flush_timer.start()

    def _stop_device(self) -> None:
        """Idle: stop the feed timer and suspend the sink (resumed by play()).

        Suspend, not stop: with Qt 6.10's WASAPI backend each stop()/start() cycle leaves
        an MMCSS registration behind, and from the 33rd start on the registration fails
        ("AvSetMmThreadCharacteristics failed"), leaving the audio thread without its
        real-time priority (reset()/start() did not leak in the same test). A suspended
        sink keeps its stream (and the audio still queued in it) and costs no CPU.
        """
        self._flush_timer.stop()
        try:
            self.audio.suspend()
        except Exception:
            pass
        self._device_stopped = True

    def _resume_device(self) -> None:
        if not self._device_stopped:
            return
        self._device_stopped = False
        try:
            self.audio.resume()
        except Exception:
            pass
        if self.dev is None or self.audio.state() == QtMultimedia.QtAudio.State.StoppedState:
            self._start_sink()  # the sink died while suspended (e.g. device loss)
        self._flush_timer.start()

    def close(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
        self._playing = False
        self.dev = None
        try:
            self.audio.stop()
        except Exception:
            pass

    def _reset_sink(self) -> None:
        """Drop everything queued in the sink (and the device) and start it again, empty."""
        try:
            self.audio.reset()
        except Exception:
            pass
        self._device_stopped = False
        self._start_sink()
        if self._flush_timer is not None and not self._flush_timer.isActive():
            self._flush_timer.start()

    def _start_sink(self) -> None:
        self._timeline.reset()
        self._end_out_frame = None
        self._audible_until = 0
        try:
            self.dev = self.audio.start()
        except Exception as exc:
            print(f"[Feeder] Audio sink start failed: {exc}")
            self.dev = None

    def _sink_running(self) -> bool:
        """Restart the sink (at most once per second) only if it died on its own, e.g. device loss."""
        if self.dev is not None and self.audio.state() != QtMultimedia.QtAudio.State.StoppedState:
            return True
        now = time.monotonic()
        if now < self._restart_at:
            return False
        self._restart_at = now + 1.0
        print(f"[Feeder] Audio sink stopped (error={self.audio.error()}); restarting")
        self._start_sink()
        return self.dev is not None

    # Configuration
    def set_track(self, pcm: Optional[np.ndarray]) -> None:
        """Load decoded PCM ([N, ch] float32), or None to unload. Stops playback at frame 0."""
        self.pause()
        if pcm is None:
            self._pcm = None
            self._len = 0
        else:
            assert pcm.ndim == 2 and pcm.shape[1] == self.ch, "pcm shape must be [N, ch]"
            self._pcm = np.asarray(pcm, dtype=np.float32, order="C")
            self._len = int(self._pcm.shape[0])
        self._pos = 0.0
        self._end_out_frame = None
        self._sent_finished = False
        self._seeked_while_paused = True
        self._jump_start = None
        self._jump_dest = None
        self.clicks.clear_voices()

    def set_mode(self, mode: str) -> None:
        m = (mode or "speed").lower()
        self._mode = m if m in ("none", "speed") else "speed"

    def set_timestretch(self, enabled: bool) -> None:
        """Pitch-preserving time stretch (True) or varispeed (False) for the "speed" mode."""
        enabled = bool(enabled)
        if enabled == (not isinstance(self._resampler, SpeedResampler)):
            return
        self._resampler = make_speed_renderer(self.ch, self.rate, timestretch=enabled)
        # A time stretcher restarts (pre-roll, tens of ms) on every seek, so while
        # scrubbing the original is played at unity instead (its direct read), and the
        # stretch resumes as soon as scrubbing ends.
        self._unity_while_scrubbing = enabled

    def set_factor(self, f: float) -> None:
        """Applies to audio rendered from now on; queued audio keeps the speed it was rendered at."""
        self._factor = float(max(0.25, min(4.0, f)))

    def set_music_gain(self, gain: float) -> None:
        self._music_gain = float(max(0.0, gain))

    def set_soft_clip(self, enabled: bool) -> None:
        """tanh soft clip of the mixed output (music + clicks) above SOFT_CLIP_KNEE."""
        self._soft_clip = bool(enabled)

    def set_scrubbing(self, scrubbing: bool) -> None:
        """Clicks only sound during normal playback, so they are muted while scrubbing."""
        self._scrubbing = bool(scrubbing)
        if self._scrubbing:
            self.clicks.clear_voices()

    def _step(self) -> float:
        if self._mode != "speed" or (self._scrubbing and self._unity_while_scrubbing):
            return 1.0
        return self._factor

    # Transport (positions in input frames)
    def is_playing(self) -> bool:
        return self._playing

    def play(self) -> bool:
        if self._playing:
            return True
        if self._pcm is None or self._pos >= self._len:
            return False
        if self._device_stopped and (
            self.dev is None or self.audio.state() == QtMultimedia.QtAudio.State.StoppedState
        ):
            self._resume_device()  # the sink died while suspended: restart it before rendering
        self._playing = True
        self._obs_time = None
        self._seeked_while_paused = False
        self._sent_finished = False
        self._end_out_frame = None
        head = self._render(self._pos, self._fade_frames)
        head *= fade_ramp(head.shape[0], self._fade_frames, rising=True)
        self._append(head, self._pos, self._step())
        self._pos += self._step() * head.shape[0]
        # Render a first block before the device runs again: a time stretcher restarts
        # here (pre-roll, ~35 ms), and a device resumed first would play the fade-in and
        # then run dry until that is done.
        if self._pos < self._len:
            self._render_block(self._chunk_frames)
        self._resume_device()
        self._flush()  # start feeding now, not on the next timer tick
        return True

    def pause(self) -> None:
        """Stop at the frame being heard now, mid-waveform (no fade-out).

        Everything queued after it is dropped (sink reset), so the pause is heard at once;
        playback resumes from that frame.
        """
        if not self._playing:
            return
        heard_pos = self.playhead_frame()
        self._playing = False
        self._obs_time = None
        self._reset_sink()
        self.clicks.clear_voices()  # clicks were mixed into the dropped audio
        self._pos = float(max(0.0, min(self._len, heard_pos)))
        # Feed silence for a while before the idle suspend, so the device's own buffer
        # (not counted in bytesFree) holds only silence when it resumes.
        self._audible_until = self._timeline.rendered_frames + self._device_tail_frames
        self._flush()

    def seek(self, frame: float) -> None:
        target = float(max(0, min(frame, self._len)))
        self._end_out_frame = None
        self._sent_finished = False
        if self._playing and self._pcm is not None:
            self._crossfade_to(target)
        else:
            self._pos = target
            self._seeked_while_paused = True

    def arm_jump(self, start_sec, dest_sec) -> None:
        if start_sec is None or dest_sec is None:
            self.disarm_jump()
            return
        self._jump_start = float(start_sec) * self.rate
        self._jump_dest = float(dest_sec) * self.rate

    def disarm_jump(self) -> None:
        self._jump_start = None
        self._jump_dest = None

    # What is being heard
    def _queued_bytes(self) -> int:
        try:
            buf_sz = max(self._bpf, int(self.audio.bufferSize()))
            bytes_free = max(0, int(self.audio.bytesFree()))
            return max(0, buf_sz - bytes_free)
        except Exception:
            return 0

    def _capacity_frames(self) -> int:
        try:
            return max(1, int(self.audio.bufferSize()) // self._bpf)
        except Exception:
            return 1

    def _observe_buffer(self) -> None:
        """Buffer level reading while playing; accumulates the time since the last one."""
        if not self._playing:
            self._obs_time = None
            return
        now = time.perf_counter()
        frames = self._queued_bytes() // self._bpf
        if self._obs_time is not None:
            self._accumulate_buffer(self._obs_frames, frames, now - self._obs_time)
        self._obs_time = now
        self._obs_frames = frames

    def _accumulate_buffer(self, start_frames: int, end_frames: int, seconds: float) -> None:
        """Level trajectory between two readings: drains at real time from the first
        reading until it reaches the second one (a write in between is read separately)."""
        if seconds <= 0.0:
            return
        if self._obs_hist is None:
            self._obs_hist = np.zeros(self._capacity_frames() * 1000 // self.rate + 2, np.float64)
        hist = self._obs_hist
        top = len(hist) - 1
        a = start_frames * 1000.0 / self.rate            # level in ms
        b = max(end_frames * 1000.0 / self.rate, a - seconds * 1000.0)
        drop = max(0.0, a - b)                           # ms drained == ms elapsed
        flat = seconds - drop / 1000.0
        if drop > 0.0:
            # 1 ms of level passes per 1 ms of time: spread drop/1000 s over [b, a].
            k = int(b)
            while k <= int(a) and k <= top:
                overlap = min(a, k + 1.0) - max(b, float(k))
                if overlap > 0.0:
                    hist[k] += overlap / 1000.0
                k += 1
        if flat > 0.0:
            hist[min(top, int(b))] += flat
        self._obs_level_ms_seconds += (a + b) * 0.5 * drop / 1000.0 + b * max(0.0, flat)
        self._obs_seconds += seconds

    def take_buffer_stats(self) -> tuple[int, Optional[dict], int]:
        """(frames queued now, observations since the last call or None, buffer capacity in
        frames); resets the observations. The observations are {"hist": seconds spent at
        each 1 ms level, "seconds": time covered, "mean_ms": time-weighted mean level}."""
        queued = 0 if self._device_stopped else self._queued_bytes() // self._bpf
        stats = None
        if self._obs_hist is not None and self._obs_seconds > 0.0:
            stats = {
                "hist": self._obs_hist,
                "seconds": self._obs_seconds,
                "mean_ms": self._obs_level_ms_seconds / self._obs_seconds,
            }
        self._obs_hist = None
        self._obs_level_ms_seconds = 0.0
        self._obs_seconds = 0.0
        return queued, stats, self._capacity_frames()

    def _heard_out_frame(self) -> float:
        return self._timeline.heard_frame(self._queued_bytes())

    def playhead_frame(self) -> float:
        """Input frame currently being heard."""
        if self._device_stopped or (not self._playing and self._seeked_while_paused):
            return self._pos
        out_frame = self._heard_out_frame()
        seg = self._timeline.segment_at(out_frame)
        if seg is None:
            first = self._timeline.first_segment()
            return first.in_start if first is not None else self._pos
        if seg.step == 0.0:
            return self._pos if not self._playing else seg.in_start
        return seg.in_start + min(out_frame - seg.out_start, seg.frames) * seg.step

    def heard_peak_dbfs(self) -> float:
        """Pre-volume peak of the block being heard."""
        if self._device_stopped:
            return self.PEAK_FLOOR_DBFS
        seg = self._timeline.segment_at(self._heard_out_frame())
        return seg.peak_dbfs if seg is not None else self.PEAK_FLOOR_DBFS

    # Rendering
    def _render(self, pos: float, n_out: int) -> np.ndarray:
        if self._pcm is None:
            return np.zeros((0, self.ch), np.float32)
        return self._resampler.render(self._pcm, pos, self._step(), n_out)

    def _append(self, out: np.ndarray, in_start: float, step: float) -> None:
        n = int(out.shape[0])
        if n <= 0:
            return
        peak = float(np.max(np.abs(out))) if step > 0.0 else 0.0
        peak_dbfs = 20.0 * math.log10(min(max(peak, 1e-12), 1.0))
        peak_dbfs = max(self.PEAK_FLOOR_DBFS, min(0.0, peak_dbfs))
        if step > 0.0:
            if self._music_gain != 1.0:
                out = out * np.float32(self._music_gain)
            if self._playing and not self._scrubbing:
                self.clicks.queue(in_start, step, n)
        out = self.clicks.mix(out)
        if self._soft_clip:
            out = soft_clip(out, self.SOFT_CLIP_KNEE)
        self._timeline.append(out, in_start, step, peak_dbfs)

    def _append_silence(self, n: int) -> None:
        hold = self._pos if self._pos < self._len else float(self._len)
        self._append(np.zeros((n, self.ch), np.float32), hold, 0.0)

    def _crossfade_to(self, target: float) -> None:
        """Fade out the audio continuing from the render head while fading in `target`."""
        target = float(max(0.0, min(target, self._len)))
        n = self._fade_frames
        tail = self._render(self._pos, n) if self._pos < self._len else np.zeros((0, self.ch), np.float32)
        head = self._render(target, n)
        k = head.shape[0]
        mix = head * fade_ramp(k, n, rising=True)
        if tail.shape[0]:
            t = min(k, tail.shape[0])
            mix[:t] += tail[:t] * fade_ramp(t, n, rising=False)
        self._append(mix, target, self._step())
        self._pos = target + self._step() * k

    def _render_block(self, n_out: int) -> None:
        step = self._step()
        js, jd = self._jump_start, self._jump_dest
        if js is not None and jd is not None and self._pos < js <= self._pos + step * n_out:
            # Cut exactly at the jump point and keep the sub-frame phase at the destination.
            n1 = int(math.ceil((js - self._pos) / step))
            if n1 > 0:
                out = self._render(self._pos, n1)
                self._append(out, self._pos, step)
                self._pos += step * out.shape[0]
            self._crossfade_to(jd + (self._pos - js))
            return

        out = self._render(self._pos, n_out)
        self._append(out, self._pos, step)
        self._pos += step * out.shape[0]
        if self._pos >= self._len and self._end_out_frame is None:
            self._pos = float(self._len)
            self._end_out_frame = self._timeline.rendered_frames

    # Feed loop
    @QtCore.Slot()
    def _flush(self) -> None:
        if self._device_stopped or not self._sink_running():
            return
        for _ in range(4):
            self._observe_buffer()
            written = self._timeline.write_to(self.dev)
            self._observe_buffer()
            if not written:
                break  # device full; retry next tick

            try:
                frames_free = max(0, int(self.audio.bytesFree())) // self._bpf
            except Exception:
                frames_free = 0
            if not self._playing:
                # Keep only a short silence queue so play() is heard promptly.
                frames_free = min(frames_free, self._idle_queue_frames - self._queued_bytes() // self._bpf)
            n = min(frames_free, self._chunk_frames)
            if n <= 0:
                break
            if self._playing and self._pos < self._len:
                self._render_block(n)
            else:
                self._append_silence(n)

        heard = self._heard_out_frame()
        self._timeline.prune(heard)

        if not self._playing and heard >= self._audible_until:
            self._stop_device()
            return

        if (
            self._playing
            and self._end_out_frame is not None
            and not self._sent_finished
            and heard >= self._end_out_frame
        ):
            self._sent_finished = True
            self.finished.emit()
