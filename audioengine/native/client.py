"""The native audio engine (audioengine/native, Rust): playback in a process of its own, so it never waits
for this process's GIL, garbage collector or busy frames.

The engine is started with the name of a shared memory block made here (the control block: commands in,
its state out; layout in audioengine/native/src/ipc.rs) and opens each track's PCM from a block of its own,
which is also the PCM this process keeps (DataModel.predecoded_pcm): no copy. NativeFeeder stands in for
PCMFeeder in the player's audio worker (the same methods), NativeClicks for its ClickTrack (the metronome's
mixer), NativeTones for its ToneTrack. Settings > Playback's Audio engine chooses it (the default); without
its executable (not built), or if it cannot start, the player uses PCMFeeder (the Python engine).
"""
from __future__ import annotations

import math
import mmap
import os
import secrets
import struct
import subprocess
import threading
import time
import weakref
import zlib
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore

from core import os_support
from core.resource_paths import resource_path

MAGIC = int.from_bytes(b"MXAUDIC1", "little")
VERSION = 1
STATE_OFFSET = 64
RING_OFFSET = 65536
RING_HEADER = 64
RING_BYTES = 4 << 20
CONTROL_SIZE = RING_OFFSET + RING_HEADER + RING_BYTES
HIST_MAX = 4096
START_TIMEOUT_SEC = 5.0
SEND_TIMEOUT_SEC = 0.5

# state fields (offsets inside the state area)
# the state fields, SEQ .. LATENCY_MS (offsets 0 .. 168: ipc.rs st)
_STATE = struct.Struct("<Q IIIIII Q ddddddd QQQ dd Q II d d")
_HIST = 256

# commands (ipc.rs op)
OP_PAD, OP_LOAD, OP_UNLOAD, OP_PLAY, OP_PAUSE, OP_STOP, OP_SEEK = 0, 1, 2, 3, 4, 5, 6
OP_SCRUB, OP_SCRATCH, OP_SCRATCH_TO, OP_ARM_JUMP, OP_DISARM_JUMP = 7, 8, 9, 10, 11
OP_SET_MODE, OP_SET_FACTOR, OP_SET_TIMESTRETCH, OP_SET_SOFT_CLIP, OP_SET_BUFFER_MS = 12, 13, 14, 15, 16
OP_TONE_HOLD, OP_SET_MUSIC_GAIN = 17, 18
OP_CLICK_SAMPLE, OP_CLICK_PITCHES, OP_CLICK_BEATS, OP_CLICK_ENABLED = 19, 20, 21, 22
OP_CLICK_OFFSET, OP_CLICK_DUCKING, OP_CLICK_CLEAR, OP_SHUTDOWN, OP_WAKE = 23, 24, 25, 26, 27
OP_SET_DEVICE, OP_LIST_DEVICES = 28, 29
DEVICES_OFFSET = 40960   # ipc.rs: [generation u64, length u32, pad] + the names, the one in use first
DEVICES_TIMEOUT_SEC = 1.0


def executable() -> Optional[Path]:
    """The engine's executable (audioengine/native/bin, built by build/audio_engine.py), None if there is none."""
    name = "mixlyzer-audio.exe" if os_support.IS_WINDOWS else "mixlyzer-audio"
    path = Path(resource_path(f"audioengine/native/bin/{name}"))
    return path if path.is_file() else None


def available() -> bool:
    """The native engine is built (whether it starts is known when it does)."""
    return executable() is not None


class EngineState:
    """One consistent reading of the engine's state."""

    __slots__ = ("ready", "rate", "channels", "playing", "track_gen", "at_end", "cmd_done", "playhead",
                 "display", "display_at", "display_rate", "peak", "queued", "capacity", "jumped", "finished",
                 "buffer_gen", "obs_seconds", "obs_level", "underruns", "hist_len", "device_errors", "render_ms",
                 "latency_ms")

    def __init__(self, values: tuple) -> None:
        (_seq, self.ready, self.rate, self.channels, self.playing, self.track_gen, self.at_end, self.cmd_done,
         self.playhead, self.display, self.display_at, self.display_rate, self.peak, self.queued, self.capacity,
         self.jumped, self.finished, self.buffer_gen, self.obs_seconds, self.obs_level, self.underruns,
         self.hist_len, self.device_errors, self.render_ms, self.latency_ms) = values


class NativeEngine:
    """The engine's process and its control block. Commands may be sent from any thread (one at a time)."""

    def __init__(self, exe: Path, device: str = "") -> None:
        self._shm = shared_memory.SharedMemory(create=True, size=CONTROL_SIZE)
        self._buf = self._shm.buf
        struct.pack_into("<QI", self._buf, 0, MAGIC, VERSION)
        self._lock = threading.Lock()
        self._seq = 0
        args = [str(exe), "--shm", self._shm.name, "--size", str(CONTROL_SIZE), "--parent", str(os.getpid())]
        if device:
            args += ["--device", device]
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, cwd=str(exe.parent),
            **os_support.hidden_subprocess_kwargs(),
        )
        self.rate = 0
        self.channels = 0

    @classmethod
    def start(cls, device: str = "") -> Optional["NativeEngine"]:
        """The engine running with the output device open (``device``: its name, "" the system's default; the
        session's format is that device's), or None (no executable, or it failed)."""
        exe = executable()
        if exe is None:
            return None
        try:
            engine = cls(exe, device)
        except Exception as exc:
            print(f"[AudioEngine] start failed: {exc}")
            return None
        deadline = time.monotonic() + START_TIMEOUT_SEC
        while time.monotonic() < deadline:
            state = engine.state()
            if state is not None and state.ready == 1:
                engine.rate, engine.channels = int(state.rate), int(state.channels)
                print(f"[AudioEngine] native engine: {engine.rate} Hz, {engine.channels} ch")
                return engine
            if (state is not None and state.ready == 2) or engine._proc.poll() is not None:
                break
            time.sleep(0.01)
        code = engine._proc.poll()
        reason = "timed out" if code is None else f"exit code {code}"
        print(f"[AudioEngine] native engine did not start ({reason}); using the Python player")
        engine.close()
        return None

    def alive(self) -> bool:
        return self._proc.poll() is None

    def state(self) -> Optional[EngineState]:
        buf = self._buf
        if buf is None:
            return None
        for _ in range(1000):
            s1 = struct.unpack_from("<Q", buf, STATE_OFFSET)[0]
            if s1 & 1:
                continue
            values = _STATE.unpack_from(buf, STATE_OFFSET)
            if struct.unpack_from("<Q", buf, STATE_OFFSET)[0] == s1:
                return EngineState(values)
        return None

    def histogram(self) -> Optional[tuple[int, float, float, np.ndarray]]:
        """(buffer generation, cumulative seconds, cumulative level integral, cumulative seconds per 1 ms)."""
        buf = self._buf
        for _ in range(1000):
            s1 = struct.unpack_from("<Q", buf, STATE_OFFSET)[0]
            if s1 & 1:
                continue
            values = _STATE.unpack_from(buf, STATE_OFFSET)
            n = min(int(values[21]), HIST_MAX)
            hist = np.frombuffer(buf, np.float64, n, STATE_OFFSET + _HIST).copy()
            if struct.unpack_from("<Q", buf, STATE_OFFSET)[0] == s1:
                return int(values[17]), float(values[18]), float(values[19]), hist
        return None

    def devices(self) -> tuple[str, list[str]]:
        """(the output device in use, the devices listed), asked of the engine; ("", []) if it does not answer.
        Any thread."""
        buf = self._buf
        if buf is None:
            return "", []
        before = struct.unpack_from("<Q", buf, DEVICES_OFFSET)[0]
        if not self.send(OP_LIST_DEVICES):
            return "", []
        deadline = time.monotonic() + DEVICES_TIMEOUT_SEC
        while struct.unpack_from("<Q", buf, DEVICES_OFFSET)[0] == before:
            if time.monotonic() > deadline or not self.alive():
                return "", []
            time.sleep(0.002)
        length = struct.unpack_from("<I", buf, DEVICES_OFFSET + 8)[0]
        text = bytes(buf[DEVICES_OFFSET + 16:DEVICES_OFFSET + 16 + length]).decode("utf-8", "replace")
        lines = text.split("\n")
        return lines[0], [n for n in lines[1:] if n]

    def send(self, op: int, payload: bytes = b"") -> bool:
        """Queues a command; False if the engine is gone or did not take its earlier ones in time."""
        buf = self._buf
        if buf is None:
            return False
        size = (24 + len(payload) + 7) & ~7
        with self._lock:
            cap = RING_BYTES
            data = RING_OFFSET + RING_HEADER
            deadline = None
            while True:
                write, read = struct.unpack_from("<QQ", buf, RING_OFFSET)
                at = write % cap
                need = size + (cap - at if at + size > cap else 0)
                if cap - (write - read) >= need:
                    break
                if not self.alive():
                    return False
                if deadline is None:
                    deadline = time.monotonic() + SEND_TIMEOUT_SEC
                elif time.monotonic() > deadline:
                    print(f"[AudioEngine] command {op} dropped: the engine is not taking commands")
                    return False
                time.sleep(0.001)
            if at + size > cap:   # a record never wraps: pad to the end
                struct.pack_into("<II", buf, data + at, cap - at, OP_PAD)
                write += cap - at
                struct.pack_into("<Q", buf, RING_OFFSET, write)
                at = 0
            self._seq += 1
            head = struct.pack("<IQ", op, self._seq)
            checksum = zlib.crc32(payload, zlib.crc32(head))
            struct.pack_into("<IIQII", buf, data + at, size, op, self._seq, checksum, len(payload))
            buf[data + at + 24:data + at + 24 + len(payload)] = payload
            struct.pack_into("<Q", buf, RING_OFFSET, write + size)
            return True

    def close(self) -> None:
        """Stops the engine (asked first, then ended) and frees the control block."""
        if self._buf is None:
            return
        self.send(OP_SHUTDOWN)
        try:
            self._proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._buf = None
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


class SharedPCM(np.ndarray):
    """A track's PCM in a shared memory block the engine opens by ``shm_name``."""

    shm_name: str = ""


def _unlink(name: str) -> None:
    """Removes a POSIX block's name (the mappings stay); nothing on Windows, where a block is gone with its
    last handle."""
    if os_support.IS_WINDOWS or not name:
        return
    import _posixshmem
    try:
        _posixshmem.shm_unlink("/" + name)
    except FileNotFoundError:
        pass


def share_pcm(pcm: np.ndarray) -> SharedPCM:
    """``pcm`` ([frames, channels] float32) copied into a new shared memory block, mapped here as plain mmap
    (freed with the returned array and every view of it, no close to call). Its name is removed once the
    engine has opened it (NativeFeeder.poll) or with the array."""
    pcm = np.asarray(pcm, dtype=np.float32)
    size = max(4, pcm.nbytes)
    name = "mxpcm_" + secrets.token_hex(8)
    if os_support.IS_WINDOWS:
        mm = mmap.mmap(-1, size, tagname=name)
    else:
        import _posixshmem
        fd = _posixshmem.shm_open("/" + name, os.O_CREAT | os.O_EXCL | os.O_RDWR, mode=0o600)
        try:
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
    view = np.frombuffer(mm, dtype=np.float32, count=pcm.size).reshape(pcm.shape).view(SharedPCM)
    view[...] = pcm
    view.shm_name = name
    weakref.finalize(view, _unlink, name)
    return view


class NativeClicks:
    """The metronome's mixer (ClickTrack's methods), its state sent to the engine."""

    def __init__(self, feeder: "NativeFeeder") -> None:
        self._feeder = feeder
        self.rate = feeder.rate
        self.ch = feeder.ch

    def set_sample(self, sample: Optional[np.ndarray]) -> None:
        if sample is None:
            payload = struct.pack("<II", 0, 0)
        else:
            s = np.ascontiguousarray(sample, dtype="<f4")
            payload = struct.pack("<II", s.shape[0], s.shape[1]) + s.tobytes()
        self._feeder.setting(OP_CLICK_SAMPLE, payload)

    def set_pitches(self, beat_semitones: float, accent_semitones: float) -> None:
        self._feeder.setting(OP_CLICK_PITCHES, struct.pack("<dd", float(beat_semitones), float(accent_semitones)))

    def set_beats(self, beats_sec, gains, accents=None) -> None:
        if beats_sec is None or len(beats_sec) == 0:
            payload = struct.pack("<I", 0)
        else:
            beats = np.asarray(beats_sec, dtype="<f8")
            n = beats.size
            g = np.asarray(gains, dtype="<f4").reshape(-1)[:n]
            a = np.zeros(n, np.uint8) if accents is None else np.asarray(accents, dtype=bool).astype(np.uint8)[:n]
            payload = struct.pack("<I", n) + beats.tobytes() + g.tobytes() + a.tobytes()
        self._feeder.setting(OP_CLICK_BEATS, payload)

    def set_enabled(self, enabled: bool) -> None:
        self._feeder.setting(OP_CLICK_ENABLED, struct.pack("<I", int(bool(enabled))))

    def set_offset_sec(self, offset_sec: float) -> None:
        self._feeder.setting(OP_CLICK_OFFSET, struct.pack("<d", float(offset_sec)))

    def set_ducking(self, enabled: bool) -> None:
        self._feeder.setting(OP_CLICK_DUCKING, struct.pack("<I", int(bool(enabled))))

    def clear_voices(self) -> None:
        self._feeder.send(OP_CLICK_CLEAR)


class NativeTones:
    """The reference tones (ToneTrack's hold / active)."""

    def __init__(self, feeder: "NativeFeeder") -> None:
        self._feeder = feeder
        self._hold: tuple[int, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self._hold)

    def hold(self, notes) -> None:
        self._hold = tuple(int(n) for n in (notes or ()))
        self._feeder.setting(OP_TONE_HOLD, struct.pack(f"<I{len(self._hold)}i", len(self._hold), *self._hold))

    def stop(self) -> None:
        self.hold(())


class NativeFeeder(QtCore.QObject):
    """PCMFeeder's methods on the native engine. Lives in the player's audio worker thread; poll() (the
    worker's frame tick) reads the engine's state and reports the track's end and the jumps played. A
    engine that died is started again, with the track and the settings it had."""

    finished = QtCore.Signal()
    jumped = QtCore.Signal()

    PEAK_FLOOR_DBFS = -24.0
    BUFFER_MS_RANGE = (10, 500)
    RESTART_SEC = 2.0

    def __init__(self, engine: NativeEngine, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self.rate = engine.rate
        self.ch = engine.channels
        self.clicks = NativeClicks(self)
        self.tones = NativeTones(self)
        self._settings: dict[int, bytes] = {}   # the last of each setting, sent again to a restarted engine
        self._pcm: Optional[SharedPCM] = None
        self._gen = 0
        self._len = 0
        self._state: Optional[EngineState] = engine.state()
        self._counts = (0, 0)
        self._hist_prev: Optional[tuple[int, float, float, np.ndarray]] = None
        self._underruns_prev: Optional[int] = None
        self._restart_at = 0.0
        self._unlink_gen = 0   # the load whose block's name is removed once the engine has it

    # Commands
    def send(self, op: int, payload: bytes = b"") -> None:
        self._engine.send(op, payload)

    def setting(self, op: int, payload: bytes = b"") -> None:
        self._settings[op] = payload
        self._engine.send(op, payload)

    # Lifecycle
    def open(self) -> None:
        pass

    def close(self) -> None:
        self._engine.close()

    def adopt(self, pcm: np.ndarray) -> np.ndarray:
        """The decoded PCM as the engine reads it (a shared block). Any thread."""
        return pcm if isinstance(pcm, SharedPCM) and pcm.shm_name else share_pcm(pcm)

    def set_track(self, pcm: Optional[np.ndarray]) -> None:
        self._gen += 1
        if pcm is None:
            self._pcm = None
            self._len = 0
            self.send(OP_UNLOAD, struct.pack("<I", self._gen))
            return
        shared = self.adopt(pcm)
        self._pcm = shared   # kept until the next track: the engine maps it meanwhile
        self._len = int(shared.shape[0])
        self._unlink_gen = self._gen
        name = shared.shm_name.encode("utf-8")
        self.send(OP_LOAD, struct.pack("<IIQI", self._gen, int(shared.shape[1]), self._len, len(name)) + name)

    def set_mode(self, mode: str) -> None:
        self.setting(OP_SET_MODE, struct.pack("<I", 0 if (mode or "speed").lower() == "none" else 1))

    def set_timestretch(self, enabled: bool) -> None:
        self.setting(OP_SET_TIMESTRETCH, struct.pack("<I", int(bool(enabled))))

    def set_buffer_ms(self, ms: float) -> None:
        lo, hi = self.BUFFER_MS_RANGE
        self.setting(OP_SET_BUFFER_MS, struct.pack("<d", float(min(hi, max(lo, ms)))))

    def set_factor(self, f: float) -> None:
        f = float(f)
        self.setting(OP_SET_FACTOR, struct.pack("<d", f if math.isfinite(f) else 1.0))

    def set_music_gain(self, gain: float) -> None:
        self.setting(OP_SET_MUSIC_GAIN, struct.pack("<d", max(0.0, float(gain))))

    def set_soft_clip(self, enabled: bool) -> None:
        self.setting(OP_SET_SOFT_CLIP, struct.pack("<I", int(bool(enabled))))

    def set_output_device(self, name: str) -> None:
        """Play on the output device ``name`` ("" the system's default) from now (what is queued plays on)."""
        data = (name or "").encode("utf-8")
        self.setting(OP_SET_DEVICE, struct.pack("<I", len(data)) + data)

    def output_devices(self) -> tuple[str, list[str]]:
        return self._engine.devices()

    def set_scrubbing(self, scrubbing: bool) -> None:
        self.send(OP_SCRUB, struct.pack("<I", int(bool(scrubbing))))

    def set_scratching(self, scratching: bool, frames_per_px: float = 0.0) -> None:
        self.send(OP_SCRATCH, struct.pack("<Id", int(bool(scratching)), float(frames_per_px)))

    def scratch_to(self, frame: float, at: float) -> None:
        self.send(OP_SCRATCH_TO, struct.pack("<dd", float(frame), float(at)))

    # Transport
    def is_playing(self) -> bool:
        return bool(self._state and self._state.playing)

    def play(self) -> bool:
        st = self._state
        if self._pcm is None or (st is not None and st.track_gen == self._gen and st.at_end and not st.playing):
            return False
        self.send(OP_PLAY)
        return True

    def pause(self) -> None:
        self.send(OP_PAUSE)

    def seek(self, frame: float) -> None:
        self.send(OP_SEEK, struct.pack("<d", float(frame)))

    def arm_jump(self, start_sec, dest_sec, once: bool = False, starts_sec=None) -> None:
        if dest_sec is None or (start_sec is None and not starts_sec):
            self.disarm_jump()
            return
        starts = np.asarray(starts_sec if starts_sec else [], dtype="<f8") * self.rate
        start = float(start_sec) * self.rate if start_sec is not None else math.nan
        self.send(OP_ARM_JUMP, struct.pack("<ddII", start, float(dest_sec) * self.rate, int(bool(once)), starts.size)
                  + starts.tobytes())

    def disarm_jump(self) -> None:
        self.send(OP_DISARM_JUMP)

    def wake(self) -> None:
        self.send(OP_WAKE)

    # What is being heard
    def poll(self) -> None:
        """Reads the engine's state (the worker's frame tick): the track's end and the jumps played."""
        if not self._engine.alive():
            self._restart()
            return
        st = self._engine.state()
        if st is None:
            return
        self._state = st
        if self._unlink_gen and st.track_gen >= self._unlink_gen and self._pcm is not None:
            _unlink(self._pcm.shm_name)   # opened by the engine: the name is no longer needed
            self._unlink_gen = 0
        jumped, finished = int(st.jumped), int(st.finished)
        old_jumped, old_finished = self._counts
        self._counts = (jumped, finished)
        if jumped > old_jumped:
            self.jumped.emit()
        if finished > old_finished and st.track_gen == self._gen:
            self.finished.emit()

    def _restart(self) -> None:
        now = time.monotonic()
        if now < self._restart_at:
            return
        self._restart_at = now + self.RESTART_SEC
        print("[AudioEngine] the engine stopped; starting it again")
        engine = NativeEngine.start()
        if engine is None:
            return
        if (engine.rate, engine.channels) != (self.rate, self.ch):
            print("[AudioEngine] the output format changed; the track stays unloaded")
            engine.close()
            return
        self._engine = engine
        self._counts = (0, 0)
        self._hist_prev = None
        for op, payload in self._settings.items():
            engine.send(op, payload)
        if self._pcm is not None:
            pcm, self._pcm = self._pcm, None
            self.set_track(pcm)

    def playhead_frame(self) -> float:
        return float(self._state.playhead) if self._state is not None else 0.0

    def display_frame(self) -> float:
        st = self._state
        if st is None:
            return 0.0
        if st.display_rate > 0.0:
            return float(st.display + (time.perf_counter() - st.display_at) * st.display_rate)
        return float(st.display)

    def heard_peak_dbfs(self) -> float:
        return float(self._state.peak) if self._state is not None else self.PEAK_FLOOR_DBFS

    def take_buffer_stats(self) -> tuple[int, Optional[dict], int]:
        """(frames queued now, the observations since the last call or None, buffer capacity in frames), as
        PCMFeeder.take_buffer_stats, of the engine's own buffer (the setting; the device's fixed latency comes
        after it); the observations also hold "dropout_sec" (the device played silence for want of audio: the
        engine's underruns, not the buffer's lowest level, which the device's whole-period reads touch without a
        gap) and "device_ms" (that latency)."""
        st = self._state
        queued = int(st.queued) if st is not None else 0
        capacity = max(1, int(st.capacity)) if st is not None else 1
        cur = self._engine.histogram()
        stats = None
        prev, self._hist_prev = self._hist_prev, cur
        underruns = int(st.underruns) if st is not None else 0
        prev_underruns, self._underruns_prev = self._underruns_prev, underruns
        if cur is not None and prev is not None and prev[0] == cur[0] and prev[3].size == cur[3].size:
            seconds = cur[1] - prev[1]
            if seconds > 0.0:
                stats = {"hist": cur[3] - prev[3], "seconds": seconds, "mean_ms": (cur[2] - prev[2]) / seconds,
                         "dropout_sec": max(0, underruns - (prev_underruns or underruns)) / max(1, self.rate),
                         "device_ms": float(st.latency_ms) if st is not None else 0.0}
        return queued, stats, capacity
