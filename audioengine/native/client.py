"""The native audio engine (audioengine/native, Rust): playback in a process of its own, so it never waits
for this process's GIL, garbage collector or busy frames.

The engine is started with the name of a shared memory block made here (the control block: commands in,
its state out; layout in audioengine/native/src/ipc.rs) and opens each track's PCM from a block of its own,
which is also the PCM this process keeps (DataModel.predecoded_pcm): no copy. NativeFeeder stands in for
PCMFeeder in the player's audio worker (the same methods), NativeClicks for its ClickTrack (the metronome's
mixer), NativeTones for its ToneTrack. Settings > Playback's Audio engine chooses it (the default); without
its executable (not built), or if it cannot start, the player uses PCMFeeder (the Python engine).

The deck (src/deck.rs) combines the requests of the GUI (its buttons, keys and faders: NativeFeeder's deck
methods) and of a MIDI controller, which the engine reads itself (never through this process); what the GUI
shows (playing, tempo, volume, the cue point, the lamps, the armed JumpCUE) is the deck's report (deck_report).
The controller's messages bound to the GUI's own actions, and all of them for MIDI learn, come through a ring
(NativeEngine.midi_events).
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
VERSION = 3
STATE_OFFSET = 64
RING_OFFSET = 65536
RING_HEADER = 64
RING_BYTES = 4 << 20
CONTROL_SIZE = RING_OFFSET + RING_HEADER + RING_BYTES
HIST_MAX = 4096
START_TIMEOUT_SEC = 5.0
SEND_TIMEOUT_SEC = 0.5

# state fields (offsets inside the state area)
# the state fields, SEQ .. MAX_GAP_MS (offsets 0 .. 248: ipc.rs st)
_STATE = struct.Struct("<Q IIIIII Q ddddddd QQQ dd Q II d d I 4x d ddd dd I 4x Q d II")
_HIST = 256
# the deck's state after the histogram (ipc.rs st::DECK): factor, volume, cue, jump dest, lamps, tempo range,
# MIDI state
_DECK = struct.Struct("<dddd III 4x")
_DECK_AT = 256 + 8 * 4096

# commands (ipc.rs op)
OP_PAD, OP_LOAD, OP_UNLOAD, OP_PLAY, OP_PAUSE, OP_STOP, OP_SEEK = 0, 1, 2, 3, 4, 5, 6
OP_SCRUB, OP_SCRATCH, OP_SCRATCH_TO, OP_ARM_JUMP, OP_DISARM_JUMP = 7, 8, 9, 10, 11
OP_SET_MODE, OP_SET_FACTOR, OP_SET_TIMESTRETCH, OP_SET_SOFT_CLIP, OP_SET_BUFFER_MS = 12, 13, 14, 15, 16
OP_TONE_HOLD, OP_SET_MUSIC_GAIN = 17, 18
OP_CLICK_SAMPLE, OP_CLICK_PITCHES, OP_CLICK_BEATS, OP_CLICK_ENABLED = 19, 20, 21, 22
OP_CLICK_OFFSET, OP_CLICK_DUCKING, OP_CLICK_CLEAR, OP_SHUTDOWN, OP_WAKE = 23, 24, 25, 26, 27
OP_SET_DEVICE, OP_LIST_DEVICES, OP_SET_OUTPUT_MODE, OP_SET_PROCESS = 28, 29, 30, 31
OP_DECK_ACTION, OP_SET_TEMPO, OP_SET_VOLUME, OP_SET_VOLUME_TRIM, OP_DECK_MARKS = 32, 33, 34, 35, 36
OP_JUMP_TO, OP_CALL_AT, OP_MIDI_OPEN, OP_MIDI_MAP, OP_MIDI_LIST, OP_MIDI_OPTIONS, OP_SET_TEMPO_RANGE = (
    37, 38, 39, 40, 41, 42, 43)
OP_SET_METER_GAIN = 44
# the deck's actions (ipc.rs act): the GUI's buttons and keys and the MIDI controller's
ACT_PLAY_PAUSE, ACT_CUE, ACT_STOP, ACT_BEAT_BACK, ACT_BEAT_FORWARD, ACT_BAR_BACK, ACT_BAR_FORWARD = 1, 2, 3, 4, 5, 6, 7
ACT_TEMPO_DOWN, ACT_TEMPO_UP, ACT_TEMPO_RESET, ACT_CALL_PREVIOUS, ACT_CALL_NEXT, ACT_TEMPO_RANGE = 8, 9, 10, 11, 12, 13
ACT_JUMP_A = 16   # .. + 7: H
# a MIDI controller's alone: a fader, a jog wheel
ACT_TEMPO_FADER, ACT_VOLUME_FADER, ACT_JOG_TOUCH, ACT_JOG_SCRATCH, ACT_JOG_BEND, ACT_JOG_SEARCH = 32, 33, 34, 35, 36, 37
ACT_LEVEL_METER = 38   # an output: the controller's level meter (a CC)
# a binding's lamp (src/deck.rs): the controller's light at its message; 1 - 26 the JumpCUE A - Z (lit while set)
LAMP_PLAY, LAMP_CUE = 32, 33
ACT_GUI = 0x1000   # the GUI's own actions from here (core.midi: their numbering), passed on to it
# the tempo fader's ranges (+-%; 100: WIDE, 0 - 200 %)
TEMPO_RANGES = (6, 10, 16, 100)
# the engine's priority (ipc.rs op::SET_PROCESS; sched.rs: what each system makes of it)
PRIORITY_CODES = {"normal": 0, "above_normal": 1, "high": 2}
# a thread's scheduling as the engine got it (st::THREADS, sched.rs) and power throttling's (st::POWER)
THREAD_STATES = {0: "api", 1: "normal", 2: "raised", 3: "raised_rtkit", 4: "realtime", 5: "realtime_rtkit",
                 6: "refused"}
POWER_STATES = {0: "left", 1: "off", 2: "partly", 3: "refused"}
# the audio APIs (ipc.rs op::SET_OUTPUT_MODE, --api): "wasapi" the engine's own (Windows), "cpal" cpal's for the
# system (WASAPI's usual shared stream, ALSA, CoreAudio), "jack" (Linux)
API_CODES = {"wasapi": 0, "cpal": 1, "jack": 2}
DEVICES_OFFSET = 40960   # ipc.rs: [generation u64, length u32, pad] + the names, the one in use first
DEVICES_TIMEOUT_SEC = 1.0
MIDI_PORTS_OFFSET = 34816   # ipc.rs: as the devices, one MIDI input port a line
MIDI_EVENTS_OFFSET = 36864  # ipc.rs: [written u64, pad] + MIDI_EVENTS slots of 16 bytes
MIDI_EVENTS = 240
MIDI_EVENT_GUI = 1   # the message is bound to one of the GUI's actions
_MIDI_EVENT = struct.Struct("<IBBBBI4x")


class DeckReport:
    """What the deck is (src/deck.rs), as the GUI shows it: ``cue`` and ``jump_dest`` (the armed JumpCUE's
    destination) in seconds, NaN for none; ``lamps`` bit 0 PLAY lit, bit 1 CUE lit (blinking: as they are now);
    ``midi_state`` 0 closed, 1 open, 2 the port did not open."""

    __slots__ = ("playing", "factor", "volume", "cue", "jump_dest", "lamps", "tempo_range", "midi_state")

    def __init__(self, playing: bool, factor: float, volume: float, cue: float, jump_dest: float, lamps: int,
                 tempo_range: int, midi_state: int) -> None:
        self.playing, self.factor, self.volume = bool(playing), float(factor), float(volume)
        self.cue, self.jump_dest, self.lamps = float(cue), float(jump_dest), int(lamps)
        self.tempo_range, self.midi_state = int(tempo_range), int(midi_state)

    def key(self) -> tuple:
        return (self.playing, self.factor, self.volume, self.cue, self.jump_dest, self.lamps, self.tempo_range,
                self.midi_state)


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
                 "latency_ms", "output_mode", "period_ms", "response_play", "response_pause", "response_seek",
                 "hist_bucket_ms", "lowest_ms", "output_open", "device_late", "max_gap_ms", "threads", "power",
                 "factor", "volume", "cue", "jump_dest", "lamps", "tempo_range", "midi_state")

    def __init__(self, values: tuple, deck: tuple) -> None:
        (_seq, self.ready, self.rate, self.channels, self.playing, self.track_gen, self.at_end, self.cmd_done,
         self.playhead, self.display, self.display_at, self.display_rate, self.peak, self.queued, self.capacity,
         self.jumped, self.finished, self.buffer_gen, self.obs_seconds, self.obs_level, self.underruns,
         self.hist_len, self.device_errors, self.render_ms, self.latency_ms, self.output_mode,
         self.period_ms, self.response_play, self.response_pause, self.response_seek, self.hist_bucket_ms,
         self.lowest_ms, self.output_open, self.device_late, self.max_gap_ms, self.threads, self.power) = values
        (self.factor, self.volume, self.cue, self.jump_dest, self.lamps, self.tempo_range, self.midi_state) = deck


class NativeEngine:
    """The engine's process and its control block. Commands may be sent from any thread (one at a time)."""

    def __init__(self, exe: Path, device: str = "", api: str = "wasapi", exclusive: bool = False,
                 rate: int = 0) -> None:
        self._shm = shared_memory.SharedMemory(create=True, size=CONTROL_SIZE)
        self._buf = self._shm.buf
        struct.pack_into("<QI", self._buf, 0, MAGIC, VERSION)
        self._lock = threading.Lock()
        self._seq = 0
        self._midi_read = 0   # the MIDI messages read from the ring (its count)
        args = [str(exe), "--shm", self._shm.name, "--size", str(CONTROL_SIZE), "--parent", str(os.getpid())]
        if device:
            args += ["--device", device]
        args += ["--api", str(API_CODES.get(api, 1)), "--exclusive", str(int(bool(exclusive))), "--rate", str(int(rate))]
        self.asked_rate = int(rate)   # the session's rate asked for (exclusive mode; 0: the device's)
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, cwd=str(exe.parent),
            **os_support.hidden_subprocess_kwargs(),
        )
        self.rate = 0
        self.channels = 0

    @classmethod
    def start(cls, device: str = "", api: str = "wasapi", exclusive: bool = False, rate: int = 0
              ) -> Optional["NativeEngine"]:
        """The engine running with the output device open (``device``: its name, "" the system's default; ``api``:
        API_CODES; ``exclusive``: WASAPI's exclusive mode; the session's format is the device's, at ``rate`` in
        exclusive mode when given), or None (no executable, or it failed)."""
        exe = executable()
        if exe is None:
            return None
        try:
            engine = cls(exe, device, api, exclusive, rate)
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
            deck = _DECK.unpack_from(buf, STATE_OFFSET + _DECK_AT)
            if struct.unpack_from("<Q", buf, STATE_OFFSET)[0] == s1:
                return EngineState(values, deck)
        return None

    @property
    def sent(self) -> int:
        """The sequence number of the last command sent (EngineState.cmd_done reaches it once all are taken)."""
        return self._seq

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

    def midi_ports(self) -> list[str]:
        """The MIDI input ports, asked of the engine; [] if it does not answer. Any thread."""
        buf = self._buf
        if buf is None:
            return []
        before = struct.unpack_from("<Q", buf, MIDI_PORTS_OFFSET)[0]
        if not self.send(OP_MIDI_LIST):
            return []
        deadline = time.monotonic() + DEVICES_TIMEOUT_SEC
        while struct.unpack_from("<Q", buf, MIDI_PORTS_OFFSET)[0] == before:
            if time.monotonic() > deadline or not self.alive():
                return []
            time.sleep(0.002)
        length = struct.unpack_from("<I", buf, MIDI_PORTS_OFFSET + 8)[0]
        text = bytes(buf[MIDI_PORTS_OFFSET + 16:MIDI_PORTS_OFFSET + 16 + length]).decode("utf-8", "replace")
        return [n for n in text.split("\n") if n]

    def midi_events(self) -> list[tuple[int, int, int, int, int]]:
        """The MIDI messages received since the last call: (status, data1, data2, flags, action); the oldest
        are lost past MIDI_EVENTS. One reader (the player's audio thread)."""
        buf = self._buf
        if buf is None:
            return []
        written = struct.unpack_from("<Q", buf, MIDI_EVENTS_OFFSET)[0]
        k = max(self._midi_read, written - MIDI_EVENTS)
        out = []
        while k < written:
            seq, status, d1, d2, flags, action = _MIDI_EVENT.unpack_from(
                buf, MIDI_EVENTS_OFFSET + 16 + (k % MIDI_EVENTS) * 16)
            if seq != (k + 1) & 0xFFFFFFFF:
                break   # not all written yet: the next call
            out.append((status, d1, d2, flags, action))
            k += 1
        self._midi_read = k
        return out

    def send(self, op: int, payload: bytes = b"") -> bool:
        """Queues a command; False if the engine is gone or did not take its earlier ones in time."""
        buf = self._buf
        if buf is None:
            return False
        size = (32 + len(payload) + 7) & ~7
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
            sent_at = time.perf_counter()   # the engine's clock too (clock.rs): the response measurement
            head = struct.pack("<IQd", op, self._seq, sent_at)
            checksum = zlib.crc32(payload, zlib.crc32(head))
            struct.pack_into("<IIQIId", buf, data + at, size, op, self._seq, checksum, len(payload), sent_at)
            buf[data + at + 32:data + at + 32 + len(payload)] = payload
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
        """The click as [frames, channels] at ``rate`` (None disables)."""
        self._sample = None if sample is None else np.ascontiguousarray(sample, dtype="<f4")
        self._sample_rate = self.rate
        self._send_sample()

    def _send_sample(self) -> None:
        s = getattr(self, "_sample", None)
        if s is None:
            payload = struct.pack("<II", 0, 0)
        else:
            payload = struct.pack("<II", s.shape[0], s.shape[1]) + s.tobytes()
        self._feeder.setting(OP_CLICK_SAMPLE, payload)

    def follow_rate(self, rate: int) -> None:
        """The engine plays at ``rate`` now: the click resampled for it (it was loaded at the old one)."""
        self.rate = rate
        s = getattr(self, "_sample", None)
        if s is not None and self._sample_rate != rate and s.shape[0] > 1:
            n = max(1, int(round(s.shape[0] * rate / self._sample_rate)))
            t = np.arange(n) * (self._sample_rate / rate)
            src = np.arange(s.shape[0])
            self._sample = np.ascontiguousarray(
                np.stack([np.interp(t, src, s[:, c]) for c in range(s.shape[1])], axis=1), dtype="<f4")
        self._sample_rate = rate
        self._send_sample()

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
    BUFFER_MS_RANGE = (2, 500)   # the engine fills its buffer every 1 ms below 10 ms (engine.rs)
    RESTART_SEC = 2.0
    EXCLUSIVE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)

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
        # the output asked for: device, API, exclusive mode and its rate (the engine is started again with them)
        self._device = ""
        self._api = "wasapi"
        self._exclusive = False
        self._exclusive_rate = 0
        # the deck: its marks (seconds: sent as frames at the engine's rate) and the last report (the state of every
        # command sent: a request still on its way is not shown undone)
        self._marks: Optional[tuple] = None
        self._report: Optional[DeckReport] = None

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

    def set_timestretch(self, enabled: bool) -> None:
        self.setting(OP_SET_TIMESTRETCH, struct.pack("<I", int(bool(enabled))))

    def set_buffer_ms(self, ms: float) -> None:
        lo, hi = self.BUFFER_MS_RANGE
        self.setting(OP_SET_BUFFER_MS, struct.pack("<d", float(min(hi, max(lo, ms)))))

    def set_soft_clip(self, enabled: bool) -> None:
        self.setting(OP_SET_SOFT_CLIP, struct.pack("<I", int(bool(enabled))))

    def set_output_device(self, name: str) -> None:
        """Play on the output device ``name`` ("" the system's default) from now (what is queued plays on)."""
        self._device = name or ""
        data = (name or "").encode("utf-8")
        self.setting(OP_SET_DEVICE, struct.pack("<I", len(data)) + data)

    def set_output(self, api: str, exclusive: bool) -> None:
        """The audio API (API_CODES) and WASAPI's exclusive mode, opened again at once at the session's rate (one
        the device cannot open falls back to cpal's shared output); a rate of its own waits for the next track."""
        self._api, self._exclusive = api, bool(exclusive)
        self.setting(OP_SET_OUTPUT_MODE, struct.pack("<II", API_CODES.get(api, 1), int(bool(exclusive))))

    def set_exclusive_rate(self, rate: int) -> None:
        """The session's rate in exclusive mode (0: the device's), from the next track loaded on."""
        self._exclusive_rate = int(rate)

    def set_process(self, priority: str, power_throttling: bool) -> None:
        """The engine's priority (PRIORITY_CODES: Windows the process's class, Linux its threads' real-time / nice,
        macOS its thread's time constraint / QoS) and whether the system may power-throttle it, at once."""
        self.setting(OP_SET_PROCESS, struct.pack("<II", PRIORITY_CODES.get(priority, 2), int(bool(power_throttling))))

    def apply_session_format(self) -> bool:
        """Before a track is decoded: the engine started again where the rate asked for differs from its own
        (exclusive mode's rate, or the device's again); True if the format (rate, channels) changed."""
        wanted = self._exclusive_rate if self._exclusive else 0
        if wanted == self._engine.asked_rate:
            return False
        # the old engine first let go of the device (an exclusive one holds it), then the new one; back to the old
        # format if the new one does not start
        previous = self._engine.asked_rate
        self._engine.close()
        engine = NativeEngine.start(self._device, self._api, self._exclusive, wanted)
        if engine is None:
            engine = NativeEngine.start(self._device, self._api, self._exclusive, previous)
            if engine is None:
                return False
        self._engine = engine
        changed = (engine.rate, engine.channels) != (self.rate, self.ch)
        self.rate, self.ch = engine.rate, engine.channels
        self._counts = (0, 0)
        self._hist_prev = None
        self._underruns_prev = None
        self._pcm = None
        self._replay()
        self.clicks.ch = self.ch
        self.clicks.follow_rate(self.rate)
        return changed

    def _replay(self) -> None:
        """A new engine: the settings, the deck as it was reported, its marks at this engine's rate."""
        for op, payload in self._settings.items():
            self._engine.send(op, payload)
        r = self._report
        if r is not None:
            self._engine.send(OP_SET_TEMPO_RANGE, struct.pack("<I", r.tempo_range))
            self._engine.send(OP_SET_TEMPO, struct.pack("<d", r.factor))
            self._engine.send(OP_SET_VOLUME, struct.pack("<d", r.volume))
        self._send_marks()

    def output_devices(self) -> tuple[str, list[str]]:
        return self._engine.devices()

    # The deck (src/deck.rs): requests, combined there with the MIDI controller's; what it is: deck_report
    def deck_action(self, code: int, pressed: bool = True) -> None:
        self.send(OP_DECK_ACTION, struct.pack("<II", int(code), int(bool(pressed))))

    def set_tempo(self, factor: float) -> None:
        f = float(factor)
        self.send(OP_SET_TEMPO, struct.pack("<d", f if math.isfinite(f) else 1.0))

    def set_tempo_range(self, percent: int) -> None:
        self.send(OP_SET_TEMPO_RANGE, struct.pack("<I", int(percent)))

    def set_volume(self, position: float) -> None:
        self.send(OP_SET_VOLUME, struct.pack("<d", min(1.0, max(0.0, float(position)))))

    def set_volume_trim(self, db: float) -> None:
        self.setting(OP_SET_VOLUME_TRIM, struct.pack("<d", float(db)))

    def set_marks(self, beats_sec, downbeats, cue_points_sec, jumpcues_sec) -> None:
        """The track's beats (with their downbeat flags), CUE points and JumpCUEs A-Z (NaN: none), in seconds."""
        self._marks = (np.asarray(beats_sec, dtype=np.float64).reshape(-1),
                       np.asarray(downbeats, dtype=bool).reshape(-1),
                       np.asarray(cue_points_sec, dtype=np.float64).reshape(-1),
                       np.asarray(jumpcues_sec, dtype=np.float64).reshape(-1))
        self._send_marks()

    def _send_marks(self) -> None:
        if self._marks is None:
            return
        beats, down, cues, jumps = self._marks
        n = min(beats.size, down.size)
        jc = np.full(26, np.nan)
        jc[:min(26, jumps.size)] = jumps[:26]
        rate = float(self.rate)
        self.send(OP_DECK_MARKS, struct.pack("<I", n) + (beats[:n] * rate).astype("<f8").tobytes()
                  + down[:n].astype(np.uint8).tobytes() + struct.pack("<I", cues.size)
                  + (cues * rate).astype("<f8").tobytes() + (jc * rate).astype("<f8").tobytes())

    def jump_to(self, sec: float) -> None:
        self.send(OP_JUMP_TO, struct.pack("<d", float(sec) * self.rate))

    def call_at(self, sec: float) -> None:
        self.send(OP_CALL_AT, struct.pack("<d", float(sec) * self.rate))

    def new_track(self) -> None:
        """The engine resets its deck itself when it loads a track."""

    def tick(self) -> None:
        """The deck runs in the engine."""

    def deck_report(self) -> Optional[DeckReport]:
        return self._report

    # The MIDI controller (read by the engine)
    def midi_open(self, port: str) -> None:
        """Read the MIDI input ``port`` ("": none)."""
        data = (port or "").encode("utf-8")
        self.setting(OP_MIDI_OPEN, struct.pack("<I", len(data)) + data)

    def midi_map(self, bindings) -> None:
        """``bindings``: (status, data1, 14-bit, lamp, action code) each."""
        payload = struct.pack("<I", len(bindings)) + b"".join(
            struct.pack("<BBBBI", int(st), int(d1), int(bool(hires)), int(lamp), int(code))
            for st, d1, hires, lamp, code in bindings)
        self.setting(OP_MIDI_MAP, payload)

    def set_meter_gain(self, gain: float) -> None:
        """The controller's level meter shows the heard peak times this (the GUI peak meter's gain)."""
        self.setting(OP_SET_METER_GAIN, struct.pack("<d", max(0.0, float(gain))))

    def midi_options(self, jog_ticks: int) -> None:
        self.setting(OP_MIDI_OPTIONS, struct.pack("<I", max(1, int(jog_ticks))))

    def midi_ports(self) -> list[str]:
        return self._engine.midi_ports()

    def midi_events(self) -> list[tuple[int, int, int, int, int]]:
        return self._engine.midi_events()

    def set_scrubbing(self, scrubbing: bool) -> None:
        self.send(OP_SCRUB, struct.pack("<I", int(bool(scrubbing))))

    def set_scratching(self, scratching: bool, frames_per_px: float = 0.0) -> None:
        self.send(OP_SCRATCH, struct.pack("<Id", int(bool(scratching)), float(frames_per_px)))

    def scratch_to(self, frame: float, at: float) -> None:
        self.send(OP_SCRATCH_TO, struct.pack("<dd", float(frame), float(at)))

    # Transport (whether it plays: deck_report)
    def play(self) -> None:
        self.send(OP_PLAY)

    def pause(self) -> None:
        self.send(OP_PAUSE)

    def seek(self, frame: float) -> None:
        self.send(OP_SEEK, struct.pack("<d", float(frame)))

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
        if st.cmd_done >= self._engine.sent:
            self._report = DeckReport(st.playing, st.factor, st.volume, st.cue / max(1, self.rate),
                                      st.jump_dest / max(1, self.rate), st.lamps, st.tempo_range, st.midi_state)
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
        wanted = self._exclusive_rate if self._exclusive else 0
        engine = NativeEngine.start(self._device, self._api, self._exclusive, wanted)
        if engine is None:
            return
        if (engine.rate, engine.channels) != (self.rate, self.ch):
            print("[AudioEngine] the output format changed; the track stays unloaded")
            engine.close()
            return
        self._engine = engine
        self._counts = (0, 0)
        self._hist_prev = None
        self._replay()
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
                         "device_ms": float(st.latency_ms) if st is not None else 0.0,
                         "output_api": {v: k for k, v in API_CODES.items()}.get(int(st.output_mode) & 0xFF, "cpal")
                         if st is not None else "cpal",
                         "exclusive": bool(int(st.output_mode) >> 8) if st is not None else False,
                         "response_ms": {"play": st.response_play, "pause": st.response_pause,
                                         "seek": st.response_seek} if st is not None else {},
                         "bucket_ms": float(st.hist_bucket_ms) if st is not None else 1.0,
                         "lowest_ms": float(st.lowest_ms) if st is not None else -1.0,
                         "output_open": bool(st.output_open) if st is not None else True,
                         "late": int(st.device_late) if st is not None else 0,
                         "max_gap_ms": float(st.max_gap_ms) if st is not None else 0.0,
                         "engine_thread": THREAD_STATES.get(int(st.threads) & 0xFF, "api") if st is not None else "api",
                         "output_thread": THREAD_STATES.get(int(st.threads) >> 8 & 0xFF, "api")
                         if st is not None else "api",
                         "power": POWER_STATES.get(int(st.power), "left") if st is not None else "left"}
        return queued, stats, capacity
