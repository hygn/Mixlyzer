"""The native audio engine (audioengine/native, Rust): playback in a process of its own, so it never waits
for this process's GIL, garbage collector or busy frames.

The engine is started with the name of a shared memory block made here (the control block: commands in,
its state out; layout in audioengine/native/src/ipc.rs) and opens each track's PCM from a block of its own,
which is also the PCM this process keeps (DataModel.predecoded_pcm): no copy. NativeFeeder stands in for
PCMFeeder in the player's audio worker (the same methods), NativeClicks for its ClickTrack (the metronome's
mixer), NativeTones for its ToneTrack. Settings > Audio > Engine chooses it (the default); without
its executable (not built), or if it cannot start, the player uses PCMFeeder (the Python engine).

The headphones and the booth may play on devices of their own (src/bridge.rs: each fed from the main device's mix,
followed at its own clock); each has its own buffer and buffer monitor (take_buffer_stats' "outputs").

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

from system import os_support
from system.resource_paths import resource_path
from audioengine import controls
from audioengine.controls import LEVEL_UNITY, MIX_KEY, MIX_KEYS, DeckReport
from audioengine.volume import trim_db_to_knob

MAGIC = int.from_bytes(b"MXAUDIC1", "little")
VERSION = 8
STATE_OFFSET = 64
RING_OFFSET = 172032
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
# MIDI state, the output's channels; its mixer channel's: headphones CUE, TRIM / EQ LOW, MID, HIGH knobs, COLOR knob
_DECK = struct.Struct("<dddd IIII I 4x dddd d")
_DECK_AT = 256 + 8 * 4096

# commands (ipc.rs op)
OP_PAD, OP_LOAD, OP_UNLOAD, OP_PLAY, OP_PAUSE, OP_STOP, OP_SEEK = 0, 1, 2, 3, 4, 5, 6
OP_SCRUB, OP_SCRATCH, OP_SCRATCH_TO, OP_ARM_JUMP, OP_DISARM_JUMP = 7, 8, 9, 10, 11
OP_SET_MODE, OP_SET_FACTOR, OP_SET_TIMESTRETCH, OP_SET_SOFT_CLIP, OP_SET_BUFFER_MS = 12, 13, 14, 15, 16
OP_TONE_HOLD = 17
OP_CLICK_SAMPLE, OP_CLICK_PITCHES, OP_CLICK_BEATS, OP_CLICK_ENABLED = 19, 20, 21, 22
OP_CLICK_OFFSET, OP_CLICK_DUCKING, OP_CLICK_CLEAR, OP_SHUTDOWN, OP_WAKE = 23, 24, 25, 26, 27
OP_SET_DEVICE, OP_LIST_DEVICES, OP_SET_OUTPUT_MODE, OP_SET_PROCESS = 28, 29, 30, 31
OP_DECK_ACTION, OP_SET_TEMPO, OP_SET_VOLUME, OP_DECK_MARKS = 32, 33, 34, 36
OP_JUMP_TO, OP_CALL_AT, OP_MIDI_OPEN, OP_MIDI_MAP, OP_MIDI_LIST, OP_MIDI_OPTIONS, OP_SET_TEMPO_RANGE = (
    37, 38, 39, 40, 41, 42, 43)
OP_SET_METER_GAIN = 44
# the mix: the routing, the channel's headphones CUE
OP_SET_ROUTING, OP_SET_CHANNEL_CUE = 47, 49
OP_SET_AUX_OUTPUT = 50   # a bus on a device of its own
OP_SET_HEADPHONES = 51   # the master mixed into the headphones, the CUE / MASTER mix, their level
OP_SET_MIXER_KNOB = 52   # the deck's mixer channel knob: u32 0 TRIM, 1 EQ LOW, 2 MID, 3 HIGH, 4 COLOR; f64 its position
OP_SET_MIXER = 53   # the mixer's setting: u32 key (MIX_KEYS' place), f64 value
OP_SET_SPECTRUM = 54   # u32 the master's spectrum made (1) or not
NO_CHANNEL = 0xFF   # a bus not routed
# the buses that may play on a device of their own (ipc.rs OUTPUTS_OFFSET: a state block each, out::*)
AUX_BUSES = ("headphones", "booth")
OUTPUTS_OFFSET = 65536
# the mixer's settings as it has them (ipc.rs mix: f64 each, MIX_OFFSET + 8 * key: audioengine.controls.MIX_KEYS)
# and its meters (from METERS on)
MIX_OFFSET = 165888
MIX_METERS_AT = 64
MIX_METERS = ("master", "booth")
_OUT = struct.Struct("<II dddd QQ d II d Q dd dd II I")
_OUT_HIST = 384
_OUT_SIZE = _OUT_HIST + 8 * 4096
_OUT_NAME = 128
_OUT_RATE = 376
AUX_STATES = {0: "off", 1: "open", 2: "failed", 3: "opening"}
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
_MIDI_EVENT = struct.Struct("<IBBBBIi")
# the master's spectrum (ipc.rs SPECTRUM_*): [written u64, pad] + SPECTRUM_FRAMES slots of SPECTRUM_SLOT bytes:
# [seq u32, pad 12, SPECTRUM_BINS u8 (SPECTRUM_DB_FLOOR + v * SPECTRUM_DB_STEP dB)]
SPECTRUM_OFFSET = 166912
SPECTRUM_FRAMES = 16
SPECTRUM_SLOT = 256
SPECTRUM_DB_FLOOR = -160.0
SPECTRUM_DB_STEP = 0.625


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
                 "factor", "volume", "cue", "jump_dest", "lamps", "tempo_range", "midi_state", "out_channels",
                 "channel_cue", "knobs", "color")

    def __init__(self, values: tuple, deck: tuple) -> None:
        (_seq, self.ready, self.rate, self.channels, self.playing, self.track_gen, self.at_end, self.cmd_done,
         self.playhead, self.display, self.display_at, self.display_rate, self.peak, self.queued, self.capacity,
         self.jumped, self.finished, self.buffer_gen, self.obs_seconds, self.obs_level, self.underruns,
         self.hist_len, self.device_errors, self.render_ms, self.latency_ms, self.output_mode,
         self.period_ms, self.response_play, self.response_pause, self.response_seek, self.hist_bucket_ms,
         self.lowest_ms, self.output_open, self.device_late, self.max_gap_ms, self.threads, self.power) = values
        (self.factor, self.volume, self.cue, self.jump_dest, self.lamps, self.tempo_range, self.midi_state,
         self.out_channels, self.channel_cue) = deck[:9]
        self.knobs = tuple(deck[9:13])   # TRIM, EQ LOW, MID, HIGH
        self.color = deck[13]


def _relay_console(stream) -> None:
    """The engine's console lines (src/log.rs: written by a thread of its own, so this reader being late holds up no
    audio) printed here: into the app's log and the startup splash. Ends with the process."""
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                try:
                    print(line, flush=True)
                except Exception:
                    pass
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


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
        self._spectrum_read = 0   # the spectrum frames read (their count)
        args = [str(exe), "--shm", self._shm.name, "--size", str(CONTROL_SIZE), "--parent", str(os.getpid())]
        if device:
            args += ["--device", device]
        args += ["--api", str(API_CODES.get(api, 1)), "--exclusive", str(int(bool(exclusive))), "--rate", str(int(rate))]
        self.asked_rate = int(rate)   # the session's rate asked for (exclusive mode; 0: the device's)
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=str(exe.parent),
            **os_support.hidden_subprocess_kwargs(),
        )
        threading.Thread(target=_relay_console, args=(self._proc.stderr,), name="AudioEngineConsole",
                         daemon=True).start()
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
        started = time.monotonic()
        deadline = started + START_TIMEOUT_SEC
        while time.monotonic() < deadline:
            state = engine.state()
            if state is not None and state.ready == 1:
                engine.rate, engine.channels = int(state.rate), int(state.channels)
                print(f"[AudioEngine] native engine ready in {1000 * (time.monotonic() - started):.0f} ms: "
                      f"{engine.rate} Hz, {engine.channels} ch")
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

    def mixer(self) -> Optional[tuple[tuple, dict]]:
        """One reading of the mixer's settings (MIX_KEYS' order) and its meters (MIX_METERS: linear peaks since the
        engine's last publish)."""
        buf = self._buf
        if buf is None:
            return None
        for _ in range(1000):
            s1 = struct.unpack_from("<Q", buf, STATE_OFFSET)[0]
            if s1 & 1:
                continue
            values = struct.unpack_from(f"<{len(MIX_KEYS)}d", buf, MIX_OFFSET)
            meters = struct.unpack_from(f"<{len(MIX_METERS)}d", buf, MIX_OFFSET + 8 * MIX_METERS_AT)
            if struct.unpack_from("<Q", buf, STATE_OFFSET)[0] == s1:
                return values, dict(zip(MIX_METERS, meters))
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

    def devices(self, api: Optional[str] = None) -> tuple[str, list[str], dict[str, int]]:
        """(the main output's device in use, the devices of ``api`` listed (None: the main output's API), each
        device's channels by name, "" the system's default's), asked of the engine; ("", [], {}) if it does not
        answer. Any thread."""
        buf = self._buf
        if buf is None:
            return "", [], {}
        before = struct.unpack_from("<Q", buf, DEVICES_OFFSET)[0]
        payload = b"" if api is None else struct.pack("<I", API_CODES.get(api, 1))
        if not self.send(OP_LIST_DEVICES, payload):
            return "", [], {}
        deadline = time.monotonic() + DEVICES_TIMEOUT_SEC
        while struct.unpack_from("<Q", buf, DEVICES_OFFSET)[0] == before:
            if time.monotonic() > deadline or not self.alive():
                return "", [], {}
            time.sleep(0.002)
        length = struct.unpack_from("<I", buf, DEVICES_OFFSET + 8)[0]
        text = bytes(buf[DEVICES_OFFSET + 16:DEVICES_OFFSET + 16 + length]).decode("utf-8", "replace")
        lines = text.split("\n")
        names: list[str] = []
        channels: dict[str, int] = {}
        for k, line in enumerate(lines[1:]):
            name, _, ch = line.rpartition("\t")
            if not name:
                continue
            count = int(ch) if ch.isdigit() else 2
            if k == 0:
                channels[""] = count   # the system's default
            else:
                names.append(name)
                channels[name] = count
        return lines[0], names, channels

    def aux_states(self) -> list[Optional[dict]]:
        """Each bus's output on a device of its own (AUX_BUSES; None: it has none): its state (out::*), with its
        histogram."""
        buf = self._buf
        if buf is None:
            return [None] * len(AUX_BUSES)
        keys = ("state", "mode", "period_ms", "latency_ms", "queued", "capacity", "underruns", "late", "max_gap_ms",
                "thread", "channels", "drift_ppm", "buffer_gen", "obs_seconds", "obs_level", "hist_bucket_ms",
                "lowest_ms", "hist_len", "errors", "name_len")
        for _ in range(1000):
            s1 = struct.unpack_from("<Q", buf, STATE_OFFSET)[0]
            if s1 & 1:
                continue
            out = []
            for k in range(len(AUX_BUSES)):
                at = OUTPUTS_OFFSET + k * _OUT_SIZE
                d = dict(zip(keys, _OUT.unpack_from(buf, at)))
                if not d["state"]:
                    out.append(None)
                    continue
                n = min(int(d["name_len"]), 240)
                d["rate"] = struct.unpack_from("<I", buf, at + _OUT_RATE)[0]
                d["name"] = bytes(buf[at + _OUT_NAME + 4:at + _OUT_NAME + 4 + n]).decode("utf-8", "replace")
                d["hist"] = np.frombuffer(buf, np.float64, min(int(d["hist_len"]), HIST_MAX), at + _OUT_HIST).copy()
                out.append(d)
            if struct.unpack_from("<Q", buf, STATE_OFFSET)[0] == s1:
                return out
        return [None] * len(AUX_BUSES)

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

    def midi_events(self) -> list[tuple[int, int, int, int, int, int]]:
        """The MIDI messages received since the last call: (status, data1, data2, flags, action, steps: a knob's
        turn, + clockwise); the oldest are lost past MIDI_EVENTS. One reader (the player's audio thread)."""
        buf = self._buf
        if buf is None:
            return []
        written = struct.unpack_from("<Q", buf, MIDI_EVENTS_OFFSET)[0]
        k = max(self._midi_read, written - MIDI_EVENTS)
        out = []
        while k < written:
            seq, status, d1, d2, flags, action, steps = _MIDI_EVENT.unpack_from(
                buf, MIDI_EVENTS_OFFSET + 16 + (k % MIDI_EVENTS) * 16)
            if seq != (k + 1) & 0xFFFFFFFF:
                break   # not all written yet: the next call
            out.append((status, d1, d2, flags, action, steps))
            k += 1
        self._midi_read = k
        return out

    def spectrum_frames(self) -> np.ndarray:
        """The master's spectrum frames made since the last call (op SET_SPECTRUM on), oldest first: ``[k,
        controls.SPECTRUM_BINS]`` float32 dB (controls.spectrum_edges); the oldest are lost past SPECTRUM_FRAMES.
        One reader (the GUI thread)."""
        buf = self._buf
        bins = controls.SPECTRUM_BINS
        if buf is None:
            return np.zeros((0, bins), np.float32)
        written = struct.unpack_from("<Q", buf, SPECTRUM_OFFSET)[0]
        k = max(self._spectrum_read, written - SPECTRUM_FRAMES)
        out = []
        while k < written:
            at = SPECTRUM_OFFSET + 16 + (k % SPECTRUM_FRAMES) * SPECTRUM_SLOT
            want = (k + 1) & 0xFFFFFFFF
            if struct.unpack_from("<I", buf, at)[0] != want:
                break   # being written: the next call
            values = np.frombuffer(bytes(buf[at + 16:at + 16 + bins]), np.uint8)
            if struct.unpack_from("<I", buf, at)[0] != want:
                break   # written over while copied
            out.append(values)
            k += 1
        self._spectrum_read = k
        if not out:
            return np.zeros((0, bins), np.float32)
        return (SPECTRUM_DB_FLOOR + np.stack(out).astype(np.float32) * SPECTRUM_DB_STEP).astype(np.float32)

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
    """PCMFeeder's methods on the native engine: its deck, and the engine's own settings, the mixer's, the outputs'
    and the MIDI controller's. Lives in its player's audio worker thread; poll() (the worker's frame tick) reads the
    engine's state and reports the track's end and the jumps played. An engine that died is started again, with the
    track and the settings it had."""

    finished = QtCore.Signal()
    jumped = QtCore.Signal()

    PEAK_FLOOR_DBFS = controls.PEAK_FLOOR_DBFS   # src/engine.rs'
    BUFFER_MS_RANGE = controls.NATIVE_BUFFER_MS_RANGE
    RESTART_SEC = 2.0
    EXCLUSIVE_RATES = controls.NATIVE_EXCLUSIVE_RATES

    def __init__(self, engine: NativeEngine, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self.rate = engine.rate
        self.ch = engine.channels
        self.clicks = NativeClicks(self)
        self.tones = NativeTones(self)
        # the last of each setting (by its op, and what it is for), sent again to a restarted engine
        self._settings: dict[tuple, tuple[int, bytes]] = {}
        self._jog_ticks, self._jog_vinyl = 720, True   # the jog wheel's messages per turn, its mode (midi_options)
        self._pcm: Optional[SharedPCM] = None
        self._gen = 0
        self._len = 0
        self._state: Optional[EngineState] = engine.state()
        self._counts = (0, 0)
        self._hist_prev: Optional[tuple[int, float, float, np.ndarray]] = None
        self._underruns_prev: Optional[int] = None
        self._aux_prev: list[Optional[dict]] = [None] * len(AUX_BUSES)
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

    def setting(self, op: int, payload: bytes = b"", key=None) -> None:
        self._settings[(op, key)] = (op, payload)
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
        macOS its thread's time constraint / QoS) and whether the system may power-throttle it, at once. Sent
        when it changes (every deck.json save applies the playback settings again: the engine applied and logged
        it each time); a new engine gets it from _replay."""
        payload = struct.pack("<II", PRIORITY_CODES.get(priority, 2), int(bool(power_throttling)))
        if self._settings.get((OP_SET_PROCESS, None)) != (OP_SET_PROCESS, payload):
            self.setting(OP_SET_PROCESS, payload)

    def set_spectrum(self, on: bool) -> None:
        """The master's spectrum made (``on``: while Settings > View shows it) or not; a new engine gets it from
        _replay."""
        self.setting(OP_SET_SPECTRUM, struct.pack("<I", int(bool(on))))

    def spectrum_frames(self) -> np.ndarray:
        """NativeEngine.spectrum_frames of the engine running (the GUI thread reads them)."""
        return self._engine.spectrum_frames()

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
        self._aux_prev = [None] * len(AUX_BUSES)
        self._pcm = None
        self._replay()
        self.clicks.ch = self.ch
        self.clicks.follow_rate(self.rate)
        return changed

    def _replay(self) -> None:
        """A new engine: the settings, the deck and the mixer as they were reported (the GUI's settings and a
        controller's alike), its marks at this engine's rate."""
        for op, payload in self._settings.values():
            self.send(op, payload)
        r = self._report
        if r is not None:
            for key, value in enumerate(r.mix):
                self.send(OP_SET_MIXER, struct.pack("<Id", key, float(value)))
            self.send(OP_SET_TEMPO_RANGE, struct.pack("<I", r.tempo_range))
            self.send(OP_SET_TEMPO, struct.pack("<d", r.factor))
            self.send(OP_SET_VOLUME, struct.pack("<d", r.volume))
            for which, position in enumerate((*r.knobs, r.color)):     # TRIM, EQ LOW, MID, HIGH, COLOR
                self.send(OP_SET_MIXER_KNOB, struct.pack("<Id", which, position))
            self.send(OP_SET_CHANNEL_CUE, struct.pack("<I", int(r.channel_cue)))
        self._send_marks()

    def output_devices(self, api: Optional[str] = None) -> tuple[str, list[str], dict[str, int]]:
        return self._engine.devices(api)

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

    # its mixer channel
    def set_knob(self, which: int, position: float) -> None:
        """Its channel's TRIM (0), EQ LOW (1), MID (2), HIGH (3) or COLOR (4) knob at ``position`` (0 .. 1, the middle
        0 dB / off)."""
        self.send(OP_SET_MIXER_KNOB, struct.pack("<Id", int(which), min(1.0, max(0.0, float(position)))))

    def set_volume_trim(self, db: float) -> None:
        """Its level before the fader (Settings > Audio > Mixer's trim): its channel's TRIM knob at ``db`` (held to
        audioengine.volume.TRIM_DB_RANGE)."""
        self.set_knob(0, trim_db_to_knob(db))

    def set_channel_cue(self, on: bool) -> None:
        """Its channel's headphones CUE."""
        self.send(OP_SET_CHANNEL_CUE, struct.pack("<I", int(bool(on))))

    def set_mixer(self, key: str, value: float) -> None:
        """A setting of the mixer's (MIX_KEYS: knobs 0 .. 1, switches 0 / 1, choices their code)."""
        self.send(OP_SET_MIXER, struct.pack("<Id", MIX_KEY[key], float(value)))

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

    def output_channels(self) -> int:
        """The output device's channels (the deck plays stereo on two of them)."""
        st = self._state
        return int(st.out_channels) if st is not None and st.out_channels else 2

    # The MIDI controller (read by the engine)
    def midi_open(self, port: str) -> None:
        """Read the MIDI input ``port`` ("": none)."""
        data = (port or "").encode("utf-8")
        self.setting(OP_MIDI_OPEN, struct.pack("<I", len(data)) + data)

    def midi_map(self, bindings) -> None:
        """``bindings``: (status, data1, flags (BIND_*), lamp, action code) each."""
        payload = struct.pack("<I", len(bindings)) + b"".join(
            struct.pack("<BBBBI", int(st), int(d1), int(flags) & 0xFF, int(lamp), int(code))
            for st, d1, flags, lamp, code in bindings)
        self.setting(OP_MIDI_MAP, payload)

    def set_routing(self, main: tuple, cue: tuple, sub: tuple) -> None:
        """Where each bus goes on the main device: (left, right) channels (0-based; NO_CHANNEL: not there) of the
        master, the headphones (the CUE'd channel and the master, mixed as their knobs set them) and the booth (the
        master again)."""
        chans = [int(c) & 0xFF for pair in (main, cue, sub) for c in pair]
        self.setting(OP_SET_ROUTING, struct.pack("<6B", *chans))

    def set_aux_output(self, bus: int, device: Optional[str], api: str = "wasapi", exclusive: bool = False,
                       buffer_ms: float = 20.0, pair: tuple = (0, 1), rate: int = 0) -> None:
        """Bus ``bus`` (AUX_BUSES' index) on a device of its own: ``device`` (its name, "" the system's default;
        None: none, the bus plays where set_routing puts it, if anywhere), through ``api`` (exclusive mode with
        WASAPI), ``buffer_ms`` kept ahead of it, on its channels ``pair`` (0-based), at ``rate`` (0: the device's
        mix format's; the bus is converted to it)."""
        on = device is not None
        name = (device or "").encode("utf-8")
        lo, hi = self.BUFFER_MS_RANGE
        payload = (struct.pack("<IIIId", int(bus), int(on), API_CODES.get(api, 1), int(bool(exclusive)),
                               float(min(hi, max(lo, buffer_ms))))
                   + struct.pack("<BBHI", int(pair[0]) & 0xFF, int(pair[1]) & 0xFF, 0, max(0, int(rate)))
                   + struct.pack("<I", len(name)) + name)
        self.setting(OP_SET_AUX_OUTPUT, payload, key=int(bus))

    def set_headphones(self, master: bool, mix: float = 0.5, level: float = LEVEL_UNITY) -> None:
        """The headphones: the CUE'd channel before its fader, and the master mixed in by ``mix`` (0 CUE .. 1
        MASTER) with ``master`` (off: a device that mixes the master into its headphones itself, as the DDJ-400 does
        in analog), at ``level``; a controller's CUE / MASTER and level knobs move them later."""
        self.setting(OP_SET_HEADPHONES, struct.pack("<Idd", int(bool(master)), min(1.0, max(0.0, float(mix))),
                                                    max(0.0, float(level))))

    def set_meter_gain(self, gain: float) -> None:
        """The controller's level meter shows the heard peak times this (the GUI peak meter's gain)."""
        self.setting(OP_SET_METER_GAIN, struct.pack("<d", max(0.0, float(gain))))

    def midi_options(self, jog_ticks: Optional[int] = None, vinyl: Optional[bool] = None) -> None:
        """The jog wheel: its messages per turn and its mode (VINYL / CDJ); None: as it was."""
        if jog_ticks is not None:
            self._jog_ticks = max(1, int(jog_ticks))
        if vinyl is not None:
            self._jog_vinyl = bool(vinyl)
        self.setting(OP_MIDI_OPTIONS, struct.pack("<II", self._jog_ticks, int(self._jog_vinyl)))

    def midi_ports(self) -> list[str]:
        return self._engine.midi_ports()

    def midi_events(self) -> list[tuple[int, int, int, int, int, int]]:
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
        mixer = self._engine.mixer()
        if st.cmd_done >= self._engine.sent:
            self._report = DeckReport(st.playing, st.factor, st.volume, st.cue / max(1, self.rate),
                                      st.jump_dest / max(1, self.rate), st.lamps, st.tempo_range, st.midi_state,
                                      st.knobs, bool(st.channel_cue), st.peak, st.color,
                                      mixer[0] if mixer is not None else ())
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

    def restart_engine(self) -> bool:
        """The engine started again by hand (Settings > Audio > Engine: the last resort when a device or the engine
        stays stuck): the old one stopped (ended when it does not answer), a new one on the same output with the
        settings, the track, its place and its play as they were. False when the new one did not start (tried
        again every RESTART_SEC, as a dead one is)."""
        report = self._report
        playing = bool(report.playing) if report is not None else False
        frame = self.playhead_frame()
        self._engine.close()
        self._restart_at = 0.0
        self._restart("restarting it as asked")
        if not self._engine.alive():
            return False
        if self._pcm is not None:
            self.seek(frame)
            if playing:
                self.play()
        return True

    def _restart(self, why: str = "the engine stopped") -> None:
        now = time.monotonic()
        if now < self._restart_at:
            return
        self._restart_at = now + self.RESTART_SEC
        print(f"[AudioEngine] {why}; starting it again")
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
        self._aux_prev = [None] * len(AUX_BUSES)
        self._replay()
        if self._pcm is not None:
            pcm, self._pcm = self._pcm, None
            self.set_track(pcm)

    def playhead_frame(self) -> float:
        return float(self._state.playhead) if self._state is not None else 0.0

    def display_rate(self) -> float:
        """Frames a second display_frame() runs at (0: held)."""
        return float(self._state.display_rate) if self._state is not None else 0.0

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
                         "rate": int(self.rate), "period_ms": float(st.period_ms) if st is not None else 0.0,
                         "channels": int(st.out_channels) if st is not None else 0,
                         "underruns": underruns,
                         "late": int(st.device_late) if st is not None else 0,
                         "max_gap_ms": float(st.max_gap_ms) if st is not None else 0.0,
                         "engine_thread": THREAD_STATES.get(int(st.threads) & 0xFF, "api") if st is not None else "api",
                         "output_thread": THREAD_STATES.get(int(st.threads) >> 8 & 0xFF, "api")
                         if st is not None else "api",
                         "power": POWER_STATES.get(int(st.power), "left") if st is not None else "left"}
        outputs = self._aux_stats()
        if stats is not None:
            stats["outputs"] = outputs
        elif any(o is not None for o in outputs.values()):
            stats = {"outputs": outputs}   # the outputs' state without the main output's observations
        return queued, stats, capacity

    def _aux_stats(self) -> dict:
        """Each bus's own output (AUX_BUSES' names): None where it has none, else its state and, while it plays,
        its buffer's observations since the last call (as the main output's: "hist", "seconds", "mean_ms",
        "dropout_sec", ...)."""
        out = {}
        states = self._engine.aux_states()
        rate = max(1, self.rate)
        for k, (bus, cur) in enumerate(zip(AUX_BUSES, states)):
            prev, self._aux_prev[k] = self._aux_prev[k], cur
            if cur is None:
                out[bus] = None
                continue
            ms = 1000.0 / rate
            d = {"open": cur["state"] == 1, "state": AUX_STATES.get(int(cur["state"]), "failed"),
                 "device": cur["name"], "errors": int(cur["errors"]), "rate": int(cur["rate"]),
                 "output_api": {v: key for key, v in API_CODES.items()}.get(int(cur["mode"]) & 0xFF, "cpal"),
                 "exclusive": bool(int(cur["mode"]) >> 8), "device_ms": float(cur["latency_ms"]),
                 "period_ms": float(cur["period_ms"]), "queued_ms": float(cur["queued"]) * ms,
                 "capacity_ms": float(cur["capacity"]) * ms, "channels": int(cur["channels"]),
                 "drift_ppm": float(cur["drift_ppm"]), "late": int(cur["late"]),
                 "max_gap_ms": float(cur["max_gap_ms"]), "bucket_ms": float(cur["hist_bucket_ms"]),
                 "lowest_ms": float(cur["lowest_ms"]),
                 "output_thread": THREAD_STATES.get(int(cur["thread"]) & 0xFF, "api"), "observed": None}
            if (prev is not None and prev["buffer_gen"] == cur["buffer_gen"]
                    and prev["hist"].size == cur["hist"].size):
                seconds = float(cur["obs_seconds"] - prev["obs_seconds"])
                if seconds > 0.0:
                    d["observed"] = {"hist": cur["hist"] - prev["hist"], "seconds": seconds,
                                     "mean_ms": float(cur["obs_level"] - prev["obs_level"]) / seconds,
                                     "dropout_sec": max(0, int(cur["underruns"]) - int(prev["underruns"])) / rate}
            out[bus] = d
        return out
