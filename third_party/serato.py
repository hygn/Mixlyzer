"""A track's analysis as Serato DJ keeps it in the audio file's tags (no import file of its own): the beatgrid
("Serato BeatGrid") and the hot cues ("Serato Markers2"), as the payloads core.tags.writer puts in each format (an
ID3 GEOB frame's data; FLAC's and MP4's base64 of them). Laid out after the community's reverse engineering
(Holzhaus/serato-tags, Mixxx's SeratoBeatGrid / SeratoMarkers2):

- BeatGrid: 01 00, the markers' count (u32 BE), each marker but the last its position (f32 s) and the beats to the
  next marker (u32), the last its position and its BPM (f32), a footer byte. Serato spaces the beats between two
  markers evenly and runs the last BPM on; it counts bars from the first marker (no bar restarts later, every bar 4
  beats). So the first marker is the first downbeat, a steady stretch one marker, a ramp a marker a beat (each where
  even spacing would put a beat more than TOLERANCE_SEC away); the beats before the first downbeat are Serato's
  run back at the first marker's tempo.
- Markers2: 01 01, then base64 (no padding, a line break every 72 characters) of 01 01 and its entries (a name, NUL,
  the data's length u32 BE, the data), and a NUL, padded with NULs to MARKERS2_MIN bytes. CUE: 00, its index (0-7),
  its position (ms u32), 00, its colour (RGB), 00 00, its name (NAME_MAX bytes), NUL. BPMLOCK: 1 (Serato keeps the
  grid: no reanalysis changes it). The file's other entries (saved loops, the track's colour, Flips) are kept as
  they are; its hot cues are replaced. The old Markers_ (its first 5 cues and 9 loops, no names), which Serato reads
  before Markers2, is emptied with it (core.tags.writer's serato_markers_old).
- Tempo Map (newer Serato versions write it when they analyze a track, and read it before the BeatGrid; its layout
  found from the ones Serato wrote: a track with 15 tempo segments, single-segment ones): 02 00, 00 01, the time of
  beat 0 (f64 s), the segments' count (u32), each segment its first beat in TICKS (u64), its BPM (f64), 0.25 (f64:
  the beat a quarter note), 01, 0 (u64); a beat's time is beat 0's plus the segments' beats at their tempo. Written
  with the BeatGrid, from the same markers.
- Times: Serato decodes a lossy file with another start than FFmpeg (Mixlyzer's times): each time is written in
  Serato's (time_offset, measured per format).

The hot cues (Settings > Library > Serato: libconfig.serato_hot_cues, serato_keep_cues, serato_bpm_lock) are the
JumpCUEs (their letter A-H the slot, as the Rekordbox XML has them; their group's colour the nearest of Serato's;
their name: its comment, else the letter), the CUE points (Serato has no memory cues: in time order, their colour
and label), or both (the CUE points in the slots the JumpCUEs leave); the file's own hot cues kept in the slots
left, or all replaced. What Serato cannot hold is listed (``losses``): the CUE points not written, a JumpCUE past
H, CUE points with no slot left, the bars that are not 4 beats from the first downbeat."""
from __future__ import annotations

import base64
import re
import struct
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from analyzer.music.beat_geometry import downbeat_beat_indices
from analyzer.music.marks import JumpCueGroup
from core.marks.colors import get_cue_point_color, jump_cue_colors

TOLERANCE_SEC = 0.0005       # how far even spacing may put a beat from the grid's before another marker
SLOTS = 8
MARKERS2_MIN = 470           # a Markers2 payload's length at least (Serato pads it with NULs)
NAME_MAX = 50                # a hot cue name's bytes at most (51 with its NUL)
# hot cue colours as the tag stores them (Serato shows brighter versions of these)
CUE_COLORS = (
    (0xCC, 0x00, 0x00), (0xCC, 0x44, 0x00), (0xCC, 0x88, 0x00), (0xCC, 0xCC, 0x00),
    (0x88, 0xCC, 0x00), (0x44, 0xCC, 0x00), (0x00, 0xCC, 0x00), (0x00, 0xCC, 0x44),
    (0x00, 0xCC, 0x88), (0x00, 0xCC, 0xCC), (0x00, 0x88, 0xCC), (0x00, 0x44, 0xCC),
    (0x00, 0x00, 0xCC), (0x44, 0x00, 0xCC), (0x88, 0x00, 0xCC), (0xCC, 0x00, 0xCC),
    (0xCC, 0x00, 0x88), (0xCC, 0x00, 0x44),
)


@dataclass(frozen=True)
class HotCue:
    slot: int                       # 0-7
    time_sec: float
    color: tuple[int, int, int]     # one of CUE_COLORS
    name: str
    kept: bool = False              # one of the file's kept (hot_cues' keep), not made of the marks


# --- Serato's time ---------------------------------------------------------------------------------------------

# Serato's time against Mixlyzer's (FFmpeg's decode), measured on click tracks (a click every 0.5 s from 1.000 s)
# Serato DJ analyzed: AAC 2112 samples earlier whether the file has an edit list or not; MP3 with a Xing / Info
# frame (FFmpeg drops its encoder delay) 50.0 ms later at 44.1 kHz, 44.7 at 48; MP3 without one 5.1 ms earlier;
# FLAC, WAV, AIFF the same
_AAC_SAMPLES = -2112
_MP3_GAPLESS_MS = {44100: 50.0, 48000: 44.74}
_MP3_PLAIN_MS = -5.06


def _mp3_gapless(path) -> bool:
    """The MP3's first frame is a Xing / Info frame with a LAME tag of an encoder FFmpeg takes the delay from (LAME,
    Lavf, Lavc): FFmpeg drops that encoder delay."""
    with open(path, "rb") as f:
        head = f.read(10)
        offset = 0
        while len(head) == 10 and head[:3] == b"ID3":
            size = ((head[6] & 0x7F) << 21) | ((head[7] & 0x7F) << 14) | ((head[8] & 0x7F) << 7) | (head[9] & 0x7F)
            offset += 10 + size + (10 if head[5] & 0x10 else 0)
            f.seek(offset)
            head = f.read(10)
        f.seek(offset)
        frame = f.read(4096)
    start = next((i for i in range(len(frame) - 1) if frame[i] == 0xFF and frame[i + 1] & 0xE0 == 0xE0), -1)
    if start < 0:
        return False
    at = next((frame.find(tag, start, start + 64) for tag in (b"Xing", b"Info")
               if frame.find(tag, start, start + 64) >= 0), -1)
    if at < 0 or at + 8 > len(frame):
        return False
    flags = struct.unpack(">I", frame[at + 4:at + 8])[0]
    at += 8 + 4 * bool(flags & 1) + 4 * bool(flags & 2) + 100 * bool(flags & 4) + 4 * bool(flags & 8)
    return frame[at:at + 4] in (b"LAME", b"Lavf", b"Lavc")


def time_offset(path, sample_rate: int) -> float:
    """Serato's time minus Mixlyzer's in the file ``path`` (s): what is added to a time written for Serato."""
    suffix = str(path).lower().rsplit(".", 1)[-1]
    if suffix in ("m4a", "mp4", "aac", "m4b"):
        return _AAC_SAMPLES / float(sample_rate or 44100)
    if suffix == "mp3":
        if _mp3_gapless(path):
            ms = _MP3_GAPLESS_MS.get(int(sample_rate), 50.0 * 44100.0 / float(sample_rate or 44100))
        else:
            ms = _MP3_PLAIN_MS * 44100.0 / float(sample_rate or 44100)
        return ms / 1000.0
    return 0.0


def shifted(markers: list[tuple[float, float]], offset: float) -> list[tuple[float, float]]:
    """The grid's markers in Serato's time (``offset``: time_offset's)."""
    return [(pos + offset, value) for pos, value in markers]


# --- BeatGrid ------------------------------------------------------------------------------------------------

def grid_markers(beats, tempo_segments) -> tuple[list[tuple[float, float]], list[str]]:
    """The grid's markers: (position s, beats to the next marker) for each but the last, (position s, BPM) for the
    last; and what Serato loses of it. [] without two beats."""
    grid = np.sort(np.asarray(beats, dtype=float).ravel())
    grid = grid[np.isfinite(grid)]
    if grid.size < 2:
        return [], []
    downs = downbeat_beat_indices(grid, tempo_segments)
    first = int(downs[0]) if downs.size else 0
    if grid.size - first < 2:
        first = 0
    pts = grid[first:]
    spans, i = [], 0
    while i < pts.size - 1:
        j = i + 1
        while j + 1 < pts.size:
            even = np.linspace(pts[i], pts[j + 1], j + 2 - i)
            if float(np.max(np.abs(even - pts[i:j + 2]))) > TOLERANCE_SEC:
                break
            j += 1
        spans.append((i, j))
        i = j
    markers: list[tuple[float, float]] = [(float(pts[a]), float(b - a)) for a, b in spans[:-1]]
    a, b = spans[-1]
    markers.append((float(pts[a]), 60.0 * (b - a) / float(pts[b] - pts[a])))
    # Serato's bars: every 4 beats from the first marker; a downbeat off them is a bar Serato cannot hold
    losses = ["bars"] if any((int(d) - first) % 4 for d in downs if d >= first) else []
    return markers, losses


def build_beatgrid(markers: list[tuple[float, float]]) -> bytes:
    out = b"\x01\x00" + struct.pack(">I", len(markers))
    for k, (pos, value) in enumerate(markers):
        out += struct.pack(">fI", pos, int(value)) if k < len(markers) - 1 else struct.pack(">ff", pos, value)
    return out + b"\x00"


TICKS = 960     # a Tempo Map's position: its beat counted in 960ths


def build_tempo_map(markers: list[tuple[float, float]]) -> bytes:
    """The Tempo Map of the grid ``markers`` (grid_markers'): its first marker the time of beat 0, then a segment
    from each marker (its first beat in TICKS, its BPM: the even spacing up to the next marker, the last's own)."""
    out = struct.pack(">HHdI", 0x0200, 1, markers[0][0], len(markers))
    tick = 0
    for k, (pos, value) in enumerate(markers):
        last = k == len(markers) - 1
        bpm = value if last else 60.0 * value / (markers[k + 1][0] - pos)
        out += struct.pack(">QddBQ", tick, bpm, 0.25, 1, 0)
        if not last:
            tick += int(value) * TICKS
    return out


def parse_tempo_map(payload: bytes) -> tuple[float, list[tuple[int, float]]]:
    """(the time of beat 0, [(the segment's first beat, its BPM)]) of a Tempo Map; (0, []) for one not read."""
    if len(payload) < 16 or payload[:2] != b"\x02\x00":
        return 0.0, []
    _version, _one, anchor, count = struct.unpack(">HHdI", payload[:16])
    if 16 + 33 * count > len(payload):
        return 0.0, []
    out = []
    for k in range(count):
        tick, bpm = struct.unpack(">Qd", payload[16 + 33 * k:32 + 33 * k])
        out.append((tick // TICKS, bpm))
    return anchor, out


def tempo_map_beats(payload: bytes, until: float) -> np.ndarray:
    """The beat times a Tempo Map gives up to ``until`` (s)."""
    anchor, segments = parse_tempo_map(payload)
    out, t = [], anchor
    for k, (beat, bpm) in enumerate(segments):
        end = segments[k + 1][0] if k + 1 < len(segments) else None
        n = (end - beat) if end is not None else int(max(0.0, until - t) * bpm / 60.0) + 1
        out += list(t + 60.0 / bpm * np.arange(n))
        t += 60.0 / bpm * n
    return np.array(out)


def parse_beatgrid(payload: bytes) -> list[tuple[float, float]]:
    """The markers of a BeatGrid payload (as ``grid_markers`` gives them); [] for one not read."""
    if len(payload) < 6 or payload[:2] != b"\x01\x00":
        return []
    (n,) = struct.unpack(">I", payload[2:6])
    if 6 + 8 * n > len(payload):
        return []
    out = []
    for k in range(n):
        chunk = payload[6 + 8 * k:14 + 8 * k]
        pos, value = struct.unpack(">fI", chunk) if k < n - 1 else struct.unpack(">ff", chunk)
        out.append((float(pos), float(value)))
    return out


# --- Markers2 ------------------------------------------------------------------------------------------------

def _slot(label: str) -> Optional[int]:
    m = re.match(r"\s*([A-H])\b", (label or "").upper())
    return ord(m.group(1)) - ord("A") if m else None


def _nearest(rgb) -> tuple[int, int, int]:
    return min(CUE_COLORS, key=lambda c: sum((int(a) - b) ** 2 for a, b in zip(rgb, c)))


HOT_CUE_SOURCES = ("jumpcues", "cue_points", "both")


def hot_cues(groups: list[JumpCueGroup], cue_points: list = (), source: str = "jumpcues", existing: bytes = b"",
             keep: bool = True, offset: float = 0.0) -> tuple[list[HotCue], list[str]]:
    """The hot cues (8 slots) and what Serato loses of the marks. ``source``: the JumpCUEs (``groups``: A-H in their
    slots, "jumpcues"), the CUE points (``cue_points``: CuePoint, in time order from the first slot, "cue_points"),
    or both (the CUE points in the slots the JumpCUEs leave, "both"). ``keep``: the ``existing`` Markers2's hot cues
    kept in the slots still free (they are in Serato's time already; ``kept`` set: the file's own and those an
    earlier write put there alike, so a mark removed since stays); the cues written in Serato's time (``offset``:
    time_offset's). Losses: "slots" (a JumpCUE past H), "cue_points" (the CUE points not written), "full" (CUE
    points with no slot left)."""
    source = source if source in HOT_CUE_SOURCES else "jumpcues"
    cues: dict[int, HotCue] = {}
    losses = []
    if source != "cue_points":
        past = 0
        for cue, color in jump_cue_colors(list(groups)):
            slot = _slot(cue.label)
            if slot is None or slot >= SLOTS or not np.isfinite(cue.point):
                past += 1
                continue
            name = (cue.comment or "").strip() or cue.label.strip()
            cues[slot] = HotCue(slot, max(0.0, float(cue.point) + offset), _nearest(color), name)
        if past:
            losses.append("slots")
    points = sorted((p for p in cue_points if np.isfinite(p.time_sec)), key=lambda p: p.time_sec)
    if source == "jumpcues":
        if points:
            losses.append("cue_points")
    else:
        free = [s for s in range(SLOTS) if s not in cues]
        for slot, point in zip(free, points):
            color = _nearest(get_cue_point_color(point.color_idx))
            cues[slot] = HotCue(slot, max(0.0, float(point.time_sec) + offset), color,
                                str(point.label or point.comment or "").strip())
        if len(points) > len(free):
            losses.append("full")
    if keep:
        for old in parse_cues(existing):
            if 0 <= old.slot < SLOTS and old.slot not in cues:
                cues[old.slot] = replace(old, kept=True)
    return [cues[s] for s in sorted(cues)], losses


def _ms(seconds: float) -> int:
    """A hot cue's position as Markers2 holds it (ms)."""
    return int(round(seconds * 1000.0))


def _entry(name: str, data: bytes) -> bytes:
    return name.encode("ascii") + b"\x00" + struct.pack(">I", len(data)) + data


def _entries(payload: bytes) -> list[tuple[str, bytes]]:
    """The entries of a Markers2 payload ([] for one not read)."""
    if payload[:2] != b"\x01\x01":
        return []
    b64 = payload[2:].split(b"\x00", 1)[0].replace(b"\n", b"")
    try:
        body = base64.b64decode(b64 + b"=" * (-len(b64) % 4))
    except ValueError:
        return []
    if body[:2] != b"\x01\x01":
        return []
    out, i = [], 2
    while i < len(body) and body[i] != 0:
        end = body.find(b"\x00", i)
        if end < 0 or end + 5 > len(body):
            break
        (n,) = struct.unpack(">I", body[end + 1:end + 5])
        if end + 5 + n > len(body):
            break
        out.append((body[i:end].decode("ascii", "replace"), body[end + 5:end + 5 + n]))
        i = end + 5 + n
    return out


def build_markers2(cues: list[HotCue], existing: bytes = b"", bpm_lock: bool = True) -> bytes:
    """A Markers2 payload: ``cues`` and BPMLOCK (``bpm_lock``), the ``existing`` payload's other entries kept (its
    loops, the track's colour, Flips) in their order; its hot cues are ``cues``' (hot_cues keeps them)."""
    kept = [(n, d) for n, d in _entries(existing) if n not in ("CUE", "BPMLOCK")]
    body = b"\x01\x01"
    for name, data in [(n, d) for n, d in kept if n == "COLOR"]:
        body += _entry(name, data)
    for c in cues:
        data = (b"\x00" + bytes([c.slot]) + struct.pack(">I", _ms(c.time_sec)) + b"\x00"
                + bytes(c.color) + b"\x00\x00" + _name(c.name) + b"\x00")
        body += _entry("CUE", data)
    for name, data in [(n, d) for n, d in kept if n != "COLOR"]:
        body += _entry(name, data)
    body += _entry("BPMLOCK", b"\x01" if bpm_lock else b"\x00") + b"\x00"
    b64 = base64.b64encode(body).rstrip(b"=")
    payload = b"\x01\x01" + b"\n".join(b64[i:i + 72] for i in range(0, len(b64), 72)) + b"\x00"
    return payload.ljust(MARKERS2_MIN, b"\x00")


def _name(name: str) -> bytes:
    """A hot cue's name as Serato takes it: UTF-8, NAME_MAX bytes at most (cut between characters)."""
    raw = name.encode("utf-8")
    while len(raw) > NAME_MAX:
        name = name[:-1]
        raw = name.encode("utf-8")
    return raw


def parse_cues(payload: bytes) -> list[HotCue]:
    """The hot cues of a Markers2 payload."""
    out = []
    for name, data in _entries(payload):
        if name == "CUE" and len(data) >= 13:
            slot, pos = data[1], struct.unpack(">I", data[2:6])[0]
            label = data[12:].split(b"\x00", 1)[0].decode("utf-8", "replace")
            out.append(HotCue(slot, pos / 1000.0, tuple(data[7:10]), label))
    return out


def bpm_locked(payload: bytes) -> bool:
    return any(n == "BPMLOCK" and d[:1] == b"\x01" for n, d in _entries(payload))
