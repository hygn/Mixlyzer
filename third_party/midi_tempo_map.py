"""A track's beatgrid as a DAW reads it: a Standard MIDI File (format 0, one track, no notes) holding the tempo map,
the time signatures and the keys, for the Export tab's DAW Tempo Map. The audio dropped at the DAW's start (bar 1)
lines up with it.

- Each beat is a quarter note. The beats are the stored ones (the editor's grid): a tempo event wherever the tempo
  so far would put the next beat more than TOLERANCE_SEC away from it, so a steady stretch is one event, a ramp one
  a beat, and no beat ends up further than that from its place (each new tempo measured from where the file has
  the beat, not from where it should be: the error never adds up).
- Before the first beat, beats of the first beat's length are counted back to the audio's start; what is left at
  the start joins the beat after it (one beat 0.5 to 1.5 of the first's long), or is a beat of its own when it is
  half a beat or more. A first beat closer to the start than half a beat is a short beat: its tempo is fast.
- A bar starts at each downbeat (analyzer.music.beat_geometry.downbeat_beat_indices: the tempo rows' time
  signatures, the bars as the editor shows them): a time signature event wherever the bar's length in beats
  changes (a bar of 3 at a junction, a 7/4 row); the beats before the first downbeat whole bars of the first bar's
  length, a shorter one first (a pickup).
- Each key change (the key rows) a key signature event (sharps or flats, major or minor; an enharmonic key the
  one with fewer accidentals, F# major as 6 sharps) and a marker naming it as the main screen does ("Am (8A)").
- Texts in UTF-8 (the track's name, the markers)."""
from __future__ import annotations

import struct

import numpy as np

from analyzer.music.beat_geometry import downbeat_beat_indices
from analyzer.music.key_names import KEY_DISPLAY_LABELS

PPQ = 960                    # ticks per quarter note (per beat)
TOLERANCE_SEC = 0.0001       # how far the file's beat may be from the grid's before a new tempo event
_MAX_TEMPO_US = 0xFFFFFF     # a tempo event's 24 bits (a beat of 16.8 s)
# a major key's accidentals by its pitch class (C = 0): sharps positive, flats negative
_MAJOR_ACCIDENTALS = (0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5)
# events at one tick in this order: the name, the bar, the tempo, the key, its marker
_ORDER = {0x03: 0, 0x58: 1, 0x51: 2, 0x59: 3, 0x06: 4}


class TempoMapError(ValueError):
    """The track has no beatgrid to write (fewer than two beats)."""


def _vlq(value: int) -> bytes:
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(out))


def _meta(kind: int, data: bytes) -> bytes:
    return bytes((0xFF, kind)) + _vlq(len(data)) + data


def beat_points(beats) -> tuple[np.ndarray, int]:
    """The file's beats in seconds (the beats counted back to the audio's start, then the grid's) and the index of
    the grid's first beat among them."""
    grid = np.sort(np.asarray(beats, dtype=float).ravel())
    grid = grid[np.isfinite(grid)]
    if grid.size < 2:
        raise TempoMapError("fewer than two beats")
    first, period = float(grid[0]), float(grid[1] - grid[0])
    if period <= 0:
        raise TempoMapError("the first two beats at one time")
    before = int(np.floor(first / period + 1e-9)) if first > 0 else 0
    pre = [first - k * period for k in range(before, 0, -1)]
    pre = [p for p in pre if p > 0]
    if pre and pre[0] < 0.5 * period:
        pre = pre[1:]              # the sliver at the start joins the next beat
    points = np.concatenate(([0.0] if first > 0 else [], pre, grid))
    return points, len(points) - grid.size


def tempo_events(points: np.ndarray) -> list[tuple[int, int]]:
    """(tick, microseconds a beat) where the tempo changes, so each beat is within TOLERANCE_SEC of its point."""
    events: list[tuple[int, int]] = []
    at, tempo = 0.0, None             # where the file has beat i, the tempo in force
    for i in range(points.size - 1):
        nxt = float(points[i + 1])
        if tempo is None or abs(at + tempo / 1e6 - nxt) > TOLERANCE_SEC:
            tempo = int(min(_MAX_TEMPO_US, max(1, round((nxt - at) * 1e6))))
            events.append((i * PPQ, tempo))
        at += tempo / 1e6
    return events


def bar_events(grid_first: int, beats, tempo_segments) -> list[tuple[int, int]]:
    """(tick, beats in the bar) where the bar's length changes: a bar from each downbeat (the last as long as the one
    before it), the beats before the first downbeat whole bars of the first bar's length, a shorter one first."""
    downs = sorted(int(d) + grid_first for d in downbeat_beat_indices(beats, tempo_segments))
    if not downs:
        return [(0, 4)]
    bars = [(a, b - a) for a, b in zip(downs, downs[1:])]
    bars.append((downs[-1], bars[-1][1] if bars else 4))
    first = bars[0][1]
    short = downs[0] % first
    lead = ([(0, short)] if short else []) + [(short + k * first, first) for k in range(downs[0] // first)]
    events: list[tuple[int, int]] = []
    for start, length in lead + bars:
        length = max(1, min(255, int(length)))
        if not events or events[-1][1] != length:
            events.append((start * PPQ, length))
    return events


def _tick_of(seconds: float, points: np.ndarray) -> int:
    """The tick at ``seconds`` (between beats in proportion; past the last at its tempo)."""
    if seconds >= points[-1]:
        last = float(points[-1] - points[-2])
        beat = (points.size - 1) + (seconds - points[-1]) / last
    else:
        beat = float(np.interp(seconds, points, np.arange(points.size)))
    return int(round(beat * PPQ))


def key_events(key_segments, points: np.ndarray) -> list[tuple[int, int]]:
    """(tick, key: pitch + 12 * minor) at each key change."""
    rows = np.asarray(key_segments, dtype=float).reshape(-1, 4) if np.size(key_segments) else np.empty((0, 4))
    rows = rows[np.all(np.isfinite(rows[:, :3]), axis=1)]
    events: list[tuple[int, int]] = []
    for pitch, minor, start, _end in rows[np.argsort(rows[:, 2], kind="stable")]:
        key = int(pitch) % 12 + 12 * int(bool(minor))
        if not events or events[-1][1] != key:
            events.append((_tick_of(max(0.0, float(start)), points), key))
    return events


def midi_tempo_map(beats, tempo_segments, key_segments, title: str = "", duration_sec: float = 0.0,
                   speed: float = 1.0) -> bytes:
    """The Standard MIDI File of a track's beats, tempo rows ([start, end, bpm, inizio, ts_num, bpm_end]) and key
    rows ([pitch, minor, start, end]); ``duration_sec`` (0: unknown) ends its track after the audio's end.
    ``speed``: the audio played that many times as fast (time-stretched, the Export tab's rendering): every time
    divided by it, every tempo multiplied (the bars, counted in beats, as they are). TempoMapError without a
    beatgrid."""
    if not speed > 0:
        raise ValueError("speed must be positive")
    points, grid_first = beat_points(np.asarray(beats, dtype=float) / speed)
    if np.size(key_segments):
        key_segments = np.asarray(key_segments, dtype=float).reshape(-1, 4) * (1.0, 1.0, 1.0 / speed, 1.0 / speed)
    duration_sec = float(duration_sec) / speed
    events: list[tuple[int, bytes]] = []
    if title:
        events.append((0, _meta(0x03, title.encode("utf-8"))))
    for tick, length in bar_events(grid_first, beats, tempo_segments):
        events.append((tick, _meta(0x58, bytes((length, 2, 24, 8)))))
    for tick, tempo in tempo_events(points):
        events.append((tick, _meta(0x51, tempo.to_bytes(3, "big"))))
    for tick, key in key_events(key_segments, points):
        minor = key >= 12
        accidentals = _MAJOR_ACCIDENTALS[(key % 12 + 3) % 12 if minor else key % 12]
        events.append((tick, _meta(0x59, struct.pack(">bB", accidentals, int(minor)))))
        events.append((tick, _meta(0x06, KEY_DISPLAY_LABELS[key].encode("utf-8"))))
    events.sort(key=lambda e: (e[0], _ORDER[e[1][1]]))
    end = max([(points.size - 1) * PPQ] + [t for t, _ in events]
              + ([_tick_of(duration_sec, points)] if duration_sec > 0 else []))
    body, previous = bytearray(), 0
    for tick, data in events:
        body += _vlq(tick - previous) + data
        previous = tick
    body += _vlq(end - previous) + _meta(0x2F, b"")
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, PPQ)
    return header + b"MTrk" + struct.pack(">I", len(body)) + bytes(body)
