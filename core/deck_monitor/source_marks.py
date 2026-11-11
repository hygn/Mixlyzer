"""The sender's own analysis of a deck's track (OSC Full, Pro DJ Link): its beat grid, CUE points and key,
shown for a track the library does not hold, through the app's own views (a StoredAnalysis without audio).

The grid is a list of tempo changes, rekordbox XML's TEMPO: from ``start`` (a beat, seconds) on at ``bpm``, ``beat``
that beat's place in its bar (1 = a downbeat, 4/4). A protocol giving every beat (Pro DJ Link) is read into
the same changes (``grid_from_beats``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from analyzer.music.key_names import ALPHANUMERIC_LABELS, CLASSICAL_LABELS
from analyzer.music.marks import CuePoint
from analyzer.music.tempo_rows import rebuild_grid_from_segments
from core.analysis_library.stored import StoredAnalysis
from core.marks.colors import CUE_POINT_COLORS

METER = 4                     # the senders' grids are 4/4 (rekordbox's beat numbers 1 - 4)
MAX_CHANGES = 20000           # a grid's tempo changes at most (a beat each: a long live set)
MAX_CUES = 1000
MAX_BPM = 999.99
# beats in a row are one tempo while their spacing stays this close to the first's (a source rounding beats to the
# millisecond moves one by 0.2 % at 128 BPM; 1 BPM more is 0.8 %)
SAME_TEMPO = 0.005


@dataclass(frozen=True)
class TempoChange:
    start: float                  # a beat (s)
    bpm: float
    beat: int = 1                 # its place in its bar, 1 ..= METER


@dataclass(frozen=True)
class SourceCue:
    seconds: float
    name: str = ""
    color: Optional[tuple[int, int, int]] = None     # None: the sender gives none


@dataclass(frozen=True)
class SourceMarks:
    grid: tuple[TempoChange, ...] = ()               # by start
    cues: tuple[SourceCue, ...] = ()                 # by time
    key: Optional[int] = None                        # 0 ..= 23 (analyzer.music.key_names' order)

    def empty(self) -> bool:
        return not self.grid and not self.cues and self.key is None


_FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}


def parse_key(text: str) -> Optional[int]:
    """A key as a sender names it, Alphanumeric (8A) or classical (Am, C#m, Dbmin, F# major), as 0 ..= 23; None
    when it names none."""
    text = (text or "").strip()
    if not text:
        return None
    if text.upper() in ALPHANUMERIC_LABELS:
        return ALPHANUMERIC_LABELS.index(text.upper())
    lowered = text.replace(" ", "")
    minor = False
    for suffix, is_minor in (("minor", True), ("major", False), ("min", True), ("maj", False), ("m", True)):
        if lowered.lower().endswith(suffix) and len(lowered) > len(suffix):
            lowered, minor = lowered[: -len(suffix)], is_minor
            break
    note = lowered[:1].upper() + lowered[1:].replace("♯", "#").replace("♭", "b")
    note = _FLATS.get(note, note)
    if note not in CLASSICAL_LABELS[:12]:
        return None
    return CLASSICAL_LABELS.index(note) + (12 if minor else 0)


def cue_color_index(color: Optional[tuple[int, int, int]]) -> int:
    """The CUE point colour preset nearest a sender's colour (core.marks.colors; none: the first)."""
    if color is None:
        return 0
    return min(range(len(CUE_POINT_COLORS)),
               key=lambda i: sum((a - b) ** 2 for a, b in zip(CUE_POINT_COLORS[i], color)))


def grid_from_beats(times: Sequence[float], bar_beats: Sequence[int]) -> tuple[TempoChange, ...]:
    """Every beat's time (s) and place in its bar (1 = a downbeat) read into tempo changes: beats in a row spaced
    alike (SAME_TEMPO) are one, its tempo the line through them (a source's rounding of each beat evened out)."""
    t = np.asarray(times, dtype=float)
    places = np.asarray(bar_beats, dtype=int)
    if t.size < 2 or places.size != t.size or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        return ()
    changes: list[TempoChange] = []
    first = 0
    while first < t.size - 1:
        gap = t[first + 1] - t[first]
        last = first + 1
        while last + 1 < t.size and abs((t[last + 1] - t[last]) - gap) <= SAME_TEMPO * gap:
            last += 1
        k = np.arange(last - first + 1, dtype=float)
        period, start = np.polyfit(k, t[first:last + 1], 1) if last > first + 1 else (gap, t[first])
        bpm = 60.0 / float(period)
        if 0.0 < bpm <= MAX_BPM:
            place = int(places[first])
            changes.append(TempoChange(float(start), bpm, place if 1 <= place <= METER else 1))
        first = last
    return tuple(changes[:MAX_CHANGES])


def grid_rows(grid: Sequence[TempoChange], duration: float) -> np.ndarray:
    """The app's tempo rows ``[start, end, bpm, inizio, ts, bpm_end]`` of a grid on a track ``duration`` long."""
    rows = []
    changes = [c for c in grid if math.isfinite(c.start) and 0.0 < c.bpm <= MAX_BPM and c.start < duration]
    for i, change in enumerate(changes):
        end = changes[i + 1].start if i + 1 < len(changes) else duration
        if end <= change.start:
            continue
        period = 60.0 / change.bpm
        inizio = change.start + ((METER - change.beat + 1) % METER) * period     # its first downbeat
        rows.append([change.start, end, change.bpm, inizio, float(METER), change.bpm])
    return np.asarray(rows, dtype=float).reshape(-1, 6)


def source_analysis(marks: SourceMarks, duration: float) -> Optional[StoredAnalysis]:
    """The sender's marks as the app's analysis of a track ``duration`` long (None: its length not known)."""
    if not duration or duration <= 0.0 or not math.isfinite(duration):
        return None
    rows = grid_rows(marks.grid, duration)
    beats = rebuild_grid_from_segments(rows) if rows.size else np.empty(0)
    keys = (np.array([[marks.key % 12, 1.0 if marks.key >= 12 else 0.0, 0.0, duration]])
            if marks.key is not None else np.empty((0, 4)))
    cues = tuple(CuePoint(time_sec=c.seconds, label=c.name, color_idx=cue_color_index(c.color))
                 for c in sorted(marks.cues, key=lambda c: c.seconds) if 0.0 <= c.seconds <= duration)
    main = float(rows[0, 2]) if rows.size else 0.0
    return StoredAnalysis(duration_sec=float(duration), main_tempo=main, beats=beats, tempo_segments=rows,
                          key_segments=keys, cue_points=cues)
