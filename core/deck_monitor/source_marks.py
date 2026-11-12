"""The sender's own analysis of a deck's track (OSC Full, Pro DJ Link): its beat grid, cues and key, shown for a
track the library does not hold, through the app's own views (a StoredAnalysis without audio); for a track it holds,
the sender's cues shown in place of Mixlyzer's when one differs, as set (``with_sender_cues``).

The grid is a list of tempo changes, rekordbox XML's TEMPO: from ``start`` (a beat, seconds) on at ``bpm``, ``beat``
that beat's place in its bar (1 = a downbeat, 4/4). A protocol giving every beat (Pro DJ Link) is read into
the same changes (``grid_from_beats``). The cues are rekordbox's: a hot cue (its slot, rekordbox XML's Num) is a
JumpCUE, a memory cue a CUE point, as the app's export writes them (third_party.rekordbox)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np

from analyzer.music.key_names import ALPHANUMERIC_LABELS, CLASSICAL_LABELS
from analyzer.music.marks import CuePoint, JumpCue, JumpCueGroup
from analyzer.music.tempo_rows import rebuild_grid_from_segments
from core.analysis_library.stored import StoredAnalysis
from core.marks.colors import CUE_POINT_COLORS, JUMPCUE_PAIR_COLORS
from core.marks.jump_cues import SLOT_LETTERS

METER = 4                     # the senders' grids are 4/4 (rekordbox's beat numbers 1 - 4)
MAX_CHANGES = 20000           # a grid's tempo changes at most (a beat each: a long live set)
MAX_BEATS = 20000             # a grid's beats at most, as a source sending each one gives them
MAX_CUES = 1000
MAX_BPM = 999.99
# beats in a row are one tempo while their spacing stays this close to the first's (a source rounding beats to the
# millisecond moves one by 0.2 % at 128 BPM; 1 BPM more is 0.8 %)
SAME_TEMPO = 0.005
MEMORY = -1                   # a cue's slot when it is a memory cue (rekordbox XML's Num -1)
# a sender's cue is at Mixlyzer's within this (rekordbox XML's 1 ms, a nexus player's 1/150 s)
SAME_CUE_SEC = 0.01
MIXLYZER, SENDER = "mixlyzer", "sender"     # whose cues a library track's deck shows first (deckconfig)
FIRSTS = (MIXLYZER, SENDER)


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
    slot: int = MEMORY            # a hot cue's slot (0: A, 1: B ...: rekordbox XML's Num), MEMORY: a memory cue


@dataclass(frozen=True)
class SourceMarks:
    grid: tuple[TempoChange, ...] = ()               # by start
    # by time: its hot cues and memory cues; None: the sender gives no cue list (() it gives one with none)
    cues: Optional[tuple[SourceCue, ...]] = None
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


def _nearest(presets: Sequence[tuple[int, int, int]], color: Optional[tuple[int, int, int]]) -> int:
    if color is None:
        return 0
    return min(range(len(presets)), key=lambda i: sum((a - b) ** 2 for a, b in zip(presets[i], color)))


def cue_color_index(color: Optional[tuple[int, int, int]]) -> int:
    """The CUE point colour preset nearest a sender's colour (core.marks.colors; none: the first)."""
    return _nearest(CUE_POINT_COLORS, color)


def jump_color_index(color: Optional[tuple[int, int, int]]) -> int:
    """The JumpCUE colour preset nearest a sender's colour (core.marks.colors; none: the first)."""
    return _nearest(JUMPCUE_PAIR_COLORS, color)


def memory_cue_points(cues: Optional[Sequence[SourceCue]], duration: float) -> tuple[CuePoint, ...]:
    """The sender's memory cues within the track as CUE points, by time."""
    return tuple(CuePoint(time_sec=c.seconds, label=c.name, color_idx=cue_color_index(c.color))
                 for c in sorted(cues or (), key=lambda c: c.seconds)
                 if c.slot == MEMORY and 0.0 <= c.seconds <= duration)


def hot_cue_groups(cues: Optional[Sequence[SourceCue]], duration: float) -> tuple[JumpCueGroup, ...]:
    """The sender's hot cues within the track as JumpCUEs (its slot the letter, its name the comment, the letter where
    it has none), a group per colour as the app's export gives a group's cues its colour; a slot taken once (its
    first by time)."""
    groups: dict[int, list[JumpCue]] = {}
    taken = set()
    for c in sorted(cues or (), key=lambda c: c.seconds):
        if not 0 <= c.slot < len(SLOT_LETTERS) or c.slot in taken or not 0.0 <= c.seconds <= duration:
            continue
        taken.add(c.slot)
        letter = SLOT_LETTERS[c.slot]
        groups.setdefault(jump_color_index(c.color), []).append(
            JumpCue(label=letter, comment=c.name or letter, start=c.seconds, end=c.seconds, point=c.seconds))
    return tuple(JumpCueGroup(cues=tuple(members), color_idx=index) for index, members in groups.items())


def _same(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> bool:
    """Two sides' cues at the same places: as many, each (slot letter, time) within SAME_CUE_SEC of the other's."""
    return len(a) == len(b) and all(x[0] == y[0] and abs(x[1] - y[1]) <= SAME_CUE_SEC
                                    for x, y in zip(sorted(a), sorted(b)))


def _first(own: tuple, sent: tuple, first: str, places: Callable[[tuple], list[tuple[str, float]]]) -> tuple:
    """Mixlyzer's marks ``own`` or the sender's ``sent``. SENDER: the sender's, all of them (none where it has none),
    when one differs, so what is shown is what the device's buttons play (Mixlyzer's when every one is at the same
    place: their names and groups); MIXLYZER: Mixlyzer's, the sender's where it has none."""
    if first == SENDER:
        return own if _same(places(own), places(sent)) else sent
    return own if own or not sent else sent


def with_sender_cues(analysis: StoredAnalysis, marks: Optional[SourceMarks], jump_cues_first: str,
                     cue_points_first: str) -> StoredAnalysis:
    """A library track's analysis with the sender's hot cues as its JumpCUEs and its memory cues as its CUE points
    as set (``FIRSTS``, ``_first``); ``analysis`` itself where nothing changes (the views not loaded again) or the
    sender gives no cue list."""
    if marks is None or marks.cues is None:
        return analysis
    duration = float(analysis.duration_sec or 0.0)
    jump_cues = _first(analysis.jump_cues, hot_cue_groups(marks.cues, duration), jump_cues_first,
                       lambda groups: [(str(c.label).strip().upper(), float(c.point)) for g in groups for c in g.cues])
    cue_points = _first(analysis.cue_points, memory_cue_points(marks.cues, duration), cue_points_first,
                        lambda points: [("", p.time_sec) for p in points])
    if jump_cues is analysis.jump_cues and cue_points is analysis.cue_points:
        return analysis
    return replace(analysis, jump_cues=jump_cues, cue_points=cue_points)


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
    main = float(rows[0, 2]) if rows.size else 0.0
    return StoredAnalysis(duration_sec=float(duration), main_tempo=main, beats=beats, tempo_segments=rows,
                          key_segments=keys, cue_points=memory_cue_points(marks.cues, duration),
                          jump_cues=hot_cue_groups(marks.cues, duration))
