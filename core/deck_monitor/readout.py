"""What a deck's panel reads out of its track at its position (Deck Monitor's window): the tempo there, the key there,
and the phrase it is in and the next one, by section (each phrase) or by label (a run of phrases with one label, a
chorus of two eight-bar phrases: CHORUS 16): each's name and colour, its length in bars, and the beats left until
the next one (shown as bars.beats, as rekordbox counts down to a cue)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from analyzer.music.beat_geometry import downbeat_beat_indices
from analyzer.music.marks import Phrase
from analyzer.music.tempo_rows import bpm_at as row_bpm_at
from core.marks.phrases import UNASSIGNED, normalize_phrase_segments, number_phrase_labels, phrase_color

EPS = 1e-3       # a beat this close to a phrase's start is on it


def bpm_at(tempo_rows, t: float) -> Optional[float]:
    """The tempo at ``t`` (a ramp's as it is there; before the first row its start, after the last its end), None
    without rows."""
    rows = np.asarray(tempo_rows, dtype=float) if tempo_rows is not None else np.empty((0, 6))
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 3:
        return None
    i = int(np.searchsorted(rows[:, 0], float(t), side="right")) - 1
    i = min(max(i, 0), rows.shape[0] - 1)
    row = rows[i]
    bpm = row_bpm_at(row, min(max(float(t), float(row[0])), float(row[1])))
    return bpm if math.isfinite(bpm) and bpm > 0.0 else None


def key_at(key_rows: Sequence[tuple[int, float, float]], t: float) -> Optional[int]:
    """The key index (0 .. 23) of the key row holding ``t`` (views.render.keystrip.key_segment_rows' rows)."""
    for idx, start, end in key_rows:
        if start <= t < end:
            return int(idx)
    return None


SECTION, LABEL = "section", "label"     # the phrase now and next: each phrase, or each run of one label
UNITS = (SECTION, LABEL)


@dataclass(frozen=True)
class PhraseSpan:
    label: str                   # as stored
    name: str                    # as shown (numbered: CHORUS2)
    color: tuple[int, int, int]
    start: float
    end: float
    bars: int                    # its length


@dataclass(frozen=True)
class PhraseNow:
    now: Optional[PhraseSpan]
    next: Optional[PhraseSpan]
    left_beats: Optional[int]    # beats until the next one starts (None: no next, or no beats)
    meter: int                   # beats in the bar at the position


class TrackPhrases:
    """A track's phrases with what is read of them once (their names and bars), as sections (each phrase) and as
    label runs (the phrases of one label in a row as one: a chorus of two eight-bar phrases, CHORUS 16): ``at(t,
    unit)`` the one now and the next. ``phrases`` cover the track in time order (the model's), ``beats`` and
    ``tempo_rows`` its grid."""

    def __init__(self, phrases, beats, tempo_rows) -> None:
        sections = normalize_phrase_segments(phrases or ())    # as number_phrase_labels names them
        self.beats = np.asarray(beats, dtype=float) if beats is not None else np.empty(0)
        rows = np.asarray(tempo_rows, dtype=float) if tempo_rows is not None else np.empty((0, 6))
        down = downbeat_beat_indices(self.beats, rows) if self.beats.size and rows.size else np.empty(0, dtype=int)
        self.down_idx = np.asarray(down, dtype=int)
        self.downbeats = self.beats[self.down_idx] if self.down_idx.size else np.empty(0)
        runs = []
        for p in sections:
            if runs and runs[-1].label == p.label and abs(runs[-1].end - p.start) < EPS:
                runs[-1] = Phrase(runs[-1].start, p.end, p.label)
            else:
                runs.append(Phrase(p.start, p.end, p.label))
        self.spans = {SECTION: self._spans(sections), LABEL: self._spans(runs)}
        self.starts = {unit: np.array([s.start for s in spans], dtype=float) for unit, spans in self.spans.items()}

    def _spans(self, phrases: list) -> list[PhraseSpan]:
        """The phrases as shown, UNASSIGNED left out (where none is: no phrase now, none next)."""
        names = number_phrase_labels(phrases) if phrases else []
        return [PhraseSpan(p.label, names[k] if k < len(names) else p.label, tuple(phrase_color(p.label)),
                           float(p.start), float(p.end), self._bars(p.start, p.end)) for k, p in enumerate(phrases)
                if p.label != UNASSIGNED]

    def _bars(self, start: float, end: float) -> int:
        """The bars starting in [start, end) (a phrase starts on a downbeat)."""
        if not self.downbeats.size:
            return 0
        return int(np.searchsorted(self.downbeats, end - EPS) - np.searchsorted(self.downbeats, start - EPS))

    def _meter(self, t: float) -> int:
        d = self.down_idx
        if d.size < 2:
            return 4
        i = int(np.searchsorted(self.downbeats, t, side="right")) - 1
        i = min(max(i, 0), d.size - 2)
        return max(1, int(d[i + 1] - d[i]))

    def at(self, t: float, unit: str = SECTION) -> PhraseNow:
        meter = self._meter(t)
        unit = unit if unit in self.spans else SECTION
        spans = self.spans[unit]
        if not spans:
            return PhraseNow(None, None, None, meter)
        i = int(np.searchsorted(self.starts[unit], float(t), side="right")) - 1
        now = spans[i] if 0 <= i < len(spans) and t < spans[i].end else None
        nxt = spans[i + 1] if i + 1 < len(spans) else None
        left = None
        if nxt is not None and self.beats.size and t >= 0.0:     # before the track's start: no beats to count
            # the beats still to come before it, its own start counted: 0 once it has started
            left = int(np.searchsorted(self.beats, nxt.start + EPS) - np.searchsorted(self.beats, t, side="right"))
            left = max(0, left)
        return PhraseNow(now, nxt, left, meter)


def bars_beats(beats: int, meter: int) -> str:
    """A count of beats as bars.beats (rekordbox's countdown: 13 beats in 4/4 is 3.1)."""
    meter = max(1, int(meter))
    return f"{beats // meter}.{beats % meter}"
