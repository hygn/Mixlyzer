"""The marks a track carries, as typed values the analysis gives and the app keeps (its stored forms are the app's:
core.analysis_library)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CuePoint:
    """A CUE point (rekordbox's memory cue): where it is, its name and comment, its colour preset. ``id`` orders
    points at one time (None: its place in the list it came in)."""
    time_sec: float
    label: str = ""
    comment: str = ""
    color_idx: int = 0
    id: int | None = None


@dataclass(frozen=True)
class Phrase:
    """A phrase of the track: from ``start`` to ``end`` (s), its label (core.marks.phrases.PHRASE_LABELS,
    or UNASSIGNED where none is)."""
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class JumpCue:
    """A JumpCUE (rekordbox's hot cue): its slot letter (A-Z), its name (its comment; the letter itself: no name), the
    passage it marks and the point jumped to and from."""
    label: str
    comment: str
    start: float
    end: float
    point: float


@dataclass(frozen=True)
class JumpCueGroup:
    """JumpCUEs that jump between each other (one cue alone: a cue to jump to), shown in one colour preset
    (core.marks.colors); its cues by point."""
    cues: tuple[JumpCue, ...]
    color_idx: int


def cues_by_point(groups) -> list[tuple[JumpCue, int]]:
    """Every JumpCUE of ``groups`` (JumpCueGroup) with its group's index, by point."""
    return sorted(((cue, gi) for gi, group in enumerate(groups) for cue in group.cues), key=lambda item: item[0].point)
