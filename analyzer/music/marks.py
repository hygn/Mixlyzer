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
