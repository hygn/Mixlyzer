"""What a search gives, importing nothing of the search (its kernels are numba's): the app holds these while it waits for one."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MusicMatch:
    uid: str
    kind: str              # "recording" (landmarks) or "melody"
    score: float           # votes for a recording; the mean frame cost (semitones, lower better) for a melody
    start_sec: float       # where in the song the query matched
    end_sec: float
    transposition: float   # semitones the song is above the query (a melody's); 0 for a recording
    places: tuple[float, ...] = ()   # every place in the song it matched about as well (a song repeats), best first


@dataclass(frozen=True)
class SearchResult:
    kind: str                       # the best match's: "recording", "melody", or "none"
    matches: tuple[MusicMatch, ...]
    voiced_sec: float               # seconds of pitch in the query (the voice's)
