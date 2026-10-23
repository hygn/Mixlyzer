"""Listening goes on: the searches of the last seconds heard, made every few seconds, joined over time.

A person sings a song on and on, and a phone plays it on: the right song comes back search after search, its matched
place moving on with the time between them (at the voice's tempo against the song's: half to double, the DTW's
range). A song a stretch matched by chance comes and goes, its place anywhere. So each song keeps a score: what each
search gave it (EVIDENCE: the better its match, the more), the older halved every HALF_LIFE_SEC (a new song sung
takes over within a few searches), twice as much when it was found in the search before at a place that moved on as
the time did (CONTINUITY). A song repeats itself (a chorus sung again matches as well as the first), so a match
carries every place it fits about as well, and the continuity is any of the places before to any of them now.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.music_search.search import MusicMatch, SearchResult

HALF_LIFE_SEC = 3.0          # what a search gave counts half this much later
CONTINUITY = 2.0             # found in the search before, its place moved on with the time
TEMPO_RANGE = (0.5, 2.0)     # the voice's tempo against the song's (the DTW's range)
PLACE_SLACK_SEC = 1.0        # the matched place's own uncertainty (frames, the path's ends)
MELODY_COST_ZERO = 0.45      # a melody match this costly gives nothing (unrelated songs: 0.3 - 0.5)
RECORDING_EVIDENCE = 2.0     # a recording's landmarks at one offset: certain
KEEP = 20
FRESH_SEC = 10.0             # a find this far back searched other sound (the window's length)
HISTORY_EXTRA_SEC = 6.0      # finds kept a little longer than that


@dataclass
class _Song:
    score: float
    match: MusicMatch
    found_at: float          # the time of the search that found it last
    history: list = field(default_factory=list)   # (time, places) of its recent finds


@dataclass
class Listening:
    fresh_sec: float = FRESH_SEC    # the listening window: a find this far back searched other sound
    songs: dict[str, _Song] = field(default_factory=dict)
    last: float | None = None       # the time of the last search

    def reset(self) -> None:
        self.songs.clear()
        self.last = None

    def update(self, result: SearchResult, now: float) -> list[MusicMatch]:
        """Join a search made at ``now`` (s) to the ones before; the songs best first, each its latest match."""
        dt = 0.0 if self.last is None else max(0.0, now - self.last)
        fade = 0.5 ** (dt / HALF_LIFE_SEC)
        for song in self.songs.values():
            song.score *= fade
        for m in result.matches:
            if m.kind == "recording":
                evidence = RECORDING_EVIDENCE
            else:
                evidence = max(0.0, 1.0 - m.score / MELODY_COST_ZERO)
            before = self.songs.get(m.uid)
            places_now = m.places or (m.start_sec,)
            if before is not None and CONTINUITY != 1.0:
                # a find of a window that shares no sound with this one (FRESH_SEC back or more: windows a search
                # apart share most of it, and a wrong song follows that shared sound as well as the right one),
                # any of its places then to any of them now (the song repeats itself)
                for then, places_then in before.history:
                    gap = now - then
                    if gap < self.fresh_sec:
                        continue
                    lo, hi = TEMPO_RANGE[0] * gap - PLACE_SLACK_SEC, TEMPO_RANGE[1] * gap + PLACE_SLACK_SEC
                    if any(lo <= a - b <= hi for b in places_then for a in places_now):
                        evidence *= CONTINUITY
                        break
            if before is None:
                before = self.songs[m.uid] = _Song(0.0, m, now)
            before.score += evidence
            before.match = m
            before.found_at = now
            before.history = [(t, p) for t, p in before.history if now - t <= self.fresh_sec + HISTORY_EXTRA_SEC]
            before.history.append((now, places_now))
        self.last = now
        for uid in [u for u, s in self.songs.items() if s.score < 0.01]:
            del self.songs[uid]
        ranked = sorted(self.songs.values(), key=lambda s: -s.score)[:KEEP]
        return [s.match for s in ranked]
