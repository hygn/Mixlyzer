"""A deck's track found in the library: by its path when the sender gives one (a file on this computer: the path the
library stores, or the same file under another spelling), else by its title and artist, its length telling versions
apart (a TV size, an extended mix) when the sender gives it."""

from __future__ import annotations

import unicodedata
from typing import Iterable, Optional

from core.deck_monitor.decks import TrackInfo
from core.library.records import TrackRow
from system.os_support import network_path   # noqa: F401  (a network path is never looked up)

DURATION_TOLERANCE_SEC = 2.0     # a length this close (or within DURATION_TOLERANCE_SHARE) is the same version
DURATION_TOLERANCE_SHARE = 0.01


def normalized(text: str) -> str:
    """A name as compared: NFKC, case folded, its spaces collapsed."""
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def same_length(a: float, b: float) -> bool:
    return abs(a - b) <= max(DURATION_TOLERANCE_SEC, DURATION_TOLERANCE_SHARE * max(a, b))


class TrackMatcher:
    """Matches against one library snapshot's tracks (made again for a new snapshot: ``for_snapshot``)."""

    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self._by_name: dict[tuple[str, str], list[TrackRow]] = {}
        self._by_title: dict[str, list[TrackRow]] = {}
        for row in getattr(snapshot, "tracks", ()) or ():
            title = normalized(row.title)
            if not title:
                continue
            self._by_name.setdefault((title, normalized(row.artist)), []).append(row)
            self._by_title.setdefault(title, []).append(row)
        self._found: dict[TrackInfo, Optional[TrackRow]] = {}

    def find(self, track: TrackInfo) -> Optional[TrackRow]:
        if track not in self._found:
            self._found[track] = self._path(track.path) or self._name(track)
        return self._found[track]

    def _path(self, path: str) -> Optional[TrackRow]:
        path = str(path or "").strip()
        if not path or "\x00" in path or network_path(path):
            return None
        row = self.snapshot.get(path)
        if row is not None:
            return row
        return self.snapshot.find_file(path) if PureWindowsPath(path).is_absolute() or path.startswith("/") else None

    def _name(self, track: TrackInfo) -> Optional[TrackRow]:
        title = normalized(track.title)
        if not title:
            return None
        artist = normalized(track.artist)
        rows = self._by_name.get((title, artist), []) if artist else self._by_title.get(title, [])
        return _closest(rows, float(track.duration_sec or 0.0))


def _closest(rows: Iterable[TrackRow], duration: float) -> Optional[TrackRow]:
    """The row of the closest length (one within the tolerance when the length is known), else the first."""
    rows = list(rows)
    if not rows:
        return None
    if duration <= 0.0:
        return rows[0]
    timed = [r for r in rows if r.duration and same_length(float(r.duration), duration)]
    return min(timed, key=lambda r: abs(float(r.duration) - duration)) if timed else None
