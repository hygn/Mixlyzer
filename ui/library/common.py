"""What the library's parts share: a track as the table shows it, the roles and timings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from PySide6 import QtCore, QtWidgets
from analyzer.music.key_names import ALPHANUMERIC_LABELS
from views.render.keystrip import key_segment_label
from core.search import reading
from core.library.records import TrackRow
from core.tieups import domain as tieup


# the columns found by their reading too (core.search.reading): the text search's columns -> Track fields (the tie-ups':
# their works' titles)
READ_FIELDS = ("title", "artist", "album", "tieup")
READ_COLUMNS = {"Title": "title", "Artist": "artist", "Album": "album", "Tie-up": "tieup"}
_SOURCE_KIND_ROLE = QtCore.Qt.UserRole
_SOURCE_VALUE_ROLE = QtCore.Qt.UserRole + 1
_DIRECTORY_LOADED_ROLE = QtCore.Qt.UserRole + 2
_SORT_ROLE = QtCore.Qt.UserRole + 10
_ANALYZED_ROLE = QtCore.Qt.UserRole + 11
_PLAYLIST_MIME = "application/x-mixlyzer-playlist"
_PLAYLIST_TRACKS_MIME = "application/x-mixlyzer-playlist-tracks"
# The filters hide the rows they leave out (rows are made only when the source or the library
# changes: about 60 ms a thousand rows); a typed query is applied once typing pauses.
SEARCH_DEBOUNCE_MS = 120
# Library updates in a row (an analysis finished, Edit Track saved, a re-read) are applied once.
LIBRARY_UPDATE_DELAY_MS = 100
# A library update changing up to this many tracks changes their rows; more rebuild the table.
INCREMENTAL_ROWS_MAX = 300


def _mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


# Data schema
@dataclass
class Track:
    track_id: str
    title: str
    artist: str
    album: str = ""
    bpm: Optional[float] = None
    key: Optional[int] = None
    duration_sec: Optional[float] = None
    rating: int = 0
    added: Optional[QtCore.QDateTime] = None
    path: str = ""
    comment: str = ""
    tieup: str = ""          # its song's tie-ups as shown (core.tieups.domain): "Title OP; Other ED"
    tieup_text: str = ""     # every title of their works and the roles, for the plain text search
    uid: str = ""
    playlist_index: Optional[int] = None
    analyzed: bool = True
    # search keys of the title / artist / album readings (core.search.reading): field -> one per reading
    reading_keys: dict = field(default_factory=dict)
    # the 초성 search's keys (the library's snapshot: LibrarySnapshot.initials): field -> keys
    initials: dict = field(default_factory=dict)
    # title / artist / album -> the key sorting it by its reading (LibrarySnapshot.sort_keys; none: as written)
    sort_keys: dict = field(default_factory=dict)
    _disp_playlist_index: str = field(init=False, default="")

    _disp_bpm: str = field(init=False, default="--")
    _disp_key: str = field(init=False, default="--")      # Alphanumeric / classical, as the app shows a key
    _disp_duration: str = field(init=False, default="00:00")
    _disp_rating: str = field(init=False, default="☆☆☆☆☆")
    _disp_added: str = field(init=False, default="")

    def __post_init__(self):
        # BPM
        self._disp_bpm = f"{self.bpm:.2f}" if isinstance(self.bpm, (int, float)) else "--"
        # Key
        try:
            if isinstance(self.key, int) and 0 <= self.key <= 23:
                self._disp_key = key_segment_label(self.key)
        except Exception:
            pass
        # Duration
        if self.duration_sec and self.duration_sec > 0:
            s = int(round(self.duration_sec))
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            self._disp_duration = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        # Rating
        r = max(0, min(5, int(self.rating or 0)))
        self._disp_rating = "★"*r + "☆"*(5-r)
        # Added
        if self.added:
            self._disp_added = self.added.toString("yyyy-MM-dd HH:mm")
        if self.playlist_index is not None:
            self._disp_playlist_index = str(self.playlist_index + 1)


def key_order(key: Optional[int]) -> int:
    """A key's place in the Alphanumeric order, 1A 1B 2A .. 12B (-1 for none): the Key column's sort."""
    if not isinstance(key, int) or not 0 <= key <= 23:
        return -1
    label = ALPHANUMERIC_LABELS[key]
    return int(label[:-1]) * 2 + (label[-1] == "B")


def row_to_track(row: TrackRow, readings: Optional[dict] = None,
                 tieups: Optional[tieup.TrackTieups] = None, initials: Optional[dict] = None,
                 sort_keys: Optional[dict] = None) -> Track:
    """``readings``: the track's readings by field (LibraryDB.all_readings), for the search; ``tieups``: its
    song's tie-ups (tieup.track_texts); ``initials``: its 초성 search's keys (LibrarySnapshot.initials);
    ``sort_keys``: its names' keys sorting them by their readings (LibrarySnapshot.sort_keys)."""
    added_qt = QtCore.QDateTime.fromSecsSinceEpoch(int(row.added_ts)) if row.added_ts else None
    keys = {f: [reading.keys(r) for r in (readings or {}).get(f, [])] for f in READ_FIELDS if f != "tieup"}
    keys["tieup"] = [reading.keys(r) for r in tieups.readings] if tieups is not None else []
    return Track(
        track_id=row.path, title=row.title or "", artist=row.artist or "",
        album=row.album or "", bpm=row.bpm, key=row.key, duration_sec=row.duration,
        rating=int(row.rating or 0), added=added_qt, path=row.path, comment=row.comment or "",
        tieup=tieups.shown if tieups is not None else "", tieup_text=tieups.text if tieups is not None else "",
        uid=row.uid or "", reading_keys=keys, initials=initials or {}, sort_keys=sort_keys or {},
    )


class _SortItem(QtWidgets.QTableWidgetItem):
    def __lt__(self, other):
        left = self.data(_SORT_ROLE)
        right = other.data(_SORT_ROLE)
        if left is not None and right is not None:
            return left < right
        return self.text().casefold() < other.text().casefold()
