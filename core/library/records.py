"""The library's records as the app passes them around: a track's row, a playlist's, the segment rows and the
readings' rule. Plain data, no storage (core.library.db reads and writes them)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from core.search import reading
from system.ids import canonical_uuid4


@dataclass
class TrackRow:
    uid: str                   # PRIMARY KEY (canonical UUIDv4): NPZ, thumbnail and playlist linkage
    path: str                  # the audio file (UNIQUE, editable: a moved file is given its new path)
    title: str = ""
    artist: str = ""
    album: str = ""
    bpm: Optional[float] = None
    key: Optional[int] = None          # integer Key
    duration: Optional[float] = None   # seconds
    total_samples: Optional[int] = None
    rating: int = 0
    added_ts: int = 0                  # epoch seconds
    comment: str = ""
    file_mtime: float = 0.0
    file_size: int = 0

    @staticmethod
    def _stable_uid_from_meta() -> Optional[str]:
        return str(uuid4())

    @staticmethod
    def from_meta(meta: Dict[str, Any]) -> "TrackRow":
        """Analysis meta (dict) → storable TrackRow. Accepts int/str key (uses int here)."""
        key = meta.get("key")
        uid = meta.get("uid")
        if uid is None:
            uid = TrackRow._stable_uid_from_meta()
        uid = canonical_uuid4(uid)
        added_ts = meta.get("added_ts") or int(time.time())
        return TrackRow(
            path=str(meta.get("path") or meta.get("track_id")),
            uid=uid,
            title=meta.get("title",""),
            artist=meta.get("artist",""),
            album=meta.get("album",""),
            bpm=float(meta["bpm"]) if meta.get("bpm") is not None else None,
            key=int(key) if key is not None else None,
            duration=( float(meta["duration_sec"])),
            total_samples=(int(meta["total_samples"]) if meta.get("total_samples") is not None else None),
            rating=int(meta.get("rating", 0)),
            added_ts=int(added_ts),
            comment=meta.get("comment",""),
            file_mtime=float(meta.get("file_mtime", 0.0)),
            file_size=int(meta.get("file_size", 0)),
        )

    def to_meta(self) -> Dict[str, Any]:
        """
        TrackRow → dict (compatible with analysis meta format).
        Use for writing dict or linking JSON/NPZ.
        """
        return {
            "path": self.path,
            "uid": self.uid,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "bpm": self.bpm,
            "key": self.key,
            "duration_sec": self.duration,
            "total_samples": self.total_samples,
            "rating": self.rating,
            "added_ts": self.added_ts,
            "comment": self.comment,
            "file_mtime": self.file_mtime,
            "file_size": self.file_size,
        }


# the fields with readings (track_readings)
READING_FIELDS = ("title", "artist", "album")
USER_SOURCE = "user"   # track_readings.source of the readings the user added


def field_readings(user: List[str], automatic: List[Tuple[str, str]],
                   removed: Iterable[str] = ()) -> List[Tuple[str, str]]:
    """A field's readings as (reading, source) in slot order: the user's first (all kept), then the
    automatic ones (``reading.readings``) not among them nor ``removed`` by the user,
    READING_SLOTS in all."""
    removed = set(removed)
    out = [(text, USER_SOURCE) for text in user]
    out += [(text, source) for text, source in automatic if text not in user and text not in removed]
    return out[:max(reading.READING_SLOTS, len(user))]


@dataclass(frozen=True)
class IndexTrack:
    """A library track Index Now indexes (core.library.jobs.index_plan)."""
    uid: str
    path: str            # the audio file
    size: int            # its size and time, kept with Search by Music's index
    mtime: float
    music: bool          # Search by Music's index to make (from the audio)
    variants: bool       # the variants' fingerprint to make (from the stored analysis)


PLAYLIST_NODE, FOLDER_NODE = "playlist", "folder"     # the playlists' tree's two kinds of node


@dataclass
class PlaylistRow:
    uid: str
    name: str
    sort_index: int                     # its place among its folder's children
    folder_uid: Optional[str] = None    # the folder it is in (None: the top)


@dataclass
class PlaylistFolderRow:
    uid: str
    name: str
    parent_uid: Optional[str] = None    # the folder it is in (None: the top)
    sort_index: int = 0                 # its place among its folder's children


@dataclass(frozen=True)
class PlaylistTree:
    """The playlists as a whole (core.library.jobs.playlist_tree): the folders, the playlists, each one's entries."""
    folders: tuple = ()          # PlaylistFolderRow
    playlists: tuple = ()        # PlaylistRow
    entries: dict = field(default_factory=dict)   # playlist uid -> ((position, track uid), ...)

    def children(self) -> dict:
        return playlist_children(self.folders, self.playlists)


def playlist_children(folders: Iterable[PlaylistFolderRow], playlists: Iterable[PlaylistRow]) -> dict:
    """The playlists' tree: a folder's uid (None: the top) -> its children in order, each (kind, row)."""
    out: dict = {}
    nodes = [(f.parent_uid, f.sort_index, 0, f.name.casefold(), FOLDER_NODE, f) for f in folders]
    nodes += [(p.folder_uid, p.sort_index, 1, p.name.casefold(), PLAYLIST_NODE, p) for p in playlists]
    for parent, *_order, kind, row in sorted(nodes, key=lambda n: (n[1], n[2], n[3])):
        out.setdefault(parent, []).append((kind, row))
    return out


@dataclass
class BpmSegmentRow:
    track_uid: str
    seq_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    bpm: Optional[float] = None
    bpm_rounded: Optional[int] = None
    time_signature: int = 4
    bpm_end: Optional[float] = None   # tempo at end_sec for a linear tempo ramp (None: steady)


@dataclass
class KeySegmentRow:
    track_uid: str
    seq_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    key_value: Optional[int] = None
    key_label: str = ""


@dataclass
class PhraseIndexRow:
    track_uid: str
    phrase_seq_index: int
    section_index: int
    phrase_label: str
    phrase_start_sec: float
    phrase_end_sec: float
    phrase_duration_sec: float
    start_sec: float
    end_sec: float
    duration_sec: float
    bpm_seq_index: Optional[int] = None
    bpm: Optional[float] = None
    bpm_rounded: Optional[int] = None
    time_signature: Optional[int] = None
    key_seq_index: Optional[int] = None
    key_value: Optional[int] = None
    key_label: str = ""


@dataclass
class LinearTransitionRow:
    track_uid: str
    path: str
    title: str
    artist: str
    from_seq_index: int
    to_seq_index: int
    from_start_sec: float
    from_end_sec: float
    from_duration_sec: float
    from_bpm: Optional[float]
    from_key_value: Optional[int]
    from_key_label: str
    to_start_sec: float
    to_end_sec: float
    to_duration_sec: float
    to_bpm: Optional[float]
    to_key_value: Optional[int]
    to_key_label: str
