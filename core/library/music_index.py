"""The music search's index, `{libpath}/music_search.db`: kept apart from library.db because it is made from the
audio alone (rebuilt whenever its method changes, never edited by the user) and large (the landmarks, about 130 KB a
song). Its SQL, used by the library's owner alone (core.library.jobs: an analysis stored, Index Now, a track removed);
the search reads the landmarks through a `LandmarkIndex` the owner opens and lends it (WAL: reads go on while the
owner writes).

Tables: meta (key, value: the methods it was made with); melody (uid, the file's size and time it was made from,
the reference lines at each frame size: lines x frames u8); landmark_song (id, uid) and landmark (hash, song, t),
clustered by hash.
"""
from __future__ import annotations

import os
import sqlite3
from typing import TYPE_CHECKING

import numpy as np

from analyzer.search.music import landmarks, melody
from analyzer.search.music.matcher import MelodyIndex

if TYPE_CHECKING:
    from analyzer.search.music.index import IndexEntry

FILE_NAME = "music_search.db"
WRITE_WAIT_SEC = 60.0
METHOD = f"melody={melody.METHOD};landmarks={landmarks.METHOD}"
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS melody (
    uid TEXT PRIMARY KEY, file_size INTEGER, file_mtime REAL, lines INTEGER, m04 BLOB, m02 BLOB, m01 BLOB);
CREATE TABLE IF NOT EXISTS landmark_song (id INTEGER PRIMARY KEY, uid TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS landmark (hash INTEGER, song INTEGER, t INTEGER, PRIMARY KEY (hash, song, t)) WITHOUT ROWID;
"""
_COLUMNS = {0.4: "m04", 0.2: "m02", 0.1: "m01"}


def path_for(libpath: str) -> str:
    return os.path.join(str(libpath), FILE_NAME)


def open_writer(path: str) -> sqlite3.Connection:
    """A writer's connection: the schema made, everything dropped when made by another method."""
    con = sqlite3.connect(path, timeout=WRITE_WAIT_SEC)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    row = con.execute("SELECT value FROM meta WHERE key = 'method'").fetchone()
    if row is None or row[0] != METHOD:
        with con:
            con.execute("DELETE FROM melody")
            con.execute("DELETE FROM landmark")
            con.execute("DELETE FROM landmark_song")
            con.execute("INSERT OR REPLACE INTO meta VALUES ('method', ?)", (METHOD,))
    return con


def open_reader(path: str) -> sqlite3.Connection | None:
    """A read-only connection, or None while there is no index (of this method) yet."""
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    try:
        row = con.execute("SELECT value FROM meta WHERE key = 'method'").fetchone()
    except sqlite3.Error:
        row = None
    if row is None or row[0] != METHOD:
        con.close()
        return None
    return con


def indexed_uids(path: str) -> set[str]:
    """The songs indexed (by this method); none while there is no index."""
    con = open_reader(path)
    if con is None:
        return set()
    try:
        return {u for (u,) in con.execute("SELECT uid FROM melody")}
    finally:
        con.close()


def put(con: sqlite3.Connection, uid: str, size: int, mtime: float, entry: "IndexEntry") -> None:
    """One song's index (replacing what it had), in the caller's transaction."""
    lines, hashes, times = entry.lines, entry.hashes, entry.times
    k = int(lines[melody.FRAMES[0]].shape[0])
    con.execute("INSERT OR REPLACE INTO melody VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, int(size), float(mtime), k, *(np.ascontiguousarray(lines[f], np.uint8).tobytes()
                                                    for f in (0.4, 0.2, 0.1))))
    con.execute("INSERT OR IGNORE INTO landmark_song (uid) VALUES (?)", (uid,))
    sid = con.execute("SELECT id FROM landmark_song WHERE uid = ?", (uid,)).fetchone()[0]
    con.execute("DELETE FROM landmark WHERE song = ?", (sid,))
    con.executemany("INSERT OR IGNORE INTO landmark VALUES (?, ?, ?)",
                    zip(hashes.tolist(), [sid] * int(hashes.size), times.tolist()))


def drop(con: sqlite3.Connection, uid: str) -> None:
    con.execute("DELETE FROM melody WHERE uid = ?", (uid,))
    row = con.execute("SELECT id FROM landmark_song WHERE uid = ?", (uid,)).fetchone()
    if row is not None:
        con.execute("DELETE FROM landmark WHERE song = ?", (row[0],))
        con.execute("DELETE FROM landmark_song WHERE id = ?", (row[0],))


def melody_index(con: sqlite3.Connection) -> MelodyIndex:
    """Every song's reference lines, in memory (about 7 KB a song)."""
    idx = MelodyIndex()
    for uid, k, *blobs in con.execute(f"SELECT uid, lines, {', '.join(_COLUMNS.values())} FROM melody ORDER BY uid"):
        idx.add(uid, {f: np.frombuffer(b, np.uint8).reshape(int(k), -1) for f, b in zip(_COLUMNS, blobs)})
    return idx.freeze()


class LandmarkIndex:
    """The landmarks as the search reads them (analyzer.search.music.landmarks), on a read-only connection of their own:
    the owner opens it (core.library.jobs.open_music_search) and lends it to the search's thread."""

    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def rows(self, hashes: np.ndarray) -> np.ndarray:
        """The index's (hash, song, t) rows of ``hashes`` (each looked up once)."""
        con = self._con
        con.execute("CREATE TEMP TABLE IF NOT EXISTS qh (hash INTEGER PRIMARY KEY)")
        con.execute("DELETE FROM qh")
        con.executemany("INSERT INTO qh VALUES (?)", ((h,) for h in np.unique(hashes).tolist()))
        # CROSS JOIN keeps this order (SQLite would otherwise scan the whole index against the query's few hashes)
        return np.array(con.execute("SELECT l.hash, l.song, l.t FROM qh CROSS JOIN landmark l ON l.hash = qh.hash")
                        .fetchall(), np.int64).reshape(-1, 3)

    def songs(self) -> dict[int, str]:
        """The songs' ids in the landmark rows -> their uids."""
        return dict(self._con.execute("SELECT id, uid FROM landmark_song"))

    def close(self) -> None:
        self._con.close()


def open_search(path: str) -> tuple[LandmarkIndex, MelodyIndex] | None:
    """The search's view of the index: its landmarks (a read-only connection for the search's thread) and every
    song's reference lines; None while there is no index of this method."""
    con = open_reader(path)
    if con is None:
        return None
    return LandmarkIndex(con), melody_index(con)
