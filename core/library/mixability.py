"""The mix candidates' index, `{libpath}/mixability.db`: kept apart from library.db because it is made from the
audio and the analysis (never edited by the user), as the Variants' index. Its SQL, used by the library's owner alone
(core.library.jobs: an analysis stored, an editor's save, Index Now, a track removed, the candidates read); what the
rows mean is analyzer.search.mixability's.

One row a track: its beats' features (``BeatFeatures``, from the audio: the analysis's frame features, or Index Now's
own reading of them) with the beat times they were made on and their method (``FEATURES``), and the bars made of
them with the analysis's downbeats and overview (``Bars``) and theirs (``analyzer.search.mixability.METHOD``). The
bars are made again without the audio whenever the beats are those of the features (an edit of the downbeats, a
change of the bars' method); other beats (a beatgrid edited, re-analysed elsewhere) need the features read again
(Index Now)."""
from __future__ import annotations

import os
import sqlite3

import numpy as np

from analyzer.search import mixability as mx

FILE_NAME = "mixability.db"
WRITE_WAIT_SEC = 60.0
# the beats' features' method: what they are made of (the frame features' chroma and percussive bands, POSITIONS a
# beat); a change reads every track's audio again
FEATURES = f"bf1;positions={mx.POSITIONS}"
SCHEMA = """
CREATE TABLE IF NOT EXISTS mix_track (
    uid TEXT PRIMARY KEY, features TEXT NOT NULL, beat_times BLOB, chroma BLOB, onsets BLOB,
    bars TEXT, bar_start BLOB, bar_end BLOB, bar_beats BLOB, bar_rows BLOB);
"""


def path_for(libpath: str) -> str:
    return os.path.join(str(libpath), FILE_NAME)


def open_writer(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=WRITE_WAIT_SEC)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def open_reader(path: str) -> sqlite3.Connection | None:
    """A read-only connection, or None while there is no index yet."""
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        con.execute("SELECT uid FROM mix_track LIMIT 1").fetchall()
    except sqlite3.Error:
        con.close()
        return None
    return con


def stored_uids(path: str) -> set[str]:
    """Every track with a row (of any method)."""
    con = open_reader(path)
    if con is None:
        return set()
    try:
        return {u for (u,) in con.execute("SELECT uid FROM mix_track")}
    finally:
        con.close()


def feature_beats(path: str) -> dict[str, np.ndarray]:
    """uid -> the beat times its features of the current method were made on."""
    con = open_reader(path)
    if con is None:
        return {}
    try:
        return {u: np.frombuffer(b, np.float64) for u, b in con.execute(
            "SELECT uid, beat_times FROM mix_track WHERE features = ?", (FEATURES,))}
    finally:
        con.close()


def same_beats(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether features made on beats ``a`` fit beats ``b`` (stored as float64: as written, to a microsecond)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) < 1e-6))


def put(con: sqlite3.Connection, uid: str, beat_times, features: mx.BeatFeatures, bars: mx.Bars) -> None:
    """A track's row (replacing the one it had), in the caller's transaction."""
    con.execute("INSERT OR REPLACE INTO mix_track VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        uid, FEATURES, np.asarray(beat_times, dtype=np.float64).tobytes(),
        features.chroma.astype(np.float16).tobytes(), features.onsets.astype(np.float16).tobytes(),
        mx.METHOD, *_bar_blobs(bars)))


def put_bars(con: sqlite3.Connection, uid: str, bars: mx.Bars) -> None:
    """A track's bars made again (its features kept), in the caller's transaction."""
    con.execute("UPDATE mix_track SET bars = ?, bar_start = ?, bar_end = ?, bar_beats = ?, bar_rows = ? WHERE uid = ?",
                (mx.METHOD, *_bar_blobs(bars), uid))


def _bar_blobs(bars: mx.Bars) -> tuple:
    return (bars.start.astype(np.float32).tobytes(), bars.end.astype(np.float32).tobytes(),
            bars.beats.astype(np.int16).tobytes(), bars.rows.astype(np.float16).tobytes())


def features_of(con: sqlite3.Connection, uid: str) -> tuple[np.ndarray, mx.BeatFeatures] | None:
    """A track's features of the current method and the beat times they were made on, None without."""
    row = con.execute("SELECT beat_times, chroma, onsets FROM mix_track WHERE uid = ? AND features = ?",
                      (uid, FEATURES)).fetchone()
    if row is None:
        return None
    times = np.frombuffer(row[0], np.float64)
    n = max(times.size - 1, 0)
    chroma = np.frombuffer(row[1], np.float16).astype(np.float32).reshape(n, 12)
    onsets = np.frombuffer(row[2], np.float16).astype(np.float32).reshape(n, 3, mx.POSITIONS)
    return times, mx.BeatFeatures(chroma, onsets)


def stale_bars(path: str) -> list[str]:
    """The tracks whose features are current but whose bars are of another method (made again from them)."""
    con = open_reader(path)
    if con is None:
        return []
    try:
        return [u for (u,) in con.execute("SELECT uid FROM mix_track WHERE features = ? AND bars IS NOT ?",
                                          (FEATURES, mx.METHOD))]
    finally:
        con.close()


def all_bars(path: str) -> dict[str, mx.Bars]:
    """Every track's bars of the current method."""
    con = open_reader(path)
    if con is None:
        return {}
    try:
        out = {}
        for uid, start, end, beats, rows in con.execute(
                "SELECT uid, bar_start, bar_end, bar_beats, bar_rows FROM mix_track WHERE bars = ?", (mx.METHOD,)):
            s = np.frombuffer(start, np.float32)
            out[uid] = mx.Bars(s, np.frombuffer(end, np.float32), np.frombuffer(beats, np.int16),
                               np.frombuffer(rows, np.float16).astype(np.float32).reshape(s.size, mx.COLUMNS))
        return out
    finally:
        con.close()


def drop(con: sqlite3.Connection, uid: str) -> None:
    """A track out of the index, in the caller's transaction."""
    con.execute("DELETE FROM mix_track WHERE uid = ?", (uid,))
