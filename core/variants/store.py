"""The variants' index, `{libpath}/variants.db`: kept apart from library.db because it is made from the analysis's
features (rebuilt whenever its method changes, never edited by the user). Written by the analysis (each track it
analyzes) and Index Now, several processes at once (SQLite serializes them; a writer waits up to WRITE_WAIT_SEC); read
by the library's Variants.

Tables: meta (key, value: the method it was made with); variant_song (id, uid, the fingerprint's codes); variant_hash
(hash, song, frame), clustered by hash; variant_pair (a, b: uids, a < b; the relation, b's to a, and the share of
each covered by the other's material).

A track is stored first, then queried against all stored (itself left out), each in its own transaction: of two
tracks indexed at once by two processes, the one stored second finds the other.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sqlite3
from typing import Mapping

import numpy as np

from core.variants import fingerprint as fpm

FILE_NAME = "variants.db"
WRITE_WAIT_SEC = 60.0
METHOD = fpm.METHOD
CANDIDATES = 100     # the tracks with the most votes verified
OFFSETS = 8          # a candidate's offsets with the most votes verified (an edit's sections have several)
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS variant_song (
    id INTEGER PRIMARY KEY, uid TEXT UNIQUE, frames INTEGER, code BLOB, rel BLOB, voc BLOB);
CREATE TABLE IF NOT EXISTS variant_hash (hash INTEGER, song INTEGER, pos INTEGER, PRIMARY KEY (hash, song, pos))
    WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS variant_pair (
    a TEXT, b TEXT, kind TEXT, cover_a REAL, cover_b REAL, full REAL, PRIMARY KEY (a, b)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS variant_pair_b ON variant_pair (b);
"""


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    kind: str          # fingerprint.relation's: b's to a
    cover_a: float
    cover_b: float
    full: float


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
            con.execute("DELETE FROM variant_song")
            con.execute("DELETE FROM variant_hash")
            con.execute("DELETE FROM variant_pair")
            con.execute("INSERT OR REPLACE INTO meta VALUES ('method', ?)", (METHOD,))
    return con


def open_reader(path: str) -> sqlite3.Connection | None:
    """A read-only connection, or None while there is no index (of this method) yet."""
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT value FROM meta WHERE key = 'method'").fetchone()
    except sqlite3.Error:
        row = None
    if row is None or row[0] != METHOD:
        con.close()
        return None
    return con


def indexed_uids(path: str) -> set[str]:
    """The tracks fingerprinted (by this method); none while there is no index."""
    con = open_reader(path)
    if con is None:
        return set()
    try:
        return {u for (u,) in con.execute("SELECT uid FROM variant_song")}
    finally:
        con.close()


def drop(con: sqlite3.Connection, uid: str) -> None:
    """A track out of the index, its relations with it, in the caller's transaction."""
    row = con.execute("SELECT id FROM variant_song WHERE uid = ?", (uid,)).fetchone()
    if row is not None:
        con.execute("DELETE FROM variant_hash WHERE song = ?", (row[0],))
        con.execute("DELETE FROM variant_song WHERE id = ?", (row[0],))
    con.execute("DELETE FROM variant_pair WHERE a = ? OR b = ?", (uid, uid))


def _stored(con: sqlite3.Connection, song: int) -> tuple[str, fpm.Fingerprint] | None:
    row = con.execute("SELECT uid, code, rel, voc FROM variant_song WHERE id = ?", (song,)).fetchone()
    if row is None:
        return None
    uid, code, rel, voc = row
    return uid, fpm.Fingerprint(np.frombuffer(code, np.uint16), np.frombuffer(rel, np.uint16),
                                np.frombuffer(voc, np.int8))


def _find(con: sqlite3.Connection, query: fpm.Fingerprint, me: int) -> dict[str, tuple[fpm.Verification, str]]:
    """The stored tracks (but ``me``) that are versions of the query: uid -> (verification, relation)."""
    hashes, frames = fpm.query_rows(query)
    if hashes.size == 0:
        return {}
    con.execute("CREATE TEMP TABLE IF NOT EXISTS q (hash INTEGER, pos INTEGER)")
    con.execute("DELETE FROM q")
    con.executemany("INSERT INTO q VALUES (?, ?)", zip(hashes.tolist(), frames.tolist()))
    rows = con.execute("""SELECT h.song, h.pos - q.pos AS off, count(*) AS n FROM q CROSS JOIN variant_hash h
                          ON h.hash = q.hash WHERE h.song != ? GROUP BY h.song, off HAVING n >= ?""",
                       (me, fpm.MIN_VOTES)).fetchall()
    con.execute("DELETE FROM q")
    con.commit()
    votes: dict[int, list[tuple[int, int]]] = {}
    for song, off, n in rows:
        votes.setdefault(int(song), []).append((int(n), int(off)))
    found = {}
    for song, offs in sorted(votes.items(), key=lambda kv: -max(kv[1])[0])[:CANDIDATES]:
        stored = _stored(con, song)
        if stored is None:
            continue
        offs.sort(reverse=True)
        v = fpm.verify(query, stored[1], [o for _, o in offs[:OFFSETS]])
        kind = fpm.relation(v)
        if kind is not None:
            found[stored[0]] = (v, kind)
    con.commit()
    return found


def put(con: sqlite3.Connection, uid: str, features: Mapping) -> int:
    """One track fingerprinted from its features (replacing what it had) and queried against the stored; its relations
    written. Commits as it goes (store, then query: see the module's note). Returns the variants found."""
    fp = fpm.fingerprint(features)
    hashes, frames = fpm.index_rows(fp)
    with con:
        drop(con, uid)
        cur = con.execute("INSERT INTO variant_song (uid, frames, code, rel, voc) VALUES (?, ?, ?, ?, ?)",
                          (uid, int(fp.code.size), fp.code.tobytes(), fp.rel.tobytes(), fp.voc.tobytes()))
        song = int(cur.lastrowid)
        con.executemany("INSERT OR IGNORE INTO variant_hash VALUES (?, ?, ?)",
                        zip(hashes.tolist(), [song] * int(hashes.size), frames.tolist()))
    found: dict[str, tuple[fpm.Verification, str]] = {}
    for k, sub in enumerate(fpm.subdivisions(features, query=True)):
        query = fp if k == 0 else fpm.fingerprint(features, sub)
        for other, (v, kind) in _find(con, query, song).items():
            if other not in found or v.covered > found[other][0].covered:
                found[other] = (v, kind)
    with con:
        for other, (v, kind) in found.items():
            con.execute("INSERT OR REPLACE INTO variant_pair VALUES (?, ?, ?, ?, ?, ?)", _as_pair(uid, other, v, kind))
    return len(found)


_SWAPPED = {fpm.A_INSTRUMENTAL: fpm.B_INSTRUMENTAL, fpm.B_INSTRUMENTAL: fpm.A_INSTRUMENTAL}


def _as_pair(uid: str, other: str, v: fpm.Verification, kind: str) -> tuple:
    """The stored row (a < b) of ``uid``'s verification against ``other``."""
    if uid < other:
        return uid, other, kind, v.cover_a, v.cover_b, v.full
    return other, uid, _SWAPPED.get(kind, kind), v.cover_b, v.cover_a, v.full


def pairs(path: str) -> list[Pair]:
    """Every relation found; none while there is no index."""
    con = open_reader(path)
    if con is None:
        return []
    try:
        return [Pair(*row) for row in con.execute("SELECT a, b, kind, cover_a, cover_b, full FROM variant_pair")]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def groups(found: list[Pair], uids: set[str]) -> list[list[str]]:
    """The tracks of ``uids`` joined by relations into groups (versions of one song), two tracks or more each."""
    parent: dict[str, str] = {}

    def root(u: str) -> str:
        while parent.setdefault(u, u) != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for p in found:
        if p.a in uids and p.b in uids:
            ra, rb = root(p.a), root(p.b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, list[str]] = {}
    for u in parent:
        members.setdefault(root(u), []).append(u)
    return [sorted(g) for g in members.values() if len(g) > 1]
