"""The variants' index, `{libpath}/variants.db`: kept apart from library.db because it is made from the analysis's
features (never edited by the user). Written by the analysis (each track it analyzes) and Index Now, several
processes at once (SQLite serializes them; a writer waits up to WRITE_WAIT_SEC); read by the library's Variants.

Each fingerprint carries its method (fingerprint.METHOD: the algorithm and the analysis's features it reads), so a
method's change leaves the index as it is: a track is fingerprinted by the current method when it is analyzed or
indexed (Index Now takes the tracks without one), compared only with fingerprints of the same method, and the
relations found by an earlier method stay until every track has the current one (`prune_old_methods`).

Tables: variant_method (id, name); variant_song (id, uid, method, the fingerprint's codes); variant_hash (method,
hash, song, frame), clustered by method and hash; variant_pair (a, b: uids, a < b; the method that found it; the
relation, b's to a, and the share of each covered by the other's material).

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
CREATE TABLE IF NOT EXISTS variant_method (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS variant_song (
    id INTEGER PRIMARY KEY, uid TEXT NOT NULL, method INTEGER NOT NULL, frames INTEGER, code BLOB, rel BLOB, voc BLOB,
    UNIQUE (uid, method));
CREATE TABLE IF NOT EXISTS variant_hash (method INTEGER, hash INTEGER, song INTEGER, pos INTEGER,
    PRIMARY KEY (method, hash, song, pos)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS variant_pair (
    a TEXT, b TEXT, method INTEGER, kind TEXT, cover_a REAL, cover_b REAL, full REAL, PRIMARY KEY (a, b, method))
    WITHOUT ROWID;
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
    """A writer's connection: the schema made (an index made before fingerprints carried their method is dropped
    whole and made again), the current method known."""
    con = sqlite3.connect(path, timeout=WRITE_WAIT_SEC)
    con.execute("PRAGMA journal_mode=WAL")
    columns = {row[1] for row in con.execute("PRAGMA table_info(variant_song)")}
    if columns and "method" not in columns:
        with con:
            for table in ("meta", "variant_song", "variant_hash", "variant_pair"):
                con.execute(f"DROP TABLE IF EXISTS {table}")
    con.executescript(SCHEMA)
    with con:
        con.execute("INSERT OR IGNORE INTO variant_method (name) VALUES (?)", (METHOD,))
    return con


def open_reader(path: str) -> sqlite3.Connection | None:
    """A read-only connection, or None while there is no index yet."""
    if not os.path.exists(path):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        con.execute("SELECT method FROM variant_song LIMIT 1").fetchall()
    except sqlite3.Error:
        con.close()
        return None
    return con


def _method(con: sqlite3.Connection) -> int | None:
    """The current method's id here, None when no fingerprint was made by it."""
    row = con.execute("SELECT id FROM variant_method WHERE name = ?", (METHOD,)).fetchone()
    return int(row[0]) if row is not None else None


def indexed_uids(path: str, any_method: bool = False) -> set[str]:
    """The tracks fingerprinted by the current method (``any_method``: by any); none while there is no index."""
    con = open_reader(path)
    if con is None:
        return set()
    try:
        if any_method:
            return {u for (u,) in con.execute("SELECT DISTINCT uid FROM variant_song")}
        return {u for (u,) in con.execute(
            "SELECT s.uid FROM variant_song s JOIN variant_method m ON m.id = s.method WHERE m.name = ?", (METHOD,))}
    finally:
        con.close()


def drop(con: sqlite3.Connection, uid: str) -> None:
    """A track out of the index (its fingerprints of every method), its relations with it, in the caller's
    transaction."""
    for song, method in con.execute("SELECT id, method FROM variant_song WHERE uid = ?", (uid,)).fetchall():
        con.execute("DELETE FROM variant_hash WHERE method = ? AND song = ?", (method, song))
        con.execute("DELETE FROM variant_song WHERE id = ?", (song,))
    con.execute("DELETE FROM variant_pair WHERE a = ? OR b = ?", (uid, uid))


def prune_old_methods(con: sqlite3.Connection) -> int:
    """The fingerprints of earlier methods of the tracks the current method has fingerprinted, and the relations
    only they found, removed; a method left with none forgotten. Returns the fingerprints removed."""
    current = _method(con)
    if current is None:
        return 0
    with con:
        old = con.execute("""SELECT s.id, s.method FROM variant_song s WHERE s.method != ? AND s.uid IN
                             (SELECT uid FROM variant_song WHERE method = ?)""", (current, current)).fetchall()
        for song, method in old:
            con.execute("DELETE FROM variant_hash WHERE method = ? AND song = ?", (method, song))
            con.execute("DELETE FROM variant_song WHERE id = ?", (song,))
        con.execute("""DELETE FROM variant_pair WHERE method != ? AND NOT EXISTS (SELECT 1 FROM variant_song s
                       WHERE s.method = variant_pair.method AND s.uid IN (variant_pair.a, variant_pair.b))""", (current,))
        con.execute("DELETE FROM variant_method WHERE id != ? AND id NOT IN (SELECT DISTINCT method FROM variant_song)",
                    (current,))
    return len(old)


def _stored(con: sqlite3.Connection, song: int) -> tuple[str, fpm.Fingerprint] | None:
    row = con.execute("SELECT uid, code, rel, voc FROM variant_song WHERE id = ?", (song,)).fetchone()
    if row is None:
        return None
    uid, code, rel, voc = row
    return uid, fpm.Fingerprint(np.frombuffer(code, np.uint16), np.frombuffer(rel, np.uint16),
                                np.frombuffer(voc, np.int8))


def _find(con: sqlite3.Connection, query: fpm.Fingerprint, method: int,
          me: int) -> dict[str, tuple[fpm.Verification, str]]:
    """The stored tracks (fingerprints of ``method``, but ``me``) that are versions of the query: uid ->
    (verification, relation)."""
    hashes, frames = fpm.query_rows(query)
    if hashes.size == 0:
        return {}
    con.execute("CREATE TEMP TABLE IF NOT EXISTS q (hash INTEGER, pos INTEGER)")
    con.execute("DELETE FROM q")
    con.executemany("INSERT INTO q VALUES (?, ?)", zip(hashes.tolist(), frames.tolist()))
    rows = con.execute("""SELECT h.song, h.pos - q.pos AS off, count(*) AS n FROM q CROSS JOIN variant_hash h
                          ON h.method = ? AND h.hash = q.hash WHERE h.song != ? GROUP BY h.song, off HAVING n >= ?""",
                       (method, me, fpm.MIN_VOTES)).fetchall()
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
    """One track fingerprinted from its features by the current method (replacing every fingerprint it had: its
    features are these now) and queried against the stored of that method; its relations written. Commits as it goes
    (store, then query: see the module's note). Returns the variants found."""
    fp = fpm.fingerprint(features)
    hashes, frames = fpm.index_rows(fp)
    with con:
        con.execute("INSERT OR IGNORE INTO variant_method (name) VALUES (?)", (METHOD,))
        method = _method(con)
        # its fingerprints all go (they read its former features), its relations of this method (found again
        # below); an earlier method's stay until pruned (the other track may have no fingerprint of this one yet)
        for song, old_method in con.execute("SELECT id, method FROM variant_song WHERE uid = ?", (uid,)).fetchall():
            con.execute("DELETE FROM variant_hash WHERE method = ? AND song = ?", (old_method, song))
            con.execute("DELETE FROM variant_song WHERE id = ?", (song,))
        con.execute("DELETE FROM variant_pair WHERE method = ? AND (a = ? OR b = ?)", (method, uid, uid))
        cur = con.execute("INSERT INTO variant_song (uid, method, frames, code, rel, voc) VALUES (?, ?, ?, ?, ?, ?)",
                          (uid, method, int(fp.code.size), fp.code.tobytes(), fp.rel.tobytes(), fp.voc.tobytes()))
        song = int(cur.lastrowid)
        con.executemany("INSERT OR IGNORE INTO variant_hash VALUES (?, ?, ?, ?)",
                        zip([method] * int(hashes.size), hashes.tolist(), [song] * int(hashes.size), frames.tolist()))
    found = _find_all(con, features, fp, method, song)
    with con:
        for other, (v, kind) in found.items():
            con.execute("INSERT OR REPLACE INTO variant_pair VALUES (?, ?, ?, ?, ?, ?, ?)",
                        _as_pair(uid, other, method, v, kind))
    return len(found)


def _find_all(con: sqlite3.Connection, features: Mapping, fp: fpm.Fingerprint, method: int,
              me: int) -> dict[str, tuple[fpm.Verification, str]]:
    """The stored tracks (fingerprints of ``method``, but ``me``) that are versions of the track with these
    features, each at its best (the octave's neighbour tried near its edge): uid -> (verification, relation)."""
    found: dict[str, tuple[fpm.Verification, str]] = {}
    for k, sub in enumerate(fpm.subdivisions(features, query=True)):
        query = fp if k == 0 else fpm.fingerprint(features, sub)
        for other, (v, kind) in _find(con, query, method, me).items():
            if other not in found or v.covered > found[other][0].covered:
                found[other] = (v, kind)
    return found


_SWAPPED = {fpm.A_INSTRUMENTAL: fpm.B_INSTRUMENTAL, fpm.B_INSTRUMENTAL: fpm.A_INSTRUMENTAL}


def _as_pair(uid: str, other: str, method: int, v: fpm.Verification, kind: str) -> tuple:
    """The stored row (a < b) of ``uid``'s verification against ``other``."""
    if uid < other:
        return uid, other, method, kind, v.cover_a, v.cover_b, v.full
    return other, uid, method, _SWAPPED.get(kind, kind), v.cover_b, v.cover_a, v.full


def pairs(path: str) -> list[Pair]:
    """Every relation found, by any method (a pair two methods found once: the current method's, else the
    latest's); none while there is no index."""
    con = open_reader(path)
    if con is None:
        return []
    try:
        current = _method(con) or -1
        rows = con.execute("SELECT a, b, kind, cover_a, cover_b, full FROM variant_pair "
                           "ORDER BY method = ? DESC, method DESC", (current,)).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    out: dict[tuple[str, str], Pair] = {}
    for row in rows:
        out.setdefault((row[0], row[1]), Pair(*row))
    return list(out.values())


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
