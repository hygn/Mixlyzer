"""The tie-ups' tables in library.db: every read and write of them (their SQL). Used by the library's owner alone
(core.library.jobs on its thread); what a tie-up and a work are, their roles and categories, is core.tieup's.

Tables (made by LibraryDB.connect):
  tieup_work  (id, category)                         a work: an anime, a game, a drama ...
  tieup_name  (work_id, name, lang, type, is_primary, rank, readings)
                                                     its names (after AniDB's title types and MusicBrainz's aliases):
                                                     titles (type official; a language's first its primary: the one
                                                     shown in that language) and names never shown (synonym, short,
                                                     search_hint); every name's language an ll-RR tag (ko-KR, ja-JP:
                                                     ISO 639-1 + ISO 3166-1 alpha-2); their readings (core.reading)
  tieup_url   (work_id, url, label, rank)            the work's web references
  tieup       (track_uid, work_id, role, seq, detail, rank)
                                                     the track used in the work: its role and number (OP2), one row
                                                     per (track, work, role, number)
Ids are the library's own (UUIDv4); a work found under another name is the same row, never a copy. Writes are in the
caller's transaction (the owner commits a job's).
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, Optional
import uuid

from core import reading
from core.i18n import tr
from core.tieup import (
    IDENTITY_URLS, OFFICIAL, UNSPECIFIED, Catalog, TieUp, TrackTieups, Work, describe, is_identity_url, is_lang,
    name_key, name_readings, versions_of,
)

_FK = "ON UPDATE CASCADE ON DELETE CASCADE"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tieup_work (
    id       TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT 'other'
);
CREATE TABLE IF NOT EXISTS tieup_name (
    work_id    TEXT NOT NULL REFERENCES tieup_work(id) {_FK},
    name       TEXT NOT NULL,
    lang       TEXT NOT NULL CHECK (lang GLOB '[a-z][a-z]-[A-Z][A-Z]'),
    type       TEXT NOT NULL DEFAULT '{OFFICIAL}',
    is_primary INTEGER NOT NULL DEFAULT 0,
    rank       INTEGER NOT NULL DEFAULT 0,
    readings   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (work_id, name)
);
CREATE TABLE IF NOT EXISTS tieup_url (
    work_id TEXT NOT NULL REFERENCES tieup_work(id) {_FK},
    url     TEXT NOT NULL,
    label   TEXT NOT NULL DEFAULT '',
    rank    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (work_id, url)
);
CREATE TABLE IF NOT EXISTS tieup (
    track_uid TEXT NOT NULL REFERENCES tracks(uid) ON DELETE CASCADE,
    work_id   TEXT NOT NULL REFERENCES tieup_work(id) {_FK},
    role      TEXT NOT NULL DEFAULT 'unspecified',
    seq       INTEGER NOT NULL DEFAULT 0,
    detail    TEXT NOT NULL DEFAULT '',
    rank      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (track_uid, work_id, role, seq)
);
CREATE INDEX IF NOT EXISTS idx_tieup_work ON tieup(work_id);
"""


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)


# ---- reading ----

def _names(con: sqlite3.Connection, where: str = "", args: tuple = ()) -> dict[str, list[tuple]]:
    """Works' names: id -> [(name, lang, type, readings)] in their order."""
    out: dict[str, list[tuple]] = {}
    for work_id, name, lang, kind, readings in con.execute(
            f"SELECT work_id, name, lang, type, readings FROM tieup_name {where} ORDER BY work_id, rank;", args):
        out.setdefault(work_id, []).append((name, lang, kind, readings))
    return out


def _work(work_id: str, category: str, names: list[tuple], urls: list[tuple[str, str]]) -> Work:
    return Work(work_id, category, [(lang, n) for n, lang, kind, _r in names if kind == OFFICIAL],
                [(kind, lang, n) for n, lang, kind, _r in names if kind != OFFICIAL], urls)


def work(con: sqlite3.Connection, work_id: str) -> Optional[Work]:
    row = con.execute("SELECT category FROM tieup_work WHERE id=?;", (work_id,)).fetchone()
    if row is None:
        return None
    urls = [(label, url) for label, url in con.execute(
        "SELECT label, url FROM tieup_url WHERE work_id=? ORDER BY rank;", (work_id,))]
    return _work(work_id, row[0], _names(con, "WHERE work_id=?", (work_id,)).get(work_id, []), urls)


def work_keys(con: sqlite3.Connection) -> list[tuple[str, tuple[str, str]]]:
    """(work id, keys) of every work's readings, for the search."""
    return [(w, reading.keys(r)) for w, text in con.execute("SELECT work_id, readings FROM tieup_name;")
            for r in text.split("\n") if r]


def catalog(con: sqlite3.Connection) -> Catalog:
    """Every work with its names and references, the search's keys, how many tracks use each."""
    names = _names(con)
    urls: dict[str, list[tuple[str, str]]] = {}
    for work_id, label, url in con.execute("SELECT work_id, label, url FROM tieup_url ORDER BY work_id, rank;"):
        urls.setdefault(work_id, []).append((label, url))
    works = {i: _work(i, c, names.get(i, []), urls.get(i, []))
             for i, c in con.execute("SELECT id, category FROM tieup_work;")}
    uses = {w: int(n) for w, n in con.execute("SELECT work_id, COUNT(DISTINCT track_uid) FROM tieup GROUP BY work_id;")}
    return Catalog(works, work_keys(con), uses)


def find_work(con: sqlite3.Connection, w: Work) -> Optional[str]:
    """The library's work that ``w`` is: the one whose references hold the same Wikidata item or Wikipedia page
    (Wikidata's first), else the one sharing an official title or a synonym with it (name_key; abbreviations and search
    hints never: short names collide); None where none is, or where two works are (left to the user to merge)."""
    urls = sorted((u for _label, u in w.urls if is_identity_url(u)), key=lambda u: not u.startswith(IDENTITY_URLS[0]))
    for url in urls:
        ids = {i for (i,) in con.execute("SELECT DISTINCT work_id FROM tieup_url WHERE url=?;", (url,))}
        if ids:
            return ids.pop() if len(ids) == 1 else None
    kinds = (OFFICIAL, "synonym")
    keys = ({name_key(t) for _lang, t in w.titles}
            | {name_key(a) for kind, _lang, a in w.aliases if kind in kinds}) - {""}
    ids = {i for i, name in con.execute("SELECT work_id, name FROM tieup_name WHERE type IN (?, ?);", kinds)
           if name_key(name) in keys}
    return ids.pop() if len(ids) == 1 else None


def track_tieups(con: sqlite3.Connection, uid: str) -> list[TieUp]:
    return [TieUp(work(con, work_id) or Work(work_id), role, int(seq), detail)
            for work_id, role, seq, detail in con.execute(
                "SELECT work_id, role, seq, detail FROM tieup WHERE track_uid=? ORDER BY rank;", (uid,)).fetchall()]


def track_texts(con: sqlite3.Connection) -> dict[str, TrackTieups]:
    """Every tagged track's tie-ups as the library shows and searches them (core.tieup.describe)."""
    names = _names(con)
    works = {i: _work(i, c, names.get(i, []), []) for i, c in con.execute("SELECT id, category FROM tieup_work;")}
    by_track: dict[str, list[tuple]] = {}
    for uid, work_id, role, seq, detail in con.execute(
            "SELECT track_uid, work_id, role, seq, detail FROM tieup ORDER BY track_uid, rank;"):
        readings = [r for *_n, text in names.get(work_id, []) for r in text.split("\n") if r]
        by_track.setdefault(uid, []).append((works.get(work_id) or Work(work_id), role, int(seq), detail, readings))
    return {uid: describe(entries) for uid, entries in by_track.items()}


def comment_suffixes(con: sqlite3.Connection) -> dict[str, str]:
    """Every tagged track's comment suffix: "[Title OP; Other ED]"."""
    return {uid: f"[{t.shown}]" for uid, t in track_texts(con).items()}


def work_tracks(con: sqlite3.Connection, work_id: str) -> list[str]:
    """The tracks with a tie-up to the work."""
    return [u for (u,) in con.execute("SELECT DISTINCT track_uid FROM tieup WHERE work_id=? ORDER BY track_uid;",
                                      (work_id,))]


def has_tieups(con: sqlite3.Connection, uid: str) -> bool:
    return con.execute("SELECT 1 FROM tieup WHERE track_uid=?;", (uid,)).fetchone() is not None


# ---- writing (each in the caller's transaction) ----

def add_names(con: sqlite3.Connection, work_id: str, titles: Iterable[tuple[str, str]] = (),
              aliases: Iterable[tuple[str, str, str]] = (), urls: Iterable[tuple[str, str]] = (),
              languages: Optional[list[str]] = None, han_run: int = reading.HAN_RUN_MIN) -> None:
    """Names and references added after the work's (one it has already kept as it is: never overwritten), each
    language's first title its primary. A language not ll-RR raises ValueError."""
    names = [(t.strip(), lang, OFFICIAL) for lang, t in titles if t.strip()]
    names += [(a.strip(), lang, kind) for kind, lang, a in aliases if a.strip()]
    bad = [lang for _n, lang, _k in names if not is_lang(lang)]
    if bad:
        raise ValueError(tr("A language is written as ko-KR (language-REGION): {code}").format(code=bad[0]))
    con.executemany("INSERT OR IGNORE INTO tieup_name(work_id, name, lang, type, rank, readings) VALUES(?, ?, ?, ?, "
                    "(SELECT COALESCE(MAX(rank) + 1, 0) FROM tieup_name WHERE work_id=?), ?);",
                    ((work_id, n, lang, kind, work_id, name_readings(n, languages, han_run)) for n, lang, kind in names))
    con.execute("""UPDATE tieup_name SET is_primary = (type = :o AND rank = (SELECT MIN(rank) FROM tieup_name n
                   WHERE n.work_id = tieup_name.work_id AND n.type = :o AND n.lang = tieup_name.lang))
                   WHERE work_id = :w;""", {"o": OFFICIAL, "w": work_id})
    con.executemany("INSERT OR IGNORE INTO tieup_url(work_id, url, label, rank) VALUES(?, ?, ?, "
                    "(SELECT COALESCE(MAX(rank) + 1, 0) FROM tieup_url WHERE work_id=?));",
                    ((work_id, url.strip(), label.strip(), work_id) for label, url in urls if url.strip()))


def save_work(con: sqlite3.Connection, w: Work, languages: Optional[list[str]] = None,
              han_run: int = reading.HAN_RUN_MIN) -> str:
    """A work written as edited (a new one when it has no id), its names read. Returns its id."""
    w.id = w.id or str(uuid.uuid4())
    con.execute("INSERT INTO tieup_work(id, category) VALUES(?, ?) ON CONFLICT(id) DO UPDATE SET category=excluded.category;",
                (w.id, w.category))
    con.execute("DELETE FROM tieup_name WHERE work_id=?;", (w.id,))
    con.execute("DELETE FROM tieup_url WHERE work_id=?;", (w.id,))
    add_names(con, w.id, w.titles, w.aliases, w.urls, languages, han_run)
    return w.id


def _put(con: sqlite3.Connection, uid: str, t: TieUp, rank: Optional[int] = None) -> bool:
    """The tie-up on the track (its detail updated when the track has it). Whether a row was added."""
    exists = con.execute("SELECT 1 FROM tieup WHERE track_uid=? AND work_id=? AND role=? AND seq=?;",
                         (uid, *t.key())).fetchone() is not None
    con.execute("""INSERT INTO tieup(track_uid, work_id, role, seq, detail, rank)
                   VALUES(?, ?, ?, ?, ?, COALESCE(?, (SELECT COALESCE(MAX(rank) + 1, 0) FROM tieup WHERE track_uid=?)))
                   ON CONFLICT(track_uid, work_id, role, seq) DO UPDATE SET detail=excluded.detail;""",
                (uid, *t.key(), t.detail.strip(), rank, uid))
    return not exists


def save_tieup(con: sqlite3.Connection, uid: str, t: TieUp, old: Optional[tuple[str, str, int]] = None,
               versions: Iterable[str] = (), languages: Optional[list[str]] = None,
               han_run: int = reading.HAN_RUN_MIN) -> list[str]:
    """A tie-up of the track written as edited with its work: ``old`` the key it had (an edit), else an addition,
    which ``versions`` (other tracks: the track's Variants) get too where they lack it. Returns the tracks
    written."""
    save_work(con, t.work, languages, han_run)
    rank = None
    if old is not None and tuple(old) != t.key():
        row = con.execute("SELECT rank FROM tieup WHERE track_uid=? AND work_id=? AND role=? AND seq=?;",
                          (uid, *old)).fetchone()
        rank = row[0] if row is not None else None
        delete_tieup(con, uid, old)
    _put(con, uid, t, rank)
    written = [uid]
    if old is None:
        for other in versions:
            if other != uid and con.execute("SELECT 1 FROM tracks WHERE uid=?;", (other,)).fetchone() \
                    and con.execute("SELECT 1 FROM tieup WHERE track_uid=? AND work_id=? AND role=? AND seq=?;",
                                    (other, *t.key())).fetchone() is None:
                _put(con, other, t)
                written.append(other)
    return written


def add_found(con: sqlite3.Connection, uid: str, found: TieUp, languages: Optional[list[str]] = None,
              han_run: int = reading.HAN_RUN_MIN) -> None:
    """A tie-up found (on the web) added to the track: its work the library's it is (find_work: the same Wikidata
    item or Wikipedia page, else one title or synonym in common; its other names and references added, nothing
    overwritten, its category kept), else a new one. On the track: none with no role where the track has the work
    already; a role found replaces the work's tie-up with none (its place kept)."""
    w = found.work
    work_id = find_work(con, w)
    if work_id is None:
        work_id = save_work(con, Work("", w.category, w.titles, w.aliases, w.urls), languages, han_run)
    else:
        add_names(con, work_id, w.titles, w.aliases, w.urls, languages, han_run)
    roles = {(r, int(s)) for r, s in con.execute("SELECT role, seq FROM tieup WHERE track_uid=? AND work_id=?;",
                                                 (uid, work_id))}
    if (found.role, found.seq) in roles or (found.role == UNSPECIFIED and roles):
        return
    if (UNSPECIFIED, 0) in roles:
        con.execute("UPDATE tieup SET role=?, seq=?, detail=? WHERE track_uid=? AND work_id=? AND role=? AND seq=0;",
                    (found.role, found.seq, found.detail, uid, work_id, UNSPECIFIED))
        return
    con.execute("INSERT INTO tieup(track_uid, work_id, role, seq, detail, rank) VALUES(?, ?, ?, ?, ?, "
                "(SELECT COALESCE(MAX(rank) + 1, 0) FROM tieup WHERE track_uid=?));",
                (uid, work_id, found.role, found.seq, found.detail, uid))


def delete_tieup(con: sqlite3.Connection, uid: str, key: tuple[str, str, int]) -> None:
    """A tie-up taken off the track (its work stays: other tracks may use it)."""
    con.execute("DELETE FROM tieup WHERE track_uid=? AND work_id=? AND role=? AND seq=?;", (uid, *key))


def delete_work(con: sqlite3.Connection, work_id: str) -> list[str]:
    """A work deleted with its names, references and every track's tie-up to it. Returns those tracks."""
    tracks = work_tracks(con, work_id)
    con.execute("DELETE FROM tieup WHERE work_id=?;", (work_id,))
    con.execute("DELETE FROM tieup_work WHERE id=?;", (work_id,))     # its names and references cascade
    return tracks


def merge_works(con: sqlite3.Connection, keep: str, drop: str, languages: Optional[list[str]] = None,
                han_run: int = reading.HAN_RUN_MIN) -> list[str]:
    """Two entries of one work made one: ``drop``'s names and references added after ``keep``'s, its tie-ups moved
    to ``keep`` (one a track has under both kept once, ``keep``'s), ``drop`` deleted. Returns the tracks moved."""
    other = work(con, drop)
    if other is None or keep == drop:
        return []
    tracks = work_tracks(con, drop)
    add_names(con, keep, other.titles, other.aliases, other.urls, languages, han_run)
    con.execute("UPDATE OR IGNORE tieup SET work_id=? WHERE work_id=?;", (keep, drop))
    delete_work(con, drop)
    return tracks


def work_names(con: sqlite3.Connection) -> list[str]:
    """Every work's names, each once (Re-read Library reads them)."""
    return [n for (n,) in con.execute("SELECT DISTINCT name FROM tieup_name;")]


def refresh_work_readings(con: sqlite3.Connection, languages: Optional[list[str]], han_run: int,
                          read: Optional[dict] = None) -> None:
    """Every name read again (the stored languages changed); ``read``: the names' readings read already (Re-read
    Library's process), by name."""
    def readings(name: str) -> str:
        if read is not None and name in read:
            return "\n".join(r for r, _source in read[name])
        return name_readings(name, languages, han_run)

    con.executemany("UPDATE tieup_name SET readings=? WHERE work_id=? AND name=?;",
                    [(readings(n), w, n) for w, n in con.execute("SELECT work_id, name FROM tieup_name;").fetchall()])


def match_new_track(con: sqlite3.Connection, uid: str, pairs) -> int:
    """A new track's tie-ups taken from its versions (theirs together, each once), where it has none. Returns the
    tie-ups added."""
    others = sorted(versions_of(uid, pairs))
    if not others or has_tieups(con, uid):
        return 0
    marks = ",".join("?" * len(others))
    rows = con.execute(f"SELECT work_id, role, seq, detail FROM tieup WHERE track_uid IN ({marks}) "
                       f"ORDER BY track_uid, rank;", others).fetchall()
    added = 0
    for work_id, role, seq, detail in rows:
        added += _put(con, uid, TieUp(Work(work_id), role, int(seq), detail))
    return added
