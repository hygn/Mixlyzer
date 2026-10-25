"""Tie-ups: the works a track is used in (an anime's opening, a game's title screen, a character song), kept in
library.db. The track owns its tie-ups, as it owns everything else: what the library shows is its rows.

Tables (library.db, made by LibraryDB.connect):
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
Ids are the library's own (UUIDv4); a work found under another name is the same row, never a copy.

Tagged automatically only where the user acts: a new track's first analysis takes the tie-ups of its versions in
the library (its Variants: `tag_new_track`), and a tie-up added in the Edit tab goes to the track's versions too
when Settings > Library says so (`save_tieup`'s versions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import sqlite3
from typing import Iterable, Optional
import unicodedata
import uuid

from core import i18n, reading
from core.i18n import tr, tr_mark

WRITE_WAIT_SEC = 60.0

# (code, name): stored as the code, shown translated
CATEGORIES = (
    ("anime", tr_mark("Anime")),
    ("game", tr_mark("Game")),
    ("drama", tr_mark("Drama")),
    ("movie", tr_mark("Movie")),
    ("media_mix", tr_mark("Media mix")),
    ("idol_2d", tr_mark("2D idol")),
    ("commercial", tr_mark("Commercial")),
    ("other", tr_mark("Other")),
)
# (code, name, short: the library's column and the comment; none: the work alone, "Title")
UNSPECIFIED = "unspecified"
ROLES = (
    (UNSPECIFIED, tr_mark("Not specified"), ""),
    ("opening", tr_mark("Opening"), tr_mark("OP")),
    ("ending", tr_mark("Ending"), tr_mark("ED")),
    ("insert", tr_mark("Insert song"), tr_mark("Insert")),
    ("theme", tr_mark("Theme song"), tr_mark("Theme")),
    ("character_song", tr_mark("Character song"), tr_mark("Chara")),
    ("unit_song", tr_mark("Unit song"), tr_mark("Unit")),
    ("image_song", tr_mark("Image song"), tr_mark("Image")),
    ("title_screen", tr_mark("Title / login screen"), tr_mark("Title screen")),
    ("main_menu", tr_mark("Main screen"), tr_mark("Menu")),
    ("event", tr_mark("Event song"), tr_mark("Event")),
    ("bgm", tr_mark("BGM"), tr_mark("BGM")),
    ("other", tr_mark("Other"), tr_mark("Other")),
)
# a work's names: its titles, and the other names it is found by (never shown)
OFFICIAL = "official"
ALIAS_TYPES = (
    ("synonym", tr_mark("Synonym")),            # a name it is commonly called by; a romanization of its title
    ("short", tr_mark("Short")),                # an abbreviation
    ("search_hint", tr_mark("Search hint")),    # a misspelling, a typing slip: only found by
)
# the languages the editor offers (any other ll-RR tag may be typed)
LANGUAGES = ("ja-JP", "ko-KR", "en-US", "zh-CN", "zh-TW", "zh-HK")
# the references that name a work itself (find_work): its Wikidata item first, its Wikipedia page
IDENTITY_URLS = ("https://www.wikidata.org/wiki/", "https://ja.wikipedia.org/wiki/")
_LANG = re.compile(r"[a-z]{2}-[A-Z]{2}\Z")
_ROLE_SHORT = {code: short for code, _name, short in ROLES}
_ROLE_NAME = {code: name for code, name, _short in ROLES}
_CATEGORY_NAME = dict(CATEGORIES)

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


def is_lang(code: str) -> bool:
    """Whether ``code`` is an ll-RR language tag (ko-KR)."""
    return bool(_LANG.match(str(code or "")))


def guess_lang(text: str) -> str:
    """The language a name is likely in, from its script (the user corrects it)."""
    if any("぀" <= c <= "ヿ" for c in text):
        return "ja-JP"
    if any("가" <= c <= "힣" for c in text):
        return "ko-KR"
    if any("一" <= c <= "鿿" for c in text):
        return "ja-JP"
    return "en-US"


@dataclass
class Work:
    id: str = ""
    category: str = "other"
    titles: list[tuple[str, str]] = field(default_factory=list)          # (lang, title): a language's first its primary
    aliases: list[tuple[str, str, str]] = field(default_factory=list)    # (type, lang, name): searched, never shown
    urls: list[tuple[str, str]] = field(default_factory=list)            # (label, url)

    def title(self, lang: Optional[str] = None) -> str:
        """The title shown in ``lang`` (default: the UI's): that language and region's first title, else the
        language's in any region, else the first title."""
        want = i18n.language() if lang is None else lang
        for match in (lambda code: code == want, lambda code: code.split("-")[0] == want.split("-")[0]):
            for code, text in self.titles:
                if match(code):
                    return text
        return self.titles[0][1] if self.titles else ""

    def names(self) -> list[str]:
        """Every title, then every other name: what the work is found by."""
        return [t for _lang, t in self.titles] + [a for _type, _lang, a in self.aliases]


@dataclass
class TieUp:
    work: Work = field(default_factory=Work)
    role: str = UNSPECIFIED     # the work is what matters; the role when known
    seq: int = 0                # OP2: 2 (0: none)
    detail: str = ""            # "season 2, episodes 1-12", "TV size"

    def key(self) -> tuple[str, str, int]:
        """What makes it one tie-up of a track: its work, role and number."""
        return self.work.id, self.role, int(self.seq)

    def short(self) -> str:
        """The work's title and the role, as the library shows it: "Title OP2"."""
        return f"{self.work.title()} {_short(self.role, self.seq)}".strip()


def _short(role: str, seq: int) -> str:
    short = _ROLE_SHORT.get(role, "Other")
    return tr(short) + (str(seq) if seq else "") if short else ""


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)


def connect(library_db: str) -> sqlite3.Connection:
    """A connection of a worker's own to library.db (the schema made, foreign keys on)."""
    con = sqlite3.connect(library_db, timeout=WRITE_WAIT_SEC)
    con.execute("PRAGMA foreign_keys=ON;")
    create_schema(con)
    return con


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


def works(con: sqlite3.Connection) -> list[Work]:
    """Every work with its names (no references)."""
    names = _names(con)
    return [_work(i, c, names.get(i, []), []) for i, c in con.execute("SELECT id, category FROM tieup_work;")]


def work_keys(con: sqlite3.Connection) -> list[tuple[str, tuple[str, str]]]:
    """(work id, keys) of every work's readings, for the search."""
    return [(w, reading.keys(r)) for w, text in con.execute("SELECT work_id, readings FROM tieup_name;")
            for r in text.split("\n") if r]


def name_key(name: str) -> str:
    """A name as two works' names are compared: NFKC, case folded, its letters and digits in their own script
    (さくひん・ざ・わーく！ and さくひん・ざ・わーく! alike; two Han spellings read alike never: no reading involved)."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", str(name or "")).casefold() if ch.isalnum())


def _identity(url: str) -> bool:
    """A reference naming the work itself: its Wikidata item, its Wikipedia page."""
    return url.startswith(IDENTITY_URLS)


def find_work(con: sqlite3.Connection, w: Work) -> Optional[str]:
    """The library's work that ``w`` is: the one whose references hold the same Wikidata item or Wikipedia page
    (Wikidata's first), else the one sharing an official title or a synonym with it (name_key; abbreviations and search
    hints never: short names collide); None where none is, or where two works are (left to the user to merge)."""
    urls = sorted((u for _label, u in w.urls if _identity(u)), key=lambda u: not u.startswith(IDENTITY_URLS[0]))
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


@dataclass(frozen=True)
class TrackTieups:
    shown: str                   # the library's column: "Title OP; Other ED"
    text: str                    # every name and role, for the plain text search
    readings: tuple[str, ...]    # the works' readings, for the search by sound


def track_texts(con: sqlite3.Connection) -> dict[str, TrackTieups]:
    """Every tagged track's tie-ups as the library shows and searches them."""
    names = _names(con)
    works_ = {i: _work(i, c, names.get(i, []), []) for i, c in con.execute("SELECT id, category FROM tieup_work;")}
    by_track: dict[str, list[tuple]] = {}
    for uid, work_id, role, seq, detail in con.execute(
            "SELECT track_uid, work_id, role, seq, detail FROM tieup ORDER BY track_uid, rank;"):
        by_track.setdefault(uid, []).append((works_.get(work_id) or Work(work_id), role, int(seq), detail))
    out = {}
    for uid, entries in by_track.items():
        shown, text, keys = [], [], []
        for w, role, seq, detail in entries:
            short = _short(role, seq)
            shown.append(f"{w.title()} {short}".strip())
            # the text search finds the role and the category by their names too, translated and in English
            role_name, category = _ROLE_NAME.get(role, "Other"), _CATEGORY_NAME.get(w.category, "Other")
            text += w.names() + [short, tr(role_name), role_name, tr(category), category, detail]
            keys += [r for *_n, readings in names.get(w.id, []) for r in readings.split("\n") if r and r not in keys]
        out[uid] = TrackTieups("; ".join(shown), " ".join(text), tuple(keys))
    return out


def comment_suffix(tieups: Iterable[TieUp]) -> str:
    shown = [t.short() for t in tieups]
    return f"[{'; '.join(shown)}]" if shown else ""


def with_suffix(comment: str, suffix: str) -> str:
    """The comment with the tie-ups' suffix after it (once: not again when it already ends with it)."""
    comment = str(comment or "").rstrip()
    if not suffix or comment.endswith(suffix):
        return comment
    return f"{comment} {suffix}" if comment else suffix


def comment_suffixes(con: sqlite3.Connection) -> dict[str, str]:
    """Every tagged track's comment suffix: "[Title OP; Other ED]"."""
    return {uid: f"[{t.shown}]" for uid, t in track_texts(con).items()}


# ---- writing (each in the caller's transaction) ----

def _readings(text: str, languages: Optional[list[str]], han_run: int) -> str:
    return "\n".join(r for r, _source in reading.readings(text, languages, han_run=han_run))


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
                    ((work_id, n, lang, kind, work_id, _readings(n, languages, han_run)) for n, lang, kind in names))
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


def delete_tieup(con: sqlite3.Connection, uid: str, key: tuple[str, str, int]) -> None:
    """A tie-up taken off the track (its work stays: other tracks may use it)."""
    con.execute("DELETE FROM tieup WHERE track_uid=? AND work_id=? AND role=? AND seq=?;", (uid, *key))


def work_tracks(con: sqlite3.Connection, work_id: str) -> list[str]:
    """The tracks with a tie-up to the work."""
    return [u for (u,) in con.execute("SELECT DISTINCT track_uid FROM tieup WHERE work_id=? ORDER BY track_uid;",
                                      (work_id,))]


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


def refresh_work_readings(con: sqlite3.Connection, languages: Optional[list[str]], han_run: int) -> None:
    """Every name read again (the stored languages changed)."""
    con.executemany("UPDATE tieup_name SET readings=? WHERE work_id=? AND name=?;",
                    [(_readings(n, languages, han_run), w, n)
                     for w, n in con.execute("SELECT work_id, name FROM tieup_name;").fetchall()])


# ---- versions ----

def versions_of(uid: str, pairs) -> set[str]:
    """The track's versions: the tracks its Variants relations (core.variants.store.Pair) join it to, directly or
    through others."""
    neighbours: dict[str, set[str]] = {}
    for p in pairs:
        neighbours.setdefault(p.a, set()).add(p.b)
        neighbours.setdefault(p.b, set()).add(p.a)
    group, todo = {uid}, [uid]
    while todo:
        for v in neighbours.get(todo.pop(), ()):
            if v not in group:
                group.add(v)
                todo.append(v)
    return group - {uid}


def match_new_track(con: sqlite3.Connection, uid: str, pairs) -> int:
    """A new track's tie-ups taken from its versions (theirs together, each once), where it has none. Returns the
    tie-ups added."""
    others = sorted(versions_of(uid, pairs))
    if not others or con.execute("SELECT 1 FROM tieup WHERE track_uid=?;", (uid,)).fetchone():
        return 0
    marks = ",".join("?" * len(others))
    rows = con.execute(f"SELECT work_id, role, seq, detail FROM tieup WHERE track_uid IN ({marks}) "
                       f"ORDER BY track_uid, rank;", others).fetchall()
    added = 0
    for work_id, role, seq, detail in rows:
        added += _put(con, uid, TieUp(Work(work_id), role, int(seq), detail))
    return added


def tag_new_track(library_db: str, variants_db: str, uid: str) -> int:
    """`match_new_track` on its own connection (the analysis's, after the track's fingerprint is indexed)."""
    from core.variants import store as variant_store

    con = connect(library_db)
    try:
        with con:
            return match_new_track(con, uid, variant_store.pairs(variants_db))
    finally:
        con.close()
