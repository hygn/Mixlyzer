"""library.db: the tracks, their readings and tag sources, the playlists and the segment tables (`LibraryDB`); the
tie-ups' tables are core.library.tieups'. Used by the library's owner alone (core.library.service), on its thread:
the records it reads and writes are core.library.records'."""
from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterable, List, Optional, Dict, Any, Tuple
from uuid import uuid4

from core.search import reading
from core.library import tieups
from core.library.records import (
    FOLDER_NODE, PLAYLIST_NODE, READING_FIELDS, USER_SOURCE, BpmSegmentRow, KeySegmentRow, LinearTransitionRow,
    PhraseIndexRow, PlaylistFolderRow, PlaylistRow, TrackRow, field_readings,
)
from system.ids import canonical_uuid4 as _canonical_uuid4, canonical_uuid4_or_none as _canonical_uuid4_or_none
from core.library.segments import harmonic_compatible_keys
from analyzer.music.key_names import idx_to_labels
from system.i18n import tr


# Columns that may appear in an ORDER BY clause. Used to validate the
# `order_by` argument, which is interpolated into SQL and must never carry
# untrusted text.
_ORDERABLE_COLUMNS = frozenset({
    "path", "uid", "title", "artist", "album", "bpm", "key", "duration",
    "total_samples", "rating", "added_ts", "comment", "file_mtime", "file_size",
})


def _safe_order_by(order_by: str, default: str = "added_ts DESC") -> str:
    """Validate an ORDER BY clause against a column whitelist.

    Accepts ``<column>`` or ``<column> ASC|DESC``. Anything else (unknown
    column, extra tokens, injection attempts) falls back to ``default``.
    """
    tokens = str(order_by or "").split()
    if not tokens or len(tokens) > 2:
        return default
    column = tokens[0]
    if column not in _ORDERABLE_COLUMNS:
        return default
    if len(tokens) == 2 and tokens[1].upper() not in ("ASC", "DESC"):
        return default
    return order_by.strip()


def _track_row_from_db_row(row: sqlite3.Row) -> TrackRow:
    return TrackRow(**dict(row))


# DB Helper
class LibraryDB:
    def __init__(self, db_path: str, reading_languages: Optional[List[str]] = None,
                 han_run: int = reading.HAN_RUN_MIN):
        """``reading_languages``: the languages Han is read in for the stored readings (the
        search settings' db_languages; all by default); ``han_run``: the Han runs beside kana read
        in Korean Hanja too (searchconfig.han_run(); 0: none)."""
        self.db_path = db_path
        self.reading_languages = list(reading_languages) if reading_languages is not None else None
        self.han_run = int(han_run)
        self._conn: Optional[sqlite3.Connection] = None
        self._tx_depth = 0
        self._index_queries: Optional[List[str]] = None     # track_index_rows' queries

    def connect(self):
        # used on the thread that opened it alone (the owner's): SQLite refuses it on another
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._index_queries = None
        self._apply_pragmas()
    
        # Create tracks table if missing
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            uid         TEXT PRIMARY KEY,
            path        TEXT NOT NULL UNIQUE,
            title       TEXT NOT NULL DEFAULT '',
            artist      TEXT DEFAULT '',
            album       TEXT DEFAULT '',
            bpm         REAL,
            key         INTEGER,
            duration    REAL,
            total_samples INTEGER,
            rating      INTEGER DEFAULT 0,
            added_ts    INTEGER NOT NULL,
            comment     TEXT DEFAULT '',
            file_mtime  REAL DEFAULT 0,
            file_size   INTEGER DEFAULT 0
        );
        """)
        # a playlist's name is one in its folder, not in the library (_check_playlist_name)
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS playlists (
            uid        TEXT PRIMARY KEY,
            name       TEXT NOT NULL COLLATE NOCASE,
            sort_index INTEGER NOT NULL DEFAULT 0
        );
        """)
        # one row per entry: a track may be in a playlist more than once
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS playlist_tracks (
            playlist_uid TEXT NOT NULL,
            position     INTEGER NOT NULL,
            track_uid    TEXT NOT NULL,
            PRIMARY KEY (playlist_uid, position),
            FOREIGN KEY (playlist_uid) REFERENCES playlists(uid) ON DELETE CASCADE,
            FOREIGN KEY (track_uid) REFERENCES tracks(uid) ON DELETE CASCADE
        );
        """)
        # the playlists' folders, a tree: a folder in a folder (parent_uid; NULL at the top), a playlist in a folder
        # (playlist_folder_items; none: at the top). A folder's children, its folders and playlists, are in one
        # order: their sort_index (a playlist's playlists.sort_index), numbered within the folder.
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS playlist_folders (
            uid        TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            parent_uid TEXT REFERENCES playlist_folders(uid) ON DELETE CASCADE,
            sort_index INTEGER NOT NULL DEFAULT 0
        );
        """)
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS playlist_folder_items (
            playlist_uid TEXT PRIMARY KEY REFERENCES playlists(uid) ON DELETE CASCADE,
            folder_uid   TEXT NOT NULL REFERENCES playlist_folders(uid) ON DELETE CASCADE
        );
        """)
        # The romanized readings (core.search.reading) of each track's title / artist / album: the
        # search's intermediate representation, up to READING_SLOTS a field. The ones the user
        # added (source "user") are kept; the automatic ones are made anew whenever the track is
        # written, except those the user removed (track_reading_removals).
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS track_readings (
            track_uid TEXT NOT NULL,
            field     TEXT NOT NULL,
            slot      INTEGER NOT NULL,
            reading   TEXT NOT NULL,
            source    TEXT NOT NULL,
            PRIMARY KEY (track_uid, field, slot),
            FOREIGN KEY (track_uid) REFERENCES tracks(uid) ON DELETE CASCADE
        );
        """)
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS track_reading_removals (
            track_uid TEXT NOT NULL,
            field     TEXT NOT NULL,
            reading   TEXT NOT NULL,
            PRIMARY KEY (track_uid, field, reading),
            FOREIGN KEY (track_uid) REFERENCES tracks(uid) ON DELETE CASCADE
        );
        """)
        # Where the track's file holds each of its title / artist / album / comment (the analysis's
        # reading: core.tags.writer's spellings, as "vorbis:PERFORMER" or "mp4:titl"; one row with
        # source "" when none does: a title from the file name, a field read empty): the file's tags are
        # written back there. BPM and key are not read from the tags.
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS track_tag_sources (
            track_uid TEXT NOT NULL,
            field     TEXT NOT NULL,
            slot      INTEGER NOT NULL,
            source    TEXT NOT NULL,
            PRIMARY KEY (track_uid, field, slot),
            FOREIGN KEY (track_uid) REFERENCES tracks(uid) ON DELETE CASCADE
        );
        """)
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS track_bpm_segments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            track_uid     TEXT NOT NULL,
            seq_index     INTEGER NOT NULL,
            start_sec     REAL NOT NULL,
            end_sec       REAL NOT NULL,
            duration_sec  REAL NOT NULL,
            bpm           REAL,
            bpm_rounded   INTEGER,
            time_signature INTEGER DEFAULT 4,
            bpm_end       REAL,
            UNIQUE(track_uid, seq_index)
        );
        """)
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS track_key_segments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            track_uid     TEXT NOT NULL,
            seq_index     INTEGER NOT NULL,
            start_sec     REAL NOT NULL,
            end_sec       REAL NOT NULL,
            duration_sec  REAL NOT NULL,
            key_value     INTEGER,
            key_label     TEXT DEFAULT '',
            UNIQUE(track_uid, seq_index)
        );
        """)
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS track_phrase_sections (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            track_uid           TEXT NOT NULL,
            phrase_seq_index    INTEGER NOT NULL,
            section_index       INTEGER NOT NULL,
            phrase_label        TEXT NOT NULL,
            phrase_start_sec    REAL NOT NULL,
            phrase_end_sec      REAL NOT NULL,
            phrase_duration_sec REAL NOT NULL,
            start_sec           REAL NOT NULL,
            end_sec             REAL NOT NULL,
            duration_sec        REAL NOT NULL,
            bpm_seq_index       INTEGER,
            bpm                 REAL,
            bpm_rounded         INTEGER,
            time_signature      INTEGER,
            key_seq_index       INTEGER,
            key_value           INTEGER,
            key_label           TEXT DEFAULT '',
            UNIQUE(track_uid, phrase_seq_index, section_index)
        );
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tbs_track_uid ON track_bpm_segments(track_uid);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tbs_bpm_duration ON track_bpm_segments(bpm_rounded, duration_sec);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tks_track_uid ON track_key_segments(track_uid);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tks_key_duration ON track_key_segments(key_value, duration_sec);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tps_track_phrase ON track_phrase_sections(track_uid, phrase_seq_index, section_index);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tps_label_bpm ON track_phrase_sections(phrase_label, bpm_rounded, phrase_duration_sec);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tps_label_bpm_value ON track_phrase_sections(phrase_label, bpm, phrase_duration_sec);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tps_label_key ON track_phrase_sections(phrase_label, key_value, phrase_duration_sec);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_playlists_sort ON playlists(sort_index);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_tracks_track ON playlist_tracks(track_uid);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_folders_parent ON playlist_folders(parent_uid);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_folder_items_folder "
                           "ON playlist_folder_items(folder_uid);")
        self._conn.commit()
        # the songs' tie-ups (core.tieups.domain): works, their names, tie-ups, references, a track's song
        tieups.create_schema(self._conn)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if not self._conn:
            raise RuntimeError("DB not connected. Call connect() first.")
        return self._conn

    def _apply_pragmas(self):
        c = self.conn.cursor()
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA foreign_keys=ON;")
        c.execute("PRAGMA temp_store=MEMORY;")
        c.execute("PRAGMA mmap_size=3000000000;")
        c.close()

    @contextmanager
    def tx(self):
        """One transaction: committed at its end, rolled back on an exception. Nested (a job's writes in one, each
        method's own inside it), an inner one is a savepoint: only the outermost commits, so the job's writes go in
        together or not at all."""
        depth = self._tx_depth
        self._tx_depth += 1
        savepoint = f"tx{depth}"
        try:
            if depth:
                self.conn.execute(f"SAVEPOINT {savepoint};")
            elif not self.conn.in_transaction:
                self.conn.execute("BEGIN;")      # a savepoint inside commits nothing (one outside a BEGIN would)
            try:
                yield
            except BaseException:
                if depth:
                    self._rollback_to(savepoint)
                else:
                    self.conn.rollback()
                raise
            if depth:
                self.conn.execute(f"RELEASE {savepoint};")
            else:
                self.conn.commit()
        finally:
            self._tx_depth -= 1

    def _rollback_to(self, savepoint: str) -> None:
        """A savepoint's writes undone on an exception, which goes on as it is: SQLite may have rolled the whole
        transaction back itself (a full disk, an I/O error), and the savepoint is gone with it."""
        if not self.conn.in_transaction:
            return
        try:
            self.conn.execute(f"ROLLBACK TO {savepoint};")
            self.conn.execute(f"RELEASE {savepoint};")
        except sqlite3.Error:
            pass

    # UPSERT / CRUD
    def upsert(self, row: TrackRow):
        """Insert the track, or update the one with its uid (its path too: a file another
        track has raises sqlite3.IntegrityError)."""
        q = """
        INSERT INTO tracks(uid, path, title, artist, album, bpm, key, duration, total_samples, rating, added_ts, comment, file_mtime, file_size)
        VALUES(:uid, :path, :title, :artist, :album, :bpm, :key, :duration, :total_samples, :rating, :added_ts, :comment, :file_mtime, :file_size)
        ON CONFLICT(uid) DO UPDATE SET
            path          = excluded.path,
            title         = COALESCE(excluded.title, tracks.title),
            artist        = COALESCE(excluded.artist, tracks.artist),
            album         = COALESCE(excluded.album, tracks.album),
            bpm           = COALESCE(excluded.bpm, tracks.bpm),
            key           = COALESCE(excluded.key, tracks.key),
            duration      = COALESCE(excluded.duration, tracks.duration),
            total_samples = COALESCE(excluded.total_samples, tracks.total_samples),
            rating        = COALESCE(excluded.rating, tracks.rating),
            added_ts      = CASE WHEN tracks.added_ts IS NULL OR tracks.added_ts=0 THEN excluded.added_ts ELSE tracks.added_ts END,
            comment       = COALESCE(excluded.comment, tracks.comment),
            file_mtime    = COALESCE(excluded.file_mtime, tracks.file_mtime),
            file_size     = COALESCE(excluded.file_size, tracks.file_size)
        ;
        """
        values = asdict(row)
        values["uid"] = _canonical_uuid4(row.uid)
        self.conn.execute(q, values)
        self.refresh_readings(values["uid"])

    # Readings (core.search.reading)
    def refresh_readings(self, uid: str, read: Optional[Dict[str, list]] = None) -> None:
        """Read the track's title / artist / album again: each field's automatic readings made anew
        in the stored languages (but those the user removed), after the readings the user added.
        ``read``: the texts' readings read already (Re-read Library's process), by text."""
        uid = _canonical_uuid4(uid)
        row = self.conn.execute("SELECT title, artist, album FROM tracks WHERE uid=?;", (uid,)).fetchone()
        if row is None:
            return
        removed = self.removed_readings(uid)
        for field in READING_FIELDS:
            user = [text for (text,) in self.conn.execute(
                "SELECT reading FROM track_readings WHERE track_uid=? AND field=? AND source=? ORDER BY slot;",
                (uid, field, USER_SOURCE))]
            text = row[field] or ""
            automatic = read[text] if read is not None and text in read else \
                reading.readings(text, self.reading_languages, han_run=self.han_run)
            self._write_readings(uid, field, field_readings(user, automatic, removed[field]))

    def _write_readings(self, uid: str, field: str, readings: list) -> None:
        self.conn.execute("DELETE FROM track_readings WHERE track_uid=? AND field=?;", (uid, field))
        self.conn.executemany(
            "INSERT INTO track_readings(track_uid, field, slot, reading, source) VALUES(?, ?, ?, ?, ?);",
            ((uid, field, slot, text, source) for slot, (text, source) in enumerate(readings[:reading.READING_SLOTS])),
        )

    def readings(self, uid: str) -> Dict[str, List[Tuple[str, str]]]:
        """The track's readings by field: (reading, source) in slot order."""
        out: Dict[str, List[Tuple[str, str]]] = {f: [] for f in READING_FIELDS}
        for field, text, source in self.conn.execute(
                "SELECT field, reading, source FROM track_readings WHERE track_uid=? ORDER BY field, slot;",
                (_canonical_uuid4(uid),)):
            out.setdefault(field, []).append((text, source))
        return out

    def all_readings(self) -> Dict[str, Dict[str, List[str]]]:
        """Every track's readings: uid -> field -> readings in slot order (for the library's search)."""
        out: Dict[str, Dict[str, List[str]]] = {}
        for uid, field, text in self.conn.execute(
                "SELECT track_uid, field, reading FROM track_readings ORDER BY track_uid, field, slot;"):
            out.setdefault(uid, {}).setdefault(field, []).append(text)
        return out

    def removed_readings(self, uid: str) -> Dict[str, List[str]]:
        """The automatic readings the user removed, by field."""
        out: Dict[str, List[str]] = {f: [] for f in READING_FIELDS}
        for field, text in self.conn.execute(
                "SELECT field, reading FROM track_reading_removals WHERE track_uid=? ORDER BY field, reading;",
                (_canonical_uuid4(uid),)):
            out.setdefault(field, []).append(text)
        return out

    def set_user_readings(self, uid: str, field: str, readings: Optional[List[str]],
                          removed: Iterable[str] = ()) -> None:
        """A field's readings as the user edited them: the ones they added (up to READING_SLOTS,
        kept from then on; None or empty: none) and the automatic ones they removed (left out
        from then on, whatever the languages); the other automatic ones made anew after them."""
        uid = _canonical_uuid4(uid)
        texts = list(dict.fromkeys(t.strip() for t in (readings or []) if t and t.strip()))
        with self.tx():
            self._write_readings(uid, field, [(t, USER_SOURCE) for t in texts])
            self.conn.execute("DELETE FROM track_reading_removals WHERE track_uid=? AND field=?;", (uid, field))
            self.conn.executemany(
                "INSERT OR IGNORE INTO track_reading_removals(track_uid, field, reading) VALUES(?, ?, ?);",
                ((uid, field, t.strip()) for t in removed if t and t.strip() and t.strip() not in texts))
            self.refresh_readings(uid)

    def upsert_meta(self, meta: Dict[str, Any]):
        row = TrackRow.from_meta(meta)
        self.upsert(row)
        return row.uid

    def set_tag_sources(self, uid: str, sources: Dict[str, List[str]]) -> None:
        """Where the track's file holds each field given (``tag_writer.tag_sources``: field -> spellings, []
        none); the fields not given as they are."""
        uid = _canonical_uuid4(uid)
        with self.tx():
            for field, places in sources.items():
                self.conn.execute("DELETE FROM track_tag_sources WHERE track_uid=? AND field=?;", (uid, field))
                rows = [(uid, field, i, str(s)) for i, s in enumerate(places)] or [(uid, field, 0, "")]
                self.conn.executemany(
                    "INSERT INTO track_tag_sources(track_uid, field, slot, source) VALUES(?, ?, ?, ?);", rows)

    def tag_sources(self, uid: str) -> Dict[str, List[str]]:
        """field -> where the track's file holds it, for the fields recorded ([]: in none of its tags)."""
        uid = _canonical_uuid4(uid)
        out: Dict[str, List[str]] = {}
        for field, source in self.conn.execute(
                "SELECT field, source FROM track_tag_sources WHERE track_uid=? ORDER BY field, slot;", (uid,)):
            out.setdefault(field, [])
            if source:
                out[field].append(source)
        return out

    def upsert_many(self, rows: Iterable[TrackRow]):
        with self.tx():
            for row in rows:
                self.upsert(row)

    def delete_tracks(self, uids: Iterable[str]):
        """Remove the tracks with their segment rows and playlist entries (the playlists close up)."""
        keys = [(uid,) for uid in dict.fromkeys(_canonical_uuid4(uid) for uid in uids)]
        with self.tx():
            playlists = set()
            for key in keys:
                playlists.update(row[0] for row in self.conn.execute(
                    "SELECT DISTINCT playlist_uid FROM playlist_tracks WHERE track_uid=?;", key))
            self.conn.executemany("DELETE FROM tracks WHERE uid=?;", keys)  # playlist entries and readings cascade
            self.conn.executemany("DELETE FROM track_bpm_segments WHERE track_uid=?;", keys)
            self.conn.executemany("DELETE FROM track_key_segments WHERE track_uid=?;", keys)
            self.conn.executemany("DELETE FROM track_phrase_sections WHERE track_uid=?;", keys)
            for playlist_uid in playlists:
                self._write_playlist_entries(
                    playlist_uid, [uid for _position, uid in self._playlist_entries(playlist_uid)])

    def drop_orphan_rows(self) -> int:
        """The rows of tracks no longer in the library, in every table keyed by a track (the tie-ups' too), dropped
        and the playlists holding one closed up: left by a removal from before every table followed its track.
        Returns the rows dropped."""
        tables = [name for (name,) in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
                  if name != "playlist_tracks"
                  and any(col[1] == "track_uid" for col in self.conn.execute(f'PRAGMA table_info("{name}");'))]
        gone = "track_uid NOT IN (SELECT uid FROM tracks)"
        dropped = 0
        with self.tx():
            for name in tables:
                dropped += self.conn.execute(f'DELETE FROM "{name}" WHERE {gone};').rowcount
            for (playlist_uid,) in self.conn.execute(
                    f"SELECT DISTINCT playlist_uid FROM playlist_tracks WHERE {gone};").fetchall():
                entries = [uid for _position, uid in self._playlist_entries(playlist_uid)]
                kept = [uid for uid in entries
                        if self.conn.execute("SELECT 1 FROM tracks WHERE uid=?;", (uid,)).fetchone()]
                dropped += len(entries) - len(kept)
                self._write_playlist_entries(playlist_uid, kept)
        return dropped

    def get(self, path: str) -> Optional[TrackRow]:
        """The row stored under exactly ``path`` (paths are kept as given, never
        normalized: case and separators mean different files on some file systems)."""
        cur = self.conn.execute("SELECT * FROM tracks WHERE path=?;", (path,))
        row = cur.fetchone()
        return _track_row_from_db_row(row) if row else None

    def find_file(self, path: str) -> Optional[TrackRow]:
        """The row of the file ``path`` points to, however it is spelled: stored
        under ``path`` itself, else the stored path that opens the same file
        (another case, separators, a link, a mapped drive). None if the file is
        not in the library or does not exist."""
        row = self.get(path)
        if row is not None:
            return row
        try:
            st = os.stat(path)
        except (OSError, ValueError):
            return None
        cur = self.conn.execute(
            "SELECT * FROM tracks WHERE file_size=? OR file_size IS NULL;", (st.st_size,)
        )
        for db_row in cur.fetchall():
            try:
                other = os.stat(db_row["path"])
            except (OSError, ValueError):
                continue
            if os.path.samestat(st, other):
                return _track_row_from_db_row(db_row)
        return None

    def get_by_uid(self, uid: str) -> Optional[TrackRow]:
        try:
            uid = _canonical_uuid4(uid)
        except ValueError:
            return None
        cur = self.conn.execute("SELECT * FROM tracks WHERE uid=?;", (uid,))
        row = cur.fetchone()
        return _track_row_from_db_row(row) if row else None

    def set_path(self, uid: str, path: str) -> None:
        """Give the track ``uid`` the file ``path`` (a moved or renamed file); the file's
        size and mtime follow. ValueError if another track already has that file."""
        uid = _canonical_uuid4(uid)
        path = str(path or "").strip()
        if not path:
            raise ValueError(tr("The path cannot be empty."))
        other = self.find_file(path)
        if other is not None and other.uid != uid:
            raise ValueError(tr('This file is already in the library as "{track}".').format(track=other.title or other.path))
        try:
            st = os.stat(path)
            stats = (st.st_mtime, st.st_size)
        except (OSError, ValueError):
            stats = (None, None)
        with self.tx():
            self.conn.execute(
                "UPDATE tracks SET path=?, file_mtime=COALESCE(?, file_mtime), "
                "file_size=COALESCE(?, file_size) WHERE uid=?;",
                (path, *stats, uid),
            )
            # another file: where it holds the fields is found again when its tags are written
            self.conn.execute("DELETE FROM track_tag_sources WHERE track_uid=?;", (uid,))

    # Playlists. Both playlists and their entries carry explicit indexes so the
    # user's order is stable and independent of the library table's sorting. An
    # entry is addressed by its position: a track may be in a playlist more than once.
    def create_playlist(self, name: str, folder_uid: Optional[str] = None) -> PlaylistRow:
        """A playlist at the end of ``folder_uid`` (None: the top)."""
        name = str(name or "").strip()
        if not name:
            raise ValueError(tr("Playlist name cannot be empty"))
        folder = self._folder_or_top(folder_uid)
        self._check_playlist_name(name, folder)
        playlist_uid = str(uuid4())
        with self.tx():
            next_index = len(self._children(folder))
            self.conn.execute(
                "INSERT INTO playlists(uid, name, sort_index) VALUES(?, ?, ?);",
                (playlist_uid, name, next_index),
            )
            if folder is not None:
                self.conn.execute("INSERT INTO playlist_folder_items(playlist_uid, folder_uid) VALUES(?, ?);",
                                  (playlist_uid, folder))
        return PlaylistRow(playlist_uid, name, next_index, folder)

    def list_playlists(self) -> List[PlaylistRow]:
        """Every playlist with its folder (its sort_index its place there)."""
        cur = self.conn.execute(
            "SELECT p.uid, p.name, p.sort_index, f.folder_uid FROM playlists p "
            "LEFT JOIN playlist_folder_items f ON f.playlist_uid = p.uid "
            "ORDER BY p.sort_index, p.name COLLATE NOCASE;"
        )
        return [PlaylistRow(**dict(row)) for row in cur.fetchall()]

    # ---- the playlists' folders ----
    def list_playlist_folders(self) -> List[PlaylistFolderRow]:
        cur = self.conn.execute(
            "SELECT uid, name, parent_uid, sort_index FROM playlist_folders ORDER BY sort_index, name COLLATE NOCASE;"
        )
        return [PlaylistFolderRow(**dict(row)) for row in cur.fetchall()]

    def create_playlist_folder(self, name: str, parent_uid: Optional[str] = None) -> PlaylistFolderRow:
        """A folder at the end of ``parent_uid`` (None: the top)."""
        name = str(name or "").strip()
        if not name:
            raise ValueError(tr("Folder name cannot be empty"))
        parent = self._folder_or_top(parent_uid)
        folder_uid = str(uuid4())
        with self.tx():
            next_index = len(self._children(parent))
            self.conn.execute("INSERT INTO playlist_folders(uid, name, parent_uid, sort_index) VALUES(?, ?, ?, ?);",
                              (folder_uid, name, parent, next_index))
        return PlaylistFolderRow(folder_uid, name, parent, next_index)

    def rename_playlist_folder(self, folder_uid: str, name: str) -> None:
        name = str(name or "").strip()
        if not name:
            raise ValueError(tr("Folder name cannot be empty"))
        with self.tx():
            self.conn.execute("UPDATE playlist_folders SET name=? WHERE uid=?;", (name, _canonical_uuid4(folder_uid)))

    def delete_playlist_folder(self, folder_uid: str) -> None:
        """The folder deleted with everything in it: its folders and their playlists (the tracks stay in the
        library); its siblings close up."""
        folder_uid = _canonical_uuid4(folder_uid)
        parent = self._parent_of(FOLDER_NODE, folder_uid)
        inside = self._subtree(folder_uid)
        with self.tx():
            marks = ",".join("?" * len(inside))
            self.conn.execute(f"DELETE FROM playlists WHERE uid IN (SELECT playlist_uid FROM playlist_folder_items "
                              f"WHERE folder_uid IN ({marks}));", tuple(inside))
            self.conn.execute("DELETE FROM playlist_folders WHERE uid=?;", (folder_uid,))
            self._write_children(self._children(parent))

    def move_playlist_node(self, kind: str, uid: str, folder_uid: Optional[str],
                           before: Optional[tuple[str, str]] = None) -> bool:
        """A playlist or folder (``kind``: PLAYLIST_NODE / FOLDER_NODE) moved into ``folder_uid`` (None: the top),
        before its child ``before`` ((kind, uid); None or not there: at its end). A folder never goes into itself or
        a folder inside it. False when nothing moved."""
        if kind not in (PLAYLIST_NODE, FOLDER_NODE):
            raise ValueError(f"not a playlist node: {kind}")
        uid = _canonical_uuid4(uid)
        folder = self._folder_or_top(folder_uid)
        if kind == FOLDER_NODE and folder is not None and folder in self._subtree(uid):
            return False
        old_parent = self._parent_of(kind, uid)
        old_children = self._children(old_parent)
        if (kind, uid) not in old_children:
            return False
        children = [child for child in self._children(folder) if child != (kind, uid)]
        before = (before[0], _canonical_uuid4(before[1])) if before else None
        children.insert(children.index(before) if before in children else len(children), (kind, uid))
        if old_parent == folder and children == old_children:
            return False
        if kind == PLAYLIST_NODE and old_parent != folder:
            name = self.conn.execute("SELECT name FROM playlists WHERE uid=?;", (uid,)).fetchone()[0]
            self._check_playlist_name(name, folder)
        with self.tx():
            if kind == FOLDER_NODE:
                self.conn.execute("UPDATE playlist_folders SET parent_uid=? WHERE uid=?;", (folder, uid))
            else:
                self.conn.execute("DELETE FROM playlist_folder_items WHERE playlist_uid=?;", (uid,))
                if folder is not None:
                    self.conn.execute("INSERT INTO playlist_folder_items(playlist_uid, folder_uid) VALUES(?, ?);",
                                      (uid, folder))
            self._write_children(children)
            if old_parent != folder:
                self._write_children(self._children(old_parent))
        return True

    def move_playlist_node_by(self, kind: str, uid: str, offset: int) -> bool:
        """A playlist or folder moved ``offset`` places among its folder's children."""
        uid = _canonical_uuid4(uid)
        parent = self._parent_of(kind, uid)
        children = self._children(parent)
        if (kind, uid) not in children:
            return False
        old = children.index((kind, uid))
        new = max(0, min(len(children) - 1, old + int(offset)))
        if new == old:
            return False
        children.insert(new, children.pop(old))
        with self.tx():
            self._write_children(children)
        return True

    def _check_playlist_name(self, name: str, folder: Optional[str], playlist_uid: Optional[str] = None) -> None:
        """A playlist's name is one in its folder (as rekordbox's: the same name in another folder is another
        playlist), letter case aside; ValueError when another playlist there has it."""
        rows = self.conn.execute(
            "SELECT p.uid FROM playlists p LEFT JOIN playlist_folder_items f ON f.playlist_uid = p.uid "
            "WHERE p.name = ? COLLATE NOCASE AND f.folder_uid IS ?;", (name, folder)).fetchall()
        if any(row[0] != playlist_uid for row in rows):
            raise ValueError(tr('A playlist named "{name}" is already in this folder.').format(name=name))

    def _folder_or_top(self, folder_uid: Optional[str]) -> Optional[str]:
        """A folder's uid in its one spelling (it must exist), or None for the top."""
        if folder_uid is None or str(folder_uid) == "":
            return None
        folder = _canonical_uuid4(folder_uid)
        if self.conn.execute("SELECT 1 FROM playlist_folders WHERE uid=?;", (folder,)).fetchone() is None:
            raise ValueError(f"no playlist folder {folder}")
        return folder

    def _parent_of(self, kind: str, uid: str) -> Optional[str]:
        if kind == FOLDER_NODE:
            row = self.conn.execute("SELECT parent_uid FROM playlist_folders WHERE uid=?;", (uid,)).fetchone()
        else:
            row = self.conn.execute("SELECT folder_uid FROM playlist_folder_items WHERE playlist_uid=?;",
                                    (uid,)).fetchone()
        return row[0] if row is not None else None

    def _subtree(self, folder_uid: str) -> set[str]:
        """The folder and every folder inside it."""
        rows = self.conn.execute(
            "WITH RECURSIVE inside(uid) AS (SELECT ? UNION "
            "SELECT f.uid FROM playlist_folders f JOIN inside ON f.parent_uid = inside.uid) SELECT uid FROM inside;",
            (folder_uid,)).fetchall()
        return {row[0] for row in rows}

    def _children(self, folder_uid: Optional[str]) -> List[Tuple[str, str]]:
        """A folder's children in their order, (kind, uid): its folders and playlists (None: the top's)."""
        if folder_uid is None:
            folders = self.conn.execute(
                "SELECT uid, sort_index, name FROM playlist_folders WHERE parent_uid IS NULL;").fetchall()
            playlists = self.conn.execute(
                "SELECT p.uid, p.sort_index, p.name FROM playlists p LEFT JOIN playlist_folder_items f "
                "ON f.playlist_uid = p.uid WHERE f.folder_uid IS NULL;").fetchall()
        else:
            folders = self.conn.execute(
                "SELECT uid, sort_index, name FROM playlist_folders WHERE parent_uid=?;", (folder_uid,)).fetchall()
            playlists = self.conn.execute(
                "SELECT p.uid, p.sort_index, p.name FROM playlists p JOIN playlist_folder_items f "
                "ON f.playlist_uid = p.uid WHERE f.folder_uid=?;", (folder_uid,)).fetchall()
        nodes = [(int(r[1]), 0, str(r[2]).casefold(), FOLDER_NODE, str(r[0])) for r in folders]
        nodes += [(int(r[1]), 1, str(r[2]).casefold(), PLAYLIST_NODE, str(r[0])) for r in playlists]
        return [(kind, uid) for *_order, kind, uid in sorted(nodes)]

    def _write_children(self, children: List[Tuple[str, str]]) -> None:
        for sort_index, (kind, uid) in enumerate(children):
            table = "playlist_folders" if kind == FOLDER_NODE else "playlists"
            self.conn.execute(f"UPDATE {table} SET sort_index=? WHERE uid=?;", (sort_index, uid))

    def rename_playlist(self, playlist_uid: str, name: str) -> None:
        name = str(name or "").strip()
        if not name:
            raise ValueError(tr("Playlist name cannot be empty"))
        playlist_uid = _canonical_uuid4(playlist_uid)
        self._check_playlist_name(name, self._parent_of(PLAYLIST_NODE, playlist_uid), playlist_uid)
        with self.tx():
            self.conn.execute("UPDATE playlists SET name=? WHERE uid=?;", (name, playlist_uid))

    def delete_playlist(self, playlist_uid: str) -> None:
        """The playlist deleted (the tracks stay in the library); its folder's other children close up."""
        playlist_uid = _canonical_uuid4(playlist_uid)
        parent = self._parent_of(PLAYLIST_NODE, playlist_uid)
        with self.tx():
            self.conn.execute("DELETE FROM playlists WHERE uid=?;", (playlist_uid,))
            self._write_children(self._children(parent))

    def _playlist_entries(self, playlist_uid: str) -> List[Tuple[int, str]]:
        """(position, track uid) of each entry, in order."""
        rows = self.conn.execute(
            "SELECT position, track_uid FROM playlist_tracks WHERE playlist_uid=? ORDER BY position;",
            (playlist_uid,),
        ).fetchall()
        return [(int(row[0]), str(row[1])) for row in rows]

    def _write_playlist_entries(self, playlist_uid: str, track_uids: List[str]) -> None:
        """Replace the playlist's entries by ``track_uids`` at positions 0, 1, ..."""
        self.conn.execute("DELETE FROM playlist_tracks WHERE playlist_uid=?;", (playlist_uid,))
        self.conn.executemany(
            "INSERT INTO playlist_tracks(playlist_uid, position, track_uid) VALUES(?, ?, ?);",
            ((playlist_uid, position, uid) for position, uid in enumerate(track_uids)),
        )

    def add_playlist_tracks(self, playlist_uid: str, track_uids: Iterable[str]) -> int:
        """Append the tracks in order; a track already in the playlist is added again."""
        playlist_uid = _canonical_uuid4(playlist_uid)
        uids = [_canonical_uuid4(uid) for uid in track_uids]
        with self.tx():
            next_position = int(self.conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM playlist_tracks WHERE playlist_uid=?;",
                (playlist_uid,),
            ).fetchone()[0])
            self.conn.executemany(
                "INSERT INTO playlist_tracks(playlist_uid, position, track_uid) VALUES(?, ?, ?);",
                ((playlist_uid, next_position + i, uid) for i, uid in enumerate(uids)),
            )
        return len(uids)

    def playlist_entries(self) -> Dict[str, Tuple[Tuple[int, str], ...]]:
        """Every playlist's entries, (position, track uid) in order, by playlist (the snapshot's: no rows built)."""
        out: Dict[str, List[Tuple[int, str]]] = {}
        for playlist_uid, position, track_uid in self.conn.execute(
                "SELECT playlist_uid, position, track_uid FROM playlist_tracks ORDER BY playlist_uid, position;"):
            out.setdefault(playlist_uid, []).append((position, track_uid))
        return {uid: tuple(entries) for uid, entries in out.items()}

    def list_playlist_tracks(self, playlist_uid: str) -> List[Tuple[int, TrackRow]]:
        """(position, track) of each entry, in order."""
        cur = self.conn.execute(
            """
            SELECT pt.position AS playlist_position, t.*
            FROM playlist_tracks pt
            JOIN tracks t ON t.uid = pt.track_uid
            WHERE pt.playlist_uid=?
            ORDER BY pt.position;
            """,
            (_canonical_uuid4(playlist_uid),),
        )
        result: List[Tuple[int, TrackRow]] = []
        for row in cur.fetchall():
            data = dict(row)
            position = int(data.pop("playlist_position"))
            result.append((position, TrackRow(**data)))
        return result

    def move_playlist_tracks_before(
        self,
        playlist_uid: str,
        positions: Iterable[int],
        before_position: Optional[int],
    ) -> bool:
        """Move the entries at ``positions`` as one block before the entry at
        ``before_position``, or to the end for ``None``."""
        playlist_uid = _canonical_uuid4(playlist_uid)
        requested = {int(position) for position in positions}
        entries = self._playlist_entries(playlist_uid)
        moving = [entry for entry in entries if entry[0] in requested]
        if not moving or (before_position is not None and int(before_position) in requested):
            return False
        remaining = [entry for entry in entries if entry[0] not in requested]
        anchors = [position for position, _uid in remaining]
        anchor = None if before_position is None else int(before_position)
        insert_at = anchors.index(anchor) if anchor in anchors else len(remaining)
        reordered = remaining[:insert_at] + moving + remaining[insert_at:]
        if reordered == entries:
            return False
        with self.tx():
            self._write_playlist_entries(playlist_uid, [uid for _position, uid in reordered])
        return True

    def remove_playlist_tracks(self, playlist_uid: str, positions: Iterable[int]) -> None:
        """Remove the entries at ``positions``; the rest close up."""
        playlist_uid = _canonical_uuid4(playlist_uid)
        removed = {int(position) for position in positions}
        kept = [uid for position, uid in self._playlist_entries(playlist_uid) if position not in removed]
        with self.tx():
            self._write_playlist_entries(playlist_uid, kept)

    def replace_bpm_segments(self, track_uid: str, segments: Iterable[BpmSegmentRow | Dict[str, Any]]):
        uid = _canonical_uuid4_or_none(track_uid)
        if not uid:
            return
        rows = []
        for raw in segments:
            row = asdict(raw) if isinstance(raw, BpmSegmentRow) else dict(raw)
            bpm = row.get("bpm")
            ts = row.get("time_signature", 4)
            bpm_end = row.get("bpm_end")
            rows.append(
                (
                    uid,
                    int(row.get("seq_index", len(rows))),
                    float(row.get("start_sec", 0.0)),
                    float(row.get("end_sec", 0.0)),
                    float(row.get("duration_sec", 0.0)),
                    None if bpm is None else float(bpm),
                    row.get("bpm_rounded", None if bpm is None else int(round(float(bpm)))),
                    int(ts) if ts is not None else 4,
                    None if bpm_end is None else float(bpm_end),
                )
            )
        with self.tx():
            self.conn.execute("DELETE FROM track_bpm_segments WHERE track_uid=?;", (uid,))
            if rows:
                self.conn.executemany(
                    """
                    INSERT INTO track_bpm_segments(
                        track_uid, seq_index, start_sec, end_sec, duration_sec, bpm, bpm_rounded, time_signature, bpm_end
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    rows,
                )

    def replace_key_segments(self, track_uid: str, segments: Iterable[KeySegmentRow | Dict[str, Any]]):
        uid = _canonical_uuid4_or_none(track_uid)
        if not uid:
            return
        rows = []
        for raw in segments:
            row = asdict(raw) if isinstance(raw, KeySegmentRow) else dict(raw)
            key_value = row.get("key_value")
            key_label = str(row.get("key_label") or "")
            if not key_label and key_value is not None:
                try:
                    key_label = idx_to_labels(int(key_value))[0]
                except Exception:
                    key_label = ""
            rows.append(
                (
                    uid,
                    int(row.get("seq_index", len(rows))),
                    float(row.get("start_sec", 0.0)),
                    float(row.get("end_sec", 0.0)),
                    float(row.get("duration_sec", 0.0)),
                    None if key_value is None else int(key_value),
                    key_label,
                )
            )
        with self.tx():
            self.conn.execute("DELETE FROM track_key_segments WHERE track_uid=?;", (uid,))
            if rows:
                self.conn.executemany(
                    """
                    INSERT INTO track_key_segments(
                        track_uid, seq_index, start_sec, end_sec, duration_sec, key_value, key_label
                    ) VALUES(?, ?, ?, ?, ?, ?, ?);
                    """,
                    rows,
                )

    def replace_phrase_index(
        self,
        track_uid: str,
        sections: Iterable[PhraseIndexRow | Dict[str, Any]],
    ):
        """Replace the phrase/BPM/key search materialization for one track."""
        uid = _canonical_uuid4_or_none(track_uid)
        if not uid:
            return
        rows = []
        for raw in sections:
            row = asdict(raw) if isinstance(raw, PhraseIndexRow) else dict(raw)
            rows.append(
                (
                    uid,
                    int(row.get("phrase_seq_index", 0)),
                    int(row.get("section_index", 0)),
                    str(row.get("phrase_label") or "").strip().upper(),
                    float(row.get("phrase_start_sec", 0.0)),
                    float(row.get("phrase_end_sec", 0.0)),
                    float(row.get("phrase_duration_sec", 0.0)),
                    float(row.get("start_sec", 0.0)),
                    float(row.get("end_sec", 0.0)),
                    float(row.get("duration_sec", 0.0)),
                    None if row.get("bpm_seq_index") is None else int(row["bpm_seq_index"]),
                    None if row.get("bpm") is None else float(row["bpm"]),
                    None if row.get("bpm_rounded") is None else int(row["bpm_rounded"]),
                    None if row.get("time_signature") is None else int(row["time_signature"]),
                    None if row.get("key_seq_index") is None else int(row["key_seq_index"]),
                    None if row.get("key_value") is None else int(row["key_value"]),
                    str(row.get("key_label") or ""),
                )
            )
        with self.tx():
            self.conn.execute("DELETE FROM track_phrase_sections WHERE track_uid=?;", (uid,))
            if rows:
                self.conn.executemany(
                    """
                    INSERT INTO track_phrase_sections(
                        track_uid, phrase_seq_index, section_index, phrase_label,
                        phrase_start_sec, phrase_end_sec, phrase_duration_sec,
                        start_sec, end_sec, duration_sec,
                        bpm_seq_index, bpm, bpm_rounded, time_signature,
                        key_seq_index, key_value, key_label
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    rows,
                )

    def get_bpm_segments(self, track_uid: str) -> List[BpmSegmentRow]:
        track_uid = _canonical_uuid4(track_uid)
        cur = self.conn.execute(
            """
            SELECT track_uid, seq_index, start_sec, end_sec, duration_sec, bpm, bpm_rounded,
                   COALESCE(time_signature, 4) AS time_signature, bpm_end
            FROM track_bpm_segments
            WHERE track_uid=?
            ORDER BY seq_index ASC;
            """,
            (track_uid,),
        )
        return [BpmSegmentRow(**dict(row)) for row in cur.fetchall()]

    def mix_marks(self) -> dict[str, tuple[list, list, list]]:
        """Every track's marks the mix candidates read (core.search.mixability): uid -> (its phrases (start, end,
        label), its tempo rows (start, end, bpm, bpm_end), its key rows (start, end, key)), each in order."""
        out: dict[str, tuple[list, list, list]] = {}

        def rows(uid: str) -> tuple[list, list, list]:
            return out.setdefault(uid, ([], [], []))

        for uid, start, end, label in self.conn.execute(
                """SELECT track_uid, MIN(phrase_start_sec), MAX(phrase_end_sec), phrase_label
                   FROM track_phrase_sections GROUP BY track_uid, phrase_seq_index
                   ORDER BY track_uid, phrase_seq_index;"""):
            rows(uid)[0].append((float(start), float(end), str(label)))
        for uid, start, end, bpm, bpm_end in self.conn.execute(
                """SELECT track_uid, start_sec, end_sec, bpm, bpm_end FROM track_bpm_segments
                   WHERE bpm IS NOT NULL ORDER BY track_uid, seq_index;"""):
            rows(uid)[1].append((float(start), float(end), float(bpm), float(bpm_end) if bpm_end is not None else 0.0))
        for uid, start, end, key in self.conn.execute(
                """SELECT track_uid, start_sec, end_sec, key_value FROM track_key_segments
                   WHERE key_value IS NOT NULL ORDER BY track_uid, seq_index;"""):
            rows(uid)[2].append((float(start), float(end), int(key)))
        return out

    def search_phrase_sections(
        self,
        phrase_label: str,
        *,
        bpm_min: float | None = None,
        bpm_max: float | None = None,
        key_value: int | None = None,
        compatible_keys: bool = False,
        match_all: bool = True,
    ) -> List[TrackRow]:
        """Tracks with a labelled phrase section matching optional BPM/key.

        With both BPM and key active, ``match_all`` requires both on the same
        constant phrase section; otherwise either value may match.
        ``compatible_keys`` also accepts the harmonically compatible keys.
        """
        value_clauses: list[str] = []
        params: list[Any] = [str(phrase_label or "").strip().upper()]
        if bpm_min is not None and bpm_max is not None:
            lo, hi = sorted((float(bpm_min), float(bpm_max)))
            value_clauses.append("p.bpm BETWEEN ? AND ?")
            params.extend((lo, hi))
        if key_value is not None:
            keys = (sorted(harmonic_compatible_keys(key_value)) if compatible_keys
                    else [int(key_value)])
            value_clauses.append(f"p.key_value IN ({','.join('?' for _ in keys)})")
            params.extend(keys)
        values_sql = ""
        if value_clauses:
            operator = " AND " if match_all else " OR "
            values_sql = f" AND ({operator.join(value_clauses)})"
        cur = self.conn.execute(
            f"""
            SELECT DISTINCT t.*
            FROM track_phrase_sections p
            JOIN tracks t ON t.uid = p.track_uid
            WHERE p.phrase_label = ?{values_sql}
            ORDER BY t.added_ts DESC, t.title COLLATE NOCASE ASC;
            """,
            tuple(params),
        )
        return [_track_row_from_db_row(row) for row in cur.fetchall()]

    def search_bpm_transitions(
        self,
        from_bpm_min: float,
        from_bpm_max: float,
        to_bpm_min: float,
        to_bpm_max: float,
        *,
        min_duration_sec: float = 4.0,
        require_first_segment: bool = False,
    ) -> List[LinearTransitionRow]:
        first_clause = "AND a.seq_index = 0" if require_first_segment else ""
        ramp_first_clause = "AND r.seq_index = 0" if require_first_segment else ""
        cur = self.conn.execute(
            f"""
            SELECT
                t.uid AS track_uid,
                t.path,
                t.title,
                t.artist,
                a.seq_index AS from_seq_index,
                b.seq_index AS to_seq_index,
                a.start_sec AS from_start_sec,
                a.end_sec AS from_end_sec,
                a.duration_sec AS from_duration_sec,
                COALESCE(a.bpm_end, a.bpm) AS from_bpm,
                NULL AS from_key_value,
                '' AS from_key_label,
                b.start_sec AS to_start_sec,
                b.end_sec AS to_end_sec,
                b.duration_sec AS to_duration_sec,
                b.bpm AS to_bpm,
                NULL AS to_key_value,
                '' AS to_key_label,
                t.added_ts AS sort_added,
                a.seq_index AS sort_seq
            FROM track_bpm_segments a
            JOIN track_bpm_segments b
              ON b.track_uid = a.track_uid
             AND b.seq_index = a.seq_index + 1
            JOIN tracks t
              ON t.uid = a.track_uid
            WHERE a.duration_sec >= ?
              AND b.duration_sec >= ?
              AND a.bpm IS NOT NULL
              AND b.bpm IS NOT NULL
              AND COALESCE(a.bpm_end, a.bpm) BETWEEN ? AND ?
              AND b.bpm BETWEEN ? AND ?
              {first_clause}
            UNION ALL
            -- A tempo ramp is itself a change from its start tempo to its end tempo.
            SELECT
                t.uid, t.path, t.title, t.artist,
                r.seq_index, r.seq_index,
                r.start_sec, r.end_sec, r.duration_sec, r.bpm, NULL, '',
                r.start_sec, r.end_sec, r.duration_sec, r.bpm_end, NULL, '',
                t.added_ts, r.seq_index
            FROM track_bpm_segments r
            JOIN tracks t
              ON t.uid = r.track_uid
            WHERE r.bpm_end IS NOT NULL
              AND r.bpm IS NOT NULL
              AND r.duration_sec >= ?
              AND r.bpm BETWEEN ? AND ?
              AND r.bpm_end BETWEEN ? AND ?
              {ramp_first_clause}
            ORDER BY sort_added DESC, title COLLATE NOCASE ASC, sort_seq ASC;
            """,
            (
                float(min_duration_sec),
                float(min_duration_sec),
                float(from_bpm_min),
                float(from_bpm_max),
                float(to_bpm_min),
                float(to_bpm_max),
                float(min_duration_sec),
                float(from_bpm_min),
                float(from_bpm_max),
                float(to_bpm_min),
                float(to_bpm_max),
            ),
        )
        rows = []
        for row in cur.fetchall():
            values = dict(row)
            values.pop("sort_added", None)
            values.pop("sort_seq", None)
            rows.append(LinearTransitionRow(**values))
        return rows

    def search_harmonic_key_transitions(
        self,
        from_anchor_key: int,
        to_anchor_key: int,
        *,
        min_duration_sec: float = 4.0,
        require_first_segment: bool = False,
        compatible: bool = True,
    ) -> List[LinearTransitionRow]:
        """Key changes from one key to another; ``compatible`` widens both ends
        to their harmonically compatible keys, otherwise the keys are exact."""
        if compatible:
            from_keys = sorted(harmonic_compatible_keys(from_anchor_key))
            to_keys = sorted(harmonic_compatible_keys(to_anchor_key))
        else:
            from_keys = [int(from_anchor_key) % 24]
            to_keys = [int(to_anchor_key) % 24]
        if not from_keys or not to_keys:
            return []
        from_placeholders = ",".join("?" for _ in from_keys)
        to_placeholders = ",".join("?" for _ in to_keys)
        first_clause = "AND a.seq_index = 0" if require_first_segment else ""
        query = f"""
            SELECT
                t.uid AS track_uid,
                t.path,
                t.title,
                t.artist,
                a.seq_index AS from_seq_index,
                b.seq_index AS to_seq_index,
                a.start_sec AS from_start_sec,
                a.end_sec AS from_end_sec,
                a.duration_sec AS from_duration_sec,
                NULL AS from_bpm,
                a.key_value AS from_key_value,
                a.key_label AS from_key_label,
                b.start_sec AS to_start_sec,
                b.end_sec AS to_end_sec,
                b.duration_sec AS to_duration_sec,
                NULL AS to_bpm,
                b.key_value AS to_key_value,
                b.key_label AS to_key_label
            FROM track_key_segments a
            JOIN track_key_segments b
              ON b.track_uid = a.track_uid
             AND b.seq_index > a.seq_index
            JOIN tracks t
              ON t.uid = a.track_uid
            WHERE a.duration_sec >= ?
              AND b.duration_sec >= ?
              AND a.key_value IN ({from_placeholders})
              AND b.key_value IN ({to_placeholders})
              {first_clause}
            ORDER BY t.added_ts DESC, t.title COLLATE NOCASE ASC, a.seq_index ASC;
        """
        params: list[Any] = [float(min_duration_sec), float(min_duration_sec), *from_keys, *to_keys]
        cur = self.conn.execute(query, tuple(params))
        return [LinearTransitionRow(**dict(row)) for row in cur.fetchall()]

    # Query / Search 
    def list_all(self, order_by: str = "added_ts DESC", limit: Optional[int]=None, offset: int=0) -> List[TrackRow]:
        q = f"SELECT * FROM tracks ORDER BY {_safe_order_by(order_by)}"
        params: Tuple[Any, ...] = ()
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        cur = self.conn.execute(q, params)
        return [_track_row_from_db_row(row) for row in cur.fetchall()]

    def set_tempo_key(self, uid: str, bpm: float, key: Optional[int]) -> None:
        """The track's BPM and key (made from its marks: core.library.segments.track_index); committed with the
        job's other writes, by the owner."""
        self.conn.execute("UPDATE tracks SET bpm=?, key=? WHERE uid=?;", (float(bpm), key, _canonical_uuid4(uid)))

    def write_track_index(self, uid: str, index) -> None:
        """The track's rows made from its marks (a core.library.segments.TrackIndex: the BPM, the key, the segment and
        phrase tables), in one transaction."""
        with self.tx():
            self.set_tempo_key(uid, index.bpm, index.key)
            self.replace_bpm_segments(uid, index.bpm_rows)
            self.replace_key_segments(uid, index.key_rows)
            self.replace_phrase_index(uid, index.phrase_rows)

    def update_track_index(self, uid: str, index) -> bool:
        """write_track_index where its rows differ from the stored ones; the same, nothing is kept (the writes undone
        to a savepoint: no commit, no row id used). Whether they differed."""
        with self.tx():
            before = self.track_index_rows(uid)
            self.conn.execute("SAVEPOINT track_index;")
            try:
                self.write_track_index(uid, index)
                same = self.track_index_rows(uid) == before
            except BaseException:
                self._rollback_to("track_index")
                raise
            if same:
                self.conn.execute("ROLLBACK TO track_index;")
            self.conn.execute("RELEASE track_index;")
            return not same

    def track_index_rows(self, uid: str) -> tuple:
        """The track's rows write_track_index writes, as stored, their row ids left out (compared before and after a
        sync)."""
        uid = _canonical_uuid4(uid)
        if self._index_queries is None:     # the tables' columns, read once a connection (the schema stays)
            self._index_queries = []
            for table, order in (("track_bpm_segments", "seq_index"), ("track_key_segments", "seq_index"),
                                 ("track_phrase_sections", "phrase_seq_index, section_index")):
                columns = ", ".join(col[1] for col in self.conn.execute(f"PRAGMA table_info({table});")
                                    if col[1] != "id")
                self._index_queries.append(f"SELECT {columns} FROM {table} WHERE track_uid=? ORDER BY {order};")
        head = self.conn.execute("SELECT bpm, key FROM tracks WHERE uid=?;", (uid,)).fetchone()
        tables = [tuple(tuple(row) for row in self.conn.execute(query, (uid,)).fetchall())
                  for query in self._index_queries]
        return (tuple(head) if head is not None else None, *tables)

    # Utilities
    def touch_file_stats(self, path: str):
        """Refresh file mtime/size (for change detection)."""
        try:
            st = os.stat(path)
            self.conn.execute(
                "UPDATE tracks SET file_mtime=?, file_size=? WHERE path=?;",
                (st.st_mtime, st.st_size, path)
            )
            self.conn.commit()
        except FileNotFoundError:
            pass

    # NPZ/JSON path