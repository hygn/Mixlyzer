from __future__ import annotations
import os, sqlite3, time, csv
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

from core.linear_segments import harmonic_compatible_keys
from utils.labels import idx_to_labels


def _canonical_uuid4(value: object) -> str:
    parsed = UUID(str(value).strip())
    if parsed.version != 4:
        raise ValueError(f"Expected UUIDv4, got version {parsed.version}")
    return str(parsed)


def _canonical_uuid4_or_none(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return _canonical_uuid4(text)


def _track_row_from_db_row(row: sqlite3.Row) -> "TrackRow":
    data = dict(row)
    try:
        data["uid"] = _canonical_uuid4_or_none(data.get("uid"))
    except ValueError:
        data["uid"] = None
    return TrackRow(**data)

@dataclass
class TrackRow:
    path: str                  # PRIMARY KEY
    uid: Optional[str] = None  # UID for NPZ/JSON linkage (unique string)
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
        uid = _canonical_uuid4(uid)
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

@dataclass
class BpmSegmentRow:
    track_uid: str
    seq_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    bpm: Optional[float] = None
    bpm_rounded: Optional[int] = None


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

# DB Helper
class LibraryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
    
        # Create tracks table if missing
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            path        TEXT PRIMARY KEY,
            uid         TEXT,
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
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tbs_track_uid ON track_bpm_segments(track_uid);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tbs_bpm_duration ON track_bpm_segments(bpm_rounded, duration_sec);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tks_track_uid ON track_key_segments(track_uid);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tks_key_duration ON track_key_segments(key_value, duration_sec);")
        self._conn.commit()

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

    def _column_exists(self, table: str, col: str) -> bool:
        cur = self.conn.execute(f"PRAGMA table_info({table});")
        return any(row["name"] == col for row in cur.fetchall())

    @contextmanager
    def tx(self):
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # UPSERT / CRUD
    def upsert(self, row: TrackRow):
        q = """
        INSERT INTO tracks(path, uid, title, artist, album, bpm, key, duration, total_samples, rating, added_ts, comment, file_mtime, file_size)
        VALUES(:path, :uid, :title, :artist, :album, :bpm, :key, :duration, :total_samples, :rating, :added_ts, :comment, :file_mtime, :file_size)
        ON CONFLICT(path) DO UPDATE SET
            uid           = COALESCE(excluded.uid, tracks.uid),
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
        self.conn.execute(q, asdict(row))

    def upsert_meta(self, meta: Dict[str, Any]):
        tr = TrackRow.from_meta(meta)
        self.upsert(tr)
        return tr.uid

    def upsert_many(self, rows: Iterable[TrackRow]):
        with self.tx():
            for row in rows:
                self.upsert(row)

    def delete_paths(self, paths: Iterable[str]):
        with self.tx():
            uid_rows = []
            for path in paths:
                row = self.get(path)
                if row and row.uid:
                    uid_rows.append((row.uid,))
            self.conn.executemany("DELETE FROM tracks WHERE path = ?;", ((p,) for p in paths))
            if uid_rows:
                self.conn.executemany("DELETE FROM track_bpm_segments WHERE track_uid = ?;", uid_rows)
                self.conn.executemany("DELETE FROM track_key_segments WHERE track_uid = ?;", uid_rows)

    def get(self, path: str) -> Optional[TrackRow]:
        cur = self.conn.execute("SELECT * FROM tracks WHERE path=?;", (path,))
        row = cur.fetchone()
        return _track_row_from_db_row(row) if row else None

    def get_by_uid(self, uid: str) -> Optional[TrackRow]:
        try:
            uid = _canonical_uuid4(uid)
        except ValueError:
            return None
        cur = self.conn.execute("SELECT * FROM tracks WHERE uid=?;", (uid,))
        row = cur.fetchone()
        return _track_row_from_db_row(row) if row else None

    def update_uid(self, path: str, uid: Optional[str]):
        """Set/update UID for the given path (allows None)."""
        uid = _canonical_uuid4_or_none(uid)
        self.conn.execute("UPDATE tracks SET uid=? WHERE path=?;", (uid, path))
        self.conn.commit()

    def rekey_path(self, old_path: str, new_path: str):
        """
        When path (PK) changes (e.g., file move), swap only the PK on the same row.
        uid stays the same so NPZ/JSON linkage is preserved.
        """
        with self.tx():
            row = self.get(old_path)
            if not row:
                return
            # Insert with new path (overwrite on conflict)
            row.path = new_path
            self.upsert(row)
            # Delete old row
            self.conn.execute("DELETE FROM tracks WHERE path=?;", (old_path,))

    def replace_bpm_segments(self, track_uid: str, segments: Iterable[BpmSegmentRow | Dict[str, Any]]):
        uid = _canonical_uuid4_or_none(track_uid)
        if not uid:
            return
        rows = []
        for raw in segments:
            row = asdict(raw) if isinstance(raw, BpmSegmentRow) else dict(raw)
            bpm = row.get("bpm")
            rows.append(
                (
                    uid,
                    int(row.get("seq_index", len(rows))),
                    float(row.get("start_sec", 0.0)),
                    float(row.get("end_sec", 0.0)),
                    float(row.get("duration_sec", 0.0)),
                    None if bpm is None else float(bpm),
                    row.get("bpm_rounded", None if bpm is None else int(round(float(bpm)))),
                )
            )
        with self.tx():
            self.conn.execute("DELETE FROM track_bpm_segments WHERE track_uid=?;", (uid,))
            if rows:
                self.conn.executemany(
                    """
                    INSERT INTO track_bpm_segments(
                        track_uid, seq_index, start_sec, end_sec, duration_sec, bpm, bpm_rounded
                    ) VALUES(?, ?, ?, ?, ?, ?, ?);
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

    def get_bpm_segments(self, track_uid: str) -> List[BpmSegmentRow]:
        track_uid = _canonical_uuid4(track_uid)
        cur = self.conn.execute(
            """
            SELECT track_uid, seq_index, start_sec, end_sec, duration_sec, bpm, bpm_rounded
            FROM track_bpm_segments
            WHERE track_uid=?
            ORDER BY seq_index ASC;
            """,
            (track_uid,),
        )
        return [BpmSegmentRow(**dict(row)) for row in cur.fetchall()]

    def get_key_segments(self, track_uid: str) -> List[KeySegmentRow]:
        track_uid = _canonical_uuid4(track_uid)
        cur = self.conn.execute(
            """
            SELECT track_uid, seq_index, start_sec, end_sec, duration_sec, key_value, key_label
            FROM track_key_segments
            WHERE track_uid=?
            ORDER BY seq_index ASC;
            """,
            (track_uid,),
        )
        return [KeySegmentRow(**dict(row)) for row in cur.fetchall()]

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
                a.bpm AS from_bpm,
                NULL AS from_key_value,
                '' AS from_key_label,
                b.start_sec AS to_start_sec,
                b.end_sec AS to_end_sec,
                b.duration_sec AS to_duration_sec,
                b.bpm AS to_bpm,
                NULL AS to_key_value,
                '' AS to_key_label
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
              AND a.bpm BETWEEN ? AND ?
              AND b.bpm BETWEEN ? AND ?
              {first_clause}
            ORDER BY t.added_ts DESC, t.title COLLATE NOCASE ASC, a.seq_index ASC;
            """,
            (
                float(min_duration_sec),
                float(min_duration_sec),
                float(from_bpm_min),
                float(from_bpm_max),
                float(to_bpm_min),
                float(to_bpm_max),
            ),
        )
        return [LinearTransitionRow(**dict(row)) for row in cur.fetchall()]

    def search_harmonic_key_transitions(
        self,
        from_anchor_key: int,
        to_anchor_key: int,
        *,
        min_duration_sec: float = 4.0,
        require_first_segment: bool = False,
    ) -> List[LinearTransitionRow]:
        from_keys = sorted(harmonic_compatible_keys(from_anchor_key))
        to_keys = sorted(harmonic_compatible_keys(to_anchor_key))
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
        q = f"SELECT * FROM tracks ORDER BY {order_by}"
        params: Tuple[Any, ...] = ()
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        cur = self.conn.execute(q, params)
        return [_track_row_from_db_row(row) for row in cur.fetchall()]

    def search_like(self, query: str, cols: Tuple[str,...]=("title","artist","album","comment"),
                    order_by: str = "added_ts DESC", limit: Optional[int]=200, offset:int=0) -> List[TrackRow]:
        kw = f"%{query}%"
        where = " OR ".join([f"{c} LIKE ?" for c in cols])
        params: List[Any] = [kw]*len(cols)
        q = f"SELECT * FROM tracks WHERE {where} ORDER BY {order_by}"
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        cur = self.conn.execute(q, tuple(params))
        return [_track_row_from_db_row(row) for row in cur.fetchall()]

    # Utilities 
    def export_csv(self, out_path: str):
        cur = self.conn.execute("SELECT * FROM tracks ORDER BY added_ts DESC;")
        cols = [d[0] for d in cur.description]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for row in cur.fetchall():
                w.writerow([row[c] for c in cols])

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
    @staticmethod
    def npz_path_for(base_dir: str, uid: str) -> str:
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{_canonical_uuid4(uid)}.npz")

    @staticmethod
    def json_path_for(base_dir: str, uid: str) -> str:
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{_canonical_uuid4(uid)}.json")
