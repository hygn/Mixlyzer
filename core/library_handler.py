from __future__ import annotations
import os, sqlite3, time, csv
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Dict, Any, Tuple
from uuid import uuid4
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
    rating: int = 0
    added_ts: int = 0                  # epoch seconds
    comment: str = ""
    file_mtime: float = 0.0
    file_size: int = 0

    @staticmethod
    def _stable_uid_from_meta() -> Optional[str]:
       return uuid4().__str__()

    @staticmethod
    def from_meta(meta: Dict[str, Any]) -> "TrackRow":
        """Analysis meta (dict) → storable TrackRow. Accepts int/str key (uses int here)."""
        key = meta.get("key")
        uid = meta.get("uid")
        if uid is None:
            uid = TrackRow._stable_uid_from_meta()

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
            "rating": self.rating,
            "added_ts": self.added_ts,
            "comment": self.comment,
            "file_mtime": self.file_mtime,
            "file_size": self.file_size,
        }

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
            rating      INTEGER DEFAULT 0,
            added_ts    INTEGER NOT NULL,
            comment     TEXT DEFAULT '',
            file_mtime  REAL DEFAULT 0,
            file_size   INTEGER DEFAULT 0
        );
        """)
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
        """
        UPSERT keyed by PRIMARY KEY(path).
        None values are left untouched via COALESCE.
        """
        q = """
        INSERT INTO tracks(path, uid, title, artist, album, bpm, key, duration, rating, added_ts, comment, file_mtime, file_size)
        VALUES(:path, :uid, :title, :artist, :album, :bpm, :key, :duration, :rating, :added_ts, :comment, :file_mtime, :file_size)
        ON CONFLICT(path) DO UPDATE SET
            uid        = COALESCE(excluded.uid,        tracks.uid),
            title      = COALESCE(excluded.title,      tracks.title),
            artist     = COALESCE(excluded.artist,     tracks.artist),
            album      = COALESCE(excluded.album,      tracks.album),
            bpm        = COALESCE(excluded.bpm,        tracks.bpm),
            key        = COALESCE(excluded.key,        tracks.key),
            duration   = COALESCE(excluded.duration,   tracks.duration),
            rating     = COALESCE(excluded.rating,     tracks.rating),
            added_ts   = CASE WHEN tracks.added_ts IS NULL OR tracks.added_ts=0 THEN excluded.added_ts ELSE tracks.added_ts END,
            comment    = COALESCE(excluded.comment,    tracks.comment),
            file_mtime = COALESCE(excluded.file_mtime, tracks.file_mtime),
            file_size  = COALESCE(excluded.file_size,  tracks.file_size)
        ;
        """
        self.conn.execute(q, asdict(row))

    def upsert_meta(self, meta: Dict[str, Any]):
        tr = TrackRow.from_meta(meta)
        self.upsert(tr)
        return tr.uid

    def upsert_many(self, rows: Iterable[TrackRow]):
        with self.tx():
            for r in rows:
                self.upsert(r)

    def delete_paths(self, paths: Iterable[str]):
        with self.tx():
            self.conn.executemany("DELETE FROM tracks WHERE path = ?;", ((p,) for p in paths))

    def get(self, path: str) -> Optional[TrackRow]:
        cur = self.conn.execute("SELECT * FROM tracks WHERE path=?;", (path,))
        r = cur.fetchone()
        return TrackRow(**dict(r)) if r else None

    def get_by_uid(self, uid: str) -> Optional[TrackRow]:
        cur = self.conn.execute("SELECT * FROM tracks WHERE uid=?;", (uid,))
        r = cur.fetchone()
        return TrackRow(**dict(r)) if r else None

    def update_uid(self, path: str, uid: Optional[str]):
        """Set/update UID for the given path (allows None)."""
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

    # Query / Search 
    def list_all(self, order_by: str = "added_ts DESC", limit: Optional[int]=None, offset: int=0) -> List[TrackRow]:
        q = f"SELECT * FROM tracks ORDER BY {order_by}"
        params: Tuple[Any, ...] = ()
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        cur = self.conn.execute(q, params)
        return [TrackRow(**dict(row)) for row in cur.fetchall()]

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
        return [TrackRow(**dict(row)) for row in cur.fetchall()]

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
        return os.path.join(base_dir, f"{uid}.npz")

    @staticmethod
    def json_path_for(base_dir: str, uid: str) -> str:
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"{uid}.json")
