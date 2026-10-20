"""One-off development-library migration for 0.6.0: the library keyed by UID.

``tracks`` gets its ``uid`` as the PRIMARY KEY (``path`` stays UNIQUE and becomes an
editable attribute), ``playlists`` a UUIDv4 ``uid`` in place of the integer id, and
``playlist_tracks`` one row per entry, (playlist_uid, position) -> track_uid, so a track
may be in a playlist more than once. Each playlist keeps its order (positions renumbered
0, 1, ...). The segment tables and the feature files do not change.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from core.library_handler import LibraryDB
from core.library_version import read_library_version, write_library_version

SOURCE_VERSION = "0.5.0"
TARGET_VERSION = "0.6.0"
TRACK_COLUMNS = ("uid", "path", "title", "artist", "album", "bpm", "key", "duration", "total_samples",
                 "rating", "added_ts", "comment", "file_mtime", "file_size")


def _sqlite_snapshot(source: sqlite3.Connection, destination: Path) -> None:
    target = sqlite3.connect(str(destination))
    try:
        source.backup(target)
    finally:
        target.close()


def _check_source(con: sqlite3.Connection) -> dict[str, int]:
    """Every track has a canonical UUIDv4 of its own and a path; the counts to compare after."""
    uids = set()
    for uid, path in con.execute("SELECT uid, path FROM tracks"):
        parsed = UUID(str(uid))
        if parsed.version != 4 or str(parsed) != uid:
            raise RuntimeError(f"track uid not a canonical UUIDv4: {uid!r}")
        if uid in uids:
            raise RuntimeError(f"track uid used twice: {uid}")
        if not path:
            raise RuntimeError(f"track {uid} has no path")
        uids.add(uid)
    orphans = con.execute(
        "SELECT COUNT(*) FROM playlist_tracks pt LEFT JOIN tracks t ON t.path = pt.track_path "
        "WHERE t.path IS NULL").fetchone()[0]
    if orphans:
        raise RuntimeError(f"{orphans} playlist entries name no track")
    return {
        "tracks": len(uids),
        "playlists": int(con.execute("SELECT COUNT(*) FROM playlists").fetchone()[0]),
        "playlist_entries": int(con.execute("SELECT COUNT(*) FROM playlist_tracks").fetchone()[0]),
    }


def _rebuild(staging_path: Path) -> dict[str, int]:
    """Move the old tables aside, let LibraryDB create the new ones, copy, drop the old ones."""
    con = sqlite3.connect(str(staging_path))
    try:
        con.execute("PRAGMA foreign_keys=OFF")
        for table in ("playlist_tracks", "playlists", "tracks"):
            con.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
        # their indexes went with them; the new tables get their own under these names
        con.execute("DROP INDEX IF EXISTS idx_playlists_sort")
        con.execute("DROP INDEX IF EXISTS idx_playlist_tracks_position")
        con.commit()
    finally:
        con.close()

    db = LibraryDB(str(staging_path))
    db.connect()
    try:
        con = db.conn
        columns = ", ".join(TRACK_COLUMNS)
        con.execute(f"INSERT INTO tracks({columns}) SELECT {columns} FROM tracks_old")
        playlists = con.execute(
            "SELECT id, name, sort_index FROM playlists_old ORDER BY sort_index, id").fetchall()
        for sort_index, (old_id, name, _old_index) in enumerate(playlists):
            playlist_uid = str(uuid4())
            con.execute("INSERT INTO playlists(uid, name, sort_index) VALUES(?, ?, ?)",
                        (playlist_uid, name, sort_index))
            entries = con.execute(
                "SELECT t.uid FROM playlist_tracks_old pt JOIN tracks_old t ON t.path = pt.track_path "
                "WHERE pt.playlist_id=? ORDER BY pt.position, pt.track_path", (old_id,)).fetchall()
            con.executemany(
                "INSERT INTO playlist_tracks(playlist_uid, position, track_uid) VALUES(?, ?, ?)",
                ((playlist_uid, position, uid) for position, (uid,) in enumerate(entries)))
        for table in ("playlist_tracks_old", "playlists_old", "tracks_old"):
            con.execute(f"DROP TABLE {table}")
        con.commit()
        problems = con.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError(f"foreign key check failed: {problems[:5]}")
        integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.lower() != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        con.execute("VACUUM")
        return {
            "tracks": int(con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]),
            "playlists": int(con.execute("SELECT COUNT(*) FROM playlists").fetchone()[0]),
            "playlist_entries": int(con.execute("SELECT COUNT(*) FROM playlist_tracks").fetchone()[0]),
        }
    finally:
        db.close()


def migrate_library(lib_path: str | Path, dry_run: bool = False) -> dict[str, object]:
    """Back up and migrate one development library from 0.5.0 to 0.6.0.

    The database is rebuilt in a staging copy (checked: foreign keys, integrity, the same
    numbers of tracks, playlists and entries) that replaces ``library.db`` only when all
    succeeded. Run it with the app closed: a library with a pending write-ahead log is refused.
    The backup is kept regardless of success.
    """
    library_dir = Path(lib_path).resolve()
    db_path = (library_dir / "library.db").resolve()
    if db_path.parent != library_dir or not db_path.is_file():
        raise FileNotFoundError(db_path)
    current = read_library_version(library_dir)
    if current != SOURCE_VERSION:
        raise RuntimeError(f"library version {current}, expected {SOURCE_VERSION}")
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise RuntimeError(f"{wal.name} is not empty: close the app (or let it finish writing) first")

    source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        before = _check_source(source)
    finally:
        source.close()
    report: dict[str, object] = {"library": str(library_dir), "before": before}
    if dry_run:
        return report

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (library_dir / "backups").resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    db_backup = backup_dir / f"library_pre_{TARGET_VERSION}_{stamp}.db"
    fd, staging_name = tempfile.mkstemp(prefix="library_0_6_0_", suffix=".db", dir=library_dir)
    os.close(fd)
    staging_path = Path(staging_name).resolve()
    if staging_path.parent != library_dir:
        raise RuntimeError("staging database escaped the library directory")
    source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        _sqlite_snapshot(source, db_backup)
        _sqlite_snapshot(source, staging_path)
    finally:
        source.close()
    report["backup_db"] = str(db_backup)

    try:
        after = _rebuild(staging_path)
        if after != before:
            raise RuntimeError(f"counts changed: {before} -> {after}")
        os.replace(staging_path, db_path)
        write_library_version(library_dir, TARGET_VERSION)
    except Exception:
        if staging_path.exists():
            staging_path.unlink()
        raise
    report["after"] = after
    report["library_version"] = TARGET_VERSION
    return report


if __name__ == "__main__":
    import sys

    print(migrate_library(sys.argv[1] if len(sys.argv) > 1 else "library", dry_run="--dry-run" in sys.argv))
