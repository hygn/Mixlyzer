"""One-off development-library migration for the 0.4.0 phrase search index."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from core.analysis_lib_handler import FeatureNPZStore
from core.library_handler import LibraryDB
from core.library_version import CURRENT_LIBRARY_VERSION, write_library_version
from core.linear_segments import build_bpm_segments, build_key_segments, build_phrase_index_rows
from utils.phrases import assigned_phrases, extract_phrase_segments


def _sqlite_snapshot(source: sqlite3.Connection, destination: Path) -> None:
    target = sqlite3.connect(str(destination))
    try:
        source.backup(target)
    finally:
        target.close()


def migrate_test_library(lib_path: str | Path) -> dict[str, object]:
    """Back up and atomically replace one development library with 0.4.0.

    All feature files are read and the staging database is fully indexed and
    integrity-checked before it replaces ``library.db``.  The backup is kept
    regardless of success.
    """
    library_dir = Path(lib_path).resolve()
    db_path = (library_dir / "library.db").resolve()
    if db_path.parent != library_dir or not db_path.is_file():
        raise FileNotFoundError(db_path)

    backup_dir = (library_dir / "backups").resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"library_pre_{CURRENT_LIBRARY_VERSION}_{stamp}.db"
    fd, staging_name = tempfile.mkstemp(prefix="library_0_4_0_", suffix=".db", dir=library_dir)
    os.close(fd)
    staging_path = Path(staging_name).resolve()
    if staging_path.parent != library_dir:
        raise RuntimeError("staging database escaped the library directory")

    source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        _sqlite_snapshot(source, backup_path)
        _sqlite_snapshot(source, staging_path)
    finally:
        source.close()

    indexed_tracks = 0
    indexed_sections = 0
    total_tracks = 0
    try:
        store = FeatureNPZStore(str(library_dir), compressed=True)
        db = LibraryDB(str(staging_path))
        db.connect()
        try:
            tracks = db.list_all()
            total_tracks = len(tracks)
            for track in tracks:
                if not track.uid:
                    continue
                features = store.load(track.uid)
                phrases = assigned_phrases(extract_phrase_segments(features, track.duration))
                bpm_rows = build_bpm_segments(features.get("tempo_segments"))
                key_rows = build_key_segments(features.get("key_segments"))
                sections = build_phrase_index_rows(phrases, bpm_rows, key_rows)
                db.replace_phrase_index(track.uid, sections)
                if sections:
                    indexed_tracks += 1
                    indexed_sections += len(sections)

            actual_sections = int(
                db.conn.execute("SELECT COUNT(*) FROM track_phrase_sections").fetchone()[0]
            )
            if actual_sections != indexed_sections:
                raise RuntimeError(
                    f"phrase index row mismatch: expected {indexed_sections}, got {actual_sections}"
                )
            integrity = str(db.conn.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity.lower() != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        finally:
            db.close()

        os.replace(staging_path, db_path)
        write_library_version(library_dir, CURRENT_LIBRARY_VERSION)
    except Exception:
        if staging_path.exists():
            staging_path.unlink()
        raise

    return {
        "backup_path": str(backup_path),
        "library_version": CURRENT_LIBRARY_VERSION,
        "total_tracks": total_tracks,
        "indexed_tracks": indexed_tracks,
        "indexed_sections": indexed_sections,
    }


if __name__ == "__main__":
    print(migrate_test_library("library"))
