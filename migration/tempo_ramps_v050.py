"""One-off development-library migration for 0.5.0: native linear tempo ramps.

Tempo segments may now carry a sixth column, the tempo at the segment's end (a linear tempo
ramp when it differs from ``bpm``), and ``track_bpm_segments`` gets a ``bpm_end`` column.

Ramps written into a 0.4.0 library could only be stored as runs of short constant-tempo
rows (one row per bar). Each run of at least ``MIN_RUN_ROWS`` consecutive rows of at most
``MAX_RUN_BEATS`` beats whose tempo moves the same way is replaced by one ramp row when that
row reproduces the run's beats within ``MAX_BEAT_ERROR_SEC``; otherwise the run is kept as
it is. Steady rows, longer step changes and everything outside the runs stay untouched.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from core.analysis_lib_handler import FeatureNPZStore
from core.beat_geometry import downbeat_times
from core.library_handler import LibraryDB
from core.library_version import read_library_version, write_library_version
from core.linear_segments import build_bpm_segments, build_key_segments, build_phrase_index_rows
from core.tempo_rows import ramp_beat_times, rebuild_grid_from_segments, with_end_bpm
from utils.phrases import assigned_phrases, extract_phrase_segments

SOURCE_VERSION = "0.4.0"
TARGET_VERSION = "0.5.0"
MIN_RUN_ROWS = 3
MAX_RUN_BEATS = 4.5
MAX_BEAT_ERROR_SEC = 0.005


def _sqlite_snapshot(source: sqlite3.Connection, destination: Path) -> None:
    target = sqlite3.connect(str(destination))
    try:
        source.backup(target)
    finally:
        target.close()


def _chunk_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i, j] of per-bar runs: >= MIN_RUN_ROWS short rows, tempo monotone."""
    beats_per_row = (rows[:, 1] - rows[:, 0]) * rows[:, 2] / 60.0
    short = (beats_per_row <= MAX_RUN_BEATS) & (np.abs(rows[:, 5] - rows[:, 2]) < 1e-9)
    runs = []
    i = 0
    while i < rows.shape[0]:
        if not short[i]:
            i += 1
            continue
        j = i
        while j + 1 < rows.shape[0] and short[j + 1]:
            j += 1
        if j - i + 1 >= MIN_RUN_ROWS:
            steps = np.diff(rows[i:j + 1, 2])
            if np.all(steps > 0) or np.all(steps < 0):
                runs.append((i, j))
        i = j + 1
    return runs


def _fit_ramp_row(rows: np.ndarray, i: int, j: int, beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray | None:
    """One ramp row over rows[i..j] whose beats match ``beats`` in that span, or None."""
    start, end = float(rows[i, 0]), float(rows[j, 1])
    span = beats[(beats >= start - 1e-6) & (beats < end - 1e-6)]
    if span.size < 3:
        return None
    anchor = float(span[0])

    def ramp(params):
        b0, b1 = params
        if b0 <= 0 or b1 <= 0:
            return None
        return ramp_beat_times(start, end, b0, b1, anchor, end - 1e-6)

    def cost(params):
        got = ramp(params)
        if got is None or got.size != span.size:
            return 1e3
        return float(np.max(np.abs(got - span)))

    best = minimize(cost, x0=[float(rows[i, 2]), float(rows[j, 2])], method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    if best.fun > MAX_BEAT_ERROR_SEC:
        return None
    b0, b1 = (float(v) for v in best.x)
    ts = float(rows[i, 4])
    # The bars keep counting as before: the run's first downbeat becomes the ramp's
    # downbeat reference (the first per-bar row may start mid-bar).
    inside = downbeats[(downbeats >= start - 1e-6) & (downbeats < end - 1e-6)]
    inizio = float(inside[0]) if inside.size else anchor
    return np.asarray([start, end, b0, inizio, ts, b1], dtype=np.float64)


def convert_track_segments(segments, duration_sec: float) -> tuple[np.ndarray, list[str]]:
    """The track's rows with every per-bar ramp run replaced by one ramp row (when exact)."""
    rows = with_end_bpm(segments)
    notes: list[str] = []
    if rows.shape[0] == 0:
        return rows, notes
    beats = rebuild_grid_from_segments(rows)
    beats = beats[(beats >= 0.0) & (beats <= float(duration_sec) + 1e-6)]
    downbeats = np.asarray(downbeat_times(beats, rows) if beats.size else [], dtype=np.float64)
    out = rows.copy()
    for i, j in reversed(_chunk_runs(rows)):
        ramp = _fit_ramp_row(rows, i, j, beats, downbeats)
        if ramp is None:
            notes.append(f"kept run {i}..{j} ({rows[i, 0]:.1f}-{rows[j, 1]:.1f}s): no exact ramp")
            continue
        out = np.vstack([out[:i], ramp[None, :], out[j + 1:]])
        notes.append(f"run {i}..{j} ({rows[i, 0]:.1f}-{rows[j, 1]:.1f}s) -> ramp {ramp[2]:.2f} -> {ramp[5]:.2f} BPM")
    if notes and any(n.startswith("run") for n in notes):
        new_beats = rebuild_grid_from_segments(out)
        new_beats = new_beats[(new_beats >= 0.0) & (new_beats <= float(duration_sec) + 1e-6)]
        if new_beats.size != beats.size or np.max(np.abs(new_beats - beats)) > MAX_BEAT_ERROR_SEC:
            return rows, [f"conversion rejected: beats moved (count {beats.size} -> {new_beats.size})"]
        new_downbeats = np.asarray(downbeat_times(new_beats, out), dtype=np.float64)
        if new_downbeats.size != downbeats.size or np.max(np.abs(new_downbeats - downbeats)) > MAX_BEAT_ERROR_SEC:
            return rows, [f"conversion rejected: downbeats moved (count {downbeats.size} -> {new_downbeats.size})"]
    return out, notes


def migrate_library(lib_path: str | Path, dry_run: bool = False) -> dict[str, object]:
    """Back up and migrate one development library from 0.4.0 to 0.5.0.

    The database is migrated in a staging copy (column added, BPM segment and phrase index
    rows rebuilt for converted tracks, integrity-checked) that replaces ``library.db`` only
    when everything succeeded. Feature files that change are backed up before they are
    rewritten and restored if anything fails. Backups are kept regardless of success.
    """
    library_dir = Path(lib_path).resolve()
    db_path = (library_dir / "library.db").resolve()
    if db_path.parent != library_dir or not db_path.is_file():
        raise FileNotFoundError(db_path)
    current = read_library_version(library_dir)
    if current != SOURCE_VERSION:
        raise RuntimeError(f"library version {current}, expected {SOURCE_VERSION}")

    store = FeatureNPZStore(str(library_dir), compressed=True)
    source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        uids = [row[0] for row in source.execute("SELECT uid FROM tracks WHERE uid IS NOT NULL")]
    finally:
        source.close()

    planned: dict[str, tuple[dict, np.ndarray, list[str]]] = {}
    for uid in uids:
        features = store.load(uid)
        segments = features.get("tempo_segments")
        if segments is None or np.asarray(segments).size == 0:
            continue
        converted, notes = convert_track_segments(segments, float(features.get("duration_sec", 0.0)))
        if any(n.startswith("run") for n in notes):
            planned[uid] = (features, converted, notes)
        elif notes:
            planned.setdefault(uid, (features, None, notes))
    report = {
        "library": str(library_dir),
        "tracks": len(uids),
        "converted": {uid: notes for uid, (_f, rows, notes) in planned.items() if rows is not None},
        "kept": {uid: notes for uid, (_f, rows, notes) in planned.items() if rows is None},
    }
    if dry_run:
        return report

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = (library_dir / "backups").resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    db_backup = backup_dir / f"library_pre_{TARGET_VERSION}_{stamp}.db"
    npz_backup = backup_dir / f"features_pre_{TARGET_VERSION}_{stamp}"
    fd, staging_name = tempfile.mkstemp(prefix="library_0_5_0_", suffix=".db", dir=library_dir)
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

    converted = {uid: (f, rows) for uid, (f, rows, _n) in planned.items() if rows is not None}
    npz_backup.mkdir(parents=True, exist_ok=True)
    for uid in converted:
        shutil.copy2(library_dir / f"{uid}.npz", npz_backup / f"{uid}.npz")
    report["backup_db"] = str(db_backup)
    report["backup_features"] = str(npz_backup)

    written: list[str] = []
    try:
        db = LibraryDB(str(staging_path))
        db.connect()
        try:
            columns = {row[1] for row in db.conn.execute("PRAGMA table_info(track_bpm_segments)")}
            if "bpm_end" not in columns:
                db.conn.execute("ALTER TABLE track_bpm_segments ADD COLUMN bpm_end REAL")
                db.conn.commit()
            for uid, (features, rows) in converted.items():
                features["tempo_segments"] = rows.astype(np.float32)
                beats = rebuild_grid_from_segments(features["tempo_segments"])
                duration = float(features.get("duration_sec", 0.0))
                features["beats_time_sec"] = beats[(beats >= 0.0) & (beats <= duration + 1e-6)]
                bpm_rows = build_bpm_segments(features["tempo_segments"])
                key_rows = build_key_segments(features.get("key_segments"))
                phrases = assigned_phrases(extract_phrase_segments(features, duration))
                db.replace_bpm_segments(uid, bpm_rows)
                db.replace_phrase_index(uid, build_phrase_index_rows(phrases, bpm_rows, key_rows))
            ramp_rows = int(db.conn.execute(
                "SELECT COUNT(*) FROM track_bpm_segments WHERE bpm_end IS NOT NULL").fetchone()[0])
            integrity = str(db.conn.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity.lower() != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        finally:
            db.close()
        for uid, (features, _rows) in converted.items():
            store.save(uid, features)
            written.append(uid)
        os.replace(staging_path, db_path)
        write_library_version(library_dir, TARGET_VERSION)
    except Exception:
        for uid in written:
            shutil.copy2(npz_backup / f"{uid}.npz", library_dir / f"{uid}.npz")
        if staging_path.exists():
            staging_path.unlink()
        raise
    report["ramp_rows_in_db"] = ramp_rows
    report["library_version"] = TARGET_VERSION
    return report


if __name__ == "__main__":
    import sys

    print(migrate_library(sys.argv[1] if len(sys.argv) > 1 else "library", dry_run="--dry-run" in sys.argv))
