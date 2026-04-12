from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_VERSION = "0.1.0"
TARGET_VERSION = "0.1.1"


def _windows_subprocess_kwargs() -> dict:
    creationflags = 0
    startupinfo = None
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {
        "creationflags": creationflags,
        "startupinfo": startupinfo,
    }


def _probe_total_samples_ffmpeg(path: str) -> int:
    ffmpeg_exe = ROOT / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        return 0
    args = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-i",
        path,
    ]
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            **_windows_subprocess_kwargs(),
        )
        text = (proc.stderr or b"").decode(errors="ignore")
        if not text:
            return 0

        sample_rate_match = re.search(r"(\d+)\s*Hz", text, flags=re.IGNORECASE)
        if not sample_rate_match:
            return 0
        sample_rate = int(sample_rate_match.group(1))
        if sample_rate <= 0:
            return 0

        duration_match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if not duration_match:
            return 0
        hours = int(duration_match.group(1))
        minutes = int(duration_match.group(2))
        seconds = float(duration_match.group(3))
        duration = (hours * 3600.0) + (minutes * 60.0) + seconds
        if duration <= 0.0:
            return 0
        total = int(round(duration * sample_rate))
        if total > 0:
            return total
    except Exception:
        pass
    return 0


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return any(str(row[1]) == column for row in cur.fetchall())


def _ensure_total_samples_column(conn: sqlite3.Connection) -> None:
    if _column_exists(conn, "tracks", "total_samples"):
        return
    conn.execute("ALTER TABLE tracks ADD COLUMN total_samples INTEGER;")
    conn.commit()


def _iter_targets(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    cur = conn.execute(
        """
        SELECT path, COALESCE(title, '')
        FROM tracks
        WHERE total_samples IS NULL OR total_samples <= 0
        ORDER BY added_ts DESC, path ASC
        """
    )
    return [(str(row[0]), str(row[1])) for row in cur.fetchall()]


def migrate_library(lib_path: Path, logger=print) -> int:
    return migrate_library_with_progress(lib_path, logger=logger, progress_callback=None)


def migrate_library_with_progress(lib_path: Path, logger=print, progress_callback=None) -> int:
    db_path = Path(lib_path) / "library.db"
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_total_samples_column(conn)
        targets = _iter_targets(conn)
        if not targets:
            logger("No tracks need total_samples migration.")
            if progress_callback is not None:
                progress_callback(100)
            return 0

        logger(f"Migrating total_samples for {len(targets)} track(s)")
        updated = 0
        missing = 0
        if progress_callback is not None:
            progress_callback(0)

        for idx, (path, title) in enumerate(targets, start=1):
            total_samples = int(_probe_total_samples_ffmpeg(path) or 0)
            label = title or Path(path).name
            if total_samples > 0:
                conn.execute(
                    "UPDATE tracks SET total_samples=? WHERE path=?;",
                    (total_samples, path),
                )
                updated += 1
                logger(f"[{idx}/{len(targets)}] OK {label} -> {total_samples}")
            else:
                missing += 1
                logger(f"[{idx}/{len(targets)}] SKIP {label} -> could not probe total_samples")
            if progress_callback is not None:
                progress_callback(int(round((idx / len(targets)) * 100)))

        conn.commit()
        logger(f"Done. updated={updated} skipped={missing}")
        if progress_callback is not None:
            progress_callback(100)
        return 0 if missing == 0 else 1
    finally:
        conn.close()
