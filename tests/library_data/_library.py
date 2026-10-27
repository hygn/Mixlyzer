"""The library the checks read, read only, and how a failing track is named in a report (never by its title, path or,
unless asked, its uid)."""
from __future__ import annotations

import json
import os
import sqlite3
import unittest
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENABLED = os.environ.get("MIXLYZER_LIBRARY_TESTS") == "1"
SHOW_UIDS = os.environ.get("MIXLYZER_TEST_SHOW_UIDS") == "1"
REPORT_MAX = 12


def library_dir() -> Path | None:
    env = os.environ.get("MIXLYZER_LIBRARY")
    if env:
        path = Path(env)
    else:
        path = ROOT / "library"
        try:
            text = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["libconfig"]["libpath"]
            path = Path(text) if Path(text).is_absolute() else ROOT / text
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return path if (path / "library.db").is_file() else None


def skip_reason() -> str | None:
    if not ENABLED:
        return "the library's checks are off (MIXLYZER_LIBRARY_TESTS=1)"
    if library_dir() is None:
        return "no library found (MIXLYZER_LIBRARY or config.json's libconfig.libpath)"
    return None


def needs_library(cls):
    reason = skip_reason()
    return unittest.skip(reason)(cls) if reason else cls


def connect() -> sqlite3.Connection:
    """library.db read only (a running app may hold it: reading is safe)."""
    path = library_dir() / "library.db"
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


@lru_cache(maxsize=1)
def uids() -> tuple[str, ...]:
    with connect() as db:
        return tuple(sorted(r[0] for r in db.execute("SELECT uid FROM tracks")))


def name(uid: str) -> str:
    """How a report names a track: its place among the uids (the uid itself only when asked for)."""
    try:
        k = uids().index(uid)
    except ValueError:
        k = -1
    return f"track #{k} ({uid})" if SHOW_UIDS else f"track #{k}"


@lru_cache(maxsize=1)
def stored() -> dict[str, dict]:
    """Every track's stored arrays (the app's own reader: no pickle); a file that does not read is left out
    (test_integrity reports it)."""
    from core.analysis_library.store import FeatureNPZStore
    store = FeatureNPZStore(str(library_dir()))
    out = {}
    for uid in uids():
        try:
            out[uid] = store.load(uid)
        except Exception:
            pass
    return out


def report(test: unittest.TestCase, problems: list[tuple[str, str]], what: str) -> None:
    """Fails with the problems found, a few of them named (anonymously) with their reason."""
    if not problems:
        return
    lines = [f"  {name(uid)}: {why}" for uid, why in problems[:REPORT_MAX]]
    more = len(problems) - REPORT_MAX
    test.fail(f"{len(problems)} of {len(uids())} tracks: {what}\n" + "\n".join(lines)
              + (f"\n  ... and {more} more" if more > 0 else ""))
