from __future__ import annotations

from pathlib import Path


SOURCE_VERSION = "0.2.0"
TARGET_VERSION = "0.3.0"


def migrate_library(lib_path: Path, logger=print) -> int:
    return migrate_library_with_progress(lib_path, logger=logger, progress_callback=None)


def migrate_library_with_progress(lib_path: Path, logger=print, progress_callback=None) -> int:
    """0.2.0 -> 0.3.0: introduce phrase information.

    Phrases are stored only inside each track's feature NPZ and default to empty,
    so there is no per-track or database work to perform here. The version stamp
    is advanced by the migration runner.
    """
    if progress_callback is not None:
        progress_callback(0)
    logger("Phrases enabled (default empty); no data conversion required.")
    if progress_callback is not None:
        progress_callback(100)
    return 0
