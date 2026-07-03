from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from utils.cue_points import CUE_POINT_PREFIX, empty_cue_points_np


SOURCE_VERSION = "0.2.0"
TARGET_VERSION = "0.3.0"


def migrate_library(lib_path: Path, logger=print) -> int:
    return migrate_library_with_progress(lib_path, logger=logger, progress_callback=None)


def migrate_library_with_progress(lib_path: Path, logger=print, progress_callback=None) -> int:
    """0.2.0 -> 0.3.0: introduce phrase and CUEPoint information.

    Phrase data remains optional. CUEPoint arrays are added to every feature NPZ
    and default to empty.
    """
    files = sorted(Path(lib_path).glob("*.npz"))
    empty = empty_cue_points_np()
    converted = 0
    if progress_callback is not None:
        progress_callback(0)
    for index, npz_path in enumerate(files, start=1):
        with np.load(npz_path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        for key, value in empty.items():
            payload.setdefault(f"{CUE_POINT_PREFIX}{key}", value)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{npz_path.stem}.migration_0_3_0_",
            suffix=".npz",
            dir=npz_path.parent,
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            np.savez_compressed(temp_path, **payload)
            os.replace(temp_path, npz_path)
        finally:
            temp_path.unlink(missing_ok=True)
        converted += 1
        if progress_callback is not None and files:
            progress_callback(int(round(index * 100 / len(files))))
    logger(f"Phrases enabled; canonical CUEPoint arrays added to {converted} track(s).")
    if progress_callback is not None:
        progress_callback(100)
    return converted
