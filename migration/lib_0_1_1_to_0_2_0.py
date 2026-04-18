from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from core.analysis_lib_handler import FeatureNPZStore
from core.library_handler import LibraryDB
from core.linear_segments import build_bpm_segments, build_key_segments
from utils.jump_cues import build_jump_cues_np, extract_jump_cue_pairs


SOURCE_VERSION = "0.1.1"
TARGET_VERSION = "0.2.0"


def migrate_library(lib_path: Path, logger=print) -> int:
    return migrate_library_with_progress(lib_path, logger=logger, progress_callback=None)


def migrate_library_with_progress(lib_path: Path, logger=print, progress_callback=None) -> int:
    db_path = Path(lib_path) / "library.db"
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    db = LibraryDB(str(db_path))
    db.connect()
    store = FeatureNPZStore(base_dir=str(lib_path), compressed=True)
    try:
        tracks = db.list_all(order_by="added_ts DESC")
        if progress_callback is not None:
            progress_callback(0)
        if not tracks:
            logger("No tracks found for 0.2.0 migration.")
            if progress_callback is not None:
                progress_callback(100)
            return 0

        updated = 0
        failed = 0
        for index, track in enumerate(tracks, start=1):
            uid = str(track.uid or "").strip()
            if not uid:
                failed += 1
                logger(f"[{index}/{len(tracks)}] SKIP {track.title or track.path} -> missing uid")
            else:
                try:
                    features = store.load(uid)
                    bpm_segments = build_bpm_segments(features.get("tempo_segments"))
                    key_segments = build_key_segments(features.get("key_segments"))
                    jump_pairs = extract_jump_cue_pairs(features)
                    features["jump_cues_np"] = build_jump_cues_np(jump_pairs, canonicalize_labels=True)
                    for key in list(features.keys()):
                        if isinstance(key, str) and key.startswith("jump_cues_np."):
                            features.pop(key, None)
                    features.pop("wave_img_np", None)
                    store.save(uid, features)
                    db.replace_bpm_segments(uid, bpm_segments)
                    db.replace_key_segments(uid, key_segments)
                    updated += 1
                    logger(
                        f"[{index}/{len(tracks)}] OK {track.title or track.path} "
                        f"-> bpm={len(bpm_segments)} key={len(key_segments)} jumpcue={len(jump_pairs)}"
                    )
                except Exception as exc:
                    failed += 1
                    logger(f"[{index}/{len(tracks)}] SKIP {track.title or track.path} -> {exc}")
            if progress_callback is not None:
                progress_callback(int(round((index / len(tracks)) * 100)))
        logger(f"Done. updated={updated} failed={failed}")
        if progress_callback is not None:
            progress_callback(100)
        return 0 if failed == 0 else 1
    finally:
        db.close()
