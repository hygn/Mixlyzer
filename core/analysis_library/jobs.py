"""Every operation on the analysis library ({libpath}/{uid}.npz: each track's stored analysis), by name. As the
library's jobs (core.library.jobs), a job is ``job(lib, *args)`` run on the library's owner's thread
(core.library.service: one owner for every file of the library), and the app asks the owner for it:

    library.read(analysis_jobs.features, uid, done=show)        # a track's stored features
    library.write(analysis_jobs.save_edits, uid, edits)          # the editor's save

Below it, core.analysis_library.store is how the files are kept, which nothing outside the library touches. The
jobs that store a whole analysis or remove tracks write library.db and these files together: they are the library's
(core.library.jobs.store_analysis, remove_tracks).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from core.linear_segments import build_bpm_segments, build_key_segments, build_phrase_index_rows
from utils.phrases import assigned_phrases, extract_phrase_segments


def features(lib, uid: str) -> Optional[dict]:
    """A track's stored features (its NPZ, flattened keys), None when there are none to read."""
    try:
        return lib.features.load(uid) if uid else None
    except (OSError, ValueError):
        return None


def features_stamp(lib, uid: str) -> Optional[tuple[int, int]]:
    """When and how large a track's stored features were written (the Rekordbox XML's change check)."""
    path = lib.features.path(uid)
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size


# ---- the editor's save ----

def _replace_block(stored: dict, name: str, block: dict) -> None:
    """Replace a stored block (flattened as ``name.<field>``) as a whole, so no field of the previous block is left
    behind."""
    for key in [k for k in stored if k.startswith(name + ".")]:
        del stored[key]
    stored[name] = block


EDIT_BLOCKS = ("beatgrid", "segments", "key_segments", "key_segments_auto", "jump_cues_np", "jump_cues_auto_np",
               "phrase_segments_np", "cue_points_np")


def save_edits(lib, uid: str, edits: dict) -> str:
    """The editor's save: the blocks in ``edits`` (EDIT_BLOCKS; one left out or None: as stored) replaced in the
    track's stored features, its segment tables in library.db written from them. Returns the uid. Unmarked (the
    snapshot read whole, sig_lib_updated sent): the library's phrase / transition filters read those tables and run
    again on it."""
    unknown = set(edits) - set(EDIT_BLOCKS)
    if unknown:
        raise ValueError(f"not an editor block: {sorted(unknown)}")
    beatgrid, segments, key_segments, key_segments_auto, jump_cues_np, jump_cues_auto_np, phrase_segments_np, \
        cue_points_np = (edits.get(name) for name in EDIT_BLOCKS)
    stored = lib.features.load(uid)
    if beatgrid is not None:
        stored["beats_time_sec"] = beatgrid
    if segments is not None:
        stored["tempo_segments"] = segments
    if key_segments is not None:
        stored["key_segments"] = key_segments
    if key_segments_auto is not None:
        # the automatic key result (full analysis / whole-track reanalysis); the key optimizer learns from the tracks
        # whose keys differ from it
        stored["key_segments_auto"] = key_segments_auto
    if jump_cues_np is not None:
        _replace_block(stored, "jump_cues_np", jump_cues_np)
    if jump_cues_auto_np is not None:
        # a JumpCUE reanalysis replaces the stored automatic result
        _replace_block(stored, "jump_cues_auto_np", jump_cues_auto_np)
    if phrase_segments_np is not None:
        _replace_block(stored, "phrase_segments_np", phrase_segments_np)
    if cue_points_np is not None:
        _replace_block(stored, "cue_points_np", cue_points_np)
    lib.features.save(uid, stored)
    bpm_rows = build_bpm_segments(stored.get("tempo_segments"))
    key_rows = build_key_segments(stored.get("key_segments"))
    phrase_rows = assigned_phrases(extract_phrase_segments(stored))
    lib.db.replace_bpm_segments(uid, bpm_rows)
    lib.db.replace_key_segments(uid, key_rows)
    lib.db.replace_phrase_index(uid, build_phrase_index_rows(phrase_rows, bpm_rows, key_rows))
    return uid


# ---- Reoptimize (the models trained on the library's edits) ----

# what the optimizers read of a track's stored features: the grids, keys, phrases and JumpCUEs the user annotated
# (a few kB a track; the waveform overview and the rest are left out)
TRAINING_KEYS = ("beats_time_sec", "sr", "duration_sec", "tempo_segments", "key_segments", "key_segments_auto")
TRAINING_PREFIXES = ("phrase_segments_np.", "jump_cues_np.", "jump_cues_auto_np.")
# tracks read a job (training_features): 3 - 14 ms a track measured, so the owner's other jobs (the GUI's) wait
# at most about a second between two
TRAINING_BATCH = 50


def training_tracks(lib) -> list[dict]:
    """The library's tracks as the optimizers read them, in uid order: uid, title, artist, audio_path; their stored
    features are read in batches after (training_features). Handed to their process (optimizer.library_tracks),
    which reads the audio from its path and never opens the library's files."""
    return [{"uid": row.uid, "title": row.title or "", "artist": row.artist or "",
             "audio_path": Path(row.path) if row.path else Path()}
            for row in lib.db.list_all(order_by="uid ASC") if row.uid]


def training_features(lib, uids: list[str]) -> list[tuple[Optional[dict], str]]:
    """Each track's TRAINING_KEYS / TRAINING_PREFIXES of its stored features (None when it has none) and why they
    could not be read ("" when they were)."""
    out = []
    for uid in uids:
        try:
            out.append((lib.features.load_keys(uid, TRAINING_KEYS, TRAINING_PREFIXES), ""))
        except (OSError, ValueError) as exc:
            out.append((None, str(exc)))
    return out
