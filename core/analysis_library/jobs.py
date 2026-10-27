"""Every operation on the analysis library ({libpath}/{uid}.npz: each track's stored analysis), by name. As the
library's jobs (core.library.jobs), a job is ``job(lib, *args)`` run on the library's owner's thread
(core.library.service: one owner for every file of the library), and the app asks the owner for it:

    library.read(analysis_jobs.analysis, uid, done=show)        # a track's stored analysis (StoredAnalysis)

Below it, core.analysis_library.store is how the files are kept, which nothing outside the library touches, and
core.analysis_library.stored what their arrays hold. The jobs that write library.db and these files together are
the library's (core.library.jobs: store_analysis, save_edits, remove_tracks): this package imports nothing of
core.library (tests/test_library_boundary.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from analyzer.training.library_tracks import Annotation
from core.analysis_library.stored import ANNOTATION_BLOCKS, ANNOTATION_KEYS, FINGERPRINT_KEYS, INDEX_BLOCKS, \
    INDEX_KEYS, MIX_KEYS, StoredAnalysis, annotation_of, index_marks


def analysis(lib, uid: str) -> Optional[StoredAnalysis]:
    """A track's stored analysis, None when there is none to read."""
    try:
        return StoredAnalysis.from_arrays(lib.features.load(uid)) if uid else None
    except (OSError, ValueError):
        return None


def fingerprint_source(lib, uid: str) -> StoredAnalysis:
    """What the Variants' fingerprint reads of a track's stored analysis: its beats, length and overview (the rest
    is not decompressed). Raises what the read raised; FileNotFoundError when the track has none stored."""
    arrays = lib.features.load_keys(uid, FINGERPRINT_KEYS)
    if arrays is None:
        raise FileNotFoundError(lib.features.path(uid))
    return StoredAnalysis.from_arrays(arrays)


def mix_source(lib, uid: str) -> StoredAnalysis:
    """What the mix candidates' bars read of a track's stored analysis: its beats, tempo rows, length and overview.
    Raises what the read raised; FileNotFoundError when the track has none stored."""
    arrays = lib.features.load_keys(uid, MIX_KEYS)
    if arrays is None:
        raise FileNotFoundError(lib.features.path(uid))
    return StoredAnalysis.from_arrays(arrays)


def stored_marks(lib, uid: str):
    """The marks library.db's rows of a track are made from (stored.index_marks: the tempo and key rows, the
    phrases; the rest is not decompressed). None when the track has none stored; raises what the read raised."""
    arrays = lib.features.load_keys(uid, INDEX_KEYS, INDEX_BLOCKS)
    return index_marks(arrays) if arrays is not None else None


def features_stamp(lib, uid: str) -> Optional[tuple[int, int]]:
    """When and how large a track's stored analysis was written (the Rekordbox XML's change check)."""
    path = lib.features.path(uid)
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size


# ---- Reoptimize (the models trained on the library's edits) ----

# tracks read a job (training_annotations): 3 - 14 ms a track measured, so the owner's other jobs (the GUI's) wait
# at most about a second between two
TRAINING_BATCH = 50


def training_tracks(lib) -> list[dict]:
    """The library's tracks as the optimizers read them, in uid order: uid, title, artist, audio_path; their stored
    annotations are read in batches after (training_annotations). Handed to their process (analyzer.training.library_tracks),
    which reads the audio from its path and never opens the library's files."""
    return [{"uid": row.uid, "title": row.title or "", "artist": row.artist or "",
             "audio_path": Path(row.path) if row.path else Path()}
            for row in lib.db.list_all(order_by="uid ASC") if row.uid]


def training_annotations(lib, uids: list[str]) -> list[tuple[Optional[Annotation], str]]:
    """Each track's annotation (core.analysis_library.stored.annotation_of: only what the optimizers read is
    decompressed; None when it has no stored analysis) and why it could not be read ("" when it was)."""
    out = []
    for uid in uids:
        try:
            arrays = lib.features.load_keys(uid, ANNOTATION_KEYS, ANNOTATION_BLOCKS)
            out.append((annotation_of(arrays) if arrays is not None else None, ""))
        except (OSError, ValueError) as exc:
            out.append((None, str(exc)))
    return out
