"""Every operation on the analysis library ({libpath}/{uid}.npz: each track's stored analysis), by name. As the
library's jobs (core.library.jobs), a job is ``job(lib, *args)`` run on the library's owner's thread
(core.library.service: one owner for every file of the library), and the app asks the owner for it:

    library.read(analysis_jobs.analysis, uid, done=show)        # a track's stored analysis (StoredAnalysis)
    library.write(analysis_jobs.save_edits, uid, edits)          # the editor's save (TrackEdits)

Below it, core.analysis_library.store is how the files are kept, which nothing outside the library touches, and
core.analysis_library.stored what their arrays hold. The jobs that store a whole analysis or remove tracks write
library.db and these files together: they are the library's (core.library.jobs.store_analysis, remove_tracks).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from analyzer.training.library_tracks import Annotation
from core.library.segments import build_bpm_segments, build_key_segments, build_phrase_index_rows
from core.analysis_library.phrases import assigned_phrases, extract_phrase_segments
from core.analysis_library.stored import ANNOTATION_BLOCKS, ANNOTATION_KEYS, StoredAnalysis, TrackEdits,     annotation_of, apply_edits


def analysis(lib, uid: str) -> Optional[StoredAnalysis]:
    """A track's stored analysis, None when there is none to read."""
    try:
        return StoredAnalysis.from_arrays(lib.features.load(uid)) if uid else None
    except (OSError, ValueError):
        return None


def features_stamp(lib, uid: str) -> Optional[tuple[int, int]]:
    """When and how large a track's stored analysis was written (the Rekordbox XML's change check)."""
    path = lib.features.path(uid)
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return st.st_mtime_ns, st.st_size


# ---- the editor's save ----

def save_edits(lib, uid: str, edits: TrackEdits) -> str:
    """The editor's save: ``edits`` into the track's stored analysis (each mark's block replaced whole), its segment
    tables in library.db written from them. Returns the uid. Unmarked (the snapshot read whole, sig_lib_updated
    sent): the library's phrase / transition filters read those tables and run again on it."""
    stored = lib.features.load(uid)
    apply_edits(stored, edits)
    lib.features.save(uid, stored)
    bpm_rows = build_bpm_segments(stored.get("tempo_segments"))
    key_rows = build_key_segments(stored.get("key_segments"))
    phrase_rows = assigned_phrases(extract_phrase_segments(stored))
    lib.db.replace_bpm_segments(uid, bpm_rows)
    lib.db.replace_key_segments(uid, key_rows)
    lib.db.replace_phrase_index(uid, build_phrase_index_rows(phrase_rows, bpm_rows, key_rows))
    return uid


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
