"""The phrases' stored form: a track's ``phrase_segments_np`` block (start, end, label; float32 times), made from the
phrases and read back (what a phrase is and how it is shown: core.marks.phrases)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from analyzer.music.marks import Phrase
from core.marks.phrases import MIN_PHRASE_DUR, fill_unassigned, normalize_phrase_segments


def build_phrase_segments_np(phrases) -> Dict[str, np.ndarray]:
    """Serialize phrases (base labels) into an NPZ-friendly block."""
    rows = normalize_phrase_segments(phrases)
    if not rows:
        return {
            "start": np.asarray([], dtype=np.float32),
            "end": np.asarray([], dtype=np.float32),
            "label": np.asarray([], dtype="U1"),
        }
    starts = np.asarray([r.start for r in rows], dtype=np.float32)
    ends = np.asarray([r.end for r in rows], dtype=np.float32)
    # Validate after quantization too: distinct float64 boundaries can map to
    # the same float32 timestamp at the end of a long track.
    valid = (ends.astype(np.float64) - starts.astype(np.float64)) > MIN_PHRASE_DUR
    rows = [row for row, keep in zip(rows, valid) if bool(keep)]
    starts = starts[valid]
    ends = ends[valid]
    if not rows:
        return {
            "start": np.asarray([], dtype=np.float32),
            "end": np.asarray([], dtype=np.float32),
            "label": np.asarray([], dtype="U1"),
        }
    label_width = max(1, max(len(r.label) for r in rows))
    return {
        "start": starts,
        "end": ends,
        "label": np.asarray([r.label for r in rows], dtype=f"U{label_width}"),
    }


def _extract_phrase_block(features: Dict[str, Any]) -> dict[str, Any]:
    if not isinstance(features, dict):
        return {}
    block = features.get("phrase_segments_np")
    if isinstance(block, dict) and block:
        return block
    prefix = "phrase_segments_np."
    return {
        key[len(prefix):]: value
        for key, value in features.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def stored_phrase_rows(features: Dict[str, Any]) -> List[Phrase]:
    """The phrases of a stored block (its rows as stored; UNASSIGNED stretches are gaps)."""
    block = _extract_phrase_block(features)
    rows: list[Phrase] = []
    try:
        starts = np.asarray(block["start"], dtype=float).ravel()
        ends = np.asarray(block["end"], dtype=float).ravel()
        labels = np.asarray(block["label"]).ravel()
    except Exception:
        starts = ends = labels = np.asarray([])
    for i in range(min(len(starts), len(ends), len(labels))):
        rows.append(Phrase(float(starts[i]), float(ends[i]), str(labels[i]).strip()))
    return rows


def extract_phrase_segments(features: Dict[str, Any], duration_sec: float | None = None) -> List[Phrase]:
    """Read phrases (base labels) from a features dict; returns a sorted list
    covering the whole track (``duration_sec``, else the features' own), the
    stretches without a phrase as UNASSIGNED."""
    rows = stored_phrase_rows(features)
    if not duration_sec and isinstance(features, dict):
        duration_sec = features.get("duration_sec")
    return fill_unassigned(rows, duration_sec)
