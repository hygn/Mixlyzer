"""The loaded track's marks in the form the editor works on, made from its stored features and written back to them
(core.track.model holds them): the tempo rows, the key rows, the phrases covering the track, the JumpCUE groups and
the CUE points. Pure functions: no state, no signals."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from analyzer.music.tempo_rows import with_end_bpm
from analyzer.music.marks import CuePoint, JumpCueGroup, Phrase
from core.analysis_library.cue_points import build_cue_points_np, extract_cue_points
from core.analysis_library.jump_cues import build_jump_cue_block, extract_jump_cue_groups, has_slot_letters, \
    lettered_by_point
from core.analysis_library.phrases import build_phrase_segments_np, extract_phrase_segments, fill_unassigned, \
    normalize_phrase_segments


def beat_times(beats) -> np.ndarray:
    return np.asarray(beats if beats is not None else [], dtype=float).reshape(-1)


def tempo_rows(segments) -> np.ndarray:
    """(N, 6) ``[start, end, bpm, inizio, ts_num, bpm_end]`` (ts_num 4 / steady backfilled), the valid rows (finite,
    end > start) by start."""
    if segments is None:
        return np.empty((0, 6), dtype=float)
    arr = np.asarray(segments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.empty((0, 6), dtype=float)
    arr = with_end_bpm(arr)
    arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 1] > arr[:, 0])]
    return arr[np.argsort(arr[:, 0], kind="stable")]


def key_rows(segments) -> np.ndarray:
    """(N, 4) ``[pitch, minor, start, end]`` by start."""
    if segments is None:
        return np.empty((0, 4), dtype=float)
    arr = np.asarray(segments, dtype=float)
    if arr.ndim == 1 and arr.size % 4 == 0:
        arr = arr.reshape((-1, 4))
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.empty((0, 4), dtype=float)
    arr = arr[:, :4].copy()
    return arr[np.argsort(arr[:, 2], kind="stable")]


def phrase_rows(phrases, duration_sec: Optional[float], *, normalize: bool = True) -> list[Phrase]:
    """The phrases covering the track (its stretches without one UNASSIGNED); ``normalize``: first ordered, slivers
    and overlaps taken out (phrases read or found; an edit's rows are taken as they are)."""
    rows = normalize_phrase_segments(phrases or []) if normalize else list(phrases or [])
    return fill_unassigned(rows, duration_sec if duration_sec and duration_sec > 0.0 else None)


def cue_point_rows(points) -> list[CuePoint]:
    """The CUE points by time."""
    return sorted(points or [], key=lambda p: float(p.time_sec))


def from_features(features: dict, duration_sec: float) -> dict[str, Any]:
    """Every mark of a track's stored features, in the editor's form."""
    return {
        "beats": beat_times(features.get("beats_time_sec")),
        "tempo_segments": tempo_rows(features.get("tempo_segments")),
        "key_segments": key_rows(features.get("key_segments")),
        "phrases": phrase_rows(extract_phrase_segments(features, duration_sec), duration_sec),
        "jump_cues": jump_cue_groups(features),
        "cue_points": cue_point_rows(extract_cue_points(features)),
    }


# each mark's stored form (the features' keys it is written to)
def stored_beatgrid(beats: np.ndarray, segments: np.ndarray) -> dict[str, Any]:
    return {"beats_time_sec": np.asarray(beats, dtype=float), "tempo_segments": np.asarray(segments, dtype=float)}


def stored_key_segments(segments: np.ndarray) -> dict[str, Any]:
    return {"key_segments": np.asarray(segments, dtype=float)}


def stored_phrases(rows: list[Phrase]) -> dict[str, Any]:
    return {"phrase_segments_np": build_phrase_segments_np(rows)}


def jump_cue_groups(features: dict) -> list[JumpCueGroup]:
    """The stored JumpCUE groups; their cues lettered by point where the stored letters are not slots' (each a
    single letter A-Z, no two alike)."""
    groups = extract_jump_cue_groups(features)
    return jump_cue_group_rows(groups if has_slot_letters(groups) else lettered_by_point(groups))


def jump_cue_group_rows(groups) -> list[JumpCueGroup]:
    """The groups as the model keeps them: none empty, each one's cues by point, the groups by their first cue."""
    rows = [replace(group, cues=tuple(sorted(group.cues, key=lambda cue: cue.point)))
            for group in (groups or []) if group.cues]
    return sorted(rows, key=lambda group: group.cues[0].point)


def stored_jump_cues(groups: list[JumpCueGroup]) -> dict[str, Any]:
    """ValueError when a cue's letter is not a slot's (A-Z, no two alike)."""
    return {"jump_cues_np": build_jump_cue_block(groups)}


def stored_cue_points(points: list[CuePoint]) -> dict[str, Any]:
    return {"cue_points_np": build_cue_points_np(points)}
