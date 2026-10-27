"""Key rows [pitch class, mode, start, end] (mode 0 major / 1 minor), as the editor changes them."""
from __future__ import annotations

import numpy as np


def main_key(key_segments, beats) -> int | None:
    """The track's key as the analysis gives it: the key (pitch class + 12 * minor) of the most beats, the lowest of
    a tie; None without keys or beats. A beat on a boundary is the later row's, the track's end the last row's."""
    rows = np.asarray(key_segments, dtype=float)
    t = np.asarray(beats, dtype=float).ravel()
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 4 or t.size == 0:
        return None
    rows = rows[np.isfinite(rows[:, :4]).all(axis=1)]
    rows = rows[np.argsort(rows[:, 2], kind="stable")]
    if rows.shape[0] == 0:
        return None
    at = np.searchsorted(rows[:, 2], t, side="right") - 1
    inside = (at >= 0) & ((t < rows[np.maximum(at, 0), 3]) | ((at == rows.shape[0] - 1) & (t <= rows[-1, 3])))
    if not inside.any():
        return None
    row = rows[at[inside]]
    values, counts = np.unique(row[:, 0].astype(int) + 12 * (row[:, 1] == 1), return_counts=True)
    return int(values[np.argmax(counts)])


def update_key_segments_with_selection(
    key_segments,
    selection: tuple[float, float, float, float],
    *,
    beat_times=None,
) -> np.ndarray:
    """Return new key segments where the provided selection overwrites its time span."""
    seg_arr = np.asarray(key_segments, dtype=float)
    if seg_arr.ndim != 2 or seg_arr.shape[1] < 4:
        raise ValueError("key_segments must be (N, 4)")
    pitch_class, mode_flag, start, end = selection
    start = float(start)
    end = float(end)
    if end <= start:
        raise ValueError("selection end must be greater than start")
    new_seg = np.array([float(pitch_class), float(mode_flag), start, end], dtype=float)
    result: list[np.ndarray] = []
    inserted = False
    for seg in seg_arr:
        s = float(seg[2])
        e = float(seg[3])
        if e <= start:
            result.append(seg.copy())
            continue
        if s >= end:
            if not inserted:
                result.append(new_seg.copy())
                inserted = True
            result.append(seg.copy())
            continue
        # overlaps
        if s < start:
            left = seg.copy()
            left[3] = start
            if left[3] - left[2] > 1e-9:
                result.append(left)
        if not inserted:
            result.append(new_seg.copy())
            inserted = True
        if e > end:
            right = seg.copy()
            right[2] = end
            if right[3] - right[2] > 1e-9:
                result.append(right)
    if not inserted:
        result.append(new_seg.copy())
    result_arr = np.asarray(result, dtype=float)
    order = np.argsort(result_arr[:, 2])

    # Estimate beat length (if provided) to guide tolerance and filtering.
    beat_len = 0.0
    if beat_times is not None:
        bt = np.asarray(beat_times, dtype=float).ravel()
        if bt.size >= 2:
            diffs = np.diff(np.unique(np.sort(bt)))
            diffs = diffs[diffs > 1e-6]
            if diffs.size:
                beat_len = float(np.median(diffs))

    # Merge adjacent segments that share the same key.
    tol_base = 1e-6
    tolerance = tol_base
    if beat_len > 0.0:
        tolerance = max(tol_base, min(0.25 * beat_len, 0.5))
    merged: list[np.ndarray] = []
    for seg in result_arr[order]:
        if not merged:
            merged.append(seg.copy())
            continue
        prev = merged[-1]
        same_key = prev[0] == seg[0] and prev[1] == seg[1]
        gap = float(seg[2] - prev[3])
        if same_key and gap <= tolerance:
            prev[3] = max(prev[3], seg[3], prev[3])
        else:
            merged.append(seg.copy())

    # Drop segments shorter than one beat, if beat grid is provided.
    if beat_len > 0.0:
        filtered = []
        min_len = max(0.0, beat_len - tolerance)
        for seg in merged:
            dur = float(seg[3] - seg[2])
            if dur < min_len:
                continue
            filtered.append(seg)
        merged = filtered

    return np.asarray(merged, dtype=float)
