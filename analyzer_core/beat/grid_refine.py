"""Beat-grid refinements the beat tracker applies after its decoding, on tempo segments
(``[start, end, bpm, inizio, ts, bpm_end]`` rows, see ``foundation.tempo_rows``):

* Boundary refinement: each tempo change moves to where the two grids fit the tempo-change
  onset activation best (``learned_onset.refine_onset_activation``).
* Slow tail: a slowdown in the last seconds of the track gives way to the main tempo held
  to the end.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from foundation.tempo_rows import is_ramp_row, with_end_bpm

BOUNDARY_SEARCH_SEC = 8.0
BOUNDARY_MIN_STEP = 0.01
BOUNDARY_MIN_GAP = 0.6

# Slow tail: a slowdown just before the end (rubato, a last ritardando) is of little use on a
# DJ grid; the main tempo is held to the end instead. Rows starting in the last TAIL_SEC,
# all slower than TAIL_RATIO x a main section of at least TAIL_MAIN_SEC before them.
TAIL_SEC = 30.0
TAIL_RATIO = 0.97
TAIL_MAIN_SEC = 20.0


def _activation_reader(activation: np.ndarray, hop_t: float, time_offset: float):
    pooled = maximum_filter1d(np.asarray(activation, dtype=np.float64), 3)
    last = pooled.size - 1

    def read(times: np.ndarray) -> np.ndarray:
        idx = np.round((np.asarray(times, dtype=np.float64) - time_offset) / hop_t - 0.5).astype(np.int64)
        return pooled[np.clip(idx, 0, last)]

    return read


def _segment_rows(segments) -> np.ndarray:
    """Rows ``[start, end, bpm, inizio, ts, bpm_end]`` (a ramp when bpm_end differs from bpm)."""
    return with_end_bpm(segments)


def refine_boundaries(
    segments,
    activation: np.ndarray,
    hop_t: float,
    duration_sec: float,
    time_offset: float = 0.0,
    search_sec: float = BOUNDARY_SEARCH_SEC,
    min_step: float = BOUNDARY_MIN_STEP,
) -> tuple[np.ndarray, int]:
    """Move each tempo change to where the two grids fit the onset activation best.

    Each grid keeps its own phase; only the hand-over beat moves (within ``search_sec``).
    A candidate is scored by the activation on its beats minus the local mean activation
    per beat, so a denser grid gains nothing from beats that land on nothing.
    """
    rows = _segment_rows(segments)
    if rows.shape[0] < 2:
        return rows, 0
    pooled = maximum_filter1d(np.asarray(activation, dtype=np.float64), 3)
    read = _activation_reader(activation, hop_t, time_offset)
    moved = 0
    for k in range(1, rows.shape[0]):
        left, right = rows[k - 1], rows[k]
        if is_ramp_row(left) or is_ramp_row(right):
            continue
        if left[2] <= 0 or right[2] <= 0 or abs(right[2] - left[2]) / left[2] < min_step:
            continue
        pl, pr = 60.0 / left[2], 60.0 / right[2]
        b = float(right[0])
        lo = max(b - search_sec, float(left[0]) + 2.0 * pl)
        hi = min(b + search_sec, float(right[1]) - 2.0 * pr)
        if hi <= lo:
            continue
        span_lo = max(float(left[0]), lo - pl)
        span_hi = min(float(right[1]), hi + pr)
        frames = pooled[int(max((span_lo - time_offset) / hop_t, 0)):int(max((span_hi - time_offset) / hop_t, 1))]
        if frames.size == 0:
            continue
        mean = float(frames.mean())
        left_grid = left[3] + pl * np.arange(np.ceil((span_lo - left[3]) / pl), np.floor((span_hi - left[3]) / pl) + 1)
        right_grid = right[3] + pr * np.arange(np.ceil((span_lo - right[3]) / pr), np.floor((span_hi - right[3]) / pr) + 1)
        best = None
        for first in right_grid[(right_grid >= lo) & (right_grid <= hi)]:
            kept = left_grid[left_grid < first - 1e-6]
            if kept.size and first - kept[-1] < BOUNDARY_MIN_GAP * min(pl, pr):
                # The left grid's last beat would crowd the first right beat: drop it.
                limit = float(kept[-1])
                kept = kept[:-1]
            else:
                limit = float(first)
            after = right_grid[right_grid >= first - 1e-6]
            score = float(read(kept).sum() + read(after).sum()) - mean * (kept.size + after.size)
            floor = max(float(kept[-1]) if kept.size else span_lo, float(first) - pr)
            if best is None or score > best[0] + 1e-9:
                best = (score, 0.5 * (floor + limit))
        if best is None or abs(best[1] - b) < 1e-6:
            continue
        # Keep the current boundary unless the move gains something.
        kept = left_grid[left_grid < b - 1e-3]
        after = right_grid[right_grid >= b - 1e-3]
        current = float(read(kept).sum() + read(after).sum()) - mean * (kept.size + after.size)
        if best[0] <= current + 1e-9:
            continue
        rows[k - 1, 1] = best[1]
        rows[k, 0] = best[1]
        moved += 1
    return rows, moved


def hold_slow_tail(segments, duration_sec: float) -> tuple[np.ndarray, bool]:
    """Hold the main tempo to the end over a slow tail: the rows after the last main section
    all start in the last ``TAIL_SEC`` and are all slower than ``TAIL_RATIO`` x its tempo. The
    main section (same-tempo rows, a split by a phase change included) lasts at least
    ``TAIL_MAIN_SEC``; its grid then runs to the end."""
    rows = _segment_rows(segments)
    end = max(float(duration_sec), float(rows[-1, 1]))
    for m in range(rows.shape[0] - 2, -1, -1):
        tail = rows[m + 1:]
        if is_ramp_row(rows[m]):
            return rows, False
        if not (np.all(tail[:, 0] >= end - TAIL_SEC)
                and np.all(np.maximum(tail[:, 2], tail[:, 5]) < TAIL_RATIO * rows[m, 2])):
            continue
        j = m
        while j > 0 and not is_ramp_row(rows[j - 1]) and abs(rows[j - 1, 2] - rows[m, 2]) / rows[m, 2] < 0.005:
            j -= 1
        if rows[m, 1] - rows[j, 0] < TAIL_MAIN_SEC:
            return rows, False
        held = rows[:m + 1].copy()
        held[-1, 1] = end
        return held, True
    return rows, False
