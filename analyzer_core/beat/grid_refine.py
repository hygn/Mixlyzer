"""Beat-grid refinements applied after the tempo tracker and the beat-phase correction.

All of them work on tempo segments (``[start, end, bpm, inizio, ts, bpm_end]`` rows, see
``core.tempo_rows``) so the stored segments rebuild exactly the stored beats:

* Intro transplant, the simple-ratio check (a section read at 2/3, 3/2, 3/4 or 4/3 of its
  beat takes that tempo when its onsets clearly say so), segment verification (a segment
  whose own grid is not clearly better than a neighbour's is absorbed by it, unless its
  simple-ratio tempo is), boundary refinement and the whole-track half-beat check.
* Phase re-lock: within a constant-BPM grid the local phase can move after a section the
  tracker did not model (a short tempo change, a fill, a phase reset). The grid's offset is
  tracked, per steady segment, in windows of beats over -0.45..0.5 beat with a switching
  penalty, and a prior that keeps the tracker's own phase unless another offset is clearly
  better. Where the offset changes, the segment is split and the new part gets its own
  ``inizio``.
* Tempo ramps (ritardando / accelerando), used by the tracker: only where the segments
  already step the tempo in the same direction by small amounts at least twice. That
  stretch becomes native ramp rows (a continuous piecewise-linear tempo). Everything else
  is left exactly as it was, so steady sections stay static; ramp rows pass through the
  other refinements unchanged.
* Step ramps: a single tempo step between two steady sections is tested as a linear ramp
  (from the grid before to a beat of the grid after, or inside the step followed by the
  grid after) and replaced when the ramp's beats fit the onsets clearly better.
* Slow tail: a slowdown in the last seconds of the track gives way to the main tempo held
  to the end.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from analyzer_core.beat.beat import (
    bpm_at,
    is_ramp_row,
    ramp_beat_times,
    ramp_rows,
    rebuild_grid_from_segments,
    shift_row_beats,
    with_end_bpm,
)

RELOCK_OFFSETS = np.round(np.arange(-0.45, 0.5001, 0.05), 3)
RELOCK_WINDOW_BEATS = 8
RELOCK_HOP_BEATS = 4
RELOCK_SWITCH_PENALTY = 2.0
# Per window off the segment's own phase (each steady segment is tracked on its own).
RELOCK_OFFSET_PRIOR = 0.1
# Offsets near half a beat are where off-beat onsets (8ths at a doubled tempo,
# syncopation) compete with the beat, so they need much stronger evidence.
RELOCK_LARGE_OFFSET = 0.4
RELOCK_LARGE_OFFSET_PRIOR_SCALE = 5.0

# Per own beat, on the refinement onset's probability scale (sparser than the tracker onset).
VERIFY_MIN_GAIN = 0.08

# Simple-ratio misreads (a dotted rhythm read as the beat, 3:2 / 4:3): a segment's tempo times
# one of these, folded into the analysis range, with its best phase. A segment takes the
# alternative when it beats its own grid by RATIO_MIN_GAIN; a segment verification would
# absorb takes it instead when it beats its own grid by RATIO_SWITCH_GAIN and the neighbour's
# by the verification's own margin.
RATIO_ALTERNATIVES = (1.5, 2.0 / 3.0, 4.0 / 3.0, 0.75)
RATIO_MIN_GAIN = 0.08
RATIO_SWITCH_GAIN = 0.02
RATIO_MIN_SEC = 4.0

RAMP_MIN_STEPS = 2
RAMP_STEP_RANGE = (0.003, 0.04)
# The line has two tempos and one phase against one tempo and phase per step: it replaces
# the steps when its beats collect at least this share of the steps' onset activation.
RAMP_SUPPORT_RATIO = 0.95
RAMP_SMOOTH = 4.0

# Whole-track half-beat check: the grid moves half a beat when the onset activation
# halfway between its beats is this much stronger than on them (well clear of what
# steady, correctly phased songs reach).
HALF_BEAT_RATIO = 1.35
HALF_BEAT_EDGE_SEC = 5.0

BOUNDARY_SEARCH_SEC = 8.0
BOUNDARY_MIN_STEP = 0.01
BOUNDARY_MIN_GAP = 0.6

# Step -> ramp: a single tempo step between two steady sections that is really an
# accelerando / ritardando. Per beat, over the span around the step, on the refinement
# onset's scale; the ramp must change the tempo by at least STEP_RAMP_MIN_CHANGE so a
# steady section never becomes one.
STEP_RAMP_MIN_GAIN = 0.05
STEP_RAMP_MIN_CHANGE = 0.03
STEP_RAMP_MAX_SEC = 30.0
STEP_RAMP_INNER_MIN_SEC = 4.0
STEP_RAMP_MEAN_TOL = 0.02

# Slow tail: a slowdown just before the end (rubato, a last ritardando) is of little use on a
# DJ grid; the main tempo is held to the end instead. Rows starting in the last TAIL_SEC,
# all slower than TAIL_RATIO x a main section of at least TAIL_MAIN_SEC before them.
TAIL_SEC = 30.0
TAIL_RATIO = 0.97
TAIL_MAIN_SEC = 20.0

EDGE_MIN_GAIN = 0.1
EDGE_MAX_SEC = 20.0
EDGE_MIN_BEATS = 8


def _activation_reader(activation: np.ndarray, hop_t: float, time_offset: float):
    pooled = maximum_filter1d(np.asarray(activation, dtype=np.float64), 3)
    last = pooled.size - 1

    def read(times: np.ndarray) -> np.ndarray:
        idx = np.round((np.asarray(times, dtype=np.float64) - time_offset) / hop_t - 0.5).astype(np.int64)
        return pooled[np.clip(idx, 0, last)]

    return read


def _track_offsets(scores: np.ndarray, switch_penalty: float, offset_prior: float) -> np.ndarray:
    """Viterbi over windows x offsets (switch cost; every window off zero costs the prior)."""
    n_win, n_off = scores.shape
    zero = int(np.argmin(np.abs(RELOCK_OFFSETS)))
    prior = np.full(n_off, -offset_prior)
    prior[np.abs(RELOCK_OFFSETS) > RELOCK_LARGE_OFFSET + 1e-9] *= RELOCK_LARGE_OFFSET_PRIOR_SCALE
    prior[zero] = 0.0
    score = scores[0] + prior
    back = np.zeros((n_win, n_off), dtype=np.int64)
    for w in range(1, n_win):
        best = int(np.argmax(score))
        switch = score[best] - switch_penalty
        back[w] = np.where(score >= switch, np.arange(n_off), best)
        score = np.maximum(score, switch) + scores[w] + prior
    path = np.empty(n_win, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for w in range(n_win - 1, 0, -1):
        path[w - 1] = back[w, path[w]]
    return RELOCK_OFFSETS[path]


def _segment_rows(segments) -> np.ndarray:
    """Rows ``[start, end, bpm, inizio, ts, bpm_end]`` (a ramp when bpm_end differs from bpm)."""
    return with_end_bpm(segments)


def _grid_beats(t0: float, t1: float, bpm: float, anchor: float) -> np.ndarray:
    period = 60.0 / bpm
    grid = anchor + period * np.arange(np.ceil((t0 - anchor) / period), np.floor((t1 - anchor) / period) + 1)
    return grid[(grid >= t0) & (grid < t1)]


def _grid_support(read, t0: float, t1: float, bpm: float, anchor: float) -> float:
    grid = _grid_beats(t0, t1, bpm, anchor)
    return float(read(grid).mean()) if grid.size else 0.0


def _ratio_alternative(score, row, bpm_range) -> tuple[float, float, float]:
    """(support, bpm, anchor) of the best simple-ratio alternative of a steady row's grid;
    ``score(bpm, anchor)`` is the row's support function."""
    best = (-np.inf, float(row[2]), float(row[3]))
    lo, hi = bpm_range
    for ratio in RATIO_ALTERNATIVES:
        bpm = float(row[2]) * ratio
        while bpm > hi:
            bpm /= 2.0
        while bpm < lo:
            bpm *= 2.0
        period = 60.0 / bpm
        for offset in np.arange(0.0, 1.0, 0.02):
            value = score(bpm, float(row[3]) + offset * period)
            if value > best[0]:
                best = (value, bpm, float(row[3]) + offset * period)
    return best


def ratio_check(
    segments,
    activation: np.ndarray,
    hop_t: float,
    bpm_range: tuple[float, float],
    time_offset: float = 0.0,
    min_gain: float = RATIO_MIN_GAIN,
) -> tuple[np.ndarray, int]:
    """Give a steady segment the 3:2 / 2:3 / 4:3 / 3:4 tempo of its own (folded into the
    analysis range) when that grid explains its onsets clearly better than its own grid
    (verification's score: activation minus the segment's mean, summed per own beat)."""
    rows = _segment_rows(segments).copy()
    read = _activation_reader(activation, hop_t, time_offset)
    pooled = maximum_filter1d(np.asarray(activation, dtype=np.float64), 3)
    switched = 0
    for row in rows:
        t0, t1 = float(row[0]), float(row[1])
        if is_ramp_row(row) or t1 - t0 < RATIO_MIN_SEC:
            continue
        span = pooled[int(max((t0 - time_offset) / hop_t, 0)):int(max((t1 - time_offset) / hop_t, 1))]
        mean = float(span.mean()) if span.size else 0.0
        per = max(_grid_beats(t0, t1, row[2], row[3]).size, 1)

        def score(bpm, anchor):
            return float((read(_grid_beats(t0, t1, bpm, anchor)) - mean).sum()) / per

        own = score(row[2], row[3])
        value, bpm, anchor = _ratio_alternative(score, row, bpm_range)
        if value - own >= min_gain:
            row[2] = row[5] = bpm
            row[3] = anchor
            switched += 1
    return rows, switched


def verify_segments(
    segments,
    activation: np.ndarray,
    hop_t: float,
    time_offset: float = 0.0,
    min_gain: float = VERIFY_MIN_GAIN,
    bpm_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, int]:
    """Merge segments whose own grid is not clearly better than a neighbour's grid there.

    A dynamic tracker can cut a steady song into sections (a quiet intro, a fill, a rubato
    passage) whose BPM is not really different, or read a section's accents as a slower
    pulse (two beats of a 3/4 bar, a 3:2 or 4:3 misread). Repeatedly, the segment whose own
    grid beats the best neighbouring grid by the least (under ``min_gain``) is absorbed by
    that neighbour, so an unclear section keeps the better of the stable tempos around it.

    A grid scores the activation on its beats minus the segment's mean activation, summed
    and divided by the segment's own beat count: a sparser grid that only lands on the
    accents does not win over the grid that also explains the beats between them.

    With ``bpm_range``, a segment about to be absorbed first tries its simple-ratio tempo
    (``_ratio_alternative``): a section read at 2/3 or 4/3 of its beat keeps its place with
    the right tempo instead of taking the neighbour's grid.
    """
    rows = [list(r) for r in _segment_rows(segments)]
    read = _activation_reader(activation, hop_t, time_offset)
    pooled = maximum_filter1d(np.asarray(activation, dtype=np.float64), 3)

    def support(t0, t1, bpm, anchor, mean, per):
        return float((read(_grid_beats(t0, t1, bpm, anchor)) - mean).sum()) / per

    def span_stats(row):
        span = pooled[int(max((row[0] - time_offset) / hop_t, 0)):int(max((row[1] - time_offset) / hop_t, 1))]
        return (float(span.mean()) if span.size else 0.0), max(_grid_beats(row[0], row[1], row[2], row[3]).size, 1)

    merged = 0
    switched: set[float] = set()
    while len(rows) > 1:
        weakest = None
        for k, row in enumerate(rows):
            if is_ramp_row(row):
                continue
            mean, per = span_stats(row)
            own = support(row[0], row[1], row[2], row[3], mean, per)
            for j in (k - 1, k + 1):
                if 0 <= j < len(rows) and not is_ramp_row(rows[j]):
                    gain = own - support(row[0], row[1], rows[j][2], rows[j][3], mean, per)
                    if gain < min_gain and (weakest is None or gain < weakest[0]):
                        weakest = (gain, k, j)
        if weakest is None:
            break
        gain, k, j = weakest
        row = rows[k]
        if bpm_range is not None and row[0] not in switched and row[1] - row[0] >= RATIO_MIN_SEC:
            mean, per = span_stats(row)
            own = support(row[0], row[1], row[2], row[3], mean, per)
            value, bpm, anchor = _ratio_alternative(lambda b, a: support(row[0], row[1], b, a, mean, per), row, bpm_range)
            if value >= own + RATIO_SWITCH_GAIN and value >= own - gain + min_gain:
                rows[k] = [row[0], row[1], bpm, anchor, row[4], bpm]
                switched.add(row[0])
                continue
        lo, hi = min(k, j), max(k, j)
        keep = rows[j]
        rows[lo] = [rows[lo][0], rows[hi][1], keep[2], keep[3], keep[4], keep[5]]
        del rows[hi]
        merged += 1
    return np.asarray(rows, dtype=np.float64), merged


def relock_segments(
    segments,
    activation: np.ndarray,
    hop_t: float,
    duration_sec: float,
    time_offset: float = 0.0,
    switch_penalty: float = RELOCK_SWITCH_PENALTY,
    offset_prior: float = RELOCK_OFFSET_PRIOR,
) -> tuple[np.ndarray, int]:
    """Split segments where the grid's local phase settles at a new offset.

    Each steady segment is tracked on its own: a phase reset that lasts until the next tempo
    change needs one switch, not a switch in and another back out for the next segment's grid.
    Runs of constant offset become their own segments; tempo ramps are kept as they are."""
    rows = _segment_rows(segments)
    read = _activation_reader(activation, hop_t, time_offset)
    half = RELOCK_HOP_BEATS // 2
    out: list[list[float]] = []
    for row in rows:
        beats = np.sort(rebuild_grid_from_segments(row[None, :]))
        beats = beats[(beats >= row[0]) & (beats < row[1]) & (beats >= 0.0) & (beats <= float(duration_sec))]
        if is_ramp_row(row) or beats.size < RELOCK_WINDOW_BEATS + RELOCK_HOP_BEATS:
            out.append(list(row))
            continue
        bpm = float(row[2])
        period = 60.0 / bpm
        starts = np.arange(0, beats.size - RELOCK_WINDOW_BEATS + 1, RELOCK_HOP_BEATS)
        scores = np.empty((starts.size, RELOCK_OFFSETS.size))
        for w, s in enumerate(starts):
            b = beats[s:s + RELOCK_WINDOW_BEATS]
            scores[w] = read(b[:, None] + RELOCK_OFFSETS[None, :] * period).mean(axis=0)
        window_offset = _track_offsets(scores, switch_penalty, offset_prior)
        if not np.any(window_offset != 0.0):
            out.append(list(row))
            continue
        beat_offset = np.empty(beats.size)
        for w, s in enumerate(starts):
            beat_offset[s + half:s + half + RELOCK_HOP_BEATS] = window_offset[w]
        beat_offset[:starts[0] + half] = window_offset[0]
        beat_offset[starts[-1] + half:] = window_offset[-1]
        runs = []
        i = 0
        while i < beats.size:
            j = i
            while j + 1 < beats.size and beat_offset[j + 1] == beat_offset[i]:
                j += 1
            shift = float(beat_offset[i]) * period
            runs.append([beats[i] + shift, beats[j] + shift, bpm, beats[i] + shift, float(row[4]), bpm])
            i = j + 1
        for r_i in range(1, len(runs)):
            boundary = 0.5 * (runs[r_i - 1][1] + runs[r_i][0])
            runs[r_i - 1][1] = boundary
            runs[r_i][0] = boundary
        runs[0][0], runs[-1][1] = float(row[0]), float(row[1])
        out.extend(runs)
    merged = [out[0]]
    for run in out[1:]:
        prev = merged[-1]
        if (not is_ramp_row(run) and not is_ramp_row(prev) and run[2] == prev[2]
                and abs(((run[3] - prev[3]) * prev[2] / 60.0) % 1.0) < 1e-6):
            prev[1] = run[1]
            continue
        merged.append(run)
    final = np.asarray(merged, dtype=np.float64)
    final[0, 0] = min(float(rows[0, 0]), final[0, 0])
    final[-1, 1] = max(float(rows[-1, 1]), float(duration_sec))
    return final, int(final.shape[0] - rows.shape[0])


def _best_anchor(read, t0: float, t1: float, bpm: float, anchor: float) -> tuple[float, float]:
    """The anchor (within half a beat of ``anchor``) whose grid on [t0, t1) has the most support."""
    period = 60.0 / bpm
    best = (-np.inf, anchor)
    for offset in np.arange(-0.5, 0.5, 0.02):
        value = _grid_support(read, t0, t1, bpm, anchor + offset * period)
        if value > best[0]:
            best = (value, anchor + offset * period)
    return best


def _current_support(read, rows: np.ndarray, t0: float, t1: float) -> float:
    beats = rebuild_grid_from_segments(rows)
    beats = beats[(beats >= t0) & (beats < t1)]
    return float(read(beats).mean()) if beats.size else 0.0


def transplant_edges(
    segments,
    finer_segments,
    activation: np.ndarray,
    hop_t: float,
    duration_sec: float,
    time_offset: float = 0.0,
    min_gain: float = EDGE_MIN_GAIN,
    max_edge_sec: float = EDGE_MAX_SEC,
    edges: tuple[str, ...] = ("intro",),
) -> tuple[np.ndarray, int]:
    """Take an intro / outro tempo from a finer tracker candidate when it clearly fits better.

    The candidate chosen for the whole song can miss a short intro or outro in another
    tempo that the shortest-window tracker caught. Such an edge segment (at most
    ``max_edge_sec`` long, at least EDGE_MIN_BEATS beats, its tempo more than 1% off the
    grid there) replaces the grid on its span when its own grid, re-phased on the onset
    activation, beats the current grid there by ``min_gain``.
    """
    rows = _segment_rows(segments)
    fine = _segment_rows(finer_segments)
    if rows.shape[0] == 0 or fine.shape[0] < 2:
        return rows, 0
    read = _activation_reader(activation, hop_t, time_offset)
    duration = float(duration_sec)
    spans: list[tuple[np.ndarray, float, float]] = []
    if "intro" in edges:
        spans.append((fine[0], 0.0, float(fine[0, 1])))
    if "middle" in edges:
        spans.extend((row, float(row[0]), float(row[1])) for row in fine[1:-1])
    if "outro" in edges:
        spans.append((fine[-1], float(fine[-1, 0]), duration))
    added = 0
    for row, t0, t1 in spans:
        bpm = float(row[2])
        if bpm <= 0 or t1 - t0 > max_edge_sec or (t1 - t0) * bpm / 60.0 < EDGE_MIN_BEATS:
            continue
        mid = 0.5 * (t0 + t1)
        here = rows[np.clip(np.searchsorted(rows[:, 0], mid, side="right") - 1, 0, rows.shape[0] - 1)]
        if abs(float(here[2]) - bpm) / bpm < 0.01:
            continue
        own, anchor = _best_anchor(read, t0, t1, bpm, float(row[3]))
        if own - _current_support(read, rows, t0, t1) < min_gain:
            continue
        before = rows[rows[:, 0] < t0].copy()
        after = rows[rows[:, 1] > t1].copy()
        if before.shape[0] == 0 and after.shape[0] == 0:
            continue
        before[:, 1] = np.minimum(before[:, 1], t0)
        after[:, 0] = np.maximum(after[:, 0], t1)
        end = t1 if after.shape[0] else max(duration, float(rows[-1, 1]))
        rows = np.vstack([before, [[t0, end, bpm, anchor, row[4], bpm]], after])
        added += 1
    return rows, added


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


def half_beat_check(
    segments,
    activation: np.ndarray,
    hop_t: float,
    duration_sec: float,
    time_offset: float = 0.0,
    ratio: float = HALF_BEAT_RATIO,
) -> tuple[np.ndarray, bool]:
    """Move the whole grid half a beat when the activation clearly sits between its beats."""
    rows = _segment_rows(segments)
    grid = np.sort(rebuild_grid_from_segments(rows))
    beats = grid[(grid >= 0.0) & (grid <= float(duration_sec))]
    inner = beats[(beats > HALF_BEAT_EDGE_SEC) & (beats < float(duration_sec) - HALF_BEAT_EDGE_SEC)]
    if inner.size >= 16:
        beats = inner
    if beats.size < 16:
        return rows, False
    read = _activation_reader(activation, hop_t, time_offset)
    on = float(read(beats[1:]).mean())
    off = float(read(0.5 * (beats[1:] + beats[:-1])).mean())
    if off <= ratio * max(on, 1e-9):
        return rows, False
    # Each boundary moves to halfway between the moved last beat before it and the
    # moved first beat after it, so every segment keeps its own beats.
    for k in range(1, rows.shape[0]):
        b = rows[k, 0]
        before, after = grid[grid < b], grid[grid >= b]
        if before.size == 0 or after.size == 0 or rows[k - 1, 2] <= 0 or rows[k, 2] <= 0:
            continue
        moved = 0.5 * (before[-1] + 30.0 / bpm_at(rows[k - 1], b) + after[0] + 30.0 / bpm_at(rows[k], b))
        rows[k - 1, 1] = moved
        rows[k, 0] = moved
    for row in rows:
        if row[2] > 0:
            row[3] = shift_row_beats(row, 0.5)
    return rows, True


def _ramp_regions(rows: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [k0, k1] of segments forming a same-direction run of small tempo steps."""
    if rows.shape[0] < RAMP_MIN_STEPS + 1:
        return []
    steps = np.diff(rows[:, 2]) / rows[:-1, 2]
    small = (np.abs(steps) >= RAMP_STEP_RANGE[0]) & (np.abs(steps) <= RAMP_STEP_RANGE[1])
    steady = np.asarray([not is_ramp_row(r) for r in rows])
    small &= steady[:-1] & steady[1:]
    regions: list[tuple[int, int]] = []
    k = 0
    while k < steps.size:
        if not small[k]:
            k += 1
            continue
        j = k
        while j + 1 < steps.size and small[j + 1] and np.sign(steps[j + 1]) == np.sign(steps[k]):
            j += 1
        if j - k + 1 >= RAMP_MIN_STEPS:
            regions.append((k, j + 1))
        k = j + 1
    return regions


def _tempo_curve(rows: np.ndarray, bpm_in: float, bpm_out: float, smooth: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous piecewise-linear tempo over the steps between two steady sections.

    The curve starts at the steady tempo before (``bpm_in``) and ends at the steady tempo
    after (``bpm_out``); its knots at the inner step boundaries are the least-squares fit of
    each step's tempo to the curve's average over that step, with a second-difference
    penalty, so a straight ramp stays one line and a bent one gets more pieces."""
    knots = np.r_[rows[:, 0], rows[-1, 1]]
    m = knots.size
    free = np.arange(1, m - 1)
    n = rows.shape[0]
    A = np.zeros((n, m))
    for k in range(n):
        A[k, k] = A[k, k + 1] = 0.5          # average of a linear piece = mean of its ends
    w = np.sqrt(np.maximum(rows[:, 1] - rows[:, 0], 1e-3))
    D = np.zeros((max(m - 2, 0), m))
    for k in range(m - 2):
        D[k, k], D[k, k + 1], D[k, k + 2] = 1.0, -2.0, 1.0
    fixed = np.zeros(m); fixed[0], fixed[-1] = bpm_in, bpm_out
    lhs = np.vstack([(A * w[:, None])[:, free], np.sqrt(smooth) * D[:, free]])
    rhs = np.r_[(rows[:, 2] - A @ fixed) * w, -np.sqrt(smooth) * (D @ fixed)]
    tempo = fixed.copy()
    if free.size:
        tempo[free], *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return knots, tempo


def _fit_ramp_region(rows: np.ndarray, k0: int, k1: int, read, baseline: np.ndarray):
    """Rows k0 .. k1 with the steps between the steady rows k0 and k1 replaced by a smooth
    tempo curve (native ramp rows, one per linear piece), or None when the curve's beats
    collect less than RAMP_SUPPORT_RATIO of what the ``baseline`` beats collect over the
    same span."""
    if k1 - k0 < 2:
        return None
    inner = rows[k0 + 1:k1]
    t0, t1 = float(inner[0, 0]), float(inner[-1, 1])
    ts = float(rows[k0, 4])
    bpm_in, bpm_out = float(rows[k0, 2]), float(rows[k1, 2])
    # the ramp continues the steady grid before it
    period0 = 60.0 / bpm_in
    anchor0 = float(rows[k0, 3])
    start_beat = anchor0 + period0 * np.floor((t0 - anchor0) / period0)
    knots, tempo = _tempo_curve(inner, bpm_in, bpm_out, RAMP_SMOOTH)
    best = (-np.inf, None)
    for scale in np.arange(0.998, 1.00201, 0.0005):
        for off in np.arange(-0.1, 0.101, 0.02):
            ramp = ramp_rows(knots, tempo * scale, start_beat + period0 + off * period0, ts)
            grid = rebuild_grid_from_segments(ramp)
            grid = grid[(grid >= t0) & (grid < t1)]
            if grid.size:
                value = float(read(grid).mean())
                if value > best[0]:
                    best = (value, ramp, grid)
    base = baseline[(baseline >= t0) & (baseline < t1)]
    if best[1] is None or best[2].size < 4 or (base.size and best[0] < RAMP_SUPPORT_RATIO * float(read(base).mean())):
        return None
    out = [list(rows[k0])]
    out[0][1] = t0
    # native ramp rows: one per linear piece of the tempo curve
    out.extend(best[1].tolist())
    last_row = list(rows[k1])
    last_row[0] = t1
    out.append(last_row)
    return out


def fit_ramp_run(rows, activation: np.ndarray, hop_t: float, time_offset: float, baseline: np.ndarray):
    """Fit the whole run (first and last rows steady) as one ramp; None when rejected."""
    rows = _segment_rows(rows)
    read = _activation_reader(activation, hop_t, time_offset)
    out = _fit_ramp_region(rows, 0, rows.shape[0] - 1, read, np.sort(np.asarray(baseline, dtype=np.float64)))
    return None if out is None else np.asarray(out, dtype=np.float64)


def ramp_segments(
    segments,
    activation: np.ndarray,
    hop_t: float,
    duration_sec: float,
    time_offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Re-fit runs of small same-direction tempo steps as smooth tempo ramps.

    Only the steps between two steady segments are replaced, and only when the ramp's grid
    collects about as much onset activation as the steps did; other segments are unchanged."""
    rows = _segment_rows(segments)
    regions = _ramp_regions(rows)
    if not regions:
        return rows, 0
    read = _activation_reader(activation, hop_t, time_offset)
    beats = np.sort(rebuild_grid_from_segments(rows))
    out: list[list[float]] = []
    cursor = 0
    fitted = 0
    for k0, k1 in regions:
        replaced = _fit_ramp_region(rows, k0, k1, read, beats)
        if replaced is None:
            continue
        out.extend(rows[cursor:k0].tolist())
        out.extend(replaced[:-1])
        cursor = k1
        rows = rows.copy()
        rows[k1] = replaced[-1]
        fitted += 1
    out.extend(rows[cursor:].tolist())
    final = np.asarray(out, dtype=np.float64)
    for k in range(1, final.shape[0]):
        final[k, 0] = final[k - 1, 1]
    final[-1, 1] = max(final[-1, 1], float(duration_sec))
    return final, fitted


def _best_step_ramp(before, step, after, read, mean_of):
    """Best linear ramp for ``step`` between the steady rows around it: (gain, ramp row) or None.

    Two shapes: from a beat of the grid before to a beat of the grid after (an accelerando /
    ritardando between the two tempos; the beat count in between is whole), or inside the step
    continuing the grid before, ending where the step ends (a ritardando / accelerando that is
    then reset; the step's tempo is the ramp's mean). Both are scored with the grids around
    them by the onset contrast per beat over the span, against the current rows."""
    p_in, p_out = 60.0 / before[2], 60.0 / after[2]
    lo, hi = step[0] - 4 * p_in, step[1] + 4 * p_out
    mean = mean_of(lo, hi)

    def score(ramp_t0, ramp_t1, b0, b1):
        beats = np.r_[_grid_beats(lo, ramp_t0, before[2], before[3]),
                      ramp_beat_times(ramp_t0, ramp_t1, b0, b1, ramp_t0),
                      _grid_beats(ramp_t1, hi, after[2], after[3])]
        return float((read(beats) - mean).mean())

    current = np.r_[_grid_beats(lo, step[0], before[2], before[3]),
                    _grid_beats(step[0], step[1], step[2], step[3]),
                    _grid_beats(step[1], hi, after[2], after[3])]
    base = float((read(current) - mean).mean())
    starts = _grid_beats(step[0] - 3 * p_in, step[0] + 3 * p_in, before[2], before[3])
    ends = _grid_beats(step[1] - 3 * p_out, step[1] + 3 * p_out, after[2], after[3])
    t_lo, t_hi = min(before[2], after[2]) * 0.97, max(before[2], after[2]) * 1.03
    best = None
    for ta in starts:
        for tb in ends:
            length = tb - ta
            if length < 2.0:
                continue
            for n in range(int(np.ceil(length * t_lo / 60.0)), int(np.floor(length * t_hi / 60.0)) + 1):
                avg = 60.0 * n / length
                for b0 in before[2] * np.arange(0.98, 1.0301, 0.005):
                    b1 = 2.0 * avg - b0
                    if not (after[2] * 0.97 <= b1 <= after[2] * 1.03) or abs(b1 - b0) / b0 < STEP_RAMP_MIN_CHANGE:
                        continue
                    value = score(ta, tb, b0, b1)
                    if best is None or value > best[0]:
                        best = (value, ta, tb, b0, b1)
    if step[1] - step[0] >= STEP_RAMP_INNER_MIN_SEC:
        for ta in starts[starts < step[1] - STEP_RAMP_INNER_MIN_SEC]:
            for b0 in step[2] * np.arange(0.85, 1.1501, 0.01):
                for b1 in step[2] * np.arange(0.85, 1.1501, 0.01):
                    if abs(b1 - b0) / b0 < STEP_RAMP_MIN_CHANGE:
                        continue
                    if abs(0.5 * (b0 + b1) - step[2]) / step[2] > STEP_RAMP_MEAN_TOL:
                        continue
                    value = score(ta, step[1], b0, b1)
                    if best is None or value > best[0]:
                        best = (value, ta, step[1], b0, b1)
    if best is None:
        return None
    value, ta, tb, b0, b1 = best
    return value - base, np.array([ta, tb, b0, ta, step[4], b1])


def step_ramps(
    segments,
    activation: np.ndarray,
    hop_t: float,
    time_offset: float = 0.0,
    min_gain: float = STEP_RAMP_MIN_GAIN,
) -> tuple[np.ndarray, int]:
    """Replace single tempo steps between steady sections by linear ramps where the onsets
    clearly prefer the ramp (see ``_best_step_ramp``). Only inner steady segments whose tempo
    differs from a neighbour's are tested; the grids around a replaced step keep their phase
    and only give up the time the ramp now covers."""
    rows = _segment_rows(segments)
    read = _activation_reader(activation, hop_t, time_offset)
    pooled = maximum_filter1d(np.asarray(activation, dtype=np.float64), 3)

    def mean_of(t0, t1):
        span = pooled[int(max((t0 - time_offset) / hop_t, 0)):int(max((t1 - time_offset) / hop_t, 1))]
        return float(span.mean()) if span.size else 0.0

    replaced = 0
    k = 1
    while k < rows.shape[0] - 1:
        before, step, after = rows[k - 1], rows[k], rows[k + 1]
        steady = not (is_ramp_row(before) or is_ramp_row(step) or is_ramp_row(after))
        duration = step[1] - step[0]
        differs = (abs(step[2] - before[2]) / before[2] >= STEP_RAMP_MIN_CHANGE
                   or abs(step[2] - after[2]) / after[2] >= STEP_RAMP_MIN_CHANGE)
        found = _best_step_ramp(before, step, after, read, mean_of) if steady and differs and 2.0 <= duration <= STEP_RAMP_MAX_SEC else None
        if found is None or found[0] < min_gain:
            k += 1
            continue
        ramp = found[1]
        new_before = before.copy(); new_before[1] = ramp[0]
        new_after = after.copy(); new_after[0] = ramp[1]
        rows = np.vstack([rows[:k - 1], new_before[None, :], ramp[None, :], new_after[None, :], rows[k + 2:]])
        replaced += 1
        k += 2
    return rows, replaced


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
