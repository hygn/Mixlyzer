"""Tempo-segment rows ``[start, end, bpm, inizio, ts, bpm_end]`` and the beats they give.

``inizio`` is a downbeat (and so a beat) of the row's grid, ``ts`` the beats per bar. The
optional sixth column is the tempo at the row's end: the tempo goes linearly in time from
``bpm`` at ``start`` to it (a tempo ramp). Rows without it, or with it equal to ``bpm``, are
steady. NumPy only, so the views, editor and exporters can use it too.
"""

from __future__ import annotations

import numpy as np


# Optional sixth column of a tempo-segment row: the tempo at the row's end. The tempo goes
# linearly in time from ``bpm`` (at ``start``) to it; absent or equal to ``bpm``: steady.
END_BPM_COLUMN = 5


def segment_end_bpm(row) -> float:
    """Tempo at the end of a ``[start, end, bpm, inizio, ts, bpm_end]`` row (steady: ``bpm``)."""
    if len(row) > END_BPM_COLUMN and np.isfinite(row[END_BPM_COLUMN]) and row[END_BPM_COLUMN] > 0:
        return float(row[END_BPM_COLUMN])
    return float(row[2])


def is_ramp_row(row, tolerance: float = 1e-6) -> bool:
    return abs(segment_end_bpm(row) - float(row[2])) > tolerance


def with_end_bpm(segments) -> np.ndarray:
    """Tempo-segment rows as ``[start, end, bpm, inizio, ts, bpm_end]`` (steady rows: bpm_end = bpm)."""
    arr = np.asarray(segments, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float64)
    out = np.empty((arr.shape[0], 6), dtype=np.float64)
    out[:, :4] = arr[:, :4]
    out[:, 4] = arr[:, 4] if arr.shape[1] > 4 else 4.0
    if arr.shape[1] > END_BPM_COLUMN:
        end_bpm = arr[:, END_BPM_COLUMN]
        out[:, 5] = np.where(np.isfinite(end_bpm) & (end_bpm > 0), end_bpm, arr[:, 2])
    else:
        out[:, 5] = arr[:, 2]
    return out


def bpm_at(row, t: float) -> float:
    """Tempo of a row at time ``t`` (linear for a ramp)."""
    start, end, bpm = float(row[0]), float(row[1]), float(row[2])
    if not is_ramp_row(row) or end <= start:
        return bpm
    return bpm + (segment_end_bpm(row) - bpm) * (float(t) - start) / (end - start)


def split_row(row, t: float):
    """Two rows covering ``row`` cut at ``t``; a ramp keeps its tempo line (both parts keep
    the same beat anchor, so the beats do not move)."""
    left = list(np.asarray(row, dtype=np.float64))
    right = list(left)
    if len(left) <= END_BPM_COLUMN:
        return left[:1] + [float(t)] + left[2:], [float(t)] + right[1:]
    mid = bpm_at(row, t)
    left[1] = float(t); left[END_BPM_COLUMN] = mid
    right[0] = float(t); right[2] = mid
    return left, right


def shift_row_beats(row, beats: float) -> float:
    """``inizio`` of ``row`` moved ``beats`` beats later along its own grid."""
    start, end, bpm, inizio = (float(x) for x in row[:4])
    if not is_ramp_row(row):
        return inizio + float(beats) * 60.0 / bpm
    b1 = segment_end_bpm(row)
    slope = (b1 - bpm) / (end - start)
    phase = lambda x: (bpm * x + 0.5 * slope * x * x) / 60.0
    target = phase(inizio - start) + float(beats)
    disc = bpm * bpm + 2.0 * slope * 60.0 * target
    return start + (-bpm + np.sqrt(max(disc, 0.0))) / slope


def beats_elapsed(row, t: float) -> float:
    """Beats of the row's tempo line from its start to ``t`` (fractional)."""
    start, end, bpm = float(row[0]), float(row[1]), float(row[2])
    x = float(t) - start
    if not is_ramp_row(row) or end <= start:
        return bpm * x / 60.0
    slope = (segment_end_bpm(row) - bpm) / (end - start)
    return (bpm * x + 0.5 * slope * x * x) / 60.0


def retempo_row(row, bpm_start: float, bpm_end: float, keep: float) -> np.ndarray:
    """``row`` with the tempo line ``bpm_start`` -> ``bpm_end`` (equal: steady). The beat at ``keep``
    stays where it is and the downbeat stays the same number of beats from it."""
    out = with_end_bpm(np.asarray(row, dtype=np.float64)[None, :])[0]
    k = round(beats_elapsed(out, out[3]) - beats_elapsed(out, keep))
    out[2], out[END_BPM_COLUMN] = float(bpm_start), float(bpm_end)
    out[3] = float(keep)
    out[3] = shift_row_beats(out, k)
    return out


def ramp_beat_times(start: float, end: float, bpm_start: float, bpm_end: float,
                    anchor: float, limit: float | None = None) -> np.ndarray:
    """Beats of a tempo ramp: the tempo goes linearly in time from ``bpm_start`` at ``start``
    to ``bpm_end`` at ``end`` and ``anchor`` is one of its beats (it may lie outside the
    row; the tempo line is extended there). Returns the beats in ``[start, limit)``."""
    limit = float(end) if limit is None else float(limit)
    length = float(end) - float(start)
    slope = (float(bpm_end) - float(bpm_start)) / length if length > 0 else 0.0   # BPM per second

    def phase(x):            # beats elapsed from ``start`` to ``start + x``
        return (bpm_start * x + 0.5 * slope * x * x) / 60.0

    c = phase(float(anchor) - float(start))
    n_lo = int(np.ceil(phase(0.0) - c - 1e-9))
    n_hi = int(np.floor(phase(limit - float(start)) - c - 1e-9))
    if n_hi < n_lo:
        return np.empty(0, dtype=float)
    target = c + np.arange(n_lo, n_hi + 1, dtype=np.float64)
    if abs(slope) < 1e-12:
        x = 60.0 * target / bpm_start
    else:
        disc = bpm_start * bpm_start + 2.0 * slope * 60.0 * target
        x = (-bpm_start + np.sqrt(np.maximum(disc, 0.0))) / slope
    beats = float(start) + x
    return beats[(beats >= float(start) - 1e-9) & (beats < limit)]


def rebuild_grid_from_segments(segments) -> np.ndarray:
    """Rebuild beat times from ``[start, end, bpm, inizio, ...]`` rows.

    ``inizio`` is a downbeat reference, and therefore also a valid beat-phase
    reference.  Keeping this helper beside the tempo-segment cleanup makes it
    possible to update the grid whenever that cleanup changes the segments.
    A row with a different end tempo (sixth column) is a linear tempo ramp.
    """
    try:
        arr = np.asarray(segments, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.asarray([], dtype=float)

    beats: list[float] = []
    valid_ends = arr[:, 1][np.isfinite(arr[:, 1])]
    track_end = float(np.max(valid_ends)) if valid_ends.size else float("inf")
    for row in arr:
        start, end, bpm, inizio = row[:4]
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            continue
        if not (np.isfinite(bpm) and bpm > 0):
            continue
        if is_ramp_row(row):
            end_limit = float(end) + (1e-3 if float(end) < track_end else 1e-6)
            anchor = float(inizio) if np.isfinite(inizio) else float(start)
            beats.extend(ramp_beat_times(float(start), float(end), float(bpm), segment_end_bpm(row),
                                         anchor, end_limit).tolist())
            continue
        period = 60.0 / float(bpm)
        if not np.isfinite(period) or period <= 0:
            continue
        phase = float(inizio) if np.isfinite(inizio) else float(start)
        beat = phase
        if beat < start:
            beat += np.ceil((start - beat) / period - 1e-9) * period
        else:
            beat -= np.floor((beat - start) / period + 1e-9) * period

        # Keep a beat that lands on a shared segment boundary despite float error
        # (its duplicate from the next segment is pruned below). Only a hair past
        # the end: a wider overshoot adds a phantom beat whenever the next
        # segment's first beat comes later than that (e.g. after nudging it).
        # The final segment must not produce inaudible beats beyond the track end.
        end_limit = float(end) + (1e-3 if float(end) < track_end else 1e-6)
        while beat < end_limit:
            beats.append(float(beat))
            beat += period

    return prune_close_beats(np.asarray(beats, dtype=float))


MIN_BEAT_GAP_SEC = 0.05


def prune_close_beats(beats, min_gap: float = MIN_BEAT_GAP_SEC) -> np.ndarray:
    """Sort beats and drop any beat closer than ``min_gap`` to the one kept before it
    (two segments' beats meeting at a shared boundary)."""
    beat_arr = np.sort(np.asarray(beats, dtype=float).ravel())
    if beat_arr.size <= 1:
        return beat_arr
    pruned: list[float] = []
    for beat in beat_arr:
        if pruned and (beat - pruned[-1]) < min_gap:
            continue
        pruned.append(float(beat))
    return np.asarray(pruned, dtype=float)


# ---- the editor's grid moves ----

def build_grid_from_period_phase(total_len_frames: int,
                                 hop_t: float,
                                 period_frames: float,
                                 phase_frames: float,
                                 subdiv: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Make beatgrid in sec using period and phase."""
    T = total_len_frames
    step = period_frames / max(1, int(subdiv))
    if not np.isfinite(step) or step <= 0:
        return np.array([]), np.array([])
    base = float(phase_frames) % step
    idx = base + np.arange(0.0, T - base + step, step, dtype=float)
    idx = idx[(idx >= 0) & (idx < T)]
    if subdiv > 1:
        main = float(phase_frames) % period_frames
        main_idx = main + np.arange(0.0, T - main + period_frames, period_frames, dtype=float)
        main_idx = main_idx[(main_idx >= 0) & (main_idx < T)]
        return (main_idx * hop_t, idx * hop_t)
    else:
        return (idx * hop_t, np.array([]))


def canonical_inizio(inz: float, start: float, bpm: float, beats_per_bar: float = 4.0) -> float:
        """Wrap inizio within one bar (``beats_per_bar`` beats) of the provided start.

        Downbeats fall every ``ts_num`` beats from inizio, so the bar must be the
        segment's own; wrapping a 3/4 segment by 4 beats would move its downbeat.
        """
        if not np.isfinite(inz):
            return float(start)
        if not (np.isfinite(bpm) and bpm > 0):
            return float(inz)
        period = 60.0 / float(bpm)
        if not np.isfinite(period) or period <= 0:
            return float(inz)
        meter = float(beats_per_bar) if np.isfinite(beats_per_bar) and beats_per_bar >= 1 else 4.0
        bar = round(meter) * period
        if bar <= 0:
            return float(inz)
        rel = (inz - start) % bar
        return float(start + rel)


def shift_grid_in_seg(beatgrid_sec: np.ndarray, segments, segment_index: int, offset: float):
    """Move one segment's beats by ``offset`` seconds (less than a beat). A tempo ramp moves
    along its own tempo line by the beats ``offset`` is at the tempo of its downbeat."""
    _beatgrid = np.asarray(beatgrid_sec, copy=True)
    _segments = np.asarray(segments, copy=True)
    if _segments.ndim == 2 and _segments.shape[1] > 5 and is_ramp_row(_segments[segment_index]):
        return _shift_ramp_in_seg(_beatgrid, _segments, segment_index, offset)
    st, ed, bpm, inz = _segments[segment_index][:4]
    assert 60/bpm > abs(offset)
    # Same span rebuild_grid_from_segments gives a segment: [st, ed) plus a hair
    # (the last segment keeps a beat sitting exactly on the track end).
    valid_ends = _segments[:, 1][np.isfinite(_segments[:, 1])]
    track_end = float(np.max(valid_ends)) if valid_ends.size else float("inf")
    end_limit = float(ed) + (1e-3 if float(ed) < track_end else 1e-6)
    # Indices delimiting that span in the current grid; a beat within the hair
    # past the previous segment's end is that segment's (as in the rebuild)
    sti = int(np.searchsorted(_beatgrid, st + (1e-3 if segment_index > 0 else 0.0), side="left"))
    edi = int(np.searchsorted(_beatgrid, end_limit, side="left"))
    sti = max(0, min(sti, len(_beatgrid)))
    edi = max(0, min(edi, len(_beatgrid)))

    # Update inizio in absolute track time; fall back to segment start if invalid
    T = 60.0 / bpm if bpm else np.inf
    cur_inz = _segments[segment_index, 3]
    if not np.isfinite(cur_inz):
        cur_inz = float(st)
    ts_num = float(_segments[segment_index, 4]) if _segments.shape[1] >= 5 else 4.0
    new_inz = canonical_inizio(float(cur_inz) + float(offset), st, bpm, ts_num)
    _segments[segment_index, 3] = new_inz

    # Rebuild beats inside [st, ed) using build_grid_from_period_phase
    if not np.isfinite(T) or T <= 0 or ed <= st:
        return _beatgrid, _segments

    hop_t = 1e-3  # 1 ms resolution
    total_len_frames = int(np.ceil(end_limit / hop_t))
    period_frames = T / hop_t
    phase_frames = new_inz / hop_t
    grid_main_sec, _ = build_grid_from_period_phase(
        total_len_frames=total_len_frames,
        hop_t=hop_t,
        period_frames=period_frames,
        phase_frames=phase_frames,
        subdiv=1,
    )
    if grid_main_sec.size:
        mask = (grid_main_sec >= st) & (grid_main_sec < end_limit)
        new_beats = grid_main_sec[mask]
    else:
        new_beats = np.array([], dtype=float)

    _beatgrid = prune_close_beats(np.concatenate([_beatgrid[:sti], new_beats, _beatgrid[edi:]]))
    return _beatgrid, _segments


def _shift_ramp_in_seg(beatgrid: np.ndarray, segments: np.ndarray, segment_index: int, offset: float):
    row = segments[segment_index]
    st, ed, inz = float(row[0]), float(row[1]), float(row[3])
    if not np.isfinite(inz):
        inz = st
    beats = float(offset) * bpm_at(row, inz) / 60.0
    segments[segment_index, 3] = shift_row_beats(row, beats)
    valid_ends = segments[:, 1][np.isfinite(segments[:, 1])]
    track_end = float(np.max(valid_ends)) if valid_ends.size else float("inf")
    end_limit = ed + (1e-3 if ed < track_end else 1e-6)
    sti = int(np.searchsorted(beatgrid, st + (1e-3 if segment_index > 0 else 0.0), side="left"))
    edi = int(np.searchsorted(beatgrid, end_limit, side="left"))
    row = segments[segment_index]
    new_beats = ramp_beat_times(st, ed, float(row[2]), segment_end_bpm(row), float(row[3]), end_limit)
    return prune_close_beats(np.concatenate([beatgrid[:sti], new_beats, beatgrid[edi:]])), segments
