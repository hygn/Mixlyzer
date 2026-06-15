from __future__ import annotations

import numpy as np


DEFAULT_TIME_SIGNATURE = 4


def _normalize_segments(tempo_segments) -> np.ndarray | None:
    """Coerce tempo_segments into a 2-D array with at least [start, end, bpm, inizio].

    Accepts N×3 (legacy, no inizio/ts), N×4 (start,end,bpm,inizio) and
    N×5 (… , ts_num). Returns None when the input is empty or malformed.
    """
    if tempo_segments is None:
        return None
    try:
        arr = np.asarray(tempo_segments, dtype=float)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        for width in (5, 4, 3):
            if arr.size % width == 0:
                arr = arr.reshape((-1, width))
                break
        else:
            return None
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return arr


def segment_time_signature(seg_row) -> int:
    """Read the per-segment time-signature numerator (default 4) from a row.

    Tolerates legacy rows that lack the 5th column.
    """
    row = np.asarray(seg_row, dtype=float).ravel()
    if row.size >= 5 and np.isfinite(row[4]) and int(round(row[4])) >= 1:
        return int(round(row[4]))
    return DEFAULT_TIME_SIGNATURE


def build_downbeats_from_segments(tempo_segments) -> np.ndarray | None:
    """Compute downbeat (bar-start) times honoring each segment's time signature.

    A downbeat is drawn every ``ts_num`` beats, where ``ts_num`` is the 5th
    column of the segment when present (defaults to 4). Returns a sorted unique
    array of seconds, or None when there is nothing to draw.
    """
    arr = _normalize_segments(tempo_segments)
    if arr is None:
        return None

    downbeats: list[float] = []
    for seg in arr:
        start, end, bpm = seg[0], seg[1], seg[2]
        inizio = seg[3] if seg.size >= 4 else start
        if not (np.isfinite(inizio) and np.isfinite(end) and np.isfinite(bpm)):
            continue
        if bpm <= 0:
            continue
        beats_per_bar = segment_time_signature(seg)
        t = max(0.0, float(inizio))
        stop = max(float(end), t)
        bar = beats_per_bar * 60.0 / float(bpm)
        if bar <= 0:
            continue
        while t <= stop + 1e-6:
            downbeats.append(t)
            t += bar
    if not downbeats:
        return None
    return np.asarray(sorted(set(downbeats)), dtype=float)


def _clean_beats(beat_times) -> np.ndarray:
    """Return a sorted array of finite beat times (possibly empty)."""
    if beat_times is None:
        return np.empty(0, dtype=float)
    beats = np.asarray(beat_times, dtype=float).ravel()
    return np.sort(beats[np.isfinite(beats)])


def downbeat_beat_indices(beat_times, tempo_segments) -> np.ndarray:
    """Indices into ``beat_times`` that are downbeats (bar starts).

    Computed entirely in beat-index space: within each tempo segment a downbeat
    falls every ``ts_num`` beats counted from the beat nearest the segment's
    ``inizio`` reference. Because it steps over integer beat indices rather than
    accumulating bar durations in seconds, the result never drifts away from the
    real beats over a long track. Returns a sorted, unique int array.
    """
    beats = _clean_beats(beat_times)
    if beats.size == 0:
        return np.empty(0, dtype=np.int64)

    arr = _normalize_segments(tempo_segments)
    if arr is None:
        # No tempo info: assume a 4/4 grid anchored on the first beat.
        return np.arange(0, beats.size, DEFAULT_TIME_SIGNATURE, dtype=np.int64)

    out: list[int] = []
    for seg in arr:
        start, end, bpm = seg[0], seg[1], seg[2]
        inizio = seg[3] if seg.size >= 4 else start
        if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(inizio)):
            continue
        ts = segment_time_signature(seg)
        # Beat-index range covered by this segment: [lo, hi).
        lo = int(np.searchsorted(beats, float(start), side="left"))
        hi = int(np.searchsorted(beats, float(end), side="left"))
        if hi <= lo:
            continue
        # Reference downbeat = beat nearest the segment's inizio.
        ref = int(np.argmin(np.abs(beats - float(inizio))))
        # First index in [lo, hi) congruent to ref (mod ts), then every ts beats.
        j = lo + (ref - lo) % ts
        while j < hi:
            out.append(j)
            j += ts
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.asarray(out, dtype=np.int64))


def bar_beat_position(
    beat_times,
    current_time,
    tempo_segments=None,
    *,
    downbeat_indices=None,
) -> tuple[int, int] | None:
    """Musical (bar, beat) position at ``current_time``.

    The bar number counts downbeats (bar starts) at or before the time; the
    beat number counts beats elapsed since that bar's downbeat. Both are
    1-based. Provide either ``tempo_segments`` (downbeats derived via
    :func:`downbeat_beat_indices`) or precomputed ``downbeat_indices`` into the
    same ``beat_times`` array (cheaper for repeated calls). Everything is
    computed in beat-index space so the bar grid never drifts.

    Returns ``None`` when beat or downbeat data is unavailable. Before the
    first bar starts, returns ``(0, beats_until_first_bar)``.
    """
    beats = _clean_beats(beat_times)
    if beats.size == 0:
        return None

    if downbeat_indices is None:
        downbeat_indices = downbeat_beat_indices(beats, tempo_segments)
    db_idx = np.asarray(downbeat_indices, dtype=np.int64).ravel()
    if db_idx.size == 0:
        return None

    t = float(current_time)
    # Current beat index: the last beat at or before t (-1 when before beat 0).
    cur = int(np.searchsorted(beats, t, side="right")) - 1

    bar = int(np.searchsorted(db_idx, cur, side="right"))
    if bar <= 0:
        # Before the first downbeat: beats remaining until it starts.
        return (0, max(0, int(db_idx[0]) - cur))
    beat_in_bar = cur - int(db_idx[bar - 1]) + 1
    return (bar, max(1, beat_in_bar))


def bar_beat_label(
    beat_times,
    current_time,
    tempo_segments=None,
    *,
    downbeat_indices=None,
) -> str:
    """Format :func:`bar_beat_position` as a ``"bar.beat"`` string.

    Returns ``""`` when the position cannot be computed.
    """
    pos = bar_beat_position(
        beat_times,
        current_time,
        tempo_segments,
        downbeat_indices=downbeat_indices,
    )
    if pos is None:
        return ""
    return f"{pos[0]}.{pos[1]}"
