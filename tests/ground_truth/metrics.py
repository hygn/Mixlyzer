"""How an analysis is scored against a song's ground truth: beats and downbeats by F-measure within 25 ms (a beat
that far off is still usable on a DJ grid), keys by the share of the track in the same key, phrases by their
boundaries within one beat and their labels by the share of time. NumPy only.

The audio a runner provides may be another encode of the song than the one annotated: an MP3's encoder delay moves
every time by a few tens of ms. ``offset`` finds that constant shift from the beats (at most ``MAX_OFFSET_SEC``) and
the scores take it out first."""
from __future__ import annotations

import numpy as np

BEAT_WINDOW_SEC = 0.025
MAX_OFFSET_SEC = 0.06


def _sorted(times) -> np.ndarray:
    t = np.asarray(times, dtype=float).ravel()
    return np.sort(t[np.isfinite(t)])


def matches(reference, estimate, window: float) -> int:
    """Pairs of a reference and an estimated time within ``window``, each used once (greedy by distance, as
    mir_eval's matching gives for well-separated events)."""
    ref, est = _sorted(reference), _sorted(estimate)
    if ref.size == 0 or est.size == 0:
        return 0
    pairs = []
    for i, t in enumerate(ref):
        lo = np.searchsorted(est, t - window, side="left")
        hi = np.searchsorted(est, t + window, side="right")
        pairs.extend((abs(est[j] - t), i, j) for j in range(lo, hi))
    used_r, used_e, n = set(), set(), 0
    for _d, i, j in sorted(pairs):
        if i not in used_r and j not in used_e:
            used_r.add(i)
            used_e.add(j)
            n += 1
    return n


def f_measure(reference, estimate, window: float = BEAT_WINDOW_SEC) -> float:
    ref, est = _sorted(reference), _sorted(estimate)
    if ref.size == 0 and est.size == 0:
        return 1.0
    if ref.size == 0 or est.size == 0:
        return 0.0
    m = matches(ref, est, window)
    return 2.0 * m / (ref.size + est.size)


def offset(reference, estimate, max_offset: float = MAX_OFFSET_SEC) -> float:
    """The constant shift (estimate - reference) the beats agree on best, within ``max_offset``: the median of each
    reference beat's nearest estimate where that is within ``max_offset``; 0 when too few agree."""
    ref, est = _sorted(reference), _sorted(estimate)
    if ref.size < 8 or est.size < 8:
        return 0.0
    k = np.clip(np.searchsorted(est, ref), 1, est.size - 1)
    nearest = np.where(np.abs(est[k] - ref) < np.abs(est[k - 1] - ref), est[k], est[k - 1]) - ref
    close = nearest[np.abs(nearest) <= max_offset]
    return float(np.median(close)) if close.size >= ref.size // 4 else 0.0


def scale_of(pitch: int, minor: int) -> int:
    """A key's scale (its key signature, the Alphanumeric number - 1): a major key and its relative minor alike."""
    return (7 * (int(pitch) + (3 if minor else 0))) % 12


def key_agreement(reference_rows, estimate_rows, duration: float, shift: float = 0.0) -> float:
    """The share of the reference's time where both name the same scale (key signature: a chart's "Key: Gb" names
    the scale, its mode is not told apart from the relative minor's). Only the time the reference covers is scored:
    it leaves out the stretches its source does not time for sure. Rows [pitch, minor, start, end]."""
    if duration <= 0:
        return 0.0
    grid = np.arange(0.0, duration, 0.05)

    def scale_at(rows, extra):
        rows = np.asarray(rows, dtype=float).reshape(-1, 4)
        out = np.full(grid.size, -1, dtype=int)
        for pitch, minor, start, end in rows:
            out[(grid >= start + extra) & (grid < end + extra)] = scale_of(pitch, minor)
        return out

    ref, est = scale_at(reference_rows, 0.0), scale_at(estimate_rows, -shift)
    covered = ref >= 0
    return float(np.mean(ref[covered] == est[covered])) if covered.any() else 0.0


def boundary_f(reference_bounds, estimate_bounds, beats) -> float:
    """Phrase boundaries (their inner starts) by F-measure within one beat, the beat period read where each
    reference boundary is."""
    ref, est, b = _sorted(reference_bounds), _sorted(estimate_bounds), _sorted(beats)
    if ref.size == 0 and est.size == 0:
        return 1.0
    if ref.size == 0 or est.size == 0 or b.size < 2:
        return 0.0
    period = float(np.median(np.diff(b)))
    return f_measure(ref, est, window=period * 1.0 + 1e-9)


def label_agreement(reference, estimate, duration: float, shift: float = 0.0) -> float:
    """The share of [0, duration] whose phrase label is the same in both (phrases as (start, end, label))."""
    if duration <= 0:
        return 0.0
    grid = np.arange(0.0, duration, 0.05)

    def label_at(rows, extra):
        out = np.full(grid.size, "", dtype=object)
        for start, end, label in rows:
            out[(grid >= float(start) + extra) & (grid < float(end) + extra)] = str(label)
        return out

    ref, est = label_at(reference, 0.0), label_at(estimate, -shift)
    return float(np.mean((ref == est) & (ref != "")))
