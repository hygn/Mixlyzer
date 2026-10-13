"""Second-pass phrase refinement features (boundary auto-context + repeat-aware labels).

Pass 1 is the two-stage model (boundary GBM -> decode -> label GBM). Pass 2
re-predicts the boundaries from pass-1 outputs and re-labels the segments:

* repetition: for every beat pair, how similar the 8/16 beats after (and
  before) them are, per family. Chroma takes the best of 12 circular shifts per
  window, so a phrase repeated after a key change still matches.
* learned similarity: a GBM predicts P(two bars carry the same phrase label)
  from those stripes, giving a bar x bar learned self-similarity matrix.
* pass-2 boundary features: pass-1 probability context, distance to the
  decoded boundaries, left/right window label posteriors, repetition, repeat
  partners' pass-1 outputs, learned checkerboard novelty.
* pass-2 label features: the segment's own / neighbours' / repeat partners'
  pass-1 label posteriors and the learned similarity to the other segments.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np

from analyzer_core.cue_and_phrase.model_features import segment_feature_matrix

REFINEMENT_VERSION = 1
REPEAT_WINDOWS = (8, 16)
REPEAT_MIN_LAG = 16
REPEAT_PARTNERS = 3
WINDOW_POSTERIOR_BEATS = 16
SIMILARITY_FAMILIES = ("all", "chroma", "timbre", "rhythm")
SIMILARITY_NOVELTY_BARS = (2, 4, 8)
PAIR_WINDOW_BEATS = 16
# Beat-feature family layout (structural_features.DEFAULT_FAMILY_SIZES = 73, 18, 27, 15).
_TIMBRE = slice(0, 73)
_CHROMA = slice(73, 85)
_RHYTHM = slice(91, 118)
_FAMILY_SLICES = (slice(0, 73), slice(73, 91), slice(91, 118), slice(118, 133))


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _window_means(d: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """after[t] = mean d[t:t+w], before[t] = mean d[t-w:t] (truncated; nan when empty)."""
    c = np.concatenate([[0.0], np.cumsum(d)])
    t = np.arange(d.size)
    hi = np.minimum(t + w, d.size)
    lo = np.maximum(t - w, 0)
    after = (c[hi] - c[t]) / np.maximum(hi - t, 1)
    before = np.full(d.size, np.nan)
    has = t > 0
    before[has] = (c[t[has]] - c[lo[has]]) / (t[has] - lo[has])
    return after, before


def repetition_matrices(feature_z: np.ndarray) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Symmetric beat x beat stripe similarities ``{window: {family: (after, before)}}``.

    after[i, j]: mean similarity of the ``window`` beats starting at i and j;
    before[i, j]: the same for the beats ending there. nan within
    ``REPEAT_MIN_LAG`` of the diagonal.
    """
    x = np.asarray(feature_z, dtype=np.float64)
    n = x.shape[0]
    families = {"timbre": _unit_rows(x[:, _TIMBRE]), "rhythm": _unit_rows(x[:, _RHYTHM])}
    chroma = _unit_rows(x[:, _CHROMA])
    out: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for w in REPEAT_WINDOWS:
        mats = {k: (np.full((n, n), np.nan, np.float32), np.full((n, n), np.nan, np.float32))
                for k in ("timbre", "rhythm", "chroma")}
        for lag in range(REPEAT_MIN_LAG, n):
            m = n - lag
            i = np.arange(m)
            j = i + lag
            for name, f in families.items():
                for M, v in zip(mats[name], _window_means(np.sum(f[:m] * f[lag:], axis=1), w)):
                    M[i, j] = v
                    M[j, i] = v
            best_after = np.full(m, -np.inf)
            best_before = np.full(m, -np.inf)
            for shift in range(12):
                after, before = _window_means(np.sum(chroma[:m] * np.roll(chroma[lag:], shift, axis=1), axis=1), w)
                best_after = np.fmax(best_after, after)
                best_before = np.fmax(best_before, before)
            for M, v in zip(mats["chroma"], (best_after, best_before)):
                v[np.isinf(v)] = np.nan
                M[i, j] = v
                M[j, i] = v
        mats["all"] = tuple(
            np.mean([mats[k][s] for k in ("timbre", "rhythm", "chroma")], axis=0).astype(np.float32)
            for s in (0, 1)
        )
        out[w] = mats
    return out


def repetition_features(mats: dict) -> np.ndarray:
    """Per beat: best after / before / continued match, repeat onset and offset strength."""
    cols: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for w in REPEAT_WINDOWS:
            for name in SIMILARITY_FAMILIES:
                A, B = mats[w][name]
                cols += [np.nanmax(A, 1), np.nanmax(B, 1), np.nanmax(np.fmin(A, B), 1),
                         np.nanmax(A - B, 1), np.nanmax(B - A, 1)]
    return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def repeat_partners(mats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Top repeat partners of every beat (16-beat combined stripes) and their weights."""
    A, B = mats[16]["all"]
    s = np.nan_to_num(np.fmax(A, B), nan=-1.0)
    idx = np.argsort(-s, axis=1, kind="stable")[:, :REPEAT_PARTNERS]
    w = np.clip(np.take_along_axis(s, idx, 1), 0.0, None).astype(np.float64) ** 2
    return idx.astype(np.int32), w


def bar_beats(downbeat_mask: np.ndarray, n_beats: int) -> np.ndarray:
    bars = np.flatnonzero(np.asarray(downbeat_mask, dtype=bool)[:n_beats])
    return bars if bars.size >= 4 else np.arange(0, int(n_beats), 4)


def pair_features(feature_z: np.ndarray, mats: dict, bars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features for every bar pair (a < b), stored at float16 precision so
    training caches and inference see identical values."""
    x = np.asarray(feature_z, dtype=np.float64)
    n = x.shape[0]
    a, b = np.triu_indices(bars.size, k=1)
    ia, ib = bars[a], bars[b]
    cols: list[np.ndarray] = []
    for w in REPEAT_WINDOWS:
        for name in SIMILARITY_FAMILIES:
            A, B = mats[w][name]
            cols += [A[ia, ib], B[ia, ib]]
    means = np.vstack([x[i:min(i + PAIR_WINDOW_BEATS, n)].mean(0) for i in bars])
    for sl in _FAMILY_SLICES:
        cols.append(np.abs(means[a, sl] - means[b, sl]).mean(1))
    cols += [ia / n, ib / n, (ib - ia) / n, ((ib - ia) % 32 == 0).astype(np.float64)]
    X = np.nan_to_num(np.column_stack(cols), nan=0.0).astype(np.float16)
    return X, a.astype(np.int32), b.astype(np.int32)


def similarity_matrix(pair_clf, X: np.ndarray, a: np.ndarray, b: np.ndarray, n_bars: int) -> np.ndarray:
    S = np.eye(int(n_bars), dtype=np.float64)
    if a.size:
        p = np.asarray(pair_clf.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)[:, 1]
        S[a, b] = p
        S[b, a] = p
    return S


def similarity_boundary_features(S: np.ndarray, bars: np.ndarray, n_beats: int) -> np.ndarray:
    """Per beat (nan off the bar grid): learned checkerboard novelty, within-block
    similarity and adjacent-bar change."""
    m = bars.size
    cols: list[np.ndarray] = []
    for k in SIMILARITY_NOVELTY_BARS:
        novelty = np.full(m, np.nan)
        within = np.full(m, np.nan)
        for j in range(1, m):
            lo, hi = max(0, j - k), min(m, j + k)
            within[j] = 0.5 * (S[lo:j, lo:j].mean() + S[j:hi, j:hi].mean())
            novelty[j] = within[j] - S[lo:j, j:hi].mean()
        cols += [novelty, within]
    change = np.full(m, np.nan)
    if m > 1:
        change[1:] = 1.0 - S[np.arange(1, m), np.arange(m - 1)]
    cols.append(change)
    out = np.full((int(n_beats), len(cols)), np.nan, np.float32)
    out[bars] = np.column_stack(cols)
    return out


def similarity_segment_features(S: np.ndarray, bars: np.ndarray, bounds: np.ndarray,
                                segment_posterior: np.ndarray) -> np.ndarray:
    """Per segment: similarity-weighted label posterior of the other segments,
    best / mean similarity to another segment, number of strong repeats."""
    bounds = np.asarray(bounds, dtype=np.int32)
    seg_of_bar = np.searchsorted(bounds, bars, side="right") - 1
    k = bounds.size - 1
    members = [np.flatnonzero(seg_of_bar == s) for s in range(k)]
    rows = []
    for s in range(k):
        sims = np.zeros(k)
        if members[s].size:
            for o in range(k):
                if o != s and members[o].size:
                    sims[o] = S[np.ix_(members[s], members[o])].max(1).mean()
        w = sims ** 2
        dist = (w[:, None] * segment_posterior).sum(0) / max(float(w.sum()), 1e-9)
        rows.append(np.concatenate([dist, [sims.max(initial=0.0), sims.mean() if k else 0.0,
                                           float(np.sum(sims > 0.5))]]))
    return np.vstack(rows)


def beat_posterior(segment_posterior: np.ndarray, bounds: np.ndarray, n_beats: int) -> np.ndarray:
    out = np.zeros((int(n_beats), segment_posterior.shape[1]), np.float32)
    for k, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        out[int(s):int(e)] = segment_posterior[k]
    return out


def window_label_posteriors(label_clf, feature_z: np.ndarray, valid: np.ndarray, downbeat: np.ndarray,
                            label_context_beats: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Label posterior of the WINDOW_POSTERIOR_BEATS beats before / after every
    valid downbeat (nan elsewhere)."""
    fz = np.asarray(feature_z, dtype=np.float64)
    n = fz.shape[0]
    w = WINDOW_POSTERIOR_BEATS
    n_labels = len(getattr(label_clf, "classes_", [])) or int(label_clf.baseline.size)
    left = np.full((n, n_labels), np.nan, np.float32)
    right = np.full((n, n_labels), np.nan, np.float32)
    at = [int(i) for i in np.flatnonzero(np.asarray(valid, bool) & np.asarray(downbeat, bool)[:n])
          if i - w >= 0 and i + w <= n]
    if at:
        rows = np.vstack([segment_feature_matrix(fz, np.asarray([i - w, i, i + w]), label_context_beats)
                          for i in at])
        p = np.asarray(label_clf.predict_proba(rows), dtype=np.float32)
        left[at], right[at] = p[0::2], p[1::2]
    return left, right


def _shifted(p: np.ndarray, d: int) -> np.ndarray:
    """out[i] = p[i + d], zero outside the track."""
    out = np.zeros(p.size)
    if d >= 0:
        out[:p.size - d] = p[d:]
    else:
        out[-d:] = p[:p.size + d]
    return out


def boundary_refinement_features(
    boundary_feature_z: np.ndarray,
    probability: np.ndarray,
    decoded_bounds: np.ndarray,
    beat_post: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    repetition: np.ndarray,
    partners: tuple[np.ndarray, np.ndarray],
    similarity: np.ndarray,
) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    n = p.size
    ctx = [_shifted(p, d) for d in (0, 1, -1, 2, -2, 4, -4, 8, -8)]
    local2 = np.asarray([p[i] >= p[max(0, i - 2):i + 3].max() for i in range(n)], np.float64)
    local8 = np.asarray([p[i] >= p[max(0, i - 8):i + 9].max() for i in range(n)], np.float64)
    inner = np.asarray(decoded_bounds, dtype=np.int64)[1:-1]
    dist = (np.full(n, 32.0) if inner.size == 0
            else np.minimum(np.min(np.abs(np.arange(n)[:, None] - inner[None]), 1), 32).astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        same = np.sum(left * right, 1)
        mid = 0.5 * (left + right)
        js = 0.5 * np.sum(left * np.log((left + 1e-7) / (mid + 1e-7))
                          + right * np.log((right + 1e-7) / (mid + 1e-7)), 1)
    differ = (np.argmax(np.nan_to_num(left), 1) != np.argmax(np.nan_to_num(right), 1)).astype(np.float64)
    differ[np.isnan(same)] = np.nan
    idx, w = partners
    wsum = w.sum(1) + 1e-9
    return np.hstack([
        np.asarray(boundary_feature_z, dtype=np.float32),
        np.column_stack(ctx + [local2, local8, dist, same, js, differ]).astype(np.float32),
        left, right, repetition,
        np.column_stack([(w * p[idx]).sum(1) / wsum, p[idx].max(1), p[idx[:, 0]]]).astype(np.float32),
        ((w[..., None] * beat_post[idx]).sum(1) / wsum[:, None]).astype(np.float32),
        similarity,
    ])


def label_refinement_features(
    feature_z: np.ndarray,
    bounds: np.ndarray,
    label_context_beats: Iterable[int],
    beat_post: np.ndarray,
    repetition: np.ndarray,
    partners: tuple[np.ndarray, np.ndarray],
    S: np.ndarray,
    bars: np.ndarray,
) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.int32)
    base = segment_feature_matrix(np.asarray(feature_z, dtype=np.float64), bounds, label_context_beats)
    idx, w = partners
    partner = (w[..., None] * beat_post[idx]).sum(1) / (w.sum(1) + 1e-9)[:, None]
    segs = list(zip(bounds[:-1], bounds[1:]))
    own = np.vstack([beat_post[s:e].mean(0) for s, e in segs])
    zero = np.zeros_like(own[0])
    rows = [np.concatenate([own[k], own[k - 1] if k > 0 else zero, own[k + 1] if k + 1 < len(segs) else zero,
                            partner[s:e].mean(0), repetition[s:e].mean(0)])
            for k, (s, e) in enumerate(segs)]
    return np.hstack([base, np.vstack(rows), similarity_segment_features(S, bars, bounds, own)])


def label_change_features(
    feature_z: np.ndarray,
    bounds: np.ndarray,
    label_context_beats: Iterable[int],
    beat_post: np.ndarray,
    S: np.ndarray,
    bars: np.ndarray,
) -> np.ndarray:
    """One row per interior boundary for the label-change GBM: both sides'
    segment features and pass-1 label posteriors, and the learned same-label
    similarity between the two segments."""
    bounds = np.asarray(bounds, dtype=np.int32)
    seg = segment_feature_matrix(np.asarray(feature_z, dtype=np.float64), bounds, label_context_beats)
    post = np.vstack([beat_post[s:e].mean(0) for s, e in zip(bounds[:-1], bounds[1:])])
    seg_of_bar = np.searchsorted(bounds, bars, side="right") - 1
    rows = []
    for k in range(1, bounds.size - 1):
        a, b = np.flatnonzero(seg_of_bar == k - 1), np.flatnonzero(seg_of_bar == k)
        if a.size and b.size:
            block = S[np.ix_(a, b)]
            sim = [block.mean(), block.max(), block.max(1).mean(), S[a[-1], b[0]]]
        else:
            sim = [np.nan] * 4
        left, right = post[k - 1], post[k]
        rows.append(np.concatenate([seg[k - 1], seg[k], seg[k] - seg[k - 1], left, right,
                                    [np.sum(left * right), float(np.argmax(left) != np.argmax(right))], sim]))
    width = 3 * seg.shape[1] + 2 * post.shape[1] + 6
    return np.vstack(rows) if rows else np.zeros((0, width))


def phrase_grid_features(bounds: np.ndarray, n_beats: int, exclude_beats: int = 12) -> np.ndarray:
    """Where beat j sits on the phrase grid of the other boundaries.

    Sections tend to start a multiple of 4/8 bars from the song's other
    boundaries, while the fill before a section does not. Boundaries within
    ``exclude_beats`` of j are ignored so a boundary and its twin peak don't
    vote for themselves. Columns: share of boundaries a multiple of 8/16/32
    beats away (plain and distance-weighted), whether the nearest boundary on
    each side is a multiple of 16/32 beats away, the dominant 16/32-beat
    boundary phase (match flag and share) and the boundary count.
    """
    b = np.asarray(bounds, dtype=np.int64)
    b = b[(b > 0) & (b < n_beats)]
    if b.size == 0:
        return np.full((n_beats, 13), np.nan, dtype=np.float32)
    d = np.arange(n_beats)[:, None] - b[None, :]
    use = np.abs(d) > exclude_beats
    w = np.exp(-np.abs(d) / 64.0) * use
    count = np.maximum(use.sum(axis=1), 1)
    cols = []
    for m in (8, 16, 32):
        on = (d % m == 0) & use
        cols += [on.sum(axis=1) / count, (on * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)]
    far = 10 ** 6
    left = np.where(use & (d > 0), d, far).min(axis=1)
    right = np.where(use & (d < 0), -d, far).min(axis=1)
    for x in (left, right):
        found = x < far
        cols += [np.where(found, x % 16 == 0, np.nan), np.where(found, x % 32 == 0, np.nan)]
    j = np.arange(n_beats)
    for m in (16, 32):
        phase = np.bincount(b % m, minlength=m)
        cols += [(j % m == np.argmax(phase)).astype(np.float64), phase[j % m] / b.size]
    cols.append(np.full(n_beats, float(b.size)))
    return np.column_stack(cols).astype(np.float32)


def shift_candidates(center: int, low: int, high: int, n_beats: int, offsets: Iterable[int],
                     edge_beats: int = 8) -> list[int]:
    return [center + int(o) for o in offsets
            if low <= center + int(o) <= high and edge_beats <= center + int(o) < n_beats - edge_beats]


def shift_key_changes(
    bounds: np.ndarray,
    labels: list[str],
    shift_clf,
    boundary_matrix: np.ndarray,
    offsets: Iterable[int],
    key_labels: Iterable[str],
    min_gap_beats: int = 8,
) -> np.ndarray:
    """Move every change between two different key labels to the bar-offset
    candidate the shift GBM ranks highest (kept ``min_gap_beats`` from the
    neighbouring boundaries). Labels stay with their segments."""
    keys = set(key_labels)
    out = np.asarray(bounds, dtype=np.int32).copy()
    n = int(out[-1])
    for k in range(1, out.size - 1):
        left, right = labels[k - 1], labels[k]
        if left == right or left not in keys or right not in keys:
            continue
        c = int(out[k])
        cands = shift_candidates(c, int(out[k - 1]) + min_gap_beats, int(out[k + 1]) - min_gap_beats, n, offsets)
        if c not in cands or len(cands) < 2:
            continue
        p = np.asarray(shift_clf.predict_proba(boundary_matrix[cands]), dtype=np.float64)[:, 1]
        best = int(np.argmax(p))
        if p[best] > p[cands.index(c)]:
            out[k] = cands[best]
    return out


__all__ = [
    "REFINEMENT_VERSION",
    "label_change_features",
    "shift_candidates",
    "shift_key_changes",
    "bar_beats",
    "beat_posterior",
    "boundary_refinement_features",
    "label_refinement_features",
    "pair_features",
    "phrase_grid_features",
    "repeat_partners",
    "repetition_features",
    "repetition_matrices",
    "similarity_boundary_features",
    "similarity_matrix",
    "window_label_posteriors",
]
