"""Non-neural whole-song structure features for phrase boundary models.

The implementation combines three classical music-structure ideas:

* beat-synchronous, family-balanced self-similarity;
* lag-domain median filtering and low-rank latent repetition factors;
* multi-scale novelty, recurrence turnover, and repeat-path endpoints.

All returned arrays are fixed-width and can be appended to GBM inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors


EPS = 1e-9
DEFAULT_FAMILY_SIZES = (73, 18, 27, 15)
DEFAULT_NOVELTY_WINDOWS = (4, 8, 16, 32)
DEFAULT_REPEAT_LENGTHS = (8, 16, 32, 64)
STRUCTURE_FEATURE_VERSION = 2
MULTI_VIEW_STRUCTURE_FEATURE_DIM = 180


def _normalized_positive(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values if int(value) > 0}))


def _family_balanced_rows(
    feature_z: np.ndarray,
    family_sizes: Sequence[int] = DEFAULT_FAMILY_SIZES,
) -> np.ndarray:
    x = np.asarray(feature_z, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_z must be a 2-D matrix")
    sizes = tuple(max(0, int(value)) for value in family_sizes)
    if sum(sizes) != x.shape[1] or not any(sizes):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, EPS)

    chunks: list[np.ndarray] = []
    start = 0
    for size in sizes:
        if size <= 0:
            continue
        chunk = x[:, start : start + size]
        start += size
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        chunks.append(chunk / np.maximum(norms, EPS))
    return np.hstack(chunks) / math.sqrt(max(len(chunks), 1))


def _delay_embed(rows: np.ndarray, delay: int = 1) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    lag = max(1, int(delay))
    delayed = np.vstack([x[:1].repeat(lag, axis=0), x[:-lag]])
    embedded = np.hstack([x, delayed])
    norms = np.linalg.norm(embedded, axis=1, keepdims=True)
    return embedded / np.maximum(norms, EPS)


def _cosine_knn_affinity(rows: np.ndarray, *, k: int, width: int) -> np.ndarray:
    """Dense ``librosa.segment.recurrence_matrix(rows.T, k=k, width=width,
    metric="cosine", sym=True, mode="affinity", bandwidth="med_k_scalar",
    self=True)``.

    Same steps and arithmetic as librosa, on a dense matrix instead of its
    per-row sparse (LIL) indexing, which dominated the phrase analysis time.
    A zero entry means "no link", as in the sparse matrix.
    """
    t = rows.shape[0]
    if width < 1 or width >= (t - 1) // 2:
        raise librosa.util.exceptions.ParameterError(
            f"width={width} must be at least 1 and at most (data.shape[-1] - 1) // 2={(t - 1) // 2}"
        )
    try:
        knn = NearestNeighbors(n_neighbors=min(t - 1, k + 2 * width), metric="cosine", algorithm="auto")
    except ValueError:
        knn = NearestNeighbors(n_neighbors=min(t - 1, k + 2 * width), metric="cosine", algorithm="brute")
    knn.fit(rows)
    rec = knn.kneighbors_graph(mode="distance").toarray()

    # Drop links within ``width`` of the diagonal, then keep each row's k nearest.
    positions = np.arange(t)
    rec[np.abs(positions[:, np.newaxis] - positions[np.newaxis, :]) < width] = 0.0
    for i in range(t):
        links = np.flatnonzero(rec[i])
        order = links[np.argsort(rec[i, links][np.newaxis, :])][0]
        rec[i, order[k:]] = 0.0
    np.fill_diagonal(rec, -1.0)
    rec = np.minimum(rec, rec.T)
    rec[rec < 0] = 0.0

    # "med_k_scalar": median over rows of the distance to the k-th nearest link.
    dist_to_k = np.full(t, np.nan)
    for i in range(t):
        links = np.flatnonzero(rec[i])
        if links.size:
            dist_to_k[i] = np.sort(rec[i, links])[:k][-1]
    if not np.any(np.isfinite(dist_to_k)):
        raise librosa.util.exceptions.ParameterError("Cannot estimate bandwidth from an empty graph")
    bandwidth = float(np.nanmedian(dist_to_k))

    linked = rec != 0.0
    np.fill_diagonal(linked, True)  # self links (stored as 0 after clipping) -> exp(0) = 1
    affinity = np.zeros_like(rec)
    affinity[linked] = np.exp(rec[linked] / (-1 * bandwidth))
    return affinity.T


def _recurrence_affinity(
    feature_z: np.ndarray,
    *,
    family_sizes: Sequence[int],
    neighbor_fraction: float,
    exclusion_beats: int,
) -> np.ndarray:
    rows = _delay_embed(_family_balanced_rows(feature_z, family_sizes))
    n = rows.shape[0]
    if n < 3:
        return np.eye(n, dtype=np.float64)
    k = int(np.clip(round(float(neighbor_fraction) * n), 2, max(2, n - 1)))
    result = _cosine_knn_affinity(rows, k=k, width=max(1, int(exclusion_beats)))
    np.fill_diagonal(result, 1.0)
    return np.clip(result, 0.0, 1.0)


def _mutual_knn_affinity(
    similarity: np.ndarray,
    *,
    neighbor_fraction: float,
    exclusion_beats: int,
) -> np.ndarray:
    """Convert a dense similarity matrix to a symmetric mutual-kNN affinity."""

    score = np.asarray(similarity, dtype=np.float64).copy()
    n = int(score.shape[0])
    if score.shape != (n, n):
        raise ValueError("similarity must be a square matrix")
    if n < 3:
        return np.eye(n, dtype=np.float64)
    positions = np.arange(n)
    score[np.abs(positions[:, None] - positions[None, :]) < max(1, int(exclusion_beats))] = -np.inf
    k = int(np.clip(round(float(neighbor_fraction) * n), 2, max(2, n - 1)))
    selected = np.zeros((n, n), dtype=bool)
    for row in range(n):
        candidates = np.flatnonzero(np.isfinite(score[row]))
        if candidates.size == 0:
            continue
        take = min(k, int(candidates.size))
        local = np.argpartition(score[row, candidates], -take)[-take:]
        selected[row, candidates[local]] = True
    mutual = selected & selected.T
    finite = score[np.isfinite(score)]
    if finite.size:
        low, high = np.percentile(finite, [10.0, 95.0])
        scale = max(float(high - low), EPS)
        affinity = np.clip((similarity - low) / scale, 0.0, 1.0)
    else:
        affinity = np.zeros_like(score)
    affinity = np.where(mutual, affinity, 0.0)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _transposition_invariant_harmony_affinity(
    feature_z: np.ndarray,
    *,
    neighbor_fraction: float,
    exclusion_beats: int,
) -> np.ndarray:
    """Return an affinity invariant to circular pitch-class transposition."""

    x = np.asarray(feature_z, dtype=np.float64)
    harmony_start = int(DEFAULT_FAMILY_SIZES[0])
    chroma = x[:, harmony_start : harmony_start + 12]
    if chroma.shape[1] != 12:
        return _recurrence_affinity(
            x,
            family_sizes=(x.shape[1],),
            neighbor_fraction=neighbor_fraction,
            exclusion_beats=exclusion_beats,
        )
    chroma = chroma - chroma.mean(axis=1, keepdims=True)
    chroma /= np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), EPS)
    similarity = np.full((x.shape[0], x.shape[0]), -1.0, dtype=np.float64)
    for shift in range(12):
        similarity = np.maximum(similarity, chroma @ np.roll(chroma, shift, axis=1).T)
    return _mutual_knn_affinity(
        similarity,
        neighbor_fraction=neighbor_fraction,
        exclusion_beats=exclusion_beats,
    )


def _block_contrast(ssm: np.ndarray, scale: int) -> np.ndarray:
    n = int(ssm.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for boundary in range(1, n):
        lo = max(0, boundary - int(scale))
        hi = min(n, boundary + int(scale))
        left = ssm[lo:boundary, lo:boundary]
        right = ssm[boundary:hi, boundary:hi]
        cross = ssm[lo:boundary, boundary:hi]
        if left.size and right.size and cross.size:
            out[boundary] = max(
                0.0,
                0.5 * (float(left.mean()) + float(right.mean())) - float(cross.mean()),
            )
    return out


def _recurrence_turnover(ssm: np.ndarray, scale: int) -> np.ndarray:
    n = int(ssm.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for boundary in range(1, n):
        lo = max(0, boundary - int(scale))
        hi = min(n, boundary + int(scale))
        if lo == boundary or hi == boundary:
            continue
        left = ssm[lo:boundary].mean(axis=0)
        right = ssm[boundary:hi].mean(axis=0)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        similarity = float(np.dot(left, right) / denominator) if denominator > EPS else 1.0
        out[boundary] = np.clip(1.0 - similarity, 0.0, 2.0)
    return out


def _aligned_similarity(ssm: np.ndarray, first: int, second: int, length: int) -> float:
    if length <= 0:
        return 0.0
    idx = np.arange(int(length))
    return float(np.mean(ssm[int(first) + idx, int(second) + idx]))


def _local_repeat_features(ssm: np.ndarray, lengths: Iterable[int]) -> np.ndarray:
    n = int(ssm.shape[0])
    columns: list[np.ndarray] = []
    for length in _normalized_positive(lengths):
        closure = np.zeros(n, dtype=np.float64)
        crossing = np.zeros(n, dtype=np.float64)
        for boundary in range(1, n):
            if boundary >= 2 * length:
                closure[boundary] = _aligned_similarity(
                    ssm, boundary - 2 * length, boundary - length, length
                )
            if boundary >= length and boundary + length <= n:
                crossing[boundary] = _aligned_similarity(
                    ssm, boundary - length, boundary, length
                )
        columns.extend([closure, np.clip(closure - crossing, 0.0, None)])
    return np.vstack(columns).T if columns else np.zeros((n, 0), dtype=np.float64)


def _repeat_path_endpoints(
    ssm: np.ndarray,
    *,
    min_length: int,
    exclusion: int,
    neighbor_fraction: float,
) -> np.ndarray:
    n = int(ssm.shape[0])
    if n < 3:
        return np.zeros(n, dtype=np.float64)
    score = np.asarray(ssm, dtype=np.float64).copy()
    positions = np.arange(n)
    score[np.abs(positions[:, None] - positions[None, :]) <= int(exclusion)] = -np.inf
    k = int(np.clip(round(float(neighbor_fraction) * n), 1, max(1, n - 1)))
    row_mask = np.zeros((n, n), dtype=bool)
    for row in range(n):
        candidates = np.flatnonzero(np.isfinite(score[row]))
        if candidates.size == 0:
            continue
        take = min(k, candidates.size)
        selected = np.argpartition(score[row, candidates], -take)[-take:]
        row_mask[row, candidates[selected]] = True
    recurrence = row_mask & row_mask.T
    endpoint = np.zeros(n, dtype=np.float64)
    for offset in range(int(exclusion) + 1, n):
        diagonal = np.diagonal(recurrence, offset=offset)
        padded = np.concatenate([[False], diagonal, [False]])
        changes = np.diff(padded.astype(np.int8))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            length = int(end - start)
            if length < int(min_length):
                continue
            first = int(start)
            second = first + offset
            quality = _aligned_similarity(ssm, first, second, length) * math.sqrt(length)
            for boundary in (first, second, first + length, second + length):
                if 0 <= boundary < n:
                    endpoint[boundary] += quality
    peak = float(endpoint.max(initial=0.0))
    return endpoint / peak if peak > 0.0 else endpoint


def _latent_repetition(
    ssm: np.ndarray,
    *,
    components: int,
    median_width: int,
) -> np.ndarray:
    n = int(ssm.shape[0])
    count = min(max(0, int(components)), max(0, n - 1))
    if count == 0:
        return np.zeros((n, 0), dtype=np.float64)
    lag = librosa.segment.recurrence_to_lag((ssm > 0.0).astype(np.float64), pad=True)
    filtered = median_filter(
        np.asarray(lag, dtype=np.float64),
        size=(1, max(1, int(median_width))),
        mode="nearest",
    )
    try:
        _u, singular, vt = svds(
            csr_matrix(filtered),
            k=count,
            which="LM",
            random_state=0,
            return_singular_vectors=True,
        )
        order = np.argsort(singular)[::-1]
        singular = singular[order]
        vt = vt[order]
        latent = singular[:, None] * vt
        latent /= max(float(singular[0]), EPS)
    except Exception:
        _u, singular, vt = np.linalg.svd(filtered, full_matrices=False)
        latent = singular[:count, None] * vt[:count]
        latent /= max(float(singular[0]), EPS)
    # Singular-vector signs are arbitrary; orient each component deterministically.
    for row in latent:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return latent.T


def _global_recurrence_summary(ssm: np.ndarray, exclusion: int) -> np.ndarray:
    """Summarize where each beat finds its strongest past/future repetitions."""

    similarity = np.asarray(ssm, dtype=np.float64)
    n = int(similarity.shape[0])
    out = np.zeros((n, 8), dtype=np.float64)
    for beat in range(n):
        past_hi = max(0, beat - int(exclusion))
        future_lo = min(n, beat + int(exclusion) + 1)
        past = similarity[beat, :past_hi]
        future = similarity[beat, future_lo:]
        if past.size:
            best = int(np.argmax(past))
            ranked = np.sort(past)[-min(3, past.size) :]
            out[beat, 0] = float(past[best])
            out[beat, 1] = float(np.mean(ranked))
            out[beat, 2] = (beat - best) / max(n, 1)
            out[beat, 3] = 1.0
        if future.size:
            best_local = int(np.argmax(future))
            best = future_lo + best_local
            ranked = np.sort(future)[-min(3, future.size) :]
            out[beat, 4] = float(future[best_local])
            out[beat, 5] = float(np.mean(ranked))
            out[beat, 6] = (best - beat) / max(n, 1)
            out[beat, 7] = 1.0
    return out


def _ssm_boundary_core(
    ssm: np.ndarray,
    *,
    novelty_windows: Iterable[int],
    repeat_lengths: Iterable[int],
    exclusion_beats: int,
    neighbor_fraction: float,
    median_width: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for scale in _normalized_positive(novelty_windows):
        columns.append(_block_contrast(ssm, scale))
        columns.append(_recurrence_turnover(ssm, scale))
    columns.append(
        _repeat_path_endpoints(
            ssm,
            min_length=max(4, int(median_width)),
            exclusion=exclusion_beats,
            neighbor_fraction=neighbor_fraction,
        )
    )
    boundary = np.vstack(columns).T
    return np.hstack([boundary, _local_repeat_features(ssm, repeat_lengths)])


def _ssm_boundary_summary(
    ssm: np.ndarray,
    *,
    novelty_windows: Iterable[int],
    repeat_lengths: Iterable[int],
    exclusion_beats: int,
    neighbor_fraction: float,
    median_width: int,
) -> np.ndarray:
    return np.hstack(
        [
            _ssm_boundary_core(
                ssm,
                novelty_windows=novelty_windows,
                repeat_lengths=repeat_lengths,
                exclusion_beats=exclusion_beats,
                neighbor_fraction=neighbor_fraction,
                median_width=median_width,
            ),
            _global_recurrence_summary(ssm, exclusion_beats),
        ]
    )


def _song_position_features(n: int) -> np.ndarray:
    position = np.arange(int(n), dtype=np.float64) / max(int(n) - 1, 1)
    remaining = 1.0 - position
    from_start = np.arange(int(n), dtype=np.float64) + 1.0
    to_end = np.arange(int(n), 0, -1, dtype=np.float64)
    scale = math.log1p(max(int(n), 1))
    return np.column_stack(
        [
            position,
            remaining,
            position * position,
            remaining * remaining,
            np.sqrt(position),
            np.sqrt(remaining),
            np.log1p(from_start) / scale,
            np.log1p(to_end) / scale,
            np.sin(math.pi * position),
            np.cos(math.pi * position),
        ]
    )


def whole_song_structure_features(
    feature_z: np.ndarray,
    *,
    family_sizes: Sequence[int] = DEFAULT_FAMILY_SIZES,
    novelty_windows: Iterable[int] = DEFAULT_NOVELTY_WINDOWS,
    repeat_lengths: Iterable[int] = DEFAULT_REPEAT_LENGTHS,
    latent_components: int = 8,
    neighbor_fraction: float = 0.08,
    exclusion_beats: int = 8,
    median_width: int = 9,
) -> np.ndarray:
    """Return per-beat latent repetition and boundary evidence features."""

    x = np.asarray(feature_z, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_z must be a 2-D matrix")
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    ssm = _recurrence_affinity(
        x,
        family_sizes=family_sizes,
        neighbor_fraction=neighbor_fraction,
        exclusion_beats=exclusion_beats,
    )
    latent = _latent_repetition(
        ssm,
        components=latent_components,
        median_width=median_width,
    )
    previous_latent = np.vstack([latent[:1], latent[:-1]])
    latent_delta = latent - previous_latent
    latent_context_change: list[np.ndarray] = []
    for scale in _normalized_positive(novelty_windows):
        curve = np.zeros(n, dtype=np.float64)
        for boundary_index in range(1, n):
            lo = max(0, boundary_index - scale)
            hi = min(n, boundary_index + scale)
            left = latent[lo:boundary_index]
            right = latent[boundary_index:hi]
            if left.size and right.size:
                curve[boundary_index] = float(
                    np.linalg.norm(right.mean(axis=0) - left.mean(axis=0))
                    / math.sqrt(max(latent.shape[1], 1))
                )
        latent_context_change.append(curve)
    boundary = _ssm_boundary_core(
        ssm,
        novelty_windows=novelty_windows,
        repeat_lengths=repeat_lengths,
        exclusion_beats=exclusion_beats,
        neighbor_fraction=neighbor_fraction,
        median_width=median_width,
    )
    return np.nan_to_num(
        np.hstack(
            [
                latent,
                latent_delta,
                np.abs(latent_delta),
                np.vstack(latent_context_change).T,
                boundary,
            ]
        ),
        copy=False,
    )


def multi_view_structure_features(
    feature_z: np.ndarray,
    *,
    family_sizes: Sequence[int] = DEFAULT_FAMILY_SIZES,
    novelty_windows: Iterable[int] = DEFAULT_NOVELTY_WINDOWS,
    repeat_lengths: Iterable[int] = DEFAULT_REPEAT_LENGTHS,
    latent_components: int = 8,
    neighbor_fraction: float = 0.08,
    exclusion_beats: int = 8,
    median_width: int = 9,
) -> np.ndarray:
    """Return fused plus family-specific and transposition-invariant structure.

    Each acoustic family gets its own recurrence representation so that a
    boundary may be supported by harmony, rhythm, timbre, or texture without
    being diluted by the other families.  Song-position columns make truncated
    right context explicit near the tail instead of silently treating it as a
    normal full-context observation.
    """

    x = np.asarray(feature_z, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_z must be a 2-D matrix")
    n = int(x.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    base = whole_song_structure_features(
        x,
        family_sizes=family_sizes,
        novelty_windows=novelty_windows,
        repeat_lengths=repeat_lengths,
        latent_components=latent_components,
        neighbor_fraction=neighbor_fraction,
        exclusion_beats=exclusion_beats,
        median_width=median_width,
    )
    views: list[np.ndarray] = [base]
    start = 0
    for raw_size in family_sizes:
        size = max(0, int(raw_size))
        if size <= 0:
            continue
        chunk = x[:, start : start + size]
        start += size
        if chunk.shape[1] == 0:
            continue
        ssm = _recurrence_affinity(
            chunk,
            family_sizes=(chunk.shape[1],),
            neighbor_fraction=neighbor_fraction,
            exclusion_beats=exclusion_beats,
        )
        views.append(
            _ssm_boundary_summary(
                ssm,
                novelty_windows=novelty_windows,
                repeat_lengths=repeat_lengths,
                exclusion_beats=exclusion_beats,
                neighbor_fraction=neighbor_fraction,
                median_width=median_width,
            )
        )
    harmony_ssm = _transposition_invariant_harmony_affinity(
        x,
        neighbor_fraction=neighbor_fraction,
        exclusion_beats=exclusion_beats,
    )
    views.append(
        _ssm_boundary_summary(
            harmony_ssm,
            novelty_windows=novelty_windows,
            repeat_lengths=repeat_lengths,
            exclusion_beats=exclusion_beats,
            neighbor_fraction=neighbor_fraction,
            median_width=median_width,
        )
    )
    views.append(_song_position_features(n))
    return np.nan_to_num(np.hstack(views), copy=False)


__all__ = [
    "DEFAULT_FAMILY_SIZES",
    "DEFAULT_NOVELTY_WINDOWS",
    "DEFAULT_REPEAT_LENGTHS",
    "MULTI_VIEW_STRUCTURE_FEATURE_DIM",
    "STRUCTURE_FEATURE_VERSION",
    "multi_view_structure_features",
    "whole_song_structure_features",
]
