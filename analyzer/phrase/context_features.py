"""Compact, multi-scale context features for the phrase GBMs.

The phrase classifiers consume fixed-width rows, so they cannot ingest a whole
song as a variable-length sequence.  These helpers summarize a wider region
around each boundary/segment without introducing a learned global embedding.
The features are appended in the fixed order expected by the current Phrase
NPZ artifact.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


EPS = 1e-9
CONTEXT_FEATURE_VERSION = 2

# These windows reach roughly 8/16/32 bars in 4/4 while remaining local enough
# to avoid turning normalized song position into the main decision signal.
DEFAULT_BOUNDARY_REGIONAL_CONTEXT_BEATS = (32, 64, 128)
DEFAULT_LABEL_CONTEXT_BEATS = (16, 32, 64)


def _normalized_windows(windows: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in windows if int(value) > 0}))


def boundary_context_coverage_features(
    n_beats: int,
    local_windows: Iterable[int],
    regional_windows: Iterable[int] = (),
) -> np.ndarray:
    """Return raw left/right availability for every requested context scale.

    Coverage deliberately bypasses per-track standardization.  A value of 0.25
    must remain 0.25 so a classifier can distinguish genuinely short context
    from an ordinary feature value close to the track median.
    """

    n = max(0, int(n_beats))
    windows = _normalized_windows((*local_windows, *regional_windows))
    if n == 0 or not windows:
        return np.zeros((n, 0), dtype=np.float64)

    out = np.zeros((n, 2 * len(windows)), dtype=np.float64)
    for boundary in range(n):
        values: list[float] = []
        for window in windows:
            left_size = min(boundary, window)
            right_size = min(n - boundary, window)
            values.extend([left_size / float(window), right_size / float(window)])
        out[boundary] = values
    return out


def _mean_std_or_reference(
    block: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if block.shape[0] == 0:
        return reference, np.zeros_like(reference)
    return block.mean(axis=0), block.std(axis=0)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPS:
        return 0.0
    similarity = float(np.dot(left, right) / denominator)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def boundary_regional_context_features(
    feature_z: np.ndarray,
    windows: Iterable[int] = DEFAULT_BOUNDARY_REGIONAL_CONTEXT_BEATS,
) -> np.ndarray:
    """Return broad left/right context summaries for every beat boundary.

    Each scale contributes eight scalar features: left/right coverage, mean
    absolute and RMS change, cosine distance, spread change, and the spread on
    each side.  Coverage makes truncated head/tail context explicit instead of
    silently treating it as a normal full window.
    """

    x = np.asarray(feature_z, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_z must be a 2-D matrix")
    n, d = x.shape
    wins = _normalized_windows(windows)
    if n == 0 or not wins:
        return np.zeros((n, 0), dtype=np.float64)

    out = np.zeros((n, len(wins) * 8), dtype=np.float64)
    for boundary in range(n):
        reference = x[boundary]
        values: list[float] = []
        for window in wins:
            left = x[max(0, boundary - window) : boundary]
            right = x[boundary : min(n, boundary + window)]
            left_mean, left_std = _mean_std_or_reference(left, reference)
            right_mean, right_std = _mean_std_or_reference(right, reference)
            delta = right_mean - left_mean
            values.extend(
                [
                    left.shape[0] / float(window),
                    right.shape[0] / float(window),
                    float(np.mean(np.abs(delta))) if d else 0.0,
                    float(np.sqrt(np.mean(delta * delta))) if d else 0.0,
                    _cosine_distance(left_mean, right_mean),
                    float(np.mean(np.abs(right_std - left_std))) if d else 0.0,
                    float(np.mean(left_std)) if d else 0.0,
                    float(np.mean(right_std)) if d else 0.0,
                ]
            )
        out[boundary] = values
    return np.nan_to_num(out, copy=False)


def segment_regional_context_features(
    feature_z: np.ndarray,
    bounds: np.ndarray,
    windows: Iterable[int] = DEFAULT_LABEL_CONTEXT_BEATS,
) -> np.ndarray:
    """Return surrounding-context and repetition summaries per phrase segment.

    Context is summarized on both sides at several beat scales.  A small set of
    segment-to-segment cosine features lets the label GBM recognize repeated
    material elsewhere in the song without receiving a full-song embedding.
    """

    x = np.asarray(feature_z, dtype=np.float64)
    boundary = np.asarray(bounds, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise ValueError("feature_z must be a 2-D matrix")
    wins = _normalized_windows(windows)
    if not wins:
        return np.zeros((max(0, boundary.size - 1), 0), dtype=np.float64)
    if boundary.size < 2:
        return np.zeros((0, len(wins) * 8 + 8), dtype=np.float64)

    n = x.shape[0]
    segments: list[tuple[int, int]] = []
    means: list[np.ndarray] = []
    for start_raw, end_raw in zip(boundary[:-1], boundary[1:]):
        start = int(np.clip(start_raw, 0, max(n - 1, 0)))
        end = int(np.clip(end_raw, start + 1, n))
        block = x[start:end]
        segments.append((start, end))
        means.append(block.mean(axis=0))

    segment_means = np.vstack(means)
    rows: list[np.ndarray] = []
    for index, ((start, end), segment_mean) in enumerate(zip(segments, segment_means, strict=True)):
        values: list[float] = []
        for window in wins:
            left = x[max(0, start - window) : start]
            right = x[end : min(n, end + window)]
            left_mean, _left_std = _mean_std_or_reference(left, segment_mean)
            right_mean, _right_std = _mean_std_or_reference(right, segment_mean)
            values.extend(
                [
                    left.shape[0] / float(window),
                    right.shape[0] / float(window),
                    float(np.mean(np.abs(segment_mean - left_mean))),
                    _cosine_distance(segment_mean, left_mean),
                    float(np.mean(np.abs(segment_mean - right_mean))),
                    _cosine_distance(segment_mean, right_mean),
                    float(np.mean(np.abs(right_mean - left_mean))),
                    _cosine_distance(left_mean, right_mean),
                ]
            )

        candidate_indices = [
            other for other in range(len(segments))
            if other != index and abs(other - index) > 1
        ]
        if not candidate_indices:
            candidate_indices = [other for other in range(len(segments)) if other != index]

        similarities: list[tuple[int, float]] = []
        norm = float(np.linalg.norm(segment_mean))
        for other in candidate_indices:
            other_mean = segment_means[other]
            denominator = norm * float(np.linalg.norm(other_mean))
            similarity = 0.0 if denominator <= EPS else float(
                np.clip(np.dot(segment_mean, other_mean) / denominator, -1.0, 1.0)
            )
            similarities.append((other, similarity))

        previous = [score for other, score in similarities if other < index]
        following = [score for other, score in similarities if other > index]
        ranked = sorted((score for _other, score in similarities), reverse=True)
        if similarities:
            best_other, best_score = max(similarities, key=lambda item: item[1])
            repeat_fraction = (
                sum(score >= 0.8 for _other, score in similarities)
                / len(similarities)
            )
            best_distance = abs(
                0.5 * (segments[best_other][0] + segments[best_other][1])
                - 0.5 * (start + end)
            ) / max(n, 1)
        else:
            best_score = repeat_fraction = best_distance = 0.0
        values.extend(
            [
                best_score,
                max(previous) if previous else 0.0,
                max(following) if following else 0.0,
                float(np.mean(ranked[:3])) if ranked else 0.0,
                float(repeat_fraction),
                float(best_distance),
                float(bool(previous)),
                float(bool(following)),
            ]
        )
        rows.append(np.asarray(values, dtype=np.float64))

    return np.nan_to_num(np.vstack(rows), copy=False)


__all__ = [
    "CONTEXT_FEATURE_VERSION",
    "DEFAULT_BOUNDARY_REGIONAL_CONTEXT_BEATS",
    "DEFAULT_LABEL_CONTEXT_BEATS",
    "boundary_context_coverage_features",
    "boundary_regional_context_features",
    "segment_regional_context_features",
]
