"""Canonical feature assembly shared by Phrase training and inference."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np

from analyzer.phrase.context_features import (
    boundary_context_coverage_features,
    boundary_regional_context_features,
    segment_regional_context_features,
)


EPS = 1e-9


def robust_standardize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return (x - med) / scale


def column_standardize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return (x - med) / scale


def feature_z_from_acoustic(acoustic) -> np.ndarray:
    rows = np.vstack(
        [
            np.asarray(acoustic.family_beat["timbre"], dtype=np.float64),
            np.asarray(acoustic.family_beat["harmony"], dtype=np.float64),
            np.asarray(acoustic.family_beat["rhythm"], dtype=np.float64),
            np.asarray(acoustic.family_beat["texture"], dtype=np.float64),
        ]
    )
    # The optimizer cache persists float32 rows.  Inference uses the same dtype
    # before all downstream functions promote to float64, keeping training and
    # inference on the same numeric representation.
    return robust_standardize_rows(rows).T.astype(np.float32)


def grid_context_feature_matrix(feature_z: np.ndarray, grid) -> np.ndarray:
    n = int(np.asarray(feature_z).shape[0])
    beat_in_bar = np.asarray(grid.beat_in_bar, dtype=np.float64).reshape(-1)
    bar_index = np.asarray(grid.bar_index_of_beat, dtype=np.float64).reshape(-1)
    downbeat = np.asarray(grid.downbeat_mask, dtype=bool).reshape(-1)
    if beat_in_bar.size != n or bar_index.size != n or downbeat.size != n:
        raise ValueError("Current beat-grid context arrays do not match feature_z")

    meters = np.asarray(grid.bar_meters, dtype=np.float64).reshape(-1)
    if not meters.size:
        raise ValueError("Current beat-grid context is missing bar_meters")
    beat_meters = np.full(n, 4.0, dtype=np.float64)
    valid_bar = (bar_index >= 0) & (bar_index < meters.size)
    beat_meters[valid_bar] = np.maximum(
        meters[bar_index[valid_bar].astype(int)], 1.0
    )
    phase = np.mod(beat_in_bar, beat_meters) / np.maximum(beat_meters, 1.0)

    downbeat_idx = np.flatnonzero(downbeat)
    prev_dist = np.full(n, n, dtype=np.float64)
    next_dist = np.full(n, n, dtype=np.float64)
    if downbeat_idx.size:
        positions = np.arange(n)
        previous = np.searchsorted(downbeat_idx, positions, side="right") - 1
        has_previous = previous >= 0
        prev_dist[has_previous] = (
            positions[has_previous] - downbeat_idx[previous[has_previous]]
        )
        following = np.searchsorted(downbeat_idx, positions, side="left")
        has_following = following < downbeat_idx.size
        next_dist[has_following] = (
            downbeat_idx[following[has_following]] - positions[has_following]
        )
    dist_to_downbeat = np.minimum(prev_dist, next_dist) / np.maximum(
        beat_meters, 1.0
    )

    rows = [
        downbeat.astype(np.float64),
        (beat_in_bar == 1).astype(np.float64),
        np.sin(2.0 * math.pi * phase),
        np.cos(2.0 * math.pi * phase),
        dist_to_downbeat,
    ]
    bar_pos = np.maximum(bar_index, 0.0)
    for period in (2.0, 4.0, 8.0, 16.0):
        rows.extend(
            [
                (np.mod(bar_pos, period) == 0.0).astype(np.float64),
                np.sin(2.0 * math.pi * bar_pos / period),
                np.cos(2.0 * math.pi * bar_pos / period),
            ]
        )
    return column_standardize(np.vstack(rows).T)


def boundary_feature_matrix(
    feature_z: np.ndarray,
    windows: Iterable[int],
    grid_context: np.ndarray,
    regional_windows: Iterable[int],
) -> np.ndarray:
    """Build the canonical vectorized boundary feature matrix."""

    x = np.asarray(feature_z, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_z must be a 2-D matrix")
    n, d = x.shape
    wins = tuple(sorted({max(1, int(window)) for window in windows}))
    grid_context_arr = np.asarray(grid_context, dtype=np.float64)
    if grid_context_arr.ndim != 2 or grid_context_arr.shape[0] != n:
        raise ValueError("Current grid context does not match feature_z")

    global_std = np.std(x, axis=0) + EPS
    regional_context = boundary_regional_context_features(x, regional_windows)
    prefix = np.vstack(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(x, axis=0)]
    )
    prefix_sq = np.vstack(
        [np.zeros((1, d), dtype=np.float64), np.cumsum(x * x, axis=0)]
    )
    positions = np.arange(n, dtype=np.int64)
    previous = x[np.maximum(positions - 1, 0)]
    delta = x - previous
    matrix_parts: list[np.ndarray] = [x, delta, np.abs(delta)]
    scalar_columns: list[np.ndarray] = []

    for window in wins:
        left_widths = np.minimum(positions, window)
        right_widths = np.minimum(n - positions, window)
        left_starts = positions - left_widths
        right_ends = positions + right_widths
        safe_left_counts = np.maximum(left_widths, 1)[:, None]
        safe_right_counts = np.maximum(right_widths, 1)[:, None]
        left_mean = (prefix[positions] - prefix[left_starts]) / safe_left_counts
        right_mean = (prefix[right_ends] - prefix[positions]) / safe_right_counts
        left_second = (
            prefix_sq[positions] - prefix_sq[left_starts]
        ) / safe_left_counts
        right_second = (
            prefix_sq[right_ends] - prefix_sq[positions]
        ) / safe_right_counts

        left_empty = left_widths == 0
        right_empty = right_widths == 0
        left_mean[left_empty] = x[left_empty]
        right_mean[right_empty] = x[right_empty]
        left_second[left_empty] = x[left_empty] * x[left_empty]
        right_second[right_empty] = x[right_empty] * x[right_empty]

        diff = right_mean - left_mean
        absdiff = np.abs(diff)
        left_std = np.sqrt(np.maximum(left_second - left_mean * left_mean, 0.0))
        right_std = np.sqrt(
            np.maximum(right_second - right_mean * right_mean, 0.0)
        )
        matrix_parts.extend([diff, absdiff])
        scalar_columns.extend(
            [
                np.mean(absdiff, axis=1),
                np.linalg.norm(diff / global_std, axis=1) / math.sqrt(d),
                np.mean(right_std - left_std, axis=1),
            ]
        )

    if scalar_columns:
        matrix_parts.append(np.column_stack(scalar_columns))
    if grid_context_arr.shape[1]:
        matrix_parts.append(grid_context_arr)
    if regional_context.shape[1]:
        matrix_parts.append(regional_context)
    standardized = np.nan_to_num(
        column_standardize(np.hstack(matrix_parts)), copy=False
    )
    coverage = boundary_context_coverage_features(n, wins, regional_windows)
    return (
        np.hstack([standardized, coverage])
        if coverage.shape[1]
        else standardized
    )


def segment_feature_matrix(
    feature_z: np.ndarray,
    bounds: np.ndarray,
    context_windows: Iterable[int],
) -> np.ndarray:
    beat_features = np.asarray(feature_z, dtype=np.float64)
    if beat_features.ndim != 2:
        raise ValueError("feature_z must be a 2-D matrix")
    n = beat_features.shape[0]
    rows: list[np.ndarray] = []
    for start_raw, end_raw in zip(bounds[:-1], bounds[1:]):
        start = int(np.clip(start_raw, 0, max(n - 1, 0)))
        end = int(np.clip(end_raw, start + 1, n))
        block = beat_features[start:end]
        head = block[: max(1, min(4, block.shape[0]))].mean(axis=0)
        tail = block[-max(1, min(4, block.shape[0])) :].mean(axis=0)
        length = float(end - start)
        rows.append(
            np.concatenate(
                [
                    block.mean(axis=0),
                    block.std(axis=0),
                    np.max(block, axis=0),
                    tail - head,
                    np.asarray(
                        [
                            math.log(max(length, 1.0)),
                            length / max(n, 1),
                            start / max(n, 1),
                            end / max(n, 1),
                            0.5 * (start + end) / max(n, 1),
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
        )
    base = np.nan_to_num(np.vstack(rows), copy=False)
    regional = segment_regional_context_features(
        beat_features, bounds, context_windows
    )
    return np.hstack([base, regional]) if regional.shape[1] else base


def valid_boundary_mask(n_beats: int, edge_beats: int) -> np.ndarray:
    n = int(n_beats)
    valid = np.ones(n, dtype=bool)
    edge = min(max(int(edge_beats), 0), n // 2)
    valid[:edge] = False
    if edge:
        valid[n - edge :] = False
    return valid


__all__ = [
    "boundary_feature_matrix",
    "column_standardize",
    "feature_z_from_acoustic",
    "grid_context_feature_matrix",
    "robust_standardize_rows",
    "segment_feature_matrix",
    "valid_boundary_mask",
]
