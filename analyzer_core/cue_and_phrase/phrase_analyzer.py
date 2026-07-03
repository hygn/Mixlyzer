"""Production beat-level two-stage GBM phrase detector."""

from __future__ import annotations

import math
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from analyzer_core.cue_and_phrase.structure import (
    FeatureConfig,
    build_predictor_grid,
    extract_song_features,
)


EPS = 1e-9
NPZ_FORMAT = "mixlyzer_phrase_weight_v1"


def _parse_feature_config(settings: dict[str, object]) -> FeatureConfig:
    sr = int(settings.get("sr", 22050))
    return FeatureConfig(
        sample_rate=sr,
        hop_length=int(settings.get("hop_length", 512)),
        n_fft=int(settings.get("n_fft", 2048)),
        n_mels=int(settings.get("n_mels", 48)),
        n_mfcc=int(settings.get("n_mfcc", 20)),
        fmax=0.5 * sr,
    )


class NumpyHistGradientBoostingClassifier:
    """Small NumPy runtime for sklearn HistGradientBoostingClassifier exports."""

    def __init__(
        self,
        *,
        classes: np.ndarray,
        baseline: np.ndarray,
        tree_classes: np.ndarray,
        tree_offsets: np.ndarray,
        node_value: np.ndarray,
        node_feature_idx: np.ndarray,
        node_num_threshold: np.ndarray,
        node_missing_go_to_left: np.ndarray,
        node_left: np.ndarray,
        node_right: np.ndarray,
        node_is_leaf: np.ndarray,
    ) -> None:
        self.classes_ = np.asarray(classes)
        self.baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
        self.tree_classes = np.asarray(tree_classes, dtype=np.int32).reshape(-1)
        self.tree_offsets = np.asarray(tree_offsets, dtype=np.int64).reshape(-1)
        self.node_value = np.asarray(node_value, dtype=np.float64).reshape(-1)
        self.node_feature_idx = np.asarray(node_feature_idx, dtype=np.int32).reshape(-1)
        self.node_num_threshold = np.asarray(node_num_threshold, dtype=np.float64).reshape(-1)
        self.node_missing_go_to_left = np.asarray(node_missing_go_to_left, dtype=bool).reshape(-1)
        self.node_left = np.asarray(node_left, dtype=np.int32).reshape(-1)
        self.node_right = np.asarray(node_right, dtype=np.int32).reshape(-1)
        self.node_is_leaf = np.asarray(node_is_leaf, dtype=bool).reshape(-1)

    def _predict_tree(self, X: np.ndarray, start: int, end: int) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        out = np.empty(x.shape[0], dtype=np.float64)
        value = self.node_value[start:end]
        feature_idx = self.node_feature_idx[start:end]
        threshold = self.node_num_threshold[start:end]
        missing_left = self.node_missing_go_to_left[start:end]
        left = self.node_left[start:end]
        right = self.node_right[start:end]
        is_leaf = self.node_is_leaf[start:end]
        for row in range(x.shape[0]):
            node = 0
            while not bool(is_leaf[node]):
                data_val = x[row, int(feature_idx[node])]
                if np.isnan(data_val):
                    node = int(left[node] if missing_left[node] else right[node])
                elif data_val <= threshold[node]:
                    node = int(left[node])
                else:
                    node = int(right[node])
            out[row] = value[node]
        return out

    def raw_predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        raw = np.tile(self.baseline.reshape(1, -1), (x.shape[0], 1)).astype(np.float64)
        for tree_index, tree_class in enumerate(self.tree_classes):
            start = int(self.tree_offsets[tree_index])
            end = int(self.tree_offsets[tree_index + 1])
            raw[:, int(tree_class)] += self._predict_tree(x, start, end)
        return raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_predict(X)
        if raw.shape[1] == 1:
            encoded = (raw.ravel() > 0.0).astype(np.int32)
        else:
            encoded = np.argmax(raw, axis=1).astype(np.int32)
        return self.classes_[encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.raw_predict(X)
        if raw.shape[1] == 1:
            p1 = 1.0 / (1.0 + np.exp(-np.clip(raw.ravel(), -50.0, 50.0)))
            return np.vstack([1.0 - p1, p1]).T
        shifted = raw - np.max(raw, axis=1, keepdims=True)
        exp = np.exp(np.clip(shifted, -50.0, 50.0))
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def _hgb_to_npz_payload(prefix: str, clf, payload: dict[str, np.ndarray]) -> None:
    values: list[np.ndarray] = []
    feature_idx: list[np.ndarray] = []
    thresholds: list[np.ndarray] = []
    missing_left: list[np.ndarray] = []
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    is_leaf: list[np.ndarray] = []
    tree_classes: list[int] = []
    offsets = [0]

    for predictors_at_iteration in clf._predictors:
        for class_index, predictor in enumerate(predictors_at_iteration):
            nodes = predictor.nodes
            if np.any(np.asarray(nodes["is_categorical"], dtype=bool)):
                raise ValueError("Categorical HistGradientBoosting splits are not supported.")
            values.append(np.asarray(nodes["value"], dtype=np.float64))
            feature_idx.append(np.asarray(nodes["feature_idx"], dtype=np.int32))
            thresholds.append(np.asarray(nodes["num_threshold"], dtype=np.float64))
            missing_left.append(np.asarray(nodes["missing_go_to_left"], dtype=np.uint8))
            left.append(np.asarray(nodes["left"], dtype=np.int32))
            right.append(np.asarray(nodes["right"], dtype=np.int32))
            is_leaf.append(np.asarray(nodes["is_leaf"], dtype=np.uint8))
            tree_classes.append(int(class_index))
            offsets.append(offsets[-1] + int(nodes.shape[0]))

    payload[f"{prefix}_classes"] = np.asarray(clf.classes_).astype(str)
    payload[f"{prefix}_baseline"] = np.asarray(clf._baseline_prediction, dtype=np.float64).reshape(-1)
    payload[f"{prefix}_tree_classes"] = np.asarray(tree_classes, dtype=np.int32)
    payload[f"{prefix}_tree_offsets"] = np.asarray(offsets, dtype=np.int64)
    payload[f"{prefix}_node_value"] = np.concatenate(values).astype(np.float64)
    payload[f"{prefix}_node_feature_idx"] = np.concatenate(feature_idx).astype(np.int32)
    payload[f"{prefix}_node_num_threshold"] = np.concatenate(thresholds).astype(np.float64)
    payload[f"{prefix}_node_missing_go_to_left"] = np.concatenate(missing_left).astype(np.uint8)
    payload[f"{prefix}_node_left"] = np.concatenate(left).astype(np.int32)
    payload[f"{prefix}_node_right"] = np.concatenate(right).astype(np.int32)
    payload[f"{prefix}_node_is_leaf"] = np.concatenate(is_leaf).astype(np.uint8)


def export_two_stage_npz_artifact(artifact: dict[str, object], output_path: str | Path) -> None:
    settings = dict(artifact.get("settings", {}))
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(NPZ_FORMAT),
        "feature_source": np.asarray(str(artifact.get("feature_source", ""))),
        "fill_policy": np.asarray(str(artifact.get("fill_policy", ""))),
        "long_segment_split": np.asarray(bool(artifact.get("long_segment_split", False))),
        "settings_json": np.asarray(json.dumps(settings, ensure_ascii=False, sort_keys=True)),
        "label_labels": np.asarray(list(artifact["label_labels"])).astype(str),
        "label_transition": np.asarray(artifact["label_transition"], dtype=np.float64),
        "label_length_mu": np.asarray(artifact["label_length_mu"], dtype=np.float64),
        "label_length_sigma": np.asarray(artifact["label_length_sigma"], dtype=np.float64),
    }
    _hgb_to_npz_payload("boundary", artifact["boundary_clf"], payload)
    _hgb_to_npz_payload("label", artifact["label_clf"], payload)
    out = Path(output_path)
    if out.suffix.lower() != ".npz":
        raise ValueError(f"Phrase artifact output must end with .npz: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)


def _load_numpy_hgb(archive: np.lib.npyio.NpzFile, prefix: str) -> NumpyHistGradientBoostingClassifier:
    return NumpyHistGradientBoostingClassifier(
        classes=np.asarray(archive[f"{prefix}_classes"]).astype(str),
        baseline=np.asarray(archive[f"{prefix}_baseline"], dtype=np.float64),
        tree_classes=np.asarray(archive[f"{prefix}_tree_classes"], dtype=np.int32),
        tree_offsets=np.asarray(archive[f"{prefix}_tree_offsets"], dtype=np.int64),
        node_value=np.asarray(archive[f"{prefix}_node_value"], dtype=np.float64),
        node_feature_idx=np.asarray(archive[f"{prefix}_node_feature_idx"], dtype=np.int32),
        node_num_threshold=np.asarray(archive[f"{prefix}_node_num_threshold"], dtype=np.float64),
        node_missing_go_to_left=np.asarray(archive[f"{prefix}_node_missing_go_to_left"], dtype=np.uint8),
        node_left=np.asarray(archive[f"{prefix}_node_left"], dtype=np.int32),
        node_right=np.asarray(archive[f"{prefix}_node_right"], dtype=np.int32),
        node_is_leaf=np.asarray(archive[f"{prefix}_node_is_leaf"], dtype=np.uint8),
    )


@lru_cache(maxsize=2)
def load_two_stage_model(path: str) -> dict[str, object]:
    model_path = Path(path)
    if model_path.suffix.lower() != ".npz":
        raise ValueError(f"Phrase model must be a NumPy .npz artifact, got: {path}")
    with np.load(model_path, allow_pickle=False) as archive:
        fmt = str(np.asarray(archive["format"]).item())
        if fmt != NPZ_FORMAT:
            raise ValueError(f"Unsupported phrase model artifact: {path}")
        settings = json.loads(str(np.asarray(archive["settings_json"]).item()))
        return {
            "format": fmt,
            "feature_source": str(np.asarray(archive["feature_source"]).item()),
            "fill_policy": str(np.asarray(archive["fill_policy"]).item()),
            "long_segment_split": bool(np.asarray(archive["long_segment_split"]).item()),
            "settings": settings,
            "boundary_clf": _load_numpy_hgb(archive, "boundary"),
            "label_clf": _load_numpy_hgb(archive, "label"),
            "label_labels": [str(x) for x in np.asarray(archive["label_labels"]).astype(str)],
            "label_transition": np.asarray(archive["label_transition"], dtype=np.float64),
            "label_length_mu": np.asarray(archive["label_length_mu"], dtype=np.float64),
            "label_length_sigma": np.asarray(archive["label_length_sigma"], dtype=np.float64),
        }


def _robust_standardize_rows(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return (x - med) / scale


def _column_standardize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return (x - med) / scale


def _feature_z_from_acoustic(acoustic) -> np.ndarray:
    rows = np.vstack(
        [
            np.asarray(acoustic.family_beat["timbre"], dtype=np.float64),
            np.asarray(acoustic.family_beat["harmony"], dtype=np.float64),
            np.asarray(acoustic.family_beat["rhythm"], dtype=np.float64),
            np.asarray(acoustic.family_beat["texture"], dtype=np.float64),
        ]
    )
    return _robust_standardize_rows(rows).T.astype(np.float64)


def _grid_context_feature_matrix(feature_z: np.ndarray, grid) -> np.ndarray:
    n = int(np.asarray(feature_z).shape[0])
    beat_in_bar = np.asarray(grid.beat_in_bar, dtype=np.float64).reshape(-1)
    bar_index = np.asarray(grid.bar_index_of_beat, dtype=np.float64).reshape(-1)
    downbeat = np.asarray(grid.downbeat_mask, dtype=bool).reshape(-1)
    if beat_in_bar.size != n or bar_index.size != n or downbeat.size != n:
        return np.zeros((n, 0), dtype=np.float64)

    meters = np.asarray(grid.bar_meters, dtype=np.float64).reshape(-1)
    beat_meters = np.full(n, 4.0, dtype=np.float64)
    valid_bar = (bar_index >= 0) & (bar_index < meters.size)
    if meters.size:
        beat_meters[valid_bar] = np.maximum(meters[bar_index[valid_bar].astype(int)], 1.0)
    phase = np.mod(beat_in_bar, beat_meters) / np.maximum(beat_meters, 1.0)

    downbeat_idx = np.flatnonzero(downbeat)
    prev_dist = np.full(n, n, dtype=np.float64)
    next_dist = np.full(n, n, dtype=np.float64)
    if downbeat_idx.size:
        pos = np.searchsorted(downbeat_idx, np.arange(n), side="right") - 1
        ok = pos >= 0
        prev_dist[ok] = np.arange(n, dtype=np.float64)[ok] - downbeat_idx[pos[ok]]
        pos_next = np.searchsorted(downbeat_idx, np.arange(n), side="left")
        ok_next = pos_next < downbeat_idx.size
        next_dist[ok_next] = downbeat_idx[pos_next[ok_next]] - np.arange(n, dtype=np.float64)[ok_next]
    dist_to_downbeat = np.minimum(prev_dist, next_dist) / np.maximum(beat_meters, 1.0)

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
    return _column_standardize(np.vstack(rows).T)


def _boundary_feature_matrix(
    feature_z: np.ndarray,
    windows: Iterable[int],
    grid_context: np.ndarray | None,
) -> np.ndarray:
    x = np.asarray(feature_z, dtype=np.float64)
    n, d = x.shape
    wins = tuple(sorted({max(1, int(w)) for w in windows}))
    grid_context_arr = np.asarray(grid_context, dtype=np.float64) if grid_context is not None else None
    if grid_context_arr is not None and grid_context_arr.shape[0] != n:
        grid_context_arr = None
    rows: list[np.ndarray] = []
    global_std = np.std(x, axis=0) + EPS
    for b in range(n):
        parts = [x[b]]
        prev_beat = x[max(0, b - 1)]
        next_beat = x[min(n - 1, b)]
        parts.extend([next_beat - prev_beat, np.abs(next_beat - prev_beat)])
        scalar_parts: list[float] = []
        for w in wins:
            left = x[max(0, b - w):b]
            right = x[b:min(n, b + w)]
            if left.size == 0:
                left = x[b:b + 1]
            if right.size == 0:
                right = x[max(0, b - 1):b]
            lm = left.mean(axis=0)
            rm = right.mean(axis=0)
            diff = rm - lm
            absdiff = np.abs(diff)
            parts.extend([diff, absdiff])
            scalar_parts.extend(
                [
                    float(np.mean(absdiff)),
                    float(np.linalg.norm(diff / global_std) / math.sqrt(d)),
                    float(np.mean(right.std(axis=0) - left.std(axis=0))),
                ]
            )
        parts.append(np.asarray(scalar_parts, dtype=np.float64))
        if grid_context_arr is not None and grid_context_arr.size:
            parts.append(grid_context_arr[b])
        rows.append(np.concatenate(parts))
    return np.nan_to_num(_column_standardize(np.vstack(rows)), copy=False)


def _valid_mask(n: int, edge_beats: int) -> np.ndarray:
    valid = np.ones(int(n), dtype=bool)
    edge = min(max(int(edge_beats), 0), int(n) // 2)
    valid[:edge] = False
    if edge:
        valid[int(n) - edge:] = False
    return valid


def _predict_boundary_probability(clf, boundary_feature_z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p = np.clip(clf.predict_proba(np.asarray(boundary_feature_z, dtype=np.float64))[:, 1], 1e-6, 1.0 - 1e-6)
    p = np.where(np.asarray(valid, dtype=bool), p, 0.0)
    if p.size:
        p[0] = 0.0
        p[-1] = 0.0
    return p


def _pick_boundaries_direct(
    clf,
    boundary_feature_z: np.ndarray,
    valid: np.ndarray,
    probability: np.ndarray,
    *,
    min_distance_beats: int,
    max_boundaries: int | None,
) -> np.ndarray:
    hard = np.asarray(clf.predict(np.asarray(boundary_feature_z, dtype=np.float64)), dtype=int) > 0
    hard &= np.asarray(valid, dtype=bool)
    if hard.size:
        hard[0] = False
        hard[-1] = False
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(hard):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, hard.size))
    candidates = [max(range(s, e), key=lambda j: float(probability[j])) for s, e in runs]
    selected: list[int] = []
    for beat in sorted(candidates, key=lambda i: float(probability[i]), reverse=True):
        if all(abs(beat - prev) >= int(min_distance_beats) for prev in selected):
            selected.append(int(beat))
            if max_boundaries is not None and len(selected) >= int(max_boundaries):
                break
    selected.sort()
    return np.asarray([0, *selected, hard.size], dtype=np.int32)


def _pick_boundaries_probability(
    probability: np.ndarray,
    *,
    threshold: float,
    min_distance_beats: int,
    max_boundaries: int | None,
) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    n = int(p.size)
    candidates = [
        i for i in range(1, n - 1)
        if p[i] >= float(threshold) and p[i] >= p[i - 1] and p[i] >= p[i + 1]
    ]
    selected: list[int] = []
    for beat in sorted(candidates, key=lambda i: float(p[i]), reverse=True):
        if all(abs(beat - prev) >= int(min_distance_beats) for prev in selected):
            selected.append(int(beat))
            if max_boundaries is not None and len(selected) >= int(max_boundaries):
                break
    selected.sort()
    return np.asarray([0, *selected, n], dtype=np.int32)


def _boundary_length_prior(length: int, target_lengths: Iterable[int]) -> float:
    length = max(int(length), 1)
    targets = np.asarray([max(int(x), 1) for x in target_lengths], dtype=np.float64)
    if targets.size == 0:
        return 0.0
    z = np.log(float(length)) - np.log(targets)
    return -0.5 * float(np.min(z * z))


def _refine_boundaries(
    raw_bounds: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    downbeat_mask: np.ndarray,
    *,
    window_beats: int,
    target_lengths_beats: Iterable[int],
    length_weight: float,
    shift_penalty: float,
    downbeat_bonus: float,
) -> np.ndarray:
    raw = np.asarray(raw_bounds, dtype=np.int32).reshape(-1)
    if raw.size <= 2 or int(window_beats) <= 0:
        return raw
    n = int(raw[-1])
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    downbeat = np.asarray(downbeat_mask, dtype=bool).reshape(-1)
    if p.size != n or valid.size != n:
        return raw
    if downbeat.size != n:
        downbeat = np.zeros(n, dtype=bool)

    candidate_sets: list[np.ndarray] = []
    emissions: list[np.ndarray] = []
    for boundary in raw[1:-1]:
        center = int(boundary)
        lo = max(1, center - int(window_beats))
        hi = min(n - 1, center + int(window_beats))
        candidates = np.arange(lo, hi + 1, dtype=np.int32) if hi >= lo else np.asarray([center], dtype=np.int32)
        candidates = candidates[valid[candidates]]
        if candidates.size == 0:
            candidates = np.asarray([center], dtype=np.int32)
        candidate_sets.append(candidates)
        score = np.log(np.clip(p[candidates], 1e-7, 1.0))
        score -= float(shift_penalty) * np.abs(candidates.astype(np.float64) - float(center))
        score += float(downbeat_bonus) * downbeat[candidates].astype(np.float64)
        emissions.append(score.astype(np.float64))

    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []
    for i, candidates in enumerate(candidate_sets):
        scores = np.full(candidates.size, -1e18, dtype=np.float64)
        prev_choice = np.full(candidates.size, -1, dtype=np.int32)
        if i == 0:
            for j, beat in enumerate(candidates):
                scores[j] = emissions[i][j] + float(length_weight) * _boundary_length_prior(
                    int(beat), target_lengths_beats
                )
        else:
            prev_candidates = candidate_sets[i - 1]
            prev_scores = dp[i - 1]
            for j, beat in enumerate(candidates):
                lengths = beat - prev_candidates
                ok = lengths > 0
                if not np.any(ok):
                    continue
                trans = np.asarray(
                    [
                        float(length_weight) * _boundary_length_prior(int(length), target_lengths_beats)
                        for length in lengths[ok]
                    ],
                    dtype=np.float64,
                )
                values = prev_scores[ok] + trans
                best_local = int(np.argmax(values))
                source_indices = np.flatnonzero(ok)
                prev_choice[j] = int(source_indices[best_local])
                scores[j] = emissions[i][j] + float(values[best_local])
        dp.append(scores)
        back.append(prev_choice)

    last_candidates = candidate_sets[-1]
    final_lengths = n - last_candidates
    ok = final_lengths > 0
    if not np.any(ok):
        return raw
    final_values = dp[-1][ok] + np.asarray(
        [
            float(length_weight) * _boundary_length_prior(int(length), target_lengths_beats)
            for length in final_lengths[ok]
        ],
        dtype=np.float64,
    )
    source_indices = np.flatnonzero(ok)
    choice = int(source_indices[int(np.argmax(final_values))])
    out = [int(last_candidates[choice])]
    for i in range(len(candidate_sets) - 1, 0, -1):
        choice = int(back[i][choice])
        if choice < 0:
            return raw
        out.append(int(candidate_sets[i - 1][choice]))
    out.reverse()
    refined = np.asarray([0, *out, n], dtype=np.int32)
    if np.any(np.diff(refined) <= 0):
        return raw
    return refined


def _segment_features(feature_z: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    beat_features = np.asarray(feature_z, dtype=np.float64)
    n = beat_features.shape[0]
    rows: list[np.ndarray] = []
    for s0, e0 in zip(bounds[:-1], bounds[1:]):
        s = int(np.clip(s0, 0, max(n - 1, 0)))
        e = int(np.clip(e0, s + 1, n))
        block = beat_features[s:e]
        head = block[:max(1, min(4, block.shape[0]))].mean(axis=0)
        tail = block[-max(1, min(4, block.shape[0])):].mean(axis=0)
        length = float(e - s)
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
                            s / max(n, 1),
                            e / max(n, 1),
                            0.5 * (s + e) / max(n, 1),
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
        )
    return np.nan_to_num(np.vstack(rows), copy=False)


def _label_logp(label_clf, feature_z: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    X = _segment_features(feature_z, bounds)
    return np.log(np.clip(label_clf.predict_proba(X), 1e-7, 1.0))


def _length_ll(length_mu: np.ndarray, length_sigma: np.ndarray, length: int) -> np.ndarray:
    sigma = np.maximum(np.asarray(length_sigma, dtype=np.float64), 0.25)
    z = (math.log(max(int(length), 1)) - np.asarray(length_mu, dtype=np.float64)) / sigma
    return -0.5 * z * z - np.log(sigma)


def _decode_labels(
    labels: list[str],
    transition: np.ndarray,
    length_mu: np.ndarray,
    length_sigma: np.ndarray,
    logp: np.ndarray,
    bounds: np.ndarray,
    *,
    label_weight: float,
    transition_weight: float,
    length_weight: float,
) -> list[str]:
    m, n_labels = logp.shape
    if m == 0:
        return []
    trans = np.asarray(transition, dtype=np.float64)
    start_state, end_state = n_labels, n_labels + 1
    emit = float(label_weight) * logp
    if length_weight:
        lengths = np.diff(np.asarray(bounds, dtype=np.int32))
        emit = emit + float(length_weight) * np.vstack(
            [_length_ll(length_mu, length_sigma, int(length)) for length in lengths]
        )
    dp = np.full((m, n_labels), -1e18, dtype=np.float64)
    back = np.full((m, n_labels), -1, dtype=np.int32)
    dp[0] = emit[0] + float(transition_weight) * trans[start_state, :n_labels]
    for i in range(1, m):
        scores = dp[i - 1][:, np.newaxis] + float(transition_weight) * trans[:n_labels, :n_labels]
        back[i] = np.argmax(scores, axis=0)
        dp[i] = emit[i] + scores[back[i], np.arange(n_labels)]
    final = dp[-1] + float(transition_weight) * trans[:n_labels, end_state]
    label = int(np.argmax(final))
    out = [label]
    for i in range(m - 1, 0, -1):
        label = int(back[i, label])
        out.append(label)
    out.reverse()
    return [str(labels[i]) for i in out]


def detect_two_stage_phrase_segments(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    tempo_segments: np.ndarray,
    *,
    model_path: str | Path,
) -> list[dict[str, object]]:
    artifact = load_two_stage_model(str(Path(model_path).resolve()))
    settings = dict(artifact.get("settings", {}))
    grid = build_predictor_grid(beat_times_sec, tempo_segments)
    if grid.n_beats < 17:
        return []

    acoustic = extract_song_features(
        None,
        grid,
        _parse_feature_config(settings),
        audio_array=audio,
        audio_sr=int(sample_rate),
    )
    feature_z = _feature_z_from_acoustic(acoustic)
    boundary_feature_z = _boundary_feature_matrix(
        feature_z,
        settings.get("boundary_context_beats", (1, 2, 4, 8, 16)),
        _grid_context_feature_matrix(feature_z, grid),
    )
    valid = _valid_mask(feature_z.shape[0], int(settings.get("edge_beats", 8)))
    boundary_clf = artifact["boundary_clf"]
    probability = _predict_boundary_probability(boundary_clf, boundary_feature_z, valid)
    bounds = _pick_boundaries_direct(
        boundary_clf,
        boundary_feature_z,
        valid,
        probability,
        min_distance_beats=int(settings.get("min_distance_beats", 16)),
        max_boundaries=settings.get("max_boundaries"),
    )
    bounds = _refine_boundaries(
        bounds,
        probability,
        valid,
        np.asarray(grid.downbeat_mask, dtype=bool),
        window_beats=int(settings.get("boundary_refine_window_beats", 8)),
        target_lengths_beats=settings.get("boundary_lengths_beats", (16, 32, 64, 128)),
        length_weight=float(settings.get("boundary_length_weight", 0.45)),
        shift_penalty=float(settings.get("boundary_shift_penalty", 0.015)),
        downbeat_bonus=float(settings.get("boundary_downbeat_bonus", 0.35)),
    )

    labels = _decode_labels(
        list(artifact["label_labels"]),
        np.asarray(artifact["label_transition"], dtype=np.float64),
        np.asarray(artifact["label_length_mu"], dtype=np.float64),
        np.asarray(artifact["label_length_sigma"], dtype=np.float64),
        _label_logp(artifact["label_clf"], feature_z, bounds),
        bounds,
        label_weight=float(settings.get("label_weight", 1.0)),
        transition_weight=float(settings.get("transition_weight", 1.0)),
        length_weight=float(settings.get("length_weight", 0.0)),
    )

    beat_times = np.asarray(grid.beat_times_sec, dtype=np.float64)
    duration = float(acoustic.audio_duration_sec)
    if beat_times.size >= 2:
        beat_tail = float(beat_times[-1] + np.median(np.diff(beat_times)))
        duration = max(duration, beat_tail)
    segments: list[dict[str, object]] = []
    for (s0, e0), label in zip(zip(bounds[:-1], bounds[1:]), labels):
        s = int(s0)
        e = int(e0)
        start = float(beat_times[s]) if s < beat_times.size else duration
        end = float(beat_times[e]) if e < beat_times.size else duration
        if end - start <= 1e-6:
            continue
        segments.append({"start": start, "end": end, "label": str(label)})
    return segments
