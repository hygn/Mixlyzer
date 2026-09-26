"""Production beat-level two-stage GBM phrase detector."""

from __future__ import annotations

import math
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from analyzer_core.hpss import HpssSpectra
from analyzer_core.cue_and_phrase.structure import (
    FeatureConfig,
    build_predictor_grid,
    extract_song_features,
)
from analyzer_core.cue_and_phrase.context_features import CONTEXT_FEATURE_VERSION
from analyzer_core.cue_and_phrase.model_features import (
    boundary_feature_matrix,
    feature_z_from_acoustic,
    grid_context_feature_matrix,
    segment_feature_matrix,
    valid_boundary_mask,
)
from analyzer_core.cue_and_phrase.structural_features import (
    STRUCTURE_FEATURE_VERSION,
    multi_view_structure_features,
)
from utils.atomic_io import atomic_output_path


EPS = 1e-9
NPZ_FORMAT = "mixlyzer_phrase_weight_v1"
_REQUIRED_SETTINGS = frozenset(
    {
        "sr",
        "hop_length",
        "n_fft",
        "n_mels",
        "n_mfcc",
        "context_feature_version",
        "structure_feature_version",
        "boundary_mode",
        "boundary_context_beats",
        "boundary_regional_context_beats",
        "multi_view_structural_features",
        "threshold",
        "min_distance_beats",
        "max_boundaries",
        "edge_beats",
        "boundary_refine_window_beats",
        "boundary_lengths_beats",
        "boundary_length_weight",
        "boundary_shift_penalty",
        "boundary_downbeat_bonus",
        "label_context_beats",
        "label_weight",
        "transition_weight",
        "length_weight",
    }
)


def _parse_feature_config(settings: dict[str, object]) -> FeatureConfig:
    sr = int(settings["sr"])
    return FeatureConfig(
        sample_rate=sr,
        hop_length=int(settings["hop_length"]),
        n_fft=int(settings["n_fft"]),
        n_mels=int(settings["n_mels"]),
        n_mfcc=int(settings["n_mfcc"]),
        fmax=0.5 * sr,
    )


def _validate_current_settings(settings: object, path: str) -> dict[str, object]:
    if not isinstance(settings, dict):
        raise ValueError(f"Phrase model settings must be a JSON object: {path}")
    missing = sorted(_REQUIRED_SETTINGS.difference(settings))
    if missing:
        raise ValueError(
            f"Phrase model is missing current settings {missing}: {path}"
        )
    if int(settings["context_feature_version"]) != CONTEXT_FEATURE_VERSION:
        raise ValueError(f"Unsupported Phrase context feature version: {path}")
    if int(settings["structure_feature_version"]) != STRUCTURE_FEATURE_VERSION:
        raise ValueError(f"Unsupported Phrase structure feature version: {path}")
    if str(settings["boundary_mode"]) != "probability":
        raise ValueError(f"Phrase model does not use the current probability boundary mode: {path}")
    if not bool(settings["multi_view_structural_features"]):
        raise ValueError(f"Phrase model does not use current multi-view structure features: {path}")
    return settings


class NumpyHistGradientBoostingClassifier:
    """Small NumPy runtime for sklearn HistGradientBoostingClassifier exports."""

    def __init__(
        self,
        *,
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
        # All rows descend one level per step until every row sits on a leaf.
        rows = np.arange(x.shape[0])
        node = np.zeros(x.shape[0], dtype=np.int64)
        active = ~is_leaf[node]
        while np.any(active):
            r, n = rows[active], node[active]
            data_val = x[r, feature_idx[n]]
            go_left = np.where(np.isnan(data_val), missing_left[n], data_val <= threshold[n])
            node[active] = np.where(go_left, left[n], right[n])
            active = ~is_leaf[node]
        out[:] = value[node]
        return out

    def raw_predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        raw = np.tile(self.baseline.reshape(1, -1), (x.shape[0], 1)).astype(np.float64)
        for tree_index, tree_class in enumerate(self.tree_classes):
            start = int(self.tree_offsets[tree_index])
            end = int(self.tree_offsets[tree_index + 1])
            raw[:, int(tree_class)] += self._predict_tree(x, start, end)
        return raw

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
    settings = dict(artifact["settings"])
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(NPZ_FORMAT),
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
    try:
        with atomic_output_path(out) as temp_path:
            np.savez_compressed(temp_path, **payload)
            # Validate the complete current artifact before replacing a working
            # model. This also catches truncated ZIP members and schema drift.
            load_two_stage_model.cache_clear()
            load_two_stage_model(str(temp_path))
    finally:
        load_two_stage_model.cache_clear()


def _load_numpy_hgb(archive: np.lib.npyio.NpzFile, prefix: str) -> NumpyHistGradientBoostingClassifier:
    return NumpyHistGradientBoostingClassifier(
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
        settings = _validate_current_settings(
            json.loads(str(np.asarray(archive["settings_json"]).item())),
            path,
        )
        boundary_classes = np.asarray(archive["boundary_classes"]).astype(str).tolist()
        label_classes = np.asarray(archive["label_classes"]).astype(str).tolist()
        label_labels = np.asarray(archive["label_labels"]).astype(str).tolist()
        if boundary_classes != ["0", "1"]:
            raise ValueError(f"Unsupported Phrase boundary classes: {path}")
        if label_classes != label_labels:
            raise ValueError(f"Phrase label class order does not match the current format: {path}")
        return {
            "settings": settings,
            "boundary_clf": _load_numpy_hgb(archive, "boundary"),
            "label_clf": _load_numpy_hgb(archive, "label"),
            "label_labels": label_labels,
            "label_transition": np.asarray(archive["label_transition"], dtype=np.float64),
            "label_length_mu": np.asarray(archive["label_length_mu"], dtype=np.float64),
            "label_length_sigma": np.asarray(archive["label_length_sigma"], dtype=np.float64),
        }


def _predict_boundary_probability(clf, boundary_feature_z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p = np.clip(clf.predict_proba(np.asarray(boundary_feature_z, dtype=np.float64))[:, 1], 1e-6, 1.0 - 1e-6)
    p = np.where(np.asarray(valid, dtype=bool), p, 0.0)
    if p.size:
        p[0] = 0.0
        p[-1] = 0.0
    return p


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


def _label_logp(
    label_clf,
    feature_z: np.ndarray,
    bounds: np.ndarray,
    context_windows: Iterable[int],
) -> np.ndarray:
    X = segment_feature_matrix(feature_z, bounds, context_windows)
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
    hpss: HpssSpectra | None = None,
) -> list[dict[str, object]]:
    artifact = load_two_stage_model(str(Path(model_path).resolve()))
    settings = artifact["settings"]
    grid = build_predictor_grid(beat_times_sec, tempo_segments)
    if grid.n_beats < 17:
        return []

    acoustic = extract_song_features(
        None,
        grid,
        _parse_feature_config(settings),
        audio_array=audio,
        audio_sr=int(sample_rate),
        hpss=hpss,
    )
    feature_z = feature_z_from_acoustic(acoustic)
    boundary_feature_z = boundary_feature_matrix(
        feature_z,
        settings["boundary_context_beats"],
        grid_context_feature_matrix(feature_z, grid),
        settings["boundary_regional_context_beats"],
    )
    structural = multi_view_structure_features(feature_z)
    boundary_feature_z = np.hstack([boundary_feature_z, structural])
    valid = valid_boundary_mask(feature_z.shape[0], int(settings["edge_beats"]))
    boundary_clf = artifact["boundary_clf"]
    probability = _predict_boundary_probability(boundary_clf, boundary_feature_z, valid)
    bounds = _pick_boundaries_probability(
        probability,
        threshold=float(settings["threshold"]),
        min_distance_beats=int(settings["min_distance_beats"]),
        max_boundaries=settings["max_boundaries"],
    )
    bounds = _refine_boundaries(
        bounds,
        probability,
        valid,
        np.asarray(grid.downbeat_mask, dtype=bool),
        window_beats=int(settings["boundary_refine_window_beats"]),
        target_lengths_beats=settings["boundary_lengths_beats"],
        length_weight=float(settings["boundary_length_weight"]),
        shift_penalty=float(settings["boundary_shift_penalty"]),
        downbeat_bonus=float(settings["boundary_downbeat_bonus"]),
    )

    labels = _decode_labels(
        list(artifact["label_labels"]),
        np.asarray(artifact["label_transition"], dtype=np.float64),
        np.asarray(artifact["label_length_mu"], dtype=np.float64),
        np.asarray(artifact["label_length_sigma"], dtype=np.float64),
        _label_logp(
            artifact["label_clf"],
            feature_z,
            bounds,
            settings["label_context_beats"],
        ),
        bounds,
        label_weight=float(settings["label_weight"]),
        transition_weight=float(settings["transition_weight"]),
        length_weight=float(settings["length_weight"]),
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
