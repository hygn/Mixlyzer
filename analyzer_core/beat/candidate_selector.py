"""Classical-ML selection of static vs ordinary piecewise-tempo beat grids.

The tracker supplies four whole-track candidates in a fixed order.  A
Phrase-style presence classifier first decides whether a tempo change is
supported; only then does a regressor choose a dynamic window.  The bundled
model is trained for ordinary 4/4 static and phase-continuous piecewise-tempo
tracks.  Meter changes, phase resets, and continuous ramps are not modelled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from analyzer_core.beat.beat import (
    collapse_short_sandwiched_tempo_segments,
    rebuild_grid_from_segments,
)
from analyzer_core.beat.frame_features import FrameFeatures
from analyzer_core.beat.learned_onset import onset_channels
from core.resource_paths import resource_path
from utils.atomic_io import atomic_output_path


CANDIDATE_NAMES = ("dynamic_5s", "dynamic_10s", "dynamic_20s", "static")
STATIC_INDEX = CANDIDATE_NAMES.index("static")
TRIM_SEC = 5.0
MODEL_FORMAT = "mixlyzer_beat_candidate_selector_v1"
MODEL_PATH = resource_path("assets/weights/beat_candidate_selector.npz")
FEATURE_COUNT = 234
PRESENCE_FEATURE_COUNT = 1350


@dataclass(frozen=True)
class CandidateSelection:
    index: int
    name: str
    change_probability: float
    threshold: float
    predicted_f1: tuple[float, float, float]


def standardize_candidate_channels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    return np.clip((values - median) / scale, -10.0, 10.0)


def prepare_candidate_for_selection(
    candidate: dict[str, object], duration_sec: float, beatgrid_offset_sec: float
) -> dict[str, object]:
    """Apply the same cleanup/offset seen by training and final production output."""
    duration = float(duration_sec)
    segments = _segment_array(candidate)
    window_sec = float(candidate.get("window_s", 5.0) or 5.0)
    segments, removed = collapse_short_sandwiched_tempo_segments(
        segments, max_duration_sec=0.5 * window_sec
    )
    beats_value = candidate.get("beats_time", ())
    beats = np.asarray(beats_value, dtype=np.float64).reshape(-1)
    if removed:
        beats = rebuild_grid_from_segments(segments)
    offset = float(beatgrid_offset_sec)
    beats = beats + offset
    beats = beats[(beats >= 0.0) & (beats <= duration)]
    shifted = np.asarray(segments, dtype=np.float64).copy()
    if shifted.size:
        shifted[:, (0, 1, 3)] += offset
        shifted[:, 0] = np.clip(shifted[:, 0], 0.0, duration)
        shifted[:, 1] = np.clip(shifted[:, 1], 0.0, duration)
    return {
        "score": float(candidate["score"]),
        "segments": shifted,
        "beats_time": beats,
        "window_s": window_sec,
    }


def _segment_array(candidate: dict[str, object]) -> np.ndarray:
    value = candidate.get("tempo_segments", candidate.get("segments", ()))
    if isinstance(value, list) and (not value or isinstance(value[0], dict)):
        rows = []
        for segment in value:
            start = float(segment.get("segment_start", segment.get("start", 0.0)))
            end = float(segment.get("end", segment.get("segment_end", start)))
            bpm = float(segment.get("bpm", 0.0))
            inizio = float(segment.get("inizio", start))
            meter = float(segment.get("time_signature", 4.0) or 4.0)
            rows.append((start, end, bpm, inizio, meter))
        return (
            np.asarray(rows, dtype=np.float64)
            if rows
            else np.empty((0, 5), dtype=np.float64)
        )
    array = np.asarray(value, dtype=np.float64)
    return array if array.ndim == 2 else np.empty((0, 5), dtype=np.float64)


def _interp_channels(
    frame_times: np.ndarray, channels: np.ndarray, times: np.ndarray
) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    return np.stack(
        [np.interp(times, frame_times, channels[:, column]) for column in range(channels.shape[1])],
        axis=1,
    )


def _safe_stats(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(values.shape[1] * 4, dtype=np.float64)
    return np.concatenate(
        [
            np.mean(values, axis=0),
            np.std(values, axis=0),
            np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0),
        ]
    )


def _score_selected_index(scores: np.ndarray) -> int:
    """Match the production selector's last-candidate-wins tie behavior."""
    if scores.shape != (len(CANDIDATE_NAMES),) or not np.all(np.isfinite(scores)):
        raise ValueError("candidate scores must be four finite values")
    return int(np.flatnonzero(scores >= np.max(scores) - 1e-12)[-1])


def build_candidate_features(
    candidates: Sequence[dict[str, object]],
    frame_times: np.ndarray,
    standardized_channels: np.ndarray,
    duration_sec: float,
    bpm_bounds: tuple[float, float],
) -> np.ndarray:
    """Build the four phase-zero candidate rows used by the bundled models."""
    if len(candidates) != len(CANDIDATE_NAMES):
        raise ValueError(f"expected {len(CANDIDATE_NAMES)} candidates")
    frame_times = np.asarray(frame_times, dtype=np.float64)
    channels = np.asarray(standardized_channels, dtype=np.float64)
    if frame_times.ndim != 1 or channels.ndim != 2 or channels.shape[0] != frame_times.size:
        raise ValueError("frame times/channels shape mismatch")
    if frame_times.size == 0 or not np.all(np.isfinite(frame_times)):
        raise ValueError("candidate frame times must be non-empty and finite")
    if not np.all(np.diff(frame_times) >= 0.0) or not np.all(np.isfinite(channels)):
        raise ValueError("candidate channels must be finite on a sorted frame grid")
    if channels.shape[1] != 24:
        raise ValueError(f"expected 24 onset channels, got {channels.shape[1]}")

    segments = [_segment_array(candidate) for candidate in candidates]
    grids = [rebuild_grid_from_segments(value) for value in segments]
    scores = np.asarray([float(candidate["score"]) for candidate in candidates], dtype=np.float64)
    selected_source = _score_selected_index(scores)
    score_scale = max(float(np.std(scores)), 1e-6)
    score_z = np.clip((scores - float(np.mean(scores))) / score_scale, -5.0, 5.0)
    selected = grids[selected_source]
    selected = selected[(selected >= TRIM_SEC) & (selected <= duration_sec)]
    selected_period = (
        float(np.median(np.diff(selected)))
        if selected.size >= 2
        else 60.0 / max(float(np.mean(bpm_bounds)), 1e-6)
    )
    eye_source = np.eye(len(CANDIDATE_NAMES), dtype=np.float64)
    phase_zero = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    features = []

    for source, grid in enumerate(grids):
        beats = np.asarray(grid, dtype=np.float64)
        beats = beats[(beats >= TRIM_SEC) & (beats <= duration_sec)]
        if beats.size >= 2:
            intervals = np.diff(beats)
            starts = beats[:-1]
            phase_times = np.stack(
                [starts + fraction * intervals for fraction in (0.0, 0.25, 0.5, 0.75)],
                axis=1,
            )
            phase_values = np.stack(
                [
                    _interp_channels(frame_times, channels, phase_times[:, phase])
                    for phase in range(4)
                ],
                axis=1,
            )
            on = phase_values[:, 0]
            quarter = 0.5 * (phase_values[:, 1] + phase_values[:, 3])
            off = phase_values[:, 2]
            interval_log = np.log2(np.maximum(intervals, 1e-9) / selected_period)
            temporal = np.asarray(
                [
                    beats.size / max(duration_sec, 1e-6),
                    np.median(intervals) / selected_period,
                    np.mean(interval_log),
                    np.std(interval_log),
                    np.quantile(interval_log, 0.1),
                    np.quantile(interval_log, 0.9),
                    np.max(np.abs(np.diff(interval_log))) if interval_log.size >= 2 else 0.0,
                ],
                dtype=np.float64,
            )
            acoustic = np.concatenate(
                [
                    _safe_stats(on),
                    np.mean(on - off, axis=0),
                    np.std(on - off, axis=0),
                    np.mean(on - quarter, axis=0),
                    np.std(on - quarter, axis=0),
                    np.mean(np.abs(np.diff(on, axis=0)), axis=0)
                    if on.shape[0] >= 2
                    else np.zeros(on.shape[1]),
                ]
            )
        else:
            temporal = np.zeros(7, dtype=np.float64)
            acoustic = np.zeros(channels.shape[1] * 9, dtype=np.float64)
        metadata = np.concatenate(
            [
                eye_source[source],
                phase_zero,
                np.asarray(
                    [score_z[source], float(source == selected_source), 1.0],
                    dtype=np.float64,
                ),
            ]
        )
        features.append(np.concatenate([metadata, temporal, acoustic]))

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape != (len(CANDIDATE_NAMES), FEATURE_COUNT):
        raise RuntimeError(f"candidate feature schema mismatch: {matrix.shape}")
    return matrix


def build_presence_features(candidate_features: np.ndarray) -> np.ndarray:
    values = np.delete(np.asarray(candidate_features, dtype=np.float32), (*range(8), 10), axis=1)
    static = values[STATIC_INDEX]
    output = np.concatenate(
        [
            static,
            (values[:STATIC_INDEX] - static).reshape(-1),
            np.mean(values, axis=0),
            np.std(values, axis=0),
        ]
    ).astype(np.float32)
    if output.shape != (PRESENCE_FEATURE_COUNT,):
        raise RuntimeError(f"presence feature schema mismatch: {output.shape}")
    return output


class NumpyExtraTreesRegressor:
    """NumPy-only runtime for an exported sklearn ExtraTreesRegressor."""

    def __init__(
        self,
        *,
        feature_count: int,
        tree_offsets: np.ndarray,
        node_feature: np.ndarray,
        node_threshold: np.ndarray,
        node_left: np.ndarray,
        node_right: np.ndarray,
        node_value: np.ndarray,
    ) -> None:
        self.feature_count = int(feature_count)
        self.tree_offsets = np.asarray(tree_offsets, dtype=np.int64).reshape(-1)
        self.node_feature = np.asarray(node_feature, dtype=np.int32).reshape(-1)
        self.node_threshold = np.asarray(node_threshold, dtype=np.float64).reshape(-1)
        self.node_left = np.asarray(node_left, dtype=np.int32).reshape(-1)
        self.node_right = np.asarray(node_right, dtype=np.int32).reshape(-1)
        self.node_value = np.asarray(node_value, dtype=np.float64).reshape(-1)
        self._validate()

    def _validate(self) -> None:
        node_count = self.node_feature.size
        if self.feature_count <= 0 or self.tree_offsets.size < 2:
            raise ValueError("invalid ExtraTrees schema")
        if self.tree_offsets[0] != 0 or self.tree_offsets[-1] != node_count:
            raise ValueError("invalid ExtraTrees offsets")
        if np.any(np.diff(self.tree_offsets) <= 0):
            raise ValueError("empty ExtraTrees tree")
        for values in (self.node_threshold, self.node_left, self.node_right, self.node_value):
            if values.shape[0] != node_count:
                raise ValueError("ExtraTrees node array length mismatch")
        for tree in range(self.tree_offsets.size - 1):
            start = int(self.tree_offsets[tree])
            end = int(self.tree_offsets[tree + 1])
            count = end - start
            left = self.node_left[start:end]
            right = self.node_right[start:end]
            leaf = left < 0
            if np.any(leaf != (right < 0)):
                raise ValueError("ExtraTrees leaf child mismatch")
            if np.any((left[~leaf] >= count) | (right[~leaf] >= count)):
                raise ValueError("ExtraTrees child index out of range")
            feature = self.node_feature[start:end][~leaf]
            if np.any((feature < 0) | (feature >= self.feature_count)):
                raise ValueError("ExtraTrees feature index out of range")

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_count:
            raise ValueError(
                f"ExtraTrees expected (*, {self.feature_count}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("ExtraTrees inputs must be finite")
        return values

    def _leaf_nodes(self, X: np.ndarray, start: int, end: int) -> np.ndarray:
        feature = self.node_feature[start:end]
        threshold = self.node_threshold[start:end]
        left = self.node_left[start:end]
        right = self.node_right[start:end]
        rows = np.arange(X.shape[0])
        node = np.zeros(X.shape[0], dtype=np.int64)
        active = left[node] >= 0
        while np.any(active):
            row = rows[active]
            current = node[active]
            go_left = X[row, feature[current]] <= threshold[current]
            node[active] = np.where(go_left, left[current], right[current])
            active = left[node] >= 0
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = self._inputs(X)
        output = np.zeros(values.shape[0], dtype=np.float64)
        for tree in range(self.tree_offsets.size - 1):
            start = int(self.tree_offsets[tree])
            end = int(self.tree_offsets[tree + 1])
            output += self.node_value[start:end][self._leaf_nodes(values, start, end)]
        return output / float(self.tree_offsets.size - 1)


class NumpyExtraTreesClassifier(NumpyExtraTreesRegressor):
    """NumPy-only probability runtime for an exported ExtraTreesClassifier."""

    def __init__(self, *, classes: np.ndarray, node_value: np.ndarray, **kwargs) -> None:
        self.classes = np.asarray(classes).reshape(-1)
        values = np.asarray(node_value, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.classes.size:
            raise ValueError("ExtraTrees classifier value shape mismatch")
        # The base validates node count; retain the matrix after that validation.
        super().__init__(node_value=np.zeros(values.shape[0]), **kwargs)
        self.node_probability = values
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("ExtraTrees classifier probabilities are invalid")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-8, atol=1e-8):
            raise ValueError("ExtraTrees classifier probabilities are not normalized")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = self._inputs(X)
        output = np.zeros((values.shape[0], self.classes.size), dtype=np.float64)
        for tree in range(self.tree_offsets.size - 1):
            start = int(self.tree_offsets[tree])
            end = int(self.tree_offsets[tree + 1])
            output += self.node_probability[start:end][self._leaf_nodes(values, start, end)]
        return output / float(self.tree_offsets.size - 1)


def _extra_trees_payload(prefix: str, model, payload: dict[str, np.ndarray]) -> None:
    offsets = [0]
    feature: list[np.ndarray] = []
    threshold: list[np.ndarray] = []
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    values: list[np.ndarray] = []
    classifier = hasattr(model, "classes_")
    for estimator in model.estimators_:
        tree = estimator.tree_
        feature.append(np.asarray(tree.feature, dtype=np.int32))
        threshold.append(np.asarray(tree.threshold, dtype=np.float64))
        left.append(np.asarray(tree.children_left, dtype=np.int32))
        right.append(np.asarray(tree.children_right, dtype=np.int32))
        raw = np.asarray(tree.value[:, 0], dtype=np.float64)
        if classifier:
            raw /= np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
            values.append(raw)
        else:
            values.append(raw.reshape(-1))
        offsets.append(offsets[-1] + int(tree.node_count))
    payload[f"{prefix}_feature_count"] = np.int64(model.n_features_in_)
    payload[f"{prefix}_tree_offsets"] = np.asarray(offsets, dtype=np.int64)
    payload[f"{prefix}_node_feature"] = np.concatenate(feature)
    payload[f"{prefix}_node_threshold"] = np.concatenate(threshold)
    payload[f"{prefix}_node_left"] = np.concatenate(left)
    payload[f"{prefix}_node_right"] = np.concatenate(right)
    payload[f"{prefix}_node_value"] = np.concatenate(values, axis=0)
    if classifier:
        payload[f"{prefix}_classes"] = np.asarray(model.classes_)


def export_candidate_selector_npz_artifact(
    artifact: dict[str, object], output_path: str | Path
) -> None:
    """Export sklearn training models into the validated NumPy runtime format."""
    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError(f"Beat candidate selector output must end with .npz: {output}")
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(MODEL_FORMAT),
        "candidate_names": np.asarray(CANDIDATE_NAMES),
        "feature_count": np.int64(FEATURE_COUNT),
        "presence_feature_count": np.int64(PRESENCE_FEATURE_COUNT),
        "change_threshold": np.float64(artifact["change_threshold"]),
        "training_json": np.asarray(
            json.dumps(artifact.get("training", {}), ensure_ascii=False, sort_keys=True)
        ),
    }
    _extra_trees_payload("presence", artifact["presence_classifier"], payload)
    _extra_trees_payload("source", artifact["source_regressor"], payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with atomic_output_path(output) as temporary:
            np.savez_compressed(temporary, **payload)
            load_candidate_selector_model.cache_clear()
            load_candidate_selector_model(str(temporary))
    finally:
        load_candidate_selector_model.cache_clear()


def _forest_kwargs(archive: np.lib.npyio.NpzFile, prefix: str) -> dict[str, object]:
    return {
        "feature_count": int(np.asarray(archive[f"{prefix}_feature_count"]).item()),
        "tree_offsets": np.asarray(archive[f"{prefix}_tree_offsets"], dtype=np.int64),
        "node_feature": np.asarray(archive[f"{prefix}_node_feature"], dtype=np.int32),
        "node_threshold": np.asarray(archive[f"{prefix}_node_threshold"], dtype=np.float64),
        "node_left": np.asarray(archive[f"{prefix}_node_left"], dtype=np.int32),
        "node_right": np.asarray(archive[f"{prefix}_node_right"], dtype=np.int32),
        "node_value": np.asarray(archive[f"{prefix}_node_value"], dtype=np.float64),
    }


@lru_cache(maxsize=2)
def load_candidate_selector_model(path: str = str(MODEL_PATH)) -> dict[str, object]:
    model_path = Path(path)
    if model_path.suffix.lower() != ".npz":
        raise ValueError(f"Beat candidate selector must be a NumPy .npz: {path}")
    with np.load(model_path, allow_pickle=False) as archive:
        if str(np.asarray(archive["format"]).item()) != MODEL_FORMAT:
            raise ValueError(f"beat candidate selector format mismatch: {path}")
        if tuple(np.asarray(archive["candidate_names"]).astype(str)) != CANDIDATE_NAMES:
            raise ValueError(f"beat candidate selector candidate schema mismatch: {path}")
        if int(np.asarray(archive["feature_count"]).item()) != FEATURE_COUNT:
            raise ValueError(f"beat candidate selector feature schema mismatch: {path}")
        if int(np.asarray(archive["presence_feature_count"]).item()) != PRESENCE_FEATURE_COUNT:
            raise ValueError(f"beat candidate selector presence schema mismatch: {path}")
        threshold = float(np.asarray(archive["change_threshold"]).item())
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"beat candidate selector threshold is invalid: {path}")
        presence_kwargs = _forest_kwargs(archive, "presence")
        source_kwargs = _forest_kwargs(archive, "source")
        classifier = NumpyExtraTreesClassifier(
            classes=np.asarray(archive["presence_classes"]), **presence_kwargs
        )
        if tuple(int(value) for value in classifier.classes) != (0, 1):
            raise ValueError(f"beat candidate selector class schema mismatch: {path}")
        regressor = NumpyExtraTreesRegressor(**source_kwargs)
        if classifier.feature_count != PRESENCE_FEATURE_COUNT:
            raise ValueError(f"beat candidate selector classifier width mismatch: {path}")
        if regressor.feature_count != FEATURE_COUNT:
            raise ValueError(f"beat candidate selector regressor width mismatch: {path}")
        return {
            "change_threshold": threshold,
            "training": json.loads(str(np.asarray(archive["training_json"]).item())),
            "presence_classifier": classifier,
            "source_regressor": regressor,
        }


def select_candidate(
    candidates: Sequence[dict[str, object]],
    frames: FrameFeatures,
    duration_sec: float,
    bpm_bounds: tuple[float, float],
    model_path: str | Path = MODEL_PATH,
) -> CandidateSelection:
    """Select static or one dynamic candidate with the bundled two-stage model."""
    artifact = load_candidate_selector_model(str(Path(model_path).resolve()))
    # Training caches store onset_channels(frames); the experiment applies one
    # more robust per-track normalization when loading those caches.
    channels = standardize_candidate_channels(onset_channels(frames))
    features = build_candidate_features(
        candidates,
        frames.frame_times,
        channels,
        float(duration_sec),
        bpm_bounds,
    )
    presence = build_presence_features(features)
    classifier = artifact["presence_classifier"]
    regressor = artifact["source_regressor"]
    probability = float(classifier.predict_proba(presence[np.newaxis, :])[0, 1])
    threshold = float(artifact["change_threshold"])
    predicted = np.asarray(regressor.predict(features[:STATIC_INDEX]), dtype=np.float64)
    dynamic_index = int(np.argmax(predicted))
    index = dynamic_index if probability >= threshold else STATIC_INDEX
    return CandidateSelection(
        index=index,
        name=CANDIDATE_NAMES[index],
        change_probability=probability,
        threshold=threshold,
        predicted_f1=tuple(float(value) for value in predicted),
    )
