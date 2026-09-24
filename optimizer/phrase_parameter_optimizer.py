from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

from optimizer import (
    SkippedOptimizationTrack,
    optimizer_track_name,
    skipped_track,
)
from optimizer import phrase_backend as _phrase_backend
from analyzer_core.cue_and_phrase.phrase_analyzer import (
    export_two_stage_npz_artifact,
    load_two_stage_model,
)
from analyzer_core.cue_and_phrase.context_features import (
    CONTEXT_FEATURE_VERSION,
    DEFAULT_BOUNDARY_REGIONAL_CONTEXT_BEATS,
    DEFAULT_LABEL_CONTEXT_BEATS,
)
from analyzer_core.cue_and_phrase.model_features import (
    boundary_feature_matrix,
    grid_context_feature_matrix,
    segment_feature_matrix,
    valid_boundary_mask,
)
from analyzer_core.cue_and_phrase.structural_features import (
    MULTI_VIEW_STRUCTURE_FEATURE_DIM,
    STRUCTURE_FEATURE_VERSION,
    multi_view_structure_features,
)
from utils.atomic_io import atomic_output_path


ProgressCallback = Callable[[int, str], None]

BOUNDARY_MODE = "probability"
BOUNDARY_THRESHOLD = 0.70
BOUNDARY_CONTEXT_BEATS = (1, 2, 4, 8, 16)
BOUNDARY_REGIONAL_CONTEXT_BEATS = DEFAULT_BOUNDARY_REGIONAL_CONTEXT_BEATS
LABEL_CONTEXT_BEATS = DEFAULT_LABEL_CONTEXT_BEATS
LABEL_BOUNDARY_JITTER_VIEWS = 3
LABEL_BOUNDARY_JITTER_BEATS = 2
POSITIVE_RADIUS_BEATS = 0
NEGATIVE_GUARD_BEATS = 2
MIN_DISTANCE_BEATS = 16
MAX_BOUNDARIES = None
BOUNDARY_REFINE_WINDOW_BEATS = 2
BOUNDARY_LENGTHS_BEATS = (16, 32, 64, 128)
BOUNDARY_LENGTH_WEIGHT = 0.15
BOUNDARY_SHIFT_PENALTY = 0.015
BOUNDARY_DOWNBEAT_BONUS = 0.20
LABEL_WEIGHT = 1.0
TRANSITION_WEIGHT = 1.0
LENGTH_WEIGHT = 0.0
EDGE_BEATS = 8
DEFAULT_FEATURE_CONFIG = _phrase_backend.FeatureConfig()


@dataclass(frozen=True)
class PhraseOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    rebuild_cache: bool = False
    seed: int = 0
    # >= 2 adds a track-level cross-validation (models refit per fold) before the
    # final fit on all tracks; < 2 fits once on all tracks.
    cv_folds: int = 0
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class PhraseOptimizationResult:
    output_path: Path
    track_count: int
    feature_dim: int
    boundary_dim: int
    boundary_average_precision: float
    boundary_f1: float
    label_accuracy: float
    label_macro_f1: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    # False: the metrics are measured on the training tracks (in-sample).
    cross_validated: bool = False


def _training_metrics(
    prepared: list[dict[str, object]],
    targets: list[tuple[np.ndarray, list[str]]],
    boundary_clf,
    label_model,
) -> dict[str, float]:
    boundary_truth: list[np.ndarray] = []
    boundary_score: list[np.ndarray] = []
    label_truth: list[np.ndarray] = []
    label_predicted: list[np.ndarray] = []

    for track, (bounds, labels) in zip(prepared, targets, strict=True):
        features = np.asarray(track["boundary_feature_z"], dtype=np.float64)
        valid = np.asarray(track["valid_mask"], dtype=bool)
        truth = np.zeros(features.shape[0], dtype=bool)
        interior = np.asarray(bounds, dtype=np.int32)[1:-1]
        interior = interior[(interior >= 0) & (interior < truth.size)]
        truth[interior] = True
        probability = np.asarray(
            boundary_clf.predict_proba(features), dtype=np.float64
        )[:, 1]
        boundary_truth.append(truth[valid])
        boundary_score.append(probability[valid])

        label_features = segment_feature_matrix(
            np.asarray(track["feature_z"], dtype=np.float64),
            np.asarray(bounds, dtype=np.int32),
            track["label_context_beats"],
        )
        label_truth.append(np.asarray(labels, dtype=str))
        label_predicted.append(
            np.asarray(label_model.clf.predict(label_features), dtype=str)
        )

    all_boundary_truth = np.concatenate(boundary_truth)
    all_boundary_score = np.concatenate(boundary_score)
    all_label_truth = np.concatenate(label_truth)
    all_label_predicted = np.concatenate(label_predicted)
    return {
        "boundary_average_precision": float(
            average_precision_score(all_boundary_truth, all_boundary_score)
        ),
        "boundary_f1": float(
            f1_score(
                all_boundary_truth,
                all_boundary_score >= BOUNDARY_THRESHOLD,
                zero_division=0,
            )
        ),
        "label_accuracy": float(
            accuracy_score(all_label_truth, all_label_predicted)
        ),
        "label_macro_f1": float(
            f1_score(
                all_label_truth,
                all_label_predicted,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _fit_models(
    prepared: list[dict[str, object]],
    indices: np.ndarray,
    targets: list[tuple[np.ndarray, list[str]]],
    seed: int,
):
    boundary_clf = _phrase_backend._fit_boundary_gbm(
        prepared,
        indices,
        targets,
        positive_radius=POSITIVE_RADIUS_BEATS,
        negative_guard=NEGATIVE_GUARD_BEATS,
        seed=int(seed),
        learning_rate=0.05,
        max_iter=420,
        max_leaf_nodes=31,
        max_depth=3,
        min_samples_leaf=10,
        l2_regularization=2.0,
    )
    label_model = _phrase_backend._fit_label_model(
        prepared,
        indices,
        targets,
        seed=int(seed) + 100,
        boundary_jitter_views=LABEL_BOUNDARY_JITTER_VIEWS,
        boundary_jitter_beats=LABEL_BOUNDARY_JITTER_BEATS,
    )
    return boundary_clf, label_model


def _cross_validate(
    prepared: list[dict[str, object]],
    targets: list[tuple[np.ndarray, list[str]]],
    folds: int,
    seed: int,
    progress: ProgressCallback | None,
) -> dict[str, float]:
    """Track-level K-fold: refit both models per fold, score the held-out tracks (track-weighted mean)."""
    indices = np.arange(len(prepared), dtype=np.int32)
    n_splits = min(int(folds), len(prepared))
    totals: dict[str, float] = {}
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(indices, groups=indices), start=1
    ):
        if progress is not None:
            progress(5 + int(round((fold - 1) * 45 / n_splits)), f"Cross-validation fold {fold}/{n_splits}")
        boundary_clf, label_model = _fit_models(prepared, train, targets, seed)
        fold_metrics = _training_metrics(
            [prepared[i] for i in test], [targets[i] for i in test], boundary_clf, label_model
        )
        for key, value in fold_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(test)
    return {key: value / len(prepared) for key, value in totals.items()}


def _structure_cache_path(cache_dir: Path, uid: str) -> Path:
    return Path(cache_dir) / f"{uid}.structure_v{STRUCTURE_FEATURE_VERSION}_multi.npy"


def _structure_cache_is_current(
    structure_path: Path,
    beat_cache_path: Path,
    n_beats: int,
) -> bool:
    if not structure_path.exists() or not beat_cache_path.exists():
        return False
    try:
        # A refreshed acoustic cache invalidates structural features even when
        # the number of beats happens to stay unchanged.
        if structure_path.stat().st_mtime_ns < beat_cache_path.stat().st_mtime_ns:
            return False
        cached = np.load(structure_path, mmap_mode="r", allow_pickle=False)
        return _structure_array_is_current(cached, n_beats)
    except Exception:
        return False


def _structure_array_is_current(values: np.ndarray, n_beats: int) -> bool:
    return bool(
        values.shape == (int(n_beats), MULTI_VIEW_STRUCTURE_FEATURE_DIM)
        and values.dtype == np.dtype(np.float32)
    )


def _load_current_structure_cache(
    structure_path: Path,
    beat_cache_path: Path,
    n_beats: int,
) -> np.ndarray:
    if structure_path.stat().st_mtime_ns < beat_cache_path.stat().st_mtime_ns:
        raise ValueError(f"Structural feature cache predates beat cache: {structure_path}")
    cached = np.load(structure_path, allow_pickle=False)
    expected = (int(n_beats), MULTI_VIEW_STRUCTURE_FEATURE_DIM)
    if not _structure_array_is_current(cached, n_beats):
        raise ValueError(
            f"Current structural feature cache must have shape {expected} and "
            f"dtype float32, got {cached.shape} and {cached.dtype}: {structure_path}"
        )
    return np.asarray(cached, dtype=np.float64)


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    with atomic_output_path(path) as temporary:
        np.save(temporary, values, allow_pickle=False)


def _build_track_caches(
    track: dict[str, object],
    cache_dir_text: str,
    config_values: dict[str, object],
    rebuild_beat_cache: bool,
) -> tuple[str, bool]:
    """Process-worker entry point; build only this track's missing caches."""

    cache_dir = Path(cache_dir_text)
    config = _phrase_backend.FeatureConfig(**config_values)

    # NumPy/SciPy dependencies may otherwise start a full BLAS thread pool in
    # every process, making parallel cache construction slower than serial.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - installed with scikit-learn
        threadpool_limits = None

    def build() -> tuple[str, bool]:
        feat = _phrase_backend._current_beat_features(
            track, cache_dir, config, bool(rebuild_beat_cache)
        )
        beat_was_current = bool(feat["cache_hit"])
        uid = str(track["uid"])
        n = int(np.asarray(feat["feature_z"]).shape[0])
        structure_path = _structure_cache_path(cache_dir, uid)
        structural = multi_view_structure_features(
            np.asarray(feat["feature_z"], dtype=np.float64)
        )
        expected = (n, MULTI_VIEW_STRUCTURE_FEATURE_DIM)
        if structural.shape != expected:
            raise ValueError(
                f"Current structural feature extractor returned {structural.shape}, "
                f"expected {expected}"
            )
        _write_npy_atomic(structure_path, structural.astype(np.float32))
        return uid, beat_was_current

    if threadpool_limits is None:
        return build()
    with threadpool_limits(limits=1):
        return build()


def _auto_cache_workers(requested: int | None, job_count: int) -> int:
    if job_count <= 0:
        return 0
    if requested is not None:
        return max(1, min(int(requested), job_count))
    cpu_count = os.cpu_count() or 1
    # Four concurrent songs gives useful throughput while bounding the large
    # STFT/HPSS working sets created for uncached audio.
    return min(job_count, 4, max(1, cpu_count // 2))


def optimize_phrase_parameters(
    request: PhraseOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> PhraseOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir)
    output_path = Path(request.output_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ignored: list[SkippedOptimizationTrack] = []
    candidates = _phrase_backend._list_annotated_tracks(library_dir, ignored)
    tracks_src: list[dict[str, object]] = []
    for track in candidates:
        try:
            grid = _phrase_backend.load_predictor_grid(track["analysis_path"])
            validation_track = {
                **track,
                "beat_times_sec": np.asarray(grid.beat_times_sec, dtype=np.float64),
                "predictor_grid": grid,
            }
            validation_track["optimizer_target"] = _phrase_backend._eval_target(
                validation_track
            )
            tracks_src.append(validation_track)
        except Exception as exc:
            ignored.append(
                skipped_track(track, f"Phrase annotation validation failed: {exc}")
            )
    if len(tracks_src) < 5:
        raise RuntimeError(
            "Need at least five valid annotated tracks to train a Phrase parameter "
            f"artifact ({len(tracks_src)} valid, {len(ignored)} ignored)."
        )

    config_values: dict[str, object] = asdict(DEFAULT_FEATURE_CONFIG)
    config = _phrase_backend.FeatureConfig(**config_values)

    prepared: list[dict[str, object]] = []
    total = len(tracks_src)
    cache_jobs: list[dict[str, object]] = []
    if feature_progress is not None:
        feature_progress(0, f"Checking feature cache: {total} tracks")
    for track in tracks_src:
        uid = str(track["uid"])
        beat_cache_path = cache_dir / f"{uid}.npz"
        beat_current = bool(
            not request.rebuild_cache
            and _phrase_backend._beat_feature_cache_is_current(
                track,
                cache_dir,
                config,
                grid=track["predictor_grid"],
            )
        )
        structure_current = False
        if beat_current:
            n_beats = int(track["predictor_grid"].n_beats)
            structure_current = _structure_cache_is_current(
                _structure_cache_path(cache_dir, uid), beat_cache_path, n_beats
            )
        if not beat_current or not structure_current:
            cache_jobs.append(
                {
                    "uid": uid,
                    "title": str(track["title"]),
                    "artist": str(track["artist"]),
                    "audio_path": Path(track["audio_path"]),
                    "analysis_path": Path(track["analysis_path"]),
                    "rebuild_beat_cache": not beat_current,
                }
            )

    cached_count = total - len(cache_jobs)
    workers = _auto_cache_workers(request.max_cache_workers, len(cache_jobs))
    if feature_progress is not None:
        feature_progress(
            int(round(cached_count * 75 / total)),
            (
                f"Building {len(cache_jobs)} missing feature caches "
                f"with {workers} workers"
                if cache_jobs
                else "Feature cache is current"
            ),
        )

    if cache_jobs:
        # Explicit spawn is safe when this function runs inside the settings
        # dialog's QThread and matches Windows/PyInstaller behavior.
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = {
                executor.submit(
                    _build_track_caches,
                    track,
                    str(cache_dir),
                    config_values,
                    bool(track["rebuild_beat_cache"]),
                ): track
                for track in cache_jobs
            }
            failed_uids: set[str] = set()
            for completed, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                uid = str(track["uid"])
                try:
                    _uid, beat_hit = future.result()
                except Exception as exc:
                    failed_uids.add(uid)
                    ignored.append(
                        skipped_track(track, f"Phrase feature cache build failed: {exc}")
                    )
                    beat_hit = False
                if feature_progress is not None:
                    finished = cached_count + completed
                    if uid in failed_uids:
                        source = "skipped"
                    elif not beat_hit:
                        source = "audio + structure extracted"
                    else:
                        source = "structure extracted"
                    feature_progress(
                        int(round(finished * 75 / total)),
                        f"{source} ({finished}/{total})\n"
                        f"{optimizer_track_name(track)}",
                    )
            tracks_src = [
                track for track in tracks_src if str(track["uid"]) not in failed_uids
            ]

    prepare_total = len(tracks_src)
    for index, track in enumerate(tracks_src, start=1):
        uid = str(track["uid"])
        if feature_progress is not None:
            feature_progress(
                75 + int(round((index - 1) * 25 / max(prepare_total, 1))),
                f"Preparing optimizer features {index}/{prepare_total}\n"
                f"{optimizer_track_name(track)}",
            )
        try:
            feat = _phrase_backend._current_beat_features(
                track,
                cache_dir,
                config,
                False,
                grid=track["predictor_grid"],
            )
            n = int(np.asarray(feat["feature_z"]).shape[0])
            valid = valid_boundary_mask(n, EDGE_BEATS)
            full_boundary_features = boundary_feature_matrix(
                np.asarray(feat["feature_z"], dtype=np.float64),
                BOUNDARY_CONTEXT_BEATS,
                grid_context_feature_matrix(feat["feature_z"], feat["grid"]),
                BOUNDARY_REGIONAL_CONTEXT_BEATS,
            )
            structure_cache = _structure_cache_path(cache_dir, uid)
            beat_cache = cache_dir / f"{uid}.npz"
            structural_features = _load_current_structure_cache(
                structure_cache, beat_cache, n
            )
            full_boundary_features = np.hstack(
                [full_boundary_features, structural_features]
            )
            prepared.append(
                {
                    "uid": uid,
                    "title": str(track["title"]),
                    "artist": str(track["artist"]),
                    "beat_times_sec": feat["beat_times_sec"],
                    "phrase_starts_sec": track["phrase_starts_sec"],
                    "phrase_ends_sec": track["phrase_ends_sec"],
                    "phrase_labels": track["phrase_labels"],
                    "feature_z": feat["feature_z"],
                    "boundary_feature_z": full_boundary_features,
                    "label_context_beats": LABEL_CONTEXT_BEATS,
                    "valid_mask": valid,
                    "optimizer_target": track["optimizer_target"],
                }
            )
        except Exception as exc:
            ignored.append(
                skipped_track(track, f"Phrase feature preparation failed: {exc}")
            )
        if feature_progress is not None:
            feature_progress(
                75 + int(round(index * 25 / max(prepare_total, 1))),
                f"prepared ({index}/{prepare_total})\n{optimizer_track_name(track)}",
            )

    if len(prepared) < 5:
        raise RuntimeError(
            "Need at least five tracks after Phrase feature validation "
            f"({len(prepared)} valid, {len(ignored)} ignored)."
        )
    feature_dim = int(np.asarray(prepared[0]["feature_z"]).shape[1])
    boundary_dim = int(np.asarray(prepared[0]["boundary_feature_z"]).shape[1])
    targets = [track["optimizer_target"] for track in prepared]
    idx = np.arange(len(prepared), dtype=np.int32)

    cross_validated = int(request.cv_folds) >= 2
    cv_metrics = (
        _cross_validate(prepared, targets, request.cv_folds, int(request.seed), optimize_progress)
        if cross_validated
        else None
    )
    if optimize_progress is not None:
        optimize_progress(55 if cross_validated else 5, "Fitting boundary and label GBMs")
    boundary_clf, label_model = _fit_models(prepared, idx, targets, int(request.seed))

    artifact = {
        "boundary_clf": boundary_clf,
        "label_clf": label_model.clf,
        "label_labels": label_model.labels,
        "label_transition": label_model.transition,
        "label_length_mu": label_model.length_mu,
        "label_length_sigma": label_model.length_sigma,
        "settings": {
            "context_feature_version": CONTEXT_FEATURE_VERSION,
            "boundary_mode": BOUNDARY_MODE,
            "threshold": BOUNDARY_THRESHOLD,
            "boundary_context_beats": list(BOUNDARY_CONTEXT_BEATS),
            "boundary_regional_context_beats": list(BOUNDARY_REGIONAL_CONTEXT_BEATS),
            "multi_view_structural_features": True,
            "structure_feature_version": STRUCTURE_FEATURE_VERSION,
            "label_context_beats": list(LABEL_CONTEXT_BEATS),
            "min_distance_beats": MIN_DISTANCE_BEATS,
            "max_boundaries": MAX_BOUNDARIES,
            "boundary_refine_window_beats": BOUNDARY_REFINE_WINDOW_BEATS,
            "boundary_lengths_beats": list(BOUNDARY_LENGTHS_BEATS),
            "boundary_length_weight": BOUNDARY_LENGTH_WEIGHT,
            "boundary_shift_penalty": BOUNDARY_SHIFT_PENALTY,
            "boundary_downbeat_bonus": BOUNDARY_DOWNBEAT_BONUS,
            "label_weight": LABEL_WEIGHT,
            "transition_weight": TRANSITION_WEIGHT,
            "length_weight": LENGTH_WEIGHT,
            "edge_beats": EDGE_BEATS,
            "sr": config.sample_rate,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "n_mels": config.n_mels,
            "n_mfcc": config.n_mfcc,
        },
    }

    if optimize_progress is not None:
        optimize_progress(87, "Evaluating fitted Phrase models")
    metrics = cv_metrics if cross_validated else _training_metrics(prepared, targets, boundary_clf, label_model)
    if optimize_progress is not None:
        optimize_progress(90, "Writing parameter artifact")
    export_two_stage_npz_artifact(artifact, output_path)
    load_two_stage_model.cache_clear()
    if optimize_progress is not None:
        optimize_progress(100, "Done")

    return PhraseOptimizationResult(
        output_path=output_path,
        track_count=len(prepared),
        feature_dim=feature_dim,
        boundary_dim=boundary_dim,
        boundary_average_precision=metrics["boundary_average_precision"],
        boundary_f1=metrics["boundary_f1"],
        label_accuracy=metrics["label_accuracy"],
        label_macro_f1=metrics["label_macro_f1"],
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
