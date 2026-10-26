from __future__ import annotations

from concurrent.futures import as_completed
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.training import (
    SkippedOptimizationTrack,
    optimizer_track_name,
    skipped_track,
)
from analyzer.runtime import spawn_pool
from analyzer.training import phrase_backend as _phrase_backend
from analyzer.phrase.phrase_analyzer import (
    export_two_stage_npz_artifact,
    load_two_stage_model,
    repeat_inputs,
)
from analyzer.phrase.refinement import REFINEMENT_VERSION
from analyzer.training import phrase_refinement as _refinement
from analyzer.training.library_tracks import Annotation
from analyzer.phrase.context_features import (
    CONTEXT_FEATURE_VERSION,
    DEFAULT_BOUNDARY_REGIONAL_CONTEXT_BEATS,
    DEFAULT_LABEL_CONTEXT_BEATS,
)
from analyzer.phrase.model_features import (
    boundary_feature_matrix,
    grid_context_feature_matrix,
    valid_boundary_mask,
)
from analyzer.phrase.structural_features import (
    MULTI_VIEW_STRUCTURE_FEATURE_DIM,
    STRUCTURE_FEATURE_VERSION,
    multi_view_structure_features,
)
from analyzer.files import atomic_output_path
from analyzer.training.text import Concat, Text, TrainingError, msg


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
    # Decoded pipeline (pass 1 + pass 2) at the exact beat: all boundaries,
    # boundaries between two different labels, and beat label accuracy.
    boundary_f1: float
    label_change_f1: float
    label_accuracy: float
    # Changes between two different key labels (INTRO / VERSE / CHORUS /
    # BREAK_CHORUS / INTERLUDE / OUTRO), the ones that matter for DJ use.
    key_change_f1: float = 0.0
    # Tracks whose LEAD_IN is exactly right (none in both, or the same end beat).
    lead_in_accuracy: float = 0.0
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    # False: the metrics are measured on the training tracks (in-sample).
    cross_validated: bool = False


def _model_settings(config) -> dict[str, object]:
    return {
        "context_feature_version": CONTEXT_FEATURE_VERSION,
        "boundary_mode": BOUNDARY_MODE,
        "threshold": BOUNDARY_THRESHOLD,
        "boundary_context_beats": list(BOUNDARY_CONTEXT_BEATS),
        "boundary_regional_context_beats": list(BOUNDARY_REGIONAL_CONTEXT_BEATS),
        "multi_view_structural_features": True,
        "structure_feature_version": STRUCTURE_FEATURE_VERSION,
        "refinement_version": REFINEMENT_VERSION,
        "label_change_weight": _refinement.LABEL_CHANGE_WEIGHT,
        "shift_offsets_beats": list(_refinement.SHIFT_OFFSETS_BEATS),
        "shift_labels": list(_refinement.KEY_LABELS),
        "shift_phrase_grid": True,
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
    }


def _scaled(progress: ProgressCallback | None, lo: int, hi: int, prefix: Text | str = "") -> ProgressCallback | None:
    if progress is None:
        return None
    return lambda pct, text: progress(lo + int(round(pct * (hi - lo) / 100)), Concat((prefix, text)))


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


def _fit_pipeline(
    prepared: list[dict[str, object]],
    indices: np.ndarray,
    targets: list[tuple[np.ndarray, list[str]]],
    settings: dict[str, object],
    seed: int,
    progress: ProgressCallback | None,
) -> dict[str, object]:
    return _refinement.fit_two_pass(
        prepared, indices, targets, settings, _fit_models, int(seed), progress
    )


def _cross_validate(
    prepared: list[dict[str, object]],
    targets: list[tuple[np.ndarray, list[str]]],
    settings: dict[str, object],
    folds: int,
    seed: int,
    progress: ProgressCallback | None,
) -> dict[str, float]:
    """Track-level K-fold: refit the whole pipeline per fold and decode the held-out tracks."""
    indices = np.arange(len(prepared), dtype=np.int32)
    n_splits = min(int(folds), len(prepared))
    predictions: list = [None] * len(prepared)
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(indices, groups=indices), start=1
    ):
        lo = 5 + int(round((fold - 1) * 60 / n_splits))
        hi = 5 + int(round(fold * 60 / n_splits))
        artifact = _fit_pipeline(
            prepared, train, targets, settings, seed,
            _scaled(progress, lo, hi, msg("Cross-validation fold {fold}/{folds}: ", fold=fold, folds=n_splits)),
        )
        for i in test:
            predictions[int(i)] = _refinement.predict_track(prepared[int(i)], artifact)
    return _refinement.pipeline_metrics(prepared, targets, predictions)


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


_REPEAT_KEYS = ("repetition", "partner_idx", "partner_w", "bars", "pair_X", "pair_a", "pair_b")


def _repeat_cache_path(cache_dir: Path, uid: str) -> Path:
    return Path(cache_dir) / f"{uid}.repeat_v{REFINEMENT_VERSION}.npz"


def _load_repeat_cache(path: Path, beat_cache_path: Path, n_beats: int) -> dict[str, np.ndarray]:
    """Cached refinement repetition inputs; raises when missing, stale or malformed."""
    if path.stat().st_mtime_ns < beat_cache_path.stat().st_mtime_ns:
        raise ValueError(f"Repetition cache predates beat cache: {path}")
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in _REPEAT_KEYS}
    if values["repetition"].shape[0] != int(n_beats) or values["partner_idx"].shape[0] != int(n_beats):
        raise ValueError(f"Repetition cache does not match {n_beats} beats: {path}")
    return values


def _repeat_cache_is_current(path: Path, beat_cache_path: Path, n_beats: int) -> bool:
    try:
        _load_repeat_cache(path, beat_cache_path, n_beats)
        return True
    except Exception:
        return False


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
        repeats = repeat_inputs(
            np.asarray(feat["feature_z"], dtype=np.float64),
            np.asarray(feat["grid"].downbeat_mask, dtype=bool),
        )
        with atomic_output_path(_repeat_cache_path(cache_dir, uid)) as temporary:
            np.savez(temporary, **repeats)
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
            grid = _phrase_backend.predictor_grid(track)
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
                skipped_track(track, msg("Phrase annotation validation failed: {error}", error=exc))
            )
    if len(tracks_src) < 5:
        raise TrainingError(msg("Need at least five valid annotated tracks to train a Phrase parameter "
            "artifact ({valid} valid, {ignored} ignored).", valid=len(tracks_src), ignored=len(ignored)))

    config_values: dict[str, object] = asdict(DEFAULT_FEATURE_CONFIG)
    config = _phrase_backend.FeatureConfig(**config_values)

    prepared: list[dict[str, object]] = []
    total = len(tracks_src)
    cache_jobs: list[dict[str, object]] = []
    if feature_progress is not None:
        feature_progress(0, msg("Checking feature cache: {count} tracks", count=total))
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
            ) and _repeat_cache_is_current(
                _repeat_cache_path(cache_dir, uid), beat_cache_path, n_beats
            )
        if not beat_current or not structure_current:
            cache_jobs.append(
                {
                    "uid": uid,
                    "title": str(track["title"]),
                    "artist": str(track["artist"]),
                    "audio_path": Path(track["audio_path"]),
                    "annotation": Annotation(beats=track["annotation"].beats,
                                             tempo_segments=track["annotation"].tempo_segments),
                    "rebuild_beat_cache": not beat_current,
                }
            )

    cached_count = total - len(cache_jobs)
    workers = _auto_cache_workers(request.max_cache_workers, len(cache_jobs))
    if feature_progress is not None:
        feature_progress(
            int(round(cached_count * 75 / total)),
            (
                msg("Building {count} missing feature caches with {workers} workers", count=len(cache_jobs), workers=workers)
                if cache_jobs
                else msg("Feature cache is current")
            ),
        )

    if cache_jobs:
        # Explicit spawn is safe when this function runs inside the settings
        # dialog's QThread and matches Windows/PyInstaller behavior.
        with spawn_pool(workers) as executor:
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
                        skipped_track(track, msg("Phrase feature cache build failed: {error}", error=exc))
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
                        (f"{source} ({finished}/{total})", optimizer_track_name(track)),
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
                (msg("Preparing optimizer features {done}/{total}", done=index, total=prepare_total), optimizer_track_name(track)),
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
            ).astype(np.float32)
            repeats = _load_repeat_cache(_repeat_cache_path(cache_dir, uid), beat_cache, n)
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
                    "downbeat": np.asarray(feat["grid"].downbeat_mask, dtype=bool),
                    "repeat_inputs": repeats,
                    "optimizer_target": track["optimizer_target"],
                }
            )
        except Exception as exc:
            ignored.append(
                skipped_track(track, msg("Phrase feature preparation failed: {error}", error=exc))
            )
        if feature_progress is not None:
            feature_progress(
                75 + int(round(index * 25 / max(prepare_total, 1))),
                (msg("prepared ({done}/{total})", done=index, total=prepare_total), optimizer_track_name(track)),
            )

    if len(prepared) < 5:
        raise TrainingError(msg("Need at least five tracks after Phrase feature validation "
            "({valid} valid, {ignored} ignored).", valid=len(prepared), ignored=len(ignored)))
    feature_dim = int(np.asarray(prepared[0]["feature_z"]).shape[1])
    boundary_dim = int(np.asarray(prepared[0]["boundary_feature_z"]).shape[1])
    targets = [track["optimizer_target"] for track in prepared]
    idx = np.arange(len(prepared), dtype=np.int32)

    settings = _model_settings(config)
    cross_validated = int(request.cv_folds) >= 2
    cv_metrics = (
        _cross_validate(prepared, targets, settings, request.cv_folds, int(request.seed), optimize_progress)
        if cross_validated
        else None
    )
    fit_lo = 65 if cross_validated else 5
    artifact = _fit_pipeline(
        prepared, idx, targets, settings, int(request.seed),
        _scaled(optimize_progress, fit_lo, 88),
    )

    if cross_validated:
        metrics = cv_metrics
    else:
        if optimize_progress is not None:
            optimize_progress(88, msg("Evaluating fitted Phrase models"))
        metrics = _refinement.pipeline_metrics(
            prepared, targets, [_refinement.predict_track(track, artifact) for track in prepared]
        )
    if optimize_progress is not None:
        optimize_progress(95, msg("Writing parameter artifact"))
    export_two_stage_npz_artifact(artifact, output_path)
    load_two_stage_model.cache_clear()
    if optimize_progress is not None:
        optimize_progress(100, msg("Done"))

    return PhraseOptimizationResult(
        output_path=output_path,
        track_count=len(prepared),
        feature_dim=feature_dim,
        boundary_dim=boundary_dim,
        boundary_f1=metrics["boundary_f1"],
        label_change_f1=metrics["label_change_f1"],
        label_accuracy=metrics["label_accuracy"],
        key_change_f1=metrics["key_change_f1"],
        lead_in_accuracy=metrics["lead_in_accuracy"],
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
