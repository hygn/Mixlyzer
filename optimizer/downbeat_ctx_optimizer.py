"""Retrain the beat tracker's contextual downbeat model (a gradient-boosted classifier on the beat-level
downbeat features and their differences to the neighbouring beats) on the library's grids: the user's beats
and downbeats are the training data. With cross-validation, a group of songs is held out first: the model
trained on the others is checked end to end on them (optimizer/beat_grid_check.py) and the model is retrained
on every song and written only when those songs' grids are not worse."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from analyzer_core.beat.downbeat_offset import _downbeat_feature_matrix, _robust_standardize_columns
from analyzer_core.beat.general.bars import context, export_ctx_model
from core.beat_geometry import downbeat_beat_indices
from optimizer import SkippedOptimizationTrack, skipped_track
from optimizer.beat_frames import cache_path, ensure_frames, list_grid_tracks, read_frames
from optimizer.refine_onset_optimizer import _group_key
from optimizer import model_start as MS
from optimizer.beat_grid_check import GridCheck, ModelSet, grid_check

ProgressCallback = Callable[[int, str], None]
PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=1.0)


@dataclass(frozen=True)
class DownbeatCtxOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    rebuild_cache: bool = False
    cv_folds: int = 0
    max_cache_workers: int | None = None
    current_models: dict | None = None      # the analysis's model files in use (ModelSet fields)
    bpm_min: float = 100.0


@dataclass(frozen=True)
class DownbeatCtxOptimizationResult:
    output_path: Path
    track_count: int
    sample_count: int
    downbeat_average_precision: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False
    grid: GridCheck | None = None
    saved: bool = True


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, **PARAMS)


def track_samples(frames, beats: np.ndarray, segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(features, labels) at the annotated beats: what the tracker's bars read at its own beats."""
    beat_times, features = _downbeat_feature_matrix(frames, beats, beat_phase_bins=8, harmonic_phase_bins=8)
    x = context(_robust_standardize_columns(features)).astype(np.float32)
    y = np.zeros(x.shape[0], dtype=np.int8)
    down = downbeat_beat_indices(beat_times, segments)
    y[down[down < y.size]] = 1
    return x, y


def optimize_downbeat_ctx(
    request: DownbeatCtxOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> DownbeatCtxOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Downbeat model path must end with .npz")
    ignored: list[SkippedOptimizationTrack] = []
    tracks = list_grid_tracks(library_dir, ignored)
    MS.check_output(output_path)
    MS.require_tracks(len(tracks), "gbm")
    tracks = ensure_frames(tracks, cache_dir, ignored, rebuild=bool(request.rebuild_cache),
                           max_workers=request.max_cache_workers, progress=feature_progress, progress_span=(0, 85))

    data = []
    for index, track in enumerate(tracks, start=1):
        frames = read_frames(cache_path(cache_dir, str(track["uid"])), track)
        if frames is None:
            ignored.append(skipped_track(track, "frame feature cache is not current"))
            continue
        x, y = track_samples(frames, np.asarray(track["beats"]), np.asarray(track["segments"]))
        if y.sum() < 2:
            ignored.append(skipped_track(track, "fewer than two downbeats"))
            continue
        data.append((track, x, y))
        if feature_progress:
            feature_progress(85 + int(round(15 * index / len(tracks))), f"Samples {index}/{len(tracks)}")
    MS.require_tracks(len(data), "gbm")

    cross_validated = int(request.cv_folds) >= 2
    grid = None
    if cross_validated:
        groups = np.asarray([_group_key(track) for track, *_ in data])
        n_splits = min(int(request.cv_folds), np.unique(groups).size)
        train, held = next(GroupKFold(n_splits=n_splits).split(data, groups=groups))
        if optimize_progress:
            optimize_progress(10, "Training on the other songs")
        trial_model = _model(1).fit(np.concatenate([data[i][1] for i in train]), np.concatenate([data[i][2] for i in train]))
        trial = cache_dir / "_downbeat_ctx_held_out.npz"
        export_ctx_model(trial_model, trial, {})
        try:
            current = ModelSet(**(request.current_models or {}))
            grid = grid_check([data[i][0] for i in held], cache_dir, request.bpm_min, current,
                              current.replace(downbeat_ctx=trial), request.max_cache_workers, optimize_progress, (15, 70))
        finally:
            trial.unlink(missing_ok=True)
        if not grid.not_worse:
            if optimize_progress:
                optimize_progress(100, "Not saved: the held-out songs' grids got worse")
            return DownbeatCtxOptimizationResult(
                output_path=output_path, track_count=len(data), sample_count=0, downbeat_average_precision=float("nan"),
                skipped_tracks=tuple(ignored), cross_validated=True, grid=grid, saved=False,
            )

    if optimize_progress:
        optimize_progress(75, "Fitting all tracks")
    x = np.concatenate([item[1] for item in data])
    y = np.concatenate([item[2] for item in data])
    model = _model(0).fit(x, y)
    export_ctx_model(model, output_path, {
        "context_shifts": [-4, -3, -2, -1, 1, 2, 3, 4], "params": PARAMS, "date": date.today().isoformat(),
        "tracks": len(data), "samples": int(x.shape[0]), "cross_validated": cross_validated,
        "source": "the library's beat grids (Reoptimize)",
    })
    if optimize_progress:
        optimize_progress(100, "Done")
    return DownbeatCtxOptimizationResult(
        output_path=output_path,
        track_count=len(data),
        sample_count=int(x.shape[0]),
        downbeat_average_precision=float("nan"),
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
        grid=grid,
    )
