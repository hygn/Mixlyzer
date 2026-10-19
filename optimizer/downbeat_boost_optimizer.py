"""Retrain the gradient-boosted downbeat model on the library's beat grids.

Besides each track's own grid, the model also learns from the grid at double tempo
(a beat inserted halfway between the annotated ones), whose four-beat bars start on
the downbeat and on the half bar: the analyzer holds slow songs at double tempo when
their tempo lies below the BPM range, and a model that never saw such grids puts their
downbeats on the off-beats.
"""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from analyzer_core.beat.downbeat_offset import export_downbeat_boost_model, standardize_downbeat_features
from core.beat_geometry import downbeat_beat_indices
from core.concurrency.process_pool import create_spawn_process_pool
from optimizer import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from optimizer.downbeat_parameter_optimizer import (
    _auto_cache_workers,
    _build_track_cache,
    _cache_path,
    _list_4_4_tracks,
    _read_current_cache,
)

ProgressCallback = Callable[[int, str], None]
MIN_TRACKS = 8
# Sample weight of the double-tempo grids against the tracks' own grids.
DOUBLE_TEMPO_WEIGHT = 0.1


@dataclass(frozen=True)
class DownbeatBoostOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    use_hpss: bool = True
    rebuild_cache: bool = False
    cv_folds: int = 0
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class DownbeatBoostOptimizationResult:
    output_path: Path
    track_count: int
    beat_count: int
    average_precision: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=80,
        l2_regularization=1.0, random_state=seed,
    )


def _fit(model: HistGradientBoostingClassifier, items) -> HistGradientBoostingClassifier:
    x = np.concatenate([part for item in items for part in (item[1], item[3])])
    y = np.concatenate([part for item in items for part in (item[2], item[4])])
    weight = np.concatenate([
        np.full(part.shape[0], scale, dtype=np.float64)
        for item in items
        for part, scale in ((item[1], 1.0), (item[3], DOUBLE_TEMPO_WEIGHT))
    ])
    return model.fit(x, y, sample_weight=weight)


def _group_key(track: dict[str, object]) -> str:
    text = f"{track.get('artist', '')}|{track.get('title', '')}".casefold()
    return re.sub(r"[^0-9a-z぀-ヿ㐀-鿿]+", "", text) or str(track["uid"])


def _double_tempo_track(track: dict[str, object]) -> dict[str, object]:
    beats = np.asarray(track["beats"], dtype=np.float64)
    doubled = np.sort(np.concatenate([beats, 0.5 * (beats[1:] + beats[:-1])]))
    # Its own cache entry next to the track's (same audio, same validity checks).
    return {**track, "uid": f"{track['uid']}.double", "beats": doubled}


def _labels(beats: np.ndarray, track: dict[str, object]) -> np.ndarray:
    labels = np.zeros(beats.size, dtype=np.int8)
    downbeats = downbeat_beat_indices(beats, np.asarray(track["segments"], dtype=np.float64))
    labels[downbeats[downbeats < beats.size]] = 1
    return labels


def _double_tempo_labels(beats: np.ndarray, track: dict[str, object]) -> np.ndarray:
    """Bar starts of the double-tempo grid: the downbeats and the half bars of 4-beat bars."""
    reference = np.asarray(track["beats"], dtype=np.float64)
    downbeats = downbeat_beat_indices(reference, np.asarray(track["segments"], dtype=np.float64))
    starts: list[int] = []
    for first, nxt in zip(downbeats, np.concatenate([downbeats[1:], [reference.size]])):
        starts.append(int(first))
        if nxt - first == 4:
            starts.append(int(first) + 2)
    labels = np.zeros(beats.size, dtype=np.int8)
    if starts and beats.size:
        times = reference[np.asarray(starts)]
        labels[np.clip(np.searchsorted(beats, times - 1e-6), 0, beats.size - 1)] = 1
    return labels


def optimize_downbeat_boost(
    request: DownbeatBoostOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> DownbeatBoostOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Downbeat model path must end with .npz")
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_hpss = bool(request.use_hpss)
    ignored: list[SkippedOptimizationTrack] = []
    # Every meter: each beat is labelled downbeat or not from the stored segments.
    tracks = _list_4_4_tracks(library_dir, ignored, four_four_only=False)
    if len(tracks) < MIN_TRACKS:
        raise RuntimeError(f"At least {MIN_TRACKS} analyzed tracks are required")

    variants = [track for track in tracks] + [_double_tempo_track(track) for track in tracks]
    jobs = [
        track for track in variants
        if request.rebuild_cache
        or _read_current_cache(_cache_path(cache_dir, str(track["uid"])), track, use_hpss) is None
    ]
    if jobs:
        workers = _auto_cache_workers(request.max_cache_workers, len(jobs))
        failed: set[str] = set()
        with create_spawn_process_pool(workers) as executor:
            futures = {
                executor.submit(_build_track_cache, track, str(cache_dir), use_hpss): track for track in jobs
            }
            for done, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    uid = str(track["uid"]).removesuffix(".double")
                    if uid not in failed:
                        ignored.append(skipped_track(track, f"feature cache build failed: {exc}"))
                    failed.add(uid)
                if feature_progress:
                    feature_progress(
                        int(round(90 * done / len(jobs))),
                        f"Feature cache {done}/{len(jobs)}\n{optimizer_track_name(track)}",
                    )
        tracks = [track for track in tracks if str(track["uid"]) not in failed]

    # (track, own-grid features, labels, double-tempo features, labels)
    data = []
    for track in tracks:
        double = _double_tempo_track(track)
        cached = _read_current_cache(_cache_path(cache_dir, str(track["uid"])), track, use_hpss)
        cached_double = _read_current_cache(_cache_path(cache_dir, str(double["uid"])), double, use_hpss)
        if cached is None or cached_double is None:
            ignored.append(skipped_track(track, "feature cache is not current"))
            continue
        beats, features = cached
        double_beats, double_features = cached_double
        data.append((
            track,
            standardize_downbeat_features(features).astype(np.float32),
            _labels(beats, track),
            standardize_downbeat_features(double_features).astype(np.float32),
            _double_tempo_labels(double_beats, track),
        ))
    if len(data) < MIN_TRACKS:
        raise RuntimeError("Insufficient tracks with current feature caches")
    if feature_progress:
        feature_progress(100, f"Features of {len(data)} tracks")

    cross_validated = int(request.cv_folds) >= 2
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    if cross_validated:
        groups = np.asarray([_group_key(track) for track, *_ in data])
        n_splits = min(int(request.cv_folds), np.unique(groups).size)
        for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(data, groups=groups), start=1):
            model = _fit(_model(fold), [data[i] for i in train])
            for i in test:
                scores.append(model.predict_proba(data[i][1])[:, 1])
                targets.append(data[i][2])
            if optimize_progress:
                optimize_progress(int(round(70 * fold / n_splits)), f"Cross-validation fold {fold}/{n_splits}")

    if optimize_progress:
        optimize_progress(75, "Fitting all tracks")
    model = _fit(_model(0), data)
    beat_count = int(sum(item[1].shape[0] + item[3].shape[0] for item in data))
    if not cross_validated:
        scores = [model.predict_proba(item[1])[:, 1] for item in data]
        targets = [item[2] for item in data]
    export_downbeat_boost_model(
        model,
        output_path,
        {"date": date.today().isoformat(), "tracks": len(data), "beats": beat_count,
         "double_tempo_weight": DOUBLE_TEMPO_WEIGHT, "cross_validated": cross_validated},
    )
    if optimize_progress:
        optimize_progress(100, "Done")
    return DownbeatBoostOptimizationResult(
        output_path=output_path,
        track_count=len(data),
        beat_count=beat_count,
        average_precision=float(average_precision_score(np.concatenate(targets), np.concatenate(scores))),
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
