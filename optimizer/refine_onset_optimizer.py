"""Retrain the gradient-boosted onset used by the beat-grid refinement."""

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

from analyzer_core.beat.learned_onset import export_refine_onset_model, stack_context
from core.concurrency.process_pool import create_spawn_process_pool
from optimizer import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from optimizer.onset_parameter_optimizer import (
    _auto_cache_workers,
    _build_track_cache,
    _cache_path,
    _list_beat_tracks,
    _read_current_cache,
    beat_targets,
    samples_from_inputs,
)

ProgressCallback = Callable[[int, str], None]
MIN_TRACKS = 8
# Share of each sample kind kept for training (near a beat / beat subdivision / other),
# reweighted so the kept frames stand for all of them.
SAMPLE_SHARE = (0.3, 0.15, 0.1)


@dataclass(frozen=True)
class RefineOnsetOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    use_hpss: bool = True
    frame_hop_length: int = 256
    rebuild_cache: bool = False
    cv_folds: int = 0
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class RefineOnsetOptimizationResult:
    output_path: Path
    track_count: int
    sample_count: int
    beat_average_precision: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False


def _model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, random_state=seed,
    )


def _group_key(track: dict[str, object]) -> str:
    text = f"{track.get('artist', '')}|{track.get('title', '')}".casefold()
    return re.sub(r"[^0-9a-z぀-ヿ㐀-鿿]+", "", text) or str(track["uid"])


def _track_samples(frame_times: np.ndarray, channels: np.ndarray, beats: np.ndarray, seed: int):
    samples = samples_from_inputs(stack_context(channels).astype(np.float32), frame_times, beats)
    kind = samples["kind"]
    draw = np.random.default_rng(seed).random(kind.size)
    share = np.asarray(SAMPLE_SHARE)[kind]
    keep = draw < share
    weight = samples["w"][keep] / share[keep]
    return samples["x"][keep].astype(np.float16), (samples["y"][keep] >= 0.5).astype(np.int8), weight.astype(np.float32)


def _beat_ap(model, frame_times: np.ndarray, channels: np.ndarray, beats: np.ndarray) -> float:
    inside, target = beat_targets(frame_times, beats)
    activation = model.predict_proba(stack_context(channels).astype(np.float32)[inside])[:, 1]
    return float(average_precision_score(target >= 0.5, activation))


def optimize_refine_onset(
    request: RefineOnsetOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> RefineOnsetOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Grid refinement onset parameter path must end with .npz")
    cache_dir.mkdir(parents=True, exist_ok=True)
    hop = int(request.frame_hop_length)
    ignored: list[SkippedOptimizationTrack] = []
    tracks = _list_beat_tracks(library_dir, ignored)
    if len(tracks) < MIN_TRACKS:
        raise RuntimeError(f"At least {MIN_TRACKS} annotated tracks are required")

    jobs = [
        track for track in tracks
        if request.rebuild_cache
        or _read_current_cache(_cache_path(cache_dir, str(track["uid"])), track, bool(request.use_hpss), hop) is None
    ]
    workers = _auto_cache_workers(request.max_cache_workers, len(jobs))
    if jobs:
        with create_spawn_process_pool(workers) as executor:
            futures = {
                executor.submit(_build_track_cache, track, str(cache_dir), bool(request.use_hpss), hop): track
                for track in jobs
            }
            failed: set[str] = set()
            for done, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.add(str(track["uid"]))
                    ignored.append(skipped_track(track, f"feature cache build failed: {exc}"))
                if feature_progress:
                    feature_progress(int(round(80 * done / len(jobs))), f"Feature cache {done}/{len(jobs)}\n{optimizer_track_name(track)}")
        tracks = [track for track in tracks if str(track["uid"]) not in failed]

    data = []
    for index, track in enumerate(tracks, start=1):
        cached = _read_current_cache(_cache_path(cache_dir, str(track["uid"])), track, bool(request.use_hpss), hop)
        if cached is None:
            ignored.append(skipped_track(track, "feature cache is not current"))
            continue
        frame_times, channels = cached
        x, y, w = _track_samples(frame_times, channels, np.asarray(track["beats"]), index)
        data.append((track, x, y, w))
        if feature_progress:
            feature_progress(80 + int(round(20 * index / len(tracks))), f"Samples {index}/{len(tracks)}")
    if len(data) < MIN_TRACKS:
        raise RuntimeError("Insufficient tracks with current feature caches")

    cross_validated = int(request.cv_folds) >= 2
    ap_values: list[float] = []
    if cross_validated:
        groups = np.asarray([_group_key(track) for track, *_ in data])
        n_splits = min(int(request.cv_folds), np.unique(groups).size)
        for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(data, groups=groups), start=1):
            model = _model(fold).fit(
                np.concatenate([data[i][1] for i in train]).astype(np.float32),
                np.concatenate([data[i][2] for i in train]),
                sample_weight=np.concatenate([data[i][3] for i in train]),
            )
            for i in test:
                track = data[i][0]
                frame_times, channels = _read_current_cache(
                    _cache_path(cache_dir, str(track["uid"])), track, bool(request.use_hpss), hop
                )
                ap_values.append(_beat_ap(model, frame_times, channels, np.asarray(track["beats"])))
            if optimize_progress:
                optimize_progress(int(round(70 * fold / n_splits)), f"Cross-validation fold {fold}/{n_splits}")

    if optimize_progress:
        optimize_progress(75, "Fitting all tracks")
    x = np.concatenate([item[1] for item in data]).astype(np.float32)
    y = np.concatenate([item[2] for item in data])
    w = np.concatenate([item[3] for item in data])
    model = _model(0).fit(x, y, sample_weight=w)
    if not cross_validated:
        for track, *_ in data[: min(len(data), 40)]:
            frame_times, channels = _read_current_cache(
                _cache_path(cache_dir, str(track["uid"])), track, bool(request.use_hpss), hop
            )
            ap_values.append(_beat_ap(model, frame_times, channels, np.asarray(track["beats"])))
    export_refine_onset_model(
        model, output_path, hop,
        {"date": date.today().isoformat(), "tracks": len(data), "samples": int(x.shape[0]),
         "cross_validated": cross_validated},
    )
    if optimize_progress:
        optimize_progress(100, "Done")
    return RefineOnsetOptimizationResult(
        output_path=output_path,
        track_count=len(data),
        sample_count=int(x.shape[0]),
        beat_average_precision=float(np.mean(ap_values)) if ap_values else float("nan"),
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
