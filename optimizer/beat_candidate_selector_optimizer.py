"""Retrain the adaptive-window beat candidate selector from library annotations."""

from __future__ import annotations

from concurrent.futures import as_completed
import contextlib
from dataclasses import dataclass
from datetime import date
import io
from pathlib import Path
import re
from typing import Callable

import numpy as np
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from analyzer_core.beat.candidate_selector import (
    CANDIDATE_NAMES,
    STATIC_INDEX,
    build_candidate_features,
    build_presence_features,
    export_candidate_selector_npz_artifact,
    load_candidate_selector_model,
    prepare_candidate_for_selection,
    standardize_candidate_channels,
)
from analyzer_core.beat.learned_onset import _load_onset_model, stack_context
from core.concurrency.process_pool import create_spawn_process_pool
from optimizer import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from optimizer.onset_parameter_optimizer import (
    _auto_cache_workers,
    _build_track_cache,
    _cache_path,
    _list_beat_tracks,
    _read_current_cache,
)


ProgressCallback = Callable[[int, str], None]
PRESENCE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MIN_TRACKS = 8


@dataclass(frozen=True)
class BeatCandidateSelectorOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    onset_parameter_path: Path
    use_hpss: bool = True
    frame_hop_length: int = 256
    bpm_window_ms: int = 5000
    bpm_min: float = 100.0
    bpm_max: float = 200.0
    beatgrid_offset_msec: float = 0.0
    rebuild_cache: bool = False
    cv_folds: int = 5
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class BeatCandidateSelectorOptimizationResult:
    output_path: Path
    track_count: int
    static_track_count: int
    dynamic_track_count: int
    change_threshold: float
    macro_f1: float
    dynamic_f1: float
    static_f1: float
    static_false_multisegment: int
    baseline_macro_f1: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = True


def _phase_distance_beats(left: np.ndarray, right: np.ndarray) -> float:
    bpm = float(left[2])
    if not np.isfinite(bpm) or bpm <= 0.0:
        return float("nan")
    delta = (float(right[3]) - float(left[3])) / (60.0 / bpm)
    return float(abs(delta - round(delta)))


def _annotation_scope(segments: np.ndarray) -> tuple[str, list[str]]:
    """Keep training scope equal to the evaluated production contract."""
    values = np.asarray(segments, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 4:
        return "excluded", ["missing tempo segments"]
    bpm = values[:, 2]
    meter = values[:, 4] if values.shape[1] >= 5 else np.full(len(values), 4.0)
    relative = (
        np.abs(np.diff(bpm)) / np.maximum(np.minimum(bpm[:-1], bpm[1:]), 1e-9)
        if bpm.size >= 2
        else np.empty(0)
    )
    phase = np.asarray(
        [_phase_distance_beats(values[index], values[index + 1]) for index in range(len(values) - 1)]
    )
    reasons: list[str] = []
    if np.any(np.rint(meter).astype(int) != 4):
        reasons.append("meter change or non-4/4 annotation")
    if np.any((relative <= 0.01) & (phase >= 0.20)):
        reasons.append("same-tempo phase reset annotation")
    signed = np.diff(bpm) / np.maximum(bpm[:-1], 1e-9) if bpm.size >= 2 else np.empty(0)
    for start in range(max(0, signed.size - 3)):
        run = signed[start : start + 4]
        if (
            run.size == 4
            and np.all(np.abs(run) >= 0.001)
            and np.all(np.abs(run) <= 0.03)
            and (np.all(run > 0.0) or np.all(run < 0.0))
            and abs(float(np.sum(run))) >= 0.02
        ):
            reasons.append("continuous ramp-like annotation")
            break
    if reasons:
        return "excluded", reasons
    return ("piecewise_tempo" if np.any(relative > 0.01) else "static"), []


def _group_key(track: dict[str, object]) -> str:
    text = f"{track.get('artist', '')}|{track.get('title', '')}".casefold()
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", text) or str(track["uid"])


def _metric_f1(reference: np.ndarray, estimated: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimated, dtype=np.float64)
    ref = ref[ref >= 5.0]
    est = est[est >= 5.0]
    if ref.size < 2 or est.size < 2:
        return 0.0
    i = j = matches = 0
    while i < ref.size and j < est.size:
        delta = float(est[j] - ref[i])
        if abs(delta) <= 0.070:
            matches += 1
            i += 1
            j += 1
        elif delta < -0.070:
            j += 1
        else:
            i += 1
    precision = matches / est.size
    recall = matches / ref.size
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _track_bounds(reference: np.ndarray, configured: tuple[float, float]) -> tuple[float, float]:
    """Move the configured one-octave DJ range to contain very slow/fast annotations."""
    intervals = np.diff(np.asarray(reference, dtype=np.float64))
    median_bpm = 60.0 / float(np.median(intervals))
    low, high = (float(configured[0]), float(configured[1]))
    while median_bpm < low:
        low *= 0.5
        high *= 0.5
    while median_bpm > high:
        low *= 2.0
        high *= 2.0
    return low, high


def _load_training_tracks(
    library_dir: Path, ignored: list[SkippedOptimizationTrack]
) -> list[dict[str, object]]:
    tracks = []
    for track in _list_beat_tracks(library_dir, ignored):
        try:
            with np.load(Path(track["analysis_path"]), allow_pickle=False) as archive:
                segments = np.asarray(archive["tempo_segments"], dtype=np.float64)
                duration = float(np.asarray(archive["duration_sec"]).item())
            scope, reasons = _annotation_scope(segments)
            if scope == "excluded":
                ignored.append(skipped_track(track, "; ".join(reasons)))
                continue
            tracks.append({**track, "segments": segments, "duration": duration, "scope": scope})
        except Exception as exc:
            ignored.append(skipped_track(track, f"invalid tempo annotation: {exc}"))
    return tracks


def _prepare_candidate_rows(
    track: dict[str, object],
    cache_dir_text: str,
    onset_parameter_path_text: str,
    hop_length: int,
    bpm_window_ms: int,
    bpm_bounds: tuple[float, float],
    offset_sec: float,
    use_hpss: bool,
) -> dict[str, object]:
    """Process-worker entry: generate the four production candidates and targets."""
    from analyzer_core.beat.beat import bpm_dynamic_phase_sync, bpm_phase_sync
    from threadpoolctl import threadpool_limits

    cached = _read_current_cache(
        _cache_path(Path(cache_dir_text), str(track["uid"])), track, use_hpss, hop_length
    )
    if cached is None:
        raise ValueError("candidate-selector feature cache is not current")
    frame_times, raw_channels = cached
    weights, bias = _load_onset_model(onset_parameter_path_text)
    activation = expit(stack_context(raw_channels).astype(np.float64) @ weights + bias)
    duration = float(track["duration"])
    sample_rate = int(track["sample_rate"])
    sample_count = max(1, int(round(duration * sample_rate)))
    odf_count = max(1, 1 + (sample_count - 1) // int(hop_length))
    hop_time = int(hop_length) / float(sample_rate)
    odf_times = (np.arange(odf_count, dtype=np.float64) + 0.5) * hop_time
    odf = np.interp(odf_times, frame_times, activation).astype(np.float32)
    audio = np.empty(sample_count, dtype=np.float32)
    win_sec = float(bpm_window_ms) / 1000.0
    bounds = _track_bounds(np.asarray(track["beats"]), bpm_bounds)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), threadpool_limits(limits=1):
        candidates = [
            bpm_dynamic_phase_sync(
                sample_rate,
                hop_length,
                hop_length,
                audio=audio,
                win_s=win_sec * multiplier,
                step_s=0.25,
                bpm_bounds=bounds,
                odf_precomputed=odf,
                hop_t_precomputed=hop_time,
                validrange_precomputed=(0.0, duration),
            )
            for multiplier in (1, 2, 4)
        ]
        candidates.append(
            bpm_phase_sync(
                sample_rate,
                hop_length,
                hop_length,
                audio=audio,
                win_s=win_sec,
                step_s=0.1,
                bpm_bounds=bounds,
                odf_precomputed=odf,
                hop_t_precomputed=hop_time,
            )
        )
    prepared_candidates = [
        prepare_candidate_for_selection(candidate, duration, offset_sec)
        for candidate in candidates
    ]
    X = build_candidate_features(
        prepared_candidates,
        frame_times,
        standardize_candidate_channels(raw_channels),
        duration,
        bounds,
    )
    reference = np.asarray(track["beats"], dtype=np.float64)
    target = np.asarray(
        [_metric_f1(reference, candidate["beats_time"]) for candidate in prepared_candidates],
        dtype=np.float64,
    )
    scores = np.asarray([candidate["score"] for candidate in prepared_candidates])
    baseline = int(np.flatnonzero(scores >= np.max(scores) - 1e-12)[-1])
    return {
        "uid": str(track["uid"]),
        "X": X,
        "target": target,
        "segment_counts": np.asarray(
            [np.asarray(candidate["segments"]).shape[0] for candidate in prepared_candidates],
            dtype=np.int32,
        ),
        "baseline": baseline,
    }


def _presence_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.25,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _source_model(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.75,
        n_jobs=-1,
        random_state=seed,
    )


def _fit_models(data: list[dict[str, object]], labels: np.ndarray, indices: np.ndarray, seed: int):
    presence = _presence_model(seed).fit(
        np.asarray([build_presence_features(data[int(i)]["X"]) for i in indices]),
        labels[indices],
    )
    source = _source_model(seed + 1).fit(
        np.concatenate([data[int(i)]["X"] for i in indices]),
        np.concatenate([data[int(i)]["target"] for i in indices]),
    )
    return presence, source


def _threshold_from_group_oof(
    data: list[dict[str, object]],
    labels: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> float:
    local_groups = groups[indices]
    n_splits = min(4, np.unique(local_groups).size)
    if n_splits < 2 or np.unique(labels[indices]).size != 2:
        return 0.5
    probability = np.full(indices.size, np.nan)
    dynamic_quality = np.full(indices.size, np.nan)
    static_quality = np.full(indices.size, np.nan)
    dynamic_segments = np.ones(indices.size, dtype=np.int32)
    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=n_splits).split(indices, groups=local_groups), start=1
    ):
        train_global = indices[train]
        if np.unique(labels[train_global]).size != 2:
            return 0.5
        classifier, regressor = _fit_models(data, labels, train_global, seed + 10 * fold)
        probability[test] = classifier.predict_proba(
            np.asarray([build_presence_features(data[int(indices[i])]["X"]) for i in test])
        )[:, 1]
        for local in test:
            item = data[int(indices[local])]
            prediction = regressor.predict(item["X"][:STATIC_INDEX])
            choice = int(np.argmax(prediction))
            dynamic_quality[local] = float(item["target"][choice])
            static_quality[local] = float(item["target"][STATIC_INDEX])
            dynamic_segments[local] = int(item["segment_counts"][choice])
    positive = labels[indices] == 1
    candidates = []
    for threshold in PRESENCE_THRESHOLDS:
        select_dynamic = probability >= threshold
        quality = np.where(select_dynamic, dynamic_quality, static_quality)
        dynamic_f1 = float(np.mean(quality[positive]))
        static_f1 = float(np.mean(quality[~positive]))
        false_splits = int(np.sum(select_dynamic[~positive] & (dynamic_segments[~positive] > 1)))
        candidates.append((-dynamic_f1, false_splits, -static_f1, threshold))
    return float(min(candidates)[3])


def _evaluate_choices(
    data: list[dict[str, object]], labels: np.ndarray, choices: np.ndarray
) -> dict[str, float | int]:
    quality = np.asarray(
        [float(item["target"][int(choice)]) for item, choice in zip(data, choices, strict=True)]
    )
    static = labels == 0
    dynamic = ~static
    return {
        "macro_f1": float(np.mean(quality)),
        "dynamic_f1": float(np.mean(quality[dynamic])),
        "static_f1": float(np.mean(quality[static])),
        "static_false_multisegment": int(
            sum(
                bool(static[index]) and int(data[index]["segment_counts"][choices[index]]) > 1
                for index in range(len(data))
            )
        ),
    }


def optimize_beat_candidate_selector(
    request: BeatCandidateSelectorOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> BeatCandidateSelectorOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    onset_path = Path(request.onset_parameter_path).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Beat candidate selector parameter path must end with .npz")
    if not onset_path.is_file():
        raise FileNotFoundError(f"Onset parameter not found: {onset_path}")

    ignored: list[SkippedOptimizationTrack] = []
    tracks = _load_training_tracks(library_dir, ignored)
    if len(tracks) < MIN_TRACKS:
        raise RuntimeError(f"At least {MIN_TRACKS} supported annotated tracks are required")
    labels = np.asarray([track["scope"] == "piecewise_tempo" for track in tracks], dtype=np.uint8)
    if np.sum(labels == 0) < 2 or np.sum(labels == 1) < 2:
        raise RuntimeError("Both static and piecewise-tempo annotations need at least 2 tracks")

    cache_jobs = [
        track
        for track in tracks
        if request.rebuild_cache
        or _read_current_cache(
            _cache_path(cache_dir, str(track["uid"])),
            track,
            bool(request.use_hpss),
            int(request.frame_hop_length),
        )
        is None
    ]
    total = len(tracks)
    cached = total - len(cache_jobs)
    workers = _auto_cache_workers(request.max_cache_workers, len(cache_jobs))
    if feature_progress:
        feature_progress(
            int(round(45 * cached / total)),
            f"Building {len(cache_jobs)} candidate-selector feature caches with {workers} workers",
        )
    if cache_jobs:
        with create_spawn_process_pool(workers) as executor:
            futures = {
                executor.submit(
                    _build_track_cache,
                    track,
                    str(cache_dir),
                    bool(request.use_hpss),
                    int(request.frame_hop_length),
                ): track
                for track in cache_jobs
            }
            failed: set[str] = set()
            for completed, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.add(str(track["uid"]))
                    ignored.append(skipped_track(track, f"feature cache build failed: {exc}"))
                if feature_progress:
                    feature_progress(
                        int(round(45 * (cached + completed) / total)),
                        f"Feature cache {cached + completed}/{total}\n{optimizer_track_name(track)}",
                    )
            tracks = [track for track in tracks if str(track["uid"]) not in failed]

    jobs = len(tracks)
    workers = _auto_cache_workers(request.max_cache_workers, jobs)
    prepared_by_uid: dict[str, dict[str, object]] = {}
    with create_spawn_process_pool(workers) as executor:
        futures = {
            executor.submit(
                _prepare_candidate_rows,
                track,
                str(cache_dir),
                str(onset_path),
                int(request.frame_hop_length),
                int(request.bpm_window_ms),
                (float(request.bpm_min), float(request.bpm_max)),
                float(request.beatgrid_offset_msec) / 1000.0,
                bool(request.use_hpss),
            ): track
            for track in tracks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            track = futures[future]
            try:
                item = future.result()
                prepared_by_uid[str(track["uid"])] = item
            except Exception as exc:
                ignored.append(skipped_track(track, f"candidate preparation failed: {exc}"))
            if feature_progress:
                feature_progress(
                    45 + int(round(55 * completed / max(jobs, 1))),
                    f"Candidates {completed}/{jobs}\n{optimizer_track_name(track)}",
                )
    tracks = [track for track in tracks if str(track["uid"]) in prepared_by_uid]
    data = [prepared_by_uid[str(track["uid"])] for track in tracks]
    labels = np.asarray([track["scope"] == "piecewise_tempo" for track in tracks], dtype=np.uint8)
    if len(data) < MIN_TRACKS or np.unique(labels).size != 2:
        raise RuntimeError("Insufficient supported tracks after candidate preparation")
    groups = np.asarray([_group_key(track) for track in tracks])
    indices = np.arange(len(data), dtype=np.int32)

    cross_validated = int(request.cv_folds) >= 2
    choices = np.empty(len(data), dtype=np.int32)
    if cross_validated:
        n_splits = min(int(request.cv_folds), np.unique(groups).size)
        if optimize_progress:
            optimize_progress(0, f"Grouped {n_splits}-fold candidate selection")
        for fold, (train, test) in enumerate(
            GroupKFold(n_splits=n_splits).split(indices, groups=groups), start=1
        ):
            threshold = _threshold_from_group_oof(data, labels, groups, train, 10000 + fold * 100)
            classifier, regressor = _fit_models(data, labels, train, 12000 + fold * 100)
            for index in test:
                item = data[int(index)]
                probability = float(
                    classifier.predict_proba(build_presence_features(item["X"])[None, :])[0, 1]
                )
                dynamic = int(np.argmax(regressor.predict(item["X"][:STATIC_INDEX])))
                choices[int(index)] = dynamic if probability >= threshold else STATIC_INDEX
            if optimize_progress:
                optimize_progress(
                    int(round(80 * fold / n_splits)), f"Cross-validation fold {fold}/{n_splits}"
                )
    else:
        threshold = _threshold_from_group_oof(data, labels, groups, indices, 14000)
        classifier, regressor = _fit_models(data, labels, indices, 15000)
        for index, item in enumerate(data):
            probability = float(
                classifier.predict_proba(build_presence_features(item["X"])[None, :])[0, 1]
            )
            dynamic = int(np.argmax(regressor.predict(item["X"][:STATIC_INDEX])))
            choices[index] = dynamic if probability >= threshold else STATIC_INDEX
    metrics = _evaluate_choices(data, labels, choices)
    baseline_choices = np.asarray([item["baseline"] for item in data], dtype=np.int32)
    baseline_metrics = _evaluate_choices(data, labels, baseline_choices)

    if optimize_progress:
        optimize_progress(82, "Selecting final threshold and fitting all tracks")
    final_threshold = _threshold_from_group_oof(data, labels, groups, indices, 16000)
    classifier, regressor = _fit_models(data, labels, indices, 17000)
    artifact = {
        "change_threshold": final_threshold,
        "presence_classifier": classifier,
        "source_regressor": regressor,
        "training": {
            "date": date.today().isoformat(),
            "tracks": len(data),
            "static": int(np.sum(labels == 0)),
            "piecewise_tempo": int(np.sum(labels == 1)),
            "excluded_scope": "meter changes, same-tempo phase resets, continuous ramps",
            "protocol": "grouped OOF; dynamic F1 first, static false splits second",
        },
    }
    if optimize_progress:
        optimize_progress(96, "Writing NumPy candidate-selector parameter")
    export_candidate_selector_npz_artifact(artifact, output_path)

    # Export parity is checked with the same rows before reporting success.
    runtime = load_candidate_selector_model(str(output_path))
    presence_X = np.asarray([build_presence_features(item["X"]) for item in data])
    candidate_X = np.concatenate([item["X"] for item in data])
    if not np.allclose(
        classifier.predict_proba(presence_X),
        runtime["presence_classifier"].predict_proba(presence_X),
        rtol=1e-12,
        atol=1e-12,
    ) or not np.allclose(
        regressor.predict(candidate_X),
        runtime["source_regressor"].predict(candidate_X),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("NumPy ExtraTrees export parity check failed")
    if optimize_progress:
        optimize_progress(100, "Done")
    return BeatCandidateSelectorOptimizationResult(
        output_path=output_path,
        track_count=len(data),
        static_track_count=int(np.sum(labels == 0)),
        dynamic_track_count=int(np.sum(labels == 1)),
        change_threshold=float(final_threshold),
        macro_f1=float(metrics["macro_f1"]),
        dynamic_f1=float(metrics["dynamic_f1"]),
        static_f1=float(metrics["static_f1"]),
        static_false_multisegment=int(metrics["static_false_multisegment"]),
        baseline_macro_f1=float(baseline_metrics["macro_f1"]),
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
