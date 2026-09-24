from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import multiprocessing
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer_core.beat.beat_phase import (
    BEAT_PHASE_FEATURE_NAMES,
    BEAT_PHASE_MODEL_FORMAT,
    BEAT_PHASE_RAW_FEATURE_NAMES,
    N_CURVES,
    SUBDIVISION,
    beat_phase_design,
    beat_phase_track_features,
    clear_beat_phase_weight_cache,
    standardize_beat_phase_groups,
)
from analyzer_core.beat.frame_features import extract_frame_features
from core.audio.decoder import decode_to_memmap
from optimizer import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from optimizer.downbeat_parameter_optimizer import _fit_conditional_softmax, _group_metrics
from optimizer.onset_parameter_optimizer import _list_beat_tracks
from utils.atomic_io import atomic_output_path, atomic_write_json


ProgressCallback = Callable[[int, str], None]
BEAT_PHASE_FEATURE_CACHE_FORMAT = "mixlyzer_beat_phase_feature_cache_v4"
L2_CANDIDATES = (0.0003, 0.001, 0.01)
# Used when cross-validation is skipped (cv_folds < 2): the L2 chosen by the last cross-validated fit.
DEFAULT_L2_STRENGTH = 0.0003


@dataclass(frozen=True)
class BeatPhaseOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    use_hpss: bool = True
    rebuild_cache: bool = False
    # < 2 skips cross-validation: one fit on all tracks with l2_strength (much faster).
    cv_folds: int = 5
    # With cross-validation: True picks the L2 among L2_CANDIDATES, False evaluates l2_strength only.
    l2_sweep: bool = True
    l2_strength: float = DEFAULT_L2_STRENGTH
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class BeatPhaseOptimizationResult:
    output_path: Path
    track_count: int
    beat_count: int
    selected_l2_strength: float
    cross_entropy: float
    beat_top1_accuracy: float
    track_top1_accuracy: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    # False: the metrics are measured on the training tracks (cross-validation skipped).
    cross_validated: bool = True


def _cache_path(cache_dir: Path, uid: str) -> Path:
    return cache_dir / f"{uid}.npz"


def _read_current_cache(
    cache_path: Path, track: dict[str, object], use_hpss: bool
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "cache_format",
                "groups",
                "regularity",
                "feature_names",
                "beat_times_sec",
                "use_hpss",
                "audio_mtime_ns",
                "analysis_mtime_ns",
            }:
                return None
            groups = np.asarray(archive["groups"])
            regularity = np.asarray(archive["regularity"], dtype=np.float64)
            beats = np.asarray(archive["beat_times_sec"], dtype=np.float64)
            current = bool(
                str(np.asarray(archive["cache_format"]).item()) == BEAT_PHASE_FEATURE_CACHE_FORMAT
                and tuple(np.asarray(archive["feature_names"]).astype(str)) == BEAT_PHASE_RAW_FEATURE_NAMES
                and bool(np.asarray(archive["use_hpss"]).item()) == bool(use_hpss)
                and groups.shape == (beats.size, SUBDIVISION, len(BEAT_PHASE_RAW_FEATURE_NAMES))
                and regularity.shape == (N_CURVES,)
                and np.array_equal(beats, np.asarray(track["beats"], dtype=np.float64))
                and int(np.asarray(archive["audio_mtime_ns"]).item())
                == Path(track["audio_path"]).stat().st_mtime_ns
                and int(np.asarray(archive["analysis_mtime_ns"]).item())
                == Path(track["analysis_path"]).stat().st_mtime_ns
            )
            if current:
                return np.asarray(groups, dtype=np.float32), regularity
    except Exception:
        # A partial/corrupt cache is equivalent to a miss and is rebuilt.
        pass
    return None


def _build_track_cache(track: dict[str, object], cache_dir_text: str, use_hpss: bool) -> str:
    """Process-worker entry point for one beat-phase feature cache."""

    cache_path = _cache_path(Path(cache_dir_text), str(track["uid"]))

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - installed with scikit-learn
        threadpool_limits = None

    def build() -> None:
        import librosa

        sample_rate = int(track["sample_rate"])
        # Same input as the analyzer: stereo decode averaged to mono and its
        # HPSS parts.
        stereo = decode_to_memmap(str(track["audio_path"]), sample_rate, 2).reshape(-1, 2)
        audio = np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32)
        harmonic, percussive = librosa.effects.hpss(audio) if use_hpss else (audio, audio)
        frames = extract_frame_features(audio, sample_rate, percussive, harmonic)
        beats = np.asarray(track["beats"], dtype=np.float64)
        features = beat_phase_track_features(frames, beats)
        with atomic_output_path(cache_path) as temporary:
            np.savez_compressed(
                temporary,
                cache_format=np.asarray(BEAT_PHASE_FEATURE_CACHE_FORMAT),
                groups=features.groups.astype(np.float32),
                regularity=features.regularity.astype(np.float64),
                feature_names=np.asarray(BEAT_PHASE_RAW_FEATURE_NAMES, dtype="U64"),
                beat_times_sec=beats,
                use_hpss=np.bool_(use_hpss),
                audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns),
                analysis_mtime_ns=np.int64(Path(track["analysis_path"]).stat().st_mtime_ns),
            )

    if threadpool_limits is None:
        build()
    else:
        with threadpool_limits(limits=1):
            build()
    return str(track["uid"])


def _regularity_stats(
    tracks: Sequence[dict[str, object]], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Library mean / std of the per-curve regularity (fitted on training tracks only)."""
    values = np.stack([np.asarray(tracks[i]["regularity"], dtype=np.float64) for i in indices])
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6)


def _design(track: dict[str, object], stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    groups = np.asarray(track["groups"])[np.asarray(track["complete"])]
    return beat_phase_design(groups, np.asarray(track["regularity"]), stats[0], stats[1])


def _training_groups(
    tracks: Sequence[dict[str, object]],
    indices: Sequence[int],
    stats: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    parts = [_design(tracks[i], stats) for i in indices]
    parts = [p for p in parts if p.size]
    if not parts:
        return np.empty((0, SUBDIVISION, len(BEAT_PHASE_FEATURE_NAMES)), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _track_correct(
    track: dict[str, object], weights: np.ndarray, stats: tuple[np.ndarray, np.ndarray]
) -> bool:
    design = _design(track, stats)
    return bool(design.size and int(np.argmax((design @ weights).sum(axis=0))) == 0)


def _cross_validate(
    tracks: Sequence[dict[str, object]],
    folds: int,
    progress: ProgressCallback | None,
    candidates: Sequence[float] = L2_CANDIDATES,
) -> tuple[float, dict[str, object]]:
    n_splits = min(int(folds), len(tracks))
    if n_splits < 2:
        raise ValueError("Beat phase optimization requires at least two folds")
    indices = np.arange(len(tracks), dtype=np.int32)
    splits = list(
        GroupKFold(n_splits=n_splits).split(
            indices, np.zeros(len(tracks), dtype=np.int8), groups=indices
        )
    )
    total_fits = len(candidates) * len(splits)
    completed_fits = 0
    best: tuple[float, float, dict[str, object]] | None = None
    for l2_strength in candidates:
        fold_rows = []
        total_beats = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        tracks_correct = 0
        for fold, (train_indices, test_indices) in enumerate(splits, start=1):
            stats = _regularity_stats(tracks, train_indices)
            weights, diagnostics = _fit_conditional_softmax(
                _training_groups(tracks, train_indices, stats), l2_strength
            )
            metrics = _group_metrics(_training_groups(tracks, test_indices, stats), weights)
            wrong_uids = [str(tracks[i]["uid"]) for i in test_indices if not _track_correct(tracks[i], weights, stats)]
            correct = len(test_indices) - len(wrong_uids)
            count = int(metrics["bar_count"])
            total_beats += count
            weighted_loss += float(metrics["cross_entropy"]) * count
            weighted_accuracy += float(metrics["top1_accuracy"]) * count
            tracks_correct += correct
            fold_rows.append(
                {
                    "fold": fold,
                    "test_uids": [str(tracks[i]["uid"]) for i in test_indices],
                    "wrong_uids": wrong_uids,
                    "optimizer": diagnostics,
                    "beat_count": count,
                    "cross_entropy": float(metrics["cross_entropy"]),
                    "beat_top1_accuracy": float(metrics["top1_accuracy"]),
                    "track_top1_accuracy": correct / max(len(test_indices), 1),
                }
            )
            completed_fits += 1
            if progress is not None:
                progress(
                    int(round(completed_fits * 85 / total_fits)),
                    f"Cross-validation L2={l2_strength:g}, fold {fold}/{n_splits}",
                )
        metrics = {
            "beat_count": total_beats,
            "cross_entropy": weighted_loss / max(total_beats, 1),
            "beat_top1_accuracy": weighted_accuracy / max(total_beats, 1),
            "track_top1_accuracy": tracks_correct / max(len(tracks), 1),
            "folds": fold_rows,
        }
        if best is None or float(metrics["cross_entropy"]) < best[0]:
            best = (float(metrics["cross_entropy"]), l2_strength, metrics)
    if best is None:
        raise RuntimeError("Cross-validation produced no result")
    return best[1], best[2]


def _auto_cache_workers(requested: int | None, jobs: int) -> int:
    if jobs <= 0:
        return 1
    if requested is not None:
        return max(1, min(int(requested), jobs))
    return max(1, min(4, jobs, (os.cpu_count() or 2) // 2))


def optimize_beat_phase_parameters(
    request: BeatPhaseOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> BeatPhaseOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_hpss = bool(request.use_hpss)

    ignored: list[SkippedOptimizationTrack] = []
    tracks_src = _list_beat_tracks(library_dir, ignored)
    if len(tracks_src) < 3:
        raise RuntimeError("At least 3 analyzed tracks with beat grids are required")

    cache_jobs = [
        track
        for track in tracks_src
        if request.rebuild_cache
        or _read_current_cache(_cache_path(cache_dir, str(track["uid"])), track, use_hpss) is None
    ]
    total = len(tracks_src)
    cached_count = total - len(cache_jobs)
    workers = _auto_cache_workers(request.max_cache_workers, len(cache_jobs))
    if feature_progress is not None:
        feature_progress(
            int(round(cached_count * 90 / total)),
            (
                f"Building {len(cache_jobs)} feature caches with {workers} workers"
                if cache_jobs
                else "Beat phase feature cache is current"
            ),
        )

    if cache_jobs:
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            futures = {
                executor.submit(_build_track_cache, track, str(cache_dir), use_hpss): track
                for track in cache_jobs
            }
            failed_uids: set[str] = set()
            for completed, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed_uids.add(str(track["uid"]))
                    ignored.append(
                        skipped_track(track, f"Beat phase feature cache build failed: {exc}")
                    )
                if feature_progress is not None:
                    done = cached_count + completed
                    feature_progress(
                        int(round(done * 90 / total)),
                        f"Feature cache {done}/{total}\n{optimizer_track_name(track)}",
                    )
            tracks_src = [
                track for track in tracks_src if str(track["uid"]) not in failed_uids
            ]

    prepared: list[dict[str, object]] = []
    prepare_total = len(tracks_src)
    for index, track in enumerate(tracks_src, start=1):
        try:
            cached = _read_current_cache(_cache_path(cache_dir, str(track["uid"])), track, use_hpss)
            if cached is None:
                raise ValueError("beat phase feature cache is not current")
            groups, regularity = cached
            standardized, complete = standardize_beat_phase_groups(groups)
            if not complete.any():
                raise ValueError("no complete beats")
            prepared.append(
                {
                    "uid": str(track["uid"]),
                    "groups": standardized,
                    "complete": complete,
                    "regularity": regularity,
                }
            )
        except Exception as exc:
            ignored.append(skipped_track(track, f"beat preparation failed: {exc}"))
        if feature_progress is not None:
            feature_progress(
                90 + int(round(index * 10 / max(prepare_total, 1))),
                f"Preparing beats {index}/{prepare_total}\n{optimizer_track_name(track)}",
            )
    if len(prepared) < 3:
        raise RuntimeError("At least 3 tracks with beat grids are required")

    cross_validated = int(request.cv_folds) >= 2
    if cross_validated:
        if optimize_progress is not None:
            optimize_progress(0, "Selecting regularization by cross-validation")
        l2_strength, cross_validation = _cross_validate(
            prepared,
            request.cv_folds,
            optimize_progress,
            L2_CANDIDATES if request.l2_sweep else (float(request.l2_strength),),
        )
    else:
        l2_strength, cross_validation = float(request.l2_strength), None
    all_indices = np.arange(len(prepared))
    stats = _regularity_stats(prepared, all_indices)
    all_groups = _training_groups(prepared, all_indices, stats)
    if optimize_progress is not None:
        optimize_progress(90, "Fitting final beat phase model")
    weights, optimizer_diagnostics = _fit_conditional_softmax(all_groups, l2_strength)
    training_metrics = _group_metrics(all_groups, weights)
    training_metrics["track_top1_accuracy"] = sum(
        _track_correct(prepared[i], weights, stats) for i in all_indices
    ) / max(len(prepared), 1)
    metrics = cross_validation if cross_validated else {
        "cross_entropy": training_metrics["cross_entropy"],
        "beat_top1_accuracy": training_metrics["top1_accuracy"],
        "track_top1_accuracy": training_metrics["track_top1_accuracy"],
    }
    report = {
        "algorithm": "beat_phase_conditional_softmax_regularity_harmony_melody_v4",
        "profile": "beat-phase-correction",
        "track_count": len(prepared),
        "beat_count": int(all_groups.shape[0]),
        "objective": (
            f"Conditional softmax over the {SUBDIVISION} phase candidates of every annotated "
            "beat (the beat itself is candidate 0), each described by the local periodicity "
            "of the shared onset curves aligned to it, beat structure, chord changes and melody "
            "note lengths, and every curve's cues times its regularity in the track. Analysis "
            "sums the logits over the track."
        ),
        "use_hpss": use_hpss,
        "selected_l2_strength": l2_strength,
        "cross_validation": cross_validation,
        "optimizer": optimizer_diagnostics,
        "training_metrics": training_metrics,
        "skipped_tracks": [asdict(track) for track in ignored],
        "model": {
            "format": BEAT_PHASE_MODEL_FORMAT,
            "subdivision": SUBDIVISION,
            "feature_names": list(BEAT_PHASE_FEATURE_NAMES),
            "weights": weights.tolist(),
            "regularity_mean": stats[0].tolist(),
            "regularity_std": stats[1].tolist(),
        },
    }
    if optimize_progress is not None:
        optimize_progress(97, "Writing beat phase parameter")
    atomic_write_json(output_path, report, ensure_ascii=False, indent=2)
    clear_beat_phase_weight_cache()
    if optimize_progress is not None:
        optimize_progress(100, "Done")

    return BeatPhaseOptimizationResult(
        output_path=output_path,
        track_count=len(prepared),
        beat_count=int(all_groups.shape[0]),
        selected_l2_strength=float(l2_strength),
        cross_entropy=float(metrics["cross_entropy"]),
        beat_top1_accuracy=float(metrics["beat_top1_accuracy"]),
        track_top1_accuracy=float(metrics["track_top1_accuracy"]),
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
