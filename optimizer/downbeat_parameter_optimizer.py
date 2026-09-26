from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

from analyzer_core.beat.downbeat_offset import (
    DOWNBEAT_FEATURE_NAMES,
    clear_downbeat_weight_cache,
    extract_downbeat_feature_matrix,
    standardize_downbeat_features,
)
from core.audio.decoder import decode_to_memmap
from core.beat_geometry import downbeat_beat_indices
from core.library_handler import LibraryDB
from optimizer import (
    SkippedOptimizationTrack,
    optimizer_track_name,
    skipped_track,
)
from core.concurrency.process_pool import create_spawn_process_pool
from utils.atomic_io import atomic_output_path, atomic_write_json


ProgressCallback = Callable[[int, str], None]
DOWNBEAT_FEATURE_CACHE_FORMAT = "mixlyzer_downbeat_feature_cache_v4"
L2_CANDIDATES = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
# Used when cross-validation is skipped (cv_folds < 2): the L2 chosen by the last cross-validated fit.
DEFAULT_L2_STRENGTH = 0.001


@dataclass(frozen=True)
class DownbeatOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    use_hpss: bool = True
    rebuild_cache: bool = False
    # < 2 skips cross-validation: one fit on all tracks with l2_strength.
    cv_folds: int = 5
    # With cross-validation: True picks the L2 among L2_CANDIDATES, False evaluates l2_strength only.
    l2_sweep: bool = True
    l2_strength: float = DEFAULT_L2_STRENGTH
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class DownbeatOptimizationResult:
    output_path: Path
    track_count: int
    bar_count: int
    selected_l2_strength: float
    cross_entropy: float
    top1_accuracy: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    # False: the metrics are measured on the training tracks (cross-validation skipped).
    cross_validated: bool = True


def _list_4_4_tracks(
    library_dir: Path,
    ignored: list[SkippedOptimizationTrack] | None = None,
) -> list[dict[str, object]]:
    database_path = library_dir / "library.db"
    if not database_path.is_file():
        raise FileNotFoundError(f"Library database not found: {database_path}")

    library = LibraryDB(str(database_path))
    library.connect()
    try:
        rows = library.list_all(order_by="uid ASC")
    finally:
        library.close()

    tracks: list[dict[str, object]] = []
    for row in rows:
        if not row.uid:
            continue
        uid = str(row.uid)
        analysis_path = library_dir / f"{uid}.npz"
        audio_path = Path(row.path) if row.path else Path()
        track = {
            "uid": uid,
            "title": str(row.title or ""),
            "artist": str(row.artist or ""),
            "audio_path": audio_path,
            "analysis_path": analysis_path,
        }
        if not analysis_path.is_file():
            if ignored is not None:
                ignored.append(skipped_track(track, "analysis NPZ is missing"))
            continue
        if not audio_path.is_file():
            if ignored is not None:
                ignored.append(skipped_track(track, "audio file is missing"))
            continue
        try:
            with np.load(analysis_path, allow_pickle=False) as archive:
                beats = np.asarray(archive["beats_time_sec"], dtype=np.float64)
                segments = np.asarray(archive["tempo_segments"], dtype=np.float64)
                sample_rate = int(np.asarray(archive["sr"]).item())
        except (KeyError, OSError, ValueError) as exc:
            if ignored is not None:
                ignored.append(skipped_track(track, f"invalid beat analysis: {exc}"))
            continue
        if beats.ndim != 1 or beats.size < 4 or not np.all(np.isfinite(beats)):
            if ignored is not None:
                ignored.append(skipped_track(track, "invalid or insufficient beat grid"))
            continue
        if segments.ndim != 2 or segments.shape[1] < 5 or segments.size == 0:
            if ignored is not None:
                ignored.append(skipped_track(track, "invalid or empty tempo segments"))
            continue
        if {int(round(float(value))) for value in segments[:, 4]} != {4}:
            continue
        track.update(
            duration_sec=float(row.duration or np.max(segments[:, 1])),
            beats=beats,
            segments=segments,
            sample_rate=sample_rate,
        )
        tracks.append(track)
    return tracks


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
                "beat_times_sec",
                "feature_raw",
                "feature_names",
                "use_hpss",
                "audio_mtime_ns",
                "analysis_mtime_ns",
            }:
                return None
            beats = np.asarray(archive["beat_times_sec"], dtype=np.float64)
            features = np.asarray(archive["feature_raw"])
            current = bool(
                str(np.asarray(archive["cache_format"]).item())
                == DOWNBEAT_FEATURE_CACHE_FORMAT
                and tuple(np.asarray(archive["feature_names"]).astype(str))
                == DOWNBEAT_FEATURE_NAMES
                and bool(np.asarray(archive["use_hpss"]).item()) == bool(use_hpss)
                and beats.ndim == 1
                and features.shape == (beats.size, len(DOWNBEAT_FEATURE_NAMES))
                and features.dtype == np.dtype(np.float32)
                and np.all(np.isfinite(beats))
                and np.all(np.isfinite(features))
                and int(np.asarray(archive["audio_mtime_ns"]).item())
                == Path(track["audio_path"]).stat().st_mtime_ns
                and int(np.asarray(archive["analysis_mtime_ns"]).item())
                == Path(track["analysis_path"]).stat().st_mtime_ns
            )
            if current:
                return beats, np.asarray(features, dtype=np.float32)
    except Exception:
        # A partial/corrupt cache is equivalent to a miss and is rebuilt.
        pass
    return None


def _cache_is_current(cache_path: Path, track: dict[str, object], use_hpss: bool) -> bool:
    return _read_current_cache(cache_path, track, use_hpss) is not None


def _write_cache_atomic(
    cache_path: Path,
    track: dict[str, object],
    beats: np.ndarray,
    features: np.ndarray,
    use_hpss: bool,
) -> None:
    with atomic_output_path(cache_path) as temporary:
        np.savez_compressed(
            temporary,
            cache_format=np.asarray(DOWNBEAT_FEATURE_CACHE_FORMAT),
            beat_times_sec=np.asarray(beats, dtype=np.float64),
            feature_raw=np.asarray(features, dtype=np.float32),
            feature_names=np.asarray(DOWNBEAT_FEATURE_NAMES, dtype="U64"),
            use_hpss=np.bool_(use_hpss),
            audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns),
            analysis_mtime_ns=np.int64(
                Path(track["analysis_path"]).stat().st_mtime_ns
            ),
        )


def _build_track_cache(
    track: dict[str, object], cache_dir_text: str, use_hpss: bool
) -> str:
    """Process-worker entry point for one downbeat feature cache."""

    cache_dir = Path(cache_dir_text)
    cache_path = _cache_path(cache_dir, str(track["uid"]))

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
        beats, features = extract_downbeat_feature_matrix(
            audio,
            sample_rate,
            np.asarray(track["beats"], dtype=np.float64),
            percussive=percussive,
            harmonic=harmonic,
        )
        _write_cache_atomic(cache_path, track, beats, features, use_hpss)

    if threadpool_limits is None:
        build()
    else:
        with threadpool_limits(limits=1):
            build()
    return str(track["uid"])


def _load_current_cache(
    cache_path: Path, track: dict[str, object], use_hpss: bool
) -> tuple[np.ndarray, np.ndarray]:
    cached = _read_current_cache(cache_path, track, use_hpss)
    if cached is None:
        raise ValueError(f"Downbeat feature cache is not current: {cache_path}")
    return cached


def _complete_bar_groups(beats: np.ndarray, segments: np.ndarray) -> np.ndarray:
    downbeats = downbeat_beat_indices(beats, segments)
    if downbeats.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    groups: list[np.ndarray] = []
    for segment in segments:
        start = int(np.searchsorted(beats, float(segment[0]), side="left"))
        end = int(np.searchsorted(beats, float(segment[1]), side="left"))
        if end - start < 4:
            continue
        for downbeat in downbeats[(downbeats >= start) & (downbeats < end)]:
            group = np.arange(downbeat, downbeat + 4, dtype=np.int32)
            if group[-1] < end and group[-1] < beats.size:
                groups.append(group)
    return np.stack(groups) if groups else np.empty((0, 4), dtype=np.int32)


def _training_groups(
    tracks: Sequence[dict[str, object]], indices: Sequence[int]
) -> np.ndarray:
    groups = [
        np.asarray(tracks[index]["feature_z"])[
            np.asarray(tracks[index]["bar_groups"], dtype=np.int32)
        ]
        for index in indices
        if np.asarray(tracks[index]["bar_groups"]).size
    ]
    if not groups:
        return np.empty((0, 4, len(DOWNBEAT_FEATURE_NAMES)), dtype=np.float64)
    return np.concatenate(groups, axis=0)


def _fit_conditional_softmax(
    groups: np.ndarray, l2_strength: float
) -> tuple[np.ndarray, dict[str, object]]:
    if groups.shape[0] == 0:
        raise ValueError("No complete 4/4 bars are available for optimization")
    n_features = groups.shape[2]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = groups @ weights
        log_normalizer = logsumexp(logits, axis=1)
        loss = float(np.mean(log_normalizer - logits[:, 0]))
        loss += l2_strength * float(np.dot(weights, weights))
        probabilities = np.exp(logits - log_normalizer[:, np.newaxis])
        gradient = np.mean(
            np.sum(probabilities[:, :, np.newaxis] * groups, axis=1)
            - groups[:, 0, :],
            axis=0,
        )
        gradient += 2.0 * l2_strength * weights
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(n_features, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-3.0, 3.0)] * n_features,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Downbeat weight optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), {
        "optimizer": "L-BFGS-B",
        "iterations": int(result.nit),
        "objective_loss": float(result.fun),
        "l2_strength": float(l2_strength),
    }


def _group_metrics(groups: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    logits = groups @ weights
    probabilities = np.exp(logits - logsumexp(logits, axis=1)[:, np.newaxis])
    predicted = np.argmax(probabilities, axis=1)
    expected = np.zeros(groups.shape[0], dtype=np.int8)
    return {
        "bar_count": int(groups.shape[0]),
        "top1_accuracy": float(accuracy_score(expected, predicted)),
        "mean_downbeat_probability": float(np.mean(probabilities[:, 0])),
        "mean_strongest_other_probability": float(
            np.mean(np.max(probabilities[:, 1:], axis=1))
        ),
        "cross_entropy": float(
            log_loss(expected, probabilities, labels=np.arange(probabilities.shape[1]))
        ),
    }


def _cross_validate(
    tracks: Sequence[dict[str, object]],
    folds: int,
    progress: ProgressCallback | None,
    candidates: Sequence[float] = L2_CANDIDATES,
) -> tuple[float, dict[str, object]]:
    n_splits = min(int(folds), len(tracks))
    if n_splits < 2:
        raise ValueError("Downbeat optimization requires at least two folds")
    indices = np.arange(len(tracks), dtype=np.int32)
    splits = list(
        GroupKFold(n_splits=n_splits).split(
            indices, np.zeros(len(tracks), dtype=np.int8), groups=indices
        )
    )
    total_fits = len(candidates) * len(splits)
    completed_fits = 0
    best: tuple[tuple[float, float], float, dict[str, object]] | None = None
    for l2_strength in candidates:
        fold_rows = []
        total_bars = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        for fold, (train_indices, test_indices) in enumerate(splits, start=1):
            weights, diagnostics = _fit_conditional_softmax(
                _training_groups(tracks, train_indices), l2_strength
            )
            metrics = _group_metrics(_training_groups(tracks, test_indices), weights)
            count = int(metrics["bar_count"])
            total_bars += count
            weighted_loss += float(metrics["cross_entropy"]) * count
            weighted_accuracy += float(metrics["top1_accuracy"]) * count
            fold_rows.append(
                {
                    "fold": fold,
                    "test_uids": [str(tracks[index]["uid"]) for index in test_indices],
                    "optimizer": diagnostics,
                    "metrics": metrics,
                }
            )
            completed_fits += 1
            if progress is not None:
                progress(
                    int(round(completed_fits * 85 / total_fits)),
                    f"Cross-validation L2={l2_strength:g}, fold {fold}/{n_splits}",
                )
        metrics = {
            "bar_count": total_bars,
            "cross_entropy": weighted_loss / max(total_bars, 1),
            "top1_accuracy": weighted_accuracy / max(total_bars, 1),
            "folds": fold_rows,
        }
        rank = (
            -round(float(metrics["cross_entropy"]), 10),
            round(float(metrics["top1_accuracy"]), 10),
        )
        if best is None or rank > best[0]:
            best = (rank, l2_strength, metrics)
    if best is None:
        raise RuntimeError("Cross-validation produced no result")
    return best[1], best[2]


def _auto_cache_workers(requested: int | None, jobs: int) -> int:
    if jobs <= 0:
        return 1
    if requested is not None:
        return max(1, min(int(requested), jobs))
    return max(1, min(2, jobs, (os.cpu_count() or 2) // 2))


def _write_report_atomic(output_path: Path, report: dict[str, object]) -> None:
    atomic_write_json(output_path, report, ensure_ascii=False, indent=2)


def optimize_downbeat_parameters(
    request: DownbeatOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> DownbeatOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    ignored: list[SkippedOptimizationTrack] = []
    tracks_src = _list_4_4_tracks(library_dir, ignored)
    if len(tracks_src) < 3:
        raise RuntimeError("At least 3 analyzed 4/4 tracks are required")

    cache_jobs = [
        track
        for track in tracks_src
        if request.rebuild_cache
        or not _cache_is_current(
            _cache_path(cache_dir, str(track["uid"])), track, bool(request.use_hpss)
        )
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
                else "Downbeat feature cache is current"
            ),
        )

    if cache_jobs:
        with create_spawn_process_pool(workers) as executor:
            futures = {
                executor.submit(
                    _build_track_cache,
                    track,
                    str(cache_dir),
                    bool(request.use_hpss),
                ): track
                for track in cache_jobs
            }
            failed_uids: set[str] = set()
            for completed, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                uid = str(track["uid"])
                try:
                    future.result()
                except Exception as exc:
                    failed_uids.add(uid)
                    ignored.append(
                        skipped_track(
                            track,
                            f"Downbeat feature cache build failed: {exc}",
                        )
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
            beats, feature_raw = _load_current_cache(
                _cache_path(cache_dir, str(track["uid"])), track, bool(request.use_hpss)
            )
            bar_groups = _complete_bar_groups(
                beats, np.asarray(track["segments"], dtype=np.float64)
            )
            if not bar_groups.size:
                raise ValueError("no complete annotated 4/4 bars")
            prepared.append(
                {
                    "uid": str(track["uid"]),
                    "title": str(track["title"]),
                    "artist": str(track["artist"]),
                    "feature_z": standardize_downbeat_features(feature_raw),
                    "bar_groups": bar_groups,
                }
            )
        except Exception as exc:
            ignored.append(skipped_track(track, f"bar preparation failed: {exc}"))
        if feature_progress is not None:
            feature_progress(
                90 + int(round(index * 10 / max(prepare_total, 1))),
                f"Preparing bars {index}/{prepare_total}\n"
                f"{optimizer_track_name(track)}",
            )
    if len(prepared) < 3:
        raise RuntimeError("At least 3 tracks with complete annotated 4/4 bars are required")

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
    all_groups = _training_groups(prepared, np.arange(len(prepared)))
    if optimize_progress is not None:
        optimize_progress(90, "Fitting final downbeat model")
    weights, optimizer_diagnostics = _fit_conditional_softmax(all_groups, l2_strength)
    training_metrics = _group_metrics(all_groups, weights)
    metrics = cross_validation if cross_validated else training_metrics
    report = {
        "algorithm": "downbeat_conditional_softmax_features_v1",
        "profile": "downbeat-feature-probability",
        "track_count": len(prepared),
        "bar_count": int(all_groups.shape[0]),
        "objective": (
            "Conditional softmax within each annotated 4/4 bar; the exact "
            "downbeat is candidate 0 and the other three beats are hard negatives."
        ),
        "selected_l2_strength": l2_strength,
        "cross_validation": cross_validation,
        "optimizer": optimizer_diagnostics,
        "training_metrics": training_metrics,
        "skipped_tracks": [asdict(track) for track in ignored],
        "model": {
            "feature_names": list(DOWNBEAT_FEATURE_NAMES),
            "weights": weights.tolist(),
        },
    }
    if optimize_progress is not None:
        optimize_progress(97, "Writing downbeat parameter")
    _write_report_atomic(output_path, report)
    clear_downbeat_weight_cache()
    if optimize_progress is not None:
        optimize_progress(100, "Done")

    return DownbeatOptimizationResult(
        output_path=output_path,
        track_count=len(prepared),
        bar_count=int(all_groups.shape[0]),
        selected_l2_strength=float(l2_strength),
        cross_entropy=float(metrics["cross_entropy"]),
        top1_accuracy=float(metrics["top1_accuracy"]),
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
