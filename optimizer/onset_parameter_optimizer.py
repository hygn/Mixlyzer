from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.special import expit, log_expit, logsumexp, softmax
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from analyzer_core.beat.frame_features import extract_frame_features
from analyzer_core.beat.learned_onset import (
    ONSET_CHANNEL_NAMES,
    ONSET_CONTEXT_OFFSETS,
    ONSET_CONTEXT_DILATIONS,
    ONSET_FEATURE_NAMES,
    ONSET_MODEL_FORMAT,
    clear_onset_weight_cache,
    onset_channels,
    stack_context,
)
from core.audio.decoder import decode_to_memmap
from core.library_handler import LibraryDB
from optimizer import (
    SkippedOptimizationTrack,
    optimizer_track_name,
    skipped_track,
)
from core.concurrency.process_pool import create_spawn_process_pool
from utils.atomic_io import atomic_output_path, atomic_write_json


ProgressCallback = Callable[[int, str], None]
ONSET_FEATURE_CACHE_FORMAT = "mixlyzer_onset_feature_cache_v4"
L2_CANDIDATES = (0.0001, 0.001, 0.01, 0.1)
# Used when cross-validation is skipped (cv_folds < 2): the L2 chosen by the last cross-validated fit.
DEFAULT_L2_STRENGTH = 0.001

# Grid confusion loss. The beat tracker picks the (BPM, phase) grid on which the
# ODF folds highest, so its typical failures are grids that also collect ODF
# energy: the true period shifted by 1/4, 1/2 or 3/4 beat, and periods at 2/3,
# 3/4, 4/3 or 3/2 of the true one (BPM x 3/2, 4/3, 3/4, 2/3). In every window of
# annotated beats the mean model logit on the true grid competes with those
# grids in a softmax, which pushes the onset down wherever a wrong grid lands.
# Grids are scored like the tracker's fold: onset summed over the positions
# minus positions x the window mean (see _grid_candidates).
GRID_WINDOW_BEATS = 8
GRID_PHASE_SHIFTS = (0.25, 0.5, 0.75)        # beats, true period
GRID_PERIOD_RATIOS = (2 / 3, 3 / 4, 4 / 3, 3 / 2)  # alternative period / true period
GRID_RATIO_PHASES = (0.0, 0.25, 0.5, 0.75)   # fractions of the alternative period
GRID_LOSS_WEIGHT = 1.0
NEWTON_MAX_ITERATIONS = 50
NEWTON_TOLERANCE = 1e-6
# Frames near an annotated beat get a Gaussian target of this width (in frames).
TARGET_SIGMA_FRAMES = 0.75
TARGET_MIN = 0.01
# Everywhere else the onset energy is pushed to zero. The off-beat term is
# softplus(logit) = -log(1 - p), a convex upper bound of the onset p there, and
# weighs OFFBEAT_ENERGY_WEIGHT times the on-beat term. Only every n-th off-beat
# frame is used (weighted by n), which keeps the training set small without
# biasing the loss.
OFFBEAT_ENERGY_WEIGHT = 3.0
NEGATIVE_STRIDE = 6
# The frames at these beat subdivisions are where half-beat, 16th and triplet
# grids collect energy; each is always used and weighs SUBDIVISION_WEIGHT frames.
SUBDIVISION_POSITIONS = (1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4)
SUBDIVISION_WEIGHT = 4.0
MIN_BEATS = 16


@dataclass(frozen=True)
class OnsetOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    use_hpss: bool = True
    # The analyzer adds this offset to the tracked grid, so the model is trained
    # on the stored beats shifted earlier by it (both onset sources share it).
    beatgrid_offset_msec: float = 0.0
    rebuild_cache: bool = False
    # < 2 skips cross-validation: one fit on all tracks with l2_strength.
    cv_folds: int = 5
    # With cross-validation: True picks the L2 among L2_CANDIDATES, False evaluates l2_strength only.
    l2_sweep: bool = True
    l2_strength: float = DEFAULT_L2_STRENGTH
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class OnsetOptimizationResult:
    output_path: Path
    track_count: int
    frame_count: int
    selected_l2_strength: float
    cross_entropy: float
    beat_average_precision: float
    grid_top1_accuracy: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    # False: the metrics are measured on the training tracks (cross-validation skipped).
    cross_validated: bool = True


def _list_beat_tracks(
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
                sample_rate = int(np.asarray(archive["sr"]).item())
        except (KeyError, OSError, ValueError) as exc:
            if ignored is not None:
                ignored.append(skipped_track(track, f"invalid beat analysis: {exc}"))
            continue
        if beats.ndim != 1 or beats.size < MIN_BEATS or not np.all(np.isfinite(beats)):
            if ignored is not None:
                ignored.append(skipped_track(track, "invalid or insufficient beat grid"))
            continue
        track.update(beats=np.sort(beats), sample_rate=sample_rate)
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
                "frame_times",
                "channels",
                "channel_names",
                "use_hpss",
                "sample_rate",
                "audio_mtime_ns",
            }:
                return None
            frame_times = np.asarray(archive["frame_times"], dtype=np.float64)
            channels = np.asarray(archive["channels"])
            current = bool(
                str(np.asarray(archive["cache_format"]).item()) == ONSET_FEATURE_CACHE_FORMAT
                and tuple(np.asarray(archive["channel_names"]).astype(str)) == ONSET_CHANNEL_NAMES
                and bool(np.asarray(archive["use_hpss"]).item()) == bool(use_hpss)
                and int(np.asarray(archive["sample_rate"]).item()) == int(track["sample_rate"])
                and frame_times.ndim == 1
                and channels.shape == (frame_times.size, len(ONSET_CHANNEL_NAMES))
                and channels.dtype == np.dtype(np.float32)
                and np.all(np.isfinite(channels))
                and int(np.asarray(archive["audio_mtime_ns"]).item())
                == Path(track["audio_path"]).stat().st_mtime_ns
            )
            if current:
                return frame_times, np.asarray(channels, dtype=np.float32)
    except Exception:
        # A partial/corrupt cache is equivalent to a miss and is rebuilt.
        pass
    return None


def _build_track_cache(
    track: dict[str, object], cache_dir_text: str, use_hpss: bool
) -> str:
    """Process-worker entry point for one onset feature cache."""

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
        with atomic_output_path(cache_path) as temporary:
            np.savez_compressed(
                temporary,
                cache_format=np.asarray(ONSET_FEATURE_CACHE_FORMAT),
                frame_times=frames.frame_times.astype(np.float64),
                channels=onset_channels(frames),
                channel_names=np.asarray(ONSET_CHANNEL_NAMES, dtype="U64"),
                use_hpss=np.bool_(use_hpss),
                sample_rate=np.int64(sample_rate),
                audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns),
            )

    if threadpool_limits is None:
        build()
    else:
        with threadpool_limits(limits=1):
            build()
    return str(track["uid"])


def beat_targets(frame_times: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames inside the annotated beat range and their Gaussian beat targets."""
    beats = np.asarray(beats, dtype=np.float64)
    period = float(np.median(np.diff(beats)))
    inside = np.flatnonzero(
        (frame_times >= beats[0] - 0.5 * period) & (frame_times <= beats[-1] + 0.5 * period)
    )
    times = frame_times[inside]
    right = np.clip(np.searchsorted(beats, times), 1, beats.size - 1)
    distance = np.minimum(np.abs(times - beats[right - 1]), np.abs(beats[right] - times))
    hop_sec = float(np.median(np.diff(frame_times)))
    target = np.exp(-0.5 * np.square(distance / (TARGET_SIGMA_FRAMES * hop_sec)))
    return inside, target


def _grid_hypotheses() -> list[tuple[float, float]]:
    """(period ratio, phase in beats); the true grid (1, 0) comes first."""
    hypotheses = [(1.0, 0.0)] + [(1.0, shift) for shift in GRID_PHASE_SHIFTS]
    for ratio in GRID_PERIOD_RATIOS:
        hypotheses.extend((ratio, phase * ratio) for phase in GRID_RATIO_PHASES)
    return hypotheses


def _grid_candidates(
    inputs: np.ndarray, frame_times: np.ndarray, beats: np.ndarray
) -> np.ndarray:
    """Grid score inputs per beat window [N, H, D]; ``w @ g`` is the grid's score.

    Like the tracker's fold (peak minus the fold median), a grid scores the sum
    of the onset over its positions minus positions x the window's mean onset.
    So a faster grid (more positions) wins whenever the onset between beats is
    above the average, which is the BPM x4/3 / x3/2 failure. Positions are placed
    in beat units and mapped to time through the annotated beats, so local tempo
    changes are followed; inputs are interpolated linearly between frames, which
    keeps every score linear in the weights.
    """
    hypotheses = _grid_hypotheses()
    n_beats = beats.size
    starts = np.arange(0, n_beats - 1 - GRID_WINDOW_BEATS, GRID_WINDOW_BEATS)
    if starts.size == 0:
        return np.empty((0, len(hypotheses), inputs.shape[1]), dtype=np.float32)
    frame_axis = np.arange(frame_times.size, dtype=np.float64)
    beat_axis = np.arange(n_beats, dtype=np.float64)

    # Mean input over each window's frames (prefix sums).
    cumulative = np.zeros((inputs.shape[0] + 1, inputs.shape[1]), dtype=np.float64)
    np.cumsum(inputs, axis=0, out=cumulative[1:])
    lo = np.searchsorted(frame_times, beats[starts])
    hi = np.maximum(np.searchsorted(frame_times, beats[starts + GRID_WINDOW_BEATS]), lo + 1)
    window_mean = ((cumulative[hi] - cumulative[lo]) / (hi - lo)[:, np.newaxis]).astype(np.float32)

    out = np.empty((starts.size, len(hypotheses), inputs.shape[1]), dtype=np.float32)
    for h, (ratio, phase) in enumerate(hypotheses):
        offsets = np.arange(phase, GRID_WINDOW_BEATS, ratio)
        units = starts[:, np.newaxis] + offsets[np.newaxis, :]
        frames = np.interp(np.interp(units, beat_axis, beats), frame_times, frame_axis)
        lower = np.clip(np.floor(frames).astype(np.int64), 0, frame_times.size - 2)
        frac = (frames - lower)[..., np.newaxis].astype(np.float32)
        rows = inputs[lower] * (1.0 - frac) + inputs[lower + 1] * frac
        out[:, h] = (rows.sum(axis=1) - offsets.size * window_mean) / GRID_WINDOW_BEATS
    return out


def _subdivision_frames(frame_times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Frame nearest to every SUBDIVISION_POSITIONS point between annotated beats."""
    units = (np.arange(beats.size - 1)[:, np.newaxis] + np.asarray(SUBDIVISION_POSITIONS)).ravel()
    times = np.interp(units, np.arange(beats.size, dtype=np.float64), beats)
    return np.unique(np.clip(np.rint(np.interp(times, frame_times, np.arange(frame_times.size))), 0, frame_times.size - 1).astype(np.int64))


def _track_samples(
    frame_times: np.ndarray, channels: np.ndarray, beats: np.ndarray
) -> dict[str, np.ndarray]:
    """Frame samples (x, y, w, kind) and grid-confusion candidates (g) of one track."""
    return samples_from_inputs(stack_context(channels).astype(np.float32), frame_times, beats)


def samples_from_inputs(
    inputs: np.ndarray, frame_times: np.ndarray, beats: np.ndarray
) -> dict[str, np.ndarray]:
    """Training samples from per-frame model inputs [T, D].

    kind 0: near a beat (Gaussian target), 1: beat subdivision, 2: other off-beat.
    """
    beats = np.asarray(beats, dtype=np.float64)
    inside, target = beat_targets(frame_times, beats)
    near = target >= TARGET_MIN
    subdivision = np.isin(inside, _subdivision_frames(frame_times, beats)) & ~near
    keep = near | subdivision | (inside % NEGATIVE_STRIDE == 0)
    kind = np.where(near, 0, np.where(subdivision, 1, 2))[keep].astype(np.int8)
    weight = np.select([kind == 0, kind == 1], [1.0, SUBDIVISION_WEIGHT], float(NEGATIVE_STRIDE))
    return {
        "x": inputs[inside[keep]],
        "y": np.where(near[keep], target[keep], 0.0).astype(np.float32),
        "w": weight.astype(np.float32),
        "kind": kind,
        "g": _grid_candidates(inputs, frame_times, beats),
    }


class _TrainingSet:
    """Concatenated samples of several tracks, with the frame class balance."""

    def __init__(self, tracks: Sequence[dict[str, object]], indices: Sequence[int]):
        parts = [tracks[index] for index in indices]
        x = np.concatenate([part["x"] for part in parts])
        self.x = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float32)])
        target = np.concatenate([part["y"] for part in parts]).astype(np.float64)
        weight = np.concatenate([part["w"] for part in parts]).astype(np.float64)
        self.target = target
        self.weight = weight
        self.kind = np.concatenate([part["kind"] for part in parts])
        # On-beat term and off-beat energy term each normalized by their mass.
        positive_mass = max(float(np.sum(weight * target)), 1e-12)
        negative_mass = max(float(np.sum(weight * (1.0 - target))), 1e-12)
        total = 1.0 + OFFBEAT_ENERGY_WEIGHT
        self.w_pos = weight * target / positive_mass / total
        self.w_neg = OFFBEAT_ENERGY_WEIGHT * weight * (1.0 - target) / negative_mass / total
        grids = np.concatenate([part["g"] for part in parts])
        # The bias is the same on every grid of a window, so it gets a zero column.
        self.g = np.concatenate(
            [grids, np.zeros(grids.shape[:2] + (1,), dtype=np.float32)], axis=2
        )

    @property
    def n_params(self) -> int:
        return self.x.shape[1]


def _objective(
    data: _TrainingSet, params: np.ndarray, l2_strength: float, hessian: bool
) -> tuple[float, np.ndarray, np.ndarray | None, dict[str, float]]:
    """Frame cross-entropy + GRID_LOSS_WEIGHT * grid softmax loss + L2 (bias excluded)."""
    params32 = params.astype(np.float32)
    logits = (data.x @ params32).astype(np.float64)
    frame_loss = -float(np.sum(data.w_pos * log_expit(logits) + data.w_neg * log_expit(-logits)))
    prob = expit(logits)
    residual = (data.w_pos + data.w_neg) * prob - data.w_pos
    gradient = (residual.astype(np.float32) @ data.x).astype(np.float64)

    n_windows, _, n_params = data.g.shape
    scores = (data.g @ params32).astype(np.float64)
    grid_loss = float(np.mean(logsumexp(scores, axis=1) - scores[:, 0])) if n_windows else 0.0
    grid_prob = softmax(scores, axis=1) if n_windows else scores
    flat = data.g.reshape(-1, n_params)
    if n_windows:
        mean_grid = np.einsum("nh,nhd->nd", grid_prob.astype(np.float32), data.g).astype(np.float64)
        gradient += GRID_LOSS_WEIGHT * (mean_grid - data.g[:, 0]).mean(axis=0)

    penalty = np.ones(n_params)
    penalty[-1] = 0.0
    loss = frame_loss + GRID_LOSS_WEIGHT * grid_loss + l2_strength * float(np.sum(penalty * params**2))
    gradient += 2.0 * l2_strength * penalty * params

    hess = None
    if hessian:
        curvature = ((data.w_pos + data.w_neg) * prob * (1.0 - prob)).astype(np.float32)
        hess = ((data.x * curvature[:, np.newaxis]).T @ data.x).astype(np.float64)
        if n_windows:
            weighted = flat * grid_prob.reshape(-1, 1).astype(np.float32)
            grid_hess = (weighted.T @ flat).astype(np.float64) - mean_grid.T @ mean_grid
            hess += GRID_LOSS_WEIGHT * grid_hess / n_windows
        hess[np.diag_indices(n_params)] += 2.0 * l2_strength * penalty + 1e-9
    parts = {"frame_cross_entropy": frame_loss, "grid_cross_entropy": grid_loss}
    return loss, gradient, hess, parts


def _fit(
    data: _TrainingSet, l2_strength: float, initial: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, object]]:
    """Newton's method with backtracking; the objective is convex."""
    params = np.zeros(data.n_params) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    iterations = 0
    converged = False
    for iterations in range(1, NEWTON_MAX_ITERATIONS + 1):
        loss, gradient, hess, _ = _objective(data, params, l2_strength, hessian=True)
        step = np.linalg.solve(hess, gradient)
        decrement = float(gradient @ step)
        # Scores are float32 matrix products, so ~1e-7 is the loss resolution.
        if decrement < NEWTON_TOLERANCE:
            converged = True
            break
        scale = 1.0
        while True:
            candidate = params - scale * step
            if _objective(data, candidate, l2_strength, hessian=False)[0] <= loss - 1e-4 * scale * decrement:
                params = candidate
                break
            scale *= 0.5
            if scale < 1e-4:
                break
        if scale < 1e-4:
            converged = True  # no further descent at float32 resolution
            break
    if not np.all(np.isfinite(params)):
        raise RuntimeError("Onset weight optimization diverged")
    return params, {
        "optimizer": "newton",
        "iterations": int(iterations),
        "converged": bool(converged),
        "objective_loss": float(_objective(data, params, l2_strength, hessian=False)[0]),
        "l2_strength": float(l2_strength),
    }


def _metrics(data: _TrainingSet, params: np.ndarray) -> dict[str, float | int]:
    _, _, _, parts = _objective(data, params, 0.0, hessian=False)
    logits = (data.x @ params.astype(np.float32)).astype(np.float64)
    labels = data.target >= 0.5
    average_precision = (
        float(average_precision_score(labels, logits, sample_weight=data.weight))
        if labels.any() and not labels.all()
        else float("nan")
    )
    scores = data.g @ params.astype(np.float32)
    top1 = float(np.mean(np.argmax(scores, axis=1) == 0)) if scores.shape[0] else float("nan")
    # Tracker-style energy (p ** 1.5) off the beat relative to on the beat.
    energy = np.power(expit(logits), 1.5)
    on_beat = float(np.mean(energy[data.target >= 0.5])) if labels.any() else float("nan")
    return {
        "frame_count": int(data.x.shape[0]),
        "window_count": int(data.g.shape[0]),
        "cross_entropy": parts["frame_cross_entropy"] + GRID_LOSS_WEIGHT * parts["grid_cross_entropy"],
        "frame_cross_entropy": parts["frame_cross_entropy"],
        "grid_cross_entropy": parts["grid_cross_entropy"],
        "grid_top1_accuracy": top1,
        "beat_average_precision": average_precision,
        "offbeat_energy_ratio": float(np.mean(energy[data.kind == 2])) / max(on_beat, 1e-12),
        "subdivision_energy_ratio": float(np.mean(energy[data.kind == 1])) / max(on_beat, 1e-12),
    }


def _cross_validate(
    tracks: Sequence[dict[str, object]],
    folds: int,
    progress: ProgressCallback | None,
    candidates: Sequence[float] = L2_CANDIDATES,
) -> tuple[float, dict[str, object]]:
    n_splits = min(int(folds), len(tracks))
    if n_splits < 2:
        raise ValueError("Onset optimization requires at least two folds")
    indices = np.arange(len(tracks), dtype=np.int32)
    splits = list(
        GroupKFold(n_splits=n_splits).split(
            indices, np.zeros(len(tracks), dtype=np.int8), groups=indices
        )
    )
    total_fits = len(candidates) * len(splits)
    completed_fits = 0
    rows: dict[float, list[dict[str, object]]] = {l2: [] for l2 in candidates}
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        train = _TrainingSet(tracks, train_indices)
        test = _TrainingSet(tracks, test_indices)
        params = None
        # Strongest regularization first; each fit warm-starts the next.
        for l2_strength in sorted(candidates, reverse=True):
            params, diagnostics = _fit(train, l2_strength, params)
            rows[l2_strength].append(
                {
                    "fold": fold,
                    "test_uids": [str(tracks[index]["uid"]) for index in test_indices],
                    "optimizer": diagnostics,
                    "metrics": _metrics(test, params),
                }
            )
            completed_fits += 1
            if progress is not None:
                progress(
                    int(round(completed_fits * 85 / total_fits)),
                    f"Cross-validation fold {fold}/{n_splits}, L2={l2_strength:g}",
                )
        del train, test

    best: tuple[float, float, dict[str, object]] | None = None
    for l2_strength in candidates:
        fold_rows = rows[l2_strength]
        metrics: dict[str, object] = {}
        for key in ("cross_entropy", "frame_cross_entropy", "grid_cross_entropy",
                    "grid_top1_accuracy", "beat_average_precision",
                    "offbeat_energy_ratio", "subdivision_energy_ratio"):
            weights = [float(row["metrics"]["window_count" if key.startswith("grid") else "frame_count"])
                       for row in fold_rows]
            values = [float(row["metrics"][key]) for row in fold_rows]
            metrics[key] = float(np.average(values, weights=weights))
        metrics["frame_count"] = int(sum(row["metrics"]["frame_count"] for row in fold_rows))
        metrics["window_count"] = int(sum(row["metrics"]["window_count"] for row in fold_rows))
        metrics["folds"] = fold_rows
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


def optimize_onset_parameters(
    request: OnsetOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> OnsetOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_hpss = bool(request.use_hpss)
    beatgrid_offset_sec = float(request.beatgrid_offset_msec) / 1000.0

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
                else "Onset feature cache is current"
            ),
        )

    if cache_jobs:
        with create_spawn_process_pool(workers) as executor:
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
                        skipped_track(track, f"Onset feature cache build failed: {exc}")
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
            cached = _read_current_cache(
                _cache_path(cache_dir, str(track["uid"])), track, use_hpss
            )
            if cached is None:
                raise ValueError("onset feature cache is not current")
            frame_times, channels = cached
            samples = _track_samples(
                frame_times,
                channels,
                np.asarray(track["beats"], dtype=np.float64) - beatgrid_offset_sec,
            )
            if not np.any(samples["y"] >= 0.5):
                raise ValueError("no annotated beats inside the audio")
            prepared.append({"uid": str(track["uid"]), **samples})
        except Exception as exc:
            ignored.append(skipped_track(track, f"frame preparation failed: {exc}"))
        if feature_progress is not None:
            feature_progress(
                90 + int(round(index * 10 / max(prepare_total, 1))),
                f"Preparing frames {index}/{prepare_total}\n{optimizer_track_name(track)}",
            )
    if len(prepared) < 3:
        raise RuntimeError("At least 3 tracks with annotated beats are required")

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
    everything = _TrainingSet(prepared, np.arange(len(prepared)))
    if optimize_progress is not None:
        optimize_progress(90, "Fitting final onset model")
    params, optimizer_diagnostics = _fit(everything, l2_strength)
    weights, bias = params[:-1], float(params[-1])
    training_metrics = _metrics(everything, params)
    metrics = cross_validation if cross_validated else training_metrics
    report = {
        "algorithm": "onset_logistic_grid_confusion_v3",
        "profile": "learned-beat-onset",
        "track_count": len(prepared),
        "frame_count": int(everything.x.shape[0]),
        "window_count": int(everything.g.shape[0]),
        "objective": (
            "Class-balanced logistic regression on frames (Gaussian target, sigma "
            f"{TARGET_SIGMA_FRAMES} frames, around each annotated beat) plus "
            f"{GRID_LOSS_WEIGHT:g} x a softmax over beat grids in every "
            f"{GRID_WINDOW_BEATS}-beat window: the annotated grid against the same "
            f"period shifted by {', '.join(f'{v:g}' for v in GRID_PHASE_SHIFTS)} beat and "
            "periods x2/3, x3/4, x4/3, x3/2 (4 phases each), each scored like the tracker "
            "fold: logit summed over the grid positions minus positions x the window mean."
        ),
        "use_hpss": use_hpss,
        "beatgrid_offset_msec": float(request.beatgrid_offset_msec),
        "selected_l2_strength": l2_strength,
        "cross_validation": cross_validation,
        "optimizer": optimizer_diagnostics,
        "training_metrics": training_metrics,
        "skipped_tracks": [asdict(track) for track in ignored],
        "model": {
            "format": ONSET_MODEL_FORMAT,
            "channel_names": list(ONSET_CHANNEL_NAMES),
            "context_offsets": list(ONSET_CONTEXT_OFFSETS),
            "context_dilations": list(ONSET_CONTEXT_DILATIONS),
            "feature_names": list(ONSET_FEATURE_NAMES),
            "weights": weights.tolist(),
            "bias": bias,
        },
    }
    if optimize_progress is not None:
        optimize_progress(97, "Writing onset parameter")
    atomic_write_json(output_path, report, ensure_ascii=False, indent=2)
    clear_onset_weight_cache()
    if optimize_progress is not None:
        optimize_progress(100, "Done")

    return OnsetOptimizationResult(
        output_path=output_path,
        track_count=len(prepared),
        frame_count=int(everything.x.shape[0]),
        selected_l2_strength=float(l2_strength),
        cross_entropy=float(metrics["cross_entropy"]),
        beat_average_precision=float(metrics["beat_average_precision"]),
        grid_top1_accuracy=float(metrics["grid_top1_accuracy"]),
        skipped_tracks=tuple(ignored),
        cross_validated=cross_validated,
    )
