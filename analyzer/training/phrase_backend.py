"""Phrase parameter optimization backend.

1. Boundary-specialist GBM runs at every BEAT and returns boundary
   probabilities for threshold/peak picking.
2. Label-specialist GBM runs on each segment produced by the boundary stage and
   returns a probability distribution over phrase labels.
3. A fixed-boundary label DP assigns the final label sequence using label-GBM
   log-probabilities plus learned transition and optional length priors.

Feature policy:
    Uses the current production structure feature extractor
    ``analyzer.phrase.structure.extract_song_features`` at beat
    resolution (``family_beat``).

Long segments are not split into 8-bar chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np

from analyzer.training import SkippedOptimizationTrack, library_tracks, skipped_track

from analyzer.phrase.structure import (
    FeatureConfig,
    build_predictor_grid,
    extract_song_features,
)
from analyzer.phrase.model_features import (
    feature_z_from_acoustic,
    segment_feature_matrix,
)
from analyzer.files import atomic_output_path
from analyzer.training.library_tracks import Annotation
from analyzer.training.text import msg


EPS = 1e-9
BEAT_FEATURE_CACHE_FORMAT = "mixlyzer_phrase_feature_cache_v1"
_BEAT_FEATURE_CACHE_KEYS = frozenset(
    {
        "cache_format",
        "beat_times_sec",
        "beat_edges_sec",
        "feature_z",
        "feature_names",
        "feature_config",
        "res_type",
        "audio_mtime_ns",
    }
)
_PHRASE_TIME_TOLERANCE_SEC = 1e-4   # the stored phrases' float32 resolution (core.analysis_library.phrases.MIN_PHRASE_DUR)
# Stored phrase/tempo-segment ends sit up to ~1 ms before duration_sec, so track-edge
# coverage is checked with a looser tolerance than segment contiguity.
_COVERAGE_TOLERANCE_SEC = 0.01


@dataclass
class LabelModel:
    clf: object
    labels: list[str]
    transition: np.ndarray
    length_mu: np.ndarray
    length_sigma: np.ndarray


def _load_phrase_annotation(
    annotation: Annotation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The phrase annotation of a track (as the library's owner gave it)."""
    parts = {"start": annotation.phrase_starts, "end": annotation.phrase_ends, "label": annotation.phrase_labels}
    if all(part is None for part in parts.values()):
        # A track without Phrase analysis is not optimizer training data.
        empty = np.asarray([], dtype=np.float64)
        return empty, empty.copy(), np.asarray([], dtype="U1")
    missing = [name for name, part in parts.items() if part is None]
    if missing:
        raise ValueError(f"Incomplete current Phrase structure: missing {', '.join(sorted(missing))}")
    starts = np.asarray(annotation.phrase_starts, dtype=np.float64).ravel()
    ends = np.asarray(annotation.phrase_ends, dtype=np.float64).ravel()
    labels = np.asarray(annotation.phrase_labels).astype(str).ravel()
    if not (starts.size == ends.size == labels.size):
        raise ValueError("Mismatched current Phrase arrays")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Non-finite current Phrase timestamps")

    # Float32 persistence can collapse a tiny edited fragment to a
    # zero-length row. It contains no trainable interval and must not overwrite
    # the real label at the same boundary.
    keep = (ends - starts) > _PHRASE_TIME_TOLERANCE_SEC
    starts = starts[keep]
    ends = ends[keep]
    labels = labels[keep]
    if not starts.size:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty.copy(), np.asarray([], dtype="U1")
    if any(not str(label).strip() for label in labels):
        raise ValueError("Empty current Phrase label")
    if np.any(np.diff(starts) < -_PHRASE_TIME_TOLERANCE_SEC):
        raise ValueError("Unsorted current Phrase segments")
    if np.any(starts[1:] < ends[:-1] - _PHRASE_TIME_TOLERANCE_SEC):
        raise ValueError("Overlapping current Phrase segments")
    if np.any(starts[1:] > ends[:-1] + _PHRASE_TIME_TOLERANCE_SEC):
        raise ValueError("Phrase optimizer requires contiguous current annotations")
    return starts, ends, labels


def _load_duration_sec(annotation: Annotation) -> float:
    """Audio duration stored with the analysis, or inf when it is unavailable."""
    value = annotation.duration_sec
    if value is not None and np.isfinite(value) and value > 0.0:
        return float(value)
    return float("inf")


def predictor_grid(track: dict):
    """The track's predictor grid from its annotation's beats and tempo rows (structure.build_predictor_grid)."""
    annotation = track["annotation"]
    for name in ("beats", "tempo_segments"):
        if getattr(annotation, name) is None:
            raise KeyError(f"the stored analysis has no {name}")
    return build_predictor_grid(annotation.beats, annotation.tempo_segments)


def _list_annotated_tracks(
    library_dir: Path,
    ignored: list[SkippedOptimizationTrack] | None = None,
) -> list[dict[str, object]]:
    """The tracks with phrase annotations, of the tracks the library's owner gave this run with their annotations
    (analyzer.training.library_tracks; ``library_dir`` is kept for the callers' signatures)."""
    out: list[dict[str, object]] = []
    for track in library_tracks.tracks():
        audio_path = Path(track["audio_path"])
        track.update(audio_path=audio_path)
        annotation = track["annotation"]
        if annotation is None:
            if ignored is not None:
                reason = track.get("error")
                ignored.append(skipped_track(track, msg("invalid Phrase annotation: {error}", error=reason)
                                             if reason else msg("analysis NPZ is missing")))
            continue
        if not audio_path.is_file():
            if ignored is not None:
                ignored.append(skipped_track(track, msg("audio file is missing")))
            continue
        try:
            starts, ends, labels = _load_phrase_annotation(annotation)
        except Exception as exc:
            if ignored is not None:
                ignored.append(
                    skipped_track(track, msg("invalid Phrase annotation: {error}", error=exc))
                )
            continue
        if starts.size < 2:
            if starts.size and ignored is not None:
                ignored.append(
                    skipped_track(track, msg("fewer than two usable Phrase segments"))
                )
            continue
        track.update(
            duration_sec=_load_duration_sec(annotation),
            phrase_starts_sec=starts,
            phrase_ends_sec=ends,
            phrase_labels=labels,
        )
        out.append(track)
    return sorted(out, key=lambda row: str(row["uid"]))


def _cache_array_matches(cache, key: str, current: np.ndarray) -> bool:
    cached = np.asarray(cache[key])
    expected = np.asarray(current)
    if cached.shape != expected.shape:
        return False
    if np.issubdtype(expected.dtype, np.floating):
        return bool(np.allclose(cached, expected, rtol=1e-7, atol=1e-6, equal_nan=True))
    return bool(np.array_equal(cached, expected))


def _feature_names_for(acoustic) -> np.ndarray:
    names: list[str] = []
    for family in ("timbre", "harmony", "rhythm", "texture"):
        size = int(np.asarray(acoustic.family_beat[family]).shape[0])
        names.extend(f"{family}_{index:03d}" for index in range(size))
    return np.asarray(names, dtype="U32")


def _feature_config_signature(config: FeatureConfig) -> np.ndarray:
    numeric_values = []
    for field in fields(config):
        if field.name == "res_type":
            continue
        value = getattr(config, field.name)
        numeric_values.append(float(value) if value is not None else -1.0)
    return np.asarray(
        numeric_values,
        dtype=np.float64,
    )


def _beat_feature_cache_metadata_is_current(
    cache,
    audio_path: Path,
    grid,
    config: FeatureConfig,
) -> bool:
    """Accept exactly the cache structure written by the current optimizer."""

    if set(cache.files) != _BEAT_FEATURE_CACHE_KEYS:
        return False
    if str(np.asarray(cache["cache_format"]).item()) != BEAT_FEATURE_CACHE_FORMAT:
        return False
    if int(np.asarray(cache["audio_mtime_ns"]).item()) != audio_path.stat().st_mtime_ns:
        return False
    if not _cache_array_matches(
        cache,
        "beat_times_sec",
        np.asarray(grid.beat_times_sec, dtype=np.float64),
    ):
        return False
    if not _cache_array_matches(
        cache, "beat_edges_sec", np.asarray(grid.beat_edges_sec, dtype=np.float64)
    ):
        return False
    if not _cache_array_matches(cache, "feature_config", _feature_config_signature(config)):
        return False
    return str(np.asarray(cache["res_type"]).item()) == str(config.res_type)


def _beat_feature_cache_is_current(
    track: dict[str, object],
    cache_dir: Path,
    config: FeatureConfig,
    grid=None,
) -> bool:
    """Cheaply validate cache provenance without decompressing feature_z."""

    uid = str(track["uid"])
    audio_path = Path(track["audio_path"])
    cache_path = Path(cache_dir) / f"{uid}.npz"
    if not cache_path.exists():
        return False
    try:
        current_grid = grid if grid is not None else predictor_grid(track)
        with np.load(cache_path, allow_pickle=False) as cache:
            return _beat_feature_cache_metadata_is_current(
                cache, audio_path, current_grid, config
            )
    except Exception:
        return False


def _current_beat_features(
    track: dict[str, object],
    cache_dir: Path,
    config: FeatureConfig,
    rebuild: bool,
    *,
    grid=None,
):
    uid = str(track["uid"])
    audio_path = Path(track["audio_path"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{uid}.npz"
    grid = grid if grid is not None else predictor_grid(track)
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cache:
            if _beat_feature_cache_metadata_is_current(
                cache, audio_path, grid, config
            ):
                feature_z = np.asarray(cache["feature_z"], dtype=np.float64)
                feature_names = np.asarray(cache["feature_names"]).astype(str)
                if feature_z.ndim != 2 or feature_z.shape[0] != grid.n_beats:
                    raise ValueError("Current feature_z shape does not match beat grid")
                if feature_names.ndim != 1 or feature_names.size != feature_z.shape[1]:
                    raise ValueError("Current feature_names do not match feature_z")
                return {
                    "beat_times_sec": np.asarray(cache["beat_times_sec"], dtype=np.float64),
                    "feature_z": feature_z,
                    "grid": grid,
                    "cache_hit": True,
                }

    acoustic = extract_song_features(audio_path, grid, config)
    feature_z = feature_z_from_acoustic(acoustic)
    feature_names = _feature_names_for(acoustic)
    beat_times = np.asarray(grid.beat_times_sec, dtype=np.float64)
    # Replace atomically so cancelling the optimizer cannot leave a truncated
    # cache that every later run has to rediscover and rebuild.
    with atomic_output_path(cache_path) as temporary:
        np.savez_compressed(
            temporary,
            cache_format=np.asarray(BEAT_FEATURE_CACHE_FORMAT),
            beat_times_sec=beat_times,
            beat_edges_sec=np.asarray(grid.beat_edges_sec, dtype=np.float64),
            feature_z=feature_z,
            feature_names=feature_names,
            feature_config=_feature_config_signature(config),
            res_type=np.asarray(config.res_type),
            audio_mtime_ns=np.int64(audio_path.stat().st_mtime_ns),
        )
    return {
        "beat_times_sec": beat_times,
        "feature_z": np.asarray(feature_z, dtype=np.float64),
        "grid": grid,
        "cache_hit": False,
    }


def _nearest_beat_index(beat_times: np.ndarray, time_sec: float) -> int:
    insertion = int(np.searchsorted(beat_times, float(time_sec)))
    candidates = [idx for idx in (insertion - 1, insertion) if 0 <= idx < beat_times.size]
    if not candidates:
        return int(np.clip(insertion, 0, beat_times.size - 1))
    return min(candidates, key=lambda idx: abs(float(beat_times[idx]) - float(time_sec)))


def _gt_segments_beats(track: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    beats = np.asarray(track["beat_times_sec"], dtype=np.float64)
    starts = np.asarray(track["phrase_starts_sec"], dtype=np.float64).reshape(-1)
    ends = np.asarray(track["phrase_ends_sec"], dtype=np.float64).reshape(-1)
    labels = [str(x) for x in track["phrase_labels"]]
    n = int(beats.size)
    if not (starts.size == ends.size == len(labels)):
        raise ValueError(f"Mismatched Phrase optimizer annotations for {track['uid']}")
    if starts[0] > beats[0] + _COVERAGE_TOLERANCE_SEC:
        raise ValueError(
            f"Phrase optimizer requires annotation coverage from the first beat: {track['uid']}"
        )
    # Beat grids may extend slightly past the end of the audio (edited grids used to), and
    # such beats cannot be annotated; they are simply absorbed by the last segment below.
    audio_end = float(track.get("duration_sec", float("inf")))
    audible = beats[beats <= audio_end + _COVERAGE_TOLERANCE_SEC]
    last_audible_beat = float(audible[-1]) if audible.size else float(beats[-1])
    if ends[-1] < last_audible_beat - _COVERAGE_TOLERANCE_SEC:
        raise ValueError(
            f"Phrase optimizer requires annotation coverage through the last beat: {track['uid']}"
        )
    pairs: list[tuple[int, str]] = []
    for start_raw, label in zip(starts, labels, strict=True):
        start = float(start_raw)
        beat = 0 if start <= beats[0] + 1e-6 else _nearest_beat_index(beats, start)
        pairs.append((int(np.clip(beat, 0, n)), label))

    dedup: list[tuple[int, str]] = []
    for beat, label in pairs:
        if dedup and beat <= dedup[-1][0]:
            dedup[-1] = (dedup[-1][0], label)
        else:
            dedup.append((beat, label))
    if dedup[0][0] != 0:
        dedup.insert(0, (0, labels[0]))

    bounds = [0]
    out_labels: list[str] = []
    for (s, label), (e, _next_label) in zip(dedup, dedup[1:]):
        if int(e) > int(s):
            if int(s) > bounds[-1]:
                bounds.append(int(s))
            bounds.append(int(e))
            out_labels.append(str(label))
    if dedup and dedup[-1][0] < n:
        if int(dedup[-1][0]) > bounds[-1]:
            bounds.append(int(dedup[-1][0]))
        bounds.append(n)
        out_labels.append(str(dedup[-1][1]))
    if len(bounds) < 2:
        return np.asarray([0, n], dtype=np.int32), [labels[0]]
    return np.asarray(bounds, dtype=np.int32), out_labels


def _eval_target(track: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    return _gt_segments_beats(track)


def _fit_boundary_gbm(
    tracks: list[dict[str, object]],
    indices: Iterable[int],
    targets: list[tuple[np.ndarray, list[str]]],
    *,
    positive_radius: int,
    negative_guard: int,
    seed: int,
    learning_rate: float = 0.05,
    max_iter: int = 360,
    max_leaf_nodes: int = 15,
    max_depth: int = 3,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
):
    from sklearn.ensemble import HistGradientBoostingClassifier

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sws: list[np.ndarray] = []
    for idx in indices:
        track = tracks[int(idx)]
        features = np.asarray(track["boundary_feature_z"], dtype=np.float64)
        valid = np.asarray(track["valid_mask"], dtype=bool)
        n = features.shape[0]
        y = np.zeros(n, dtype=np.int8)
        use = valid.copy()
        for boundary in targets[int(idx)][0]:
            b = int(boundary)
            if b <= 0 or b >= n:
                continue
            lo = max(0, b - int(positive_radius))
            hi = min(n, b + int(positive_radius) + 1)
            y[lo:hi] = 1
            if negative_guard > positive_radius:
                glo = max(0, b - int(negative_guard))
                ghi = min(n, b + int(negative_guard) + 1)
                guard = np.ones(ghi - glo, dtype=bool)
                plo = max(lo, glo) - glo
                phi = min(hi, ghi) - glo
                guard[plo:phi] = False
                use[glo:ghi] &= ~guard
        mask = use
        pos = max(float(np.mean(y[mask])), 1e-6)
        weights = np.where(y[mask] > 0, (1.0 - pos) / pos, 1.0)
        xs.append(features[mask])
        ys.append(y[mask])
        sws.append(weights)

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sample_weight = np.concatenate(sws, axis=0).astype(np.float64)
    sample_weight *= sample_weight.size / max(float(sample_weight.sum()), EPS)

    clf = HistGradientBoostingClassifier(
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        max_leaf_nodes=int(max_leaf_nodes),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=float(l2_regularization),
        early_stopping=False,
        random_state=int(seed),
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def _fit_label_model(
    tracks: list[dict[str, object]],
    indices: Iterable[int],
    targets: list[tuple[np.ndarray, list[str]]],
    *,
    seed: int,
    boundary_jitter_views: int = 0,
    boundary_jitter_beats: int = 2,
) -> LabelModel:
    from sklearn.ensemble import HistGradientBoostingClassifier

    xs: list[np.ndarray] = []
    ys: list[str] = []
    trans_counts: dict[tuple[str, str], float] = {}
    lengths: dict[str, list[float]] = {}
    for idx in indices:
        track = tracks[int(idx)]
        bounds, labels = targets[int(idx)]
        X = segment_feature_matrix(
            np.asarray(track["feature_z"], dtype=np.float64),
            bounds,
            track["label_context_beats"],
        )
        if X.shape[0] != len(labels):
            raise ValueError(f"segment feature/label mismatch for {track['uid']}")
        xs.append(X)
        ys.extend(str(label) for label in labels)
        rng = np.random.default_rng(int(seed) * 1009 + int(idx))
        for _view in range(max(0, int(boundary_jitter_views))):
            jittered = np.asarray(bounds, dtype=np.int32).copy()
            if jittered.size > 2:
                offsets = rng.integers(
                    -max(0, int(boundary_jitter_beats)),
                    max(0, int(boundary_jitter_beats)) + 1,
                    size=jittered.size - 2,
                )
                for boundary_index, offset in enumerate(offsets, start=1):
                    low = int(jittered[boundary_index - 1]) + 1
                    high = int(jittered[boundary_index + 1]) - 1
                    if low <= high:
                        jittered[boundary_index] = int(
                            np.clip(jittered[boundary_index] + int(offset), low, high)
                        )
            jittered_X = segment_feature_matrix(
                np.asarray(track["feature_z"], dtype=np.float64),
                jittered,
                track["label_context_beats"],
            )
            if jittered_X.shape[0] == len(labels):
                xs.append(jittered_X)
                ys.extend(str(label) for label in labels)
        prev = "START"
        for (s, e), label in zip(zip(bounds[:-1], bounds[1:]), labels):
            lab = str(label)
            trans_counts[(prev, lab)] = trans_counts.get((prev, lab), 0.0) + 1.0
            lengths.setdefault(lab, []).append(float(int(e) - int(s)))
            prev = lab
        trans_counts[(prev, "END")] = trans_counts.get((prev, "END"), 0.0) + 1.0

    X = np.concatenate(xs, axis=0)
    y = np.asarray(ys, dtype=object)
    label_names = sorted(str(v) for v in np.unique(y))
    counts = {label: int(np.sum(y == label)) for label in label_names}
    sample_weight = np.asarray([1.0 / max(counts[str(label)], 1) for label in y], dtype=np.float64)
    sample_weight *= sample_weight.size / max(float(sample_weight.sum()), EPS)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=360,
        max_leaf_nodes=15,
        max_depth=3,
        min_samples_leaf=8,
        l2_regularization=0.5,
        early_stopping=False,
        random_state=int(seed),
    )
    clf.fit(X, y, sample_weight=sample_weight)
    labels = [str(x) for x in clf.classes_]

    L = len(labels)
    start_state, end_state = L, L + 1
    index = {label: i for i, label in enumerate(labels)}
    index["START"] = start_state
    index["END"] = end_state
    counts_mat = np.full((L + 2, L + 2), 0.3, dtype=np.float64)
    for (src, dst), count in trans_counts.items():
        if src in index and dst in index:
            counts_mat[index[src], index[dst]] += count
    counts_mat[end_state, :] = EPS
    transition = np.log(counts_mat / counts_mat.sum(axis=1, keepdims=True))

    length_mu = np.zeros(L, dtype=np.float64)
    length_sigma = np.ones(L, dtype=np.float64)
    for label, i in index.items():
        if label in {"START", "END"}:
            continue
        arr = np.log(np.maximum(np.asarray(lengths.get(label, [16.0]), dtype=np.float64), 1.0))
        length_mu[i] = float(arr.mean())
        length_sigma[i] = float(max(arr.std(), 0.25))

    return LabelModel(
        clf=clf,
        labels=labels,
        transition=transition,
        length_mu=length_mu,
        length_sigma=length_sigma,
    )
