"""Per-beat downbeat features: how much each beat's harmony, bass, timbre, melody and rhythm change against
its neighbours (self-similarity novelty), its onset accents per instrument, and the bar-level periodicity of
the onset curves. The beat tracker's contextual downbeat model (``general/bars.py``) reads them."""

from __future__ import annotations

import numpy as np

from analyzer.beat.frame_features import FrameFeatures
from analyzer.beat.periodicity import (
    PERIODICITY_CURVE_NAMES,
    local_periodicity,
    periodicity_curves,
    rotate,
)


def _beat_ranges(
    beat_times_sec: np.ndarray,
    duration_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    beats = np.asarray(beat_times_sec, dtype=np.float64).reshape(-1)
    beats = np.unique(beats[np.isfinite(beats)])
    beats = beats[(beats >= 0.0) & (beats < duration_sec)]
    if beats.size < 2:
        raise ValueError("beat_times_sec must contain at least two valid beats")

    median_interval = float(np.median(np.diff(beats)))
    final_end = min(duration_sec, float(beats[-1] + median_interval))
    ends = np.concatenate([beats[1:], np.asarray([final_end])])
    valid = ends > beats + 1e-4
    return beats[valid].astype(np.float32), ends[valid].astype(np.float32)


def _robust_standardize(features: np.ndarray) -> np.ndarray:
    median = np.median(features, axis=1, keepdims=True)
    mad = np.median(np.abs(features - median), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-5)
    return ((features - median) / scale).astype(np.float32)


# Per-beat features and their SSM metric, in a fixed order. "cos" -> cosine
# SSM (tonal/normalized features), "rbf" -> RBF SSM (magnitude features). The
# log-mel "timbre" feature was dropped: a library-wide ablation showed it hurt
# downbeat-phase detection (lower F1 when included). Spectral contrast was
# dropped too (no effect on the downbeat or the learned onset).
_FEATURE_SPECS: tuple[tuple[str, str], ...] = (
    ("harmonic", "cos"),      # CQT chroma -- chord changes on the downbeat
    ("rhythm", "rbf"),        # onset bands + rms + band ratios
    ("bass_chroma", "cos"),   # low-register chroma -- root/bass changes
    ("mfcc", "rbf"),          # MFCC timbral texture
    ("tonnetz", "cos"),       # tonal centroid (complements chroma)
    ("melody", "cos"),        # melody-register chroma -- melodic phrase repetition
)
# Onset curves pooled per beat: percussive onset, harmonic (note / chord)
# onset, the kick band of the percussive flux and melody note starts.
_ONSET_SOURCES: tuple[str, ...] = ("onset", "harmonic_onset", "kick_onset", "melody_onset")
# Bar-level periodicity: every shared onset curve's local pulse at 4 and 2 beats
# over +-8 beats, rotated so this beat is the cycle start (bar / backbeat
# patterns of each instrument, like a learned drum-pattern template).
_BAR_PERIODS_BEATS: tuple[float, ...] = (4.0, 2.0)
_BAR_WINDOW_BEATS = 8.0


def _onset_column_names(source: str) -> tuple[str, ...]:
    if source == "onset":
        return (
            "onset_strength",
            "onset_local_accent_4beat",
            "onset_local_accent_8beat",
            "onset_attack",
            "pre_downbeat_fill",
            "onset_four_beat_recurrence",
        )
    return (
        f"{source}_strength",
        f"{source}_local_accent_4beat",
        f"{source}_local_accent_8beat",
        f"{source}_attack",
        f"{source}_pre_downbeat_fill",
        f"{source}_four_beat_recurrence",
    )


_DOWNBEAT_FEATURE_NAMES = tuple(
    feature_name
    for representation, _metric in _FEATURE_SPECS
    for feature_name in (
        f"{representation}_previous_change",
        f"{representation}_symmetric_change",
        f"{representation}_novelty_1beat",
        f"{representation}_novelty_2beat",
        f"{representation}_novelty_4beat",
        f"{representation}_local_distinctiveness",
        f"{representation}_four_beat_recurrence_advantage",
    )
) + _onset_column_names("onset") + (
    "rhythm_patch_energy",
    "rhythm_patch_local_accent",
) + _onset_column_names("harmonic_onset") + _onset_column_names("kick_onset") + _onset_column_names(
    "melody_onset"
) + tuple(
    f"bar_periodicity_{curve}_{period:g}beat_{part}"
    for part in ("in_phase", "quadrature")
    for curve in PERIODICITY_CURVE_NAMES
    for period in _BAR_PERIODS_BEATS
)


def _beat_patch_rows(
    frames: np.ndarray,
    frame_times: np.ndarray,
    beat_starts: np.ndarray,
    beat_ends: np.ndarray,
    phase_bins: int,
    l2: bool,
) -> np.ndarray:
    """Per-beat flattened patches: each beat resampled to ``phase_bins`` and
    either L2-normalized per phase column (cosine features) or mean-centered
    (RBF features). Each beat is sampled at ``phase_bins`` evenly spaced times
    from its start (linear interpolation, edges held)."""
    starts = np.asarray(beat_starts, dtype=np.float64)
    steps = (np.asarray(beat_ends, dtype=np.float64) - starts) / phase_bins
    # np.linspace(start, end, phase_bins, endpoint=False, dtype=float32) per beat.
    targets = (np.arange(phase_bins, dtype=np.float64)[np.newaxis, :] * steps[:, np.newaxis]
               + starts[:, np.newaxis]).astype(np.float32).reshape(-1)
    patches = np.empty((starts.size, frames.shape[0], phase_bins), dtype=np.float32)
    for channel in range(frames.shape[0]):
        patches[:, channel, :] = np.interp(
            targets,
            frame_times,
            frames[channel],
            left=float(frames[channel, 0]),
            right=float(frames[channel, -1]),
        ).reshape(starts.size, phase_bins)
    if l2:
        columns = patches.transpose(0, 2, 1)
        columns /= np.maximum(np.linalg.norm(columns, axis=2, keepdims=True), 1e-8)
        return np.ascontiguousarray(columns).reshape(starts.size, -1)
    return (patches - patches.mean(axis=(1, 2), keepdims=True)).reshape(starts.size, -1)


def _extract_beat_features(
    frames: FrameFeatures,
    beat_times_sec: np.ndarray,
    beat_phase_bins: int,
    harmonic_phase_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Pool the shared frame features to beats.

    Returns ``(beat_starts, beat_ends, rows)`` where ``rows`` are the per-beat
    feature matrices in :data:`_FEATURE_SPECS` order."""
    beat_starts, beat_ends = _beat_ranges(beat_times_sec, frames.duration_sec)
    frame_times = frames.frame_times.astype(np.float32)
    rhythm_frames = _robust_standardize(
        np.concatenate(
            [frames.onset_bands, frames.rms_db[np.newaxis, :], frames.band_ratios],
            axis=0,
        ).astype(np.float32)
    )

    frame_sets = {
        "harmonic": (frames.chroma, harmonic_phase_bins),
        "rhythm": (rhythm_frames, beat_phase_bins),
        "bass_chroma": (frames.bass_chroma, harmonic_phase_bins),
        "mfcc": (_robust_standardize(frames.mfcc), beat_phase_bins),
        "tonnetz": (frames.tonnetz, harmonic_phase_bins),
        "melody": (frames.melody_chroma, harmonic_phase_bins),
    }
    rows = [
        _beat_patch_rows(frame_sets[name][0], frame_times, beat_starts, beat_ends,
                         frame_sets[name][1], l2=(metric == "cos"))
        for name, metric in _FEATURE_SPECS
    ]
    return beat_starts, beat_ends, rows


def _row_normalize(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True),
        1e-8,
    )


def _cosine_ssm(features: np.ndarray) -> np.ndarray:
    normalized = _row_normalize(np.asarray(features, dtype=np.float32))
    return np.clip(normalized @ normalized.T, -1.0, 1.0).astype(np.float32)


def _rbf_ssm(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    squared_norm = np.sum(features * features, axis=1, keepdims=True)
    distances = np.maximum(
        squared_norm + squared_norm.T - 2.0 * (features @ features.T),
        0.0,
    )
    positive = distances[distances > 1e-10]
    scale = float(np.median(positive)) if positive.size else 1.0
    return np.exp(-distances / max(2.0 * scale, 1e-8)).astype(np.float32)


def _checkerboard_kernel(half_width: int) -> np.ndarray:
    size = 2 * half_width
    coordinates = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(coordinates, coordinates)
    sigma = max(half_width / 2.0, 1.0)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel[:half_width, half_width:] *= -1.0
    kernel[half_width:, :half_width] *= -1.0
    kernel -= kernel.mean()
    kernel /= np.sum(np.abs(kernel)) + 1e-8
    return kernel.astype(np.float32)


def _foote_novelty(
    ssm: np.ndarray,
    half_width: int,
    context_gap: int,
) -> np.ndarray:
    length = ssm.shape[0]
    novelty = np.zeros(length, dtype=np.float32)
    margin = half_width + context_gap
    if length < 2 * margin + 1:
        return novelty

    kernel = _checkerboard_kernel(half_width)
    for index in range(margin, length - margin):
        indices = np.concatenate(
            [
                np.arange(index - context_gap - half_width, index - context_gap),
                np.arange(index + context_gap, index + context_gap + half_width),
            ]
        )
        novelty[index] = float(np.sum(ssm[np.ix_(indices, indices)] * kernel))
    return np.maximum(novelty, 0.0)


def _positive_normalize(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    normalized = np.maximum(
        (values - median) / max(1.4826 * mad, 1e-6),
        0.0,
    )
    upper = float(np.percentile(normalized, 95.0)) if normalized.size else 1.0
    return np.clip(normalized / max(upper, 1e-6), 0.0, 1.0)


def _multiscale_novelty(
    ssm: np.ndarray,
    scales: tuple[int, ...],
    context_gaps: tuple[int, ...],
) -> np.ndarray:
    curves = [
        _positive_normalize(_foote_novelty(ssm, scale, gap))
        for scale in scales
        for gap in context_gaps
        if ssm.shape[0] >= 2 * (scale + gap) + 1
    ]
    if not curves:
        return np.zeros(ssm.shape[0], dtype=np.float32)
    return np.mean(curves, axis=0, dtype=np.float64)


def robust_standardize_columns(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return np.clip((values - median) / scale, -8.0, 8.0)


def _median_small(values):
    """np.median of a short float64 array: its middle value, or for an even count the mean of the middle two
    ((a + b) / 2, as np.median takes it); NaN if any value is NaN."""
    n = values.size
    for v in values:
        if np.isnan(v):
            return np.nan
    ordered = np.sort(values)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _neighbor_median_loop(signal, radius):
    n = signal.size
    output = np.zeros_like(signal)
    buffer = np.empty(2 * radius, dtype=np.float64)
    for index in range(n):
        count = 0
        for other in range(max(0, index - radius), min(n, index + radius + 1)):
            if other != index:
                buffer[count] = signal[other]
                count += 1
        output[index] = _median_small(buffer[:count]) if count else signal[index]
    return output


def _onset_recurrence_loop(onset):
    n = onset.size
    output = np.zeros_like(onset)
    periodic = np.empty(4, dtype=np.float64)
    nearby = np.empty(6, dtype=np.float64)
    for index in range(n):
        p = 0
        for offset in (-8, -4, 4, 8):
            other = index + offset
            if 0 <= other < n:
                periodic[p] = onset[other]
                p += 1
        q = 0
        for offset in (-3, -2, -1, 1, 2, 3):
            other = index + offset
            if 0 <= other < n:
                nearby[q] = onset[other]
                q += 1
        if p and q:
            output[index] = (-abs(onset[index] - _median_small(periodic[:p]))
                             + abs(onset[index] - _median_small(nearby[:q])))
    return output


try:
    from numba import njit

    _median_small = njit(cache=True, inline="always")(_median_small)
    _neighbor_median_numba = njit(cache=True, nogil=True)(_neighbor_median_loop)
    _onset_recurrence_numba = njit(cache=True, nogil=True)(_onset_recurrence_loop)
except Exception:  # pragma: no cover - numba is optional
    _neighbor_median_numba = _onset_recurrence_numba = None


def _neighbor_median(values: np.ndarray, radius: int) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if _neighbor_median_numba is not None and signal.ndim == 1:
        # compiled: the same medians (np.median's own rule) without a call per beat
        return _neighbor_median_numba(np.ascontiguousarray(signal), int(radius))
    output = np.zeros_like(signal)
    for index in range(signal.size):
        lo = max(0, index - radius)
        hi = min(signal.size, index + radius + 1)
        neighbors = np.concatenate(
            [signal[lo:index], signal[index + 1 : hi]]
        )
        output[index] = (
            float(np.median(neighbors)) if neighbors.size else signal[index]
        )
    return output


def _ssm_downbeat_features(ssm: np.ndarray) -> list[np.ndarray]:
    n_beats = ssm.shape[0]
    previous_change = np.zeros(n_beats, dtype=np.float64)
    following_change = np.zeros(n_beats, dtype=np.float64)
    if n_beats > 1:
        diagonal = np.diag(ssm, k=1)
        previous_change[1:] = 1.0 - diagonal
        following_change[:-1] = 1.0 - diagonal
    symmetric_change = 0.5 * (previous_change + following_change)

    local_distinctiveness = np.zeros(n_beats, dtype=np.float64)
    recurrence_advantage = np.zeros(n_beats, dtype=np.float64)
    for index in range(n_beats):
        nearby = [
            float(ssm[index, other])
            for offset in (-3, -2, -1, 1, 2, 3)
            if 0 <= (other := index + offset) < n_beats
        ]
        periodic = [
            float(ssm[index, other])
            for offset in (-8, -4, 4, 8)
            if 0 <= (other := index + offset) < n_beats
        ]
        nearby_mean = float(np.mean(nearby)) if nearby else 1.0
        periodic_mean = float(np.mean(periodic)) if periodic else nearby_mean
        local_distinctiveness[index] = 1.0 - nearby_mean
        recurrence_advantage[index] = periodic_mean - nearby_mean

    return [
        previous_change,
        symmetric_change,
        _multiscale_novelty(ssm, (1,), (0,)),
        _multiscale_novelty(ssm, (2,), (0,)),
        _multiscale_novelty(ssm, (4,), (0,)),
        local_distinctiveness,
        recurrence_advantage,
    ]


def _onset_columns(
    curve: np.ndarray,
    frames: FrameFeatures,
    beats: np.ndarray,
    beat_ends: np.ndarray,
) -> list[np.ndarray]:
    """Per-beat onset energy and its accent / attack / fill / recurrence columns."""
    onset = _onset_per_beat(curve, frames.frame_times, beats, beat_ends, 0.5)
    onset_accent_4 = onset - _neighbor_median(onset, 2)
    median_8 = _neighbor_median(onset, 4)
    onset_accent_8 = onset - median_8
    onset_attack = np.diff(onset, prepend=onset[:1])
    pre_fill = np.zeros_like(onset)
    if onset.size > 1:
        pre_fill[1:] = onset[:-1] - median_8[:-1]
    if _onset_recurrence_numba is not None and onset.dtype == np.float64 and onset.ndim == 1:
        return [onset, onset_accent_4, onset_accent_8, onset_attack, pre_fill,
                _onset_recurrence_numba(np.ascontiguousarray(onset))]
    onset_recurrence = np.zeros_like(onset)
    for index in range(onset.size):
        periodic = [
            onset[other]
            for offset in (-8, -4, 4, 8)
            if 0 <= (other := index + offset) < onset.size
        ]
        nearby = [
            onset[other]
            for offset in (-3, -2, -1, 1, 2, 3)
            if 0 <= (other := index + offset) < onset.size
        ]
        if periodic and nearby:
            onset_recurrence[index] = (
                -abs(onset[index] - float(np.median(periodic)))
                + abs(onset[index] - float(np.median(nearby)))
            )
    return [onset, onset_accent_4, onset_accent_8, onset_attack, pre_fill, onset_recurrence]


def downbeat_feature_matrix(
    frames: FrameFeatures,
    beat_times_sec: np.ndarray,
    *,
    beat_phase_bins: int,
    harmonic_phase_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    beats, beat_ends, rows = _extract_beat_features(
        frames,
        beat_times_sec,
        beat_phase_bins,
        harmonic_phase_bins,
    )
    columns: list[np.ndarray] = []
    for representation_rows, (_name, metric) in zip(
        rows,
        _FEATURE_SPECS,
        strict=True,
    ):
        ssm = (
            _cosine_ssm(representation_rows)
            if metric == "cos"
            else _rbf_ssm(representation_rows)
        )
        columns.extend(_ssm_downbeat_features(ssm))

    rhythm_index = next(
        index
        for index, (name, _metric) in enumerate(_FEATURE_SPECS)
        if name == "rhythm"
    )
    rhythm_energy = np.linalg.norm(rows[rhythm_index], axis=1)
    rhythm_accent = rhythm_energy - _neighbor_median(rhythm_energy, 2)
    curves = {
        "onset": frames.librosa_onset,
        "harmonic_onset": frames.harmonic_onset,
        "kick_onset": frames.percussive_bands[0],
        "melody_onset": frames.melody_onset,
    }
    columns.extend(_onset_columns(curves["onset"], frames, beats, beat_ends))
    columns.extend([rhythm_energy, rhythm_accent])
    for source in _ONSET_SOURCES[1:]:
        columns.extend(_onset_columns(curves[source], frames, beats, beat_ends))

    coefficients = local_periodicity(
        frames.frame_times,
        beats.astype(np.float64),
        periodicity_curves(frames),
        _BAR_PERIODS_BEATS,
        _BAR_WINDOW_BEATS,
    )
    aligned = rotate(coefficients, np.arange(beats.size, dtype=np.float64), _BAR_PERIODS_BEATS)
    columns.extend(aligned.real.reshape(beats.size, -1).T)
    columns.extend(aligned.imag.reshape(beats.size, -1).T)

    features = np.stack(columns, axis=1)
    if features.shape[1] != len(_DOWNBEAT_FEATURE_NAMES):
        raise RuntimeError(
            f"Downbeat feature count mismatch: {features.shape[1]}"
        )
    return beats.astype(np.float64), features.astype(np.float32)


def _onset_per_beat(
    curve: np.ndarray,
    frame_times: np.ndarray,
    beats: np.ndarray,
    beat_ends: np.ndarray,
    onset_window_beats: float,
) -> np.ndarray:
    """Onset-strength energy assigned to each beat (no cross-beat averaging).

    The onset strength is integrated inside ``+-onset_window_beats`` of every
    beat. With 0.5 the per-beat windows tile the timeline (every frame counted
    once, centred on its nearest beat), so all time information is preserved --
    this is the beat-grid-style cut the windowed Folded-Sum throws away.
    """
    onset_env = np.asarray(curve, dtype=np.float64)
    if onset_env.size == 0 or beats.size == 0:
        return np.zeros(beats.size, dtype=np.float64)
    onset_times = np.asarray(frame_times, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(onset_env)])
    last = cumulative.size - 1

    beats = np.asarray(beats, dtype=np.float64)
    half_widths = float(onset_window_beats) * (np.asarray(beat_ends, dtype=np.float64) - beats)

    lo = np.clip(
        np.searchsorted(onset_times, beats - half_widths, side="left"), 0, last
    )
    hi = np.clip(
        np.searchsorted(onset_times, beats + half_widths, side="left"), 0, last
    )
    return cumulative[hi] - cumulative[lo]
