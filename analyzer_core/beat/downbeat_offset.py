from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit

from core.resource_paths import resource_path


@dataclass(frozen=True)
class DownbeatOffsetSegment:
    """A time range with one stable downbeat offset."""

    start_sec: float
    end_sec: float
    start_beat_index: int
    end_beat_index: int
    downbeat_phase: int
    downbeat_offset_beats: int
    first_downbeat_beat_index: int | None
    first_downbeat_time_sec: float | None
    downbeat_offset_sec: float | None
    confidence: float

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "start_beat_index": self.start_beat_index,
            "end_beat_index": self.end_beat_index,
            "downbeat_phase": self.downbeat_phase,
            "downbeat_offset_beats": self.downbeat_offset_beats,
            "first_downbeat_beat_index": self.first_downbeat_beat_index,
            "first_downbeat_time_sec": self.first_downbeat_time_sec,
            "downbeat_offset_sec": self.downbeat_offset_sec,
            "confidence": self.confidence,
        }


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        mono = waveform
    elif waveform.ndim == 2:
        channel_axis = 0 if waveform.shape[0] <= 8 else 1
        mono = waveform.mean(axis=channel_axis, dtype=np.float32)
    else:
        raise ValueError("audio must be a mono or stereo NumPy array")
    # copy=True: the input may be read-only (e.g. memory-mapped decode output),
    # so never write into it in place.
    mono = np.nan_to_num(mono.reshape(-1), copy=True)
    if mono.size == 0:
        raise ValueError("audio is empty")
    return mono


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


def _resample_patch(
    features: np.ndarray,
    frame_times: np.ndarray,
    start_sec: float,
    end_sec: float,
    phase_bins: int,
) -> np.ndarray:
    targets = np.linspace(
        start_sec,
        end_sec,
        phase_bins,
        endpoint=False,
        dtype=np.float32,
    )
    patch = np.empty((features.shape[0], phase_bins), dtype=np.float32)
    for channel in range(features.shape[0]):
        patch[channel] = np.interp(
            targets,
            frame_times,
            features[channel],
            left=float(features[channel, 0]),
            right=float(features[channel, -1]),
        )
    return patch


# Per-beat features and their SSM metric, in a fixed order. "cos" -> cosine
# SSM (tonal/normalized features), "rbf" -> RBF SSM (magnitude features). The
# log-mel "timbre" feature was dropped: a library-wide ablation showed it hurt
# downbeat-phase detection (lower F1 when included).
_FEATURE_SPECS: tuple[tuple[str, str], ...] = (
    ("harmonic", "cos"),      # CQT chroma -- chord changes on the downbeat
    ("rhythm", "rbf"),        # onset bands + rms + band ratios
    ("bass_chroma", "cos"),   # low-register chroma -- root/bass changes
    ("contrast", "rbf"),      # spectral contrast
    ("mfcc", "rbf"),          # MFCC timbral texture
    ("tonnetz", "cos"),       # tonal centroid (complements chroma)
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
) + (
    "onset_strength",
    "onset_local_accent_4beat",
    "onset_local_accent_8beat",
    "onset_attack",
    "pre_downbeat_fill",
    "onset_four_beat_recurrence",
    "rhythm_patch_energy",
    "rhythm_patch_local_accent",
)
_DEFAULT_WEIGHT_PATH = resource_path(
    "assets/weights/downbeat_feature_weights.json"
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
    (RBF features)."""
    rows = np.empty((beat_starts.size, frames.shape[0] * phase_bins), dtype=np.float32)
    for i, (start_sec, end_sec) in enumerate(zip(beat_starts, beat_ends, strict=True)):
        patch = _resample_patch(frames, frame_times, float(start_sec), float(end_sec), phase_bins)
        if l2:
            patch = patch.T
            patch /= np.maximum(np.linalg.norm(patch, axis=1, keepdims=True), 1e-8)
            rows[i] = patch.reshape(-1)
        else:
            rows[i] = (patch - patch.mean()).reshape(-1)
    return rows


def _extract_beat_features(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    beat_phase_bins: int,
    harmonic_phase_bins: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return ``(beat_starts, rows)`` where ``rows`` are the per-beat feature
    matrices in :data:`_FEATURE_SPECS` order."""
    duration_sec = audio.size / float(sample_rate)
    beat_starts, beat_ends = _beat_ranges(beat_times_sec, duration_sec)

    stft = librosa.stft(
        y=audio, n_fft=n_fft, hop_length=hop_length,
        window="hann", center=True, pad_mode="reflect",
    )
    magnitude = np.abs(stft).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32)
    frame_times = librosa.frames_to_time(
        np.arange(magnitude.shape[1]), sr=sample_rate, hop_length=hop_length,
    ).astype(np.float32)

    log_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            S=power, sr=sample_rate, n_mels=n_mels,
            fmin=30.0, fmax=sample_rate / 2.0, norm="slaney", power=2.0,
        ),
        ref=np.max, top_db=100.0,
    ).astype(np.float32)

    # Sharp CQT chroma (not CENS, which over-smooths chord-change timing).
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sample_rate, hop_length=hop_length, n_chroma=12,
    ).astype(np.float32)
    bass_chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sample_rate, hop_length=hop_length, n_chroma=12,
        fmin=librosa.note_to_hz("C1"), n_octaves=3,
    ).astype(np.float32)
    contrast = librosa.feature.spectral_contrast(
        S=magnitude, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
    ).astype(np.float32)
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(
            librosa.feature.melspectrogram(S=power, sr=sample_rate, n_mels=n_mels),
            ref=np.max,
        ),
        n_mfcc=20,
    ).astype(np.float32)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sample_rate).astype(np.float32)

    aligned = min(
        frame_times.size, magnitude.shape[1], chroma.shape[1], bass_chroma.shape[1],
        contrast.shape[1], mfcc.shape[1], tonnetz.shape[1],
    )
    frame_times = frame_times[:aligned]
    magnitude = magnitude[:, :aligned]
    power = power[:, :aligned]
    log_mel = log_mel[:, :aligned]

    mel_difference = np.maximum(np.diff(log_mel, axis=1, prepend=log_mel[:, :1]), 0.0)
    band_edges = np.linspace(0, n_mels, 4, dtype=int)
    onset_bands = np.stack(
        [mel_difference[band_edges[i]:band_edges[i + 1]].mean(axis=0) for i in range(3)]
    )
    rms = librosa.feature.rms(S=magnitude, frame_length=n_fft, center=False)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max, top_db=100.0)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    total_power = np.maximum(power.sum(axis=0, keepdims=True), 1e-10)
    low_ratio = power[frequencies < 180.0].sum(axis=0, keepdims=True) / total_power
    mid_ratio = power[
        (frequencies >= 180.0) & (frequencies < 2500.0)
    ].sum(axis=0, keepdims=True) / total_power
    high_ratio = power[frequencies >= 2500.0].sum(axis=0, keepdims=True) / total_power
    rhythm_frames = _robust_standardize(
        np.concatenate([onset_bands, rms_db, low_ratio, mid_ratio, high_ratio], axis=0).astype(np.float32)
    )

    frame_sets = {
        "harmonic": (chroma[:, :aligned], harmonic_phase_bins),
        "rhythm": (rhythm_frames, beat_phase_bins),
        "bass_chroma": (bass_chroma[:, :aligned], harmonic_phase_bins),
        "contrast": (_robust_standardize(contrast[:, :aligned]), beat_phase_bins),
        "mfcc": (_robust_standardize(mfcc[:, :aligned]), beat_phase_bins),
        "tonnetz": (tonnetz[:, :aligned], harmonic_phase_bins),
    }
    rows = [
        _beat_patch_rows(frame_sets[name][0], frame_times, beat_starts, beat_ends,
                         frame_sets[name][1], l2=(metric == "cos"))
        for name, metric in _FEATURE_SPECS
    ]
    return beat_starts, rows


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


def _robust_standardize_columns(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return np.clip((values - median) / scale, -8.0, 8.0)


def _neighbor_median(values: np.ndarray, radius: int) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
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


def _downbeat_feature_matrix(
    waveform: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    *,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    beat_phase_bins: int,
    harmonic_phase_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    beats, rows = _extract_beat_features(
        waveform,
        sample_rate,
        beat_times_sec,
        hop_length,
        n_fft,
        n_mels,
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

    onset = _onset_per_beat(waveform, sample_rate, beats, hop_length, 0.5)
    onset_accent_4 = onset - _neighbor_median(onset, 2)
    onset_accent_8 = onset - _neighbor_median(onset, 4)
    onset_attack = np.diff(onset, prepend=onset[:1])
    pre_fill = np.zeros_like(onset)
    if onset.size > 1:
        pre_fill[1:] = onset[:-1] - _neighbor_median(onset, 4)[:-1]
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

    rhythm_index = next(
        index
        for index, (name, _metric) in enumerate(_FEATURE_SPECS)
        if name == "rhythm"
    )
    rhythm_energy = np.linalg.norm(rows[rhythm_index], axis=1)
    rhythm_accent = rhythm_energy - _neighbor_median(rhythm_energy, 2)
    columns.extend(
        [
            onset,
            onset_accent_4,
            onset_accent_8,
            onset_attack,
            pre_fill,
            onset_recurrence,
            rhythm_energy,
            rhythm_accent,
        ]
    )

    features = np.stack(columns, axis=1)
    if features.shape[1] != len(_DOWNBEAT_FEATURE_NAMES):
        raise RuntimeError(
            f"Downbeat feature count mismatch: {features.shape[1]}"
        )
    return beats.astype(np.float64), features.astype(np.float32)


@lru_cache(maxsize=4)
def _load_downbeat_weights(weight_path: str) -> np.ndarray:
    path = Path(weight_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    model = report.get("model", report)
    feature_names = tuple(str(name) for name in model["feature_names"])
    if feature_names != _DOWNBEAT_FEATURE_NAMES:
        raise ValueError(
            f"Downbeat weight feature schema mismatch: {path}"
        )
    weights = np.asarray(model["weights"], dtype=np.float64)
    if weights.shape != (len(_DOWNBEAT_FEATURE_NAMES),):
        raise ValueError(f"Invalid downbeat weight shape: {weights.shape}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Downbeat weights contain non-finite values")
    return weights


def _downbeat_probability(
    features: np.ndarray,
    weight_path: str | Path,
) -> np.ndarray:
    weights = _load_downbeat_weights(str(Path(weight_path).resolve()))
    standardized = _robust_standardize_columns(features)
    return expit(standardized @ weights)


def _folded_sum(values: np.ndarray, period: int) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(values, dtype=np.float64).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    phase = np.arange(signal.size, dtype=np.int32) % period
    return np.bincount(phase, weights=signal, minlength=period).astype(np.float64)


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    return values / total if total > 1e-12 else np.zeros_like(values)


def _phase_confidence(scores: np.ndarray) -> float:
    """Relative margin between the best and the second-best phase score."""
    phase = int(np.argmax(scores))
    alternatives = np.delete(scores, phase)
    alternative_score = float(np.max(alternatives)) if alternatives.size else 0.0
    selected_score = float(scores[phase])
    return (selected_score - alternative_score) / max(abs(selected_score), 1e-12)


def _window_starts(length: int, window_beats: int, hop_beats: int) -> np.ndarray:
    if length <= window_beats:
        return np.asarray([0], dtype=np.int32)
    starts = list(range(0, length - window_beats + 1, hop_beats))
    final_start = length - window_beats
    if starts[-1] != final_start:
        starts.append(final_start)
    return np.asarray(starts, dtype=np.int32)


def _phase_distance(left: int, right: int, meter: int) -> int:
    distance = abs(left - right)
    return min(distance, meter - distance)


def _transition_cost_matrix(
    meter: int,
    adjacent_phase_penalty: float,
    distant_phase_penalty: float,
) -> np.ndarray:
    """Per-(old, new) phase transition penalty for the Viterbi step.

    The downbeat phase wraps around, so in 4/4 phases 0 and 3 are adjacent
    (``_phase_distance`` returns 1). Moving to an adjacent phase (distance 1)
    costs ``adjacent_phase_penalty`` while jumping two or more beats away (e.g.
    a half-bar shift, the more common genuine correction) costs
    ``distant_phase_penalty``. Staying put costs nothing.
    """
    costs = np.zeros((meter, meter), dtype=np.float64)
    for old_phase in range(meter):
        for phase in range(meter):
            distance = _phase_distance(old_phase, phase, meter)
            if distance == 0:
                continue
            costs[old_phase, phase] = (
                adjacent_phase_penalty
                if distance == 1
                else distant_phase_penalty
            )
    return costs


def _track_phases(
    scores: np.ndarray,
    meter: int,
    adjacent_phase_penalty: float = 2.3,
    distant_phase_penalty: float = 1.15,
) -> np.ndarray:
    costs = _transition_cost_matrix(
        meter,
        adjacent_phase_penalty,
        distant_phase_penalty,
    )
    dynamic = np.full_like(scores, -np.inf, dtype=np.float64)
    previous = np.zeros_like(scores, dtype=np.int32)
    dynamic[0] = scores[0]

    for window_index in range(1, scores.shape[0]):
        for phase in range(meter):
            # costs[:, phase] is the penalty of arriving at `phase` from each
            # possible previous phase.
            candidates = dynamic[window_index - 1] - costs[:, phase]
            old_phase = int(np.argmax(candidates))
            previous[window_index, phase] = old_phase
            dynamic[window_index, phase] = (
                candidates[old_phase] + scores[window_index, phase]
            )

    tracked = np.empty(scores.shape[0], dtype=np.int32)
    tracked[-1] = int(np.argmax(dynamic[-1]))
    for window_index in range(scores.shape[0] - 1, 0, -1):
        tracked[window_index - 1] = previous[window_index, tracked[window_index]]
    return tracked


def _make_segment(
    beat_times: np.ndarray,
    duration_sec: float,
    start_beat: int,
    end_beat: int,
    phase: int,
    confidence: float,
    meter: int,
) -> DownbeatOffsetSegment:
    """Assemble one :class:`DownbeatOffsetSegment` from a beat range and phase."""
    start_beat = int(start_beat)
    end_beat = int(end_beat)
    phase = int(phase)
    offset_beats = int((phase - start_beat) % meter)
    first_downbeat = start_beat + offset_beats
    if first_downbeat < end_beat and first_downbeat < beat_times.size:
        first_index: int | None = first_downbeat
        first_time: float | None = float(beat_times[first_downbeat])
    else:
        first_index = None
        first_time = None

    start_sec = 0.0 if start_beat == 0 else float(beat_times[start_beat])
    end_sec = (
        duration_sec if end_beat >= beat_times.size else float(beat_times[end_beat])
    )
    return DownbeatOffsetSegment(
        start_sec=start_sec,
        end_sec=end_sec,
        start_beat_index=start_beat,
        end_beat_index=end_beat,
        downbeat_phase=phase,
        downbeat_offset_beats=offset_beats,
        first_downbeat_beat_index=first_index,
        first_downbeat_time_sec=first_time,
        downbeat_offset_sec=(None if first_time is None else first_time - start_sec),
        confidence=float(confidence),
    )


def _phase_credit_cumsum(
    combined_novelty: np.ndarray, phase_of_beat: np.ndarray, meter: int
) -> np.ndarray:
    """Prefix sums of novelty grouped by beat phase, shape ``(meter, n+1)``."""
    cum = np.zeros((meter, combined_novelty.size + 1), dtype=np.float64)
    for residue in range(meter):
        cum[residue, 1:] = np.cumsum(
            np.where(phase_of_beat == residue, combined_novelty, 0.0)
        )
    return cum


def _changepoint_beat(
    cum: np.ndarray,
    phase_left: int,
    phase_right: int,
    lo: int,
    hi: int,
    center: int,
    radius: int,
) -> int:
    """Beat in ``(lo, hi)`` where switching ``phase_left -> phase_right`` best
    partitions the novelty credit (a beat-resolution changepoint near
    ``center``). This replaces the fixed lookback shift with a data-driven,
    beat-accurate boundary."""
    b_lo = max(lo + 1, center - radius)
    b_hi = min(hi - 1, center + radius)
    if b_hi < b_lo:
        return int(min(max(center, lo + 1), hi - 1))
    candidates = np.arange(b_lo, b_hi + 1)
    left = cum[phase_left, candidates] - cum[phase_left, lo]
    right = cum[phase_right, hi] - cum[phase_right, candidates]
    return int(candidates[int(np.argmax(left + right))])


def _merge_runs(runs: list[list[int]], min_segment_beats: int) -> list[list[int]]:
    """Absorb runs shorter than ``min_segment_beats`` into their longer
    neighbor, then coalesce adjacent equal-phase runs. Removes over-detection
    (spurious one-off phase flips)."""
    runs = [list(run) for run in runs]
    while len(runs) > 1:
        lengths = [end - start for start, end, _ in runs]
        index = int(np.argmin(lengths))
        if lengths[index] >= min_segment_beats:
            break
        if index == 0:
            runs[1][0] = runs[0][0]
            runs.pop(0)
        elif index == len(runs) - 1:
            runs[-2][1] = runs[-1][1]
            runs.pop()
        elif lengths[index - 1] >= lengths[index + 1]:
            runs[index - 1][1] = runs[index][1]
            runs.pop(index)
        else:
            runs[index + 1][0] = runs[index][0]
            runs.pop(index)

    merged = [runs[0]]
    for run in runs[1:]:
        if run[2] == merged[-1][2]:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return merged


def _gate_runs(
    runs: list[list[int]], cum: np.ndarray, min_transition_gain: float
) -> list[list[int]]:
    """Drop boundaries whose two-phase split barely beats a single-phase fit.

    For each interior boundary the gain is how much better explaining the two
    sides with their own phases is than the best single phase over the merged
    span (as a fraction of the span's novelty). Iteratively remove the weakest
    sub-threshold boundary, re-labeling the merged run by its dominant phase.
    This rejects spurious flips (no real transition) by evidence strength rather
    than by raising the penalty, so genuine transitions survive."""
    runs = [list(run) for run in runs]
    while len(runs) > 1:
        worst_index, worst_gain = -1, np.inf
        for i in range(1, len(runs)):
            lo, boundary, hi = runs[i - 1][0], runs[i][0], runs[i][1]
            span = cum[:, hi] - cum[:, lo]
            total = float(span.sum()) + 1e-12
            two_phase = (
                (cum[runs[i - 1][2], boundary] - cum[runs[i - 1][2], lo])
                + (cum[runs[i][2], hi] - cum[runs[i][2], boundary])
            )
            gain = (two_phase - float(span.max())) / total
            if gain < worst_gain:
                worst_gain, worst_index = gain, i
        if worst_gain >= min_transition_gain:
            break
        lo, hi = runs[worst_index - 1][0], runs[worst_index][1]
        phase = int(np.argmax(cum[:, hi] - cum[:, lo]))
        runs[worst_index - 1] = [lo, hi, phase]
        runs.pop(worst_index)

    coalesced = [runs[0]]
    for run in runs[1:]:
        if run[2] == coalesced[-1][2]:
            coalesced[-1][1] = run[1]
        else:
            coalesced.append(run)
    return coalesced


def _build_segments_refined(
    beat_times: np.ndarray,
    duration_sec: float,
    window_starts: np.ndarray,
    run_window_starts: np.ndarray,
    run_phases: list[int],
    combined_novelty: np.ndarray,
    meter: int,
    lookback: int,
    refine_radius: int,
    min_segment_beats: int,
    min_transition_gain: float,
) -> list[DownbeatOffsetSegment]:
    """Refine windowed phase-change boundaries to beat resolution, drop
    over-detected short runs and weak (low-evidence) transitions, and assemble
    segments."""
    n_beats = int(beat_times.size)
    phase_of_beat = np.arange(n_beats, dtype=np.int64) % meter
    cum = _phase_credit_cumsum(combined_novelty, phase_of_beat, meter)

    # Coarse boundary beat for each phase change (window start, minus the
    # lookback prior), then refine each to the exact changepoint beat.
    coarse = [int(window_starts[w]) - lookback for w in run_window_starts[1:]]
    bounds = [0]
    for index, center in enumerate(coarse):
        lo = bounds[-1]
        hi = coarse[index + 1] if index + 1 < len(coarse) else n_beats
        beat = _changepoint_beat(
            cum, run_phases[index], run_phases[index + 1], lo, hi, center, refine_radius
        )
        bounds.append(max(beat, lo + 1))
    bounds.append(n_beats)

    runs = [[bounds[i], bounds[i + 1], int(run_phases[i])] for i in range(len(run_phases))]
    runs = _merge_runs(runs, min_segment_beats)
    if min_transition_gain > 0.0:
        runs = _gate_runs(runs, cum, min_transition_gain)

    segments: list[DownbeatOffsetSegment] = []
    for start_beat, end_beat, phase in runs:
        run_scores = np.zeros(meter, dtype=np.float64)
        np.add.at(run_scores, phase_of_beat[start_beat:end_beat], combined_novelty[start_beat:end_beat])
        total = float(run_scores.sum())
        if total > 1e-12:
            run_scores = run_scores / total
        segments.append(
            _make_segment(
                beat_times,
                duration_sec,
                int(start_beat),
                int(end_beat),
                int(phase),
                _phase_confidence(run_scores),
                meter,
            )
        )
    return segments


def _wrap_downbeat(downbeat_time: float, seg_start: float, bpm: float, ts_num: float) -> float:
    """Bring a downbeat onto the segment, keeping its phase (within one bar of start)."""
    if bpm <= 0:
        return float(seg_start)
    period = 60.0 / bpm
    beats_per_bar = int(round(ts_num)) if ts_num >= 1 else 4
    bar = beats_per_bar * period
    if bar <= 0:
        return float(seg_start)
    rel = (float(downbeat_time) - float(seg_start)) % bar
    return float(seg_start + rel)


def _first_downbeat_on_beats(
    beats: np.ndarray,
    downbeat_time: float,
    seg_start: float,
    seg_end: float,
    ts_num: float,
) -> float:
    """First actual-beat downbeat at/after ``seg_start``.

    A downbeat is every ``meter``-th beat from the reference downbeat beat, so
    the returned time is always one of the real beats (no drift across tempo
    changes). Falls back to the nearest beat to ``seg_start`` when the segment
    is shorter than a bar.
    """
    eps = 1e-4
    meter = int(round(ts_num)) if ts_num >= 1 else 4
    meter = max(1, meter)
    # Reference downbeat beat index (nearest real beat to the detected downbeat).
    ref_idx = int(np.argmin(np.abs(beats - float(downbeat_time))))
    # First beat index inside the segment.
    lo = int(np.searchsorted(beats, seg_start - eps, side="left"))
    lo = min(max(lo, 0), beats.size - 1)
    # Step forward to the first index congruent to ref_idx (mod meter).
    offset = (ref_idx - lo) % meter
    j = lo + offset
    hi = int(np.searchsorted(beats, seg_end + eps, side="right"))
    if j >= beats.size:
        j = beats.size - 1
    elif j >= hi:
        # No downbeat lands inside this (sub-bar) segment; keep the phase anyway.
        j = min(j, beats.size - 1)
    return float(beats[j])


def apply_downbeat_offset_segments(
    tempo_segments: np.ndarray,
    downbeat_segments: list[DownbeatOffsetSegment],
    beat_times_sec: np.ndarray | None = None,
) -> np.ndarray:
    """Reassign per-segment downbeats from detected downbeat-offset changes.

    Beat timings are NOT changed. A downbeat is the bar-start beat; where the
    detected downbeat phase changes at time T, the tempo segment is cut half a
    beat before T (so the beat at T belongs to the new segment) and the new
    segment's ``inizio`` (bar-start reference) is set to the detected downbeat.

    Args:
        tempo_segments: (N, 5) array of [start, end, bpm, inizio, ts_num].
        downbeat_segments: output of :func:`detect_downbeat_offset_segments`.

    Returns:
        A possibly longer (M, 5) tempo-segment array with realigned downbeats.
    """
    seg = np.asarray(tempo_segments, dtype=float)
    if seg.ndim != 2 or seg.shape[1] < 5 or seg.shape[0] == 0:
        return np.asarray(tempo_segments, dtype=np.float32)
    rows: list[list[float]] = [list(map(float, r[:5])) for r in seg]

    # Downbeat reference timeline: (region_start_time, downbeat_time) at the first
    # detected segment and at every phase change. Phase-change starts are also the
    # points where a tempo segment must be split.
    references: list[tuple[float, float]] = []
    split_points: list[float] = []
    previous_phase: int | None = None
    for ds in downbeat_segments:
        downbeat_time = ds.first_downbeat_time_sec
        if downbeat_time is None:
            previous_phase = ds.downbeat_phase
            continue
        if previous_phase is None:
            references.append((float(ds.start_sec), float(downbeat_time)))
        elif int(ds.downbeat_phase) != int(previous_phase):
            references.append((float(ds.start_sec), float(downbeat_time)))
            split_points.append(float(ds.start_sec))
        previous_phase = ds.downbeat_phase

    if not references:
        return np.asarray(rows, dtype=np.float32)

    eps = 1e-4
    # 1) Split tempo segments half a beat before each phase-change point, so the
    #    beat at the change belongs to the following segment.
    for region_start in split_points:
        starts = [r[0] for r in rows]
        idx = int(np.searchsorted(starts, region_start + eps, side="right")) - 1
        idx = max(0, min(idx, len(rows) - 1))
        start, end, bpm, inizio, ts_num = rows[idx]
        beat_period = 60.0 / bpm if bpm > 0 else 0.0
        cut_t = region_start - 0.5 * beat_period
        if cut_t <= start + eps or cut_t >= end - eps:
            continue
        rows[idx][1] = cut_t
        rows.insert(idx + 1, [cut_t, end, bpm, inizio, ts_num])

    # 2) Propagate the downbeat phase to EVERY segment (not just the first / the
    #    change points). A downbeat is every ``meter``-th BEAT counted from the
    #    reference downbeat beat, so it stays on the real beat grid even when the
    #    tempo (and thus the bar length in seconds) differs between segments.
    #    The lookup point is half a beat into the segment so the right half of a
    #    half-beat-earlier split maps to the new phase region.
    beats = None
    if beat_times_sec is not None:
        beats = np.unique(np.asarray(beat_times_sec, dtype=float).reshape(-1))
        beats = beats[np.isfinite(beats)]
        if beats.size == 0:
            beats = None

    ref_starts = [r[0] for r in references]
    for r in rows:
        seg_start, seg_end, bpm, _inizio, ts_num = r
        half_beat = 0.5 * (60.0 / bpm) if bpm > 0 else 0.0
        k = int(np.searchsorted(ref_starts, seg_start + half_beat + eps, side="right")) - 1
        k = max(0, k)
        downbeat_time = references[k][1]
        if beats is None:
            r[3] = _wrap_downbeat(downbeat_time, seg_start, bpm, ts_num)
        else:
            r[3] = _first_downbeat_on_beats(
                beats, downbeat_time, seg_start, seg_end, ts_num
            )

    return np.asarray(rows, dtype=np.float32)


def detect_downbeat_offset_segments(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    *,
    method: Literal["dynamic", "global"] = "dynamic",
    meter: int = 4,
    window_beats: int = 12,
    window_hop_beats: int = 1,
    transition_lookback_beats: int = 0,
    boundary_refine_radius_beats: int = 12,
    min_segment_beats: int = 4,
    min_transition_gain: float = 0.0,
    adjacent_phase_penalty: float = 5.0,
    distant_phase_penalty: float = 2.5,
    probability_power: float = 1.0,
    smooth_sigma_beats: float = 0.5,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 64,
    beat_phase_bins: int = 8,
    harmonic_phase_bins: int = 8,
    weight_path: str | Path = _DEFAULT_WEIGHT_PATH,
) -> list[DownbeatOffsetSegment]:
    """Detect stable downbeat-phase segments using the learned beat model.

    The model produces one learned downbeat probability per beat from 50
    harmonic, rhythmic, bass, timbral, tonal, onset, and recurrence features.
    ``dynamic`` folds those probabilities in sliding windows and tracks the
    phase with Viterbi. ``global`` folds the same probabilities over the entire
    input and returns one phase.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if method not in {"dynamic", "global"}:
        raise ValueError("method must be 'dynamic' or 'global'")
    if meter != 4:
        raise ValueError("the learned downbeat model currently supports 4/4 only")
    if method == "dynamic" and window_beats < meter:
        raise ValueError("window_beats must be at least meter")
    if method == "dynamic" and window_hop_beats < 1:
        raise ValueError("window_hop_beats must be positive")
    if method == "dynamic" and transition_lookback_beats < 0:
        raise ValueError("transition_lookback_beats must be non-negative")
    if method == "dynamic" and boundary_refine_radius_beats < 0:
        raise ValueError("boundary_refine_radius_beats must be non-negative")
    if method == "dynamic" and min_segment_beats < 1:
        raise ValueError("min_segment_beats must be at least 1")
    if method == "dynamic" and min_transition_gain < 0.0:
        raise ValueError("min_transition_gain must be non-negative")
    if method == "dynamic" and adjacent_phase_penalty < 0.0:
        raise ValueError("adjacent_phase_penalty must be non-negative")
    if method == "dynamic" and distant_phase_penalty < 0.0:
        raise ValueError("distant_phase_penalty must be non-negative")
    if probability_power <= 0.0:
        raise ValueError("probability_power must be positive")
    if smooth_sigma_beats < 0.0:
        raise ValueError("smooth_sigma_beats must be non-negative")

    waveform = _mono_audio(audio)
    duration_sec = waveform.size / float(sample_rate)
    beat_times, feature_matrix = _downbeat_feature_matrix(
        waveform,
        sample_rate,
        beat_times_sec,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
        beat_phase_bins=beat_phase_bins,
        harmonic_phase_bins=harmonic_phase_bins,
    )
    probability = _downbeat_probability(feature_matrix, weight_path)
    if smooth_sigma_beats > 0.0:
        probability = gaussian_filter1d(
            probability,
            sigma=smooth_sigma_beats,
            mode="nearest",
        )
    evidence = np.power(
        np.clip(probability, 0.0, 1.0),
        probability_power,
    )

    if method == "global":
        scores = _normalize_curve(_folded_sum(evidence, meter))
        phase = int(np.argmax(scores))
        return [
            _make_segment(
                beat_times,
                duration_sec,
                0,
                beat_times.size,
                phase,
                _phase_confidence(scores),
                meter,
            )
        ]

    starts = _window_starts(beat_times.size, window_beats, window_hop_beats)
    ends = np.minimum(starts + window_beats, beat_times.size).astype(np.int32)
    scores = np.zeros((starts.size, meter), dtype=np.float64)
    for window_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        phase_profile = _folded_sum(evidence[start:end], meter)
        scores[window_index] = np.roll(
            phase_profile,
            int(start % meter),
        )

    score_totals = scores.sum(axis=1, keepdims=True)
    scores = np.divide(
        scores,
        score_totals,
        out=np.full_like(scores, 1.0 / meter),
        where=score_totals > 1e-12,
    )
    tracked_phases = _track_phases(
        scores,
        meter,
        adjacent_phase_penalty,
        distant_phase_penalty,
    )
    # Viterbi runs (maximal constant-phase spans of windows) give the robust
    # phase sequence; their window-start beats are only coarse boundaries.
    run_starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(tracked_phases[1:] != tracked_phases[:-1]) + 1,
        ]
    )
    run_phases = [int(tracked_phases[w]) for w in run_starts]
    return _build_segments_refined(
        beat_times,
        duration_sec,
        starts,
        run_starts,
        run_phases,
        evidence,
        meter,
        int(transition_lookback_beats),
        int(boundary_refine_radius_beats),
        int(min_segment_beats),
        float(min_transition_gain),
    )


def _onset_per_beat(
    waveform: np.ndarray,
    sample_rate: int,
    beats: np.ndarray,
    hop_length: int,
    onset_window_beats: float,
) -> np.ndarray:
    """Onset-strength energy assigned to each beat (no cross-beat averaging).

    The ODF is integrated inside ``+-onset_window_beats`` of every beat. With
    0.5 the per-beat windows tile the timeline (every ODF frame counted once,
    centred on its nearest beat), so all time information is preserved -- this
    is the beat-grid-style cut the windowed Folded-Sum throws away.
    """
    onset_env = librosa.onset.onset_strength(
        y=waveform, sr=sample_rate, hop_length=hop_length
    )
    if onset_env.size == 0 or beats.size == 0:
        return np.zeros(beats.size, dtype=np.float64)
    onset_times = librosa.times_like(
        onset_env, sr=sample_rate, hop_length=hop_length
    )
    cumulative = np.concatenate([[0.0], np.cumsum(onset_env.astype(np.float64))])
    last = cumulative.size - 1

    periods = np.diff(beats)
    beat_periods = np.empty(beats.size, dtype=np.float64)
    if periods.size:
        beat_periods[:-1] = periods
        beat_periods[-1] = periods[-1]
    else:
        beat_periods[:] = 0.0
    half_widths = float(onset_window_beats) * beat_periods

    lo = np.clip(
        np.searchsorted(onset_times, beats - half_widths, side="left"), 0, last
    )
    hi = np.clip(
        np.searchsorted(onset_times, beats + half_widths, side="left"), 0, last
    )
    return cumulative[hi] - cumulative[lo]
