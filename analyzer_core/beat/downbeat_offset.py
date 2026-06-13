from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def _extract_beat_features(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    hop_length: int,
    n_fft: int,
    n_mels: int,
    beat_phase_bins: int,
    harmonic_phase_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duration_sec = audio.size / float(sample_rate)
    beat_starts, beat_ends = _beat_ranges(beat_times_sec, duration_sec)

    stft = librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    magnitude = np.abs(stft).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32)
    frame_times = librosa.frames_to_time(
        np.arange(magnitude.shape[1]),
        sr=sample_rate,
        hop_length=hop_length,
    ).astype(np.float32)

    mel_power = librosa.feature.melspectrogram(
        S=power,
        sr=sample_rate,
        n_mels=n_mels,
        fmin=30.0,
        fmax=sample_rate / 2.0,
        norm="slaney",
        power=2.0,
    )
    log_mel = librosa.power_to_db(
        mel_power,
        ref=np.max,
        top_db=100.0,
    ).astype(np.float32)
    timbre_frames = _robust_standardize(log_mel)

    chroma = librosa.feature.chroma_stft(
        S=power,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_chroma=12,
        norm=2,
    ).astype(np.float32)
    chroma = librosa.feature.chroma_cens(
        C=chroma,
        sr=sample_rate,
        hop_length=hop_length,
        n_chroma=12,
    ).astype(np.float32)

    aligned_frames = min(frame_times.size, chroma.shape[1])
    frame_times = frame_times[:aligned_frames]
    magnitude = magnitude[:, :aligned_frames]
    power = power[:, :aligned_frames]
    log_mel = log_mel[:, :aligned_frames]
    timbre_frames = timbre_frames[:, :aligned_frames]
    chroma = chroma[:, :aligned_frames]

    mel_difference = np.maximum(
        np.diff(log_mel, axis=1, prepend=log_mel[:, :1]),
        0.0,
    )
    band_edges = np.linspace(0, n_mels, 4, dtype=int)
    onset_bands = np.stack(
        [
            mel_difference[band_edges[index] : band_edges[index + 1]].mean(axis=0)
            for index in range(3)
        ]
    )
    rms = librosa.feature.rms(
        S=magnitude,
        frame_length=n_fft,
        center=False,
    )
    rms_db = librosa.amplitude_to_db(rms, ref=np.max, top_db=100.0)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    total_power = np.maximum(power.sum(axis=0, keepdims=True), 1e-10)
    low_ratio = power[frequencies < 180.0].sum(axis=0, keepdims=True) / total_power
    mid_ratio = power[
        (frequencies >= 180.0) & (frequencies < 2500.0)
    ].sum(axis=0, keepdims=True) / total_power
    high_ratio = (
        power[frequencies >= 2500.0].sum(axis=0, keepdims=True) / total_power
    )
    rhythm_frames = _robust_standardize(
        np.concatenate(
            [onset_bands, rms_db, low_ratio, mid_ratio, high_ratio],
            axis=0,
        ).astype(np.float32)
    )

    timbre_rows: list[np.ndarray] = []
    harmonic_rows: list[np.ndarray] = []
    rhythm_rows: list[np.ndarray] = []
    for start_sec, end_sec in zip(beat_starts, beat_ends, strict=True):
        timbre = _resample_patch(
            timbre_frames,
            frame_times,
            float(start_sec),
            float(end_sec),
            beat_phase_bins,
        )
        timbre -= timbre.mean()
        timbre_rows.append(timbre.reshape(-1))

        harmonic = _resample_patch(
            chroma,
            frame_times,
            float(start_sec),
            float(end_sec),
            harmonic_phase_bins,
        ).T
        harmonic /= np.maximum(
            np.linalg.norm(harmonic, axis=1, keepdims=True),
            1e-8,
        )
        harmonic_rows.append(harmonic.reshape(-1))

        rhythm = _resample_patch(
            rhythm_frames,
            frame_times,
            float(start_sec),
            float(end_sec),
            beat_phase_bins,
        )
        rhythm_rows.append(rhythm.reshape(-1))

    return (
        beat_starts,
        np.asarray(timbre_rows, dtype=np.float32),
        np.asarray(harmonic_rows, dtype=np.float32),
        np.asarray(rhythm_rows, dtype=np.float32),
    )


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


def _global_phase_scores(
    novelty_curves: tuple[np.ndarray, np.ndarray, np.ndarray],
    novelty_weights: tuple[float, float, float],
    meter: int,
) -> np.ndarray:
    """Fold all novelty curves over the entire track without window tracking."""
    scores = np.zeros(meter, dtype=np.float64)
    for weight, curve in zip(novelty_weights, novelty_curves, strict=True):
        scores += float(weight) * _folded_sum(_normalize_curve(curve), meter)
    total = float(scores.sum())
    return scores / total if total > 1e-12 else scores


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


def _track_phases(
    scores: np.ndarray,
    meter: int,
    transition_penalty: float,
) -> np.ndarray:
    dynamic = np.full_like(scores, -np.inf, dtype=np.float64)
    previous = np.zeros_like(scores, dtype=np.int32)
    dynamic[0] = scores[0]

    for window_index in range(1, scores.shape[0]):
        for phase in range(meter):
            candidates = np.asarray(
                [
                    dynamic[window_index - 1, old_phase]
                    - transition_penalty
                    * _phase_distance(old_phase, phase, meter)
                    for old_phase in range(meter)
                ]
            )
            old_phase = int(np.argmax(candidates))
            previous[window_index, phase] = old_phase
            dynamic[window_index, phase] = candidates[old_phase] + scores[
                window_index,
                phase,
            ]

    tracked = np.empty(scores.shape[0], dtype=np.int32)
    tracked[-1] = int(np.argmax(dynamic[-1]))
    for window_index in range(scores.shape[0] - 1, 0, -1):
        tracked[window_index - 1] = previous[window_index, tracked[window_index]]
    return tracked


def _build_segments(
    beat_times: np.ndarray,
    duration_sec: float,
    window_centers: np.ndarray,
    tracked_phases: np.ndarray,
    confidence: np.ndarray,
    meter: int,
) -> list[DownbeatOffsetSegment]:
    run_starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int32),
            np.flatnonzero(tracked_phases[1:] != tracked_phases[:-1]) + 1,
        ]
    )
    run_ends = np.concatenate(
        [run_starts[1:], np.asarray([tracked_phases.size], dtype=np.int32)]
    )

    transition_beats = [
        int(round((window_centers[index - 1] + window_centers[index]) / 2.0))
        for index in run_starts[1:]
    ]
    beat_boundaries = [0, *transition_beats, beat_times.size]
    segments: list[DownbeatOffsetSegment] = []

    for run_index, (window_start, window_end) in enumerate(
        zip(run_starts, run_ends, strict=True)
    ):
        start_beat = int(beat_boundaries[run_index])
        end_beat = int(beat_boundaries[run_index + 1])
        phase = int(tracked_phases[window_start])
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
            duration_sec
            if end_beat >= beat_times.size
            else float(beat_times[end_beat])
        )
        segments.append(
            DownbeatOffsetSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                start_beat_index=start_beat,
                end_beat_index=end_beat,
                downbeat_phase=phase,
                downbeat_offset_beats=offset_beats,
                first_downbeat_beat_index=first_index,
                first_downbeat_time_sec=first_time,
                downbeat_offset_sec=(
                    None if first_time is None else first_time - start_sec
                ),
                confidence=float(np.median(confidence[window_start:window_end])),
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
    window_beats: int = 32,
    window_hop_beats: int = 1,
    transition_penalty: float = 2.3,
    novelty_scales: tuple[int, ...] = (8, 16, 32, 64),
    novelty_context_gaps: tuple[int, ...] = (0,),
    novelty_weights: tuple[float, float, float] = (0.45, 0.25, 0.30),
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 64,
    beat_phase_bins: int = 8,
    harmonic_phase_bins: int = 8,
) -> list[DownbeatOffsetSegment]:
    """Detect time segments with a stable downbeat offset.

    Args:
        audio: Mono or stereo floating-point PCM samples.
        sample_rate: Audio sample rate in Hz.
        beat_times_sec: Beat start times in seconds.
        method: ``"dynamic"`` uses sliding windows and dynamic programming.
            ``"global"`` uses one Folded Sum over the entire track.

    Returns:
        Contiguous downbeat-offset segments. ``downbeat_phase`` is the
        zero-based global beat phase. ``downbeat_offset_beats`` is the number
        of beats from the segment start beat to its first downbeat.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if method not in {"dynamic", "global"}:
        raise ValueError("method must be 'dynamic' or 'global'")
    if meter < 2:
        raise ValueError("meter must be at least 2")
    if method == "dynamic" and window_beats < meter:
        raise ValueError("window_beats must be at least meter")
    if method == "dynamic" and window_hop_beats < 1:
        raise ValueError("window_hop_beats must be positive")
    if method == "dynamic" and transition_penalty < 0.0:
        raise ValueError("transition_penalty must be non-negative")
    if any(scale < 1 for scale in novelty_scales):
        raise ValueError("novelty_scales must contain positive integers")
    if any(gap < 0 for gap in novelty_context_gaps):
        raise ValueError("novelty_context_gaps must be non-negative")
    if len(novelty_weights) != 3:
        raise ValueError("novelty_weights must contain three values")

    waveform = _mono_audio(audio)
    duration_sec = waveform.size / float(sample_rate)
    beat_times, timbre, harmonic, rhythm = _extract_beat_features(
        waveform,
        sample_rate,
        beat_times_sec,
        hop_length,
        n_fft,
        n_mels,
        beat_phase_bins,
        harmonic_phase_bins,
    )

    novelty_curves = (
        _multiscale_novelty(
            _rbf_ssm(timbre),
            novelty_scales,
            novelty_context_gaps,
        ),
        _multiscale_novelty(
            _cosine_ssm(harmonic),
            novelty_scales,
            novelty_context_gaps,
        ),
        _multiscale_novelty(
            _rbf_ssm(rhythm),
            novelty_scales,
            novelty_context_gaps,
        ),
    )

    if method == "global":
        scores = _global_phase_scores(
            novelty_curves,
            novelty_weights,
            meter,
        )
        phase = int(np.argmax(scores))
        alternatives = np.delete(scores, phase)
        alternative_score = float(np.max(alternatives))
        selected_score = float(scores[phase])
        confidence = (
            selected_score - alternative_score
        ) / max(abs(selected_score), 1e-12)
        return _build_segments(
            beat_times,
            duration_sec,
            np.asarray([beat_times.size // 2], dtype=np.int32),
            np.asarray([phase], dtype=np.int32),
            np.asarray([confidence], dtype=np.float64),
            meter,
        )

    starts = _window_starts(beat_times.size, window_beats, window_hop_beats)
    ends = np.minimum(starts + window_beats, beat_times.size).astype(np.int32)
    centers = ((starts + ends - 1) // 2).astype(np.int32)
    scores = np.zeros((starts.size, meter), dtype=np.float64)
    for window_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for weight, curve in zip(novelty_weights, novelty_curves, strict=True):
            local_profile = _folded_sum(
                _normalize_curve(curve[start:end]),
                meter,
            )
            scores[window_index] += float(weight) * np.roll(
                local_profile,
                int(start % meter),
            )

    score_totals = scores.sum(axis=1, keepdims=True)
    scores = np.divide(
        scores,
        score_totals,
        out=np.zeros_like(scores),
        where=score_totals > 1e-12,
    )
    tracked_phases = _track_phases(scores, meter, transition_penalty)
    sorted_scores = np.sort(scores, axis=1)
    confidence = np.divide(
        sorted_scores[:, -1] - sorted_scores[:, -2],
        np.maximum(np.abs(sorted_scores[:, -1]), 1e-12),
    )
    return _build_segments(
        beat_times,
        duration_sec,
        centers,
        tracked_phases,
        confidence,
        meter,
    )
