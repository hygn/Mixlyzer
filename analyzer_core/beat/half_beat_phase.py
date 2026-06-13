from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HalfBeatPhaseResult:
    """Result of comparing energy in the two halves of each beat interval."""

    decision: int
    first_half_energy_sum: float
    second_half_energy_sum: float
    confidence: float
    correction_offset_beats: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "decision": self.decision,
            "first_half_energy_sum": self.first_half_energy_sum,
            "second_half_energy_sum": self.second_half_energy_sum,
            "confidence": self.confidence,
            "correction_offset_beats": self.correction_offset_beats,
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
    mono = np.nan_to_num(mono.reshape(-1), copy=True)
    if mono.size == 0:
        raise ValueError("audio is empty")
    return mono


def analyze_half_beat_phase(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
) -> HalfBeatPhaseResult:
    """Return detailed evidence for normal versus half-beat-shifted timing.

    Decision 1 means energy is larger in the first half of the supplied beat
    intervals. Decision 2 means it is larger in the second half, indicating a
    likely half-beat offset. Exact ties are classified as decision 1.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    waveform = _mono_audio(audio)
    duration_sec = waveform.size / float(sample_rate)

    beats = np.asarray(beat_times_sec, dtype=np.float64).reshape(-1)
    beats = np.unique(beats[np.isfinite(beats)])
    beats = beats[(beats >= 0.0) & (beats <= duration_sec)]
    if beats.size < 2:
        raise ValueError("beat_times_sec must contain at least two valid beats")

    first_sum = 0.0
    second_sum = 0.0
    valid_intervals = 0
    for start_sec, end_sec in zip(beats[:-1], beats[1:], strict=True):
        midpoint_sec = 0.5 * (start_sec + end_sec)
        sample_edges = np.clip(
            np.rint(
                np.asarray([start_sec, midpoint_sec, end_sec]) * sample_rate
            ).astype(np.int64),
            0,
            waveform.size,
        )
        start_sample, midpoint_sample, end_sample = sample_edges
        if midpoint_sample <= start_sample or end_sample <= midpoint_sample:
            continue

        first = waveform[start_sample:midpoint_sample].astype(np.float64)
        second = waveform[midpoint_sample:end_sample].astype(np.float64)
        first_sum += float(np.mean(first * first))
        second_sum += float(np.mean(second * second))
        valid_intervals += 1

    if valid_intervals == 0:
        raise ValueError("beat intervals are too short to contain audio samples")

    total = first_sum + second_sum
    confidence = abs(first_sum - second_sum) / max(total, 1e-12)
    decision = 1 if first_sum >= second_sum else 2
    return HalfBeatPhaseResult(
        decision=decision,
        first_half_energy_sum=first_sum,
        second_half_energy_sum=second_sum,
        confidence=confidence,
        correction_offset_beats=0.0 if decision == 1 else 0.5,
    )


def detect_half_beat_phase(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
) -> int:
    """Return 1 for normal beat phase or 2 for a likely half-beat offset."""
    return analyze_half_beat_phase(
        audio=audio,
        sample_rate=sample_rate,
        beat_times_sec=beat_times_sec,
    ).decision
