from __future__ import annotations

import numpy as np

from analyzer_core.beat.frame_features import FrameFeatures


# Beat-grid local periodicity, shared by the beat-phase and downbeat models.
#
# For every beat, a Hann window of +-window beats gives each onset curve's local
# Fourier coefficient at a few periods measured in beats (a local pulse per
# source, as in predominant-local-pulse / metrical-profile / drum-pattern
# methods). Rotating the coefficient to a candidate position tells how well that
# periodic pulse lines up with it: at 1/2 and 1 beat it separates the beat from
# the off-beat, at 2 and 4 beats it follows backbeat and bar patterns.
PERIODICITY_CURVE_NAMES: tuple[str, ...] = (
    "kick_flux",
    "snare_flux",
    "hat_flux",
    "mel_flux_low",
    "mel_flux_mid",
    "mel_flux_high",
    "harmonic_onset",
    "percussive_onset",
    "melody_onset",
    "chroma_flux",
    "bass_chroma_flux",
    "melody_chroma_flux",
)


def _robust_standardize_columns(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return np.clip((values - median) / scale, -8.0, 8.0)


def _positive_flux_sum(frames: np.ndarray) -> np.ndarray:
    return np.maximum(np.diff(frames, axis=1, prepend=frames[:, :1]), 0.0).sum(axis=0)


def periodicity_curves(frames: FrameFeatures) -> np.ndarray:
    """Onset curves [T, C] in PERIODICITY_CURVE_NAMES order, standardized per track."""
    return _robust_standardize_columns(raw_onset_curves(frames))


def raw_onset_curves(frames: FrameFeatures) -> np.ndarray:
    """Onset curves [T, C] in PERIODICITY_CURVE_NAMES order, in their own units."""
    curves = np.column_stack(
        [
            frames.percussive_bands.T,
            frames.onset_bands.T,
            frames.harmonic_onset,
            frames.librosa_onset,
            frames.melody_onset,
            _positive_flux_sum(frames.chroma),
            _positive_flux_sum(frames.bass_chroma),
            _positive_flux_sum(frames.melody_chroma),
        ]
    ).astype(np.float64)
    if curves.shape[1] != len(PERIODICITY_CURVE_NAMES):
        raise RuntimeError(f"Periodicity curve count mismatch: {curves.shape[1]}")
    return np.nan_to_num(curves)


def local_periodicity(
    frame_times: np.ndarray,
    beat_times: np.ndarray,
    curves: np.ndarray,
    periods_beats: tuple[float, ...],
    window_beats: float,
) -> np.ndarray:
    """Local Fourier coefficients [n_beats, C, P] around every beat.

    Phases are measured in beat units from beat 0, so multiplying by
    ``exp(2j * pi * position / period)`` rotates a coefficient to ``position``
    (in beats): its real part is then the pulse strength at that position.
    """
    beat_times = np.asarray(beat_times, dtype=np.float64)
    n = beat_times.size
    position = np.interp(frame_times, beat_times, np.arange(n, dtype=np.float64))
    periods = np.asarray(periods_beats, dtype=np.float64)
    phase = np.exp(-2j * np.pi * position[:, np.newaxis] / periods[np.newaxis, :])  # [T, P]
    lo = np.searchsorted(position, np.arange(n) - window_beats)
    hi = np.searchsorted(position, np.arange(n) + window_beats)
    out = np.zeros((n, curves.shape[1], periods.size), dtype=np.complex128)
    for i in range(n):
        s, e = int(lo[i]), int(hi[i])
        if e - s < 4:
            continue
        weight = 0.5 * (1.0 + np.cos(np.pi * (position[s:e] - i) / window_beats))
        out[i] = np.einsum("t,tc,tp->cp", weight, curves[s:e], phase[s:e]) / weight.sum()
    return out


def rotate(coefficients: np.ndarray, positions: np.ndarray, periods_beats: tuple[float, ...]) -> np.ndarray:
    """Rotate [n, C, P] coefficients to per-row positions (beats) -> [n, C, P]."""
    periods = np.asarray(periods_beats, dtype=np.float64)
    turn = np.exp(2j * np.pi * np.asarray(positions, dtype=np.float64)[:, np.newaxis] / periods[np.newaxis, :])
    return coefficients * turn[:, np.newaxis, :]


def sixteenth_peaks(frame_times: np.ndarray, beat_times: np.ndarray, curves: np.ndarray) -> np.ndarray:
    """Peak of each curve in every 16th cell centred on the grid, scaled to [0, 1] per curve.

    Returns [4 * n_beats + 4, C]; cell ``4 * beat + k`` is ``k`` sixteenths after
    ``beat`` (four trailing cells past the last beat). Peaks keep the raw accent
    (loud vs soft hits), which the clipped standardized curves lose on dense drums.
    """
    beat_times = np.asarray(beat_times, dtype=np.float64)
    n = beat_times.size
    position = np.interp(frame_times, beat_times, np.arange(n, dtype=np.float64))
    cell = np.rint(position * 4).astype(np.int64)
    inside = (frame_times > beat_times[0]) & (frame_times < beat_times[-1])
    peaks = np.zeros((4 * n + 4, curves.shape[1]))
    for c in range(curves.shape[1]):
        np.maximum.at(peaks[:, c], np.clip(cell[inside], 0, 4 * n + 3), curves[inside, c])
    body = peaks[4 : 4 * n]
    lo = np.percentile(body, 10, axis=0)
    hi = np.percentile(body, 95, axis=0)
    return np.clip((peaks - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0)
