from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit

from analyzer_core.beat.frame_features import FrameFeatures, model_frame_hop_length
from core.resource_paths import resource_path


# Per-frame onset sources, in model order. librosa's onset strength is one
# source; the rest come from the frame features the downbeat model also uses.
ONSET_CHANNEL_NAMES: tuple[str, ...] = (
    "librosa_onset",
    "mel_flux_low",
    "mel_flux_mid",
    "mel_flux_high",
    "rms_db",
    "rms_db_rise",
    "low_ratio",
    "mid_ratio",
    "high_ratio",
    "bass_chroma_change",
    "tonnetz_change",
    "mfcc_change",
    "percussive_flux_low",
    "percussive_flux_mid",
    "percussive_flux_high",
    "harmonic_onset",
    "percussive_share",
    "chroma_flux",
    "bass_chroma_flux",
    "flatness",
    "flatness_rise",
    "melody_onset",
    "melody_chroma_change",
    "melody_chroma_flux",
)
# Frame offsets stacked around each frame so the linear model can sharpen and
# time-align the sources (an FIR filter per channel). Offsets and dilations are
# in frames, so a model only fits the frame hop it was trained on.
ONSET_CONTEXT_OFFSETS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
# Longer, dilated context: channel means over dyadic bands of frames on each
# side, (d/2, d] frames before and after, for dilations d (256-hop frames: the
# bands tile ~0.05 s to ~3 s). Each band costs one prefix-sum lookup, so the
# span doubles per band at a constant cost. The model can weigh a frame against
# its surroundings, e.g. an on-beat kick against the off-beat hats around it.
ONSET_CONTEXT_DILATIONS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)

ONSET_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{name}@{offset:+d}" for offset in ONSET_CONTEXT_OFFSETS for name in ONSET_CHANNEL_NAMES
) + tuple(
    f"{name}@{side}{dilation}"
    for dilation in ONSET_CONTEXT_DILATIONS
    for side in ("pre", "post")
    for name in ONSET_CHANNEL_NAMES
)
ONSET_MODEL_FORMAT = "mixlyzer_learned_onset_v5"

_DEFAULT_WEIGHT_PATH = resource_path("assets/weights/onset_feature_weights.json")


def _rise(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.diff(values, axis=-1, prepend=values[..., :1]), 0.0)


def _cosine_change(frames: np.ndarray) -> np.ndarray:
    norm = frames / np.maximum(np.linalg.norm(frames, axis=0, keepdims=True), 1e-8)
    similarity = np.sum(norm[:, 1:] * norm[:, :-1], axis=0)
    return np.concatenate([[0.0], 1.0 - similarity])


def _euclidean_change(frames: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.linalg.norm(np.diff(frames, axis=1), axis=0)])


def _robust_standardize_columns(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return np.clip((values - median) / scale, -8.0, 8.0)


def onset_channels(frames: FrameFeatures) -> np.ndarray:
    """Per-frame onset sources [T, C], standardized per track (median / MAD)."""
    channels = np.column_stack(
        [
            frames.librosa_onset,
            frames.onset_bands.T,
            frames.rms_db,
            _rise(frames.rms_db),
            frames.band_ratios.T,
            _cosine_change(frames.bass_chroma),
            _euclidean_change(frames.tonnetz),
            _euclidean_change(frames.mfcc),
            frames.percussive_bands.T,
            frames.harmonic_onset,
            frames.percussive_share,
            _rise(frames.chroma).sum(axis=0),
            _rise(frames.bass_chroma).sum(axis=0),
            frames.flatness,
            _rise(frames.flatness),
            frames.melody_onset,
            _cosine_change(frames.melody_chroma),
            _rise(frames.melody_chroma).sum(axis=0),
        ]
    ).astype(np.float64)
    if channels.shape[1] != len(ONSET_CHANNEL_NAMES):
        raise RuntimeError(f"Onset channel count mismatch: {channels.shape[1]}")
    channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
    return _robust_standardize_columns(channels).astype(np.float32)


def stack_context(channels: np.ndarray, frame_index: np.ndarray | None = None) -> np.ndarray:
    """Model inputs [N, C * len(offsets)] for the given frames (all frames if None)."""
    n_frames = channels.shape[0]
    if frame_index is None:
        frame_index = np.arange(n_frames)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    blocks = [
        channels[np.clip(frame_index + offset, 0, n_frames - 1)]
        for offset in ONSET_CONTEXT_OFFSETS
    ]
    # Window means from prefix sums; windows are clipped at the track edges.
    cumulative = np.zeros((n_frames + 1, channels.shape[1]), dtype=np.float64)
    np.cumsum(channels, axis=0, out=cumulative[1:])

    def window_mean(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        lo = np.clip(lo, 0, n_frames)
        hi = np.clip(hi, 0, n_frames)
        count = np.maximum(hi - lo, 1)[:, np.newaxis]
        return ((cumulative[hi] - cumulative[lo]) / count).astype(channels.dtype)

    for dilation in ONSET_CONTEXT_DILATIONS:
        half = dilation // 2
        blocks.append(window_mean(frame_index - dilation, frame_index - half))
        blocks.append(window_mean(frame_index + half + 1, frame_index + dilation + 1))
    return np.concatenate(blocks, axis=1)


def clear_onset_weight_cache() -> None:
    """Make a newly written weight artifact visible in this process."""
    _load_onset_model.cache_clear()


@lru_cache(maxsize=4)
def _load_onset_model(weight_path: str) -> tuple[np.ndarray, float]:
    path = Path(weight_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    model = report.get("model", report)
    if str(model.get("format", "")) != ONSET_MODEL_FORMAT:
        raise ValueError(f"Onset weight format mismatch: {path}")
    if tuple(str(name) for name in model["feature_names"]) != ONSET_FEATURE_NAMES:
        raise ValueError(f"Onset weight feature schema mismatch: {path}")
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])
    if weights.shape != (len(ONSET_FEATURE_NAMES),) or not np.all(np.isfinite(weights)):
        raise ValueError(f"Invalid onset weights: {path}")
    return weights, bias


def onset_model_frame_hop_length(weight_path: str | Path) -> int:
    """Frame hop the onset model at ``weight_path`` was optimized at."""
    report = json.loads(Path(weight_path).read_text(encoding="utf-8"))
    return model_frame_hop_length(report.get("model", report))


def onset_activation(
    frames: FrameFeatures,
    weight_path: str | Path = _DEFAULT_WEIGHT_PATH,
) -> np.ndarray:
    """Learned beat-onset probability on the frame grid [T]."""
    weights, bias = _load_onset_model(str(Path(weight_path).resolve()))
    inputs = stack_context(onset_channels(frames))
    return expit(inputs.astype(np.float64) @ weights + bias).astype(np.float32)


def resample_to_odf_grid(
    values: np.ndarray,
    frames: FrameFeatures,
    hop_length: int,
    n_samples: int,
) -> tuple[np.ndarray, float]:
    """Resample a frame-grid curve onto the beat tracker's ODF grid.

    The tracker reads ODF frame ``k`` as a beat at ``(k + 0.5) * hop`` seconds,
    so the curve is sampled there.
    """
    hop_t = int(hop_length) / float(frames.sample_rate)
    n_odf = max(1, 1 + (int(n_samples) - 1) // int(hop_length))
    times = (np.arange(n_odf, dtype=np.float64) + 0.5) * hop_t
    odf = np.interp(times, frames.frame_times, np.asarray(values, dtype=np.float64))
    return odf.astype(np.float32), hop_t


def compute_learned_odf(
    frames: FrameFeatures,
    hop_length: int,
    n_samples: int,
    weight_path: str | Path = _DEFAULT_WEIGHT_PATH,
) -> tuple[np.ndarray, float]:
    """Beat tracker ODF from the learned onset model."""
    return resample_to_odf_grid(onset_activation(frames, weight_path), frames, hop_length, n_samples)


def compute_beat_odf(
    onset_source: str,
    frames: FrameFeatures | None,
    onset_audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    weight_path: str | Path = _DEFAULT_WEIGHT_PATH,
) -> tuple[np.ndarray, float]:
    """ODF for the beat tracker: librosa onset strength, or the learned onset.

    ``"optimized"`` needs ``frames``; if its weights cannot be used the librosa
    ODF is returned instead (and the reason printed) so analysis still finishes.
    """
    from analyzer_core.beat.beat import _compute_odf

    if onset_source == "optimized" and frames is not None:
        try:
            odf, hop_t = compute_learned_odf(frames, hop_length, len(onset_audio), weight_path)
            print(f"[Onset] learned onset ({weight_path})")
            return odf, hop_t
        except Exception as exc:
            print(f"[Onset] learned onset unavailable, using librosa: {exc}")
    return _compute_odf(onset_audio, sample_rate, hop_length)
