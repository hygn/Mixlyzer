"""The tempo-change onset: a gradient-boosted beat activation on the frame grid, from per-frame onset
sources with their context. The beat tracker moves each tempo change to it (``grid_refine.refine_boundaries``)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from analyzer.beat.frame_features import FrameFeatures


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
# Frame offsets stacked around each frame so the model can sharpen and
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


# Gradient-boosted onset over those inputs. It also fires on triplet subdivisions, so the
# tracker reads tempo from its TCN; this one only places the tempo changes.
REFINE_ONSET_FORMAT = "mixlyzer_refine_onset_hgb_v1"


@lru_cache(maxsize=2)
def load_refine_onset_model(path: str):
    from analyzer.phrase.phrase_analyzer import _load_numpy_hgb

    with np.load(Path(path), allow_pickle=False) as archive:
        if str(np.asarray(archive["format"]).item()) != REFINE_ONSET_FORMAT:
            raise ValueError(f"Refinement onset model format mismatch: {path}")
        if tuple(np.asarray(archive["feature_names"]).astype(str)) != ONSET_FEATURE_NAMES:
            raise ValueError(f"Refinement onset model feature schema mismatch: {path}")
        model = _load_numpy_hgb(archive, "onset")
        hop = int(np.asarray(archive["frame_hop_length"]).item())
    return model, hop


def refine_onset_activation(frames: FrameFeatures, path: str | Path) -> np.ndarray:
    """Gradient-boosted beat activation on the frame grid [T] (the model in the file at ``path``)."""
    model, hop = load_refine_onset_model(str(path))
    if hop != int(frames.hop_length):
        raise ValueError(f"Refinement onset model expects hop {hop}, frames use {frames.hop_length}")
    inputs = stack_context(onset_channels(frames)).astype(np.float64)
    return model.predict_proba(inputs)[:, 1].astype(np.float32)


def export_refine_onset_model(classifier, output_path: str | Path, frame_hop_length: int, training: dict | None = None) -> None:
    from analyzer.phrase.phrase_analyzer import _hgb_to_npz_payload
    from analyzer.files import atomic_output_path

    payload: dict[str, np.ndarray] = {
        "format": np.asarray(REFINE_ONSET_FORMAT),
        "feature_names": np.asarray(ONSET_FEATURE_NAMES),
        "frame_hop_length": np.int64(frame_hop_length),
        "training_json": np.asarray(json.dumps(training or {}, ensure_ascii=False, sort_keys=True)),
    }
    _hgb_to_npz_payload("onset", classifier, payload)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with atomic_output_path(out) as temporary:
            np.savez_compressed(temporary, **payload)
            load_refine_onset_model.cache_clear()
            load_refine_onset_model(str(temporary))
    finally:
        load_refine_onset_model.cache_clear()
