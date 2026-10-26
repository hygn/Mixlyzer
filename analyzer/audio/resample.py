"""Audio resampling through python-soxr without librosa/resampy dispatch."""

from __future__ import annotations

import numpy as np
import soxr


def resample_audio(
    audio: np.ndarray,
    orig_sr: float,
    target_sr: float,
    *,
    axis: int = -1,
    quality: str = "HQ",
) -> np.ndarray:
    """Resample ``audio`` along ``axis`` while preserving all other axes."""
    values = np.asarray(audio)
    if values.ndim == 0:
        raise ValueError("audio must have at least one dimension")
    if float(orig_sr) <= 0 or float(target_sr) <= 0:
        raise ValueError("sample rates must be positive")
    if float(orig_sr) == float(target_sr):
        return np.ascontiguousarray(values)

    axis = int(axis)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ValueError(f"axis {axis} is out of bounds for {values.ndim}-D audio")
    quality = str(quality)
    if quality.lower().startswith("soxr_"):
        quality = quality[5:]
    quality = quality.upper()
    time_first = np.moveaxis(values, axis, 0)
    input_shape = time_first.shape
    channels = int(np.prod(input_shape[1:], dtype=np.int64))
    matrix = np.ascontiguousarray(time_first.reshape(input_shape[0], channels))
    if channels == 1:
        output = soxr.resample(matrix[:, 0], float(orig_sr), float(target_sr), quality=quality)
        output = np.asarray(output).reshape(-1, *input_shape[1:])
    else:
        output = soxr.resample(matrix, float(orig_sr), float(target_sr), quality=quality)
        output = np.asarray(output).reshape(-1, *input_shape[1:])
    return np.ascontiguousarray(np.moveaxis(output, 0, axis))
