"""STFT primitives used by Mixlyzer's analysis paths.

The implementation uses SciPy's pocketfft and bounded frame blocks.  Its
centering, Hann window and overlap-add normalization match the librosa calls
the analyzer previously used, without importing librosa or Numba.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal import get_window


_MAX_FRAME_BLOCK_BYTES = 32 * 1024 * 1024


def _analysis_window(window: str | tuple | np.ndarray, win_length: int, n_fft: int, dtype) -> np.ndarray:
    if isinstance(window, np.ndarray):
        fft_window = np.asarray(window)
        if fft_window.ndim != 1 or fft_window.size != win_length:
            raise ValueError("window array must be one-dimensional with win_length elements")
    else:
        fft_window = get_window(window, win_length, fftbins=True)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        fft_window = np.pad(fft_window, (left, n_fft - win_length - left))
    return np.asarray(fft_window, dtype=dtype)


def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str | tuple | np.ndarray = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError("stft expects one-dimensional mono audio")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    n_fft = int(n_fft)
    win_length = n_fft if win_length is None else int(win_length)
    hop_length = win_length // 4 if hop_length is None else int(hop_length)
    if n_fft <= 0 or win_length <= 0 or win_length > n_fft or hop_length <= 0:
        raise ValueError("invalid FFT/window/hop length")

    if center:
        padding = n_fft // 2
        values = np.pad(values, (padding, padding), mode=pad_mode)
    if values.size < n_fft:
        raise ValueError("audio is shorter than n_fft with center=False")

    fft_window = _analysis_window(window, win_length, n_fft, values.dtype)
    frames = np.lib.stride_tricks.sliding_window_view(values, n_fft)[::hop_length]
    complex_dtype = np.complex64 if values.dtype.itemsize <= 4 else np.complex128
    result = np.empty((1 + n_fft // 2, frames.shape[0]), dtype=complex_dtype)
    block_frames = max(1, _MAX_FRAME_BLOCK_BYTES // max(1, n_fft * values.dtype.itemsize))
    for start in range(0, frames.shape[0], block_frames):
        stop = min(frames.shape[0], start + block_frames)
        # Frequency-first layout matches the analyzer's former librosa FFT
        # orientation and minimizes float32 rounding drift.
        windowed = np.asarray(fft_window[:, np.newaxis] * frames[start:stop].T, dtype=values.dtype)
        result[:, start:stop] = rfft(windowed, n=n_fft, axis=0)
    return result


def istft(
    spectrum: np.ndarray,
    *,
    hop_length: int | None = None,
    win_length: int | None = None,
    n_fft: int | None = None,
    window: str | tuple | np.ndarray = "hann",
    center: bool = True,
    dtype=None,
    length: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(spectrum)
    if matrix.ndim != 2:
        raise ValueError("istft expects a [frequency, frame] matrix")
    n_fft = 2 * (matrix.shape[0] - 1) if n_fft is None else int(n_fft)
    win_length = n_fft if win_length is None else int(win_length)
    hop_length = win_length // 4 if hop_length is None else int(hop_length)
    real_dtype = np.float32 if matrix.dtype.itemsize <= 8 else np.float64
    if dtype is not None:
        real_dtype = np.dtype(dtype)
    fft_window = _analysis_window(window, win_length, n_fft, real_dtype)

    n_frames = matrix.shape[1]
    natural_length = n_fft + hop_length * max(0, n_frames - 1)
    offset = n_fft // 2 if center else 0
    required = offset + int(length) if length is not None else natural_length
    output_length = max(natural_length, required)
    output = np.zeros(output_length, dtype=real_dtype)
    window_sum = np.zeros(output_length, dtype=real_dtype)
    window_square = np.square(fft_window, dtype=real_dtype)

    block_frames = max(1, _MAX_FRAME_BLOCK_BYTES // max(1, n_fft * np.dtype(real_dtype).itemsize))
    for block_start in range(0, n_frames, block_frames):
        block_stop = min(n_frames, block_start + block_frames)
        frames = irfft(matrix[:, block_start:block_stop].T, n=n_fft, axis=-1).astype(real_dtype, copy=False)
        frames *= fft_window
        for local_index, frame in enumerate(frames):
            frame_index = block_start + local_index
            start = frame_index * hop_length
            output[start : start + n_fft] += frame
            window_sum[start : start + n_fft] += window_square

    tiny = np.finfo(real_dtype).tiny
    valid = window_sum > tiny
    output[valid] /= window_sum[valid]
    if center:
        output = output[offset:]
    if length is not None:
        target = int(length)
        if output.size < target:
            output = np.pad(output, (0, target - output.size))
        else:
            output = output[:target]
    return np.asarray(output, dtype=real_dtype)


def softmask(
    values: np.ndarray,
    reference: np.ndarray,
    *,
    power: float = 1.0,
    split_zeros: bool = False,
) -> np.ndarray:
    values = np.asarray(values)
    reference = np.asarray(reference)
    if values.shape != reference.shape:
        raise ValueError("values and reference must have identical shapes")
    if np.any(values < 0) or np.any(reference < 0):
        raise ValueError("softmask inputs must be non-negative")
    if power <= 0:
        raise ValueError("power must be strictly positive")
    if np.isinf(power):
        return values > reference

    dtype = np.result_type(values, reference, np.float32)
    values = values.astype(dtype, copy=False)
    reference = reference.astype(dtype, copy=False)
    scale = np.maximum(values, reference)
    tiny = np.finfo(dtype).tiny
    zeros = scale < tiny
    safe_scale = np.where(zeros, 1.0, scale)
    value_power = np.power(values / safe_scale, power)
    reference_power = np.power(reference / safe_scale, power)
    denominator = value_power + reference_power
    mask = value_power / np.where(denominator > 0, denominator, 1.0)
    mask[zeros] = 0.5 if split_zeros else 0.0
    return mask
