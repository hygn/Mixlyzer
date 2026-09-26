from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import median_filter

# Same defaults as librosa.effects.hpss / librosa.decompose.hpss.
HPSS_KERNEL_SIZE = 31
HPSS_N_FFT = 2048
HPSS_HOP_LENGTH = HPSS_N_FFT // 4
# Threads for the median filters (half on each filter).
HPSS_MAX_WORKERS = 8


@dataclass(frozen=True)
class HpssSpectra:
    """Magnitude STFT of the analysed signal and its harmonic / percussive parts.

    ``librosa.stft(y, n_fft, hop_length, center=True, pad_mode="constant")``;
    the parts equal ``librosa.decompose.hpss(magnitude)``.
    """

    sample_rate: int
    n_fft: int
    hop_length: int
    magnitude: np.ndarray   # [F, T]
    harmonic: np.ndarray    # [F, T]
    percussive: np.ndarray  # [F, T]


def _median_filter_chunked(
    pool: ThreadPoolExecutor, magnitude: np.ndarray, size: tuple[int, int], axis: int, chunks: int
) -> list:
    """Futures of ``median_filter(magnitude, size, mode="reflect")`` on slices
    along ``axis``. The kernel spans one element along ``axis``, so every slice
    filters independently and the joined result equals the whole-array filter."""
    bounds = np.linspace(0, magnitude.shape[axis], chunks + 1, dtype=int)
    index = [slice(None), slice(None)]
    futures = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop > start:
            index[axis] = slice(start, stop)
            futures.append(pool.submit(median_filter, magnitude[tuple(index)], size=size, mode="reflect"))
    return futures


def hpss_masks(magnitude: np.ndarray, kernel_size: int = HPSS_KERNEL_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic / percussive soft masks of a magnitude spectrogram ``[F, T]``.

    Equal to ``librosa.decompose.hpss(magnitude, mask=True)``. scipy's median
    filter and numpy's ufuncs release the GIL: the time filter runs on
    frequency bands and the frequency filter on time spans, in parallel, and
    the masks on frequency bands.
    """
    workers = max(2, min(HPSS_MAX_WORKERS, os.cpu_count() or 2))
    chunks = max(1, workers // 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        harmonic_parts = _median_filter_chunked(pool, magnitude, (1, kernel_size), axis=0, chunks=chunks)
        percussive_parts = _median_filter_chunked(pool, magnitude, (kernel_size, 1), axis=1, chunks=chunks)
        harmonic = np.concatenate([part.result() for part in harmonic_parts], axis=0)
        percussive = np.concatenate([part.result() for part in percussive_parts], axis=1)
        # softmask is elementwise: compute it on frequency bands.
        bounds = np.linspace(0, magnitude.shape[0], chunks + 1, dtype=int)
        bands = [slice(start, stop) for start, stop in zip(bounds[:-1], bounds[1:]) if stop > start]
        mask_parts = [
            (
                pool.submit(librosa.util.softmask, harmonic[band], percussive[band], power=2.0, split_zeros=True),
                pool.submit(librosa.util.softmask, percussive[band], harmonic[band], power=2.0, split_zeros=True),
            )
            for band in bands
        ]
        mask_harmonic = np.concatenate([h.result() for h, _ in mask_parts], axis=0)
        mask_percussive = np.concatenate([p.result() for _, p in mask_parts], axis=0)
        return mask_harmonic, mask_percussive


def hpss_audio_with_spectra(
    y: np.ndarray, sample_rate: int, n_fft: int = HPSS_N_FFT
) -> tuple[np.ndarray, np.ndarray, HpssSpectra]:
    """Harmonic and percussive signals of mono ``y`` (equal to ``librosa.effects.hpss(y)``)
    and the spectra they are made from."""
    hop_length = n_fft // 4
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True, pad_mode="constant")
    magnitude, phase = librosa.magphase(stft)
    mask_harmonic, mask_percussive = hpss_masks(magnitude)
    harmonic = magnitude * mask_harmonic
    percussive = magnitude * mask_percussive

    def inverse(part: np.ndarray) -> np.ndarray:
        return librosa.istft(part * phase, dtype=y.dtype, n_fft=n_fft, hop_length=hop_length, length=y.shape[-1])

    with ThreadPoolExecutor(max_workers=2) as pool:
        y_harmonic = pool.submit(inverse, harmonic)
        y_percussive = pool.submit(inverse, percussive)
        spectra = HpssSpectra(int(sample_rate), int(n_fft), int(hop_length), magnitude, harmonic, percussive)
        return y_harmonic.result(), y_percussive.result(), spectra


def hpss_audio(y: np.ndarray, n_fft: int = HPSS_N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic and percussive signals of mono ``y``; equal to ``librosa.effects.hpss(y)``."""
    y_harmonic, y_percussive, _ = hpss_audio_with_spectra(y, 0, n_fft)
    return y_harmonic, y_percussive
