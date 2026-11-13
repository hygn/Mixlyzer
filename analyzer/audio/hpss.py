from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from analyzer.audio.running_median import median_filter_1d
from analyzer.audio.spectral import istft, softmask, stft

# Same defaults as librosa.effects.hpss / librosa.decompose.hpss.
HPSS_KERNEL_SIZE = 31
HPSS_N_FFT = 2048
HPSS_HOP_LENGTH = HPSS_N_FFT // 4
# Threads for the median filters (half on each filter).
HPSS_MAX_WORKERS = 8


@dataclass(frozen=True)
class HpssSpectra:
    """Magnitude STFT of the analysed signal and its harmonic / percussive parts.

    Mixlyzer STFT with ``center=True, pad_mode="constant"``; the parts use the
    same median filters and soft masks as the former librosa HPSS path.
    """

    sample_rate: int
    n_fft: int
    hop_length: int
    magnitude: np.ndarray   # [F, T]
    harmonic: np.ndarray    # [F, T]
    percussive: np.ndarray  # [F, T]
    # the harmonic part's signal with a wider margin (librosa.effects.harmonic's ``margin``), when asked for
    # (hpss_audio_with_spectra's ``vocal_margin``: the phrase model's song cues); None otherwise
    vocal: np.ndarray | None = None


def _median_filter_chunked(
    pool: ThreadPoolExecutor, magnitude: np.ndarray, size: tuple[int, int], axis: int, chunks: int
) -> list:
    """Futures of ``median_filter(magnitude, size, mode="reflect")`` on slices
    along ``axis``. The kernel spans one element along ``axis``, so every slice
    filters independently and the joined result equals the whole-array filter
    (a sliding median, dsp.running_median: scipy's result, several times faster)."""
    filter_axis = 1 - axis
    length = size[filter_axis]
    bounds = np.linspace(0, magnitude.shape[axis], chunks + 1, dtype=int)
    index = [slice(None), slice(None)]
    futures = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop > start:
            index[axis] = slice(start, stop)
            futures.append(pool.submit(median_filter_1d, magnitude[tuple(index)], length, filter_axis))
    return futures


def _medians(pool: ThreadPoolExecutor, magnitude: np.ndarray, kernel_size: int, chunks: int):
    harmonic_parts = _median_filter_chunked(pool, magnitude, (1, kernel_size), axis=0, chunks=chunks)
    percussive_parts = _median_filter_chunked(pool, magnitude, (kernel_size, 1), axis=1, chunks=chunks)
    harmonic = np.concatenate([part.result() for part in harmonic_parts], axis=0)
    percussive = np.concatenate([part.result() for part in percussive_parts], axis=1)
    return harmonic, percussive


def hpss_masks(magnitude: np.ndarray, kernel_size: int = HPSS_KERNEL_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic / percussive soft masks of a magnitude spectrogram ``[F, T]``.

    Equivalent to the former librosa HPSS masks. The median kernel and
    numpy's ufuncs release the GIL: the time filter runs on
    frequency bands and the frequency filter on time spans, in parallel, and
    the masks on frequency bands.
    """
    mask_harmonic, mask_percussive, _ = _masks(magnitude, kernel_size, None)
    return mask_harmonic, mask_percussive


def _masks(magnitude: np.ndarray, kernel_size: int, vocal_margin: float | None):
    """hpss_masks' two masks, and with ``vocal_margin`` the harmonic mask of that margin from the same medians
    (librosa.effects.harmonic's: the percussive median times the margin, zeros not split), else None."""
    workers = max(2, min(HPSS_MAX_WORKERS, os.cpu_count() or 2))
    chunks = max(1, workers // 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        harmonic, percussive = _medians(pool, magnitude, kernel_size, chunks)
        # softmask is elementwise: compute it on frequency bands.
        bounds = np.linspace(0, magnitude.shape[0], chunks + 1, dtype=int)
        bands = [slice(start, stop) for start, stop in zip(bounds[:-1], bounds[1:]) if stop > start]
        mask_parts = [
            (
                pool.submit(softmask, harmonic[band], percussive[band], power=2.0, split_zeros=True),
                pool.submit(softmask, percussive[band], harmonic[band], power=2.0, split_zeros=True),
                pool.submit(softmask, harmonic[band], percussive[band] * vocal_margin, power=2.0, split_zeros=False)
                if vocal_margin is not None else None,
            )
            for band in bands
        ]
        mask_harmonic = np.concatenate([h.result() for h, _, _ in mask_parts], axis=0)
        mask_percussive = np.concatenate([p.result() for _, p, _ in mask_parts], axis=0)
        mask_vocal = (np.concatenate([v.result() for _, _, v in mask_parts], axis=0)
                      if vocal_margin is not None else None)
        return mask_harmonic, mask_percussive, mask_vocal


def hpss_audio_with_spectra(
    y: np.ndarray, sample_rate: int, n_fft: int = HPSS_N_FFT, *, vocal_margin: float | None = None
) -> tuple[np.ndarray, np.ndarray, HpssSpectra]:
    """Harmonic and percussive signals of mono ``y`` and their spectra; with ``vocal_margin`` the spectra's ``vocal``
    too (the harmonic part with that margin, from the same STFT and medians)."""
    hop_length = n_fft // 4
    spectrum = stft(y, n_fft=n_fft, hop_length=hop_length, center=True, pad_mode="constant")
    magnitude = np.abs(spectrum)
    mask_harmonic, mask_percussive, mask_vocal = _masks(magnitude, HPSS_KERNEL_SIZE, vocal_margin)
    harmonic = magnitude * mask_harmonic
    percussive = magnitude * mask_percussive

    def inverse(mask: np.ndarray) -> np.ndarray:
        # Applying masks to the original complex spectrum avoids retaining a
        # second full-size complex phase matrix.
        return istft(
            spectrum * mask,
            dtype=y.dtype,
            n_fft=n_fft,
            hop_length=hop_length,
            length=y.shape[-1],
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        y_harmonic = pool.submit(inverse, mask_harmonic)
        y_percussive = pool.submit(inverse, mask_percussive)
        y_vocal = pool.submit(inverse, mask_vocal) if mask_vocal is not None else None
        spectra = HpssSpectra(int(sample_rate), int(n_fft), int(hop_length), magnitude, harmonic, percussive,
                              y_vocal.result() if y_vocal is not None else None)
        return y_harmonic.result(), y_percussive.result(), spectra


def hpss_audio(y: np.ndarray, n_fft: int = HPSS_N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic and percussive signals of mono ``y``."""
    y_harmonic, y_percussive, _ = hpss_audio_with_spectra(y, 0, n_fft)
    return y_harmonic, y_percussive
