"""librosa.cqt's magnitude approximated from STFTs, for the analysis profiles that trade a little accuracy for speed
(analyzer.settings.AnalysisProfile.cqt_gains; the onboard analysis's measurements, standalone_dj_player
docs/onboard_analysis: about half the CQTs' CPU). Each CQT bin is read from the STFT of the one of three resolutions
its frequency needs: below C3 the signal at 1/8 of the rate, C3 - C5 at 1/2, above it at the rate itself, each
N_FFT points with its hop the frame grid's (hop / decimation): the same frame times as librosa.cqt's (centred). A bin's
value is the square root of the STFT power under a triangle one CQT bin wide each side on the log-frequency axis (where
that is narrower than the FFT's bins, linear interpolation between the two nearest), times a per-bin gain fitted
against librosa 0.11's cqt at tuning 0 (the gains file: "mix", C1 + 7 octaves at 36 a octave, and "melody", C4 + 3
octaves at 12, both at 22050 Hz and hop 256).

Its matrix products run where it is called: the frame features' worker threads, under the analysis's one BLAS thread
(a BLAS splitting them across threads changes their rounding from run to run)."""
from __future__ import annotations

import functools

import numpy as np

N_FFT = 2048
# (decimation, the highest frequency it reads (Hz), its frames every this many of the grid's: the low registers'
# long windows change slowly, read between them linearly)
TIERS = ((8, 130.8128, 4), (2, 523.2511, 2), (1, np.inf, 1))
GAIN_BINS = {"mix": 252, "melody": 36}


@functools.lru_cache(maxsize=4)
def load_gains(path: str) -> dict[str, np.ndarray]:
    """The gains file's per-bin gains by CQT ("mix", "melody"), checked: each one's length, finite and positive."""
    with np.load(path, allow_pickle=False) as z:
        gains = {name: np.asarray(z[name], np.float32) for name in GAIN_BINS}
    for name, n in GAIN_BINS.items():
        g = gains[name]
        if g.shape != (n,) or not np.all(np.isfinite(g)) or np.any(g <= 0):
            raise ValueError(f"{path}: the {name} CQT's gains are not {n} positive numbers")
    return gains


@functools.lru_cache(maxsize=16)
def _weights(sr_t: float, freqs: tuple, bins_per_octave: int) -> np.ndarray:
    fb = np.arange(N_FFT // 2 + 1) * sr_t / N_FFT
    W = np.zeros((len(freqs), fb.size), np.float32)
    with np.errstate(divide="ignore"):
        lf = np.log2(np.maximum(fb, 1e-9))
    for k, f in enumerate(freqs):
        tri = np.maximum(0.0, 1.0 - np.abs(lf - np.log2(f)) * bins_per_octave)
        x = f / (sr_t / N_FFT)
        i0 = int(np.floor(x))
        lin = np.zeros_like(tri)
        if i0 + 1 < fb.size:
            lin[i0], lin[i0 + 1] = 1.0 - (x - i0), x - i0
        w = np.maximum(tri, lin) if np.count_nonzero(tri) < 2 else tri
        W[k] = w / w.sum()
    return W


def _power(y: np.ndarray, hop: int) -> np.ndarray:
    import librosa
    S = librosa.stft(y, n_fft=N_FFT, hop_length=hop, window="hann", center=True, pad_mode="constant")
    return (S.real ** 2 + S.imag ** 2).astype(np.float32)


def pseudo_cqt(y: np.ndarray, sr: int, hop_length: int, fmin: float, n_bins: int, bins_per_octave: int,
               gains: np.ndarray) -> np.ndarray:
    """[n_bins, frames] |CQT| approximated: librosa.cqt's bins at tuning 0, fmin * 2 ** (k / bins_per_octave)."""
    import soxr

    y = np.ascontiguousarray(y, np.float32)
    n_frames = 1 + y.size // hop_length
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    out = np.zeros((n_bins, n_frames), np.float32)
    lo = 0.0
    for dec, hi, step in TIERS:
        sel = np.flatnonzero((freqs >= lo) & (freqs < hi))
        lo = hi
        if not sel.size:
            continue
        sr_t = sr / dec
        W = _weights(sr_t, tuple(float(f) for f in freqs[sel]), bins_per_octave)
        cols = np.flatnonzero(W.any(axis=0))
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        yt = y if dec == 1 else soxr.resample(y, sr, sr_t, quality="HQ").astype(np.float32)
        P = _power(yt, hop_length * step // dec)[c0:c1]
        M = np.sqrt(W[:, c0:c1] @ P)
        if step > 1:  # the grid's frames between the tier's, linearly
            x = np.arange(n_frames) / step
            i0 = np.minimum(np.floor(x).astype(int), M.shape[1] - 1)
            i1 = np.minimum(i0 + 1, M.shape[1] - 1)
            fr = (x - i0).astype(np.float32)
            M = M[:, i0] * (1.0 - fr) + M[:, i1] * fr
        n = min(n_frames, M.shape[1])
        out[sel, :n] = M[:, :n]
    return out * np.asarray(gains, np.float32)[:, None]
