"""JumpCUE's repeats: where a passage of the track comes again, a transposed one (a last chorus a semitone or two up)
too. Three steps, on the beats:

  1. ``pitch_beats``: the HPSS harmonic part on a semitone axis (triangular weights on the STFT bins, MIDI LO..HI), log,
     averaged per beat; each semitone standardized over the track, each beat of unit length;
  2. ``similarity``: the cosine of beat i and beat j moved by each transposition in SHIFTS, averaged over WINDOW beats
     along the diagonal; the best transposition kept;
  3. ``repeats``: on each lag of MIN_LAG beats or more (a diagonal), the runs where that likeness's z (against the
     track's own pairs of beats) stays at Z_MIN or above for MIN_LEN beats or more; a run beside a better one, on a lag
     within NMS_LAG and mostly the same beats, dropped; only the TOP_LAGS lags that repeat the most (beats x z) kept.

A repeat is (first beat, last beat, first beat + lag, last beat + lag, its mean z): the earlier passage first; its ends
may reach up to WINDOW beats beyond the passage (a window partly on it already scores). The lowest semitones, narrower
than an STFT bin, get no bin and stay empty.
Fitted on the tracks reviewed whole (2026-11-11, 244): the user's jump pairs on a repeat 86 %, those across a key change
71 % (the former mel self-correlation: 75 %, 42 %), the automatic pairs as often wrong as with it."""
from __future__ import annotations

import numpy as np

LO, HI = 28, 100               # the semitone axis, MIDI (41 Hz .. 2.6 kHz)
SHIFTS = (-2, -1, 0, 1, 2)     # the transpositions compared, semitones
WINDOW = 8                     # beats averaged along a diagonal
MIN_LAG = 8                    # beats between a passage and its repeat at least
MIN_LEN = 8                    # beats a repeat runs at least
Z_MIN = 2.5                    # the likeness's z a repeat keeps
NMS_LAG = 2                    # runs on lags this close, mostly on the same beats, are one repeat
TOP_LAGS = 10                  # the lags kept: those that repeat the most


def _semitone_weights(sample_rate: int, n_fft: int) -> np.ndarray:
    """[HI - LO + 1, n_fft // 2 + 1] the triangular weights of the STFT bins on each semitone."""
    f = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    midi = np.full(f.shape, -1e9)
    midi[1:] = 69.0 + 12.0 * np.log2(f[1:] / 440.0)
    m = np.arange(LO, HI + 1, dtype=float)[:, None]
    W = np.clip(1.0 - np.abs(midi[None, :] - m), 0.0, None)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-9)


def pitch_beats(harmonic: np.ndarray, sample_rate: int, n_fft: int, hop: int, beats: np.ndarray) -> np.ndarray:
    """[B, semitones] the harmonic magnitude [F, T] per beat on the semitone axis (a beat past the audio's end: its
    last frame)."""
    P = np.log1p(100.0 * (_semitone_weights(sample_rate, n_fft) @ harmonic))
    T = P.shape[1]
    t = np.arange(T) * hop / sample_rate
    edges = np.append(beats, beats[-1] + (np.median(np.diff(beats)) if beats.size > 1 else 0.5))
    idx = np.searchsorted(t, edges)
    starts = np.minimum(idx[:-1], T - 1)
    X = np.stack([P[:, s:max(e, s + 1)].mean(1) for s, e in zip(starts, idx[1:])])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def similarity(X: np.ndarray) -> np.ndarray:
    """[B - WINDOW + 1, same] the best transposition's cosine of two beats, averaged over WINDOW beats along the
    diagonal."""
    n = X.shape[0] - WINDOW + 1
    best = np.full((n, n), -np.inf, np.float32)
    for s in SHIFTS:
        Y = np.zeros_like(X)
        if s > 0:
            Y[:, s:] = X[:, :-s]
        elif s < 0:
            Y[:, :s] = X[:, -s:]
        else:
            Y = X
        C = X @ Y.T
        np.maximum(best, sum(C[k:k + n, k:k + n] for k in range(WINDOW)) / WINDOW, out=best)
    return best


def repeats(harmonic: np.ndarray, sample_rate: int, n_fft: int, hop: int, beats: np.ndarray) -> np.ndarray:
    """[N, 5] the track's repeats (first beat, last beat, first + lag, last + lag, mean z), best first."""
    beats = np.asarray(beats, dtype=float)
    if beats.size < WINDOW + MIN_LAG + MIN_LEN:
        return np.zeros((0, 5))
    S = similarity(pitch_beats(harmonic, sample_rate, n_fft, hop, beats))
    n = S.shape[0]
    vals = S[np.triu_indices(n, MIN_LAG)]
    mu, sd = float(vals.mean()), float(vals.std() + 1e-9)
    runs = []
    for lag in range(MIN_LAG, n):
        z = (np.diagonal(S, lag) - mu) / sd
        on = np.concatenate([[False], z >= Z_MIN, [False]])
        starts, ends = np.flatnonzero(on[1:] & ~on[:-1]), np.flatnonzero(~on[1:] & on[:-1])
        for i, j in zip(starts, ends):
            if j - i >= MIN_LEN - WINDOW + 1:
                runs.append((float(z[i:j].mean()), lag, int(i), int(j - 1 + WINDOW - 1)))
    runs.sort(reverse=True)
    kept: list = []
    for z, lag, a0, a1 in runs:
        if not any(abs(lag - lag2) <= NMS_LAG and min(a1, b1) - max(a0, b0) > 0.5 * (a1 - a0)
                   for _z, lag2, b0, b1 in kept):
            kept.append((z, lag, a0, a1))
    strength: dict = {}
    for z, lag, a0, a1 in kept:
        strength[lag] = strength.get(lag, 0.0) + z * (a1 - a0 + 1)
    top = set(sorted(strength, key=lambda lag: -strength[lag])[:TOP_LAGS])
    kept = [k for k in kept if k[1] in top]
    return np.asarray([(a0, a1, a0 + lag, a1 + lag, z) for z, lag, a0, a1 in kept], np.float64).reshape(-1, 5)
