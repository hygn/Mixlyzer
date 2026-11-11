"""The vocal context JumpCUE's pair features read (audio only).

- Vocal estimate: REPET-SIM on the mix mel power spectrogram. Each frame's
  background is the median of the frames elsewhere in the track that share its
  low band (bass line, kick): the same accompaniment, sung with other words or
  not sung at all. What stands above that median is taken as the vocal.
  Neighbours found on the full spectrum pair up the repeats of the same sung
  line (the chorus) and hide the vocal in the background.
- Vocal activity: the vocal estimate carries more than half of the voice-band
  power.
- Formant envelope: low-order cepstrum of the vocal estimate in the voice band
  (vowel colour; the harmonic ripple of the pitch sits in higher coefficients).
- The mix's own per-beat log mel and level.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
from scipy.fft import dct
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits

from analyzer.audio.conversion import mel_frequencies
from analyzer.audio.spectral import stft


@dataclass
class VocalContextConfig:
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 96
    fmin: float = 60.0
    fmax: float = 8000.0
    # REPET-SIM: neighbours are searched on the band below neighbor_fmax.
    knn: int = 30
    min_neighbor_sec: float = 2.0
    neighbor_fmax: float = 250.0
    # Vocal band (activity and formants)
    voice_fmin: float = 150.0
    voice_fmax: float = 5000.0
    # Activity: vocal share of voice-band power above threshold. The long
    # smoothing decides which beats are sung; the short one finds the gaps
    # between lines around a jump point.
    activity_threshold: float = 0.5
    activity_smooth_sec: float = 0.7
    gap_smooth_sec: float = 0.15
    # Formant cepstrum coefficients c[1 : 1 + n_ceps]
    n_ceps: int = 12
    # Formant frames sampled per beat
    beat_subdiv: int = 8


@dataclass
class VocalContext:
    frame_times: np.ndarray        # [T]
    gap_activity: np.ndarray       # [T] bool, the activity on the short smoothing (the gaps between lines)
    beat_mel: np.ndarray           # [B, n_mels] whitened log mel of the mix, L2 per beat
    beat_db: np.ndarray            # [B] mix level per beat (dB)
    beat_ceps: np.ndarray          # [B, subdiv, n_ceps] formant cepstrum, L2 per sample


def _power_mel(y: Optional[np.ndarray], sr: int, cfg: VocalContextConfig, magnitude: Optional[np.ndarray]) -> np.ndarray:
    if magnitude is None:
        magnitude = np.abs(stft(np.asarray(y, dtype=np.float32), n_fft=cfg.n_fft, hop_length=cfg.hop))
    fb = librosa.filters.mel(sr=sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    return (fb.astype(np.float32) @ (np.asarray(magnitude, dtype=np.float32) ** 2)).astype(np.float32)


def _repet_background(M: np.ndarray, feat: np.ndarray, k: int, min_gap: int, chunk: int = 128) -> np.ndarray:
    """REPET-SIM background of ``M`` [F, T]: per frame, the median of ``M`` over
    its top-``k`` cosine neighbours on ``feat`` [D, T], at least ``min_gap``
    frames away. Frame chunks run on threads (numpy releases the GIL), with the
    BLAS pool held at one thread."""
    _, T = feat.shape
    X = (feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-9)).astype(np.float32)
    Xt = np.ascontiguousarray(X.T)
    k = int(max(1, min(k, T - 2 * min_gap - 1)))
    bg = np.empty_like(M)

    def run(s: int) -> None:
        e = min(T, s + chunk)
        sim = Xt[s:e] @ X  # [c, T]
        # Slice the exclusion band row by row.  Building the former broadcast
        # distance matrix required a second int64 [chunk, T] allocation in
        # every worker, which became hundreds of MB on long tracks.
        for row, center in enumerate(range(s, e)):
            lo = max(0, center - min_gap + 1)
            hi = min(T, center + min_gap)
            sim[row, lo:hi] = -np.inf
        nn = np.argpartition(sim, -k, axis=1)[:, -k:]
        bg[:, s:e] = np.median(M[:, nn], axis=2)

    # Four smaller chunks bound the all-pairs working set while retaining
    # parallel BLAS work.  Results are identical because chunks are disjoint.
    workers = max(1, min(4, os.cpu_count() or 1))
    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(0, T, chunk)))
    return bg


def _beat_sample(frame_times: np.ndarray, beats_time: np.ndarray, subdiv: int) -> np.ndarray:
    """Frame index at ``subdiv`` evenly spaced phases of every beat. [B, subdiv]"""
    beats = np.asarray(beats_time, dtype=float)
    if beats.size < 2:
        return np.zeros((0, subdiv), dtype=np.int64)
    gaps = np.diff(beats)
    ends = np.append(beats[1:], beats[-1] + gaps[-1])
    phases = (np.arange(subdiv) + 0.5) / subdiv
    t = beats[:, None] + (ends - beats)[:, None] * phases[None, :]
    hop_sec = float(frame_times[1] - frame_times[0]) if frame_times.size > 1 else 1.0
    idx = np.rint(t / hop_sec).astype(np.int64)
    return np.clip(idx, 0, max(0, frame_times.size - 1))


def _beat_level_db(M: np.ndarray, frame_times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Mean mix power per beat in dB. [B]"""
    T = M.shape[1]
    hop_sec = float(frame_times[1] - frame_times[0]) if frame_times.size > 1 else 1.0
    edges = np.rint(np.append(beats, beats[-1] + (beats[-1] - beats[-2])) / hop_sec).astype(np.int64)
    edges = np.clip(edges, 0, T)
    power = np.concatenate([[0.0], np.cumsum(M.sum(axis=0, dtype=np.float64))])
    n = np.maximum(edges[1:] - edges[:-1], 1)
    mean = (power[edges[1:]] - power[edges[:-1]]) / n
    return (10.0 * np.log10(mean + 1e-12)).astype(np.float32)


def _beat_mel(logM: np.ndarray, frame_times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Mean log mel per beat, whitened per band, L2 per beat. [B, F]"""
    T = logM.shape[1]
    hop_sec = float(frame_times[1] - frame_times[0]) if frame_times.size > 1 else 1.0
    edges = np.rint(np.append(beats, beats[-1] + (beats[-1] - beats[-2])) / hop_sec).astype(np.int64)
    edges = np.clip(edges, 0, T)
    csum = np.concatenate([np.zeros((logM.shape[0], 1)), np.cumsum(logM, axis=1)], axis=1)
    n = np.maximum(edges[1:] - edges[:-1], 1)
    X = ((csum[:, edges[1:]] - csum[:, edges[:-1]]) / n).T
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def compute_vocal_context(
    y_mix: Optional[np.ndarray],
    sr: int,
    beats_time: np.ndarray,
    *,
    magnitude: Optional[np.ndarray] = None,
    config: Optional[VocalContextConfig] = None,
) -> VocalContext:
    """``y_mix`` is the full mix (not the HPSS harmonic part). ``magnitude`` is
    its STFT magnitude with the config's n_fft / hop, when already computed."""
    cfg = config or VocalContextConfig()
    beats = np.asarray(beats_time, dtype=float)
    M = _power_mel(y_mix, sr, cfg, magnitude)  # [F, T]
    _, T = M.shape
    hop_sec = cfg.hop / float(sr)
    frame_times = np.arange(T) * hop_sec

    mel_f = mel_frequencies(n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    band = (mel_f >= cfg.voice_fmin) & (mel_f <= cfg.voice_fmax)
    logM = np.log(M + 1e-6 * float(M.max()) + 1e-12)

    # REPET-SIM: background = median of the frames with the same low band.
    low = mel_f < cfg.neighbor_fmax
    min_gap = int(round(cfg.min_neighbor_sec / hop_sec))
    bg = _repet_background(M, logM[low] - logM[low].mean(axis=1, keepdims=True), cfg.knn, min_gap)
    V = M - np.minimum(bg, M)

    # Vocal activity.
    v_band = V[band].sum(axis=0)
    m_band = M[band].sum(axis=0)
    loud = m_band > 1e-4 * np.percentile(m_band, 95)

    def share(sec: float) -> np.ndarray:
        n = max(1, int(round(sec / hop_sec)))
        return uniform_filter1d(v_band, n) / (uniform_filter1d(m_band, n) + 1e-12)

    ratio = share(cfg.activity_smooth_sec)
    activity = (ratio > cfg.activity_threshold) & loud
    gap_activity = (share(cfg.gap_smooth_sec) > cfg.activity_threshold) & loud

    # Formant cepstrum of the vocal estimate in the voice band.
    logV = np.log(V[band] + 1e-6 * np.percentile(V[band], 99) + 1e-12)
    ceps = dct(logV, type=2, norm="ortho", axis=0)[1 : 1 + cfg.n_ceps].T  # [T, n_ceps]
    ref = ceps[activity] if activity.any() else ceps
    ceps = (ceps - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-9)
    ceps = ceps / (np.linalg.norm(ceps, axis=1, keepdims=True) + 1e-9)

    B = beats.size
    if B >= 2:
        idx = _beat_sample(frame_times, beats, cfg.beat_subdiv)
        beat_ceps = ceps[idx].astype(np.float32)
        beat_mel = _beat_mel(logM, frame_times, beats)
        beat_db = _beat_level_db(M, frame_times, beats)
    else:
        beat_ceps = np.zeros((B, cfg.beat_subdiv, cfg.n_ceps), dtype=np.float32)
        beat_mel = np.zeros((B, cfg.n_mels), dtype=np.float32)
        beat_db = np.zeros(B, dtype=np.float32)

    ctx = VocalContext(
        frame_times=frame_times,
        gap_activity=gap_activity,
        beat_mel=beat_mel,
        beat_db=beat_db,
        beat_ceps=beat_ceps,
    )
    return ctx

