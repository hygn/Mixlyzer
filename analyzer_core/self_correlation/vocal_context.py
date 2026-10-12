"""Vocal-aware context for Jump CUE placement (audio only).

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
  Two passages sung with the same words line up beat by beat; the same melody
  with other words does not. One beat is too noisy to decide, so the comparison
  is averaged over several beats.
- Jump point: the playback leaves one passage at a beat and continues in the
  other. When the two passages agree after the jump, the track goes on as it
  would have, in either direction; agreeing only before it changes where the
  track goes next. The beats after the jump (acoustic x lyric) score it, and a
  jump inside a running vocal line is penalised unless the words agree around
  it. The beats before it are reported but not scored.
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
    # Formant frames sampled per beat, shift tolerance in samples (singer timing)
    beat_subdiv: int = 8
    beat_shift_tol: int = 1
    min_active_subdiv: int = 3
    # Pairs for the track's unrelated-lyrics similarity level
    null_pairs: int = 2000


@dataclass
class JumpContextConfig:
    window_beats: int = 16          # beats compared before and after the jump
    decay_beats: float = 6.0        # weight of a beat decays with its distance from the jump
    # Windowed formant similarity above the track's unrelated level: at or
    # below lyric_diff the words differ, at or above lyric_same they agree.
    lyric_diff: float = 0.16
    lyric_same: float = 0.36
    local_beats: int = 2            # lyric agreement right at the jump (cut check)
    cut_guard_sec: float = 0.2      # vocal on both sides of a boundary = a line runs through it
    cut_weight: float = 0.5


@dataclass
class VocalContext:
    beats_time: np.ndarray         # [B]
    frame_times: np.ndarray        # [T]
    vocal_ratio: np.ndarray        # [T] vocal share of voice-band power (long smoothing)
    activity: np.ndarray           # [T] bool
    gap_activity: np.ndarray       # [T] bool, short smoothing
    beat_mel: np.ndarray           # [B, n_mels] whitened log mel of the mix, L2 per beat
    beat_db: np.ndarray            # [B] mix level per beat (dB)
    beat_ceps: np.ndarray          # [B, subdiv, n_ceps] formant cepstrum, L2 per sample
    beat_ceps_active: np.ndarray   # [B, subdiv] bool
    lyric_null: float              # formant similarity of unrelated sung beats
    shift_tol: int
    min_active: int

    def running_vocal(self, t: float, guard: float) -> float:
        """How much a vocal line runs through time ``t``: the smaller of the
        active shares just before and just after it."""
        ft = self.frame_times
        i0 = int(np.searchsorted(ft, t - guard))
        i1 = int(np.searchsorted(ft, t))
        i2 = int(np.searchsorted(ft, t + guard))
        if i1 <= i0 or i2 <= i1:
            return 0.0
        return float(min(np.mean(self.gap_activity[i0:i1]), np.mean(self.gap_activity[i1:i2])))


@dataclass
class JumpPointScore:
    beat_a: int
    beat_b: int
    score: float
    acoustic_pre: float
    acoustic_post: float
    lyric_pre: float                # 0 = other words, 1 = same words (or nothing sung)
    lyric_post: float
    vocal_cut: float                # penalty for leaving / entering a running line


def _power_mel(y: Optional[np.ndarray], sr: int, cfg: VocalContextConfig, magnitude: Optional[np.ndarray]) -> np.ndarray:
    if magnitude is None:
        magnitude = np.abs(librosa.stft(np.asarray(y, dtype=np.float32), n_fft=cfg.n_fft, hop_length=cfg.hop))
    fb = librosa.filters.mel(sr=sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    return (fb.astype(np.float32) @ (np.asarray(magnitude, dtype=np.float32) ** 2)).astype(np.float32)


def _repet_background(M: np.ndarray, feat: np.ndarray, k: int, min_gap: int, chunk: int = 512) -> np.ndarray:
    """REPET-SIM background of ``M`` [F, T]: per frame, the median of ``M`` over
    its top-``k`` cosine neighbours on ``feat`` [D, T], at least ``min_gap``
    frames away. Frame chunks run on threads (numpy releases the GIL), with the
    BLAS pool held at one thread."""
    _, T = feat.shape
    X = (feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-9)).astype(np.float32)
    Xt = np.ascontiguousarray(X.T)
    k = int(max(1, min(k, T - 2 * min_gap - 1)))
    cols = np.arange(T)
    bg = np.empty_like(M)

    def run(s: int) -> None:
        e = min(T, s + chunk)
        sim = Xt[s:e] @ X  # [c, T]
        sim[np.abs(cols[None, :] - np.arange(s, e)[:, None]) < min_gap] = -np.inf
        nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        bg[:, s:e] = np.median(M[:, nn], axis=2)

    workers = max(1, min(8, os.cpu_count() or 1))
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


def lyric_similarity(ctx: VocalContext, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Formant similarity of beats ``i`` and ``j`` (NaN unless both are sung)."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    out = np.full(i.shape, np.nan)
    B = ctx.beat_ceps.shape[0]
    ok = (i >= 0) & (j >= 0) & (i < B) & (j < B)
    if not ok.any():
        return out
    ii, jj = i[ok], j[ok]
    both = ctx.beat_ceps_active[ii] & ctx.beat_ceps_active[jj]
    C = np.einsum("nsc,ntc->nst", ctx.beat_ceps[ii], ctx.beat_ceps[jj])
    best = np.full(ii.shape, -np.inf)
    for shift in range(-ctx.shift_tol, ctx.shift_tol + 1):
        diag = np.diagonal(C, offset=shift, axis1=1, axis2=2)
        best = np.maximum(best, diag.mean(axis=1))
    vals = np.where(both.sum(axis=1) >= ctx.min_active, best, np.nan)
    out[ok] = vals
    return out


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

    mel_f = librosa.mel_frequencies(n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
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
        beat_ceps_active = activity[idx]
        beat_mel = _beat_mel(logM, frame_times, beats)
        beat_db = _beat_level_db(M, frame_times, beats)
    else:
        beat_ceps = np.zeros((B, cfg.beat_subdiv, cfg.n_ceps), dtype=np.float32)
        beat_ceps_active = np.zeros((B, cfg.beat_subdiv), dtype=bool)
        beat_mel = np.zeros((B, cfg.n_mels), dtype=np.float32)
        beat_db = np.zeros(B, dtype=np.float32)

    ctx = VocalContext(
        beats_time=beats,
        frame_times=frame_times,
        vocal_ratio=ratio.astype(np.float32),
        activity=activity,
        gap_activity=gap_activity,
        beat_mel=beat_mel,
        beat_db=beat_db,
        beat_ceps=beat_ceps,
        beat_ceps_active=beat_ceps_active,
        lyric_null=0.0,
        shift_tol=int(cfg.beat_shift_tol),
        min_active=int(cfg.min_active_subdiv),
    )
    # Unrelated level: sung beats paired at random.
    sung = np.flatnonzero(beat_ceps_active.sum(axis=1) >= cfg.min_active_subdiv)
    if sung.size >= 8:
        rng = np.random.default_rng(0)
        a = rng.choice(sung, cfg.null_pairs)
        b = rng.choice(sung, cfg.null_pairs)
        keep = np.abs(a - b) >= 8
        vals = lyric_similarity(ctx, a[keep], b[keep])
        if np.isfinite(vals).any():
            ctx.lyric_null = float(np.nanmean(vals))
    return ctx


def _side_mean(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted mean of the finite values and their share of the weight."""
    ok = np.isfinite(values)
    total = float(weights.sum())
    w = float(weights[ok].sum())
    if w <= 0.0 or total <= 0.0:
        return float("nan"), 0.0
    return float(np.dot(values[ok], weights[ok]) / w), w / total


def jump_point_scores(
    ctx: VocalContext,
    a_range: tuple[int, int],
    lag_beats: int,
    config: Optional[JumpContextConfig] = None,
) -> list[JumpPointScore]:
    """Scores of the jumps between beat ``k`` and beat ``k - lag_beats`` for
    ``k`` in ``a_range`` (inclusive start, exclusive end). The jump happens at
    the start of both beats: beats before it are heard from one passage, beats
    from it on from the other."""
    cfg = config or JumpContextConfig()
    B = ctx.beat_mel.shape[0]
    L = int(lag_beats)
    k0 = max(int(a_range[0]), L)
    k1 = min(int(a_range[1]), B)
    if L <= 0 or k1 <= k0:
        return []
    W = int(cfg.window_beats)

    # Per-beat comparisons of beat k with beat k - L over the searched span.
    lo = max(L, k0 - W)
    hi = min(B, k1 + W)
    ks = np.arange(lo, hi)
    acoustic = np.einsum("ij,ij->i", ctx.beat_mel[ks], ctx.beat_mel[ks - L])
    lyric = lyric_similarity(ctx, ks, ks - L)

    span = float(cfg.lyric_same - cfg.lyric_diff)

    def lyric_term(values: np.ndarray, weights: np.ndarray) -> float:
        mean, coverage = _side_mean(values, weights)
        if coverage <= 0.0:
            return 1.0
        agree = float(np.clip((mean - ctx.lyric_null - cfg.lyric_diff) / span, 0.0, 1.0))
        return 1.0 - coverage + coverage * agree

    decay = np.exp(-np.arange(W) / max(float(cfg.decay_beats), 1e-6))
    scores: list[JumpPointScore] = []
    for k in range(k0, k1):
        p = k - lo
        pre = slice(max(0, p - W), p)
        post = slice(p, min(ks.size, p + W))
        w_pre = decay[: pre.stop - pre.start][::-1]
        w_post = decay[: post.stop - post.start]
        ac_pre, _ = _side_mean(acoustic[pre], w_pre)
        ac_post, _ = _side_mean(acoustic[post], w_post)
        if not np.isfinite(ac_pre) or not np.isfinite(ac_post):
            continue
        ly_pre = lyric_term(lyric[pre], w_pre)
        ly_post = lyric_term(lyric[post], w_post)
        near = slice(max(0, p - cfg.local_beats), min(ks.size, p + cfg.local_beats))
        ly_near = lyric_term(lyric[near], np.ones(near.stop - near.start))
        t_a = float(ctx.beats_time[k])
        t_b = float(ctx.beats_time[k - L])
        running = max(ctx.running_vocal(t_a, cfg.cut_guard_sec), ctx.running_vocal(t_b, cfg.cut_guard_sec))
        cut = running * (1.0 - ly_near)
        score = max(ac_post, 0.0) * ly_post - cfg.cut_weight * cut
        scores.append(
            JumpPointScore(
                beat_a=k,
                beat_b=k - L,
                score=float(score),
                acoustic_pre=float(ac_pre),
                acoustic_post=float(ac_post),
                lyric_pre=float(ly_pre),
                lyric_post=float(ly_post),
                vocal_cut=float(cut),
            )
        )
    return scores


def score_jump_points(
    ctx: VocalContext,
    a_range: tuple[int, int],
    lag_beats: int,
    config: Optional[JumpContextConfig] = None,
) -> Optional[JumpPointScore]:
    """Best of :func:`jump_point_scores`."""
    scores = jump_point_scores(ctx, a_range, lag_beats, config)
    return max(scores, key=lambda s: s.score) if scores else None
