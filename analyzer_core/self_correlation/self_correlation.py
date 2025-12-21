from __future__ import annotations
import numpy as np
import librosa
from typing import Tuple, Sequence

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def mel_beat_matrix(
    y: np.ndarray,
    beats_time: np.ndarray,
    sr: int,
    *,
    n_mels: int = 128,
    mel_fmin: float = 30.0,
    mel_fmax: float | None = None,
    n_beats_seg: int = 4,
    max_cols: int = 2000,
) -> tuple[np.ndarray, float, float]:
    """Build a beat-synchronous Mel feature matrix.

    Returns (Xmat, step_beats, avg_beat_sec)
      - Xmat: shape (F', T') where F' = n_mels * n_beats_seg if n_beats_seg>1
      - step_beats: decimation step in beat units (>=1)
      - avg_beat_sec: average seconds per beat (for unit conversions)
    """
    beats_time = np.asarray(beats_time, dtype=float)
    if beats_time.size < 2:
        return np.zeros((n_mels, 0), dtype=float), 1.0, 0.5

    beat_starts = np.round(beats_time[:-1] * sr).astype(int)
    beat_ends = np.round(beats_time[1:] * sr).astype(int)
    B = len(beat_starts)
    if B <= 0:
        return np.zeros((n_mels, 0), dtype=float), 1.0, 0.5

    # Per-beat Mel power
    max_len = int(np.max(beat_ends - beat_starts))
    max_len = int(max(512, min(max_len, 65536)))
    n_fft = 1 << int(np.ceil(np.log2(max_len)))
    mel_fb = librosa.filters.mel(sr=sr, n_fft=int(n_fft), n_mels=int(max(8, n_mels)),
                                 fmin=float(mel_fmin), fmax=(float(mel_fmax) if mel_fmax else None))
    M = mel_fb.shape[0]
    beat_mel = np.zeros((M, B), dtype=float)
    for i in range(B):
        s = int(max(0, beat_starts[i])); e = int(min(len(y), beat_ends[i]))
        if e - s < 2:
            continue
        seg = y[s:e].astype(float)
        w = np.hanning(len(seg))
        X = np.fft.rfft(seg * w, n=n_fft)
        P = (np.abs(X) ** 2)
        beat_mel[:, i] = mel_fb @ P

    # Log-compress then whiten across time (per Mel bin)
    mel_log = np.log1p(beat_mel)
    mu = np.mean(mel_log, axis=1, keepdims=True)
    sigma = np.std(mel_log, axis=1, keepdims=True) + 1e-12
    beat_feat = (mel_log - mu) / sigma

    # Concatenate n_beats_seg beats per column
    if n_beats_seg > 1 and B >= n_beats_seg:
        T_eff = B - n_beats_seg + 1
        Xmat = np.empty((M * n_beats_seg, T_eff), dtype=float)
        for i in range(T_eff):
            Xmat[:, i] = beat_feat[:, i:i + n_beats_seg].reshape(-1)
    else:
        Xmat = beat_feat

    # Optional decimation to cap width
    T_cols = Xmat.shape[1]
    decim_step = max(1, T_cols // int(max(1, max_cols)))
    if decim_step > 1:
        Xmat = Xmat[:, ::decim_step]

    beat_durs = np.diff(beats_time)
    avg_beat_sec = float(np.nanmean(beat_durs)) if beat_durs.size else 0.5
    step_beats = float(decim_step)
    return Xmat, step_beats, avg_beat_sec


def ssm_from_features(
    Xmat: np.ndarray,
    *,
    smooth_sigma: float = 1.0,
    diag_band: int = 2,
) -> np.ndarray:
    """Compute cosine SSM on whitened features (rows=time, cols=features).
    Expects Xmat as (F, T). Returns SSM as (T, T) in [-1, 1]."""
    if Xmat.size == 0:
        return np.zeros((0, 0), dtype=float)
    Xrows = Xmat.T
    # Already whitened at construction, still L2 per row to be safe
    Xn = Xrows / (np.linalg.norm(Xrows, axis=1, keepdims=True) + 1e-9)
    S = cosine_similarity(Xn)
    if float(smooth_sigma) > 0:
        S = gaussian_filter1d(S, sigma=float(smooth_sigma), axis=0)
        S = gaussian_filter1d(S, sigma=float(smooth_sigma), axis=1)
    # Diagonal band suppression
    H, W = S.shape
    ii = np.arange(H)[:, None]; jj = np.arange(W)[None, :]
    S = S.copy()
    S[np.abs(ii - jj) <= int(diag_band)] = 0.0
    return S


def diag_profile_peaks(
    S: np.ndarray,
    *,
    step_beats: float = 1.0,
    margin_beats: float = 16.0,
    prominence: float = 0.09,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute folded diagonal profile and detect interior peaks.
    Returns (lag_beats, resid_fold, peaks_idx)."""
    if S.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    H, W = S.shape
    ks = np.arange(-(W - 1), H, dtype=int)
    vals = []
    for k in ks:
        d = np.diag(S, k=int(k))
        vals.append(float(np.mean(d)) if d.size > 0 else np.nan)
    prof = np.asarray(vals, dtype=float)
    # Robust trend remove: smooth then subtract
    win = max(5, (len(prof) // 50) * 2 + 1)
    kern = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    prof_pad = np.pad(np.nan_to_num(prof, nan=np.nanmean(prof)), (pad, pad), mode="edge")
    trend = np.convolve(prof_pad, kern, mode="valid")
    resid = np.clip(prof - trend, 0.0, None)
    rmax = float(np.max(resid)) if resid.size else 1.0
    resid_norm = resid / (rmax if rmax > 0 else 1.0)

    # Fold +k with -k (average)
    ks_array = ks
    pos_mask = ks_array >= 0
    ks_pos = ks_array[pos_mask]
    idx_map = {int(v): i for i, v in enumerate(ks_array)}
    resid_fold = np.zeros_like(ks_pos, dtype=float)
    for i, k in enumerate(ks_pos):
        v_pos = resid_norm[idx_map.get(int(k), 0)]
        v_neg = resid_norm[idx_map.get(int(-k), 0)] if int(-k) in idx_map else v_pos
        resid_fold[i] = 0.5 * (float(v_pos) + float(v_neg))

    lag_beats = ks_pos.astype(float) * float(step_beats)

    # Peak detection within interior margins
    if lag_beats.size:
        max_b = float(lag_beats[-1])
        m = float(max(0.0, margin_beats))
        valid_mask = (lag_beats >= m) & (lag_beats <= (max_b - m))
    else:
        valid_mask = np.array([], dtype=bool)
    resid_use = resid_fold[valid_mask] if valid_mask.size else resid_fold
    distance = max(1, len(resid_use) // 50)
    peaks_loc, _ = find_peaks(resid_use, prominence=float(prominence), distance=distance, width=1)
    if valid_mask.size:
        idxs = np.flatnonzero(valid_mask)
        peaks_idx = idxs[peaks_loc]
    else:
        peaks_idx = peaks_loc

    return lag_beats, resid_fold, peaks_idx


def mel_frames(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 128,
    mel_fmin: float = 30.0,
    mel_fmax: float | None = None,
    whiten: bool = True,
    l2norm: bool = True,
) -> tuple[np.ndarray, float]:
    """Frame-wise Mel features (F x T) with optional whitening and L2 per frame.
    Returns (Feat, hop_sec)."""
    Mel = librosa.feature.melspectrogram(
        y=y.astype(float), sr=sr, n_fft=int(max(256, n_fft)), hop_length=int(max(1, hop)),
        n_mels=int(max(8, n_mels)), fmin=float(mel_fmin), fmax=float(mel_fmax) if mel_fmax else None,
        power=2.0, center=True
    ).astype(float)
    Feat = np.log1p(Mel)
    if whiten:
        mu = np.mean(Feat, axis=1, keepdims=True)
        sigma = np.std(Feat, axis=1, keepdims=True) + 1e-12
        Feat = (Feat - mu) / sigma
    if l2norm:
        Feat = Feat / (np.linalg.norm(Feat, axis=0, keepdims=True) + 1e-12)
    hop_sec = float(hop) / float(sr)
    return Feat, hop_sec


def beat_centers_to_frames(beats_time: np.ndarray, sr: int, hop: int) -> np.ndarray:
    beats_time = np.asarray(beats_time, dtype=float)
    if beats_time.size >= 2:
        centers = 0.5 * (beats_time[:-1] + beats_time[1:])
    else:
        centers = np.asarray([], dtype=float)
    return librosa.time_to_frames(centers, sr=sr, hop_length=int(max(1, hop))) if centers.size else np.asarray([], dtype=int)


def overlap_map_from_frames(
    Feat: np.ndarray,
    beat_frames: np.ndarray,
    lag_beats: float,
    avg_beat_sec: float,
    hop_sec: float,
    *,
    tol_beats: float = 3.0,
    anchor_restart: bool = True,
    use_beat_index_mapping: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (offset, time) similarity map using cosine over feature frames.
    Returns (sim2d, deltas, valid_idx).

    If use_beat_index_mapping=True, uses beat index offsets (robust to tempo drift):
      - baseline offset k = round(lag_beats) beats
      - tolerance deltas in integer beats
      - for each valid beat index j, compare frame at j with frame at j-k+delta
    Otherwise falls back to constant-seconds conversion using avg_beat_sec.
    """
    if Feat.size == 0 or beat_frames.size == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if use_beat_index_mapping:
        k_idx = int(round(float(lag_beats)))
        tol_idx = int(round(float(tol_beats)))
        # valid beat indices to anchor comparisons
        if anchor_restart:
            valid_idx = np.arange(k_idx, beat_frames.size, dtype=int)
        else:
            valid_idx = np.arange(0, max(0, beat_frames.size - k_idx), dtype=int)
        Tm = int(valid_idx.size)
        deltas = np.arange(-tol_idx, tol_idx + 1, dtype=int)
        Hm = deltas.size
        sim2d = np.zeros((Hm, max(1, Tm)), dtype=float)
        for jj, ii0 in enumerate(valid_idx):
            f0 = int(beat_frames[ii0])
            if f0 < 0 or f0 >= Feat.shape[1]:
                continue
            v0 = Feat[:, f0]
            for rr, dd in enumerate(deltas):
                if anchor_restart:
                    j1 = ii0 - k_idx + int(dd)
                else:
                    j1 = ii0 + k_idx + int(dd)
                if 0 <= j1 < beat_frames.size:
                    f1 = int(beat_frames[j1])
                    if 0 <= f1 < Feat.shape[1]:
                        sim2d[rr, jj] = float(np.dot(v0, Feat[:, f1]))
                    else:
                        sim2d[rr, jj] = -1.0
                else:
                    sim2d[rr, jj] = -1.0
        return sim2d, deltas, valid_idx
    else:
        lag_frames = int(round(float(lag_beats) * float(avg_beat_sec) / max(1e-9, hop_sec)))
        tol_frames = int(round(float(tol_beats) * float(avg_beat_sec) / max(1e-9, hop_sec)))
        valid_idx = np.flatnonzero(beat_frames >= lag_frames) if anchor_restart else np.arange(beat_frames.size, dtype=int)
        Tm = int(valid_idx.size)
        deltas = np.arange(-tol_frames, tol_frames + 1, dtype=int)
        Hm = deltas.size
        sim2d = np.zeros((Hm, max(1, Tm)), dtype=float)
        for jj, ii0 in enumerate(valid_idx):
            f0 = int(beat_frames[ii0])
            if f0 < 0 or f0 >= Feat.shape[1]:
                continue
            v0 = Feat[:, f0]
            for rr, dd in enumerate(deltas):
                f1 = (f0 - lag_frames + int(dd)) if anchor_restart else (f0 + lag_frames + int(dd))
                if 0 <= f1 < Feat.shape[1]:
                    sim2d[rr, jj] = float(np.dot(v0, Feat[:, int(f1)]))
                else:
                    sim2d[rr, jj] = -1.0
        return sim2d, deltas, valid_idx


def refine_offset_from_map(sim2d: np.ndarray, deltas: np.ndarray) -> tuple[float, float]:
    """Estimate best offset row by time-averaged maximum with quadratic refinement.
    Returns (r_ref, delta_ref) where r_ref is fractional row index and delta_ref is
    the refined offset in the same units as 'deltas' (frames or beats).
    """
    if sim2d.size == 0:
        return 0.0, 0.0
    row_mean = np.nanmean(sim2d, axis=1)
    r0 = int(np.nanargmax(row_mean)) if row_mean.size else 0
    if 0 < r0 < len(row_mean) - 1:
        y_m, y0, y_p = float(row_mean[r0 - 1]), float(row_mean[r0]), float(row_mean[r0 + 1])
        denom = (y_m - 2.0 * y0 + y_p)
        shift = 0.5 * (y_m - y_p) / denom if abs(denom) > 1e-12 else 0.0
        r_ref = float(r0) + float(np.clip(shift, -1.0, 1.0))
    else:
        r_ref = float(r0)
    delta_ref = float(deltas[0]) + float(r_ref)
    return r_ref, delta_ref


def flatten_band(sim2d: np.ndarray, r_ref: float, hop_sec: float, avg_beat_sec: float, *, band_beats: float = 1.0) -> np.ndarray:
    """Average rows within 짹band_beats (converted to frames) around r_ref to build 1D time series."""
    if sim2d.size == 0:
        return np.zeros(0, dtype=float)
    Hm, Tm = sim2d.shape
    beat_per_frame = hop_sec / max(1e-9, avg_beat_sec)
    band_frames = int(round(float(band_beats) / max(1e-9, beat_per_frame)))
    rr_c = int(np.clip(np.round(r_ref), 0, Hm - 1))
    lo_b = int(max(0, rr_c - band_frames))
    hi_b = int(min(Hm - 1, rr_c + band_frames))
    if hi_b >= lo_b:
        return np.nanmean(sim2d[lo_b:hi_b + 1, :], axis=0)
    return sim2d[rr_c]


def flatten_band_beats(sim2d: np.ndarray, r_ref: float, *, band_beats: float = 1.0) -> np.ndarray:
    """Average rows around r_ref using beat-indexed rows.

    This variant assumes the overlap map rows represent integer beat offsets
    (use_beat_index_mapping=True). In that case, one row ≈ one beat, so the
    window half-width is simply ±band_beats rows.
    """
    if sim2d.size == 0:
        return np.zeros(0, dtype=float)
    Hm, _Tm = sim2d.shape
    band_rows = int(max(0, round(float(band_beats))))
    rr_c = int(np.clip(np.round(r_ref), 0, Hm - 1))
    lo_b = int(max(0, rr_c - band_rows))
    hi_b = int(min(Hm - 1, rr_c + band_rows))
    if hi_b >= lo_b:
        return np.nanmean(sim2d[lo_b:hi_b + 1, :], axis=0)
    return sim2d[rr_c]
 
def link_similar_segments(
    y: np.ndarray,
    beats_time: np.ndarray,
    sr: int,
    *,
    # SSM + peak params
    n_beats_seg: int = 4,
    n_mels: int = 128,
    mel_fmin: float = 30.0,
    mel_fmax: float | None = None,
    smooth_sigma: float = 1.0,
    diag_band: int = 2,
    top_peaks: int = 5,
    prominence: float = 0.09,
    margin_beats: float = 16.0,
    # Overlap map params
    hop: int = 512,
    tol_beats: float = 3.0,
    anchor_restart: bool = True,
    band_beats: float = 1.0,
    # Spectral segment constraints
    cluster_min_beats: int = 8,
    cluster_merge_gap_beats: int = 2,
) -> list[dict]:
    """Find pairs of similar segments in a track.

    Returns a list of dicts with fields:
      - lag_beats, lag_sec, score
      - a: (start_sec, end_sec)
      - b: (start_sec, end_sec)
    """
    beats_time = np.asarray(beats_time, dtype=float)
    if beats_time.size < 2:
        return []

    # 1) Beat-wise features -> SSM -> lag peaks
    Xmat, step_beats, avg_beat_sec = mel_beat_matrix(
        y, beats_time, sr,
        n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
        n_beats_seg=n_beats_seg,
    )
    S = ssm_from_features(Xmat, smooth_sigma=smooth_sigma, diag_band=diag_band)
    lag_beats, resid_fold, peaks_idx = diag_profile_peaks(
        S, step_beats=step_beats, margin_beats=margin_beats, prominence=prominence
    )
    if peaks_idx.size == 0:
        return []

    # select top peaks by folded residual height
    order = np.argsort(resid_fold[peaks_idx])[::-1][:int(max(1, top_peaks))]
    sel_idx = peaks_idx[order]

    # 2) Frame-wise Mel features for overlap map
    Feat, hop_sec = mel_frames(y, sr, hop=hop, n_mels=n_mels, mel_fmin=mel_fmin, mel_fmax=mel_fmax)
    beat_frames = beat_centers_to_frames(beats_time, sr, hop)
    # center times for mapping indices to seconds
    centers = 0.5 * (beats_time[:-1] + beats_time[1:]) if beats_time.size >= 2 else np.asarray([], dtype=float)

    links: list[dict] = []
    for pk in sel_idx:
        k_beats = float(lag_beats[int(pk)])
        sim2d, deltas, valid_idx = overlap_map_from_frames(
            Feat, beat_frames, k_beats, avg_beat_sec, hop_sec,
            tol_beats=tol_beats, anchor_restart=anchor_restart, use_beat_index_mapping=True,
        )
        if sim2d.size == 0 or valid_idx.size == 0:
            continue

        # refine row and lag
        r_ref, delta_ref = refine_offset_from_map(sim2d, deltas)
        # deltas are in beats when use_beat_index_mapping=True
        delta_beats_ref = float(delta_ref)
        refined_lag_beats = (k_beats - delta_beats_ref) if anchor_restart else (k_beats + delta_beats_ref)
        lag_sec = refined_lag_beats * avg_beat_sec

        # Spectral segmentation on overlap map columns into similar time spans
        try:
            spec_segments, _spec_labels = spectral_segments_from_overlap(
                sim2d,
                k=2,
                smooth_sigma=3,
                band_window=0,
                min_len=int(max(1, cluster_min_beats)),
                merge_gap=int(max(0, cluster_merge_gap_beats)),
            )
        except Exception:
            spec_segments = []
        if not spec_segments:
            continue

        # Build 1D flattened line near optimal row for scoring only
        sim_line = flatten_band_beats(sim2d, r_ref, band_beats=band_beats)

        # map spectral segments to time pairs (a, b)
        for s0, e0 in spec_segments:
            jj0 = int(valid_idx[int(s0)])
            jj1 = int(valid_idx[int(max(s0, e0 - 1))])
            if jj0 >= centers.size or jj1 >= centers.size:
                continue
            a0 = float(centers[jj0]); a1 = float(centers[jj1])
            if anchor_restart:
                b0, b1 = a0 - lag_sec, a1 - lag_sec
            else:
                b0, b1 = a0 + lag_sec, a1 + lag_sec
            mean_score = float(np.mean(sim_line[int(s0):int(e0)])) if (e0 - s0) > 0 else -1.0
            links.append({
                "lag_beats": float(refined_lag_beats),
                "lag_sec": float(lag_sec),
                "a": (max(0.0, a0), max(0.0, a1)),
                "b": (max(0.0, b0), max(0.0, b1)),
                "score": mean_score,
            })

    # sort by score descending
    links.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return links

def spectral_segments_from_overlap(
    sim2d: np.ndarray,
    *,
    smooth_sigma: float = 0.5,
    band_window: int = 64,
    min_len: int = 0,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Dense-block segmentation on the overlap time-time SSM.

    Steps:
      - Per-column standardization + L2 normalization
      - Cosine similarity W on columns, clipped to [0,1]
      - Optional locality window and smoothing
      - Dense block search: pick the single interval with the highest normalized
        off-diagonal similarity (relative to the median baseline)

    Returns (segments, labels) where at most one segment is reported.
    Parameter `k` is retained for API compatibility but is ignored.
    """
    if sim2d.ndim != 2 or sim2d.shape[1] == 0:
        return [], np.zeros(0, dtype=int)
    H, T = sim2d.shape
    X = np.nan_to_num(sim2d, nan=0.0, posinf=0.0, neginf=0.0)
    # Per-column standardization (across offsets) and L2
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True) + 1e-12
    Z = (X - mu) / sd
    Z = Z / (np.linalg.norm(Z, axis=0, keepdims=True) + 1e-12)
    # Time-time affinities (non-negative for stable graph Laplacian)
    W = cosine_similarity(Z.T)
    W = np.clip(W, 0.0, 1.0)
    # Ensure strong self-loops
    np.fill_diagonal(W, 1.0)
    # Optional locality window
    if band_window and int(band_window) > 0 and T > 0:
        ii = np.arange(T)[:, None]; jj = np.arange(T)[None, :]
        W[np.abs(ii - jj) > int(band_window)] = 0.0
    if float(smooth_sigma) > 0:
        W = gaussian_filter1d(W, sigma=float(smooth_sigma), axis=0)
        W = gaussian_filter1d(W, sigma=float(smooth_sigma), axis=1)

    # Pre-compute off-diagonal sums for dense block scoring
    W_offdiag = W.copy()
    np.fill_diagonal(W_offdiag, 0.0)
    P_offdiag = np.zeros((T + 1, T + 1), dtype=float)
    P_offdiag[1:, 1:] = W_offdiag
    P_offdiag = P_offdiag.cumsum(axis=0).cumsum(axis=1)
    def _block_offdiag_sum(a: int, b: int) -> float:
        if a >= b:
            return 0.0
        return float(P_offdiag[b, b] - P_offdiag[a, b] - P_offdiag[b, a] + P_offdiag[a, a])
    base_offdiag = float(np.median(W_offdiag)) if W_offdiag.size else 0.0
    min_len_i = int(max(1, min_len))
    segments: list[tuple[int, int]] = []
    labels = np.zeros(T, dtype=int)

    try:
        best_score = -1e18
        best_pair: tuple[int, int] | None = None
        for a in range(0, max(0, T - min_len_i) + 1):
            for b in range(a + min_len_i, T + 1):
                L = b - a
                s_in = _block_offdiag_sum(a, b)
                denom = max(1.0, float(L * max(1, L - 1)))
                mean_in = s_in / denom
                score = (mean_in - base_offdiag) * np.sqrt(L)
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)
        if best_pair is not None and best_score > 0.0:
            segments = [best_pair]
            labels = np.zeros(T, dtype=int)
            labels[best_pair[0]:best_pair[1]] = 1
    except Exception:
        segments = []
        labels = np.zeros(T, dtype=int)

    try:
        if segments:
            lens = [b - a for a, b in segments]
            print(f"[DenseSSM] T={T} segments={len(segments)} lengths={lens}")
        else:
            print(f"[DenseSSM] T={T} no segments")
    except Exception:
        pass

    return segments, labels