from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import mode


@dataclass
class TempoSegment:
    start: float # Start time of Segment
    end: float # End time of Segment
    bpm: float # BPM of Segment
    inizio: float # First downbeat time of segment

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return self as tuple"""
        return (float(self.start), float(self.end), float(self.bpm), float(self.inizio))

    def to_dict(self) -> dict[str, float]:
        """Return self as dictionary"""
        return {
            "segment_start": float(self.start),
            "end": float(self.end),
            "bpm": float(self.bpm),
            "inizio": float(self.inizio),
        }

def _compute_odf(y: np.ndarray, sr: int, hop: int = 256) -> tuple[np.ndarray, float]:
    """Compute ODF from audio. \\
    Currenly using only librosa ODF but there is room for improvements"""
    odf1 = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float32)
    hop_t = hop / float(sr)
    return odf1, hop_t

def build_grid_from_period_phase(total_len_frames: int,
                                 hop_t: float,
                                 period_frames: float,
                                 phase_frames: float,
                                 subdiv: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Make beatgrid in sec using period and phase."""
    T = total_len_frames
    step = period_frames / max(1, int(subdiv))
    if not np.isfinite(step) or step <= 0:
        return np.array([]), np.array([])
    base = float(phase_frames) % step
    idx = base + np.arange(0.0, T - base + step, step, dtype=float)
    idx = idx[(idx >= 0) & (idx < T)]
    if subdiv > 1:
        main = float(phase_frames) % period_frames
        main_idx = main + np.arange(0.0, T - main + period_frames, period_frames, dtype=float)
        main_idx = main_idx[(main_idx >= 0) & (main_idx < T)]
        return (main_idx * hop_t, idx * hop_t)
    else:
        return (idx * hop_t, np.array([]))

def estimate_bpm_and_grid(odf: np.ndarray, odf_lp: np.ndarray, sr: int, hop: int,
                          bpm_lo: float, bpm_hi: float,
                          phase_bins: int = 128,
                          gamma_peak: float = 1.5,
                          prev_bpm: float = None,
                          use_only_prev_bpm:bool = False) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Get BPM, phase and Beatgrid from given ODF"""

    hop_t = hop / float(sr)
    T = len(odf)
    if T < 4:
        return float('nan'), np.array([]), np.array([]), np.array([])

    # Gamma 
    x = odf.astype(np.float32, copy=False)
    if gamma_peak != 1.0:
        xmax = float(np.max(x)) if np.max(x) > 0 else 1.0
        xw = np.power(np.clip(x, 0, xmax), float(gamma_peak), dtype=np.float32)
    else:
        xw = x
    if not use_only_prev_bpm:
        # Segment full ACF
        min_lag = int(np.floor((60.0 / (bpm_hi*1.1)) / hop_t))
        max_lag = int(np.ceil((60.0 / (bpm_lo*0.9)) / hop_t))
        max_lag = min(max_lag, T-1)
        if max_lag - min_lag < 8:
            return float('nan'), np.array([]), np.array([]), np.array([])
    
        seg = xw.astype(float)
        seg -= seg.mean()
        X = np.fft.rfft(seg)
        r = np.fft.irfft(np.abs(X)**2, n=seg.size) # Autocorrealtion IFFT(FFT^2)
        r[0] = 0.0 # DC Offset Rejection
        acf_flat = r[min_lag:max_lag+1].astype(np.float32)
        acf_flat = gaussian_filter1d(acf_flat, sigma=2.0)
        acf_flat -= acf_flat.min()
        m = acf_flat.max()
        if m > 0:
            acf_flat /= m
    
        # lag→BPM
        lags = np.arange(min_lag, max_lag+1, dtype=float)
        bpms_all = 60.0 / (lags * hop_t)
        valid = (bpms_all >= bpm_lo) & (bpms_all <= bpm_hi)
        if not np.any(valid):
            return float('nan'), np.array([]), np.array([]), np.array([])
    
        # From peaks to initial candidates
        peaks, _ = find_peaks(acf_flat[valid], prominence=0.01)
        if peaks.size == 0:
            idx = np.argmax(acf_flat[valid])
            peaks = np.array([idx], dtype=int)
        
        valid_idx = np.flatnonzero(valid)
        peaks_lag_idx = valid_idx[peaks]
        peaks_scores = acf_flat[peaks_lag_idx - min_lag]
    
        K = min(10, len(peaks_lag_idx))
        order = np.argsort(peaks_scores)[::-1][:K]
        lag_cands = lags[peaks_lag_idx[order]]
        bpm_cands = 60.0 / (lag_cands * hop_t)
        if prev_bpm:
            bpm_cands = np.unique(np.concatenate([bpm_cands, prev_bpm]))
    else:
        bpm_cands = np.asarray([prev_bpm])

    
    print(f"[sweep] bpm cands = {bpm_cands}")

    def best_phase_for_bpm(odf, bpm: float, base_bins: int = None, sigma_frac: float = 0.05) -> tuple[float, float]:
        """Get best phase and score for specific BPM using FFT (Gaussian pulse train)"""
        period = 60.0 / bpm / hop_t  # in frames(samples)
        if not np.isfinite(period) or period < 3.0:
            return 0.0, -1e18
        bins = max(128, phase_bins if base_bins is None else base_bins)
        T_pad = int(2 ** np.ceil(np.log2(T)))
        X = np.fft.rfft(odf, n=T_pad)
        freqs = np.fft.rfftfreq(T_pad, d=1.0)  # cycles/sample, so 2π*freqs is rad/sample
        Np = int(T / period)
        # finite comb
        denom = 1.0 - np.exp(-2j * np.pi * freqs * period)
        comb = (1.0 - np.exp(-2j * np.pi * freqs * Np * period)) / (denom + 1e-12)
        sigma = float(sigma_frac * period)  # sigma in samples
        G = np.exp(-0.5 * (2.0 * np.pi * freqs * sigma) ** 2) # Gaussian pulse shape in frequency
        geom = comb * G  # Gaussian pulse train spectrum
        corr = np.fft.irfft(X * np.conj(geom), n=T_pad)
        corr = corr[: int(np.floor(period))]
        step = len(corr) / bins
        vals = np.array([corr[int(i * step)] for i in range(bins)], dtype=float)
        k = int(np.argmax(vals))
        phi_best = float(k * (period / bins))
        val_best = float(vals[k])
        return phi_best, val_best

    def score_from_bestphase(odf, bpm: float, base_bins=None) -> tuple[float, float]:
        # Wrapper for best_phase_for_bpm and apply some scoring rule
        phi_b, val_b = best_phase_for_bpm(odf, bpm, base_bins=base_bins if base_bins else max(phase_bins, 128))
        if not np.isfinite(val_b):
            return phi_b, -1e18
        if abs(bpm - int(bpm)) > 0.1:
            val_b *= 0.5 # prefer integer BPM
        return phi_b, float(val_b)

    def refine_bpm_phase(bpm0: float) -> tuple[float, float, float]:
        """By sweeping BPM in specific range, get best BPM and phase"""
        bpm = float(np.clip(bpm0, bpm_lo, bpm_hi))
        phi, val = score_from_bestphase(xw, bpm)
        best_local = (val, bpm, phi)

        # coarse local sweep
        local_band = 5
        b_grid = np.arange(max(bpm_lo, bpm - local_band),
                           min(bpm_hi, bpm + local_band) + 1e-12,
                           0.05, dtype=float)
        if b_grid.size:
            for b in b_grid:
                phi_b, v = score_from_bestphase(xw, b)
                if v > best_local[0]:
                    best_local = (v, b, phi_b)
        print(f"[sweep] sweep => {best_local[1]} BPM, Φ = {best_local[2]}, v = {best_local[0]}")

        # fine sweep
        _, bpm, phi = best_local
        b_grid = np.arange(max(bpm_lo, bpm - 0.1),
                           min(bpm_hi, bpm + 0.1) + 1e-12,
                           0.01, dtype=float)
        for b in b_grid:
            phi_b, v = score_from_bestphase(xw, b)
            if v > best_local[0]:
                best_local = (v, b, phi_b)
        
        print(f"[sweep] sweep => {best_local[1]} BPM, Φ = {best_local[2]}, v = {best_local[0]}")

        # micro sweep
        _, bpm, phi = best_local
        b_grid = np.arange(max(bpm_lo, bpm - 0.02),
                           min(bpm_hi, bpm + 0.02) + 1e-12,
                           0.001, dtype=float)
        for b in b_grid:
            phi_b, v = score_from_bestphase(xw, b)
            if v > best_local[0]:
                best_local = (v, b, phi_b)
        
        print(f"[sweep] sweep => {best_local[1]} BPM, Φ = {best_local[2]}, v = {best_local[0]}")

        val, bpm, phi = best_local
        return bpm, phi, val

    best_triplet = (-1e18, bpm_cands[0], 0.0)
    for b_seed in bpm_cands: # Iterate sweep within cands
        b_ref, phi_ref, v_ref = refine_bpm_phase(b_seed)
        if v_ref > best_triplet[0]:
            best_triplet = (v_ref, b_ref, phi_ref)

    _, bpm_final, phi_final = best_triplet
    period_final = 60.0 / bpm_final / hop_t

    grid_main_sec, _ = build_grid_from_period_phase( # build grid
        total_len_frames=T, hop_t=hop_t,
        period_frames=period_final, phase_frames=phi_final,
        subdiv=1
    )
    beats_sec = grid_main_sec

    downbeat_sec = 0.0
    # downbeat detection
    if beats_sec.size:
        beat_frames = np.clip(
            np.round(beats_sec / hop_t).astype(np.int32),
            0,
            max(0, T - 1)
        )
        onset_vals = np.asarray(odf_lp, dtype=float)
        downbeat_scores = np.full(4, -np.inf, dtype=float)
        for offset in range(min(4, beat_frames.size)):
            idx = beat_frames[offset::4]
            if idx.size:
                downbeat_scores[offset] = float(np.sum(onset_vals[idx])) # get sum of ODF for all four cases
        best_offset = int(np.argmax(downbeat_scores))# Get best
        best_offset = min(best_offset, beats_sec.size - 1)
        downbeat_sec = float(beats_sec[best_offset])
    print(f"[sweep] downbeat detected => {downbeat_sec} s")

    return float(bpm_final), beats_sec, downbeat_sec

def coarse_bpm_acf(
    odf: np.ndarray, sr: int, hop: int,
    bounds: tuple[float, float], win_s, step_s ) -> tuple[np.ndarray, np.ndarray, float, int]:
    hop_t = hop / float(sr)
    q = 1
    odf_ds = odf
    hop_t_ds = hop_t * q
    N = len(odf_ds)
    if N == 0:
        return np.array([], dtype=int), hop_t_ds, q, np.empty((0, 0), dtype=np.float32)

    win_exact = win_s / hop_t_ds if hop_t_ds > 0 else 0.0
    win = max(1, int(round(win_exact))) if win_exact > 0 else 1
    half_exact = win_exact / 2.0 if win_exact > 0 else 0.0
    step_frames = max(step_s / hop_t_ds, 1e-9) if hop_t_ds > 0 else 1.0

    pad = int(np.ceil(half_exact)) if win > 1 else 0
    if win > 1:
        padded = np.pad(odf_ds.astype(float), (pad, pad), mode="edge")
    else:
        padded = odf_ds.astype(float)

    min_lag = int(np.floor((60.0 / bounds[1]) / hop_t_ds))
    max_lag = int(np.ceil((60.0 / bounds[0]) / hop_t_ds))
    acf_stack = []

    centers: list[float] = []
    center = 0.0
    last_idx = float(max(0, N - 1))
    while center <= last_idx:
        centers.append(center)
        center += step_frames
    if centers[-1] < last_idx:
        centers.append(last_idx)

    for c in centers:
        center_shifted = c + pad
        start = int(np.floor(center_shifted - half_exact))
        end = start + win
        segment = padded[start:end]
        if segment.shape[0] < win:
            segment = np.pad(segment, (0, win - segment.shape[0]), mode="edge")
        x = segment - segment.mean()

        X = np.fft.rfft(x)
        r = np.fft.irfft(np.abs(X) ** 2)
        r[0] = 0.0
        acf_stack.append(r[min_lag:max_lag])

    acf_stack = np.array(acf_stack, dtype=np.float32)

    centers = np.asarray(centers, float)
    hop_t_ds = hop_t * q

    return centers, hop_t_ds, q, acf_stack

def compute_ssm_from_acf(acf_stack: np.ndarray, blur_sigma: float = 1.0) -> np.ndarray:
    X = acf_stack / (np.max(acf_stack, axis=1, keepdims=True) + 1e-9)
    S = cosine_similarity(X)
    if blur_sigma > 0:
        S = gaussian_filter1d(S, sigma=blur_sigma, axis=0)
        S = gaussian_filter1d(S, sigma=blur_sigma, axis=1)
    return S

def _build_label_segments(labels: np.ndarray, min_segment_frames: int) -> list[tuple[int, int, int]]:
    """
    Build contiguous segments of identical labels and normalize them by
    absorbing short runs into the previous segment whenever possible.
    """
    if labels.size == 0:
        return [(0, 0, 0)]

    # Initial segmentation based on raw label changes.
    segments: list[list[int]] = []
    seg_start = 0
    current_label = int(labels[0])
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label == current_label:
            continue
        segments.append([seg_start, idx, current_label])
        seg_start = idx
        current_label = label
    segments.append([seg_start, len(labels), current_label])

    if min_segment_frames <= 1:
        min_segment_frames = 1

    # Absorb short segments into the previous one (when it exists).
    consolidated: list[list[int]] = []
    for start, end, lbl in segments:
        seg_len = end - start
        if consolidated and seg_len < min_segment_frames:
            consolidated[-1][1] = end
            continue
        consolidated.append([start, end, lbl])

    # If the very first segment is still too short, merge it forward.
    if len(consolidated) > 1:
        first_len = consolidated[0][1] - consolidated[0][0]
        if first_len < min_segment_frames:
            consolidated[1][0] = consolidated[0][0]
            consolidated.pop(0)

    # Merge neighbours that share the same label (after absorption).
    merged: list[list[int]] = []
    for start, end, lbl in consolidated:
        if merged and lbl == merged[-1][2]:
            merged[-1][1] = end
            continue
        merged.append([start, end, lbl])

    return [(int(start), int(end), int(lbl)) for start, end, lbl in merged]

def _preview_label_segments(segments: list[tuple[int, int, int]], limit: int = 6) -> str:
    if not segments:
        return "[]"
    parts = [
        f"L{lbl}[{start},{end})"
        for start, end, lbl in segments[:limit]
    ]
    if len(segments) > limit:
        parts.append("...")
    return "[" + ", ".join(parts) + "]"

def detect_boundaries_from_ssm(
    S: np.ndarray,
    hop_t: float,
    max_segments: int = 16,
    smooth_sigma: float = 0.1,
    R: float = 0.005,
    eig_min: int = 1,
    eig_max: int = 12,
    min_gap_s: float = 5.0,
) -> np.ndarray:
    
    N = S.shape[0]
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    
    S_sym = ((S + S.T) / 2.0) - 0.5
    S_smooth = gaussian_filter(S_sym, sigma=smooth_sigma)
    W = np.clip(S_smooth, 0, 1)
    D = np.diag(W.sum(axis=1)) # Laplacian
    L = D - W

    k_try = max(eig_max, 3)
    vals, vecs = eigsh(L, k=k_try, which="SM") # Eigen Decomposition
    eigvals = np.real(vals)

    diffs = np.diff(eigvals)
    k_opt = np.argmax(diffs[eig_min-1:]) + eig_min
    k_opt = np.clip(k_opt, 2, max_segments)
    print(f"[SSM] estimated optimal segments = {k_opt} (eigengap={diffs[k_opt-1]:.4f})")

    X = vecs[:, :k_opt]
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=k_opt, n_init=max_segments, random_state=42, tol=0.00001)
    raw_labels = km.fit_predict(X) # Cluster

    smoothed = gaussian_filter1d(raw_labels.astype(float), sigma=0.5)
    labels = np.round(smoothed).astype(int)
    labels = np.clip(labels, 0, k_opt - 1)

    min_gap_frames = max(int(round(min_gap_s / hop_t)), 1)
    label_segments = _build_label_segments(labels, min_gap_frames)
    if not label_segments:
        label_segments = [(0, N, 0)]

    boundaries = np.array([end for _, end, _ in label_segments], dtype=int)
    print(
        f"[SSM] final boundaries = {len(boundaries)} | label-segments={len(label_segments)} "
        f"| min_gap={min_gap_s:.2f}s ({min_gap_frames} frames)"
    )
    print(f"[SSM] label preview: {_preview_label_segments(label_segments)}")
    return boundaries, labels, label_segments

def get_seg_frames_from_boundaries(
    odf: np.ndarray,
    boundaries: np.ndarray,
    label_segments: list[tuple[int, int, int]],
    hop_t: float,
    min_seg_s: float,
):
    def _preview(frames: list[tuple[int, int]], labels: list[int] | None = None, limit: int = 4) -> str:
        if not frames:
            return "[]"
        display_entries = []
        for idx, (s, e) in enumerate(frames[:limit]):
            if labels and idx < len(labels):
                display_entries.append(f"({s},{e}):L{labels[idx]}")
            else:
                display_entries.append(f"({s},{e})")
        display = ", ".join(display_entries)
        return f"[{display}{', ...' if len(frames) > limit else ''}]"

    def _var(buf: np.ndarray) -> float:
        if buf.size == 0:
            return 0.0
        return np.var(buf, dtype=np.float64)
        #return float(np.sqrt(np.mean(np.square(buf, dtype=np.float64))))

    total = len(odf)
    print(f"[seg] building segments from {len(boundaries)} boundaries (odf_len={total})")

    if total == 0:
        return []

    if not label_segments:
        print("[seg] no label segments detected; falling back to a single range")
        seg_frames = [(0, total)]
        seg_labels = [0]
    else:
        ssm_span = label_segments[-1][1]
        scale = total / ssm_span if ssm_span else 0.0

        scaled_segments: list[list[int]] = []
        for start, end, lbl in label_segments:
            start_frame = int(round(start * scale)) if ssm_span else 0
            end_frame = int(round(end * scale)) if ssm_span else total
            start_frame = max(0, min(start_frame, total))
            end_frame = max(start_frame + 1 if total > start_frame else start_frame, min(end_frame, total))
            scaled_segments.append([start_frame, end_frame, lbl])
        scaled_segments[-1][1] = total  # Guarantee coverage of the tail.

        min_seg_frames = max(int(round(min_seg_s / hop_t)), 1) if hop_t > 0 else 1
        absorbed: list[list[int]] = []
        for start, end, lbl in scaled_segments:
            seg_len = end - start
            if absorbed and seg_len < min_seg_frames:
                absorbed[-1][1] = end
                continue
            absorbed.append([start, end, lbl])

        if len(absorbed) > 1:
            first_len = absorbed[0][1] - absorbed[0][0]
            if first_len < min_seg_frames:
                absorbed[1][0] = absorbed[0][0]
                absorbed.pop(0)

        merged_by_label: list[list[int]] = []
        for start, end, lbl in absorbed:
            if merged_by_label and lbl == merged_by_label[-1][2]:
                merged_by_label[-1][1] = end
                continue
            merged_by_label.append([start, end, lbl])

        seg_frames = []
        seg_labels = []
        for start, end, lbl in merged_by_label:
            seg_frames.append((start, end))
            seg_labels.append(lbl)

    print(f"[seg] raw label-based segments={len(seg_frames)} preview={_preview(seg_frames, labels=seg_labels)}")

    seg_mask: list[int] = []
    odf_var = _var(odf)
    threshold = odf_var * 0.1
    verbose = len(seg_frames) <= 32
    for idx, (s, e) in enumerate(seg_frames):
        local = odf[s:e]
        local_var = _var(local)
        keep = 1 if (e - s) > 0 and local_var > threshold else 0
        seg_mask.append(keep)
        if verbose:
            print(f"[seg]   seg#{idx:03d} frames=({s},{e}) var={local_var:.6f} thr={threshold:.6f} -> {'keep' if keep else 'merge'}")

    seg_filtered = [seg_frames[0]]
    for idx, keep in enumerate(seg_mask[1:], start=1):
        if keep:
            seg_filtered.append(seg_frames[idx])
        else:
            e = seg_frames[idx][1]
            latest_seg = seg_filtered[-1]
            seg_filtered[-1] = (latest_seg[0], e)
    print(f"[seg] filtered segments={len(seg_filtered)} (removed {len(seg_frames) - len(seg_filtered)}) preview={_preview(seg_filtered)}")
    return seg_filtered

def _score_beats_against_odf(beats_sec: np.ndarray, odf: np.ndarray, hop_t: float) -> float:
    if hop_t <= 0 or odf.size == 0 or beats_sec.size == 0:
        return float("-inf")
    beat_frames = np.round(beats_sec / hop_t).astype(np.int64, copy=False)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < len(odf))]
    if beat_frames.size == 0:
        return float("-inf")
    vals = odf[beat_frames]
    return float(np.mean(vals, dtype=np.float64))

def _segments_to_array(segments: list[TempoSegment]) -> np.ndarray:
    if not segments:
        return np.empty((0, 4), dtype=float)
    return np.asarray([seg.as_tuple() for seg in segments], dtype=float)

def _segments_from_array(arr: np.ndarray) -> list[TempoSegment]:
    if arr.size == 0:
        return []
    return [TempoSegment(*row[:4]) for row in arr]

def refine_segments_via_beatgrid(
    segments: list[TempoSegment],
    odf: np.ndarray,
    hop_t: float,
    win_s: float,
    step_s: float,
    valid_trange: tuple[float] = None
) -> tuple[list[TempoSegment], np.ndarray]:
    seg_arr = _segments_to_array(segments)
    if seg_arr.ndim != 2 or seg_arr.shape[0] == 0:
        return segments, np.asarray([], dtype=float)
    if hop_t <= 0 or win_s <= 0 or step_s <= 0:
        return segments, np.asarray([], dtype=float)

    from analyzer_core.editor.beatgrid import rebuild_grid_from_segments, canonical_inizio

    seg_arr = seg_arr.copy()
    beats = rebuild_grid_from_segments(seg_arr)
    best_score = _score_beats_against_odf(beats, odf, hop_t)
    if not np.isfinite(best_score):
        best_score = float("-inf")

    half_shift = max(win_s / 2.0, step_s)
    min_seg_len = max(step_s * 0.5, 0.5)
    step = max(step_s, hop_t * 2.0)

    for idx in range(len(seg_arr) - 1):
        base_boundary = seg_arr[idx, 1]
        left_min = seg_arr[idx, 0] + min_seg_len
        right_max = seg_arr[idx + 1, 1] - min_seg_len
        if left_min >= right_max:
            continue

        candidate_shifts = np.arange(-half_shift, half_shift + step * 0.5, step)
        best_local_boundary = base_boundary
        best_local_score = best_score
        best_candidate_arr: np.ndarray | None = None
        best_candidate_beats: np.ndarray | None = None

        for delta in candidate_shifts:
            new_boundary = np.clip(base_boundary + delta, left_min, right_max)
            if not np.isfinite(new_boundary):
                continue

            candidate = seg_arr.copy()
            candidate[idx, 1] = new_boundary
            candidate[idx + 1, 0] = new_boundary

            beats_candidate = rebuild_grid_from_segments(candidate)
            score = _score_beats_against_odf(beats_candidate, odf, hop_t)
            if score > best_local_score + 1e-9:
                old_inizio = candidate[idx + 1, 3]
                new_inizio = canonical_inizio(old_inizio, candidate[idx + 1, 0], candidate[idx + 1, 2])
                candidate[idx + 1, 3] = new_inizio
                best_local_score = score
                best_local_boundary = new_boundary
                best_candidate_arr = candidate
                best_candidate_beats = beats_candidate

        if best_local_boundary != base_boundary:
            if best_candidate_arr is None or best_candidate_beats is None:
                continue
            seg_arr = best_candidate_arr
            beats = best_candidate_beats
            best_score = best_local_score
            print(f"[seg] refined boundary idx={idx} {base_boundary:.3f}->{best_local_boundary:.3f} score={best_local_score:.6f}")
    
    if seg_arr[0, 1] - max(seg_arr[0, 0], valid_trange[0]) < win_s / 2:
        cand = seg_arr.copy()
        cand = cand[1:]
        cand[0, 0] = 0.0
        old_inizio = cand[0, 3]
        new_inizio = canonical_inizio(old_inizio, cand[0, 0], cand[0, 2])
        cand[0, 3] = new_inizio
        beats = rebuild_grid_from_segments(cand)
        seg_arr = cand
    
    if min(seg_arr[-1, 1], valid_trange[1]) - seg_arr[-1, 0] < win_s / 2:
        cand = seg_arr.copy()
        cand = cand[0:-1]
        cand[-1, 1] = seg_arr[-1, 1]
        beats = rebuild_grid_from_segments(cand)
        seg_arr = cand

    if beats.size == 0:
        beats = rebuild_grid_from_segments(seg_arr)

    return _segments_from_array(seg_arr), beats


def _coarse_segment_track(
    odf: np.ndarray,
    sr: int,
    hop_for_odf: int,
    hop_t: float,
    win_s: float,
    step_s: float,
    seg_min_s: float = 5,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    print(f"[coarse] win={win_s} s step={step_s} s")
    centers, hop_t_ds, q, acf_stack = coarse_bpm_acf(
        odf, sr, hop_for_odf, (1 / (win_s * 60), 2000), win_s, step_s
    )
    print(f"[coarse] centers={len(centers)} hop_t_ds={hop_t_ds:.6f}s decim={q}")
    S = compute_ssm_from_acf(acf_stack, blur_sigma=0)
    boundaries, labels, label_segments = detect_boundaries_from_ssm(
        S, hop_t=step_s, smooth_sigma=0.1, min_gap_s=seg_min_s
    )
    boundaries = np.multiply(np.array(boundaries, dtype=np.float32), len(odf) / (S.shape[0] - 1))
    boundaries = boundaries.astype(np.int32)
    boundaries = np.clip(boundaries, 0, len(odf) - 1)
    print(f"[seg] boundaries(frames): {boundaries[:8]}{' ...' if len(boundaries) > 8 else ''}")

    seg_frames = get_seg_frames_from_boundaries(
        odf, boundaries, label_segments, hop_t, min_seg_s=seg_min_s
    )
    return seg_frames, boundaries

def _estimate_tempo_segments(
    seg_frames: list[tuple[int, int]],
    odf: np.ndarray,
    odf_lp: np.ndarray,
    sr: int,
    hop_for_odf: int,
    bpm_bounds: tuple[float, float],
    fine_phase_bins: int,
    hop_t: float,
) -> tuple[list[TempoSegment], list[float], list[tuple[float, float, float]], float]:
    beats_all: list[float] = []
    tempo_segments: list[TempoSegment] = []
    seg_meta: list[tuple[float, float, float]] = []
    first_downbeat = float("nan")

    for s, e in seg_frames:
        local = odf[s:e]
        local_lp = odf_lp[s:e]
        seg_len_s = (e - s) * hop_t
        print(f"[fine] segment frames=({s},{e}) dur={seg_len_s:.2f}s")
        start_sec = s * hop_t
        end_sec = e * hop_t
        if local.size < 8:
            print("[fine] skip short segment")
            seg_meta.append((start_sec, end_sec, float("nan")))
            continue

        bpm_est, beats_sec, downbeat_sec = estimate_bpm_and_grid(
            local,
            local_lp,
            sr,
            hop_for_odf,
            bpm_lo=bpm_bounds[0],
            bpm_hi=bpm_bounds[1],
            phase_bins=fine_phase_bins,
            prev_bpm=None,
        )
        seg_meta.append((start_sec, end_sec, float(bpm_est) if np.isfinite(bpm_est) else np.nan))

        if beats_sec.size:
            beats_all.extend((beats_sec + start_sec).tolist())
            downbeat_local = float(downbeat_sec)
            if not np.isfinite(first_downbeat):
                first_downbeat = float(start_sec + beats_sec[0])
        else:
            downbeat_local = 0.0
            if not np.isfinite(first_downbeat):
                first_downbeat = float(start_sec)

        if np.isfinite(bpm_est):
            tempo_segments.append(
                TempoSegment(
                    start=float(start_sec),
                    end=float(end_sec),
                    bpm=float(bpm_est),
                    inizio=float(start_sec + downbeat_local),
                )
            )

    if not np.isfinite(first_downbeat):
        first_downbeat = 0.0

    return tempo_segments, beats_all, seg_meta, first_downbeat

def get_track_validrange(audio: np.ndarray, sr: int, window_s: float = 1, thresh: float = 0.01):
    m = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if m <= 0:
        return 0.0, 0.0
    y = (audio / m).astype(np.float32)
    frame_length = max(1, int(round(window_s * sr)))
    hop_length = frame_length

    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True)[0]

    sound = (rms >= thresh).astype(np.float32)
    cs = np.cumsum(sound)
    lo = float(np.percentile(cs, 0.1))
    hi = float(np.percentile(cs, 99.9))

    i0 = int(np.searchsorted(cs, lo, side="left"))
    i1 = int(np.searchsorted(cs, hi, side="left"))
    s0 = i0 * hop_length
    s1 = i1 * hop_length
    s0 = max(0, min(s0, len(audio) - 1))
    s1 = max(s0 + 1, min(s1, len(audio)))

    return float(s0 / sr), float(s1 / sr)

def bpm_dynamic_phase_sync(
    sr: int,
    hop_length: int,
    save_hop: int,
    win_s, step_s,
    audio: np.ndarray,
    audio_lp: np.ndarray,
    bpm_bounds: tuple[float, float] = (128.0, 260.0),
    fine_phase_bins: int = 128,
) -> dict:
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        raise ValueError("audio must be 1-D np.ndarray (mono)")

    hop_for_odf = int(hop_length)
    print(f"[seg] computing ODF via librosa.onset.onset_strength (hop={hop_for_odf})")
    odf, hop_t = _compute_odf(audio, sr, hop_for_odf)
    odf_lp, _ =  _compute_odf(audio_lp, sr, hop_for_odf)
    
    seg_frames, boundaries = _coarse_segment_track(odf, sr, hop_for_odf, hop_t, win_s, step_s, seg_min_s = win_s/2)

    tempo_segments, beats_all, seg_meta, first_beat = _estimate_tempo_segments(
        seg_frames,
        odf,
        odf_lp,
        sr,
        hop_for_odf,
        bpm_bounds,
        fine_phase_bins,
        hop_t,
    )

    beats_time = np.asarray(sorted(beats_all), dtype=float)
    beats_frames = np.clip(
        np.round(beats_time / hop_t).astype(np.int32), 0, max(0, len(odf) - 1)
    )

    track_dur = len(audio) / sr if len(audio) else 0.0
    if tempo_segments:
        tempo_segments[-1].end = float(track_dur)
    
    validrange = get_track_validrange(audio, sr, 1, 0.02)
    print(f"[fine] Valid track range = {validrange}")

    refined_segments, refined_beats = refine_segments_via_beatgrid(
        tempo_segments, odf, hop_t, win_s, step_s, validrange
    )
    if refined_segments:
        tempo_segments = refined_segments
        seg_meta = [(seg.start, seg.end, seg.bpm) for seg in tempo_segments]
        if tempo_segments:
            tempo_segments[-1].end = float(track_dur)
    if refined_beats.size:
        beats_time = refined_beats.astype(float)
        beats_frames = np.clip(
            np.round(beats_time / hop_t).astype(np.int32), 0, max(0, len(odf) - 1)
        )

    tempo_segment_dicts = [seg.to_dict() for seg in tempo_segments]

    # Tempo stats
    dur_sec = len(audio) / float(sr)
    if beats_time.size >= 2:
        dt = np.diff(beats_time)
        inst_bpm = 60.0 / np.maximum(dt, 1e-6)
        tempo_global = float(mode(inst_bpm).mode)
        tempo_bt = float(np.mean(inst_bpm))
    else:
        inst_bpm = np.array([], dtype=float)
        tempo_global = float("nan")
        tempo_bt = float("nan")
    
    score = _score_beats_against_odf(beats_time, odf, hop_t)

    return {
        "onset_env": odf.astype(float, copy=True),
        "beats": beats_frames,
        "beats_time": beats_time,
        "downbeats_sec": first_beat,
        "tempo_global": float(tempo_global),
        "tempo_bt": float(tempo_bt),
        "segments": seg_meta,           # [(t0, t1, bpm_est), ...]
        "tempo_segments": tempo_segment_dicts,
        "boundaries": boundaries,       # frame indices on ODF grid
        "odf": odf,
        "hop_t": hop_t,
        "score": score
    }

def bpm_phase_sync(
    sr: int,
    hop_length: int,
    save_hop: int,
    win_s, step_s,
    audio: np.ndarray,
    audio_lp: np.ndarray,
    bpm_bounds: tuple[float, float] = (128.0, 260.0),
    fine_phase_bins: int = 128,
) -> dict:
    if not isinstance(audio, np.ndarray) or audio.ndim != 1:
        raise ValueError("audio must be 1-D np.ndarray (mono)")

    hop_for_odf = int(hop_length)
    print(f"[seg] computing ODF via librosa.onset.onset_strength (hop={hop_for_odf})")
    odf, hop_t = _compute_odf(audio, sr, hop_for_odf)
    odf_lp, _ =  _compute_odf(audio_lp, sr, hop_for_odf)

    seg_frames = [(0, len(odf) - 1)]
    boundaries = np.array([], dtype=np.int32)

    tempo_segments, beats_all, seg_meta, first_beat = _estimate_tempo_segments(
        seg_frames,
        odf,
        odf_lp,
        sr,
        hop_for_odf,
        bpm_bounds,
        fine_phase_bins,
        hop_t,
    )

    beats_time = np.asarray(sorted(beats_all), dtype=float)
    beats_frames = np.clip(
        np.round(beats_time / hop_t).astype(np.int32), 0, max(0, len(odf) - 1)
    )

    track_dur = len(audio) / sr if len(audio) else 0.0
    if tempo_segments:
        tempo_segments[-1].end = float(track_dur)

    tempo_segment_dicts = [seg.to_dict() for seg in tempo_segments]

    if beats_time.size >= 2:
        dt = np.diff(beats_time)
        inst_bpm = 60.0 / np.maximum(dt, 1e-6)
        tempo_global = float(mode(inst_bpm).mode)
        tempo_bt = float(np.mean(inst_bpm))
    else:
        inst_bpm = np.array([], dtype=float)
        tempo_global = float("nan")
        tempo_bt = float("nan")
    
    score = _score_beats_against_odf(beats_time, odf, hop_t)

    return {
        "onset_env": odf.astype(float, copy=True),
        "beats": beats_frames,
        "beats_time": beats_time,
        "downbeats_sec": first_beat,
        "tempo_global": float(tempo_global),
        "tempo_bt": float(tempo_bt),
        "segments": seg_meta,
        "tempo_segments": tempo_segment_dicts,
        "boundaries": boundaries,
        "odf": odf,
        "hop_t": hop_t,
        "score": score
    }