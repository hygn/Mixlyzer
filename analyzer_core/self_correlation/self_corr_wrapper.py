from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from analyzer_core.self_correlation.self_correlation import (
    beat_centers_to_frames,
    flatten_band_beats,
    mel_beat_matrix,
    mel_frames,
    overlap_map_from_frames,
    refine_offset_from_map,
    spectral_segments_from_overlap,
    ssm_from_features,
    diag_profile_peaks,
)
from analyzer_core.utils import moving_average


@dataclass
class BeatSSMConfig:
    n_beats_seg: int = 4
    n_mels: int = 128
    mel_fmin: float = 30.0
    mel_fmax: Optional[float] = None
    smooth_sigma: float = 1.0
    diag_band: int = 2
    margin_beats: float = 16.0
    prominence: float = 0.09
    max_peaks: int = 6


@dataclass
class OverlapConfig:
    hop: int = 512
    tol_beats: float = 3.0
    anchor_restart: bool = True
    flatten_band_beats: float = 1.0
    cluster_min_beats: int = 8
    spectral_sigma: float = 1.0
    spectral_band_window: int = 0


@dataclass
class LinkFilterConfig:
    min_score: float = -1.0
    min_duration_sec: float = 0.0


@dataclass
class SelfCorrelationV2Config:
    beat_ssm: BeatSSMConfig = BeatSSMConfig
    overlap: OverlapConfig = OverlapConfig
    link_filter: LinkFilterConfig = LinkFilterConfig


@dataclass
class BeatSSMResult:
    ssm: np.ndarray
    lag_beats: np.ndarray
    folded_profile: np.ndarray
    peak_indices: np.ndarray
    avg_beat_sec: float
    step_beats: float


@dataclass
class OverlapAnalysis:
    lag_beats: float
    refined_lag_beats: float
    delta_ref: float
    sim2d: np.ndarray
    deltas: np.ndarray
    valid_idx: np.ndarray
    flattened: np.ndarray
    spectral_segments: list[tuple[int, int]]


@dataclass
class SimilarLink:
    lag_beats: float
    lag_sec: float
    score: float
    a: tuple[float, float]
    b: tuple[float, float]
    a_beat: tuple[float, float]
    b_beat: tuple[float, float]


@dataclass
class SelfCorrelationReport:
    beat_ssm: BeatSSMResult
    overlaps: list[OverlapAnalysis] = field(default_factory=list)
    links: list[SimilarLink] = field(default_factory=list)


def _safe_top_peaks(
    lag_beats: np.ndarray,
    folded: np.ndarray,
    peak_indices: np.ndarray,
    cfg: BeatSSMConfig,
) -> np.ndarray:
    if peak_indices.size:
        return peak_indices
    max_lag = float(lag_beats[-1]) if lag_beats.size else 0.0
    margin = float(max(0.0, cfg.margin_beats))
    valid = (lag_beats >= margin) & (lag_beats <= max_lag - margin)
    idx_pool = np.flatnonzero(valid) if valid.any() else np.arange(lag_beats.size)
    if not idx_pool.size:
        return np.zeros(0, dtype=int)
    vals = folded[idx_pool]
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return np.zeros(0, dtype=int)
    distance = max(1, len(vals) // 50)
    peaks, _ = find_peaks(vals, prominence=float(cfg.prominence), distance=distance, width=1)
    if not peaks.size:
        return np.zeros(0, dtype=int)
    peak_idx = idx_pool[peaks]
    order = np.argsort(folded[peak_idx])[::-1][: int(max(1, cfg.max_peaks))]
    return peak_idx[order]


def _fold_profile(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ks = np.arange(-(S.shape[1] - 1), S.shape[0], dtype=int)
    vals: list[float] = []
    for k in ks:
        diag = np.diag(S, int(k))
        vals.append(float(np.mean(diag)) if diag.size else 0.0)
    prof = np.asarray(vals, dtype=float)
    win = max(5, (len(prof) // 50) * 2 + 1)
    trend = moving_average(prof, win)
    resid = np.clip(prof - trend, 0.0, None)
    if resid.size:
        resid /= max(float(np.max(resid)), 1e-9)

    idx_map = {int(v): i for i, v in enumerate(ks)}
    pos_mask = ks >= 0
    ks_pos = ks[pos_mask]
    folded = np.zeros_like(ks_pos, dtype=float)
    for idx, k in enumerate(ks_pos):
        v_pos = resid[idx_map[int(k)]]
        v_neg = resid[idx_map.get(int(-k), idx_map[int(k)])]
        folded[idx] = 0.5 * (float(v_pos) + float(v_neg))
    return ks_pos.astype(float), folded


def _links_from_segments(
    segments: Iterable[tuple[int, int]],
    valid_idx: np.ndarray,
    beats_time: np.ndarray,
    lag_sec: float,
    lag_beats: float,
    sim_line: np.ndarray,
    cfg: LinkFilterConfig,
    anchor_restart: bool,
) -> list[SimilarLink]:
    links: list[SimilarLink] = []
    for s0, e0 in segments:
        if e0 <= s0:
            continue
        a_bt0 = int(valid_idx[int(s0)])
        a_bt1 = int(valid_idx[int(max(s0, e0 - 1))])
        if a_bt0 >= beats_time.size or a_bt1 >= beats_time.size:
            continue
        a0 = float(beats_time[a_bt0])
        a1 = float(beats_time[a_bt1])
        if anchor_restart:
            b0, b1 = a0 - lag_sec, a1 - lag_sec
            b_bt0, b_bt1 = a_bt0 - lag_beats, a_bt1 - lag_beats
        else:
            b0, b1 = a0 + lag_sec, a1 + lag_sec
            b_bt0, b_bt1 = a_bt0 + lag_beats, a_bt1 + lag_beats
        mean_score = float(np.mean(sim_line[int(s0) : int(e0)]))
        dur = max(0.0, a1 - a0)
        if mean_score < cfg.min_score or dur < cfg.min_duration_sec:
            continue
        links.append(
            SimilarLink(
                lag_beats=lag_beats,
                lag_sec=lag_sec,
                score=mean_score,
                a=(max(0.0, a0), max(0.0, a1)),
                b=(max(0.0, b0), max(0.0, b1)),
                a_beat=(max(0.0, a_bt0), max(0.0, a_bt1)),
                b_beat=(max(0.0, b_bt0), max(0.0, b_bt1)),
            )
        )
    return links


def analyze_self_correlation(
    y_harm: np.ndarray,
    sr: int,
    beats_time: Sequence[float],
    *,
    config: SelfCorrelationV2Config = SelfCorrelationV2Config(),
) -> SelfCorrelationReport:
    beats_time = np.asarray(beats_time, dtype=float)
    if beats_time.size < 2 or y_harm.size < 2:
        return SelfCorrelationReport(
            beat_ssm=BeatSSMResult(
                ssm=np.zeros((0, 0), dtype=float),
                lag_beats=np.zeros(0, dtype=float),
                folded_profile=np.zeros(0, dtype=float),
                peak_indices=np.zeros(0, dtype=int),
                avg_beat_sec=0.5,
                step_beats=1.0,
            ),
            overlaps=[],
            links=[],
        )

    beat_cfg = config.beat_ssm
    overlap_cfg = config.overlap
    link_cfg = config.link_filter

    Xmat, step_beats, avg_beat_sec = mel_beat_matrix(
        y_harm,
        beats_time,
        sr,
        n_mels=beat_cfg.n_mels,
        mel_fmin=beat_cfg.mel_fmin,
        mel_fmax=beat_cfg.mel_fmax,
        n_beats_seg=beat_cfg.n_beats_seg,
    )
    S = ssm_from_features(Xmat, smooth_sigma=beat_cfg.smooth_sigma, diag_band=beat_cfg.diag_band)
    lag_beats, folded_profile, peak_indices = diag_profile_peaks(
        S,
        step_beats=step_beats,
        margin_beats=beat_cfg.margin_beats,
        prominence=beat_cfg.prominence,
    )
    if lag_beats.size == 0:
        ks_pos, folded = _fold_profile(S)
        lag_beats = ks_pos * step_beats
        folded_profile = folded

    peak_indices = _safe_top_peaks(lag_beats, folded_profile, peak_indices, beat_cfg)
    beat_ssm_res = BeatSSMResult(
        ssm=S,
        lag_beats=lag_beats,
        folded_profile=folded_profile,
        peak_indices=peak_indices,
        avg_beat_sec=avg_beat_sec,
        step_beats=step_beats,
    )

    if not peak_indices.size:
        return SelfCorrelationReport(beat_ssm=beat_ssm_res, overlaps=[], links=[])

    Feat, hop_sec = mel_frames(
        y_harm,
        sr,
        n_fft=int(max(256, overlap_cfg.hop)),
        hop=int(max(1, overlap_cfg.hop)),
        n_mels=beat_cfg.n_mels,
        mel_fmin=beat_cfg.mel_fmin,
        mel_fmax=beat_cfg.mel_fmax,
    )
    beat_frames = beat_centers_to_frames(beats_time, sr, int(max(1, overlap_cfg.hop)))
    #centers = (0.5 * (beats_time[:-1] + beats_time[1:])) if beats_time.size >= 2 else np.asarray([], dtype=float)

    overlaps: list[OverlapAnalysis] = []
    links: list[SimilarLink] = []

    for pk_idx in peak_indices:
        lag_val = float(lag_beats[int(pk_idx)])
        sim2d, deltas, valid_idx = overlap_map_from_frames(
            Feat,
            beat_frames,
            lag_val,
            avg_beat_sec,
            hop_sec,
            tol_beats=overlap_cfg.tol_beats,
            anchor_restart=overlap_cfg.anchor_restart,
            use_beat_index_mapping=True,
        )
        if sim2d.size == 0 or valid_idx.size == 0:
            continue

        r_ref, delta_ref = refine_offset_from_map(sim2d, deltas)
        refined_lag_beats = (lag_val - delta_ref) if overlap_cfg.anchor_restart else (lag_val + delta_ref)
        lag_sec = refined_lag_beats * avg_beat_sec
        flattened = flatten_band_beats(sim2d, r_ref, band_beats=overlap_cfg.flatten_band_beats)

        try:
            segs, _ = spectral_segments_from_overlap(
                sim2d,
                smooth_sigma=overlap_cfg.spectral_sigma,
                band_window=overlap_cfg.spectral_band_window,
                min_len=max(2, int(round(overlap_cfg.cluster_min_beats))),
            )
        except Exception:
            segs = []

        overlaps.append(
            OverlapAnalysis(
                lag_beats=lag_val,
                refined_lag_beats=refined_lag_beats,
                delta_ref=float(delta_ref),
                sim2d=sim2d,
                deltas=deltas,
                valid_idx=valid_idx,
                flattened=flattened,
                spectral_segments=list(segs),
            )
        )

        if not segs.size if isinstance(segs, np.ndarray) else not segs:
            continue
        link_entries = _links_from_segments(
            segs,
            valid_idx,
            beats_time,
            lag_sec,
            lag_val,
            flattened,
            link_cfg,
            overlap_cfg.anchor_restart,
        )
        links.extend(link_entries)

    links.sort(key=lambda l: l.score, reverse=True)
    return SelfCorrelationReport(beat_ssm=beat_ssm_res, overlaps=overlaps, links=links), Xmat

def _beat_time(beats: np.ndarray, idx: int) -> float:
    if beats.size == 0:
        return 0.0
    if idx <= 0:
        return float(beats[0])
    if idx >= beats.size:
        if beats.size == 1:
            return float(beats[0])
        gap = beats[-1] - beats[-2]
        extra = idx - (beats.size - 1)
        return float(beats[-1] + gap * extra)
    return float(beats[int(idx)])

def refine_overlap_offset(xmat: np.ndarray, link:SimilarLink, beats_time, tol_beats:int = 2):
    """Refine overlap part offset and snap to beat using beat-synced mel matrix"""
    if xmat.ndim != 2 or xmat.shape[1] == 0:
        return link, 0

    beats = np.asarray(beats_time, dtype=float)
    cols = xmat.shape[1]
    max_col = max(0, cols - 1)

    a0 = int(np.clip(round(link.a_beat[0]), 0, max_col))
    a1 = int(np.clip(round(link.a_beat[1]), a0 + 1, cols))
    bd = max(1, a1 - a0)
    b0_base = int(np.clip(round(link.b_beat[0]), 0, max_col))
    a = xmat[:, a0 : a0 + bd]
    if a.shape[1] != bd:
        return link, 0

    best_offset = None
    best_score = -np.inf
    for offset in range(-tol_beats, tol_beats + 1):
        start = b0_base + offset
        end = start + bd
        if start < 0 or end > cols:
            continue
        b = xmat[:, start:end]
        a_cols = a.T
        b_cols = b.T
        numerators = np.einsum("ij,ij->i", a_cols, b_cols)
        norms = np.linalg.norm(a_cols, axis=1) * np.linalg.norm(b_cols, axis=1)
        similarities = np.divide(
            numerators,
            norms,
            out=np.full_like(numerators, -np.inf, dtype=float),
            where=norms > 0,
        )
        score = np.mean(similarities)
        if score > best_score:
            best_score = score
            best_offset = offset
    if best_offset is None:
        return link, 0

    b0 = b0_base + best_offset
    if b0 < 0:
        b0 = 0
    if b0 + bd > cols:
        b0 = cols - bd
    b1 = b0 + bd

    a_start = _beat_time(beats, a0)
    a_end = _beat_time(beats, a0 + bd)
    b_start = _beat_time(beats, b0)
    b_end = _beat_time(beats, b0 + bd)
    new_lag_beats = float(a0 - b0)
    new_lag_sec = float(a_start - b_start)
    clamped_score = float(np.clip(best_score, -1.0, 1.0))
    return (SimilarLink(
        lag_beats=new_lag_beats,
        lag_sec=new_lag_sec,
        score=clamped_score,
        a=(max(0.0, a_start), max(0.0, a_end)),
        b=(max(0.0, b_start), max(0.0, b_end)),
        a_beat=(max(0.0, float(a0)), max(0.0, float(a0 + bd))),
        b_beat=(max(0.0, float(b0)), max(0.0, float(b0 + bd))),
    ), best_offset)

def refine_overlap(xmat: np.ndarray, link:SimilarLink, beats_time, tol_beats:int = 2):
    """sweep position of only **a** in link and calling refine_overlap_offset multiple times to find best possible offsets for both a and b"""
    beats = np.asarray(beats_time, dtype=float)
    base_link, _ = refine_overlap_offset(xmat, link, beats, tol_beats)
    best_link = base_link
    best_score = float(base_link.score)

    tol = int(round(float(tol_beats)))
    if tol <= 0:
        return best_link

    cols = xmat.shape[1]
    max_col = max(0, cols - 1)
    a0 = int(np.clip(round(base_link.a_beat[0]), 0, max_col))
    a1 = int(np.clip(round(base_link.a_beat[1]), a0 + 1, cols))
    seg_len = max(1, a1 - a0)

    lag_baseline = float(base_link.lag_beats)
    diff_beats = float(base_link.a_beat[0]) - float(base_link.b_beat[0])
    anchor_sign = 1.0 if abs(diff_beats - lag_baseline) <= abs(-diff_beats - lag_baseline) else -1.0
    base_a_time = _beat_time(beats, a0)
    lag_sec_base = float(base_link.lag_sec)

    for shift in range(-tol, tol + 1):
        if shift == 0:
            continue
        start = a0 + shift
        end = start + seg_len
        if start < 0 or end > cols:
            continue
        delta_time = _beat_time(beats, start) - base_a_time
        candidate = SimilarLink(
            lag_beats=lag_baseline + anchor_sign * shift,
            lag_sec=lag_sec_base + anchor_sign * delta_time,
            score=best_link.score,
            a=(_beat_time(beats, start), _beat_time(beats, start + seg_len)),
            b=base_link.b,
            a_beat=(float(start), float(end)),
            b_beat=base_link.b_beat,
        )
        refined, bo_b = refine_overlap_offset(xmat, candidate, beats, tol_beats)
        cand_score = float(refined.score)
        if cand_score > best_score:
            best_score = cand_score
            best_link = refined

    return best_link

def get_best_jump_offset(xmat: np.ndarray, link:SimilarLink, beats_time):
    if xmat.ndim != 2 or xmat.shape[1] == 0:
        return link, 0

    beats = np.asarray(beats_time, dtype=float)
    cols = xmat.shape[1]
    max_col = max(0, cols - 1)

    a0 = int(np.clip(round(link.a_beat[0]), 0, max_col))
    a1 = int(np.clip(round(link.a_beat[1]), a0 + 1, cols))
    bd = max(1, a1 - a0)
    b0_base = int(np.clip(round(link.b_beat[0]), 0, max_col))
    a = xmat[:, a0 : a0 + bd]
    if a.shape[1] != bd:
        return link, 0
    start = b0_base
    end = start + bd
    if end > cols:
        end = cols
        start = max(0, end - bd)
    b = xmat[:, start:end]
    if b.shape[1] != bd:
        return link, 0
    a_cols = a.T
    b_cols = b.T
    numerators = np.einsum("ij,ij->i", a_cols, b_cols)
    norms = np.linalg.norm(a_cols, axis=1) * np.linalg.norm(b_cols, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        similarities = np.divide(
            numerators,
            norms,
            out=np.full_like(numerators, -np.inf, dtype=float),
            where=norms > 0,
        )
    if not similarities.size or not np.isfinite(similarities).any():
        return link, 0
    best_jump_offset = int(np.argmax(similarities))
    best_jump_offset = int(np.clip(best_jump_offset, 0, bd - 1))
    return _beat_time(beats, a0 + best_jump_offset), _beat_time(beats, start + best_jump_offset)
