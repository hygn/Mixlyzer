from typing import Literal, Callable, Optional, Tuple
import numpy as np
import librosa
from scipy.signal import filtfilt

from analyzer_core.global_analyzer import fast_load, _butter_bandpass
from analyzer_core.beat.beat import (
    build_grid_from_period_phase,
    _compute_odf,
    estimate_bpm_and_grid,
)
from core.config import config
import math

def canonical_inizio(inz: float, start: float, bpm: float) -> float:
        """Wrap inizio within one bar of the provided start."""
        if not np.isfinite(inz):
            return float(start)
        if not (np.isfinite(bpm) and bpm > 0):
            return float(inz)
        period = 60.0 / float(bpm)
        if not np.isfinite(period) or period <= 0:
            return float(inz)
        bar = 4.0 * period
        if bar <= 0:
            return float(inz)
        rel = (inz - start) % bar
        return float(start + rel)


def shift_grid_in_seg(beatgrid_sec: np.ndarray, segments, segment_index: int, offset: float):
    _beatgrid = np.asarray(beatgrid_sec, copy=True)
    _segments = np.asarray(segments, copy=True)
    st, ed, bpm, inz = _segments[segment_index]
    assert 60/bpm > abs(offset)
    # Indices delimiting [st, ed) in the current grid
    sti = int(np.searchsorted(_beatgrid, st, side="left"))
    edi = int(np.searchsorted(_beatgrid, ed, side="left"))
    sti = max(0, min(sti, len(_beatgrid)))
    edi = max(0, min(edi, len(_beatgrid)))

    # Update inizio in absolute track time; fall back to segment start if invalid
    T = 60.0 / bpm if bpm else np.inf
    cur_inz = _segments[segment_index, 3]
    if not np.isfinite(cur_inz):
        cur_inz = float(st)
    new_inz = canonical_inizio(float(cur_inz) + float(offset), st, bpm)
    _segments[segment_index, 3] = new_inz

    # Rebuild beats inside [st, ed) using build_grid_from_period_phase
    if not np.isfinite(T) or T <= 0 or ed <= st:
        return _beatgrid, _segments

    hop_t = 1e-3  # 1 ms resolution
    total_len_frames = int(np.ceil(ed / hop_t))
    period_frames = T / hop_t
    phase_frames = new_inz / hop_t
    grid_main_sec, _ = build_grid_from_period_phase(
        total_len_frames=total_len_frames,
        hop_t=hop_t,
        period_frames=period_frames,
        phase_frames=phase_frames,
        subdiv=1,
    )
    if grid_main_sec.size:
        mask = (grid_main_sec >= st) & (grid_main_sec < ed)
        new_beats = grid_main_sec[mask]
    else:
        new_beats = np.array([], dtype=float)

    _beatgrid = np.concatenate([_beatgrid[:sti], new_beats, _beatgrid[edi:]]).astype(float)
    _beatgrid = np.sort(_beatgrid)
    return _beatgrid, _segments


def rebuild_grid_from_segments(segments) -> np.ndarray:
    """Rebuild an absolute beatgrid from tempo segments [start, end, bpm, inizio]."""
    try:
        arr = np.asarray(segments, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.asarray([], dtype=float)
    beats = []
    for start, end, bpm, inizio in arr[:, :4]:
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            continue
        if not (np.isfinite(bpm) and bpm > 0):
            continue
        period = 60.0 / bpm
        if not np.isfinite(period) or period <= 0:
            continue
        phase = float(inizio) if np.isfinite(inizio) else float(start)
        beat = float(phase)
        # move beat near start
        if beat < start:
            delta = start - beat
            shift = math.ceil(delta / period)
            beat += shift * period
        else:
            while beat - period >= start:
                beat -= period
            while beat < start:
                beat += period
        end_limit = float(end) + 1e-1
        while beat < end_limit:
            beats.append(beat)
            beat += period
    if not beats:
        return np.asarray([], dtype=float)
    beat_arr = np.sort(np.asarray(beats, dtype=float))
    if beat_arr.size <= 1:
        return beat_arr
    min_gap = 0.05  # seconds
    pruned: list[float] = []
    for beat in beat_arr:
        if pruned and (beat - pruned[-1]) < min_gap:
            continue
        pruned.append(float(beat))
    return np.asarray(pruned, dtype=float)


def reanalyze_segment_from_file(
    path: str,
    cfg: config,
    beats_time_sec,
    segments,
    segment_index: int,
    prev_bpm: Optional[float] = None,
    use_only_prev_bpm: bool = False,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute beats/tempo info for a single segment."""
    if progress_cb:
        progress_cb("Loading audio", 0.0)
    gcf = cfg.analysisconfig
    sr = int(gcf.analysis_samp_rate)
    audio = fast_load(path, sr)
    if progress_cb:
        progress_cb("Preparing audio", 0.10)

    beats_source = beats_time_sec if beats_time_sec is not None else ()
    beats_arr = np.asarray(beats_source, dtype=float)
    seg_arr = np.asarray(segments, dtype=float)
    if seg_arr.ndim != 2 or seg_arr.shape[1] < 4:
        raise ValueError("segments must be (N,4) array")
    if not (0 <= segment_index < len(seg_arr)):
        raise IndexError("segment index out of range")

    start, end, bpm, inz = seg_arr[segment_index, :4]
    total_duration = len(audio) / float(sr) if len(audio) else 0.0
    end = min(end, total_duration)
    start = float(start)
    end = float(end)
    if end - start <= 1e-6:
        raise ValueError("segment duration too small")

    start_sample = int(max(0.0, start) * sr)
    end_sample = int(min(end, total_duration) * sr)
    section = audio[start_sample:end_sample].astype(np.float32, copy=False)
    if section.size < sr * 0.5:
        raise ValueError("segment audio too short for reanalysis")

    use_hpss = getattr(gcf, "use_hpss", False)
    if use_hpss:
        try:
            _, y_perc = librosa.effects.hpss(section)
        except Exception:
            y_perc = section
    else:
        y_perc = section

    b, a = _butter_bandpass(20, 200, sr, order=4)
    lp_y = filtfilt(b, a, y_perc).astype(np.float32)

    hop = int(gcf.bpm_hop_length)
    odf, hop_t = _compute_odf(y_perc, sr, hop)
    odf_lp, _ = _compute_odf(lp_y, sr, hop)

    local = odf
    local_lp = odf_lp
    if progress_cb:
        progress_cb("Estimating BPM", 0.70)
    bpm_lo = float(gcf.bpm_min) * 0.9
    bpm_hi = float(gcf.bpm_max) * 1.1
    prev = None
    if prev_bpm is not None and np.isfinite(prev_bpm) and prev_bpm > 0:
        prev = float(prev_bpm)
        if use_only_prev_bpm:
            bpm_lo = max(prev * 0.8, 0.1)
            bpm_hi = prev * 1.2
        else:
            window = max(0.5, prev * 0.05)
            bpm_lo = max(prev - window, min(bpm_lo, prev))
            bpm_hi = min(prev + window, max(bpm_hi, prev))
            if bpm_hi <= bpm_lo:
                bpm_lo = max(prev - window, 0.1)
                bpm_hi = prev + window
    bpm_est, beats_local, downbeat = estimate_bpm_and_grid(
        local,
        local_lp,
        sr,
        hop,
        bpm_lo=bpm_lo,
        bpm_hi=bpm_hi,
        phase_bins=128,
        prev_bpm=prev,
        use_only_prev_bpm=use_only_prev_bpm if prev is not None else False,
    )
    bpm_est = float(bpm_est) if np.isfinite(bpm_est) else float(bpm)

    beats_local = np.asarray(beats_local, dtype=float)
    if beats_local.size:
        beats_abs = beats_local + start
    else:
        beats_abs = np.array([start], dtype=float)

    if not np.isfinite(downbeat):
        downbeat = beats_local[0] if beats_local.size else start
    new_inizio = float(start + float(downbeat))

    if progress_cb:
        progress_cb("Updating beat grid", 0.70)

    keep_mask = (beats_arr < start - 1e-9) | (beats_arr >= end - 1e-9)
    beats_updated = np.concatenate([beats_arr[keep_mask], beats_abs])
    beats_updated = np.sort(beats_updated.astype(float, copy=False))

    seg_updated = seg_arr.copy()
    seg_updated[segment_index, 0] = start
    seg_updated[segment_index, 1] = end
    seg_updated[segment_index, 2] = bpm_est
    seg_updated[segment_index, 3] = new_inizio

    if progress_cb:
        progress_cb("Segment updated", 0.9)

    return beats_updated, seg_updated
