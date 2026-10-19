from typing import Callable, Optional, Tuple
import numpy as np

from analyzer_core.global_analyzer import fast_load
from analyzer_core.beat.beat import (
    build_grid_from_period_phase,
    estimate_bpm_and_grid,
    prune_close_beats,
    rebuild_grid_from_segments,
)
from analyzer_core.beat.downbeat_offset import DOWNBEAT_BOOST_PATH, detect_downbeat_offset_segments, usable_boost_path
from core.tempo_rows import bpm_at, is_ramp_row, ramp_beat_times, segment_end_bpm, shift_row_beats
from analyzer_core.beat.frame_features import FRAME_HOP_LENGTH, extract_frame_features
from analyzer_core.beat.learned_onset import compute_beat_odf
from analyzer_core.hpss import hpss_audio
from core.config import config

def canonical_inizio(inz: float, start: float, bpm: float, beats_per_bar: float = 4.0) -> float:
        """Wrap inizio within one bar (``beats_per_bar`` beats) of the provided start.

        Downbeats fall every ``ts_num`` beats from inizio, so the bar must be the
        segment's own; wrapping a 3/4 segment by 4 beats would move its downbeat.
        """
        if not np.isfinite(inz):
            return float(start)
        if not (np.isfinite(bpm) and bpm > 0):
            return float(inz)
        period = 60.0 / float(bpm)
        if not np.isfinite(period) or period <= 0:
            return float(inz)
        meter = float(beats_per_bar) if np.isfinite(beats_per_bar) and beats_per_bar >= 1 else 4.0
        bar = round(meter) * period
        if bar <= 0:
            return float(inz)
        rel = (inz - start) % bar
        return float(start + rel)


def shift_grid_in_seg(beatgrid_sec: np.ndarray, segments, segment_index: int, offset: float):
    """Move one segment's beats by ``offset`` seconds (less than a beat). A tempo ramp moves
    along its own tempo line by the beats ``offset`` is at the tempo of its downbeat."""
    _beatgrid = np.asarray(beatgrid_sec, copy=True)
    _segments = np.asarray(segments, copy=True)
    if _segments.ndim == 2 and _segments.shape[1] > 5 and is_ramp_row(_segments[segment_index]):
        return _shift_ramp_in_seg(_beatgrid, _segments, segment_index, offset)
    st, ed, bpm, inz = _segments[segment_index][:4]
    assert 60/bpm > abs(offset)
    # Same span rebuild_grid_from_segments gives a segment: [st, ed) plus a hair
    # (the last segment keeps a beat sitting exactly on the track end).
    valid_ends = _segments[:, 1][np.isfinite(_segments[:, 1])]
    track_end = float(np.max(valid_ends)) if valid_ends.size else float("inf")
    end_limit = float(ed) + (1e-3 if float(ed) < track_end else 1e-6)
    # Indices delimiting that span in the current grid; a beat within the hair
    # past the previous segment's end is that segment's (as in the rebuild)
    sti = int(np.searchsorted(_beatgrid, st + (1e-3 if segment_index > 0 else 0.0), side="left"))
    edi = int(np.searchsorted(_beatgrid, end_limit, side="left"))
    sti = max(0, min(sti, len(_beatgrid)))
    edi = max(0, min(edi, len(_beatgrid)))

    # Update inizio in absolute track time; fall back to segment start if invalid
    T = 60.0 / bpm if bpm else np.inf
    cur_inz = _segments[segment_index, 3]
    if not np.isfinite(cur_inz):
        cur_inz = float(st)
    ts_num = float(_segments[segment_index, 4]) if _segments.shape[1] >= 5 else 4.0
    new_inz = canonical_inizio(float(cur_inz) + float(offset), st, bpm, ts_num)
    _segments[segment_index, 3] = new_inz

    # Rebuild beats inside [st, ed) using build_grid_from_period_phase
    if not np.isfinite(T) or T <= 0 or ed <= st:
        return _beatgrid, _segments

    hop_t = 1e-3  # 1 ms resolution
    total_len_frames = int(np.ceil(end_limit / hop_t))
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
        mask = (grid_main_sec >= st) & (grid_main_sec < end_limit)
        new_beats = grid_main_sec[mask]
    else:
        new_beats = np.array([], dtype=float)

    _beatgrid = prune_close_beats(np.concatenate([_beatgrid[:sti], new_beats, _beatgrid[edi:]]))
    return _beatgrid, _segments


def _shift_ramp_in_seg(beatgrid: np.ndarray, segments: np.ndarray, segment_index: int, offset: float):
    row = segments[segment_index]
    st, ed, inz = float(row[0]), float(row[1]), float(row[3])
    if not np.isfinite(inz):
        inz = st
    beats = float(offset) * bpm_at(row, inz) / 60.0
    segments[segment_index, 3] = shift_row_beats(row, beats)
    valid_ends = segments[:, 1][np.isfinite(segments[:, 1])]
    track_end = float(np.max(valid_ends)) if valid_ends.size else float("inf")
    end_limit = ed + (1e-3 if ed < track_end else 1e-6)
    sti = int(np.searchsorted(beatgrid, st + (1e-3 if segment_index > 0 else 0.0), side="left"))
    edi = int(np.searchsorted(beatgrid, end_limit, side="left"))
    row = segments[segment_index]
    new_beats = ramp_beat_times(st, ed, float(row[2]), segment_end_bpm(row), float(row[3]), end_limit)
    return prune_close_beats(np.concatenate([beatgrid[:sti], new_beats, beatgrid[edi:]])), segments


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
            y_harm, y_perc = hpss_audio(section)
        except Exception:
            y_harm = y_perc = section
    else:
        y_harm = y_perc = section

    hop = FRAME_HOP_LENGTH
    # Analyzed once for this segment: the shipped learned onset and downbeat models both use them.
    frames = extract_frame_features(section, sr, y_perc, y_harm, hop_length=hop)
    odf, hop_t = compute_beat_odf("optimized", frames, y_perc, sr, hop)

    local = odf
    if progress_cb:
        progress_cb("Estimating BPM", 0.70)
    # the octave the beat tracker reads tempos in
    bpm_lo = float(gcf.bpm_min)
    bpm_hi = 2.0 * bpm_lo
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
        sr,
        hop,
        bpm_lo=bpm_lo,
        bpm_hi=bpm_hi,
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

    # Global Downbeat Analysis for THIS segment only: place the bar-start
    # (inizio) on the detected downbeat, snapped to an actual beat.
    if beats_local.size >= 2:
        try:
            if progress_cb:
                progress_cb("Downbeat analysis", 0.78)
            db_segments = detect_downbeat_offset_segments(
                np.asarray(section, dtype=np.float32),
                int(sr),
                beats_local,
                method="global",
                frames=frames,
                boost_path=usable_boost_path(DOWNBEAT_BOOST_PATH),
            )
            db_time = db_segments[0].first_downbeat_time_sec if db_segments else None
            if db_time is not None and np.isfinite(db_time):
                snapped = float(beats_local[int(np.argmin(np.abs(beats_local - float(db_time))))])
                new_inizio = float(start + snapped)
                print(f"[SegRA] global downbeat (local) => {snapped:.3f}s")
        except Exception as exc:
            print(f"[SegRA] downbeat analysis skipped: {exc}")

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
    if seg_updated.shape[1] > 5:
        seg_updated[segment_index, 5] = bpm_est   # reanalysis gives a steady tempo
    if progress_cb:
        progress_cb("Segment updated", 0.9)

    return beats_updated, seg_updated
