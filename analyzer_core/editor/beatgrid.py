from typing import Callable, Optional, Tuple
import numpy as np

from analyzer_core.global_analyzer import fast_load
from foundation.tempo_rows import (
    bpm_at,
    is_ramp_row,
    prune_close_beats,
    ramp_beat_times,
    segment_end_bpm,
    shift_row_beats,
    with_end_bpm,
)
from analyzer_core.beat.frame_features import extract_frame_features
from analyzer_core.hpss import hpss_audio
from core.config import config
from foundation.i18n import tr


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


PAD_SEC = 4.0      # music on both sides of the segment the frame TCN hears (its context)


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
    """One segment read again with the full analysis's models (the three frame-TCN trainings, averaged):
    the steady grid with the most evidence in its BPM octave (with a reference BPM: within 20 % of it),
    or a linear tempo ramp when that explains the onsets better (guesses: the neighbours' tempos, the
    tempos of the segment's two halves); then its bars (one bar length and phase) for the downbeat."""
    from analyzer_core.beat.general import bars as B, decode as D, gradual as G, load_models, tcn as T

    if progress_cb:
        progress_cb(tr("Loading audio"), 0.0)
    sr = T.SAMPLE_RATE
    audio = fast_load(path, sr)
    seg_arr = with_end_bpm(segments)
    if seg_arr.shape[0] == 0:
        raise ValueError("no tempo segments")
    if not (0 <= segment_index < len(seg_arr)):
        raise IndexError("segment index out of range")
    total = len(audio) / float(sr) if len(audio) else 0.0
    start = float(seg_arr[segment_index, 0])
    end = min(float(seg_arr[segment_index, 1]), total)
    if end - start < 2.0:
        raise ValueError(tr("segment too short for reanalysis"))

    a = max(0.0, start - PAD_SEC)
    b = min(total, end + PAD_SEC)
    section = np.ascontiguousarray(audio[int(a * sr):int(b * sr)], dtype=np.float32)
    if progress_cb:
        progress_cb(tr("Preparing audio"), 0.10)
    # the models were trained on frames of the HPSS parts
    y_harm, y_perc = hpss_audio(section)
    frames = extract_frame_features(section, sr, y_perc, y_harm, hop_length=T.FRAME_HOP_LENGTH)
    if progress_cb:
        progress_cb(tr("Beat activations"), 0.40)
    gcf = cfg.analysisconfig
    models = load_models(
        str(getattr(gcf, "downbeat_ctx_parameter_path", "") or "").strip() or None,
        str(getattr(gcf, "beat_tcn_parameter_path", "") or "").strip() or None,
        str(getattr(gcf, "beat_critic_parameter_path", "") or "").strip() or None,
        str(getattr(gcf, "beat_decoder_parameter_path", "") or "").strip() or None,
    )
    x = T.frame_inputs(frames)
    acts = [T.activations(P, x) for P in models.tcn]
    beat = np.mean([act for act, _ in acts], axis=0)
    down = np.mean([d for _, d in acts], axis=0)
    ft = np.asarray(frames.frame_times, dtype=float)
    hop = float(np.median(np.diff(ft))) if ft.size > 1 else T.FRAME_HOP_LENGTH / sr
    t0 = a + float(ft[0])
    val, fine = D.readers(beat, hop, t0)

    if progress_cb:
        progress_cb(tr("Estimating BPM"), 0.70)
    # the segment's own frames for its tempo candidates and the steady grid over it
    i0 = max(0, int(np.ceil((start - t0) / hop)))
    i1 = min(beat.size, int(np.floor((end - t0) / hop)) + 1)
    act_seg = beat[i0:i1]
    off = (t0 + i0 * hop - start) - 0.5 * hop
    if prev_bpm is not None and np.isfinite(prev_bpm) and prev_bpm > 0:
        ref = float(prev_bpm)
        near = D.Octave(ref / np.sqrt(2.0))
        found = D.coarse_tempos(act_seg, hop, near)
        tempos = np.unique(np.r_[ref, found[(found >= 0.8 * ref) & (found <= 1.2 * ref)]])
    else:
        ref = None
        octave = D.Octave(float(cfg.analysisconfig.bpm_min))
        tempos = D.with_ratios(D.coarse_tempos(act_seg, hop, octave), octave)
    row = D.single_grid(act_seg, hop, off, end - start, tempos)[0]
    row[0], row[1], row[3] = start, end, row[3] + start
    row[4] = seg_arr[segment_index, 4]
    row = D.final_refit(D.final_refit(row[None, :], val), fine)[0]

    if ref is None:
        # a ramp? from the neighbours' tempos, and from the tempos of the segment's halves
        guesses = []
        if segment_index > 0 and segment_index + 1 < len(seg_arr):
            guesses.append((segment_end_bpm(seg_arr[segment_index - 1]), float(seg_arr[segment_index + 1, 2])))
        mid = 0.5 * (start + end)
        if mid - start >= 2.0:
            halves = []
            for lo, hi in ((start, mid), (mid, end)):
                j0 = max(0, int(np.ceil((lo - t0) / hop)))
                j1 = min(beat.size, int(np.floor((hi - t0) / hop)) + 1)
                own = D.with_ratios(D.coarse_tempos(beat[j0:j1], hop, octave), octave)
                part = D.single_grid(beat[j0:j1], hop, (t0 + j0 * hop - lo) - 0.5 * hop, hi - lo,
                                     np.unique(np.r_[own, tempos, row[2]]))[0]
                halves.append(float(part[2]))
            # a linear ramp has its halves' tempos at a quarter and three quarters of the way
            h0, h1 = halves
            guesses.append((h0 - 0.5 * (h1 - h0), h1 + 0.5 * (h1 - h0)))
        row = G.ramp_or_steady(row, guesses, fine)

    if progress_cb:
        progress_cb(tr("Downbeat analysis"), 0.80)
    seg_beats = G.row_beats(row)
    seg_beats = seg_beats[(seg_beats >= start - 1e-9) & (seg_beats < end)]
    if seg_beats.size >= 2:
        # the bars on the excerpt's own clock (its frames start at a)
        dbs = B.downbeats(seg_beats - a, row[None, :], down, t0 - a, hop, frames, models.ctx, restarts=False) + a
        if dbs.size:
            if dbs.size >= 2:
                per_bar = np.diff(np.searchsorted(seg_beats, dbs - 1e-6))
                row[4] = float(np.median(per_bar))
            row[3] = float(dbs[0])

    if progress_cb:
        progress_cb(tr("Updating beat grid"), 0.90)
    beats_arr = np.asarray(beats_time_sec if beats_time_sec is not None else (), dtype=float)
    keep_mask = (beats_arr < start - 1e-9) | (beats_arr >= end - 1e-9)
    beats_updated = np.sort(np.concatenate([beats_arr[keep_mask], seg_beats]))
    seg_updated = seg_arr.copy()
    seg_updated[segment_index] = row
    print(f"[SegRA] {start:.3f}-{end:.3f}s bpm={row[2]:.2f}"
          + (f"->{row[5]:.2f}" if abs(row[5] - row[2]) > 1e-6 else "")
          + f" inizio={row[3]:.3f}s ts={int(row[4])}/4")
    if progress_cb:
        progress_cb(tr("Segment updated"), 0.95)
    return beats_updated, seg_updated
