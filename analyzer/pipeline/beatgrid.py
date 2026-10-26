"""The editor's tempo segment reanalysis: one segment read again with the full analysis's beat models."""
from typing import Optional, Tuple

import numpy as np

from analyzer.audio.hpss import hpss_audio
from analyzer.audio.load import fast_load
from analyzer.beat.frame_features import extract_frame_features
from analyzer.errors import SEGMENT_TOO_SHORT, AnalysisError
from analyzer.music.tempo_rows import segment_end_bpm, with_end_bpm
from analyzer.pipeline import stages
from analyzer.pipeline.stages import Progress, report
from analyzer.settings import AnalysisSettings

PAD_SEC = 4.0      # music on both sides of the segment the frame TCN hears (its context)


def reanalyze_segment(
    path: str,
    settings: AnalysisSettings,
    beats_time_sec,
    segments,
    segment_index: int,
    prev_bpm: Optional[float] = None,
    progress: Progress = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """One segment read again with the full analysis's models (the three frame-TCN trainings, averaged):
    the steady grid with the most evidence in its BPM octave (with a reference BPM: within 20 % of it),
    or a linear tempo ramp when that explains the onsets better (guesses: the neighbours' tempos, the
    tempos of the segment's two halves); then its bars (one bar length and phase) for the downbeat."""
    from analyzer.beat.general import bars as B, decode as D, gradual as G, load_models, tcn as T

    report(progress, stages.LOAD, 0.0)
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
        raise AnalysisError(SEGMENT_TOO_SHORT)

    a = max(0.0, start - PAD_SEC)
    b = min(total, end + PAD_SEC)
    section = np.ascontiguousarray(audio[int(a * sr):int(b * sr)], dtype=np.float32)
    report(progress, stages.HPSS, 0.10)
    # the models were trained on frames of the HPSS parts
    y_harm, y_perc = hpss_audio(section)
    frames = extract_frame_features(section, sr, y_perc, y_harm, hop_length=T.FRAME_HOP_LENGTH)
    report(progress, stages.BEAT_ACTIVATIONS, 0.40)
    paths = settings.models
    models = load_models(paths.downbeat_ctx, paths.beat_tcn, paths.beat_critic, paths.beat_decoder)
    x = T.frame_inputs(frames)
    acts = [T.activations(P, x) for P in models.tcn]
    beat = np.mean([act for act, _ in acts], axis=0)
    down = np.mean([d for _, d in acts], axis=0)
    ft = np.asarray(frames.frame_times, dtype=float)
    hop = float(np.median(np.diff(ft))) if ft.size > 1 else T.FRAME_HOP_LENGTH / sr
    t0 = a + float(ft[0])
    val, fine = D.readers(beat, hop, t0)

    report(progress, stages.TEMPO_ESTIMATE, 0.70)
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
        octave = D.Octave(float(settings.bpm_min))
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

    report(progress, stages.DOWNBEATS, 0.80)
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

    report(progress, stages.GRID_UPDATE, 0.90)
    beats_arr = np.asarray(beats_time_sec if beats_time_sec is not None else (), dtype=float)
    keep_mask = (beats_arr < start - 1e-9) | (beats_arr >= end - 1e-9)
    beats_updated = np.sort(np.concatenate([beats_arr[keep_mask], seg_beats]))
    seg_updated = seg_arr.copy()
    seg_updated[segment_index] = row
    print(f"[SegRA] {start:.3f}-{end:.3f}s bpm={row[2]:.2f}"
          + (f"->{row[5]:.2f}" if abs(row[5] - row[2]) > 1e-6 else "")
          + f" inizio={row[3]:.3f}s ts={int(row[4])}/4")
    return beats_updated, seg_updated
