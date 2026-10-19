"""The generalized beat tracker: one pipeline from frame features to tempo rows, beats and downbeats.

Per training seed of the frame TCN (three equally good trainings):
  activations -> tempo candidates from their periodicity -> section decoding (one cost J) -> same-tempo
  phase judgement -> exact fit (pooled, then fine evidence) -> sudden or gradual tempo changes (ramps) ->
  bar-supported beat phase -> metronome critic (tempo-ratio readings) -> tempo-change boundaries on the
  local onset (refine_boundaries) -> slow tail held -> drifting sections -> tiny re-syncs merged ->
  junction rules -> beats -> bars from two downbeat observers.
Then the consensus: of the three seeds' grids per track, the one that agrees most with the others (the
pipeline's discrete choices flip with small changes of the activation; the medoid keeps what most
trainings agree on). Finally the production rows: inizio = a downbeat, ts = the bar length.

Knobs: J (section cost) and CB (odd-bar cost) -- both fixed here; the BPM octave comes from the user's
BPM range setting (bpm_min .. 2 bpm_min).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.tempo_rows import rebuild_grid_from_segments

from . import bars as B
from . import critic as C
from . import decode as D
from . import gradual as G
from . import junctions as JN
from . import tcn as T

@dataclass
class SeedResult:
    rows: np.ndarray
    beats: np.ndarray
    downbeats: np.ndarray


@dataclass
class TrackerResult:
    segments: np.ndarray          # production tempo rows [start, end, bpm, inizio (downbeat), ts, bpm_end]
    beats: np.ndarray
    downbeats: np.ndarray
    seed: int                     # which seed's grid the consensus kept
    seeds: list


class Models:
    """The trained parts, loaded once: TCN seeds, critic folds, the contextual downbeat GBM."""

    def __init__(self, tcn_paths=T.WEIGHT_PATHS, critic_path=C.WEIGHT_PATH, ctx_path=B.CTX_PATH):
        self.tcn = [T.load_weights(p) for p in tcn_paths]
        self.critic = C.load_weights(critic_path)
        self.ctx = B.load_ctx_model(ctx_path)


_MODELS = None


def load_models() -> Models:
    """The shipped models, loaded once per process."""
    global _MODELS
    if _MODELS is None:
        _MODELS = Models()
    return _MODELS


def dominant_tempo(segments) -> float:
    """The track's main tempo: the tempo (within 1 %) that covers the most time (a ramp counts at its mean)."""
    rows = np.asarray(segments, float)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return 0.0
    tempo = 0.5 * (rows[:, 2] + rows[:, 5]) if rows.shape[1] > 5 else rows[:, 2]
    weight = np.maximum(rows[:, 1] - rows[:, 0], 0.0)
    order = np.argsort(tempo)
    groups = []
    for i in order:
        if groups and abs(tempo[i] / groups[-1][0][-1] - 1.0) <= 0.01:
            groups[-1][0].append(tempo[i]); groups[-1][1].append(weight[i])
        else:
            groups.append(([tempo[i]], [weight[i]]))
    t, w = max(groups, key=lambda g: sum(g[1]))
    return float(np.average(t, weights=np.asarray(w) + 1e-9))


def decode_seed(beat, down, t0, hop, dur, frames, octave, models, critic_track, refine_onset):
    """One seed's activations -> rows, beats, downbeats."""
    off = t0 - 0.5 * hop
    val, fine = D.readers(beat, hop, t0)
    base = D.coarse_tempos(beat, hop, octave)
    rows = D.partition(beat, hop, off, dur, D.with_ratios(base, octave))
    rows = D.partition(beat, hop, off, dur, D.with_ratios(np.r_[D.refit_tempos(rows, beat, hop, off), base], octave))
    rows = D.phase_merge(rows, val)
    rows = D.final_refit(rows, val)
    rows = D.final_refit(rows, fine)
    rows, _ = G.ramp_fix(rows, fine)
    rows = B.bar_check(rows, down, t0, hop)
    rows, _ = C.apply(rows, models.critic, critic_track, val, dur)
    # post-process
    from analyzer_core.beat.grid_refine import hold_slow_tail, refine_boundaries
    if refine_onset is not None:
        rows, _ = refine_boundaries(rows, refine_onset[0], refine_onset[1], dur, refine_onset[2])
    rows, _ = hold_slow_tail(rows, float(np.asarray(rows, float)[-1, 1]))
    rows, _ = G.drift_merge(rows, fine)
    rows = JN.merge_resyncs(rows)
    rows = JN.fix(rows, val)
    beats = rebuild_grid_from_segments(rows)
    beats = beats[(beats >= 0) & (beats <= dur)]
    dbs = B.downbeats(beats, rows, down, t0, hop, frames, models.ctx)
    return SeedResult(np.asarray(rows, float), beats, dbs)


def _f1(ref, est, tol=0.070):
    ref = np.sort(np.asarray(ref, float))
    est = np.sort(np.asarray(est, float))
    if ref.size < 2 or est.size < 2:
        return 0.0
    i = j = m = 0
    while i < ref.size and j < est.size:
        d = est[j] - ref[i]
        if abs(d) <= tol:
            m += 1
            i += 1
            j += 1
        elif d < -tol:
            j += 1
        else:
            i += 1
    p, r = m / est.size, m / ref.size
    return 2 * p * r / (p + r) if p + r else 0.0


def _agree(a: SeedResult, b: SeedResult) -> float:
    return _f1(a.beats, b.beats) + 0.5 * _f1(a.downbeats, b.downbeats)


def medoid(results: list) -> int:
    scores = [sum(_agree(c, d) for j, d in enumerate(results) if j != i) for i, c in enumerate(results)]
    return int(np.argmax(scores))


def track(frames, bpm_min: float, models: Models, refine_onset=None, parallel: bool = True) -> TrackerResult:
    """frames: production FrameFeatures on the models' grid (T.SAMPLE_RATE, T.FRAME_HOP_LENGTH);
    refine_onset: (activation, hop, t0) of the local onset model."""
    if (int(frames.sample_rate), int(frames.hop_length)) != (T.SAMPLE_RATE, T.FRAME_HOP_LENGTH):
        raise ValueError(f"the beat tracker reads {T.SAMPLE_RATE} Hz / hop {T.FRAME_HOP_LENGTH} frames, "
                         f"got {frames.sample_rate} Hz / hop {frames.hop_length}")
    ft = np.asarray(frames.frame_times, float)
    t0, hop = float(ft[0]), float(np.median(np.diff(ft)))
    x = T.frame_inputs(frames)
    octave = D.Octave(bpm_min)
    critic_track = C.track_features(frames)
    def one(P):
        beat, down = T.activations(P, x)
        dur = t0 + hop * (beat.size - 1)
        return decode_seed(beat, down, t0, hop, dur, frames, octave, models, critic_track, refine_onset)

    if parallel and len(models.tcn) > 1:
        # the seeds are independent; NumPy releases the GIL in its array work
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(models.tcn)) as pool:
            seeds = list(pool.map(one, models.tcn))
    else:
        seeds = [one(P) for P in models.tcn]
    k = medoid(seeds)
    best = seeds[k]
    segments = B.to_rows(best.rows, best.beats, best.downbeats)
    return TrackerResult(segments, best.beats, best.downbeats, k, seeds)
