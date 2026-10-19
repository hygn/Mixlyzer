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

Knobs: the decoder's costs (``Knobs``: a new section, a same-tempo half-beat reset, an odd bar), in evidence
units of the activation -- so they belong to the beat model and are refitted with it (optimizer/
beat_decoder_optimizer.py); the BPM octave comes from the user's BPM range setting (bpm_min .. 2 bpm_min). Static analysis (a setting): the one steady grid with the most
evidence over the whole track instead of the section decoding, none of the tempo-change stages; its bars
are static too (one bar length and phase, no restarts) unless dynamic downbeats are asked for.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

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


@dataclass(frozen=True)
class Knobs:
    """The decoder's costs in evidence units (the activation's beat contrast summed over beats): what a new
    section, a same-tempo half-beat reset (on top of a section) and an odd bar must explain to be made."""
    section: float = D.J
    half_reset: float = D.HALF_RESET
    odd_bar: float = B.CB


KNOBS_FORMAT = 'mixlyzer.beat_decoder.v1'


def load_knobs(path: str | None) -> Knobs:
    """The costs in a decoder settings file (JSON); no path: the defaults."""
    if not path:
        return Knobs()
    from core.model_files import resolve_model
    with open(resolve_model(path), encoding='utf-8') as f:
        data = json.load(f)
    if data.get('format') != KNOBS_FORMAT:
        raise ValueError(f'Beat decoder settings format mismatch: {path}')
    return Knobs(**{k: float(v) for k, v in data['knobs'].items() if k in Knobs.__dataclass_fields__})


def save_knobs(path, knobs: Knobs, training: dict | None = None) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'format': KNOBS_FORMAT, 'knobs': asdict(knobs), 'training': training or {}}, f, indent=2,
                  ensure_ascii=False)


class Models:
    """The trained parts, loaded once: TCN seeds, critic folds, the contextual downbeat GBM, the decoder's
    costs."""

    def __init__(self, tcn_path=T.WEIGHT_PATH, critic_path=C.WEIGHT_PATH, ctx_path=B.CTX_PATH, knobs_path=None):
        self.tcn = T.load_weights(tcn_path)
        self.critic = C.load_weights(critic_path)
        self.ctx = B.load_ctx_model(ctx_path)
        self.knobs = load_knobs(knobs_path)


_MODELS: dict = {}


def load_models(ctx_path: str | None = None, tcn_path: str | None = None, critic_path: str | None = None,
                knobs_path: str | None = None) -> Models:
    """The models (the shipped ones, or retrained ones at the given paths), loaded once per process and
    paths."""
    key = (ctx_path or B.CTX_PATH, tcn_path or T.WEIGHT_PATH, critic_path or C.WEIGHT_PATH, knobs_path or '')
    if key not in _MODELS:
        _MODELS[key] = Models(ctx_path=key[0], tcn_path=key[1], critic_path=key[2], knobs_path=key[3] or None)
    return _MODELS[key]


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


def decode_seed(beat, down, t0, hop, dur, frames, octave, models, critic_track, refine_onset,
                dynamic=True, dynamic_downbeat=True):
    """One seed's activations -> rows, beats, downbeats. Static analysis (``dynamic`` off) keeps one steady
    grid for the whole track; static bars (``dynamic_downbeat`` off) keep one bar length and phase."""
    k = models.knobs
    val, _ = D.readers(beat, hop, t0)
    rows = sections(beat, hop, t0, dur, octave, k.section) if dynamic else None
    rows, beats = seed_grid(rows, beat, down, t0, hop, dur, octave, models, critic_track, refine_onset, k, dynamic)
    dbs = B.downbeats(beats, rows, down, t0, hop, frames, models.ctx, restarts=dynamic or dynamic_downbeat,
                      cost=k.odd_bar)
    return SeedResult(np.asarray(rows, float), beats, dbs)


def sections(beat, hop, t0, dur, octave, section_cost):
    """The section decoding (the expensive step; depends only on the section cost)."""
    off = t0 - 0.5 * hop
    base = D.coarse_tempos(beat, hop, octave)
    rows = D.partition(beat, hop, off, dur, D.with_ratios(base, octave), section_cost)
    return D.partition(beat, hop, off, dur, D.with_ratios(np.r_[D.refit_tempos(rows, beat, hop, off), base], octave),
                       section_cost)


def seed_grid(rows, beat, down, t0, hop, dur, octave, models, critic_track, refine_onset, knobs: Knobs,
              dynamic=True):
    """From the decoded sections (``rows``; None for static analysis) to the seed's grid: rows and beats."""
    val, fine = D.readers(beat, hop, t0)
    if dynamic:
        rows = D.phase_merge(rows, val, knobs.half_reset)
    else:
        off = t0 - 0.5 * hop
        rows = D.single_grid(beat, hop, off, dur, D.with_ratios(D.coarse_tempos(beat, hop, octave), octave))
    rows = D.final_refit(rows, val)
    rows = D.final_refit(rows, fine)
    if dynamic:
        rows, _ = G.ramp_fix(rows, fine)
    rows = B.bar_check(rows, down, t0, hop)
    rows, _ = C.apply(rows, models.critic, critic_track, val, dur)
    rows = B.fold_into_octave(rows, octave.lo, down, t0, hop, val)
    if dynamic:
        # post-process
        from analyzer_core.beat.grid_refine import hold_slow_tail, refine_boundaries
        if refine_onset is not None:
            rows, _ = refine_boundaries(rows, refine_onset[0], refine_onset[1], dur, refine_onset[2])
        rows, _ = hold_slow_tail(rows, float(np.asarray(rows, float)[-1, 1]))
        rows, _ = G.drift_merge(rows, fine, J=knobs.section)
        rows = JN.merge_resyncs(rows)
        rows = JN.fix(rows, val)
    beats = rebuild_grid_from_segments(rows)
    return rows, beats[(beats >= 0) & (beats <= dur)]


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


def track(frames, bpm_min: float, models: Models, refine_onset=None, parallel: bool = True,
          dynamic: bool = True, dynamic_downbeat: bool = True) -> TrackerResult:
    """frames: production FrameFeatures on the models' grid (T.SAMPLE_RATE, T.FRAME_HOP_LENGTH);
    refine_onset: (activation, hop, t0) of the local onset model. ``dynamic`` off: one steady grid for
    the whole track; its bars are dynamic (odd bars, meter changes) only with ``dynamic_downbeat`` (a
    dynamic grid always has dynamic bars)."""
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
        return decode_seed(beat, down, t0, hop, dur, frames, octave, models, critic_track, refine_onset,
                           dynamic, dynamic_downbeat)

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
