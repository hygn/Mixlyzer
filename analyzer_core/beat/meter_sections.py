"""Sustained 3/4 sections in an otherwise 4/4 grid.

Short odd bars (one 3/4 or 5/4 bar) are left to the downbeat tracker, which handles them as
a downbeat shift. A section that stays in 3/4 for several bars cannot be written that way, so
it gets its own tempo segment with ``ts = 3``:

1. Per beat, how much the beat-synchronous features (chroma, bass chroma, MFCC and onset
   cells) repeat after 2..24 beats, smoothed over 8 / 16 / 32 beats. Bars of three repeat at
   lags 6 / 12 / 24 more than at 8 / 16.
2. A classifier (ExtraTrees, NumPy runtime) gives P(3/4 | beat).
3. A two-state Viterbi (4/4, 3/4) with a switching penalty keeps sections of at least
   ``MIN_SECTION_BEATS`` beats.
4. Inside a section the downbeat is the period-3 fold maximum of the learned downbeat
   probability.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from analyzer_core.beat.candidate_selector import NumpyExtraTreesClassifier, _forest_kwargs
from analyzer_core.beat.frame_features import FrameFeatures
from core.resource_paths import resource_path

MODEL_FORMAT = "mixlyzer_meter_section_v1"
MODEL_PATH = resource_path("assets/weights/meter_section_classifier.npz")
LAGS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24)
SMOOTH_BEATS = (8, 16, 32)
FEATURE_COUNT = len(LAGS) * len(SMOOTH_BEATS)
SWITCH_PENALTY = 60.0
THREE_FOUR_BIAS = 1.5
MIN_SECTION_BEATS = 32


def _beat_sync(frames: FrameFeatures, beats: np.ndarray, cells: int = 4) -> np.ndarray:
    ft = frames.frame_times
    n_frames = ft.size
    edges = np.r_[beats, beats[-1] + np.median(np.diff(beats))]
    index = np.searchsorted(ft, edges)
    blocks = (frames.chroma, frames.bass_chroma, frames.mfcc[1:13])
    onset = np.vstack([frames.percussive_bands, frames.melody_onset[None], frames.harmonic_onset[None]])
    rows = []
    for i in range(beats.size):
        a = min(int(index[i]), n_frames - 1)
        b = max(min(int(index[i + 1]), n_frames), a + 1)
        values = [block[:, a:b].mean(axis=1) for block in blocks]
        cuts = np.minimum(np.linspace(a, b, cells + 1).astype(int), n_frames - 1)
        for c in range(cells):
            values.append(onset[:, cuts[c]:max(cuts[c + 1], cuts[c] + 1)].max(axis=1))
        rows.append(np.concatenate(values))
    x = np.asarray(rows, dtype=np.float64)
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


def meter_feature_matrix(frames: FrameFeatures, beats: np.ndarray) -> np.ndarray:
    """[n_beats, FEATURE_COUNT] lagged self-similarity of beat-synchronous features."""
    x = _beat_sync(frames, np.asarray(beats, dtype=np.float64))
    unit = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    sim = np.zeros((x.shape[0], len(LAGS)))
    for j, lag in enumerate(LAGS):
        if lag < x.shape[0]:
            sim[:-lag, j] = np.sum(unit[:-lag] * unit[lag:], axis=1)
    return np.concatenate(
        [uniform_filter1d(sim, w, axis=0, mode="nearest") for w in SMOOTH_BEATS], axis=1
    ).astype(np.float32)


@lru_cache(maxsize=2)
def load_meter_model(path: str = str(MODEL_PATH)) -> NumpyExtraTreesClassifier:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(np.asarray(archive["format"]).item()) != MODEL_FORMAT:
            raise ValueError(f"meter section model format mismatch: {path}")
        if int(np.asarray(archive["feature_count"]).item()) != FEATURE_COUNT:
            raise ValueError(f"meter section model feature schema mismatch: {path}")
        model = NumpyExtraTreesClassifier(classes=np.asarray(archive["meter_classes"]), **_forest_kwargs(archive, "meter"))
    if tuple(int(c) for c in model.classes) != (0, 1, 2):
        raise ValueError(f"meter section model class schema mismatch: {path}")
    return model


def three_four_sections(p3: np.ndarray, switch_penalty: float = SWITCH_PENALTY,
                        bias: float = THREE_FOUR_BIAS, min_beats: int = MIN_SECTION_BEATS) -> list[tuple[int, int]]:
    """Beat index ranges [a, b) decoded as 3/4."""
    p3 = np.clip(np.asarray(p3, dtype=np.float64), 1e-3, 1 - 1e-3)
    emit = np.stack([np.log(1 - p3), np.log(p3) + bias], axis=1)
    n = p3.size
    score = emit[0].copy()
    back = np.zeros((n, 2), dtype=np.int64)
    for i in range(1, n):
        switch = score[::-1] - switch_penalty
        back[i] = np.where(score >= switch, [0, 1], [1, 0])
        score = np.maximum(score, switch) + emit[i]
    state = np.empty(n, dtype=np.int64)
    state[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        state[i - 1] = back[i, state[i]]
    sections = []
    i = 0
    while i < n:
        if state[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and state[j + 1] == 1:
            j += 1
        if j - i + 1 >= min_beats:
            sections.append((i, j + 1))
        i = j + 1
    return sections


def apply_meter_sections(
    tempo_segments: np.ndarray,
    beats: np.ndarray,
    sections: list[tuple[int, int]],
    downbeat_probability: np.ndarray,
) -> np.ndarray:
    """Give each 3/4 section its own ``ts = 3`` segments (split at half a beat before its ends).

    The bars run on across the tempo segments a section spans; each part's ``inizio`` is
    a downbeat on its own grid (a beat inside it), so the segments still rebuild the beats.
    """
    from analyzer_core.beat.beat import bpm_at, with_end_bpm

    rows = [list(map(float, r)) for r in with_end_bpm(tempo_segments)]
    beats = np.asarray(beats, dtype=np.float64)
    p = np.asarray(downbeat_probability, dtype=np.float64)
    for a, b in sections:
        idx = np.arange(a, b)
        phase = int(np.argmax([p[idx[k::3]].mean() for k in range(3)]))
        period_a = beats[a + 1] - beats[a] if a + 1 < beats.size else beats[a] - beats[a - 1]
        period_b = beats[b] - beats[b - 1] if b < beats.size else period_a
        t0 = beats[a] - 0.5 * period_a
        t1 = beats[b] - 0.5 * period_b if b < beats.size else rows[-1][1]
        new_rows: list[list[float]] = []
        for row in rows:
            start, end, bpm, inizio, ts, bpm_end = row
            if end <= t0 or start >= t1:
                new_rows.append(list(row))
                continue
            period = 60.0 / bpm if bpm > 0 else 0.0
            # A leftover shorter than a beat holds no beat of its own: the 3/4 part takes it.
            lo = start if t0 - start < period else max(start, t0)
            hi = end if end - t1 < period else min(end, t1)
            inside = np.flatnonzero((beats >= lo - 1e-3) & (beats < hi - 1e-3))
            downbeats = inside[(inside - (a + phase)) % 3 == 0]
            if downbeats.size == 0:
                new_rows.append(list(row))
                continue
            b_lo, b_hi = bpm_at(row, lo), bpm_at(row, hi)
            if start < lo:
                new_rows.append([start, lo, bpm, inizio, ts, b_lo])
            new_rows.append([lo, hi, b_lo, float(beats[downbeats[0]]), 3.0, b_hi])
            if end > hi:
                new_rows.append([hi, end, b_hi, inizio, ts, bpm_end])
        rows = new_rows
    return np.asarray(rows, dtype=np.float32)


def detect_meter_sections(
    frames: FrameFeatures,
    beats: np.ndarray,
    model_path: str | Path = MODEL_PATH,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    model = load_meter_model(str(Path(model_path).resolve()))
    x = meter_feature_matrix(frames, beats)
    p3 = model.predict_proba(x)[:, 0]
    return three_four_sections(p3), p3
