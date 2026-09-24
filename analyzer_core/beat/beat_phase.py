from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from analyzer_core.beat.frame_features import FrameFeatures
from analyzer_core.beat.melody_contour import melody_note_starts
from analyzer_core.beat.periodicity import (
    PERIODICITY_CURVE_NAMES,
    local_periodicity,
    periodicity_curves,
    raw_onset_curves,
    rotate,
    sixteenth_peaks,
)
from core.resource_paths import resource_path


# Beat-phase correction: for every beat, the candidates are the grid shifted by
# 0, 1/4, 1/2 and 3/4 beat. Each candidate is described by how the local pulse
# of every shared onset curve (+-WINDOW_BEATS around the beat) lines up with it
# at periods of 1/2, 1 and 2 beats, and by beat-structure cues measured on the
# 16th grid around it: how much each source alternates between neighbouring
# candidate beats (backbeat-like patterns), how much stronger it is on the
# candidate than on its own off-beat, and how syncopated it is with the candidate
# as the beat (onsets on e / & / a with the next stronger position empty,
# Longuet-Higgins & Lee). Two slower cues are read over longer spans
# (+-HARMONY_WINDOW_BEATS): chord changes, which listeners put on strong beats
# (Dawe et al. 1993; Temperley's harmony rule), and the held length of melody
# notes, since long notes start on strong beats (Temperley & Sleator's length
# rule; Parncutt's durational accent).
#
# Which source carries the beat changes from song to song (kick and bass on the
# & in one, only the hats on the beat in another), so every curve's cues are
# also taken times that curve's regularity in the track: how much less
# syncopated its best phase is than the others. A learned linear model scores
# the candidates per beat; the logits summed over the whole track give the
# grid's phase offset.
SUBDIVISION = 4
PERIODS_BEATS: tuple[float, ...] = (0.5, 1.0, 2.0)
WINDOW_BEATS = 4.0
HARMONY_WINDOW_BEATS = 32
MELODY_WINDOW_BEATS = 32
N_CURVES = len(PERIODICITY_CURVE_NAMES)
_BASE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{curve}_{name}"
    for name in (
        "8th_in_phase", "8th_quadrature",
        "beat_in_phase", "beat_quadrature",
        # The 2-beat cycle's own phase depends on the (unknown) bar position,
        # so only its magnitudes are used.
        "2beat_in_phase_abs", "2beat_quadrature_abs",
    )
    for curve in PERIODICITY_CURVE_NAMES
) + tuple(
    f"{curve}_{name}"
    for name in ("alternation", "on_off", "syncopation")
    for curve in PERIODICITY_CURVE_NAMES
)
_HARMONY_FEATURE_NAMES = (
    "chord_change_on", "chord_change_span_on", "chord_change_on_off", "chord_change_span_on_off",
)
_MELODY_FEATURE_NAMES = (
    "melody_long_note_on", "melody_note_length_on", "melody_short_note_on",
    "melody_long_note_on_off", "melody_note_length_on_off", "melody_short_note_on_off",
)
# Raw per-candidate features (cached by the optimizer, standardized per track).
BEAT_PHASE_RAW_FEATURE_NAMES: tuple[str, ...] = _BASE_FEATURE_NAMES + _HARMONY_FEATURE_NAMES + _MELODY_FEATURE_NAMES
# Per curve: its 6 periodicity and 3 structure columns in the base block.
_CURVE_COLUMNS = np.array(
    [
        [block * N_CURVES + c for block in range(6)] + [6 * N_CURVES + block * N_CURVES + c for block in range(3)]
        for c in range(N_CURVES)
    ]
)
# Model input: the raw features, then every curve's columns times its regularity.
BEAT_PHASE_FEATURE_NAMES: tuple[str, ...] = BEAT_PHASE_RAW_FEATURE_NAMES + tuple(
    f"{_BASE_FEATURE_NAMES[column]}_x_regularity" for c in range(N_CURVES) for column in _CURVE_COLUMNS[c]
)
BEAT_PHASE_MODEL_FORMAT = "mixlyzer_beat_phase_v4"

_DEFAULT_WEIGHT_PATH = resource_path("assets/weights/beat_phase_weights.json")


@dataclass(frozen=True)
class BeatPhaseDecision:
    shift_index: int             # the real beat is shift_index / SUBDIVISION beat later
    logit_sums: tuple[float, ...]
    beat_count: int

    @property
    def shift_beats(self) -> float:
        return self.shift_index / SUBDIVISION

    @property
    def margin(self) -> float:
        """Best minus second-best summed logit, per beat."""
        ordered = sorted(self.logit_sums, reverse=True)
        return (ordered[0] - ordered[1]) / max(self.beat_count, 1)


@dataclass(frozen=True)
class BeatPhaseTrackFeatures:
    groups: np.ndarray        # [n_beats, SUBDIVISION, len(BEAT_PHASE_RAW_FEATURE_NAMES)]; NaN rows lack context
    regularity: np.ndarray    # [N_CURVES]


def beat_phase_track_features(frames: FrameFeatures, beats: np.ndarray) -> BeatPhaseTrackFeatures:
    """Raw candidate features and per-curve regularity of one track."""
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    cells = sixteenth_peaks(frames.frame_times, beats, raw_onset_curves(frames))
    base = _base_groups(frames, beats, cells)
    period = float(np.median(np.diff(beats))) if beats.size > 1 else 0.5
    harmony = _windowed_candidates(_chord_change_cells(frames, beats, period), beats.size, HARMONY_WINDOW_BEATS)
    melody = _windowed_candidates(_melody_length_cells(frames, beats, period), beats.size, MELODY_WINDOW_BEATS)
    groups = np.concatenate([base, harmony, melody], axis=2).astype(np.float32)
    groups[np.isnan(base).any(axis=2)] = np.nan
    return BeatPhaseTrackFeatures(groups=groups, regularity=curve_regularity(cells, beats.size))


def _base_groups(frames: FrameFeatures, beats: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Periodicity and beat-structure candidate features [n_beats, SUBDIVISION, 9 * N_CURVES]."""
    coefficients = local_periodicity(
        frames.frame_times, beats, periodicity_curves(frames), PERIODS_BEATS, WINDOW_BEATS
    )
    empty = ~np.any(coefficients != 0, axis=(1, 2))
    groups = []
    for k in range(SUBDIVISION):
        aligned = rotate(coefficients, np.arange(beats.size) + k / SUBDIVISION, PERIODS_BEATS)
        groups.append(
            np.concatenate(
                [
                    aligned[:, :, 0].real, aligned[:, :, 0].imag,
                    aligned[:, :, 1].real, aligned[:, :, 1].imag,
                    np.abs(aligned[:, :, 2].real), np.abs(aligned[:, :, 2].imag),
                ],
                axis=1,
            )
        )
    periodic = np.stack(groups, axis=1)
    structure = _beat_structure(cells, beats.size)
    out = np.concatenate([periodic, structure], axis=2).astype(np.float32)
    out[empty] = np.nan
    return out


def _syncopation(cells: np.ndarray, beat: np.ndarray, k: int) -> np.ndarray:
    """Local syncopation per beat with candidate ``k`` as the beat: [len(beat), C]."""
    def at(offset: int) -> np.ndarray:
        return cells[np.minimum(beat * 4 + k + offset, cells.shape[0] - 1)]
    e, off, a, next_on = at(1), at(2), at(3), at(4)
    return e * (1.0 - off) + off * (1.0 - next_on) + 2.0 * a * (1.0 - next_on)


def _beat_structure(cells: np.ndarray, n_beats: int) -> np.ndarray:
    """Alternation / on-off / syncopation per candidate, averaged over +-WINDOW_BEATS: [n, 4, 3C]."""
    n_curves = cells.shape[1]
    beat = np.arange(n_beats)
    window = int(WINDOW_BEATS)
    lo = np.clip(beat - window, 0, n_beats)
    hi = np.clip(beat + window, 0, n_beats)
    count = np.maximum(hi - lo, 1)[:, np.newaxis]

    def window_mean(values: np.ndarray) -> np.ndarray:
        cumulative = np.vstack([np.zeros((1, n_curves)), np.cumsum(values, axis=0)])
        return (cumulative[hi] - cumulative[lo]) / count

    out = np.zeros((n_beats, SUBDIVISION, 3 * n_curves))
    for k in range(SUBDIVISION):
        on = cells[np.minimum(beat * 4 + k, cells.shape[0] - 1)]
        off = cells[np.minimum(beat * 4 + k + 2, cells.shape[0] - 1)]
        next_on = cells[np.minimum(beat * 4 + k + 4, cells.shape[0] - 1)]
        out[:, k, :n_curves] = window_mean(np.abs(on - next_on))
        out[:, k, n_curves : 2 * n_curves] = window_mean(on - off)
        out[:, k, 2 * n_curves :] = window_mean(_syncopation(cells, beat, k))
    return out


def curve_regularity(cells: np.ndarray, n_beats: int) -> np.ndarray:
    """Per curve: 1 - (syncopation of its least syncopated phase) / (mean over the four phases)."""
    beat = np.arange(n_beats)
    syncopation = np.stack([_syncopation(cells, beat, k).mean(axis=0) for k in range(SUBDIVISION)])
    return (1.0 - syncopation.min(axis=0) / np.maximum(syncopation.mean(axis=0), 1e-9)).astype(np.float64)


def _windowed_candidates(cells: np.ndarray, n_beats: int, window: int) -> np.ndarray:
    """Cue mass at every candidate and candidate minus its off-beat, averaged over +-window beats.

    ``cells`` is [4 * n_beats + 4, K]; returns [n_beats, SUBDIVISION, 2K].
    """
    n_cues = cells.shape[1]
    beat = np.arange(n_beats)
    lo = np.clip(beat - window, 0, n_beats)
    hi = np.clip(beat + window, 0, n_beats)
    count = np.maximum(hi - lo, 1)[:, np.newaxis]
    out = np.zeros((n_beats, SUBDIVISION, 2 * n_cues))
    for k in range(SUBDIVISION):
        on = cells[np.minimum(beat * 4 + k, cells.shape[0] - 1)]
        off = cells[np.minimum(beat * 4 + k + 2, cells.shape[0] - 1)]
        for j, values in enumerate((on, on - off)):
            cumulative = np.vstack([np.zeros((1, n_cues)), np.cumsum(values, axis=0)])
            out[:, k, j * n_cues:(j + 1) * n_cues] = (cumulative[hi] - cumulative[lo]) / count
    return out


def _chord_change_cells(frames: FrameFeatures, beats: np.ndarray, period: float) -> np.ndarray:
    """Chord changes on the 16th grid [4n+4, 2]: change at its local maxima, and that times the span to the next.

    The change at a grid point is 1 - correlation of the mean chroma over the
    beat before and the beat after it.
    """
    n = beats.size
    n_cells = 4 * n + 4
    frame_times = frames.frame_times
    chroma = np.asarray(frames.chroma, dtype=np.float64)
    grid = np.interp(np.arange(n_cells) / 4.0, np.arange(n), beats)
    index = np.searchsorted(frame_times, grid)
    half = max(2, int(round(period / max(float(frame_times[1] - frame_times[0]), 1e-9))))
    cumulative = np.vstack([np.zeros((1, chroma.shape[0])), np.cumsum(chroma.T, axis=0)])
    T = frame_times.size
    lo = np.clip(index - half, 0, T)
    mid = np.clip(index, 0, T)
    hi = np.clip(index + half, 0, T)
    before = (cumulative[mid] - cumulative[lo]) / np.maximum(mid - lo, 1)[:, np.newaxis]
    after = (cumulative[hi] - cumulative[mid]) / np.maximum(hi - mid, 1)[:, np.newaxis]
    before = before - before.mean(axis=1, keepdims=True)
    after = after - after.mean(axis=1, keepdims=True)
    correlation = (before * after).sum(axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-9
    )
    change = np.clip(1.0 - correlation, 0.0, 2.0)
    peak = (change >= np.roll(change, 1)) & (change >= np.roll(change, -1)) & (change > np.percentile(change, 60))
    out = np.zeros((n_cells, 2))
    out[:, 0] = change * peak
    starts = np.flatnonzero(peak)
    span = np.diff(np.append(starts, n_cells))
    out[starts, 1] = change[starts] * np.log2(1.0 + np.minimum(span, 16))
    return out


def _melody_length_cells(frames: FrameFeatures, beats: np.ndarray, period: float) -> np.ndarray:
    """Melody note starts on the 16th grid [4n+4, 3]: long (>= 1 beat), log length in 16ths, short (< 1/2 beat)."""
    n = beats.size
    out = np.zeros((4 * n + 4, 3))
    starts, lengths = melody_note_starts(frames.frame_times, frames.melody_pitch, frames.melody_contour)
    inside = (starts > beats[0]) & (starts < beats[-1])
    if not inside.any():
        return out
    cell = np.clip(np.rint(np.interp(starts[inside], beats, np.arange(n)) * 4).astype(np.int64), 0, 4 * n + 3)
    held = lengths[inside]
    np.add.at(out[:, 0], cell, (held >= period).astype(np.float64))
    np.add.at(out[:, 1], cell, np.log2(1.0 + held / (period / 4.0)))
    np.add.at(out[:, 2], cell, (held < period / 2.0).astype(np.float64))
    return out


def _robust_standardize_columns(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return np.clip((values - median) / scale, -8.0, 8.0)


def standardize_beat_phase_groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-track standardized groups (NaN as 0) and the mask of beats with context."""
    flat = groups.reshape(-1, groups.shape[2])
    present = ~np.isnan(flat).any(axis=1)
    standardized = np.zeros_like(flat, dtype=np.float64)
    if present.any():
        standardized[present] = _robust_standardize_columns(flat[present].astype(np.float64))
    complete = present.reshape(groups.shape[:2]).all(axis=1)
    return standardized.reshape(groups.shape), complete


def beat_phase_design(
    standardized: np.ndarray,
    regularity: np.ndarray,
    regularity_mean: np.ndarray,
    regularity_std: np.ndarray,
) -> np.ndarray:
    """Model input [n, SUBDIVISION, len(BEAT_PHASE_FEATURE_NAMES)] from standardized raw groups."""
    z = (np.asarray(regularity, dtype=np.float64) - regularity_mean) / np.maximum(regularity_std, 1e-9)
    gated = standardized[:, :, _CURVE_COLUMNS] * z[np.newaxis, np.newaxis, :, np.newaxis]
    return np.concatenate([standardized, gated.reshape(standardized.shape[0], SUBDIVISION, -1)], axis=2)


@dataclass(frozen=True)
class BeatPhaseModel:
    weights: np.ndarray
    regularity_mean: np.ndarray
    regularity_std: np.ndarray


def clear_beat_phase_weight_cache() -> None:
    """Make a newly written weight artifact visible in this process."""
    _load_beat_phase_weights.cache_clear()


@lru_cache(maxsize=4)
def _load_beat_phase_weights(weight_path: str) -> BeatPhaseModel:
    path = Path(weight_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    model = report.get("model", report)
    if str(model.get("format", "")) != BEAT_PHASE_MODEL_FORMAT:
        raise ValueError(f"Beat phase weight format mismatch: {path}")
    if tuple(str(name) for name in model["feature_names"]) != BEAT_PHASE_FEATURE_NAMES:
        raise ValueError(f"Beat phase weight feature schema mismatch: {path}")
    weights = np.asarray(model["weights"], dtype=np.float64)
    regularity_mean = np.asarray(model["regularity_mean"], dtype=np.float64)
    regularity_std = np.asarray(model["regularity_std"], dtype=np.float64)
    if (
        weights.shape != (len(BEAT_PHASE_FEATURE_NAMES),)
        or regularity_mean.shape != (N_CURVES,)
        or regularity_std.shape != (N_CURVES,)
        or not all(np.all(np.isfinite(v)) for v in (weights, regularity_mean, regularity_std))
    ):
        raise ValueError(f"Invalid beat phase weights: {path}")
    return BeatPhaseModel(weights, regularity_mean, regularity_std)


def detect_beat_phase(
    frames: FrameFeatures,
    beats: np.ndarray,
    weight_path: str | Path = _DEFAULT_WEIGHT_PATH,
) -> BeatPhaseDecision:
    """Pick the grid phase whose candidates score highest over the whole track."""
    model = _load_beat_phase_weights(str(Path(weight_path).resolve()))
    features = beat_phase_track_features(frames, beats)
    standardized, complete = standardize_beat_phase_groups(features.groups)
    design = beat_phase_design(standardized, features.regularity, model.regularity_mean, model.regularity_std)
    logits = design[complete] @ model.weights
    sums = logits.sum(axis=0) if logits.size else np.zeros(SUBDIVISION)
    return BeatPhaseDecision(
        shift_index=int(np.argmax(sums)),
        logit_sums=tuple(float(v) for v in sums),
        beat_count=int(complete.sum()),
    )


def shift_beat_grid(
    beats: np.ndarray,
    tempo_segments: np.ndarray,
    shift_beats: float,
    duration_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the grid ``shift_beats`` (0..1) later in beat-index space.

    Beats keep their local spacing (tempo changes included). A beat is added in
    front when the shifted grid leaves room for one, and beats past the track end
    are dropped. Segment boundaries stay; each ``inizio`` moves with the grid.
    """
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    segments = np.asarray(tempo_segments, dtype=np.float64).copy()
    if abs(shift_beats) < 1e-9 or beats.size < 2:
        return beats, segments
    n = beats.size
    first_interval = beats[1] - beats[0]
    last_interval = beats[-1] - beats[-2]
    position = np.arange(-1, n, dtype=np.float64) + float(shift_beats)
    shifted = np.interp(position, np.arange(n, dtype=np.float64), beats)
    shifted = np.where(position < 0, beats[0] + position * first_interval, shifted)
    shifted = np.where(position > n - 1, beats[-1] + (position - (n - 1)) * last_interval, shifted)
    shifted = shifted[(shifted >= 0.0) & (shifted <= float(duration_sec))]
    if segments.ndim == 2 and segments.shape[1] >= 4 and segments.size:
        for row in segments:
            if row[2] > 0:
                row[3] = row[3] + float(shift_beats) * 60.0 / row[2]
    return shifted, segments
