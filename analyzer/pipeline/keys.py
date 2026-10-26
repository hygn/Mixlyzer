"""The editor's key reanalysis: the whole track or a selected span, with the full analysis's chord-aware key model
(analyzer/key/chord_key.py) on the editor's current beatgrid, bars and phrases."""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from analyzer.pipeline.track import fast_load
from analyzer.audio.hpss import hpss_audio_with_spectra
from analyzer.key import chord_key as CK
from analyzer.music.beat_geometry import downbeat_times
from core.config import ANALYSIS_SAMPLE_RATE, config
from foundation.i18n import tr


def _track_evidence(path: str, cfg: config, beats, tempo_segments, phrases, progress_cb):
    """(model, beat grid, per-beat key evidence, duration) of the whole track."""
    if progress_cb:
        progress_cb(tr("Loading audio"), 0.0)
    gcf = cfg.analysisconfig
    sr = ANALYSIS_SAMPLE_RATE
    samp = np.ascontiguousarray(fast_load(path, target_sr=sr, stereo=True).mean(axis=1), dtype=np.float32)
    duration = samp.size / float(sr)
    if progress_cb:
        progress_cb(tr("Separating harmonics"), 0.15)
    y_harm = hpss_audio_with_spectra(samp, sr)[0]
    if progress_cb:
        progress_cb(tr("Analyzing key"), 0.5)
    beats = np.asarray(beats, dtype=float).ravel()
    segs = np.asarray(tempo_segments, dtype=float) if tempo_segments is not None else np.zeros((0, 5))
    downbeats = downbeat_times(beats, segs) if segs.ndim == 2 and segs.shape[0] else None
    starts = [float(p["start"]) for p in (phrases or []) if "start" in p]
    model, grid, L = CK.key_evidence(y_harm, sr, beats, downbeats, starts, duration,
                                     getattr(gcf, "key_parameter_path", None) or None)
    return model, grid, L, duration


def reanalyze_full(
    path: str,
    cfg: config,
    beats_time_sec,
    *,
    tempo_segments=None,
    phrases=None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> np.ndarray:
    """Key segments [n, 4] (pitch class, mode, start, end) of the whole track (Settings > Analysis > Key > Dynamic
    Analysis off: one key, the dynamic analysis's key in the intro / outro / interludes)."""
    model, grid, L, duration = _track_evidence(path, cfg, beats_time_sec, tempo_segments, phrases, progress_cb)
    _, key_segments = CK.track_keys(model, grid, L, duration, bool(getattr(cfg.analysisconfig, "key_dynamic", True)),
                                    phrases)
    if progress_cb:
        progress_cb(tr("Key segment updated"), 0.9)
    return key_segments


def reanalyze_segment_from_file(
    path: str,
    cfg: config,
    beats_time_sec,
    beat_start_index: int,
    beat_end_index: int,
    *,
    grid_beats=None,
    tempo_segments=None,
    phrases=None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> tuple[float, float, float, float]:
    """One key for a beat range (``beats_time_sec`` indices; ``grid_beats``: the track's beat grid, when
    ``beats_time_sec`` carries extra bounds): the key with the most evidence over the range's beats, the evidence
    read on the whole track (its bars and phrases). -> (pitch class, mode, start, end)"""
    beats = np.asarray(beats_time_sec, dtype=float).ravel()
    if beats.size == 0:
        raise ValueError("beats_time_sec must contain beat times")
    if not (0 <= beat_start_index < beat_end_index <= beats.size):
        raise IndexError("beat indices out of range")
    model, grid, L, duration = _track_evidence(
        path, cfg, beats if grid_beats is None else grid_beats, tempo_segments, phrases, progress_cb)
    start = float(beats[beat_start_index])
    end = float(beats[beat_end_index]) if beat_end_index < beats.size else float(duration)
    end = min(max(end, start), duration)
    if end - start <= 1e-6:
        raise ValueError(tr("selection duration too small"))
    pitch_class, mode_flag = CK.selection_key(grid, L, start, end)
    if progress_cb:
        progress_cb(tr("Key segment updated"), 0.9)
    return float(pitch_class), float(mode_flag), float(start), float(end)


def update_key_segments_with_selection(
    key_segments,
    selection: tuple[float, float, float, float],
    *,
    beat_times=None,
) -> np.ndarray:
    """Return new key segments where the provided selection overwrites its time span."""
    seg_arr = np.asarray(key_segments, dtype=float)
    if seg_arr.ndim != 2 or seg_arr.shape[1] < 4:
        raise ValueError("key_segments must be (N, 4)")
    pitch_class, mode_flag, start, end = selection
    start = float(start)
    end = float(end)
    if end <= start:
        raise ValueError("selection end must be greater than start")
    new_seg = np.array([float(pitch_class), float(mode_flag), start, end], dtype=float)
    result: list[np.ndarray] = []
    inserted = False
    for seg in seg_arr:
        s = float(seg[2])
        e = float(seg[3])
        if e <= start:
            result.append(seg.copy())
            continue
        if s >= end:
            if not inserted:
                result.append(new_seg.copy())
                inserted = True
            result.append(seg.copy())
            continue
        # overlaps
        if s < start:
            left = seg.copy()
            left[3] = start
            if left[3] - left[2] > 1e-9:
                result.append(left)
        if not inserted:
            result.append(new_seg.copy())
            inserted = True
        if e > end:
            right = seg.copy()
            right[2] = end
            if right[3] - right[2] > 1e-9:
                result.append(right)
    if not inserted:
        result.append(new_seg.copy())
    result_arr = np.asarray(result, dtype=float)
    order = np.argsort(result_arr[:, 2])

    # Estimate beat length (if provided) to guide tolerance and filtering.
    beat_len = 0.0
    if beat_times is not None:
        bt = np.asarray(beat_times, dtype=float).ravel()
        if bt.size >= 2:
            diffs = np.diff(np.unique(np.sort(bt)))
            diffs = diffs[diffs > 1e-6]
            if diffs.size:
                beat_len = float(np.median(diffs))

    # Merge adjacent segments that share the same key.
    tol_base = 1e-6
    tolerance = tol_base
    if beat_len > 0.0:
        tolerance = max(tol_base, min(0.25 * beat_len, 0.5))
    merged: list[np.ndarray] = []
    for seg in result_arr[order]:
        if not merged:
            merged.append(seg.copy())
            continue
        prev = merged[-1]
        same_key = prev[0] == seg[0] and prev[1] == seg[1]
        gap = float(seg[2] - prev[3])
        if same_key and gap <= tolerance:
            prev[3] = max(prev[3], seg[3], prev[3])
        else:
            merged.append(seg.copy())

    # Drop segments shorter than one beat, if beat grid is provided.
    if beat_len > 0.0:
        filtered = []
        min_len = max(0.0, beat_len - tolerance)
        for seg in merged:
            dur = float(seg[3] - seg[2])
            if dur < min_len:
                continue
            filtered.append(seg)
        merged = filtered

    return np.asarray(merged, dtype=float)
