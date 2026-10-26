"""The editor's key reanalysis: the whole track or a selected span, with the full analysis's chord-aware key model
(analyzer/key/chord_key.py) on the editor's current beatgrid, bars and phrases."""
from __future__ import annotations

import numpy as np

from analyzer.audio.hpss import hpss_audio_with_spectra
from analyzer.audio.load import fast_load
from analyzer.errors import SELECTION_TOO_SHORT, AnalysisError
from analyzer.key import chord_key as CK
from analyzer.music.beat_geometry import downbeat_times
from analyzer.music.key_rows import update_key_segments_with_selection
from analyzer.pipeline import stages
from analyzer.pipeline.stages import Progress, report
from analyzer.settings import ANALYSIS_SAMPLE_RATE, AnalysisSettings


def _track_evidence(path: str, settings: AnalysisSettings, beats, tempo_segments, phrases, progress: Progress):
    """(model, beat grid, per-beat key evidence, duration) of the whole track."""
    report(progress, stages.LOAD, 0.0)
    sr = ANALYSIS_SAMPLE_RATE
    samp = np.ascontiguousarray(fast_load(path, target_sr=sr, stereo=True).mean(axis=1), dtype=np.float32)
    duration = samp.size / float(sr)
    report(progress, stages.HPSS, 0.15)
    y_harm = hpss_audio_with_spectra(samp, sr)[0]
    report(progress, stages.KEY, 0.5)
    beats = np.asarray(beats, dtype=float).ravel()
    segs = np.asarray(tempo_segments, dtype=float) if tempo_segments is not None else np.zeros((0, 5))
    downbeats = downbeat_times(beats, segs) if segs.ndim == 2 and segs.shape[0] else None
    starts = [float(p["start"]) for p in (phrases or []) if "start" in p]
    model, grid, L = CK.key_evidence(y_harm, sr, beats, downbeats, starts, duration, settings.models.key)
    return model, grid, L, duration


def reanalyze_full(
    path: str,
    settings: AnalysisSettings,
    beats_time_sec,
    *,
    tempo_segments=None,
    phrases=None,
    progress: Progress = None,
) -> np.ndarray:
    """Key segments [n, 4] (pitch class, mode, start, end) of the whole track (``settings.key_dynamic`` off: one
    key, the dynamic analysis's key in the intro / outro / interludes)."""
    model, grid, L, duration = _track_evidence(path, settings, beats_time_sec, tempo_segments, phrases, progress)
    _, key_segments = CK.track_keys(model, grid, L, duration, bool(settings.key_dynamic), phrases)
    return key_segments


def reanalyze_selection(
    path: str,
    settings: AnalysisSettings,
    beats_time_sec,
    duration: float,
    selection: tuple[float, float],
    key_segments,
    *,
    tempo_segments=None,
    phrases=None,
    progress: Progress = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """One key for the selected span (start, end s), from its first beat to the beat ending it (the track's
    edges count as beats): the key with the most evidence over its beats, the evidence read on the whole track (its
    bars and phrases). -> (``key_segments`` with the span's key over it, (pitch class, mode, start, end))"""
    if selection is None:
        raise ValueError("selection end must be greater than start")
    sel_start, sel_end = map(float, selection)
    if sel_end <= sel_start:
        raise ValueError("selection end must be greater than start")
    grid_beats = np.asarray(beats_time_sec, dtype=float)
    beats = np.concatenate(([0.], grid_beats, [duration]))
    beat_start_index = int(np.searchsorted(beats, sel_start, side="left"))
    beat_end_index = int(np.searchsorted(beats, sel_end, side="right")) - 1
    beat_start_index = max(0, min(beat_start_index, max(beats.size - 1, 0)))
    beat_end_index = max(
        beat_start_index + 1,
        min(max(beat_end_index, beat_start_index + 1), beats.size),
    )
    if not (0 <= beat_start_index < beat_end_index <= beats.size):
        raise IndexError("beat indices out of range")
    model, grid, L, track_duration = _track_evidence(path, settings, grid_beats, tempo_segments, phrases, progress)
    start = float(beats[beat_start_index])
    end = float(beats[beat_end_index]) if beat_end_index < beats.size else float(track_duration)
    end = min(max(end, start), track_duration)
    if end - start <= 1e-6:
        raise AnalysisError(SELECTION_TOO_SHORT)
    pitch_class, mode_flag = CK.selection_key(grid, L, start, end)
    picked = (float(pitch_class), float(mode_flag), float(start), float(end))
    return update_key_segments_with_selection(key_segments, picked, beat_times=beats), picked
