from __future__ import annotations
from typing import Callable, Optional
import numpy as np
import librosa
from analyzer_core.global_analyzer import fast_load
from analyzer_core.key.key import _parse_cqt_key_name, keyanalyzer, chroma_to_subdiv_grid
from analyzer_core.key.key_cqt import keydetect as KeyCQTDetector
from core.config import config

def reanalyze_full(
    path: str,
    cfg: config,
    beats_time_sec,
    *,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> tuple[float, float, float, float]:
    """Recompute the key for an arbitrary beat range."""
    if progress_cb:
        progress_cb("Loading audio", 0.0)
    gcf = cfg.analysisconfig
    sr = int(getattr(gcf, "analysis_samp_rate", 44100))
    audio_full = fast_load(path, sr)
    if progress_cb:
        progress_cb("Preparing detector", 0.1)
    use_hpss = getattr(gcf, "use_hpss", False)
    if use_hpss:
        try:
            y_harm, _ = librosa.effects.hpss(audio_full)
        except Exception:
            y_harm = audio_full
    else:
        y_harm = audio_full
    
    if progress_cb:
        progress_cb("Analyzing key", 0.6)
    
    chroma = chroma_to_subdiv_grid(
        y_harm, np.divide(beats_time_sec*sr, gcf.chroma_hop_length).astype(int), sr=sr,
        hop_length=gcf.chroma_hop_length,
        bins_per_octave=gcf.chroma_cqt_bins_per_octave,
        n_octaves=gcf.chroma_cqt_octaves,
        mode=gcf.chroma_method,
        subdiv=1
    )

    key, key_12, logB, _, logPi, logA, key_segments = keyanalyzer(
        np.asarray(chroma["chroma_subdiv"]),
        chroma["t_subdiv"],
        config=cfg,
        y_harm=y_harm,
        sample_rate=sr,
        beat_times=beats_time_sec
    )
    
    if progress_cb:
        progress_cb("Key segment updated", 0.9)
    return key_segments

def reanalyze_segment_from_file(
    path: str,
    cfg: config,
    beats_time_sec,
    beat_start_index: int,
    beat_end_index: int,
    *,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> tuple[float, float, float, float]:
    """Recompute the key for an arbitrary beat range."""
    if progress_cb:
        progress_cb("Loading audio", 0.0)
    gcf = cfg.analysisconfig
    sr = int(getattr(gcf, "analysis_samp_rate", 44100))
    audio_full = fast_load(path, sr)
    duration = len(audio_full) / float(sr) if len(audio_full) else 0.0
    if progress_cb:
        progress_cb("Preparing detector", 0.1)

    beats = np.asarray(beats_time_sec, dtype=float).ravel()
    if beats.size == 0:
        raise ValueError("beats_time_sec must contain beat times")
    if not (0 <= beat_start_index < beat_end_index <= beats.size):
        raise IndexError("beat indices out of range")
    start = float(beats[beat_start_index])
    if beat_end_index < beats.size:
        end = float(beats[beat_end_index])
    else:
        end = float(duration if duration > 0 else beats[-1])
    end = max(end, start)
    if end - start <= 1e-6:
        raise ValueError("selection duration too small")

    start_sample = int(max(0.0, start) * sr)
    end_sample = int(min(end, duration) * sr)
    segment = audio_full[start_sample:end_sample].astype(np.float32, copy=False)
    if segment.size < sr * 0.5:
        raise ValueError("selection audio too short for key analysis")

    use_hpss = getattr(gcf, "use_hpss", False)
    if use_hpss:
        try:
            y_harm, _ = librosa.effects.hpss(segment)
        except Exception:
            y_harm = segment
    else:
        y_harm = segment

    segment_duration = len(y_harm) / float(sr)
    beat_times_local = beats[(beats >= start) & (beats <= end)] - start
    detector = KeyCQTDetector(y_harm, sr)
    if progress_cb:
        progress_cb("Analyzing key segment", 0.6)
    local_range = (0.0, segment_duration) if segment_duration > 0 else None
    cqt_out = detector.keydetect(songrange=local_range, beat_times=beat_times_local)
    top_candidate = None
    for cand in cqt_out.get("keys", []):
        parsed = _parse_cqt_key_name(cand.get("key"))
        if parsed is not None:
            top_candidate = parsed
            break
    if top_candidate is None:
        raise ValueError("CQT key detection failed for selected beats")

    pitch_class, mode_flag = top_candidate

    if progress_cb:
        progress_cb("Key segment updated", 0.9)
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
