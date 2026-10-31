import math
import numpy as np
from views.render.color import key_cmap_color


def transposed(key_segments, semitones: int):
    """Key rows [pitch, mode, start, end] with every key moved by ``semitones`` (its mode kept): the keys as heard."""
    if not semitones:
        return key_segments
    try:
        arr = np.array(key_segments, dtype=float, copy=True)
    except Exception:
        return key_segments
    if arr.ndim != 2 or arr.shape[1] < 4:
        return key_segments
    arr[:, 0] = np.mod(arr[:, 0] + semitones, 12)
    return arr


def scaled_tempo_segments(tempo_segments, speed: float):
    """Tempo rows [start, end, bpm, inizio, ts, bpm_end] with their tempos times ``speed`` (the times as they are):
    the tempos as heard."""
    if speed == 1.0 or not speed > 0:
        return tempo_segments
    try:
        arr = np.array(tempo_segments, dtype=float, copy=True)
    except Exception:
        return tempo_segments
    if arr.ndim != 2 or arr.shape[1] < 3:
        return tempo_segments
    arr[:, 2] *= speed
    if arr.shape[1] > 5:
        arr[:, 5] *= speed
    return arr


def key_strip_segments(key_segments) -> list[tuple[float, float, tuple[int, int, int]]]:
    """(start, end, colour) of each key segment, for a SegmentStrip."""
    return [(start, end, key_cmap_color(idx)) for idx, start, end in key_segment_rows(key_segments)]


def key_segment_rows(key_segments) -> list[tuple[int, float, float]]:
    """(key index 0..23, start, end) of the valid rows of [pitch, mode, start, end], by start."""
    try:
        arr = np.asarray(key_segments, dtype=float)
    except Exception:
        return []
    if arr.ndim != 2 or arr.shape[1] < 4:
        return []
    rows = []
    for pitch, mode, start, end in arr[:, :4]:
        if not (math.isfinite(pitch) and math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        idx = int(pitch) % 12 + (12 if math.isfinite(mode) and int(mode) == 1 else 0)
        rows.append((idx, float(start), float(end)))
    rows.sort(key=lambda r: r[1])
    return rows


def key_segment_label(idx: int) -> str:
    """Key strip label: Alphanumeric / classical, as in the track info ("3B / C#")."""
    from analyzer.music.key_names import idx_to_labels
    num, cls = idx_to_labels(idx)
    return f"{num}\u2009/\u2009{cls}"
