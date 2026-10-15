from __future__ import annotations

import math
from typing import Any

import numpy as np


def beat_quantize_grid(beat_times: Any, duration_sec: Any) -> np.ndarray:
    """Editing snap points: real beats plus the exact track start and end."""
    try:
        beats = np.asarray(beat_times, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        beats = np.empty(0, dtype=float)
    beats = beats[np.isfinite(beats) & (beats >= 0.0)]

    try:
        duration = float(duration_sec)
    except (TypeError, ValueError):
        duration = math.nan
    if math.isfinite(duration) and duration > 0.0:
        beats = beats[beats <= duration]
        beats = np.concatenate((beats, np.asarray([0.0, duration])))
    else:
        beats = np.concatenate((beats, np.asarray([0.0])))
    return np.unique(beats)


def quantize_to_beat_or_edge(t: float, beat_times: Any, duration_sec: Any) -> float:
    """Nearest edit snap point, including 0:00 and the exact track duration."""
    grid = beat_quantize_grid(beat_times, duration_sec)
    return float(grid[int(np.argmin(np.abs(grid - float(t))))])


__all__ = ["beat_quantize_grid", "quantize_to_beat_or_edge"]
