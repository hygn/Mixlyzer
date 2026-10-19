"""The grid-level check of a retrained beat model: the full tracker with the current and with the new models on
songs the new one was not trained on, scored against the user's grids. A frame-level score cannot see what
matters here -- small changes of an activation flip the tracker's discrete choices (a half-beat phase, a
tempo reading) on fragile songs -- so a new model is kept only when the grids it gives are not worse.

Scores: the beat time that is not usable (a beat of the user's grid farther than 25 ms from the tracker's,
an octave allowed, a half-beat shift at the same tempo not) and the downbeat F-measure (70 ms)."""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from core.concurrency.process_pool import create_spawn_process_pool
from optimizer.beat_frames import cache_path, read_frames

ProgressCallback = Callable[[int, str], None]
USABLE_SEC = 0.025


@dataclass(frozen=True)
class GridCheck:
    songs: int
    unusable_before_sec: float
    unusable_after_sec: float
    downbeat_f_before: float
    downbeat_f_after: float
    worse_songs: int               # songs with 5 s or more of beats newly unusable
    better_songs: int

    @property
    def not_worse(self) -> bool:
        return self.unusable_after_sec <= self.unusable_before_sec + 1e-6


def usable_error(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Per beat of the user's grid, the distance to the tracker's grid where the tracker's local beat period is
    the user's or half of it; where it runs at half tempo, its midpoints count too (an octave is allowed, a
    half-beat shift at the same tempo is not)."""
    ref = np.sort(np.asarray(ref, float))
    est = np.sort(np.asarray(est, float))
    if est.size < 2 or ref.size < 2:
        return np.full(ref.size, np.inf)
    j = np.clip(np.searchsorted(est, ref), 1, est.size - 1)
    e = np.minimum(np.abs(ref - est[j - 1]), np.abs(ref - est[j]))
    q = (est[j] - est[j - 1]) / np.r_[np.diff(ref), np.diff(ref)[-1]]
    half = (q > 1.7) & (q < 2.3)
    return np.where(half, np.minimum(e, np.abs(ref - 0.5 * (est[j - 1] + est[j]))), e)


def _f1(ref, est, tol=0.07):
    ref, est = np.sort(np.asarray(ref, float)), np.sort(np.asarray(est, float))
    if ref.size == 0 or est.size == 0:
        return 0.0
    i = j = m = 0
    while i < ref.size and j < est.size:
        d = est[j] - ref[i]
        if abs(d) <= tol:
            m, i, j = m + 1, i + 1, j + 1
        elif d < -tol:
            j += 1
        else:
            i += 1
    p, r = m / est.size, m / ref.size
    return 2 * p * r / (p + r) if p + r else 0.0


def _score_song(job):
    """Process-worker entry point: one song tracked with both model sets."""
    track, cache_dir_text, bpm_min, paths_before, paths_after = job
    from threadpoolctl import threadpool_limits
    from analyzer_core.beat.general import tracker as TR
    from core.beat_geometry import downbeat_times
    from core.tempo_rows import rebuild_grid_from_segments
    frames = read_frames(cache_path(Path(cache_dir_text), str(track["uid"])), track)
    if frames is None:
        return None
    ref = np.sort(np.asarray(track["beats"], float))
    ref_down = np.asarray(downbeat_times(ref, track["segments"]), float)
    ibi = np.r_[np.diff(ref), np.diff(ref)[-1]]
    out = []
    with threadpool_limits(limits=1):
        for paths in (paths_before, paths_after):
            # an unset path is the shipped model
            models = TR.load_models(*[None if p in (None, "", "None") else p for p in paths])
            res = TR.track(frames, bpm_min, models, None, parallel=False)
            est = rebuild_grid_from_segments(res.segments)
            out.append((float(ibi[usable_error(ref, est) > USABLE_SEC].sum()), _f1(ref_down, res.downbeats)))
    return out


def grid_check(tracks, cache_dir: Path, bpm_min: float, paths_before: tuple, paths_after: tuple,
               max_workers: int | None, progress: ProgressCallback | None, span: tuple[int, int]) -> GridCheck:
    """``paths_*``: (downbeat model, beat TCN, critic) paths of the two model sets."""
    lo, hi = span
    rows = []
    with create_spawn_process_pool(max(1, int(max_workers or 4))) as executor:
        def as_text(paths):
            return tuple(None if p is None else str(p) for p in paths)
        futures = [executor.submit(_score_song, (t, str(cache_dir), float(bpm_min), as_text(paths_before),
                                                 as_text(paths_after))) for t in tracks]
        for done, f in enumerate(as_completed(futures), start=1):
            r = f.result()
            if r is not None:
                rows.append(r)
            if progress:
                progress(lo + int(round((hi - lo) * done / len(futures))), f"Grid check {done}/{len(futures)}")
    if not rows:
        return GridCheck(0, 0.0, 0.0, float("nan"), float("nan"), 0, 0)
    a = np.asarray([r[0] for r in rows])
    b = np.asarray([r[1] for r in rows])
    return GridCheck(
        songs=len(rows),
        unusable_before_sec=float(a[:, 0].sum()), unusable_after_sec=float(b[:, 0].sum()),
        downbeat_f_before=float(a[:, 1].mean()), downbeat_f_after=float(b[:, 1].mean()),
        worse_songs=int(np.sum(b[:, 0] > a[:, 0] + 5.0)), better_songs=int(np.sum(b[:, 0] < a[:, 0] - 5.0)),
    )
