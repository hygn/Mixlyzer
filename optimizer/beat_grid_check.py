"""The end-to-end check of a retrained beat model: the whole analysis with the current and with the new models on
songs the new one was not trained on, scored against the user's grids. A model-level score cannot see what
matters to the user -- small changes of a model flip the tracker's discrete choices (a half-beat phase, a tempo
reading) on fragile songs -- so a new model is kept only when the grids it gives are not worse.

Scores (seconds of the songs): beats that are not usable (a beat of the user's grid farther than 25 ms from the
analysis's; half or double tempo allowed, a half-beat shift at the same tempo not) and bars whose downbeat is
missed (farther than 70 ms)."""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from core.concurrency.process_pool import create_spawn_process_pool
from optimizer.beat_frames import cache_path, read_frames
from foundation.i18n import tr

ProgressCallback = Callable[[int, str], None]
USABLE_SEC = 0.025
DOWNBEAT_SEC = 0.07
CHANGE_SEC = 1.0                  # a song is listed as better / worse when it changes by more than this


@dataclass(frozen=True)
class ModelSet:
    """The beat analysis's model files (None: the configured default)."""
    downbeat_ctx: str | None = None
    tcn: str | None = None
    critic: str | None = None
    decoder: str | None = None
    refine_onset: str | None = None

    def replace(self, **kw) -> "ModelSet":
        values = {**self.__dict__, **{k: (None if v is None else str(v)) for k, v in kw.items()}}
        return ModelSet(**values)

    def files(self) -> tuple[str, str, str, str]:
        """The files the tracker reads: (downbeat, TCN, critic, decoder)."""
        from core.workers.analysis_settings import model_file
        return (model_file(self.downbeat_ctx, "downbeat_ctx_parameter_path"),
                model_file(self.tcn, "beat_tcn_parameter_path"),
                model_file(self.critic, "beat_critic_parameter_path"),
                model_file(self.decoder, "beat_decoder_parameter_path"))

    def refine_onset_file(self) -> str:
        from core.workers.analysis_settings import model_file
        return model_file(self.refine_onset, "refine_onset_parameter_path")


@dataclass(frozen=True)
class SongChange:
    title: str
    beats_before: float
    beats_after: float
    bars_before: float
    bars_after: float

    @property
    def change(self) -> float:
        return (self.beats_after + self.bars_after) - (self.beats_before + self.bars_before)


@dataclass(frozen=True)
class GridCheck:
    songs: int
    beats_before_sec: float
    beats_after_sec: float
    bars_before_sec: float
    bars_after_sec: float
    changes: tuple[SongChange, ...] = field(default=())

    @property
    def wrong_before_sec(self) -> float:
        return self.beats_before_sec + self.bars_before_sec

    @property
    def wrong_after_sec(self) -> float:
        return self.beats_after_sec + self.bars_after_sec

    @property
    def not_worse(self) -> bool:
        return self.wrong_after_sec <= self.wrong_before_sec + 1e-6

    @property
    def better(self) -> list[SongChange]:
        return sorted((c for c in self.changes if c.change < -CHANGE_SEC), key=lambda c: c.change)

    @property
    def worse(self) -> list[SongChange]:
        return sorted((c for c in self.changes if c.change > CHANGE_SEC), key=lambda c: -c.change)


def from_scores(titles, before: np.ndarray, after: np.ndarray) -> GridCheck:
    """A check from per-song [beats, bars] seconds of the two model sets."""
    before, after = np.asarray(before, float).reshape(-1, 2), np.asarray(after, float).reshape(-1, 2)
    changes = tuple(SongChange(str(t), beats_before=float(b[0]), beats_after=float(a[0]), bars_before=float(b[1]),
                               bars_after=float(a[1]))
                    for t, b, a in zip(titles, before, after) if abs(a.sum() - b.sum()) > CHANGE_SEC)
    return GridCheck(songs=len(before), beats_before_sec=float(before[:, 0].sum()), beats_after_sec=float(after[:, 0].sum()),
                     bars_before_sec=float(before[:, 1].sum()), bars_after_sec=float(after[:, 1].sum()), changes=changes)


def usable_error(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Per beat of the user's grid, the distance to the analysis's grid where its local beat period is the user's
    or half of it; where it runs at half tempo, its midpoints count too (an octave is allowed, a half-beat shift
    at the same tempo is not)."""
    ref = np.sort(np.asarray(ref, float))
    est = np.sort(np.asarray(est, float))
    if est.size < 2 or ref.size < 2:
        return np.full(ref.size, np.inf)
    j = np.clip(np.searchsorted(est, ref), 1, est.size - 1)
    e = np.minimum(np.abs(ref - est[j - 1]), np.abs(ref - est[j]))
    q = (est[j] - est[j - 1]) / np.r_[np.diff(ref), np.diff(ref)[-1]]
    half = (q > 1.7) & (q < 2.3)
    return np.where(half, np.minimum(e, np.abs(ref - 0.5 * (est[j - 1] + est[j]))), e)


def missed_bar_time(ref_down: np.ndarray, est_down: np.ndarray) -> float:
    ref_down = np.sort(np.asarray(ref_down, float))
    if ref_down.size < 2:
        return 0.0
    bar = np.r_[np.diff(ref_down), np.diff(ref_down)[-1]]
    est = np.sort(np.asarray(est_down, float))
    if est.size == 0:
        return float(bar.sum())
    j = np.clip(np.searchsorted(est, ref_down), 1, max(1, est.size - 1))
    d = np.minimum(np.abs(est[j - 1] - ref_down), np.abs(est[np.minimum(j, est.size - 1)] - ref_down))
    return float(bar[d > DOWNBEAT_SEC].sum())


def song_scores(track, frames, result) -> tuple[float, float]:
    """(unusable beats, missed bars) in seconds of a tracker result against the song's grid."""
    from analyzer.music.beat_geometry import downbeat_times
    from analyzer.music.tempo_rows import rebuild_grid_from_segments
    ref = np.sort(np.asarray(track["beats"], float))
    ibi = np.r_[np.diff(ref), np.diff(ref)[-1]]
    est = rebuild_grid_from_segments(result.segments)
    beats = float(ibi[usable_error(ref, est) > USABLE_SEC].sum())
    return beats, missed_bar_time(np.asarray(downbeat_times(ref, track["segments"]), float), result.downbeats)


def track_song(frames, models: ModelSet, bpm_min: float):
    """The full analysis's beat tracking of one song with a model set."""
    from analyzer.beat.general import tracker as TR
    from analyzer.beat.learned_onset import refine_onset_activation
    loaded = TR.load_models(*models.files())
    ft = np.asarray(frames.frame_times, float)
    refine = (refine_onset_activation(frames, models.refine_onset_file()).astype(np.float64),
              float(np.median(np.diff(ft))), float(ft[0]))
    return TR.track(frames, bpm_min, loaded, refine, parallel=False)


def _score_song(job):
    """Process-worker entry point: one song tracked with both model sets."""
    track, cache_dir_text, bpm_min, before, after = job
    from threadpoolctl import threadpool_limits
    frames = read_frames(cache_path(Path(cache_dir_text), str(track["uid"])), track)
    if frames is None:
        return None
    with threadpool_limits(limits=1):
        return str(track.get("title") or track["uid"]), [song_scores(track, frames, track_song(frames, m, bpm_min))
                                                         for m in (before, after)]


def grid_check(tracks, cache_dir: Path, bpm_min: float, before: ModelSet, after: ModelSet,
               max_workers: int | None, progress: ProgressCallback | None, span: tuple[int, int]) -> GridCheck:
    lo, hi = span
    rows = []
    with create_spawn_process_pool(max(1, int(max_workers or 4))) as executor:
        futures = [executor.submit(_score_song, (t, str(cache_dir), float(bpm_min), before, after)) for t in tracks]
        for done, f in enumerate(as_completed(futures), start=1):
            r = f.result()
            if r is not None:
                rows.append(r)
            if progress:
                progress(lo + int(round((hi - lo) * done / len(futures))), tr("End-to-end check {done}/{total}").format(done=done, total=len(futures)))
    if not rows:
        return GridCheck(0, 0.0, 0.0, 0.0, 0.0)
    return from_scores([r[0] for r in rows], [r[1][0] for r in rows], [r[1][1] for r in rows])
