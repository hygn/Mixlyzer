"""Refit the beat tracker's decoder costs on the library's grids.

The decoder's costs -- what a new section, a same-tempo half-beat reset and an odd bar must explain before they
are made -- are in evidence units of the beat model's activation, so a retrained model needs them refitted. Each
song is tracked with every combination of the candidate costs (the expensive section decoding once per section
cost, the bars once per beat grid) and scored against the user's grid in time: the beats that are not usable
(farther than 25 ms; an octave allowed, a half-beat shift not) plus the bars whose downbeat is missed (70 ms).
The costs with the least wrong time are kept. Quick mode keeps the section cost (the expensive one) and refits
the half-beat reset and odd-bar costs only. With cross-validation, the costs are chosen on the other songs of
each group and scored on the held-out ones; they are written only when that held-out score is not worse than
the current costs'.
"""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import date
import itertools
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.beat.general import tracker as TR
from core.concurrency.process_pool import create_spawn_process_pool
from optimizer import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from optimizer.beat_frames import cache_path, ensure_frames, list_grid_tracks, read_frames
from optimizer.beat_grid_check import USABLE_SEC, GridCheck, from_scores, missed_bar_time, usable_error
from optimizer.refine_onset_optimizer import _group_key
from optimizer import model_start as MS
from foundation.i18n import tr

ProgressCallback = Callable[[int, str], None]
# candidates: steps of sqrt(2) around the shipped costs
SECTION_COSTS = (8.5, 12.0, 17.0)
HALF_RESET_COSTS = (34.0, 48.0, 68.0, 96.0, 136.0, 192.0, 272.0)
ODD_BAR_COSTS = (14.0, 20.0, 28.0, 40.0, 57.0, 80.0, 113.0)


@dataclass(frozen=True)
class BeatDecoderOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    start: str = MS.CURRENT                # the costs refitted around: shipped | current
    quick: bool = False                    # refit the half-beat reset and odd-bar costs only
    tcn_path: Path | None = None           # the tracker's models the costs are fitted for
    critic_path: Path | None = None
    downbeat_ctx_path: Path | None = None
    refine_onset_path: Path | None = None
    bpm_min: float = 100.0
    cv_folds: int = 0
    rebuild_cache: bool = False
    max_cache_workers: int | None = None
    table_path: Path | None = None         # also write every song's wrong time per cost combination here


@dataclass(frozen=True)
class BeatDecoderOptimizationResult:
    output_path: Path
    track_count: int
    knobs_before: dict
    knobs_after: dict
    wrong_before_sec: float                 # held-out (CV) or all songs: unusable beats + missed bars
    wrong_after_sec: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False
    saved: bool = True
    grid: GridCheck | None = None          # the end-to-end report (held-out songs with CV)


def _score_song(job):
    """Process-worker entry point: one song's wrong time for every combination of the candidate costs
    [section, half_reset, odd_bar] -> (unusable beats, missed bars) in seconds."""
    track, cache_dir_text, paths, refine_path, bpm_min, grid = job
    from threadpoolctl import threadpool_limits
    from analyzer.beat.general import bars as B, critic as C, decode as D, tcn as T
    from analyzer.beat.learned_onset import REFINE_ONSET_PATH, refine_onset_activation
    from analyzer.music.beat_geometry import downbeat_times
    from analyzer.music.tempo_rows import rebuild_grid_from_segments
    sections_c, resets_c, bars_c = grid
    frames = read_frames(cache_path(Path(cache_dir_text), str(track["uid"])), track)
    if frames is None:
        return str(track["uid"]), None
    ref = np.sort(np.asarray(track["beats"], float))
    ref_down = np.asarray(downbeat_times(ref, track["segments"]), float)
    ibi = np.r_[np.diff(ref), np.diff(ref)[-1]]
    out = np.zeros((len(sections_c), len(resets_c), len(bars_c), 2))
    with threadpool_limits(limits=1):
        models = TR.load_models(*[None if p in (None, "", "None") else p for p in paths])
        ft = np.asarray(frames.frame_times, float)
        t0, hop = float(ft[0]), float(np.median(np.diff(ft)))
        refine = (refine_onset_activation(frames, refine_path or REFINE_ONSET_PATH).astype(np.float64), hop, t0)
        x = T.frame_inputs(frames)
        octave = D.Octave(bpm_min)
        critic_track = C.track_features(frames)
        acts = [T.activations(P, x) for P in models.tcn]
        dur = t0 + hop * (acts[0][0].size - 1)
        decoded = {s: [TR.sections(beat, hop, t0, dur, octave, s) for beat, _ in acts] for s in sections_c}
        # Most costs give the same rows (most songs have no same-tempo half-beat reset, nearby section costs
        # often decode alike): everything after the half-beat judgement is computed once per distinct input.
        grids_memo: dict = {}
        prob_memo: dict = {}
        readers = [D.readers(beat, hop, t0)[0] for beat, _ in acts]
        for (i, s), (j, h) in itertools.product(enumerate(sections_c), enumerate(resets_c)):
            knobs = TR.Knobs(section=s, half_reset=h)
            grids = []
            for n, ((beat, down), rows) in enumerate(zip(acts, decoded[s])):
                merged = D.phase_merge(rows, readers[n], h)
                key = (n, s, np.round(np.asarray(merged, float), 9).tobytes())
                if key not in grids_memo:
                    grids_memo[key] = TR.seed_grid(rows, beat, down, t0, hop, dur, octave, models, critic_track, refine, knobs)
                r, b = grids_memo[key]
                pkey = (n, np.round(b, 9).tobytes())
                if pkey not in prob_memo:
                    prob_memo[pkey] = B.downbeat_probability(b, down, t0, hop, frames, models.ctx)
                grids.append((r, b, down, prob_memo[pkey]))
            for k, o in enumerate(bars_c):
                seeds = [TR.SeedResult(np.asarray(r, float), b,
                                       B.downbeats(b, r, down, t0, hop, frames, models.ctx, cost=o, p=p))
                         for r, b, down, p in grids]
                best = seeds[TR.medoid(seeds)]
                est = rebuild_grid_from_segments(B.to_rows(best.rows, best.beats, best.downbeats))
                out[i, j, k, 0] = float(ibi[usable_error(ref, est) > USABLE_SEC].sum())
                out[i, j, k, 1] = missed_bar_time(ref_down, best.downbeats)
    return str(track["uid"]), out


def optimize_beat_decoder(
    request: BeatDecoderOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> BeatDecoderOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".json":
        raise ValueError("Decoder settings path must end with .json")
    MS.check_output(output_path)
    current = TR.load_knobs(str(MS.base_model(output_path, MS.SHIPPED if request.start == MS.SHIPPED else MS.CURRENT)))
    ignored: list[SkippedOptimizationTrack] = []
    tracks = list_grid_tracks(library_dir, ignored)
    MS.require_tracks(len(tracks), "search")
    tracks = ensure_frames(tracks, cache_dir, ignored, rebuild=bool(request.rebuild_cache),
                           max_workers=request.max_cache_workers, progress=feature_progress, progress_span=(0, 100))
    # the candidates around the current costs (the current ones included)
    def around(values, now):
        return tuple(sorted(set(values) | {float(now)}))
    grid = (around(() if request.quick else SECTION_COSTS, current.section), around(HALF_RESET_COSTS, current.half_reset),
            around(ODD_BAR_COSTS, current.odd_bar))
    paths = tuple(None if p is None else str(p) for p in (request.downbeat_ctx_path, request.tcn_path, request.critic_path))
    tables, kept = {}, []
    workers = max(1, int(request.max_cache_workers or 4))
    with create_spawn_process_pool(workers) as executor:
        futures = {executor.submit(_score_song, (t, str(cache_dir), paths,
                                                 None if request.refine_onset_path is None else str(request.refine_onset_path),
                                                 float(request.bpm_min), grid)): t for t in tracks}
        for done, f in enumerate(as_completed(futures), start=1):
            uid, table = f.result()
            if table is None:
                ignored.append(skipped_track(futures[f], tr("frame feature cache is not current")))
            else:
                tables[uid] = table
            if optimize_progress:
                optimize_progress(int(round(90 * done / len(futures))),
                                  tr("Tracking with every cost combination {done}/{total}").format(
                                      done=done, total=len(futures)) + "\n" + optimizer_track_name(futures[f]))
    kept = [t for t in tracks if str(t["uid"]) in tables]
    MS.require_tracks(len(kept), "search")
    full = np.stack([tables[str(t["uid"])] for t in kept])                  # [songs, S, H, O, beats/bars] seconds
    wrong = full.sum(-1)
    if request.table_path:
        np.savez_compressed(request.table_path, uids=np.asarray([str(t["uid"]) for t in kept]),
                            tables=np.stack([tables[str(t["uid"])] for t in kept]),
                            section=np.asarray(grid[0]), half_reset=np.asarray(grid[1]), odd_bar=np.asarray(grid[2]))
    now = (grid[0].index(float(current.section)), grid[1].index(float(current.half_reset)),
           grid[2].index(float(current.odd_bar)))

    def choose(songs):
        total = wrong[songs].sum(0)
        best = np.unravel_index(int(np.argmin(total)), total.shape)
        return now if total[now] <= total[best] + 1e-6 else best           # ties keep the current costs

    cross_validated = int(request.cv_folds) >= 2
    titles = [str(t.get("title") or t["uid"]) for t in kept]
    if cross_validated:
        groups = np.asarray([_group_key(t) for t in kept])
        k = min(int(request.cv_folds), np.unique(groups).size)
        names, b_rows, a_rows = [], [], []
        for train, held in GroupKFold(n_splits=k).split(np.arange(len(kept)), groups=groups):
            c = choose(train)
            names += [titles[i] for i in held]
            b_rows.append(full[held][:, now[0], now[1], now[2]])
            a_rows.append(full[held][:, c[0], c[1], c[2]])
        check = from_scores(names, np.concatenate(b_rows), np.concatenate(a_rows))
    chosen = choose(np.arange(len(kept)))
    if not cross_validated:
        check = from_scores(titles, full[:, now[0], now[1], now[2]], full[:, chosen[0], chosen[1], chosen[2]])
    before, after = check.wrong_before_sec, check.wrong_after_sec
    knobs = TR.Knobs(section=grid[0][chosen[0]], half_reset=grid[1][chosen[1]], odd_bar=grid[2][chosen[2]])
    saved = after <= before + 1e-6
    if saved:
        TR.save_knobs(output_path, knobs, {
            "date": date.today().isoformat(), "tracks": len(kept), "cross_validated": cross_validated,
            "start": request.start, "quick": bool(request.quick),
            "wrong_seconds_before": round(before, 2), "wrong_seconds_after": round(after, 2),
            "candidates": {"section": grid[0], "half_reset": grid[1], "odd_bar": grid[2]},
            "models": {"beat_tcn": paths[1], "critic": paths[2], "downbeat": paths[0]},
        })
    if optimize_progress:
        optimize_progress(100, tr("Done") if saved else tr("Not saved: the held-out songs got worse"))
    from dataclasses import asdict
    return BeatDecoderOptimizationResult(
        output_path=output_path, track_count=len(kept), knobs_before=asdict(current), knobs_after=asdict(knobs),
        wrong_before_sec=before, wrong_after_sec=after, skipped_tracks=tuple(ignored),
        cross_validated=cross_validated, saved=saved, grid=check,
    )
