"""Fine-tune the beat tracker's frame TCN (each of its trainings) on the library's grids, in NumPy.

Each training starts from the shipped model, from the model in use, or from scratch (analyzer/training/model_start.py)
and is trained on the user's beats and downbeats with the original recipe: crops of 3000 frames (about 35 s) stretched by a random tempo factor (0.8 - 1.25, so the
model hears each song at other tempos too), binary cross-entropy on the beat and downbeat frames (each
annotated frame and half its neighbours; positives weighted 10 / 30, they are rare), dropout 0.1, Adam (from scratch: 4000 steps, the rate warmed up over the first tenth). The trainings
run in parallel processes on one shared, memory-mapped copy of the training frames.

With cross-validation, one group of songs is held out first: the current and the fine-tuned model are
compared on it -- the frame-level average precision of the activations, and the grids the whole tracker
gives with each (analyzer/training/beat_grid_check.py) -- and only when the grids are not worse is the model
fine-tuned on every song and written.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from datetime import date
import math
import multiprocessing
from pathlib import Path
import queue as queue_module
import shutil
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from analyzer.beat.general import tcn as T
from analyzer.music.beat_geometry import downbeat_beat_indices
from analyzer.runtime import spawn_pool
from analyzer.training import SkippedOptimizationTrack, skipped_track
from analyzer.training.beat_frames import cache_path, ensure_frames, list_grid_tracks, read_frames
from analyzer.training.beat_grid_check import GridCheck, ModelSet, grid_check
from analyzer.training import model_start as MS
from analyzer.training.refine_onset_optimizer import _group_key
from analyzer.training.text import msg

ProgressCallback = Callable[[int, str], None]
DROPOUT = 0.1
SCRATCH_STEPS = 4000
SCRATCH_RATE = 2e-3
CROP = 3000
BATCH = 16
POS_WEIGHT = (10.0, 30.0)
STRETCH = (0.8, 1.25)
THREADS_PER_TRAINING = 4


@dataclass(frozen=True)
class BeatTcnOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    start: str = MS.CURRENT          # shipped | current | scratch
    base_path: Path | None = None    # overrides where the trainings start (tests)
    current_path: str | None = None  # the model in use, what the new one is compared with (the app's file)
    start_path: str | None = None    # the model the trainings start from (the app's file; None: from scratch)
    current_models: dict | None = None      # the analysis's model files in use (ModelSet fields)
    bpm_min: float = 100.0
    steps: int = 0                   # 0: 600 (improving a model) or 4000 (from scratch)
    learning_rate: float = 0.0       # 0: 5e-4 or 2e-3
    cv_folds: int = 0
    keep_if_worse: bool = False
    rebuild_cache: bool = False
    max_cache_workers: int | None = None
    max_train_workers: int | None = None


@dataclass(frozen=True)
class BeatTcnOptimizationResult:
    output_path: Path
    track_count: int
    steps: int
    beat_ap_before: float = float("nan")
    beat_ap_after: float = float("nan")
    downbeat_ap_before: float = float("nan")
    downbeat_ap_after: float = float("nan")
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False
    grid: GridCheck | None = None
    saved: bool = True


# --- the network's gradients (the forward pass is analyzer.beat.general.tcn's) -------------------

def _elu_grad(x):
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0))).astype(np.float32)


def _forward(P, x, rng=None):
    """x [F, T] standardized -> logits [2, T] and the intermediates for the backward pass (``rng``: training,
    with dropout on each block's activations)."""
    h = T._conv1x1(P['inp.weight'], P['inp.bias'], x)
    cache = []
    for i, d in enumerate(T.DILATIONS):
        xp = np.pad(h, ((0, 0), (2 * d, 2 * d)))
        cols = np.concatenate([xp[:, j * d:j * d + h.shape[1]] for j in range(T.K)], axis=0)
        W = P[f'blocks.{i}.conv.weight']
        a = W.transpose(0, 2, 1).reshape(W.shape[0], -1) @ cols + P[f'blocks.{i}.conv.bias'][:, None]
        keep = None if rng is None else ((rng.random(a.shape) >= DROPOUT) / (1.0 - DROPOUT)).astype(np.float32)
        e = T.elu(a) if keep is None else T.elu(a) * keep
        h = h + T._conv1x1(P[f'blocks.{i}.mix.weight'], P[f'blocks.{i}.mix.bias'], e)
        cache.append((i, d, cols, a, e, keep))
    z = T.elu(h)
    return T._conv1x1(P['out.weight'], P['out.bias'], z), (x, cache, h, z)


def _backward(P, saved, gout):
    x, cache, h, z = saved
    G = {'out.weight': (gout @ z.T)[:, :, None], 'out.bias': gout.sum(1)}
    gh = (P['out.weight'][:, :, 0].T @ gout) * _elu_grad(h)
    for i, d, cols, a, e, keep in reversed(cache):
        G[f'blocks.{i}.mix.weight'] = (gh @ e.T)[:, :, None]
        G[f'blocks.{i}.mix.bias'] = gh.sum(1)
        ga = (P[f'blocks.{i}.mix.weight'][:, :, 0].T @ gh) * _elu_grad(a)
        if keep is not None:
            ga = ga * keep
        W = P[f'blocks.{i}.conv.weight']
        O, I, _ = W.shape
        G[f'blocks.{i}.conv.weight'] = (ga @ cols.T).reshape(O, T.K, I).transpose(0, 2, 1)
        G[f'blocks.{i}.conv.bias'] = ga.sum(1)
        gcols = W.transpose(0, 2, 1).reshape(O, -1).T @ ga
        n, pad = ga.shape[1], 2 * d
        gxp = np.zeros((I, n + 2 * pad), dtype=np.float32)
        for j in range(T.K):
            gxp[:, j * d:j * d + n] += gcols[j * I:(j + 1) * I]
        gh = gh + gxp[:, pad:pad + n]
    G['inp.weight'] = (gh @ x.T)[:, :, None]
    G['inp.bias'] = gh.sum(1)
    return G


# --- data -----------------------------------------------------------------------------------------

def _widen(y):
    w = y.copy()
    w[1:] = np.maximum(w[1:], 0.5 * y[:-1])
    w[:-1] = np.maximum(w[:-1], 0.5 * y[1:])
    return w


def track_targets(frame_times, beats, segments):
    """Beat and downbeat frame targets: 1 on the frame of each annotated beat (downbeat), 0.5 beside it."""
    t0 = float(frame_times[0])
    hop = float(np.median(np.diff(frame_times)))
    n = frame_times.size
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    yb = np.zeros(n, np.float32)
    yd = np.zeros(n, np.float32)
    idx = np.round((beats - t0) / hop).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    yb[idx[ok]] = 1.0
    down = downbeat_beat_indices(beats, segments)
    di = idx[down[down < idx.size]]
    yd[di[(di >= 0) & (di < n)]] = 1.0
    return _widen(yb), _widen(yd)


def _write_training_data(tracks, cache_dir: Path, work: Path, ignored) -> list[dict[str, object]]:
    """All songs' frame inputs (float16) and targets in one memory-mapped pair of arrays (filled song by
    song: the whole library's frames never sit in memory at once)."""
    sizes = []
    for track in tracks:
        try:
            with np.load(cache_path(cache_dir, str(track["uid"])), allow_pickle=False) as z:
                sizes.append(int(z["frame_times"].size))
        except Exception:
            sizes.append(0)
    work.mkdir(parents=True, exist_ok=True)
    total = int(sum(sizes))
    x_out = np.lib.format.open_memmap(work / "x.npy", mode="w+", dtype=np.float16, shape=(total, 77))
    y_out = np.lib.format.open_memmap(work / "y.npy", mode="w+", dtype=np.float16, shape=(total, 2))
    kept, offsets = [], [0]
    for track, size in zip(tracks, sizes):
        frames = read_frames(cache_path(cache_dir, str(track["uid"])), track) if size else None
        if frames is None or np.asarray(frames.frame_times).size != size:
            ignored.append(skipped_track(track, msg("frame feature cache is not current")))
            continue
        a = offsets[-1]
        x_out[a:a + size] = T.frame_inputs(frames)
        yb, yd = track_targets(np.asarray(frames.frame_times, float), track["beats"], track["segments"])
        y_out[a:a + size, 0], y_out[a:a + size, 1] = yb, yd
        offsets.append(a + size)
        kept.append(track)
    x_out.flush()
    y_out.flush()
    del x_out, y_out
    np.save(work / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    return kept


def _rate(lr, step, steps, warmup) -> float:
    """The learning rate: warmed up over the first ``warmup`` share of the steps, then a cosine decay."""
    w = int(round(warmup * steps))
    if step <= w:
        return lr * (0.04 + 0.96 * step / w)
    return lr * 0.5 * (1.0 + math.cos(math.pi * (step - 1 - w) / max(1, steps - w)))


def _init_net(mu, sd, seed) -> dict:
    """A new training (from scratch): weights and biases uniform in +-1 / sqrt(fan-in), the input statistics
    of the library's frames."""
    rng = np.random.default_rng(10_000 + seed)
    C, F = 32, mu.size

    def uni(shape, fan_in):
        b = 1.0 / math.sqrt(fan_in)
        return rng.uniform(-b, b, shape).astype(np.float32)
    P = {'inp.weight': uni((C, F, 1), F), 'inp.bias': uni((C,), F)}
    for i in range(len(T.DILATIONS)):
        P[f'blocks.{i}.conv.weight'] = uni((C, C, T.K), C * T.K)
        P[f'blocks.{i}.conv.bias'] = uni((C,), C * T.K)
        P[f'blocks.{i}.mix.weight'] = uni((C, C, 1), C)
        P[f'blocks.{i}.mix.bias'] = uni((C,), C)
    P['out.weight'], P['out.bias'] = uni((2, C, 1), C), uni((2,), C)
    P['mu'], P['sd'] = mu.astype(np.float32), sd.astype(np.float32)
    return P


def _input_stats(work) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(Path(work) / "x.npy", mmap_mode="r")
    s = np.zeros(x.shape[1]); s2 = np.zeros(x.shape[1]); n = 0
    for a in range(0, x.shape[0], 200_000):
        c = np.asarray(x[a:a + 200_000], np.float64)
        s += c.sum(0); s2 += (c * c).sum(0); n += c.shape[0]
    mu = s / n
    return mu, np.sqrt(np.maximum(s2 / n - mu * mu, 0.0)) + 1e-6


def _train_one(args) -> dict:
    """Process-worker entry point: one training fine-tuned on the given songs."""
    P, work, songs, steps, lr, seed, progress, warmup = args
    from threadpoolctl import threadpool_limits
    x_all = np.load(Path(work) / "x.npy", mmap_mode="r")
    y_all = np.load(Path(work) / "y.npy", mmap_mode="r")
    offsets = np.load(Path(work) / "offsets.npy")
    P = {k: np.asarray(v, dtype=np.float32).copy() for k, v in P.items()}
    mu, sd = P.pop("mu"), P.pop("sd")
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    pw = np.asarray(POS_WEIGHT, np.float32)[:, None]
    rng = np.random.default_rng(seed)
    with threadpool_limits(limits=THREADS_PER_TRAINING):
        for step in range(1, steps + 1):
            G = {k: np.zeros_like(v) for k, v in P.items()}
            for _ in range(BATCH):
                song = int(songs[rng.integers(len(songs))])
                a, b = int(offsets[song]), int(offsets[song + 1])
                n = b - a
                s = float(np.exp(rng.uniform(np.log(STRETCH[0]), np.log(STRETCH[1]))))
                src = min(int(math.ceil(CROP * s)) + 2, n - 1)
                c0 = int(rng.integers(0, max(1, n - src)))
                pos = c0 + np.arange(CROP) * (src - 1) / (CROP - 1)
                i0 = np.floor(pos).astype(np.int64)
                i1 = np.minimum(i0 + 1, n - 1)
                fr = (pos - i0)[:, None].astype(np.float32)
                xw = np.asarray(x_all[a + i0], np.float32) * (1 - fr) + np.asarray(x_all[a + i1], np.float32) * fr
                xc = ((xw - mu) / sd).T.astype(np.float32)
                yc = np.maximum(np.asarray(y_all[a + i0], np.float32), np.asarray(y_all[a + i1], np.float32)).T
                logit, saved = _forward(P, xc, rng)
                p = 1.0 / (1.0 + np.exp(-logit))
                g = ((-(pw * yc) * (1 - p) + (1 - yc) * p) / (logit.size * BATCH)).astype(np.float32)
                for k, gk in _backward(P, saved, g).items():
                    G[k] += gk
            rate = _rate(lr, step, steps, warmup)
            for k in P:
                m[k] = b1 * m[k] + (1 - b1) * G[k]
                v2[k] = b2 * v2[k] + (1 - b2) * G[k] * G[k]
                P[k] = (P[k] - rate * (m[k] / (1 - b1 ** step)) / (np.sqrt(v2[k] / (1 - b2 ** step)) + eps)).astype(np.float32)
            if progress is not None and (step % 10 == 0 or step == steps):
                progress.put((seed, step))
    P["mu"], P["sd"] = mu, sd
    return P


def _train_all(nets, work, songs, steps, lr, max_workers, progress, span, label, warmup=0.0) -> list[dict]:
    """Fine-tune every training in parallel processes, reporting the slowest one's progress."""
    lo, hi = span
    manager = multiprocessing.get_context("spawn").Manager()
    q = manager.Queue()
    done_steps = {i: 0 for i in range(len(nets))}
    out: list[dict | None] = [None] * len(nets)
    workers = max(1, min(len(nets), int(max_workers or len(nets))))
    try:
        with spawn_pool(workers) as executor:
            futures = {executor.submit(_train_one, (P, str(work), np.asarray(songs), steps, lr, i, q, warmup)): i
                       for i, P in enumerate(nets)}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
                for f in finished:
                    out[futures[f]] = f.result()
                while True:
                    try:
                        i, s = q.get_nowait()
                    except queue_module.Empty:
                        break
                    done_steps[i] = max(done_steps[i], s)
                if progress:
                    frac = sum(done_steps.values()) / (steps * len(nets))
                    progress(lo + int(round((hi - lo) * frac)), msg("{stage}: step {done}/{steps} of each training", stage=label, done=min(done_steps.values()), steps=steps))
    finally:
        manager.shutdown()
    return out


def _activations(nets, x):
    acts = [T.activations(P, x) for P in nets]
    return np.mean([a[0] for a in acts], axis=0), np.mean([a[1] for a in acts], axis=0)


def _ap(nets, work, songs) -> tuple[float, float]:
    x_all = np.load(Path(work) / "x.npy", mmap_mode="r")
    y_all = np.load(Path(work) / "y.npy", mmap_mode="r")
    offsets = np.load(Path(work) / "offsets.npy")
    beat, down = [], []
    for song in songs:
        a, b = int(offsets[song]), int(offsets[song + 1])
        pb, pd = _activations(nets, np.asarray(x_all[a:b], np.float32))
        y = np.asarray(y_all[a:b], np.float32)
        beat.append(average_precision_score(y[:, 0] >= 1.0, pb))
        if (y[:, 1] >= 1.0).any():
            down.append(average_precision_score(y[:, 1] >= 1.0, pd))
    return float(np.mean(beat)), float(np.mean(down)) if down else float("nan")


def optimize_beat_tcn(
    request: BeatTcnOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> BeatTcnOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Beat TCN model path must end with .npz")
    scratch = request.start == MS.SCRATCH
    reference_path = request.current_path
    reference = T.load_weights(str(reference_path))
    base = None if request.start_path is None else T.load_weights(str(request.start_path))
    ignored: list[SkippedOptimizationTrack] = []
    tracks = list_grid_tracks(library_dir, ignored)
    MS.require_tracks(len(tracks), MS.kind_of(request.start))
    tracks = ensure_frames(tracks, cache_dir, ignored, rebuild=bool(request.rebuild_cache),
                           max_workers=request.max_cache_workers, progress=feature_progress, progress_span=(0, 90))
    work = cache_dir / "_tcn_training"
    try:
        if feature_progress:
            feature_progress(90, msg("Training frames"))
        tracks = _write_training_data(tracks, cache_dir, work, ignored)
        MS.require_tracks(len(tracks), MS.kind_of(request.start))
        if feature_progress:
            feature_progress(100, msg("Training frames ready"))
        steps = int(request.steps) or (SCRATCH_STEPS if base is None else 600)
        lr = float(request.learning_rate) or (SCRATCH_RATE if base is None else 5e-4)
        warmup = 0.1 if base is None else 0.0
        if base is None:
            mu, sd = _input_stats(work)
            base = [_init_net(mu, sd, i) for i in range(len(reference))]
        everything = np.arange(len(tracks))
        cross_validated = int(request.cv_folds) >= 2
        before = after = (float("nan"), float("nan"))
        grid = None
        if cross_validated:
            groups = np.asarray([_group_key(t) for t in tracks])
            k = min(int(request.cv_folds), np.unique(groups).size)
            train, held = next(GroupKFold(n_splits=k).split(everything, groups=groups))
            tuned = _train_all(base, work, train, steps, lr, request.max_train_workers, optimize_progress,
                               (0, 35), msg("Held-out check"), warmup)
            before, after = _ap(reference, work, held), _ap(tuned, work, held)
            trial = work / "held_out_model.npz"
            T.save_weights(trial, tuned)
            current = ModelSet(**(request.current_models or {})).replace(tcn=reference_path)
            grid = grid_check([tracks[i] for i in held], cache_dir, request.bpm_min,
                              current, current.replace(tcn=trial),
                              request.max_cache_workers, optimize_progress, (35, 50))
            if not grid.not_worse and not request.keep_if_worse:
                if optimize_progress:
                    optimize_progress(100, msg("Not saved: the grids got worse on the held-out songs"))
                return BeatTcnOptimizationResult(
                    output_path=output_path, track_count=len(tracks), steps=steps,
                    beat_ap_before=before[0], beat_ap_after=after[0], downbeat_ap_before=before[1],
                    downbeat_ap_after=after[1], skipped_tracks=tuple(ignored), cross_validated=True,
                    grid=grid, saved=False,
                )
            span = (50, 98)
        else:
            span = (0, 98)
        nets = _train_all(base, work, everything, steps, lr, request.max_train_workers, optimize_progress, span,
                          msg("Training") if scratch else msg("Fine-tuning"), warmup)
        if not cross_validated:
            sample = everything[: min(40, everything.size)]
            before, after = _ap(reference, work, sample), _ap(nets, work, sample)
        T.save_weights(output_path, nets, {
            "start": request.start, "started_from": None if scratch else str(request.start_path), "steps": steps, "learning_rate": lr, "batch": BATCH,
            "crop_frames": CROP, "tracks": len(tracks), "date": date.today().isoformat(),
            "cross_validated": cross_validated, "source": "the library's beat grids (Reoptimize)",
        })
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if optimize_progress:
        optimize_progress(100, msg("Done"))
    return BeatTcnOptimizationResult(
        output_path=output_path, track_count=len(tracks), steps=steps,
        beat_ap_before=before[0], beat_ap_after=after[0], downbeat_ap_before=before[1], downbeat_ap_after=after[1],
        skipped_tracks=tuple(ignored), cross_validated=cross_validated, grid=grid,
    )
