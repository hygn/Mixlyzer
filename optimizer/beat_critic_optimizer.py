"""Fine-tune the beat tracker's metronome critic (its fold models) on the library's grids, in NumPy.

The critic hears the music laid out in a candidate grid's time (8 bars, 12 slots per beat) and says whether
the grid is right. Training windows come from the user's grids (right) and from wrong grids made from them:
half / quarter beat shifts, 3:2 / 2:3 / 4:3 / 3:4 tempo readings, slow drifts (+-0.5 %, +-1 %), a stretch of
8 - 32 beats read wrongly and cut back in, and the decoder's own readings with the current beat model (its
real mistakes). A window is right when all its beats are within 50 ms of the user's beats, wrong when four
or more are farther than 70 ms (others are left out). Each fold model starts from the shipped critic, from the
critic in use, or from scratch (optimizer/model_start.py; from scratch: 6000 steps, the rate warmed up over the
first tenth) and is trained with binary cross-entropy and Adam; with cross-validation, one group of songs is held out first and the current
and fine-tuned critics are compared on it -- how often the right grid sounds better than each wrong one, and
the grids the whole tracker gives with each (optimizer/beat_grid_check.py) -- and only when the grids are
not worse is the critic fine-tuned on every song and written.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, as_completed, wait
from dataclasses import dataclass
import math
import multiprocessing
from pathlib import Path
import queue as queue_module
import shutil
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.beat.general import critic as C
from core.concurrency.process_pool import create_spawn_process_pool
from optimizer import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from optimizer.beat_frames import cache_path, ensure_frames, list_grid_tracks, read_frames
from optimizer.beat_grid_check import GridCheck, ModelSet, grid_check
from optimizer import model_start as MS
from optimizer.beat_tcn_optimizer import _rate
from foundation.model_files import resolve_model
from core.workers.analysis_settings import model_file
from optimizer.refine_onset_optimizer import _group_key
from foundation.i18n import tr

ProgressCallback = Callable[[int, str], None]
SCRATCH_STEPS = 6000
SCRATCH_RATE = 2e-3
BATCH = 32
PAIRWISE_SONGS = 40           # songs the pairwise score is reported on (the grid check decides what is saved)
LOCAL_EDIT_SHARE = 0.35
KINDS = ('half', 'q1', 'q3', 'r32', 'r23', 'r43', 'r34', 'd+05', 'd-05', 'd+1', 'd-1')
THREADS_PER_TRAINING = 2


@dataclass(frozen=True)
class BeatCriticOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    start: str = MS.CURRENT          # shipped | current | scratch
    base_path: Path | None = None    # overrides where the fold models start (tests)
    tcn_path: Path | None = None                   # the beat model whose decoder readings are the critic's hard examples
    bpm_min: float = 100.0           # the octave the decoder reads tempos in
    current_models: dict | None = None      # the analysis's model files in use (ModelSet fields)
    steps: int = 0                   # 0: 1500 (improving a critic) or 6000 (from scratch)
    learning_rate: float = 0.0       # 0: 5e-4 or 2e-3
    cv_folds: int = 0
    keep_if_worse: bool = False
    rebuild_cache: bool = False
    max_cache_workers: int | None = None
    max_train_workers: int | None = None


@dataclass(frozen=True)
class BeatCriticOptimizationResult:
    output_path: Path
    track_count: int
    steps: int
    pairwise_before: float = float("nan")
    pairwise_after: float = float("nan")
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False
    grid: GridCheck | None = None
    saved: bool = True


def _save(path, nets) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"f{i}/{k}": v for i, P in enumerate(nets) for k, v in P.items()})


# --- the network's gradients (the forward pass mirrors analyzer.beat.general.critic) ---------------

def _elu_grad(x):
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0))).astype(np.float32)


def _conv3x3_back(W, x, gout, d):
    """Gradients of critic._conv3x3 (padding (1, d), dilation (1, d)): (dW, db, dx)."""
    B, _, H, Wd = x.shape
    xp = np.moveaxis(np.pad(x, ((0, 0), (0, 0), (1, 1), (d, d))), 1, -1)       # [B, H+2, W+2d, I]
    go = np.moveaxis(gout, 1, -1)                                                # [B, H, W, O]
    dW = np.zeros_like(W)
    gxp = np.zeros_like(xp)
    go2 = go.reshape(-1, go.shape[-1])
    for i in range(3):
        for j in range(3):
            patch = xp[:, i:i + H, j * d:j * d + Wd, :]
            dW[:, :, i, j] = go2.T @ patch.reshape(-1, patch.shape[-1])
            gxp[:, i:i + H, j * d:j * d + Wd, :] += go @ W[:, :, i, j]
    dx = np.moveaxis(gxp[:, 1:1 + H, d:d + Wd, :], -1, 1)
    return dW, go2.sum(0), dx


def _forward(P, x):
    """x [B, C + SLOTS, WIN * SLOTS] -> logits [B] and the intermediates for the backward pass."""
    B, Cp, L = x.shape
    img = x.reshape(B, Cp, L // C.BAR, C.BAR)
    feat, pos = img[:, :-C.SLOTS], img[:, -C.SLOTS:]
    a1 = C._conv1x1(P['inst.weight'], P['inst.bias'], feat)
    u = np.concatenate([C._elu(a1), pos], 1)
    h = C._conv1x1(P['inp.weight'], P['inp.bias'], u)
    hs = [h]
    for k, d in enumerate(C.DIL):
        h = h + C._conv3x3(P[f'convs.{k}.weight'], P[f'convs.{k}.bias'], C._elu(h), d)
        hs.append(h)
    z = C._elu(h)
    pooled = np.concatenate([z.mean((2, 3)), z.max((2, 3))], 1)
    return pooled @ P['out.weight'].T[:, 0] + P['out.bias'][0], (feat, a1, u, hs, z, pooled)


def _backward(P, saved, g):
    feat, a1, u, hs, z, pooled = saved
    B, Cc, H, Wd = z.shape
    G = {'out.weight': (g @ pooled)[None, :], 'out.bias': np.asarray([g.sum()], np.float32)}
    gp = g[:, None] * P['out.weight'][0][None, :]
    gz = np.broadcast_to((gp[:, :Cc] / (H * Wd))[:, :, None, None], z.shape).copy()
    flat = z.reshape(B, Cc, -1)
    arg = flat.argmax(-1)
    gzf = gz.reshape(B, Cc, -1)
    np.add.at(gzf, (np.arange(B)[:, None], np.arange(Cc)[None, :], arg), gp[:, Cc:])
    gh = gzf.reshape(z.shape) * _elu_grad(hs[-1])
    for k in reversed(range(len(C.DIL))):
        d = C.DIL[k]
        hp = hs[k]
        dW, db, de = _conv3x3_back(P[f'convs.{k}.weight'], C._elu(hp), gh, d)
        G[f'convs.{k}.weight'], G[f'convs.{k}.bias'] = dW, db
        gh = gh + de * _elu_grad(hp)
    G['inp.weight'] = np.einsum('bohw,bihw->oi', gh, u)[:, :, None, None]
    G['inp.bias'] = gh.sum((0, 2, 3))
    gu = np.einsum('oi,bohw->bihw', P['inp.weight'][:, :, 0, 0], gh)
    ga1 = gu[:, :a1.shape[1]] * _elu_grad(a1)
    G['inst.weight'] = np.einsum('bohw,bihw->oi', ga1, feat)[:, :, None, None]
    G['inst.bias'] = ga1.sum((0, 2, 3))
    return {k: np.asarray(v, np.float32) for k, v in G.items()}


# --- grids and windows -----------------------------------------------------------------------------

def _warp(ref, step, offset=0.0):
    idx = np.arange(offset, ref.size - 1, step)
    return np.interp(idx, np.arange(ref.size), ref)


def wrong_grid(kind, ref):
    if kind == 'half':
        return _warp(ref, 1.0, 0.5)
    if kind == 'q1':
        return _warp(ref, 1.0, 0.25)
    if kind == 'q3':
        return _warp(ref, 1.0, 0.75)
    if kind.startswith('r'):
        return _warp(ref, {'r32': 2.0 / 3.0, 'r23': 1.5, 'r43': 0.75, 'r34': 4.0 / 3.0}[kind], 0.0)
    eps = {'d+05': 0.005, 'd-05': -0.005, 'd+1': 0.01, 'd-1': -0.01}[kind]
    return ref[0] + (ref - ref[0]) * (1.0 + eps)


def local_edit(rng, ref):
    """The user's grid with one stretch of 8 - 32 beats read wrongly and cut back in."""
    n = int(rng.integers(8, 33))
    if ref.size < n + 4:
        return ref
    a = int(rng.integers(1, ref.size - n - 1))
    b = a + n
    what = int(rng.integers(6))
    if what < 2:
        p = (0.5, 0.25, 0.75)[int(rng.integers(3))]
        new = np.interp(np.arange(a, b) + p, np.arange(ref.size), ref)
    else:
        new = _warp(ref[a - 1:b + 1], (2.0 / 3.0, 1.5, 0.75, 4.0 / 3.0)[what - 2], float(rng.uniform(0, 1)))
        new = new[(new > ref[a - 1]) & (new < ref[b])]
    return np.sort(np.r_[ref[:a], new, ref[b:]])


def label(ref, b):
    j = np.clip(np.searchsorted(ref, b), 1, ref.size - 1)
    d = np.minimum(np.abs(ref[j - 1] - b), np.abs(ref[j] - b))
    if np.all(d <= 0.05):
        return 1
    if np.sum(d > 0.07) >= 4:
        return 0
    return -1


def window(feats, t0, hop, g, k0):
    """critic._window on one song's feature rows (local prefix sums: the same slot means)."""
    n = feats.shape[0]
    b = g[k0:k0 + C.WIN + 1]
    Pd = np.diff(b)
    edges = b[:-1, None] + Pd[:, None] * (np.arange(C.SLOTS + 1)[None, :] - 0.5) / C.SLOTS
    lo, hi = edges[:, :-1].ravel(), edges[:, 1:].ravel()
    i0 = np.clip(np.floor((lo - t0) / hop).astype(int), 0, n - 1)
    i1 = np.maximum(np.clip(np.ceil((hi - t0) / hop).astype(int), 1, n), i0 + 1)
    a, z = int(i0.min()), int(i1.max())
    cs = np.vstack([np.zeros((1, feats.shape[1]), np.float64), np.cumsum(np.asarray(feats[a:z], np.float64), 0)])
    f = (cs[i1 - a] - cs[i0 - a]) / (i1 - i0)[:, None]
    pos = np.eye(C.SLOTS, dtype=np.float32)[np.arange(f.shape[0]) % C.SLOTS]
    return np.concatenate([f, pos], 1).astype(np.float32).T


# --- data -------------------------------------------------------------------------------------------

def _decoder_grids(job) -> tuple[str, list[np.ndarray]]:
    """Process-worker entry point: the decoder's grids of one song with each training of the beat model."""
    track, cache_dir_text, tcn_path, bpm_min = job
    from threadpoolctl import threadpool_limits
    from analyzer.beat.general import decode as D, tcn as T
    from analyzer.music.tempo_rows import rebuild_grid_from_segments
    frames = read_frames(cache_path(Path(cache_dir_text), str(track["uid"])), track)
    if frames is None:
        return str(track["uid"]), []
    ft = np.asarray(frames.frame_times, float)
    t0, hop = float(ft[0]), float(np.median(np.diff(ft)))
    grids = []
    with threadpool_limits(limits=1):
        x = T.frame_inputs(frames)
        octave = D.Octave(bpm_min)
        for P in T.load_weights(tcn_path):
            beat, _ = T.activations(P, x)
            dur = t0 + hop * (beat.size - 1)
            off = t0 - 0.5 * hop
            val, fine = D.readers(beat, hop, t0)
            base = D.coarse_tempos(beat, hop, octave)
            rows = D.partition(beat, hop, off, dur, D.with_ratios(base, octave))
            rows = D.partition(beat, hop, off, dur, D.with_ratios(np.r_[D.refit_tempos(rows, beat, hop, off), base], octave))
            rows = D.final_refit(D.final_refit(D.phase_merge(rows, val), val), fine)
            grids.append(rebuild_grid_from_segments(rows))
    return str(track["uid"]), grids


def _write_training_data(tracks, cache_dir: Path, work: Path, ignored) -> tuple[list, list]:
    """Every song's critic input rows (float16, memory-mapped) and its time base."""
    sizes = []
    for track in tracks:
        try:
            with np.load(cache_path(cache_dir, str(track["uid"])), allow_pickle=False) as z:
                sizes.append(int(z["frame_times"].size))
        except Exception:
            sizes.append(0)
    work.mkdir(parents=True, exist_ok=True)
    channels = None
    out = None
    kept, meta, offset = [], [], 0
    for track, size in zip(tracks, sizes):
        frames = read_frames(cache_path(cache_dir, str(track["uid"])), track) if size else None
        if frames is None:
            ignored.append(skipped_track(track, tr("frame feature cache is not current")))
            continue
        track_feats = C.track_features(frames)
        feats = np.diff(track_feats['cs'], axis=0)     # the per-frame rows the prefix sums were made of
        if out is None:
            channels = feats.shape[1]
            out = np.lib.format.open_memmap(work / "x.npy", mode="w+", dtype=np.float16, shape=(int(sum(sizes)), channels))
        n = feats.shape[0]
        out[offset:offset + n] = feats
        meta.append((offset, n, track_feats['t0'], track_feats['hop']))
        offset += n
        kept.append(track)
    if out is not None:
        out.flush()
        del out
    return kept, meta


def _train_one(args) -> dict:
    """Process-worker entry point: one fold model fine-tuned on the given songs."""
    P, work, meta, refs, decs, songs, steps, lr, seed, progress, warmup = args
    from threadpoolctl import threadpool_limits
    x_all = np.load(Path(work) / "x.npy", mmap_mode="r")
    P = {k: np.asarray(v, np.float32).copy() for k, v in P.items()}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(1000 + seed)
    cache: dict = {}

    def grid_of(song, kind):
        key = (song, kind)
        if key not in cache:
            cache[key] = decs[song][int(kind[3:] or 0)] if kind.startswith('dec') else wrong_grid(kind, refs[song])
        return cache[key]

    kinds = list(KINDS) + [f'dec{i}' for i in range(max((len(d) for d in decs), default=0))]
    with threadpool_limits(limits=THREADS_PER_TRAINING):
        for step in range(1, steps + 1):
            xs, ys = [], []
            while len(xs) < BATCH:
                song = int(songs[rng.integers(len(songs))])
                ref = refs[song]
                if rng.random() < LOCAL_EDIT_SHARE:
                    g = local_edit(rng, ref)
                elif rng.random() < 0.5:
                    g = ref
                else:
                    kind = kinds[int(rng.integers(len(kinds)))]
                    if kind.startswith('dec') and int(kind[3:] or 0) >= len(decs[song]):
                        continue
                    g = grid_of(song, kind)
                if g.size < C.WIN + 2:
                    continue
                k0 = int(rng.integers(0, g.size - C.WIN - 1))
                y = label(ref, g[k0:k0 + C.WIN])
                if y < 0:
                    continue
                a, n, t0, hop = meta[song]
                xs.append(window(x_all[a:a + n], t0, hop, g, k0))
                ys.append(y)
            x = np.stack(xs)
            y = np.asarray(ys, np.float32)
            logit, saved = _forward(P, x)
            p = 1.0 / (1.0 + np.exp(-logit))
            G = _backward(P, saved, ((p - y) / y.size).astype(np.float32))
            rate = _rate(lr, step, steps, warmup)
            for k in P:
                m[k] = b1 * m[k] + (1 - b1) * G[k]
                v2[k] = b2 * v2[k] + (1 - b2) * G[k] * G[k]
                P[k] = (P[k] - rate * (m[k] / (1 - b1 ** step)) / (np.sqrt(v2[k] / (1 - b2 ** step)) + eps)).astype(np.float32)
            if progress is not None and (step % 20 == 0 or step == steps):
                progress.put((seed, step))
    return P


def _init_critic(like: dict, seed: int) -> dict:
    """A new fold model (from scratch): weights and biases uniform in +-1 / sqrt(fan-in)."""
    rng = np.random.default_rng(20_000 + seed)
    out = {}
    for k, v in like.items():
        w = like[k.replace('.bias', '.weight')]
        fan_in = int(np.prod(w.shape[1:]))
        b = 1.0 / math.sqrt(fan_in)
        out[k] = rng.uniform(-b, b, v.shape).astype(np.float32)
    return out


def _train_all(nets, work, meta, refs, decs, songs, steps, lr, max_workers, progress, span, label_text, warmup=0.0):
    lo, hi = span
    manager = multiprocessing.get_context("spawn").Manager()
    q = manager.Queue()
    done_steps = {i: 0 for i in range(len(nets))}
    out: list = [None] * len(nets)
    workers = max(1, min(len(nets), int(max_workers or len(nets))))
    try:
        with create_spawn_process_pool(workers) as executor:
            futures = {executor.submit(_train_one, (P, str(work), meta, refs, decs, np.asarray(songs), steps, lr, i, q, warmup)): i
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
                    progress(lo + int(round((hi - lo) * frac)), tr("{stage}: step {done}/{steps} of each fold model").format(stage=label_text, done=min(done_steps.values()), steps=steps))
    finally:
        manager.shutdown()
    return out


def _pairwise_song(job) -> tuple[int, int]:
    """Process-worker entry point: (right grid heard above a wrong one, comparisons) on one song."""
    nets, work, song_meta, ref, decs = job
    from threadpoolctl import threadpool_limits
    a, n, t0, hop = song_meta
    feats = np.asarray(np.load(Path(work) / "x.npy", mmap_mode="r")[a:a + n], np.float32)
    track_feats = dict(cs=np.vstack([np.zeros((1, feats.shape[1]), np.float32),
                                     np.cumsum(feats, 0, dtype=np.float64).astype(np.float32)]), n=n, t0=t0, hop=hop)

    def scored(g):
        if g.size < C.WIN + 2:
            return None
        starts = np.arange(0, g.size - C.WIN - 1, C.HOPB)
        labs = np.array([label(ref, g[s:s + C.WIN]) for s in starts])
        return C.window_logits(nets, track_feats, g), labs
    win = tot = 0
    with threadpool_limits(limits=1):
        right = scored(ref)
        if right is None or not (right[1] == 1).any():
            return 0, 0
        r = float(right[0][right[1] == 1].mean())
        for g in [wrong_grid(k, ref) for k in KINDS] + list(decs):
            w = scored(np.asarray(g))
            if w is None or not (w[1] == 0).any():
                continue
            tot += 1
            win += int(r > float(w[0][w[1] == 0].mean()))
    return win, tot


def _pairwise(nets, work, meta, refs, decs, songs, max_workers=None) -> float:
    """How often the right grid's windows sound better (mean logit) than each wrong grid's wrong windows."""
    win = tot = 0
    with create_spawn_process_pool(max(1, int(max_workers or 4))) as executor:
        jobs = [(nets, str(work), meta[s], refs[s], decs[s]) for s in songs]
        for w, t in executor.map(_pairwise_song, jobs):
            win, tot = win + w, tot + t
    return win / tot if tot else float("nan")


def optimize_beat_critic(
    request: BeatCriticOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> BeatCriticOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Critic model path must end with .npz")
    MS.check_output(output_path)
    scratch = request.start == MS.SCRATCH and request.base_path is None
    tcn_text = model_file(request.tcn_path, "beat_tcn_parameter_path")
    base_path = request.base_path or MS.base_model(output_path, request.start)
    # the critic in use: what the new one is compared with
    reference_path = resolve_model(output_path)
    reference = C.load_weights(str(reference_path))
    base = [_init_critic(reference[0], i) for i in range(len(reference))] if scratch else C.load_weights(str(resolve_model(base_path)))
    ignored: list[SkippedOptimizationTrack] = []
    tracks = list_grid_tracks(library_dir, ignored)
    MS.require_tracks(len(tracks), MS.kind_of(request.start))
    tracks = ensure_frames(tracks, cache_dir, ignored, rebuild=bool(request.rebuild_cache),
                           max_workers=request.max_cache_workers, progress=feature_progress, progress_span=(0, 60))
    work = cache_dir / "_critic_training"
    try:
        tracks, meta = _write_training_data(tracks, cache_dir, work, ignored)
        MS.require_tracks(len(tracks), MS.kind_of(request.start))
        refs = [np.sort(np.asarray(t["beats"], np.float64)) for t in tracks]
        decs: dict[str, list] = {}
        workers = max(1, min(len(tracks), int(request.max_cache_workers or 4)))
        with create_spawn_process_pool(workers) as executor:
            futures = {executor.submit(_decoder_grids, (t, str(cache_dir), tcn_text, float(request.bpm_min))): t for t in tracks}
            for done, f in enumerate(as_completed(futures), start=1):
                uid, grids = f.result()
                decs[uid] = grids
                if feature_progress:
                    feature_progress(60 + int(round(40 * done / len(tracks))),
                                     tr("Decoder readings {done}/{total}").format(done=done, total=len(tracks))
                                     + "\n" + optimizer_track_name(futures[f]))
        decs_list = [decs.get(str(t["uid"]), []) for t in tracks]
        steps = int(request.steps) or (SCRATCH_STEPS if scratch else 1500)
        lr = float(request.learning_rate) or (SCRATCH_RATE if scratch else 5e-4)
        warmup = 0.1 if scratch else 0.0
        everything = np.arange(len(tracks))
        cross_validated = int(request.cv_folds) >= 2
        before = after = float("nan")
        grid = None
        if cross_validated:
            groups = np.asarray([_group_key(t) for t in tracks])
            k = min(int(request.cv_folds), np.unique(groups).size)
            train, held = next(GroupKFold(n_splits=k).split(everything, groups=groups))
            tuned = _train_all(base, work, meta, refs, decs_list, train, steps, lr, request.max_train_workers,
                               optimize_progress, (0, 35), tr("Held-out check"), warmup)
            before = _pairwise(reference, work, meta, refs, decs_list, held[:PAIRWISE_SONGS], request.max_cache_workers)
            after = _pairwise(tuned, work, meta, refs, decs_list, held[:PAIRWISE_SONGS], request.max_cache_workers)
            trial = work / "held_out_critic.npz"
            _save(trial, tuned)
            current = ModelSet(**(request.current_models or {})).replace(tcn=tcn_text, critic=reference_path)
            grid = grid_check([tracks[i] for i in held], cache_dir, request.bpm_min,
                              current, current.replace(critic=trial),
                              request.max_cache_workers, optimize_progress, (35, 50))
            if not grid.not_worse and not request.keep_if_worse:
                if optimize_progress:
                    optimize_progress(100, tr("Not saved: the grids got worse on the held-out songs"))
                return BeatCriticOptimizationResult(
                    output_path=output_path, track_count=len(tracks), steps=steps, pairwise_before=before,
                    pairwise_after=after, skipped_tracks=tuple(ignored), cross_validated=True, grid=grid, saved=False,
                )
            span = (50, 98)
        else:
            span = (0, 98)
        nets = _train_all(base, work, meta, refs, decs_list, everything, steps, lr, request.max_train_workers,
                          optimize_progress, span, tr("Training") if scratch else tr("Fine-tuning"), warmup)
        if not cross_validated:
            sample = everything[: min(PAIRWISE_SONGS, everything.size)]
            before = _pairwise(reference, work, meta, refs, decs_list, sample, request.max_cache_workers)
            after = _pairwise(nets, work, meta, refs, decs_list, sample, request.max_cache_workers)
        _save(output_path, nets)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if optimize_progress:
        optimize_progress(100, tr("Done"))
    return BeatCriticOptimizationResult(
        output_path=output_path, track_count=len(tracks), steps=steps, pairwise_before=before,
        pairwise_after=after, skipped_tracks=tuple(ignored), cross_validated=cross_validated, grid=grid,
    )
