"""Train the beat tracker's metronome critic (analyzer.beat.general.critic: its fold models) on the library's grids,
in PyTorch.

The critic hears the music laid out in a candidate grid's time (8 bars, 12 slots per beat) and says whether
the grid is right. Training windows come from the user's grids (right) and from wrong grids made from them:
half / quarter beat shifts, 3:2 / 2:3 / 4:3 / 3:4 tempo readings, slow drifts (+-0.5 %, +-1 %), a stretch of
8 - 32 beats read wrongly and cut back in, and the decoder's own readings with the beat model in use (its
real mistakes). A window is right when all its beats are within 50 ms of the user's beats, wrong when four
or more are farther than 70 ms (others are left out). Each fold model starts from the shipped critic, from the
critic in use, or from scratch (training/model_start.py) and is trained with binary cross-entropy and Adam on
batches of 32 windows, the rate on a cosine decay: improving a critic 1500 steps from 5e-4, from scratch 6000 steps
from 2e-3 after a warmup over the first tenth. The windows are cut on the CPU from the songs' rows (memory-mapped
from a work file beside the frame cache), the network runs on the device (a CUDA GPU when there is one); the fold
models are trained one after another, each exported as the arrays the app's NumPy inference reads
(critic.load_weights) and checked against it on a song before it is written.

With cross-validation, one group of songs is held out first and the current and new critics are compared on it --
how often the right grid sounds better than each wrong one, and the grids the whole tracker gives with each
(training/beat_grid_check.py) -- and only when the grids are not worse (keep_if_worse aside) is the critic trained
on every song and written.

torch is the training group's (uv sync --group training), imported when a training runs."""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
import gc
import math
from pathlib import Path
import shutil
import time
import traceback
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.beat.general import critic as C
from analyzer.files import atomic_save_npz
from analyzer.runtime import spawn_pool
from training import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from training import model_start as MS
from training.beat_frames import cache_path, ensure_frames, list_grid_tracks, read_frames
from training.beat_grid_check import GridCheck, ModelSet, grid_check
from training.beat_tcn_optimizer import _rate as _warmup_rate, device_of
from training.refine_onset_optimizer import _group_key
from training.text import msg

ProgressCallback = Callable[[int, str], None]
FEATURES = 81                 # the critic's feature channels (critic.track_features: 77 frame inputs, 4 note events)
INSTRUMENTS = 8
CHANNELS = 32
SCRATCH_STEPS = 6000
SCRATCH_RATE = 2e-3
TUNE_STEPS = 1500
TUNE_RATE = 5e-4
WARMUP = 0.1                  # from scratch only
BATCH = 32
PAIRWISE_SONGS = 40           # songs the pairwise score is reported on (the grid check decides what is saved)
LOCAL_EDIT_SHARE = 0.35
KINDS = ('half', 'q1', 'q3', 'r32', 'r23', 'r43', 'r34', 'd+05', 'd-05', 'd+1', 'd-1')
CHECK_WINDOWS = 64            # the windows of a song each export is checked on
EXPORT_TOLERANCE = 1e-4       # the exported arrays' logits against the trained net's


@dataclass(frozen=True)
class BeatCriticOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    start: str = MS.CURRENT          # shipped | current | scratch
    current_path: str | None = None  # the critic in use, what the new one is compared with (the app's file)
    start_path: str | None = None    # the critic the fold models start from (the app's file; None: from scratch)
    current_models: dict | None = None      # the analysis's model files in use (ModelSet fields); the decoder's
                                            # readings with its beat model (tcn) are the critic's hard examples
    bpm_min: float = 100.0           # the octave the decoder reads tempos in
    steps: int = 0                   # 0: 1500 (improving a critic) or 6000 (from scratch)
    learning_rate: float = 0.0       # 0: 5e-4 or 2e-3
    cv_folds: int = 0
    keep_if_worse: bool = False
    rebuild_cache: bool = False
    max_cache_workers: int | None = None
    device: str = ""                 # "": CUDA when there is one, else the CPU


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
    device: str = ""
    seconds_per_fold: float = float("nan")      # the last training's (the final one, or the held-out check's)
    final_loss: float = float("nan")            # its fold models' mean loss over their last 100 steps


def _save(path, nets) -> None:
    # replaced whole: a cut-off write never leaves a model to read
    atomic_save_npz(path, {f"f{i}/{k}": np.asarray(v, np.float32) for i, P in enumerate(nets) for k, v in P.items()})


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


def sampler(x_all, meta, refs, decs, songs, seed: int) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """A fold model's batches: BATCH windows, each of a random song of ``songs``: from a local edit of its grid
    (LOCAL_EDIT_SHARE), its grid, or one of its wrong grids (the made ones and the decoder's), at a random beat,
    labelled against its grid (windows neither right nor wrong are drawn again)."""
    rng = np.random.default_rng(1000 + seed)
    cache: dict = {}
    kinds = list(KINDS) + [f'dec{i}' for i in range(max((len(d) for d in decs), default=0))]
    songs = np.asarray(songs)

    def grid_of(song, kind):
        key = (song, kind)
        if key not in cache:
            cache[key] = decs[song][int(kind[3:] or 0)] if kind.startswith('dec') else wrong_grid(kind, refs[song])
        return cache[key]

    def batch():
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
        return np.stack(xs), np.asarray(ys, np.float32)

    return batch


def check_windows(x_all, meta, refs, songs) -> np.ndarray:
    """The right grid's windows (every HOPB beats, at most CHECK_WINDOWS) of the first of ``songs`` long enough."""
    for song in songs:
        ref = refs[int(song)]
        if ref.size >= C.WIN + 2:
            a, n, t0, hop = meta[int(song)]
            starts = np.arange(0, ref.size - C.WIN - 1, C.HOPB)[:CHECK_WINDOWS]
            return np.stack([window(x_all[a:a + n], t0, hop, ref, s) for s in starts])
    raise ValueError("no song has the beats of a critic window")


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
            ignored.append(skipped_track(track, msg("frame feature cache is not current")))
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


# --- the network ----------------------------------------------------------------------------------

def network(features: int = FEATURES, instruments: int = INSTRUMENTS, channels: int = CHANNELS):
    """The torch module whose state dict is critic._forward's arrays (same names and shapes)."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.inst = nn.Conv2d(features, instruments, 1)
            self.inp = nn.Conv2d(instruments + C.SLOTS, channels, 1)
            self.convs = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=(1, d), dilation=(1, d))
                                       for d in C.DIL)
            self.out = nn.Linear(2 * channels, 1)

        def forward(self, x):              # x [B, features + SLOTS, WIN * SLOTS] -> logits [B]
            img = x.reshape(x.shape[0], x.shape[1], -1, C.BAR)
            feat, pos = img[:, :-C.SLOTS], img[:, -C.SLOTS:]
            h = self.inp(torch.cat([F.elu(self.inst(feat)), pos], 1))
            for conv in self.convs:
                h = h + conv(F.elu(h))
            z = F.elu(h).flatten(2)
            # the max pooled through the first largest slot (as the NumPy recipe's argmax), not shared among ties
            return self.out(torch.cat([z.mean(2), z.max(2).values], 1))[:, 0]

    return Critic()


def network_of(P: dict, device=None):
    """The torch module holding the arrays ``P`` (a fold model as critic.load_weights gives it)."""
    import torch
    net = network(P['inst.weight'].shape[1], P['inst.weight'].shape[0], P['inp.weight'].shape[0])
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, np.float32).copy()) for k, v in P.items()})
    return net if device is None else net.to(device)


def export(net) -> dict:
    """The fold model as critic.load_weights gives it: the state dict's arrays."""
    return {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()}


def check_export(net, P: dict, x: np.ndarray, device) -> float:
    """The largest difference between the NumPy inference's logits of the exported arrays and the net's, on
    windows ``x``."""
    import torch
    net.eval()
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False        # full float32 as the NumPy inference (TF32 differs by ~1e-3)
    try:
        with torch.no_grad():
            logits = net(torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(device)).double().cpu().numpy()
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    return float(np.abs(logits - C._forward(P, x)).max())


# --- training -------------------------------------------------------------------------------------

def _rate(step: int, steps: int, warmup: float) -> float:
    """The learning rate's factor: warmed up over the first ``warmup`` share of the steps, then a cosine decay (the
    beat TCN's); without a warmup the decay starts at the first step."""
    if warmup > 0:
        return _warmup_rate(step, steps, warmup)
    return 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(1, steps)))


def train_one(P: dict, x_all, meta, refs, decs, songs, steps: int, rate: float, seed: int, warmup: float, device,
              progress: Callable[[int], None] | None = None):
    """One fold model trained on ``songs`` from the arrays ``P`` (its start): (net, the last 100 steps' mean loss).
    Adam's moments as the recipe's (betas 0.9 / 0.999, eps 1e-8, no weight decay), the mean cross-entropy of the
    batch."""
    import torch
    import torch.nn.functional as F
    net = network_of(P, device)
    opt = torch.optim.Adam(net.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: _rate(i + 1, steps, warmup))
    batch = sampler(x_all, meta, refs, decs, songs, seed)
    losses = []
    net.train()
    for step in range(1, steps + 1):
        x, y = batch()      # cut on the CPU while the device still runs the step before
        loss = F.binary_cross_entropy_with_logits(net(torch.from_numpy(x).to(device)), torch.from_numpy(y).to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step > steps - 100:
            losses.append(loss.detach())
        if progress is not None and (step % 20 == 0 or step == steps):
            progress(step)
    net.eval()
    return net, float(torch.stack(losses).mean()) if losses else float("nan")


def train_all(nets, work: Path, meta, refs, decs, songs, steps: int, rate: float, warmup: float, device,
              progress: ProgressCallback | None, span: tuple[int, int], label_text) -> tuple[list[dict], float, float]:
    """The fold models one after another on the device, each from its start in ``nets``: (exported fold models,
    seconds per fold model, mean final loss). Each export checked against the NumPy inference on a song."""
    lo, hi = span
    x_all = np.load(Path(work) / "x.npy", mmap_mode="r")
    x0 = check_windows(x_all, meta, refs, songs)
    out, losses, took = [], [], []
    for k, P in enumerate(nets):
        def step_done(step, k=k):
            if progress:
                frac = (k + step / steps) / len(nets)
                progress(lo + int(round((hi - lo) * frac)),
                         msg("{stage}: fold model {k} of {n}, step {done}/{steps}", stage=label_text, k=k + 1,
                             n=len(nets), done=step, steps=steps))
        t = time.time()
        net, loss = train_one(P, x_all, meta, refs, decs, songs, steps, rate, k, warmup, device, step_done)
        took.append(time.time() - t)
        Q = export(net)
        diff = check_export(net, Q, x0, device)
        if not diff < EXPORT_TOLERANCE:
            raise RuntimeError(f"the exported critic differs from the trained one by {diff:.2g}")
        out.append(Q)
        losses.append(loss)
    del x_all
    return out, float(np.mean(took)), float(np.mean(losses))


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
    with spawn_pool(max(1, int(max_workers or 4))) as executor:
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
    if request.current_path is None:
        raise ValueError("the critic in use (current_path) is what the new one is compared with: it must be given")
    current = ModelSet(**(request.current_models or {})).replace(critic=request.current_path)
    if current.tcn is None:
        raise ValueError("the beat model in use (current_models' tcn) gives the decoder's readings: it must be given")
    scratch = request.start_path is None
    reference_path = str(request.current_path)
    reference = C.load_weights(reference_path)
    base = [_init_critic(reference[0], i) for i in range(len(reference))] if scratch else C.load_weights(str(request.start_path))
    device = device_of(request.device)
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
        with spawn_pool(workers) as executor:
            futures = {executor.submit(_decoder_grids, (t, str(cache_dir), current.tcn, float(request.bpm_min))): t
                       for t in tracks}
            for done, f in enumerate(as_completed(futures), start=1):
                uid, grids = f.result()
                decs[uid] = grids
                if feature_progress:
                    feature_progress(60 + int(round(40 * done / len(tracks))),
                                     (msg("Decoder readings {done}/{total}", done=done, total=len(tracks)),
                                      optimizer_track_name(futures[f])))
        decs_list = [decs.get(str(t["uid"]), []) for t in tracks]
        steps = int(request.steps) or (SCRATCH_STEPS if scratch else TUNE_STEPS)
        lr = float(request.learning_rate) or (SCRATCH_RATE if scratch else TUNE_RATE)
        warmup = WARMUP if scratch else 0.0
        everything = np.arange(len(tracks))
        cross_validated = int(request.cv_folds) >= 2
        before = after = float("nan")
        grid = None
        if cross_validated:
            groups = np.asarray([_group_key(t) for t in tracks])
            k = min(int(request.cv_folds), np.unique(groups).size)
            train, held = next(GroupKFold(n_splits=k).split(everything, groups=groups))
            tuned, seconds, loss = train_all(base, work, meta, refs, decs_list, train, steps, lr, warmup, device,
                                             optimize_progress, (0, 35), msg("Held-out check"))
            before = _pairwise(reference, work, meta, refs, decs_list, held[:PAIRWISE_SONGS], request.max_cache_workers)
            after = _pairwise(tuned, work, meta, refs, decs_list, held[:PAIRWISE_SONGS], request.max_cache_workers)
            trial = work / "held_out_critic.npz"
            _save(trial, tuned)
            grid = grid_check([tracks[i] for i in held], cache_dir, request.bpm_min,
                              current, current.replace(critic=trial),
                              request.max_cache_workers, optimize_progress, (35, 50))
            if not grid.not_worse and not request.keep_if_worse:
                if optimize_progress:
                    optimize_progress(100, msg("Not saved: the grids got worse on the held-out songs"))
                return BeatCriticOptimizationResult(
                    output_path=output_path, track_count=len(tracks), steps=steps, pairwise_before=before,
                    pairwise_after=after, skipped_tracks=tuple(ignored), cross_validated=True, grid=grid, saved=False,
                    device=str(device), seconds_per_fold=seconds, final_loss=loss,
                )
            span = (50, 98)
        else:
            span = (0, 98)
        nets, seconds, loss = train_all(base, work, meta, refs, decs_list, everything, steps, lr, warmup, device,
                                        optimize_progress, span, msg("Training") if scratch else msg("Fine-tuning"))
        if not cross_validated:
            sample = everything[: min(PAIRWISE_SONGS, everything.size)]
            before = _pairwise(reference, work, meta, refs, decs_list, sample, request.max_cache_workers)
            after = _pairwise(nets, work, meta, refs, decs_list, sample, request.max_cache_workers)
        _save(output_path, nets)
    except BaseException as exc:
        # the training's frames let go of the memory-mapped rows: Windows deletes no file that is mapped
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gc.collect()            # the rows' last references (reference cycles) gone before the work file is removed
        shutil.rmtree(work, ignore_errors=True)
    if optimize_progress:
        optimize_progress(100, msg("Done"))
    return BeatCriticOptimizationResult(
        output_path=output_path, track_count=len(tracks), steps=steps, pairwise_before=before,
        pairwise_after=after, skipped_tracks=tuple(ignored), cross_validated=cross_validated, grid=grid,
        device=str(device), seconds_per_fold=seconds, final_loss=loss,
    )
