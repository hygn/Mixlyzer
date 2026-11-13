"""Train the beat tracker's frame TCN (analyzer.beat.general.tcn) on the library's grids, in PyTorch.

Each of the trainings the tracker's consensus runs (three seeds) is trained on the user's beats and downbeats with
the recipe of the shipped ones: crops of 3000 frames (about 35 s) stretched by a random tempo factor
(0.8 - 1.25, so the model hears each song at other tempos too), binary cross-entropy on the beat and downbeat frames
(each annotated frame 1, its neighbours 0.5; positives weighted 10 / 30, they are rare), dropout 0.1 on each block's
activation, Adam with the rate warmed up over the first tenth and a cosine decay after, 4000 steps of 16 crops from
scratch, or 1500 at a quarter of the rate from a model's training of its place (``--start shipped`` / ``current``,
the command line's default the model in use): trained from scratch, a ritardando the shipped TCN read as one ramp
came out as two (a GT song's beats 41 ms off), fine-tuned it stays as it was. The
songs' frames sit on the device whole (float16, about 1.5 GB for 500 songs), the crops are cut there; on a CUDA GPU
a training takes about a minute. The weights are exported as the arrays the app's NumPy inference reads
(tcn.save_weights), checked against it on a song before they are written.

With cross-validation, one group of songs is held out first: the model in use and the one trained without them are
compared on the grids the whole tracker gives (training/beat_grid_check.py), and the trainings are run again on every
song and written only when those songs' grids are not worse (keep_if_worse aside). The decoder's costs are in the
activation's evidence units: after a new TCN, refit them (python -m training beat_decoder).

torch is the training group's (uv sync --group training), imported when a training runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.beat.general import tcn as T
from analyzer.music.beat_geometry import downbeat_beat_indices
from training import SkippedOptimizationTrack, skipped_track
from training import model_start as MS
from training.beat_frames import cache_path, ensure_frames, list_grid_tracks, read_frames
from training.beat_grid_check import GridCheck, ModelSet, grid_check
from training.refine_onset_optimizer import _group_key
from training.text import msg

ProgressCallback = Callable[[int, str], None]
FEATURES = 77                 # the frame-feature channels the TCN reads (tcn.frame_inputs)
CHANNELS = 32
DROPOUT = 0.1
STEPS = 4000
RATE = 2e-3
WARMUP = 0.1
CROP = 3000
BATCH = 16
POS_WEIGHT = (10.0, 30.0)
STRETCH = (0.8, 1.25)
TRAININGS = 3
# started from a model (--start shipped / current): each training from that model's training of its place, fewer steps
# at a lower rate, so what it does well stays and it takes in what changed (a feature corrected, new songs)
FINE_TUNE_STEPS = 1500
FINE_TUNE_RATE = RATE / 4
EXPORT_TOLERANCE = 1e-4       # the exported weights' activations against the trained net's, on a song


@dataclass(frozen=True)
class BeatTcnOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    current_models: dict | None = None      # the analysis's model files in use (ModelSet fields)
    bpm_min: float = 100.0
    trainings: int = TRAININGS
    steps: int = STEPS
    learning_rate: float = RATE
    seed: int = 0                    # the first training's seed (the others follow)
    start: str = MS.SCRATCH          # shipped | current | scratch (the command line's default: current)
    start_path: str | None = None    # the model the trainings start from (the app's file; None: from scratch)
    current_path: str | None = None  # the model in use (the app's file)
    fine_tune_steps: int = FINE_TUNE_STEPS
    fine_tune_rate: float = FINE_TUNE_RATE
    cv_folds: int = 0
    keep_if_worse: bool = False
    rebuild_cache: bool = False
    max_cache_workers: int | None = None
    device: str = ""                 # "": CUDA when there is one, else the CPU


@dataclass(frozen=True)
class BeatTcnOptimizationResult:
    output_path: Path
    track_count: int
    trainings: int
    steps: int
    device: str
    seconds_per_training: float
    final_loss: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False
    grid: GridCheck | None = None
    saved: bool = True


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


@dataclass
class TrainingFrames:
    """Every song's frame inputs and targets end to end on the device (float16), ``offsets`` [songs + 1] where each
    starts, ``tracks`` the songs in that order."""
    x: object
    y: object
    offsets: np.ndarray
    tracks: list

    def song(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.x[a:b].float().cpu().numpy(), self.y[a:b].float().cpu().numpy()


def training_frames(tracks, cache_dir: Path, ignored: list, device) -> TrainingFrames:
    """The cached frames of the songs put on ``device`` one by one (the whole library never sits in memory twice)."""
    import torch
    sizes, kept = [], []
    for track in tracks:
        try:
            with np.load(cache_path(cache_dir, str(track["uid"])), allow_pickle=False) as z:
                sizes.append(int(z["frame_times"].size))
        except (OSError, KeyError, ValueError):
            sizes.append(0)
    total = int(sum(sizes))
    x = torch.empty((total, FEATURES), dtype=torch.float16, device=device)
    y = torch.empty((total, 2), dtype=torch.float16, device=device)
    offsets = [0]
    for track, size in zip(tracks, sizes):
        frames = read_frames(cache_path(cache_dir, str(track["uid"])), track) if size else None
        if frames is None or np.asarray(frames.frame_times).size != size:
            ignored.append(skipped_track(track, msg("frame feature cache is not current")))
            continue
        a = offsets[-1]
        x[a:a + size] = torch.from_numpy(T.frame_inputs(frames).astype(np.float16)).to(device)
        yb, yd = track_targets(np.asarray(frames.frame_times, float), track["beats"], track["segments"])
        y[a:a + size] = torch.from_numpy(np.stack([yb, yd], axis=1).astype(np.float16)).to(device)
        offsets.append(a + size)
        kept.append(track)
    end = offsets[-1]
    return TrainingFrames(x[:end], y[:end], np.asarray(offsets, dtype=np.int64), kept)


def input_stats(data: TrainingFrames, songs) -> tuple[np.ndarray, np.ndarray]:
    """The mean and standard deviation of the songs' frame inputs (the network's input normalization)."""
    import torch
    s = torch.zeros(data.x.shape[1], dtype=torch.float64, device=data.x.device)
    s2 = torch.zeros_like(s)
    n = 0
    for i in songs:
        c = data.x[int(data.offsets[i]):int(data.offsets[i + 1])].double()
        s += c.sum(0)
        s2 += (c * c).sum(0)
        n += c.shape[0]
    mu = (s / n).cpu().numpy()
    return mu, np.sqrt(np.maximum((s2 / n).cpu().numpy() - mu * mu, 0.0)) + 1e-6


# --- the network ----------------------------------------------------------------------------------

def network(features: int = FEATURES, channels: int = CHANNELS):
    """The torch module whose state dict is tcn.activations' arrays (same names and shapes)."""
    from torch import nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, T.K, dilation=d, padding=2 * d)
            self.mix = nn.Conv1d(channels, channels, 1)

    class Tcn(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(features, channels, 1)
            self.blocks = nn.ModuleList(Block(d) for d in T.DILATIONS)
            self.out = nn.Conv1d(channels, 2, 1)

        def forward(self, x):              # x [B, F, T] standardized -> logits [B, 2, T]
            h = self.inp(x)
            for b in self.blocks:
                h = h + b.mix(F.dropout(F.elu(b.conv(h)), DROPOUT, self.training))
            return self.out(F.elu(h))

    return Tcn()


def export(net, mu: np.ndarray, sd: np.ndarray) -> dict:
    """The training as tcn.load_weights gives it: the state dict's arrays and the input statistics."""
    P = {k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()}
    P["mu"], P["sd"] = np.asarray(mu, np.float32), np.asarray(sd, np.float32)
    return P


def check_export(net, P: dict, x: np.ndarray, device) -> float:
    """The largest difference between the NumPy inference of the exported arrays and the net, on frames ``x``."""
    import torch
    net.eval()
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False        # full float32 as the NumPy inference (TF32 differs by ~1e-3)
    try:
        with torch.no_grad():
            xs = torch.from_numpy(((x - P["mu"]) / P["sd"]).T[None].astype(np.float32)).to(device)
            p = torch.sigmoid(net(xs))[0].double().cpu().numpy()
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    b, d = T.activations(P, x)
    return float(max(np.abs(p[0] - b).max(), np.abs(p[1] - d).max()))


# --- training -------------------------------------------------------------------------------------

def _rate(step: int, steps: int, warmup: float) -> float:
    """The learning rate's factor: warmed up over the first ``warmup`` share of the steps, then a cosine decay."""
    w = max(1, int(round(warmup * steps)))
    if step <= w:
        return 0.04 + 0.96 * step / w
    return 0.5 * (1.0 + math.cos(math.pi * (step - 1 - w) / max(1, steps - w)))


def _batch(data: TrainingFrames, off, songs, mu, sd, gen):
    """BATCH crops of CROP frames, each from a random song (``songs``) at a random place, stretched by a random tempo
    factor (linear interpolation of the frames; a target is the larger of the two frames it falls between);
    ``off``: data.offsets on the device."""
    import torch
    dev = data.x.device
    pick = songs[torch.randint(len(songs), (BATCH,), generator=gen, device=dev)]
    a, n = off[pick], off[pick + 1] - off[pick]
    lo, hi = math.log(STRETCH[0]), math.log(STRETCH[1])
    s = torch.exp(lo + (hi - lo) * torch.rand(BATCH, generator=gen, device=dev, dtype=torch.float64))
    src = torch.minimum(torch.ceil(CROP * s).long() + 2, n - 1)
    c0 = torch.floor(torch.rand(BATCH, generator=gen, device=dev, dtype=torch.float64)
                     * torch.clamp(n - src, min=1)).long()
    pos = c0[:, None] + torch.arange(CROP, device=dev, dtype=torch.float64)[None, :] * ((src - 1) / (CROP - 1))[:, None]
    i0 = torch.floor(pos).long()
    i1 = torch.minimum(i0 + 1, (n - 1)[:, None])
    fr = (pos - i0).float()[..., None]
    g0, g1 = a[:, None] + i0, a[:, None] + i1
    x = data.x[g0].float() * (1 - fr) + data.x[g1].float() * fr
    y = torch.maximum(data.y[g0], data.y[g1]).float()
    return ((x - mu) / sd).transpose(1, 2).contiguous(), y.transpose(1, 2).contiguous()


def train_one(data: TrainingFrames, songs, seed: int, steps: int = STEPS, rate: float = RATE,
              mu=None, sd=None, progress: Callable[[int], None] | None = None, channels: int = CHANNELS,
              start: dict | None = None):
    """One training on ``songs`` (indices into ``data``), from scratch or from ``start`` (a training as
    tcn.load_weights gives it: its weights, the input statistics made anew on these songs): (net, mu, sd, the last
    100 steps' mean loss)."""
    import torch
    import torch.nn.functional as F
    dev = data.x.device
    if mu is None or sd is None:
        mu, sd = input_stats(data, songs)
    torch.manual_seed(10_000 + seed)
    net = network(data.x.shape[1], channels).to(dev)
    if start is not None:
        net.load_state_dict({k: torch.from_numpy(np.asarray(start[k], np.float32)) for k in net.state_dict()})
    opt = torch.optim.Adam(net.parameters(), lr=rate)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: _rate(i + 1, steps, WARMUP))
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    songs_t = torch.as_tensor(np.asarray(songs), device=dev)
    off = torch.as_tensor(data.offsets, device=dev)
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=dev)
    sd_t = torch.as_tensor(sd, dtype=torch.float32, device=dev)
    pw = torch.as_tensor(POS_WEIGHT, dtype=torch.float32, device=dev)[None, :, None]
    losses = []
    net.train()
    for step in range(1, steps + 1):
        x, y = _batch(data, off, songs_t, mu_t, sd_t, gen)
        loss = F.binary_cross_entropy_with_logits(net(x), y, pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step > steps - 100:
            losses.append(float(loss.detach()))
        if progress is not None and (step % 50 == 0 or step == steps):
            progress(step)
    net.eval()
    return net, mu, sd, float(np.mean(losses)) if losses else float("nan")


def train_trainings(data: TrainingFrames, songs, trainings: int, first_seed: int, steps: int, rate: float,
                    progress: ProgressCallback | None, span: tuple[int, int], label,
                    channels: int = CHANNELS, start: list[dict] | None = None) -> tuple[list[dict], float, float]:
    """The ensemble's trainings one after another on the device, from scratch or each from ``start``'s training of
    its place: (exported trainings, seconds per training, mean final loss). Each export checked against the NumPy
    inference on the first song."""
    lo, hi = span
    nets, losses, took = [], [], []
    mu, sd = input_stats(data, songs)
    x0, _ = data.song(int(songs[0]))
    x0 = x0[: min(x0.shape[0], 6000)]
    for k in range(trainings):
        def step_done(step, k=k):
            if progress:
                frac = (k + step / steps) / trainings
                progress(lo + int(round((hi - lo) * frac)),
                         msg("{stage}: training {k} of {n}, step {done}/{steps}", stage=label, k=k + 1, n=trainings,
                             done=step, steps=steps))
        t = time.time()
        net, _, _, loss = train_one(data, songs, first_seed + k, steps, rate, mu, sd, step_done, channels,
                                    start[k % len(start)] if start else None)
        took.append(time.time() - t)
        P = export(net, mu, sd)
        diff = check_export(net, P, x0, data.x.device)
        if not diff < EXPORT_TOLERANCE:
            raise RuntimeError(f"the exported TCN differs from the trained one by {diff:.2g}")
        nets.append(P)
        losses.append(loss)
    return nets, float(np.mean(took)), float(np.mean(losses))


def device_of(text: str):
    import torch
    if text:
        return torch.device(text)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    device = device_of(request.device)
    ignored: list[SkippedOptimizationTrack] = []
    tracks = list_grid_tracks(library_dir, ignored)
    MS.require_tracks(len(tracks), "scratch")
    tracks = ensure_frames(tracks, cache_dir, ignored, rebuild=bool(request.rebuild_cache),
                           max_workers=request.max_cache_workers, progress=feature_progress, progress_span=(0, 90))
    if feature_progress:
        feature_progress(90, msg("Training frames"))
    data = training_frames(tracks, cache_dir, ignored, device)
    MS.require_tracks(len(data.tracks), "scratch")
    if feature_progress:
        feature_progress(100, msg("Training frames ready"))
    everything = np.arange(len(data.tracks))
    start = T.load_weights(str(request.start_path)) if request.start_path else None
    if start is not None:
        steps, rate = int(request.fine_tune_steps), float(request.fine_tune_rate)
    else:
        steps, rate = int(request.steps), float(request.learning_rate)
    n = int(request.trainings)
    cross_validated = int(request.cv_folds) >= 2
    grid = None
    span = (0, 98)
    if cross_validated:
        groups = np.asarray([_group_key(t) for t in data.tracks])
        k = min(int(request.cv_folds), np.unique(groups).size)
        train, held = next(GroupKFold(n_splits=k).split(everything, groups=groups))
        trial_nets, _, _ = train_trainings(data, train, n, int(request.seed), steps, rate, optimize_progress, (0, 35),
                                           msg("Held-out check"), start=start)
        trial = output_path.with_name(output_path.stem + ".held_out_check.npz")
        T.save_weights(trial, trial_nets)
        try:
            current = ModelSet(**(request.current_models or {}))
            grid = grid_check([data.tracks[i] for i in held], cache_dir, request.bpm_min, current,
                              current.replace(tcn=trial), request.max_cache_workers, optimize_progress, (35, 50))
        finally:
            trial.unlink(missing_ok=True)
        if not grid.not_worse and not request.keep_if_worse:
            if optimize_progress:
                optimize_progress(100, msg("Not saved: the grids got worse on the held-out songs"))
            return BeatTcnOptimizationResult(
                output_path=output_path, track_count=len(data.tracks), trainings=n, steps=steps, device=str(device),
                seconds_per_training=float("nan"), final_loss=float("nan"), skipped_tracks=tuple(ignored),
                cross_validated=True, grid=grid, saved=False)
        span = (50, 98)
    nets, seconds, loss = train_trainings(data, everything, n, int(request.seed), steps, rate, optimize_progress, span,
                                          msg("Training"), start=start)
    T.save_weights(output_path, nets, {
        "source": "the library's beat grids (python -m training beat_tcn, PyTorch)", "trainings": n,
        "seeds": [int(request.seed) + k for k in range(n)], "steps": steps, "learning_rate": rate, "batch": BATCH,
        "crop_frames": CROP, "tracks": len(data.tracks), "date": date.today().isoformat(),
        "cross_validated": cross_validated, "start": request.start if start is not None else "scratch",
    })
    if optimize_progress:
        optimize_progress(100, msg("Done"))
    return BeatTcnOptimizationResult(
        output_path=output_path, track_count=len(data.tracks), trainings=n, steps=steps, device=str(device),
        seconds_per_training=seconds, final_loss=loss, skipped_tracks=tuple(ignored), cross_validated=cross_validated,
        grid=grid)
