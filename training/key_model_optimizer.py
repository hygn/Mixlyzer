"""Retrain the chord-aware key model (analyzer/key/chord_key.py) on the keys the user checked, in PyTorch.

Training songs are the library's tracks whose key annotation is the user's:
- a track with the automatic key result kept beside it (``key_segments_auto``, written by the full analysis and a
  whole-track key reanalysis) is used when its keys differ from it -- the user went through that track's keys;
  a track whose keys are still the automatic result is left out (the model would learn its own output);
- a track analyzed before the automatic result was kept is used unless its annotation has the shape of the old
  analyzer's raw output (10+ segments, or short fragments with same-key neighbours).
Only keys train the model: no chord labels (users cannot record them). The chord layer stays a chord layer through
the template-chord term (KL to the fixed triad templates), as when the shipped model was trained.

The network (the root-equivariant chord CNN: one set of weights read at each of the 12 rotations of the chroma; the
fixed chord-in-key tables) is trained with Adam (L2 1e-4 on every weight) on random crops of 384 beats, 16 songs a
step: cross-entropy of every bar's key (the sum of its beats' key evidence) plus the template-chord term; 15 epochs
at 1e-3 with a cosine decay when it improves a model, 40 at 3e-3 warmed up over the first 30 % from scratch (the
new weights uniform in +-1 / sqrt(fan-in)). It starts from the model in use, the shipped model or from scratch
(training/model_start.py). The decoder settings (key-change costs) and the chord-in-key tables are kept. The songs'
beat inputs are made once (the frame chroma cached per song, about 4 s a song a worker; pooled on the beats) and the
crops cut on the CPU; the network runs on the GPU when there is one, in full float32 (a step about 15 ms: a training
from scratch on 240 songs in 10 s). The weights are exported as the arrays the analyzer's NumPy inference reads,
checked against it on a song (the chord posteriors and the key evidence) before they are used. With cross-validation
one group of songs is held out first and the current and the retrained model -- the whole key analysis, decoder
included -- are compared on it; only when the held-out keys are not worse is the model retrained on every song and
written.

torch is the training group's (uv sync --group training), imported when a training runs."""

from __future__ import annotations

from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.files import atomic_save_npz
from analyzer.key import chord_key as CK
from analyzer.music.beat_geometry import downbeat_times
from analyzer.runtime import spawn_pool
from training import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from training import model_start as MS
from training.beat_tcn_optimizer import _rate as warmup_cosine, device_of
from training.onset_training import _auto_cache_workers, _list_beat_tracks
from training.refine_onset_optimizer import _group_key
from training.text import msg

ProgressCallback = Callable[[int, str], None]
KEY_FRAME_CACHE_FORMAT = "mixlyzer_key_frame_cache_v1"
CROP = 384
BATCH = 16
WEIGHT_DECAY = 1e-4
SCRATCH_EPOCHS, TUNE_EPOCHS = 40, 15
SCRATCH_RATE, TUNE_RATE = 3e-3, 1e-3
SCRATCH_WARMUP = 0.3
TEMPLATE_TEMP, TEMPLATE_BASS = 0.08, 0.6
TRAINED = ("c1_w", "c1_b", "c2_w", "c2_b", "out_w", "out_b")    # the weights trained (chord_key.WEIGHTS but tab)
EXPORT_TOLERANCE = 1e-4       # the exported arrays' chord posteriors and key evidence against the trained net's


@dataclass(frozen=True)
class KeyModelOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    start: str = MS.CURRENT          # shipped | current | scratch
    cv_folds: int = 0
    epochs: int = 0                  # 0: 15 (improving a model) or 40 (from scratch)
    learning_rate: float = 0.0       # 0: 1e-3 or 3e-3
    distill: float = 2.0             # weight of the template-chord term
    keep_if_worse: bool = False
    rebuild_cache: bool = False
    max_cache_workers: int | None = None
    seed: int = 0
    current_path: str | None = None  # the model in use, what the new one is compared with (the app's file)
    start_path: str | None = None    # the model the training starts from (the app's file; None: from scratch)
    device: str = ""                 # "": CUDA when there is one, else the CPU


@dataclass(frozen=True)
class KeyModelOptimizationResult:
    output_path: Path
    track_count: int                 # training songs
    edited_count: int                # of them: keys differ from the kept automatic result
    legacy_count: int                # of them: no automatic result kept, accepted by the annotation's shape
    epochs: int
    accuracy_before: float = float("nan")   # share of the songs' beats whose key (signature) is right
    accuracy_after: float = float("nan")
    fifth_before: float = float("nan")      # share of beats off by a fifth (an Alphanumeric neighbour)
    fifth_after: float = float("nan")
    eval_songs: int = 0
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False
    saved: bool = True
    steps: int = 0                   # of the last training
    device: str = ""
    seconds_per_step: float = float("nan")


# --- which annotations are the user's ---------------------------------------------------------------

def _rows(ks) -> np.ndarray:
    a = np.asarray(ks, dtype=float)
    if a.size == 0:
        return np.zeros((0, 4))
    a = a.reshape(-1, a.shape[-1])[:, :4]
    return a[np.argsort(a[:, 2])]


def _same_keys(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.all(a[:, :2] == b[:, :2]) and np.allclose(a[:, 2:], b[:, 2:], atol=1e-3))


def raw_looking(ks: np.ndarray) -> bool:
    """The shape of the old analyzer's raw output: many segments, or short fragments next to same-key rows."""
    n = ks.shape[0]
    short = int(np.sum((ks[:, 3] - ks[:, 2]) < 4.0))
    keys = ks[:, 0] + 12 * ks[:, 1]
    same_adj = int(np.sum(keys[1:] == keys[:-1]))
    return n >= 10 or (same_adj >= 2 and short >= 2) or (n >= 6 and short >= 2)


def list_key_tracks(library_dir: Path, ignored: list[SkippedOptimizationTrack]) -> list[dict[str, object]]:
    """The library's tracks whose keys are the user's, with what training reads (beats, bars, phrases, keys)."""
    tracks = []
    for track in _list_beat_tracks(library_dir, ignored):
        a = track["annotation"]        # the track's annotation the library's owner gave (training.library_tracks)
        try:
            ks = _rows(a.key_segments) if a.key_segments is not None else np.zeros((0, 4))
            auto = _rows(a.key_segments_auto) if a.key_segments_auto is not None else None
            if a.tempo_segments is None or a.duration_sec is None:
                raise ValueError("no tempo rows" if a.tempo_segments is None else "no duration")
            segs = np.asarray(a.tempo_segments, dtype=np.float64)
            starts = np.asarray(a.phrase_starts, float) if a.phrase_starts is not None else np.zeros(0)
            duration = float(a.duration_sec)
        except Exception as exc:
            ignored.append(skipped_track(track, msg("unreadable analysis: {error}", error=exc)))
            continue
        if ks.shape[0] == 0:
            ignored.append(skipped_track(track, msg("no key annotation")))
            continue
        if auto is not None:
            if _same_keys(ks, auto):
                continue                       # still the automatic result: not a training song (no message)
            track["kind"] = "edited"
        elif raw_looking(ks):
            continue                           # looks like the old analyzer's raw output
        else:
            track["kind"] = "legacy"
        track.update(keys=ks, segments=segs, phrase_starts=starts, duration=duration)
        tracks.append(track)
    return tracks


# --- the frame chroma cache ------------------------------------------------------------------------

def cache_path(cache_dir: Path, uid: str) -> Path:
    return Path(cache_dir) / f"{uid}.npz"


def read_key_frames(path: Path, track: dict[str, object]) -> np.ndarray | None:
    if not Path(path).is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(np.asarray(z["cache_format"]).item()) != KEY_FRAME_CACHE_FORMAT:
                return None
            if int(np.asarray(z["audio_mtime_ns"]).item()) != Path(track["audio_path"]).stat().st_mtime_ns:
                return None
            return np.asarray(z["frames"], np.float32)
    except Exception:
        return None


def build_key_frames(track: dict[str, object], cache_dir_text: str) -> str:
    """Process-worker entry point: the frame chroma the full analysis's key model reads."""
    from threadpoolctl import threadpool_limits
    from analyzer.audio.load import fast_load
    from analyzer.audio.hpss import hpss_audio_with_spectra

    path = cache_path(Path(cache_dir_text), str(track["uid"]))
    with threadpool_limits(limits=1):
        samp = np.ascontiguousarray(fast_load(str(track["audio_path"]), target_sr=CK.SR, stereo=True).mean(axis=1),
                                    dtype=np.float32)
        y_harm = hpss_audio_with_spectra(samp, CK.SR)[0]
        frames = CK.frame_chroma(y_harm, CK.SR)
    atomic_save_npz(path, dict(cache_format=np.asarray(KEY_FRAME_CACHE_FORMAT),
                               audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns),
                               frames=frames.astype(np.float16)))
    return str(track["uid"])


def ensure_key_frames(tracks, cache_dir: Path, ignored, *, rebuild=False, max_workers=None, progress=None,
                      progress_span=(0, 100)):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [t for t in tracks if rebuild or read_key_frames(cache_path(cache_dir, str(t["uid"])), t) is None]
    failed: set[str] = set()
    if jobs:
        lo, hi = progress_span
        with spawn_pool(_auto_cache_workers(max_workers, len(jobs))) as executor:
            futures = {executor.submit(build_key_frames, t, str(cache_dir)): t for t in jobs}
            for done, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.add(str(track["uid"]))
                    ignored.append(skipped_track(track, msg("key feature cache build failed: {error}", error=exc)))
                if progress:
                    progress(lo + int(round((hi - lo) * done / len(jobs))),
                             (msg("Key features {done}/{total}", done=done, total=len(jobs)), optimizer_track_name(track)))
    return [t for t in tracks if str(t["uid"]) not in failed]


# --- training data ---------------------------------------------------------------------------------

@dataclass
class Song:
    x: np.ndarray        # [nb, 36] beat inputs
    y: np.ndarray        # [nb] key index (pc + 12 * minor)
    w: np.ndarray        # [nb] 1 where the beat has a key
    bar: np.ndarray      # [nb] bar index
    pos: np.ndarray      # [nb] phrase position
    teach: np.ndarray    # [nb, 24] template chord posteriors
    grid: CK.BeatGrid
    duration: float


def _templates() -> np.ndarray:
    T = np.zeros((24, 12), np.float32)
    for r in range(12):
        T[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1
        T[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1
    return T / np.linalg.norm(T, axis=1, keepdims=True)


def template_chords(x: np.ndarray) -> np.ndarray:
    """The fixed triad templates' chord posteriors per beat (bass and treble CQT chroma)."""
    bas, tre = x[:, :12], x[:, 12:24]
    sc = (tre / (np.linalg.norm(tre, axis=1, keepdims=True) + 1e-9)) @ _templates().T
    sc = sc + TEMPLATE_BASS * np.c_[bas, bas] / (bas.sum(1, keepdims=True) + 1e-9)
    sc = (sc - sc.max(1, keepdims=True)) / TEMPLATE_TEMP
    p = np.exp(sc)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def song_of(track: dict[str, object], frames: np.ndarray) -> Song | None:
    beats = np.asarray(track["beats"], float)
    segs = np.asarray(track["segments"], float)
    grid = CK.beat_grid(beats, downbeat_times(beats, segs) if segs.ndim == 2 and segs.shape[0] else None,
                        track["phrase_starts"], float(track["duration"]))
    if grid.beat_t.size < 16:
        return None
    x = CK.pool_beats(frames, grid, float(track["duration"]))
    ks = track["keys"]
    y = np.zeros(grid.beat_t.size, np.int64)
    w = np.zeros(grid.beat_t.size, np.float32)
    for p, m, t0, t1 in ks:
        sel = (grid.beat_t + 1e-3 >= t0) & (grid.beat_t + 1e-3 < t1)
        y[sel] = int(p) % 12 + 12 * (int(m) % 2)
        w[sel] = 1.0
    return Song(x=x, y=y, w=w, bar=grid.bar, pos=grid.pos, teach=template_chords(x), grid=grid,
                duration=float(track["duration"]))


def _batch(songs, idx, rng):
    xs, ys, ws, bs, ps, ts = [], [], [], [], [], []
    for i in idx:
        s = songs[i]
        nb = s.y.size
        a = int(rng.integers(0, nb - CROP)) if nb > CROP else 0
        e = min(nb, a + CROP)
        pad = CROP - (e - a)
        bar = s.bar[a:e] - s.bar[a]
        xs.append(np.pad(s.x[a:e], ((0, pad), (0, 0)))); ys.append(np.pad(s.y[a:e], (0, pad)))
        ws.append(np.pad(s.w[a:e], (0, pad))); ps.append(np.pad(s.pos[a:e], (0, pad), constant_values=2))
        bs.append(np.pad(bar, (0, pad), constant_values=int(bar.max()) + 1)); ts.append(np.pad(s.teach[a:e], ((0, pad), (0, 0))))
    return (np.stack(xs).astype(np.float32), np.stack(ys), np.stack(ws).astype(np.float32), np.stack(bs),
            np.stack(ps), np.stack(ts).astype(np.float32))


# --- the network ------------------------------------------------------------------------------------

def _rolls(blocks: int) -> np.ndarray:
    """[12, 12 * blocks]: row r reads every 12-block rolled down by r semitones (np.roll(block, -r))."""
    p = np.arange(12)
    return np.stack([np.concatenate([12 * b + (p + r) % 12 for b in range(blocks)]) for r in range(12)])


def network(arrays: dict):
    """The chord CNN as a torch module holding ``arrays``' weights, its state dict chord_key's arrays (same names and
    shapes; ``tab``, the chord-in-key tables, a buffer: kept, not trained). Computed as ChordKeyModel computes it:
    the 12 rotations of the input's three chroma blocks through one set of weights, the root r chords' logits the
    rotation r's, so a transposed song's chords are the song's chords transposed."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    def tensor(key):
        return torch.tensor(np.asarray(arrays[key], np.float32))

    class ChordKeyNet(nn.Module):
        def __init__(self):
            super().__init__()
            for k in TRAINED:
                setattr(self, k, nn.Parameter(tensor(k)))
            self.register_buffer("tab", tensor("tab"))
            self.register_buffer("rot", torch.as_tensor(_rolls(3)), persistent=False)
            self.register_buffer("roll", torch.as_tensor(_rolls(2)), persistent=False)

        def chords(self, x):               # x [B, n, 36] -> log P(chord | beat) [B, n, 24] (0-11 major, 12-23 minor)
            B, n, _ = x.shape
            h = x[..., self.rot].permute(2, 0, 3, 1).reshape(12 * B, 36, n)      # [12 rotations * B, 36, n]
            h = F.relu(F.conv1d(h, self.c1_w, self.c1_b, padding=self.c1_w.shape[2] // 2))
            h = F.relu(F.conv1d(h, self.c2_w, self.c2_b, padding=self.c2_w.shape[2] // 2)) + h
            o = F.conv1d(h, self.out_w, self.out_b, padding=self.out_w.shape[2] // 2)      # [12 * B, 2, n]
            return F.log_softmax(o.reshape(12, B, 2, n).permute(1, 3, 2, 0).reshape(B, n, 24), -1)

        def key_loglik(self, logpc, pos):  # -> [B, n, 24 keys]: log sum_c P(c | beat) P(c | key, phrase position)
            B, n, _ = logpc.shape
            R = F.log_softmax(self.tab, -1)                                     # [2 modes, 3 positions, 24]
            z = logpc[..., self.roll].unsqueeze(0) + R[:, pos].unsqueeze(3)     # [2, B, n, 12 tonics, 24 chords]
            return torch.logsumexp(z, -1).permute(1, 2, 0, 3).reshape(B, n, 24)

        def forward(self, x, pos):
            logpc = self.chords(x)
            return logpc, self.key_loglik(logpc, pos)

    return ChordKeyNet()


def export(net, arrays: dict) -> dict:
    """The model file's arrays: ``arrays`` (the decoder settings, the tables) with the net's weights."""
    out = {k: np.asarray(v) for k, v in arrays.items()}
    out.update({k: v.detach().float().cpu().numpy() for k, v in net.state_dict().items()})
    return out


@contextmanager
def _float32():
    """cuDNN's convolutions in full float32, as the NumPy inference and the shipped model's training computed them
    (TF32, cuDNN's default on newer GPUs, rounds them to ~1e-3; on a network this small it is no faster)."""
    import torch
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = tf32


def check_export(net, P: dict, x: np.ndarray, pos: np.ndarray) -> float:
    """The largest difference between ChordKeyModel's inference of the exported arrays and the net, on a song's beat
    inputs ``x`` [nb, 36] and phrase positions ``pos``: of P(chord | beat) and of the key evidence."""
    import torch
    net.eval()
    dev = net.tab.device
    with _float32(), torch.no_grad():
        logpc, L = net(torch.as_tensor(x, dtype=torch.float32, device=dev)[None],
                       torch.as_tensor(pos, dtype=torch.long, device=dev)[None])
        logpc, L = logpc[0].double().cpu().numpy(), L[0].double().cpu().numpy()
    model = CK.ChordKeyModel(arrays=P)
    ref = model.chords(np.asarray(x, np.float32))
    return float(max(np.abs(np.exp(logpc) - np.exp(ref)).max(), np.abs(L - model.key_loglik(ref, pos)).max()))


# --- training ---------------------------------------------------------------------------------------

def key_loss(net, batch, distill: float):
    """The loss of a batch of crops (``_batch``'s arrays): the cross-entropy of every bar's key -- the bar's evidence
    the sum of its keyed beats', its target the share of each key among them -- plus ``distill`` times the
    template-chord term, KL(templates || chords) averaged over the crops' beats. None when no beat has a key."""
    import torch
    import torch.nn.functional as F
    x, y, w, bar, pos, teach = batch
    B = x.shape[0]
    nbar = int(bar.max()) + 1
    gid = (np.arange(B)[:, None] * nbar + bar).ravel()
    cnt = np.bincount(gid, weights=w.ravel(), minlength=B * nbar)
    ok = np.flatnonzero(cnt > 0)
    if ok.size == 0:
        return None
    lab = np.zeros((B * nbar, 24)); np.add.at(lab, (gid, y.ravel()), w.ravel())
    dev = net.tab.device

    def put(a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dtype).to(dev)
    logpc, L = net(put(x), put(pos, torch.long))
    E = torch.zeros((B * nbar, 24), device=dev).index_add_(0, put(gid, torch.long),
                                                           (L * put(w)[..., None]).reshape(-1, 24))
    ce = -(put(lab[ok] / cnt[ok][:, None]) * F.log_softmax(E[put(ok, torch.long)], -1)).sum(-1).mean()
    t = put(teach)
    kl = (t * (torch.log(t + 1e-8) - logpc)).sum(-1).mean()
    return ce + distill * kl


def _init_weights(like: dict, seed: int) -> dict:
    """From scratch: weights and biases uniform in +-1 / sqrt(fan-in) (as the shipped model was started)."""
    rng = np.random.default_rng(30_000 + seed)
    out = {}
    for k in TRAINED:
        w = like[k.replace("_b", "_w")]
        bound = 1.0 / math.sqrt(int(np.prod(w.shape[1:])))
        out[k] = rng.uniform(-bound, bound, like[k].shape).astype(np.float32)
    return out


def _rate(step: int, steps: int, warmup: float) -> float:
    """The learning rate's factor: warmed up over the first ``warmup`` share of the steps, then a cosine decay
    (the beat TCN's); without a warmup the decay starts at the first step."""
    if int(round(warmup * steps)) == 0:
        return 0.5 * (1.0 + math.cos(math.pi * (step - 1) / max(1, steps)))
    return warmup_cosine(step, steps, warmup)


def optimizer(net, lr: float):
    """Adam on every weight and bias, WEIGHT_DECAY times the weight added to its gradient (L2)."""
    import torch
    return torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)


def steps_of(songs: int, epochs: int) -> int:
    return epochs * max(1, int(math.ceil(songs / BATCH)))


def train(arrays: dict, songs: list[Song], idx, epochs: int, lr: float, distill: float, seed: int, scratch: bool,
          device, progress: ProgressCallback | None, span, label) -> tuple[dict, float]:
    """One training on the songs ``idx`` from ``arrays`` (``scratch``: new weights, its shapes): (the model file's
    arrays, checked against the NumPy inference on the first song, seconds per step). An epoch is every song once
    in a random order, BATCH crops a step; the crops and the order are NumPy's (``seed``)."""
    import torch
    start = dict(arrays)
    if scratch:
        start.update(_init_weights(arrays, seed))
    net = network(start).to(device)
    opt = optimizer(net, lr)
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    per_epoch = max(1, int(math.ceil(idx.size / BATCH)))
    steps = steps_of(idx.size, epochs)
    warmup = SCRATCH_WARMUP if scratch else 0.0
    step = 0
    lo, hi = span
    net.train()
    began = time.perf_counter()
    with _float32():
        for ep in range(epochs):
            perm = rng.permutation(idx)
            total = 0.0
            for bi in range(per_epoch):
                step += 1
                loss = key_loss(net, _batch(songs, perm[bi * BATCH:(bi + 1) * BATCH], rng), distill)
                if loss is None:
                    continue
                for group in opt.param_groups:
                    group["lr"] = lr * _rate(step, steps, warmup)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total = total + loss.detach()
            if progress:
                progress(lo + int(round((hi - lo) * (ep + 1) / epochs)),
                         msg("{stage}: epoch {epoch}/{epochs}, loss {loss}", stage=label, epoch=ep + 1, epochs=epochs,
                             loss=f"{float(total) / per_epoch:.3f}"))
    if net.tab.device.type == "cuda":
        torch.cuda.synchronize(net.tab.device)
    seconds = (time.perf_counter() - began) / max(1, steps)
    P = export(net, arrays)
    first = songs[int(idx[0])]
    diff = check_export(net, P, first.x, first.pos)
    if not diff < EXPORT_TOLERANCE:
        raise RuntimeError(f"the exported key model differs from the trained one by {diff:.2g}")
    return P, seconds


# --- the end-to-end check: the whole key analysis on the songs ------------------------------------------

def evaluate(arrays: dict, songs: list[Song], idx) -> tuple[float, float]:
    """(share of the songs' keyed beats whose key signature is right, share off by a fifth) with the whole key
    analysis (chord CNN + decoder)."""
    model = CK.ChordKeyModel(arrays=arrays)
    sig_of = np.r_[np.arange(12), (np.arange(12) + 3) % 12]
    right = fifth = total = 0.0
    for i in idx:
        s = songs[i]
        L = model.key_loglik(model.chords(s.x), s.pos)
        _, key_per_bar = model.decode(L, s.grid, s.duration)
        est = sig_of[np.asarray(key_per_bar)[s.bar]]
        ref = sig_of[s.y]
        d = (7 * (est - ref)) % 12
        right += float(np.sum(s.w * (est == ref)))
        fifth += float(np.sum(s.w * ((d == 1) | (d == 11))))
        total += float(s.w.sum())
    return (right / total, fifth / total) if total else (float("nan"), float("nan"))


def _save(path: Path, arrays: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(path, {k: np.asarray(v) for k, v in arrays.items()}, compressed=False,
                    check=CK.read_model_arrays)


def optimize_key_model(
    request: KeyModelOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> KeyModelOptimizationResult:
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir).resolve()
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Key model path must end with .npz")
    scratch = request.start_path is None
    reference = CK.read_model_arrays(request.current_path)        # the model in use: what the new one is compared with
    base = reference if scratch else CK.read_model_arrays(request.start_path)
    ignored: list[SkippedOptimizationTrack] = []
    tracks = list_key_tracks(library_dir, ignored)
    MS.require_tracks(len(tracks), MS.kind_of(request.start), MS.KEY_SONGS)
    tracks = ensure_key_frames(tracks, cache_dir, ignored, rebuild=bool(request.rebuild_cache),
                               max_workers=request.max_cache_workers, progress=feature_progress, progress_span=(0, 90))
    songs, kept = [], []
    for done, t in enumerate(tracks, start=1):
        frames = read_key_frames(cache_path(cache_dir, str(t["uid"])), t)
        s = song_of(t, frames) if frames is not None else None
        if s is None:
            ignored.append(skipped_track(t, msg("too few beats or no key features")))
            continue
        songs.append(s); kept.append(t)
        if feature_progress and done % 10 == 0:
            feature_progress(90 + int(round(10 * done / len(tracks))),
                             msg("Beat inputs {done}/{total}", done=done, total=len(tracks)))
    if feature_progress:
        feature_progress(100, msg("{count} songs", count=len(songs)))
    MS.require_tracks(len(songs), MS.kind_of(request.start), MS.KEY_SONGS)
    device = device_of(request.device)
    edited = sum(t["kind"] == "edited" for t in kept)
    epochs = int(request.epochs) or (SCRATCH_EPOCHS if scratch else TUNE_EPOCHS)
    lr = float(request.learning_rate) or (SCRATCH_RATE if scratch else TUNE_RATE)
    everything = np.arange(len(songs))
    cross_validated = int(request.cv_folds) >= 2
    common = dict(edited_count=edited, legacy_count=len(kept) - edited, epochs=epochs, skipped_tracks=tuple(ignored),
                  device=str(device))
    if cross_validated:
        groups = np.asarray([_group_key(t) for t in kept])
        k = min(int(request.cv_folds), np.unique(groups).size)
        train_idx, held = next(GroupKFold(n_splits=k).split(everything, groups=groups))
        trial, seconds = train(base, songs, train_idx, epochs, lr, request.distill, request.seed, scratch, device,
                               optimize_progress, (0, 45), msg("Held-out check"))
        before, f_before = evaluate(reference, songs, held)
        after, f_after = evaluate(trial, songs, held)
        if after < before and not request.keep_if_worse:
            if optimize_progress:
                optimize_progress(100, msg("Not saved: the held-out songs' keys got worse"))
            return KeyModelOptimizationResult(
                output_path=output_path, track_count=len(songs), accuracy_before=before, accuracy_after=after,
                fifth_before=f_before, fifth_after=f_after, eval_songs=int(held.size), cross_validated=True,
                saved=False, steps=steps_of(train_idx.size, epochs), seconds_per_step=seconds, **common)
        span = (50, 98)
    else:
        span = (0, 95)
    final, seconds = train(base, songs, everything, epochs, lr, request.distill, request.seed, scratch, device,
                           optimize_progress, span, msg("Training") if scratch else msg("Fine-tuning"))
    if not cross_validated:
        before, f_before = evaluate(reference, songs, everything)
        after, f_after = evaluate(final, songs, everything)
        held = everything
    _save(output_path, final)
    if optimize_progress:
        optimize_progress(100, msg("Done"))
    return KeyModelOptimizationResult(
        output_path=output_path, track_count=len(songs), accuracy_before=before, accuracy_after=after,
        fifth_before=f_before, fifth_after=f_after, eval_songs=int(np.size(held)), cross_validated=cross_validated,
        steps=steps_of(everything.size, epochs), seconds_per_step=seconds, **common)
