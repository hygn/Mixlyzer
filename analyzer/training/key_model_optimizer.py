"""Retrain the chord-aware key model (analyzer/key/chord_key.py) on the keys the user checked, in NumPy.

Training songs are the library's tracks whose key annotation is the user's:
- a track with the automatic key result kept beside it (``key_segments_auto``, written by the full analysis and a
  whole-track key reanalysis) is used when its keys differ from it -- the user went through that track's keys;
  a track whose keys are still the automatic result is left out (the model would learn its own output);
- a track analyzed before the automatic result was kept is used unless its annotation has the shape of the old
  analyzer's raw output (10+ segments, or short fragments with same-key neighbours).
Only keys train the model: no chord labels (users cannot record them). The chord layer stays a chord layer through
the template-chord term (KL to the fixed triad templates), as when the shipped model was trained.

The network (a root-equivariant chord CNN, the fixed chord-in-key tables) is trained with Adam on random crops of
384 beats: cross-entropy of every bar's key (the sum of its beats' key evidence) plus the template-chord term. It
starts from the model in use, the shipped model or from scratch (analyzer/training/model_start.py). The decoder settings
(key-change costs) are kept. With cross-validation one group of songs is held out first and the current and the
retrained model -- the whole key analysis, decoder included -- are compared on it; only when the held-out keys are
not worse is the model retrained on every song and written.
"""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.key import chord_key as CK
from analyzer.music.beat_geometry import downbeat_times
from analyzer.runtime import spawn_pool
from analyzer.training import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from analyzer.training import model_start as MS
from analyzer.training.beat_tcn_optimizer import _rate
from analyzer.training.onset_training import _auto_cache_workers, _list_beat_tracks
from analyzer.training.refine_onset_optimizer import _group_key
from analyzer.files import atomic_output_path
from analyzer.training.text import msg

ProgressCallback = Callable[[int, str], None]
KEY_FRAME_CACHE_FORMAT = "mixlyzer_key_frame_cache_v1"
CROP = 384
BATCH = 16
WEIGHT_DECAY = 1e-4
SCRATCH_EPOCHS, TUNE_EPOCHS = 40, 15
SCRATCH_RATE, TUNE_RATE = 3e-3, 1e-3
TEMPLATE_TEMP, TEMPLATE_BASS = 0.08, 0.6


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


@dataclass(frozen=True)
class KeyModelOptimizationResult:
    output_path: Path
    track_count: int                 # training songs
    edited_count: int                # of them: keys differ from the kept automatic result
    legacy_count: int                # of them: no automatic result kept, accepted by the annotation's shape
    epochs: int
    accuracy_before: float = float("nan")   # share of the songs' beats whose key (signature) is right
    accuracy_after: float = float("nan")
    fifth_before: float = float("nan")      # share of beats off by a fifth (a Alphanumeric neighbour)
    fifth_after: float = float("nan")
    eval_songs: int = 0
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()
    cross_validated: bool = False
    saved: bool = True


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
        a = track["annotation"]        # the track's annotation the library's owner gave (analyzer.training.library_tracks)
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
    with atomic_output_path(path) as temporary:
        np.savez_compressed(temporary, cache_format=np.asarray(KEY_FRAME_CACHE_FORMAT),
                            audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns),
                            frames=frames.astype(np.float16))
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


# --- the network's forward and backward passes (the forward mirrors chord_key.ChordKeyModel) ------------

def _conv(x, W, b):
    """x [B, n, cin], W [cout, cin, k] ('same' padding) -> [B, n, cout]"""
    k = W.shape[2]
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    n = x.shape[1]
    out = np.broadcast_to(b, (x.shape[0], n, W.shape[0])).astype(x.dtype)
    for j in range(k):
        out = out + xp[:, j:j + n] @ W[:, :, j].T
    return out


def _conv_back(x, W, g, need_dx=True):
    k = W.shape[2]
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    n = x.shape[1]
    dW = np.zeros_like(W)
    g2 = g.reshape(-1, g.shape[-1])
    dxp = np.zeros_like(xp) if need_dx else None
    for j in range(k):
        dW[:, :, j] = g2.T @ xp[:, j:j + n].reshape(-1, xp.shape[-1])
        if need_dx:
            dxp[:, j:j + n] += g @ W[:, :, j]
    return dW, g2.sum(0), (dxp[:, pad:pad + n] if need_dx else None)


def _rot(x, r):
    return np.concatenate([np.roll(x[..., 12 * b:12 * (b + 1)], -r, axis=-1) for b in range(3)], axis=-1)


def _log_softmax(z):
    m = z.max(-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))


def _roll24(p, k):
    """chord axis: roots shifted by -k within each quality."""
    return np.concatenate([np.roll(p[..., :12], -k, axis=-1), np.roll(p[..., 12:], -k, axis=-1)], axis=-1)


def _unroll24(p, k):
    return np.concatenate([np.roll(p[..., :12], k, axis=-1), np.roll(p[..., 12:], k, axis=-1)], axis=-1)


def forward(P, x):
    """x [B, n, 36] -> log P(chord | beat) [B, n, 24] and the intermediates of every root's branch."""
    B, n, _ = x.shape
    logits = np.zeros((B, n, 24), x.dtype)
    saved = []
    for r in range(12):
        xr = _rot(x, r)
        a1 = _conv(xr, P["c1_w"], P["c1_b"])
        h1 = np.maximum(a1, 0)
        a2 = _conv(h1, P["c2_w"], P["c2_b"])
        h2 = np.maximum(a2, 0) + h1
        o = _conv(h2, P["out_w"], P["out_b"])
        logits[:, :, r], logits[:, :, 12 + r] = o[:, :, 0], o[:, :, 1]
        saved.append((xr, a1, h1, a2, h2))
    return _log_softmax(logits), saved


def backward(P, saved, logpc, glogpc):
    """Gradients of the weights from d loss / d log P(chord | beat)."""
    glogits = glogpc - np.exp(logpc) * glogpc.sum(-1, keepdims=True)
    G = {k: np.zeros_like(P[k]) for k in ("c1_w", "c1_b", "c2_w", "c2_b", "out_w", "out_b")}
    for r, (xr, a1, h1, a2, h2) in enumerate(saved):
        go = np.stack([glogits[:, :, r], glogits[:, :, 12 + r]], -1)
        dW, db, gh2 = _conv_back(h2, P["out_w"], go)
        G["out_w"] += dW; G["out_b"] += db
        ga2 = gh2 * (a2 > 0)
        dW, db, gh1 = _conv_back(h1, P["c2_w"], ga2)
        G["c2_w"] += dW; G["c2_b"] += db
        ga1 = (gh2 + gh1) * (a1 > 0)
        dW, db, _ = _conv_back(xr, P["c1_w"], ga1, need_dx=False)
        G["c1_w"] += dW; G["c1_b"] += db
    return G


def key_loglik(R, logpc, pos):
    """[B, n, 24 keys] log sum_c P(c | beat) P(c | key, position), with the softmax weights for the backward."""
    B, n, _ = logpc.shape
    L = np.zeros((B, n, 24), logpc.dtype)
    A = np.zeros((24, B, n, 24), logpc.dtype)
    for m in range(2):
        Rb = R[m][pos]
        for k in range(12):
            z = _roll24(logpc, k) + Rb
            zm = z.max(-1, keepdims=True)
            e = np.exp(z - zm)
            s = e.sum(-1, keepdims=True)
            L[:, :, m * 12 + k] = (zm + np.log(s))[..., 0]
            A[m * 12 + k] = e / s
    return L, A


def loss_and_grads(P, R, batch, distill):
    x, y, w, bar, pos, teach = batch
    B, n, _ = x.shape
    logpc, saved = forward(P, x)
    L, A = key_loglik(R, logpc, pos)
    # bar evidence: the sum of a bar's trusted beats; its target: the share of each key among them
    nbar = int(bar.max()) + 1
    gid = (np.arange(B)[:, None] * nbar + bar).ravel()
    wf = w.ravel()
    E = np.zeros((B * nbar, 24)); np.add.at(E, gid, (L * w[..., None]).reshape(-1, 24))
    cnt = np.bincount(gid, weights=wf, minlength=B * nbar)
    lab = np.zeros((B * nbar, 24)); np.add.at(lab, (gid, y.ravel()), wf)
    ok = cnt > 0
    if not ok.any():
        return 0.0, None
    tgt = lab[ok] / cnt[ok][:, None]
    lse = _log_softmax(E[ok])
    ce = float(-(tgt * lse).sum(-1).mean())
    gE = np.zeros_like(E)
    gE[ok] = (np.exp(lse) - tgt) / ok.sum()
    gL = (gE[gid].reshape(B, n, 24) * w[..., None]).astype(logpc.dtype)
    glogpc = np.zeros_like(logpc)
    for key in range(24):
        glogpc += _unroll24(gL[:, :, key, None] * A[key], key % 12)
    # template-chord term: KL(templates || chords), averaged over the crop's positions
    kl = float((teach * (np.log(teach + 1e-8) - logpc)).sum(-1).mean())
    glogpc -= distill * teach / float(B * n)
    G = backward(P, saved, logpc, glogpc.astype(logpc.dtype))
    return ce + distill * kl, G


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


def _init_weights(like: dict, seed: int) -> dict:
    """From scratch: weights and biases uniform in +-1 / sqrt(fan-in) (as the shipped model was started)."""
    rng = np.random.default_rng(30_000 + seed)
    out = {}
    for k in ("c1_w", "c1_b", "c2_w", "c2_b", "out_w", "out_b"):
        w = like[k.replace("_b", "_w")]
        bound = 1.0 / math.sqrt(int(np.prod(w.shape[1:])))
        out[k] = rng.uniform(-bound, bound, like[k].shape).astype(np.float32)
    return out


def train(arrays: dict, songs: list[Song], idx, epochs: int, lr: float, distill: float, seed: int, scratch: bool,
          progress: ProgressCallback | None, span, label: str) -> dict:
    P = _init_weights(arrays, seed) if scratch else {k: np.asarray(arrays[k], np.float32).copy()
                                                       for k in ("c1_w", "c1_b", "c2_w", "c2_b", "out_w", "out_b")}
    R = CK._log_softmax(np.asarray(arrays["tab"], np.float32), -1)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    per_epoch = max(1, int(math.ceil(idx.size / BATCH)))
    steps = epochs * per_epoch
    warmup = 0.3 if scratch else 0.0
    step = 0
    lo, hi = span
    for ep in range(epochs):
        perm = rng.permutation(idx)
        total = 0.0
        for bi in range(per_epoch):
            step += 1
            loss, G = loss_and_grads(P, R, _batch(songs, perm[bi * BATCH:(bi + 1) * BATCH], rng), distill)
            if G is None:
                continue
            total += loss
            rate = _rate(lr, step, steps, warmup)
            for k in P:
                g = G[k] + WEIGHT_DECAY * P[k]
                m[k] = b1 * m[k] + (1 - b1) * g
                v2[k] = b2 * v2[k] + (1 - b2) * g * g
                P[k] = (P[k] - rate * (m[k] / (1 - b1 ** step)) / (np.sqrt(v2[k] / (1 - b2 ** step)) + eps)).astype(np.float32)
        if progress:
            progress(lo + int(round((hi - lo) * (ep + 1) / epochs)), msg("{stage}: epoch {epoch}/{epochs}, loss {loss}", stage=label, epoch=ep + 1, epochs=epochs, loss=f"{total / per_epoch:.3f}"))
    out = {k: np.asarray(v) for k, v in arrays.items()}
    out.update(P)
    return out


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
    with atomic_output_path(Path(path)) as temporary:
        np.savez(temporary, **{k: np.asarray(v) for k, v in arrays.items()})


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
        feature_progress(100, f"{len(songs)} songs")
    MS.require_tracks(len(songs), MS.kind_of(request.start), MS.KEY_SONGS)
    edited = sum(t["kind"] == "edited" for t in kept)
    epochs = int(request.epochs) or (SCRATCH_EPOCHS if scratch else TUNE_EPOCHS)
    lr = float(request.learning_rate) or (SCRATCH_RATE if scratch else TUNE_RATE)
    everything = np.arange(len(songs))
    cross_validated = int(request.cv_folds) >= 2
    common = dict(edited_count=edited, legacy_count=len(kept) - edited, epochs=epochs, skipped_tracks=tuple(ignored))
    if cross_validated:
        groups = np.asarray([_group_key(t) for t in kept])
        k = min(int(request.cv_folds), np.unique(groups).size)
        train_idx, held = next(GroupKFold(n_splits=k).split(everything, groups=groups))
        trial = train(base, songs, train_idx, epochs, lr, request.distill, request.seed, scratch, optimize_progress,
                      (0, 45), msg("Held-out check"))
        before, f_before = evaluate(reference, songs, held)
        after, f_after = evaluate(trial, songs, held)
        if after < before and not request.keep_if_worse:
            if optimize_progress:
                optimize_progress(100, msg("Not saved: the held-out songs' keys got worse"))
            return KeyModelOptimizationResult(
                output_path=output_path, track_count=len(songs), accuracy_before=before, accuracy_after=after,
                fifth_before=f_before, fifth_after=f_after, eval_songs=int(held.size), cross_validated=True,
                saved=False, **common)
        span = (50, 98)
    else:
        span = (0, 95)
    final = train(base, songs, everything, epochs, lr, request.distill, request.seed, scratch, optimize_progress, span,
                  msg("Training") if scratch else msg("Fine-tuning"))
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
        **common)
