"""Train the shipped phrase model, a stack (analyzer/phrase/stack.py), on the library's phrase annotations, as the
research trained it (research/benchmark/phrase_boundary: dl_dstcn.py, dl_dstcn_film.py, ship_dstcn.py,
ship_stack.py), the same operations in the same order:

1. The songs' phrase features (phrase_parameter_optimizer.prepare_tracks: its caches, featurecache/phrase) and song
   cues (analyzer.phrase.song_cues of the audio's mono mix, as the analysis decodes it, on the same beats; cached
   beside them, cues/<uid>.npz).
2. The DS-TCN (PyTorch; the net analyzer.phrase.dstcn runs: a dilated depthwise-separable TCN whose blocks a small MLP
   of the song's character scales and shifts (FiLM), a label + boundary head and an MS-TCN refinement stage), per beat
   the 9 labels and the boundary. OUTER_FOLDS nets, each trained without one fold of the songs (a song's versions kept
   together), give every song its out-of-fold outputs: the GBMs train on these, so no song's columns come from a net
   that saw its labels. One net trained on every song is the one shipped. Each net: AdamW (2e-3, decay 1e-2) over
   whole songs one at a time; label cross-entropy (the classes weighted by the inverse square root of their
   frequency) + boundary BCE (positives x20; 1 on the annotated starts, 0.5 a beat beside) + the MS-TCN smoothing
   term, on both stages; Gaussian noise 0.1 on the inputs; the songs whose character (the mean of their inputs) is the
   rarest -- the farthest 30 % from their 10 nearest -- seen 3 times an epoch; stopped on the label accuracy of 10 %
   of the song groups held back (15 epochs without a gain, 150 at most).
3. MEMBERS GBM pipelines (phrase_parameter_optimizer's two-pass fit) on the phrase features with the vocal cues and
   the DS-TCN's out-of-fold columns (stack.columns, as the analysis adds them), seeds 0 / 1000 / 2000, the further
   members with their inner folds assigned otherwise.
4. The mix-label specialist: a GBM telling an annotated segment's mixable labels (LEAD_IN / INTRO / OUTRO / INTERLUDE)
   from the sung body's (VERSE / BRIDGE / CHORUS / BREAK_CHORUS) by stack.segment_rows of the DS-TCN's inputs and the
   riff cues.
5. The stack written (phrase_analyzer.export_stack_artifact, read back by the analysis's loader first), the shipped
   DS-TCN's export checked against the NumPy inference.

The run writes the model trained on every song; the out-of-fold label accuracy of the DS-TCN is its held-out measure
(the whole stack's cross-validation is the research's, cv_all.py: a day). On a CUDA GPU the DS-TCNs take minutes
(the song cues of 480 songs about 30 minutes on 2 workers the first time, cached after); most of the run is the GBMs
on the CPU, and they hold every song's phrase features: about 13 GB for 480 songs, so on a 16 GB machine close the
other programs first (else the run pages for hours). torch is the training group's (uv sync --group training),
imported when the DS-TCNs train."""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.files import atomic_save_npz
from analyzer.phrase import dstcn as DS
from analyzer.phrase import stack as ST
from analyzer.phrase.phrase_analyzer import _hgb_to_npz_payload, export_stack_artifact, \
    export_two_stage_npz_artifact, load_two_stage_model
from analyzer.runtime import spawn_pool
from training import SkippedOptimizationTrack, library_tracks, skipped_track
from training import phrase_parameter_optimizer as PPO
from training.refine_onset_optimizer import _group_key
from training.text import TrainingError, msg

ProgressCallback = Callable[[int, str], None]
CUE_CACHE_FORMAT = "mixlyzer_phrase_cue_cache_v1"
OUTER_FOLDS = 5
MEMBERS = 3
CHANNELS = 48
REFINE_DILATIONS = (1, 2, 4, 8, 16, 32)
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
DROPOUT = 0.2
EPOCHS = 150
PATIENCE = 15
RARE_TOP = 0.3                # the share of the training songs repeated as the rarest
RARE_REPEATS = 3
RARE_NEIGHBOURS = 10
BOUNDARY_POS_WEIGHT = 20.0
STACK_WEIGHTS = (0.5, 0.25)   # the label mix: the specialist's log-odds, the DS-TCN's mean log-probability
EXPORT_TOLERANCE = 2e-3       # the NumPy inference's outputs (rounded to float16) against the trained net's


@dataclass(frozen=True)
class PhraseStackOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    rebuild_cache: bool = False
    seed: int = 0
    members: int = MEMBERS
    outer_folds: int = OUTER_FOLDS
    channels: int = CHANNELS
    max_cache_workers: int | None = None
    device: str = ""                 # "": CUDA when there is one, else the CPU


@dataclass(frozen=True)
class PhraseStackOptimizationResult:
    output_path: Path
    track_count: int
    members: int
    dstcn_out_of_fold_accuracy: float      # the DS-TCN's beat label accuracy on songs its net did not see
    dstcn_validation_accuracy: float       # the shipped net's, on its early-stopping songs
    specialist_segments: int
    minutes: float
    skipped_tracks: tuple[SkippedOptimizationTrack, ...] = ()


# --- the song cues ---------------------------------------------------------------------------------------------------

def cue_cache_path(cache_dir: Path, uid: str) -> Path:
    return Path(cache_dir) / "cues" / f"{uid}.npz"


def song_cues(audio_path: Path, beat_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vocal (beats, 8), style (beats, 14)) of the song as the analysis makes them: its decode at the cues' rate, the
    mono mix, the frame-level cues, then per beat."""
    from analyzer.audio.load import fast_load
    from analyzer.phrase import song_cues as SC
    stereo = fast_load(str(audio_path), target_sr=SC.SAMPLE_RATE, stereo=True)
    frames = SC.cue_frames(np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32))
    return SC.vocal_cues(frames, beat_times), SC.style_cues(frames, beat_times)


def read_cues(path: Path, audio_path: Path, beat_times: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The cached cues when they are current (same format, audio file and beats), else None."""
    if not Path(path).is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(np.asarray(z["cache_format"]).item()) != CUE_CACHE_FORMAT:
                return None
            if int(np.asarray(z["audio_mtime_ns"]).item()) != Path(audio_path).stat().st_mtime_ns:
                return None
            cached = np.asarray(z["beat_times"], np.float64)
            if cached.shape != beat_times.shape or not np.array_equal(cached, beat_times):
                return None
            return np.asarray(z["vocal"], np.float32), np.asarray(z["style"], np.float32)
    except (OSError, KeyError, ValueError):
        return None


def build_cues(job: tuple[str, str, np.ndarray, str]) -> str:
    """Process-worker entry point: one song's cues into the cache."""
    uid, audio_path, beat_times, cache_dir = job
    vocal, style = song_cues(Path(audio_path), beat_times)
    path = cue_cache_path(Path(cache_dir), uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(path, {"cache_format": np.asarray(CUE_CACHE_FORMAT), "beat_times": beat_times,
                           "audio_mtime_ns": np.asarray(Path(audio_path).stat().st_mtime_ns, np.int64),
                           "vocal": np.asarray(vocal, np.float32), "style": np.asarray(style, np.float32)})
    return uid


def ensure_cues(prepared, audio_paths: dict, cache_dir: Path, ignored: list, rebuild: bool, workers: int | None,
                progress: ProgressCallback | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every prepared song's cues, made from its audio where the cache has none current (worker processes)."""
    out, jobs = {}, []
    for p in prepared:
        uid, bt = str(p["uid"]), np.asarray(p["beat_times_sec"], np.float64)
        got = None if rebuild else read_cues(cue_cache_path(cache_dir, uid), audio_paths[uid], bt)
        if got is None:
            jobs.append((uid, str(audio_paths[uid]), bt, str(cache_dir)))
        else:
            out[uid] = got
    if jobs:
        with spawn_pool(max(1, min(int(workers or 2), len(jobs)))) as executor:
            futures = {executor.submit(build_cues, job): job for job in jobs}
            for done, f in enumerate(as_completed(futures), start=1):
                uid, audio, bt, _ = futures[f]
                try:
                    f.result()
                    out[uid] = read_cues(cue_cache_path(cache_dir, uid), Path(audio), bt)
                except Exception as exc:
                    ignored.append(skipped_track({"uid": uid, "audio_path": audio},
                                                 msg("song cues failed: {error}", error=exc)))
                if progress:
                    progress(int(round(100 * done / len(jobs))), msg("Song cues {done}/{total}", done=done,
                                                                     total=len(jobs)))
    return {u: v for u, v in out.items() if v is not None}


# --- the DS-TCN ------------------------------------------------------------------------------------------------------

@dataclass
class Song:
    """A song as the DS-TCN trains on it: inputs (beats, 155), the phrase grid's downbeats and the positions made from
    them (beats, 8), labels (index in dstcn.LABELS, -1 none), boundary (1 at an annotated start), its group (versions
    together)."""
    uid: str
    x: np.ndarray
    down: np.ndarray
    pos: np.ndarray
    y: np.ndarray
    bnd: np.ndarray
    group: str


def beat_labels(beat_times: np.ndarray, starts: np.ndarray, labels) -> tuple[np.ndarray, np.ndarray]:
    """(label per beat, boundary per beat): each annotated phrase from the beat at its start (50 ms early allowed) to
    the next phrase's; beats before the first phrase and labels the net has no class for are -1."""
    bt = np.asarray(beat_times, float)
    b = np.clip(np.searchsorted(bt, np.asarray(starts, float) - 0.05), 0, bt.size - 1)
    y = np.full(bt.size, -1, np.int64)
    for k in range(len(b)):
        lab = str(labels[k])
        y[b[k]:(b[k + 1] if k + 1 < len(b) else bt.size)] = DS.LABELS.index(lab) if lab in DS.LABELS else -1
    bnd = np.zeros(bt.size, np.float32)
    bnd[b[b > 0]] = 1.0
    return y, bnd


def make_song(p: dict, vocal: np.ndarray, style: np.ndarray, group: str) -> Song:
    y, bnd = beat_labels(p["beat_times_sec"], p["phrase_starts_sec"], p["phrase_labels"])
    down = np.asarray(p["downbeat"], bool)
    return Song(str(p["uid"]), DS.inputs(p["feature_z"], vocal, style), down, DS.positions(down), y, bnd, group)


def network(d_in: int, channels: int = CHANNELS):
    """The torch module whose state dict (``w.<name>``) dstcn.DsTcn reads; ``d_in``: the inputs with the positions."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    labels = len(DS.LABELS)

    class Block(nn.Module):
        def __init__(self, c, d):
            super().__init__()
            self.a = nn.Conv1d(c, c, 3, padding=d, dilation=d, groups=c)
            self.b = nn.Conv1d(c, c, 3, padding=2 * d, dilation=2 * d, groups=c)
            self.pw = nn.Conv1d(2 * c, c, 1)
            self.norm = nn.GroupNorm(1, c)
            self.drop = nn.Dropout(DROPOUT)

        def forward(self, h):
            return h + self.drop(F.gelu(self.norm(self.pw(torch.cat([self.a(h), self.b(h)], 1)))))

    class FilmNet(nn.Module):
        def __init__(self, c=channels, e=32):
            super().__init__()
            self.inp = nn.Conv1d(d_in, c, 1)
            self.in_drop = nn.Dropout(0.1)
            self.blocks = nn.ModuleList([Block(c, d) for d in DILATIONS])
            self.song = nn.Sequential(nn.Linear(2 * d_in, e), nn.GELU(), nn.Dropout(0.1))
            self.film = nn.Linear(e, 2 * c * len(DILATIONS))
            nn.init.zeros_(self.film.weight)          # the FiLM starts as the identity
            nn.init.zeros_(self.film.bias)
            self.c = c
            self.head = nn.Conv1d(c, labels + 1, 1)
            self.rin = nn.Conv1d(labels + 1, 32, 1)
            self.refine = nn.Sequential(*[Block(32, d) for d in REFINE_DILATIONS])
            self.rhead = nn.Conv1d(32, labels + 1, 1)

        def forward(self, x):                       # (1, T, D) -> two (1, T, labels + 1)
            desc = torch.cat([x.mean(1), x.std(1)], -1)                      # (1, 2D): the song's character
            gb = self.film(self.song(desc)).view(len(DILATIONS), 2, self.c)
            h = self.inp(self.in_drop(x).transpose(1, 2))
            for k, blk in enumerate(self.blocks):
                h = blk(h)
                h = h * (1 + gb[k, 0][None, :, None]) + gb[k, 1][None, :, None]
            o1 = self.head(h)
            z = torch.cat([F.softmax(o1[:, :labels], 1), torch.sigmoid(o1[:, labels:])], 1)
            o2 = self.rhead(self.refine(self.rin(z)))
            return o1.transpose(1, 2), o2.transpose(1, 2)

    return FilmNet()


def _stage_loss(o, y, bt, w):
    import torch
    import torch.nn.functional as F
    n = len(DS.LABELS)
    lab, bnd = o[0, :, :n], o[0, :, n]
    ce = F.cross_entropy(lab, y, weight=w, ignore_index=-1)
    bce = F.binary_cross_entropy_with_logits(bnd, bt, pos_weight=torch.tensor(BOUNDARY_POS_WEIGHT, device=o.device))
    lp = F.log_softmax(lab, -1)
    sm = torch.clamp((lp[1:] - lp[:-1].detach()) ** 2, max=16).mean()
    return ce + 0.5 * bce + 0.15 * sm


def _tensor(s: Song, mu, sd, train: bool, device):
    import torch
    x = (s.x - mu) / sd
    if train:
        x = x + np.random.normal(0, 0.1, x.shape).astype(np.float32)
    return torch.from_numpy(np.hstack([x, s.pos]).astype(np.float32)[None]).to(device)


def _accuracy(pairs) -> float:
    ok = n = 0
    for y, pred in pairs:
        m = y >= 0
        ok += int((pred[m] == y[m]).sum())
        n += int(m.sum())
    return ok / max(n, 1)


def rarity_repeats(train: list[Song], mu, sd) -> np.ndarray:
    """RARE_REPEATS for the training songs whose character (the mean of their normalized inputs) lies farthest from
    its RARE_NEIGHBOURS nearest training songs (the rarest RARE_TOP share), 1 for the rest: the songs unlike the
    library's bulk (on 419 songs the rarest 30 % held 25 of the 33 electronic ones), found without labels."""
    X = np.vstack([((s.x - mu) / sd).mean(0) for s in train])
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    knn = np.sort(d, axis=1)[:, :min(RARE_NEIGHBOURS, max(1, len(train) - 1))].mean(1)
    rank = knn.argsort().argsort() / len(knn)
    return np.where(rank >= 1.0 - RARE_TOP, RARE_REPEATS, 1)


def fit_net(songs: list[Song], seed: int, device, channels: int = CHANNELS):
    """A net on ``songs`` (10 % of their groups held back for early stopping): (net, mu, sd, best validation
    accuracy)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    groups = sorted({s.group for s in songs})
    val_g = set(rng.choice(groups, max(1, len(groups) // 10), replace=False))
    val = [s for s in songs if s.group in val_g]
    train = [s for s in songs if s.group not in val_g]
    allx = np.vstack([s.x for s in train])
    mu, sd = allx.mean(0), allx.std(0) + 1e-3
    n = len(DS.LABELS)
    freq = np.bincount(np.concatenate([s.y[s.y >= 0] for s in train]), minlength=n) + 1
    w = torch.tensor((freq.sum() / freq) ** 0.5, dtype=torch.float32, device=device)
    w = w / w.mean()
    order = [s for s, r in zip(train, rarity_repeats(train, mu, sd)) for _ in range(int(r))]
    net = network(train[0].x.shape[1] + DS.POSITIONS, channels).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    best, best_state, bad = -1.0, None, 0
    for _epoch in range(EPOCHS):
        net.train()
        for j in rng.permutation(len(order)):
            s = order[j]
            bt = torch.from_numpy(np.clip(s.bnd + 0.5 * (np.roll(s.bnd, 1) + np.roll(s.bnd, -1)), 0, 1)
                                  .astype(np.float32)).to(device)
            y = torch.from_numpy(s.y).to(device)
            o1, o2 = net(_tensor(s, mu, sd, True, device))
            loss = _stage_loss(o1, y, bt, w) + _stage_loss(o2, y, bt, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = _accuracy([(s.y, net(_tensor(s, mu, sd, False, device))[1][0, :, :n].argmax(-1).cpu().numpy())
                             for s in val])
        if acc > best:
            best, best_state, bad = acc, {a: v.detach().clone() for a, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, mu, sd, best


def net_outputs(net, mu, sd, s: Song, device) -> tuple[np.ndarray, np.ndarray]:
    """(label probabilities, boundary probability) of the refined stage, rounded to float16 as the analysis rounds
    them."""
    import torch
    import torch.nn.functional as F
    n = len(DS.LABELS)
    with torch.no_grad():
        o = net(_tensor(s, mu, sd, False, device))[1][0]
        p = F.softmax(o[:, :n], -1).cpu().numpy()
        b = torch.sigmoid(o[:, n]).cpu().numpy()
    return p.astype(np.float16).astype(np.float32), b.astype(np.float16).astype(np.float32)


def export_net(net, mu, sd, best: float, channels: int) -> dict[str, np.ndarray]:
    """The arrays dstcn.DsTcn reads (the stack's ``dstcn.`` block)."""
    arrays = {f"w.{k}": v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    return {"mu": np.asarray(mu, np.float32), "sd": np.asarray(sd, np.float32),
            "dilations": np.asarray(DILATIONS, np.int64), "refine_dilations": np.asarray(REFINE_DILATIONS, np.int64),
            "labels": np.asarray(DS.LABELS), "channels": np.int64(channels), "val_acc": np.float64(best), **arrays}


def check_export(net, mu, sd, arrays: dict, songs: list[Song], device) -> float:
    """The largest difference between the NumPy inference of the exported arrays and the net, on ``songs``."""
    import torch
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False        # full float32 as the NumPy inference
    try:
        numpy_net = DS.DsTcn(arrays)
        worst = 0.0
        for s in songs:
            p, b = net_outputs(net, mu, sd, s, device)
            q, c = numpy_net(s.x, s.down)
            worst = max(worst, float(np.abs(p - q).max()), float(np.abs(b - c).max()))
        return worst
    finally:
        torch.backends.cudnn.allow_tf32 = tf32


# --- the run ---------------------------------------------------------------------------------------------------------

def _scaled(progress: ProgressCallback | None, lo: int, hi: int) -> ProgressCallback | None:
    if progress is None:
        return None
    return lambda pct, text: progress(lo + int(round(pct * (hi - lo) / 100)), text)


def optimize_phrase_stack(
    request: PhraseStackOptimizationRequest,
    *,
    feature_progress: ProgressCallback | None = None,
    optimize_progress: ProgressCallback | None = None,
) -> PhraseStackOptimizationResult:
    from training.beat_tcn_optimizer import device_of
    t_start = time.time()
    library_dir = Path(request.library_dir).resolve()
    cache_dir = Path(request.cache_dir)
    output_path = Path(request.output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Phrase model path must end with .npz")
    device = device_of(request.device)
    data = PPO.prepare_tracks(library_dir, cache_dir, bool(request.rebuild_cache), request.max_cache_workers,
                              _scaled(feature_progress, 0, 70))
    ignored = list(data.ignored)
    by_uid = {str(t["uid"]): t for t in library_tracks.tracks()}
    audio = {str(p["uid"]): Path(by_uid[str(p["uid"])]["audio_path"]) for p in data.prepared}
    cues = ensure_cues(data.prepared, audio, cache_dir, ignored, bool(request.rebuild_cache),
                       request.max_cache_workers, _scaled(feature_progress, 70, 100))
    keep = [i for i, p in enumerate(data.prepared) if str(p["uid"]) in cues]
    prepared = [data.prepared[i] for i in keep]
    targets = [data.targets[i] for i in keep]
    if len(prepared) < max(5, int(request.outer_folds)):
        raise TrainingError(msg("Need at least five tracks with song cues to train the phrase stack ({valid} "
                                "valid, {ignored} ignored).", valid=len(prepared), ignored=len(ignored)))

    def group(uid: str) -> str:
        t = by_uid[uid]
        return str(t.get("song") or _group_key(t))
    songs = [make_song(p, *cues[str(p["uid"])], group(str(p["uid"]))) for p in prepared]
    say = optimize_progress or (lambda _v, _t: None)

    # 2. the DS-TCNs: out-of-fold outputs for every song, then the shipped net
    folds = int(request.outer_folds)
    oof: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    groups = np.asarray([s.group for s in songs])
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(np.arange(len(songs)), groups=groups), start=1):
        say(int(round(30 * (k - 1) / (folds + 1))), msg("DS-TCN: fold {k}/{folds}", k=k, folds=folds))
        net, mu, sd, _ = fit_net([songs[i] for i in tr], int(request.seed) + k, device, int(request.channels))
        for i in te:
            oof[songs[i].uid] = net_outputs(net, mu, sd, songs[i], device)
    oof_acc = _accuracy([(s.y, oof[s.uid][0].argmax(1)) for s in songs])
    say(int(round(30 * folds / (folds + 1))), msg("DS-TCN: every song"))
    net, mu, sd, val_acc = fit_net(songs, int(request.seed), device, int(request.channels))
    dstcn = export_net(net, mu, sd, val_acc, int(request.channels))
    diff = check_export(net, mu, sd, dstcn, songs[:4], device)
    if not diff < EXPORT_TOLERANCE:
        raise RuntimeError(f"the exported DS-TCN differs from the trained one by {diff:.2g}")

    # 3. the GBM members on the phrase features with the stack's columns (the out-of-fold DS-TCN's)
    for p, s in zip(prepared, songs):
        vocal = cues[s.uid][0]
        p["feature_z"], p["boundary_feature_z"] = ST.columns(p["feature_z"], p["boundary_feature_z"], vocal, *oof[s.uid])
    work = output_path.parent / f".{output_path.stem}.stack_parts"
    work.mkdir(parents=True, exist_ok=True)
    try:
        idx = np.arange(len(prepared), dtype=np.int32)
        members = int(request.members)
        member_paths = []
        for m in range(members):
            lo, hi = 30 + int(round(60 * m / members)), 30 + int(round(60 * (m + 1) / members))
            say(lo, msg("GBM member {m}/{members}", m=m + 1, members=members))
            artifact = PPO._fit_pipeline(prepared, idx, targets, data.settings, int(request.seed) + 1000 * m,
                                         _scaled(optimize_progress, lo, hi), None if m == 0 else 1000 + m)
            member_paths.append(work / f"member{m}.npz")
            export_two_stage_npz_artifact(artifact, member_paths[-1])

        # 4. the mix-label specialist
        say(91, msg("Mix-label specialist"))
        from sklearn.ensemble import HistGradientBoostingClassifier
        X, y = [], []
        for s, (bounds, labels) in zip(songs, targets):
            x = np.hstack([s.x, ST.riff_cues(s.x, s.down)])
            for row, lab in zip(ST.segment_rows(x, np.asarray(bounds, int)), labels):
                if lab in ST.MIXABLE:
                    X.append(row), y.append(1)
                elif lab in ST.BODY:
                    X.append(row), y.append(0)
        specialist = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                    l2_regularization=1.0, random_state=0).fit(np.array(X), np.array(y))
        payload: dict[str, np.ndarray] = {}
        _hgb_to_npz_payload("spec", specialist, payload)
        atomic_save_npz(work / "spec.npz", payload)
        atomic_save_npz(work / "dstcn.npz", dstcn)

        # 5. the stack
        say(97, msg("Writing the phrase stack"))
        export_stack_artifact(member_paths, work / "spec.npz", work / "dstcn.npz", output_path, STACK_WEIGHTS)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        load_two_stage_model.cache_clear()
    say(100, msg("Done"))
    return PhraseStackOptimizationResult(
        output_path=output_path, track_count=len(songs), members=members, dstcn_out_of_fold_accuracy=oof_acc,
        dstcn_validation_accuracy=float(val_acc), specialist_segments=len(y),
        minutes=(time.time() - t_start) / 60, skipped_tracks=tuple(ignored))
