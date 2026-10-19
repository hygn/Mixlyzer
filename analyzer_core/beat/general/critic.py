"""The metronome critic: does this grid sound right?

The music is unrolled into the time of a candidate grid like a DAW / step-sequencer view: rows are bars,
columns the 48 slots of a 4/4 bar (12 per beat), channels the frame features averaged per slot plus melody
note events and the slot position. A small 2-D CNN (learned 1x1 'instrument' maps, four residual 3x3
convolutions dilated along the slots) judges windows of 8 bars; trained to tell the annotated grid from
half-beat, quarter-beat, 3:2 / 2:3 / 4:3 / 3:4 and drifting readings and from decoders' real mistakes.

Applied per steady section to the tempo-ratio readings only (3:2, 2:3, 4:3, 3:4): the reading that sounds
best wins, but only readings the onset evidence supports are heard (a critic must not pull a grid onto
nothing). A reading shares only part of its beats with the decoded grid (2 against 3: every other beat of
the slower reading; 3 against 4: every third), so when the activation fires on the decoded, wrong level a
right reading keeps just the evidence of its shared beats: it is heard when it keeps at least that share
of the decoded grid's evidence (SHARED). Half-beat readings are not used: different trainings of the
critic disagree on them, while their ratio judgements agree.
Weights: five fold models whose logits are averaged (assets/weights/beat_critic.npz).
"""
from __future__ import annotations

import numpy as np

from core.resource_paths import resource_path

from .decode import is_ramp
from .tcn import frame_inputs

SLOTS, BAR, HOPB = 12, 48, 8
WIN = 32                       # grid beats per window (8 bars of 4/4)
DIL = (1, 3, 6, 12)
# the share of a reading's beats that fall on the decoded grid's beats (its evidence gate)
SHARED = {'r32': 1.0 / 2.0, 'r23': 1.0 / 3.0, 'r43': 1.0 / 3.0, 'r34': 1.0 / 4.0}
READINGS = (('r32', 2.0 / 3.0, (0.0, 1.0 / 3.0, 2.0 / 3.0)), ('r23', 1.5, (0.0, 0.5, 1.0)),
            ('r43', 0.75, (0.0, 0.25, 0.5)), ('r34', 4.0 / 3.0, (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)))
WEIGHT_PATH = 'assets/weights/beat_critic.npz'


def load_weights(path: str = WEIGHT_PATH) -> list[dict]:
    """The fold models, each a dict of the torch state dict's arrays (npz keys 'f<k>/<name>')."""
    with np.load(resource_path(path)) as z:
        folds = sorted({k.split('/', 1)[0] for k in z.files})
        return [{k.split('/', 1)[1]: np.asarray(z[k], np.float32) for k in z.files if k.startswith(f + '/')} for f in folds]


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def _conv1x1(W, b, x):
    # channels last for one matrix product: [B, H, W, I] @ [I, O]
    y = np.moveaxis(x, 1, -1) @ W[:, :, 0, 0].T + b
    return np.moveaxis(y, -1, 1)


def _conv3x3(W, b, x, d):
    B, _, H, Wd = x.shape
    xp = np.moveaxis(np.pad(x, ((0, 0), (0, 0), (1, 1), (d, d))), 1, -1)     # [B, H+2, W+2d, I]
    out = np.zeros((B, H, Wd, W.shape[0]), x.dtype)
    for i in range(3):
        for j in range(3):
            out += xp[:, i:i + H, j * d:j * d + Wd, :] @ W[:, :, i, j].T
    return np.moveaxis(out + b, -1, 1)


def _forward(P, x):
    """x [B, C + SLOTS, WIN * SLOTS] -> logits [B]."""
    B, Cp, L = x.shape
    img = x.reshape(B, Cp, L // BAR, BAR)
    feat, pos = img[:, :-SLOTS], img[:, -SLOTS:]
    h = _conv1x1(P['inp.weight'], P['inp.bias'], np.concatenate([_elu(_conv1x1(P['inst.weight'], P['inst.bias'], feat)), pos], 1))
    for k, d in enumerate(DIL):
        h = h + _conv3x3(P[f'convs.{k}.weight'], P[f'convs.{k}.bias'], _elu(h), d)
    h = _elu(h)
    pooled = np.concatenate([h.mean((2, 3)), h.max((2, 3))], 1)
    return pooled @ P['out.weight'].T[:, 0] + P['out.bias'][0]


def _note_events(c, rms):
    voiced = (c >= 0).astype(np.float32)
    prev = np.r_[-1, c[:-1]]
    nxt = np.r_[c[1:], -1]
    start = ((c >= 0) & (c != prev)).astype(np.float32)
    end = ((c >= 0) & (c != nxt)).astype(np.float32)
    drop = np.maximum(0.0, -np.diff(np.r_[rms[0], rms])).astype(np.float32)
    return np.stack([voiced, start, end, drop], 1)


def track_features(frames) -> dict:
    """Per-track input of the critic: standardized frame features and melody note events, as prefix sums."""
    ft = np.asarray(frames.frame_times, float)
    x = frame_inputs(frames)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    n = x.shape[0]
    nev = _note_events(np.asarray(frames.melody_contour, np.float32)[:n], np.asarray(frames.rms_db, np.float32)[:n])
    nev = np.pad(nev, ((0, max(0, n - nev.shape[0])), (0, 0)))[:n]
    feat = np.concatenate([x, nev], 1).astype(np.float32)
    cs = np.vstack([np.zeros((1, feat.shape[1]), np.float32), np.cumsum(feat, 0, dtype=np.float64).astype(np.float32)])
    return dict(cs=cs, n=n, t0=float(ft[0]), hop=float(np.median(np.diff(ft))))


def _window(tr, g, k0):
    b = g[k0:k0 + WIN + 1]
    P = np.diff(b)
    edges = b[:-1, None] + P[:, None] * (np.arange(SLOTS + 1)[None, :] - 0.5) / SLOTS
    lo = edges[:, :-1].ravel()
    hi = edges[:, 1:].ravel()
    i0 = np.clip(np.floor((lo - tr['t0']) / tr['hop']).astype(int), 0, tr['n'] - 1)
    i1 = np.clip(np.ceil((hi - tr['t0']) / tr['hop']).astype(int), 1, tr['n'])
    i1 = np.maximum(i1, i0 + 1)
    f = (tr['cs'][i1] - tr['cs'][i0]) / (i1 - i0)[:, None]
    pos = np.eye(SLOTS, dtype=np.float32)[np.arange(f.shape[0]) % SLOTS]
    return np.concatenate([f, pos], 1).astype(np.float32)


def window_logits(nets, tr, g):
    """Mean over the fold models of the logits of every window of grid g (None when too short)."""
    if g.size < WIN + 2:
        return None
    starts = np.arange(0, g.size - WIN - 1, HOPB)
    x = np.stack([_window(tr, g, s).T for s in starts]).astype(np.float32)
    return np.mean([np.concatenate([_forward(P, x[i:i + 128]) for i in range(0, x.shape[0], 128)]) for P in nets], axis=0)


def _warp(ref, step, offset=0.0):
    idx = np.arange(offset, ref.size - 1, step)
    return np.interp(idx, np.arange(ref.size), ref)


def _section_beats(r, dur):
    P = 60.0 / r[2]
    k = np.arange(np.ceil((r[0] - r[3]) / P - 1e-9), np.floor((min(r[1], dur) - r[3]) / P - 1e-9) + 1)
    return r[3] + P * k


def apply(rows, nets, tr, val, dur):
    """Per steady section: the tempo-ratio reading the critic hears as most right (evidence-supported)."""
    rows = np.asarray(rows, float).copy()
    changed = 0
    for r in rows:
        if is_ramp(r):
            continue
        g = _section_beats(r, dur)
        wb = window_logits(nets, tr, g)
        if wb is None:
            continue
        base = float(wb.mean())
        ev0 = float(val(g).sum())
        best = (base, None)
        for name, step, offsets in READINGS:
            for off in offsets:
                alt = _warp(g, step, off)
                if ev0 > 0 and float(val(alt).sum()) < SHARED[name] * ev0:
                    continue
                w = window_logits(nets, tr, alt)
                if w is None:
                    continue
                s = float(w.mean())
                if s > best[0]:
                    best = (s, (step, alt))
        if best[1] is not None:
            step, alt = best[1]
            r[2] = r[2] / step
            if r.size > 5:
                r[5] = r[2]
            r[3] = float(alt[0])
            changed += 1
    return rows, changed
