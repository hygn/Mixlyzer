"""Section decoding: the track as a sequence of steady beat grids.

A song is a sequence of sections, each with a steady pulse (tempo, phase). A grid explains a section by
the onset contrast on its beats (the beat activation minus its 4 s local mean, so a beat on nothing costs
and a denser grid gains nothing by itself). Every new section costs J: it has to explain the music that
much better than continuing. Dynamic programming over section boundaries (every 0.5 s) and candidate
grids (tempos from the activation's own periodicity and their simple ratios, 32 phases per beat).

Then the same-tempo phase judgement (a half-beat reset needs overwhelming evidence) and the exact fit of
every steady section, first on the pooled contrast (robust), then on the activation read between frames
(precise: the pooled contrast is flat within +-1 frame).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

POS_SEC = 0.5                 # section boundaries are decided on this grid (refined later)
PHASES = 32                   # phases per beat in the decoder
RATIOS = (1.0, 1.5, 2.0 / 3.0, 4.0 / 3.0, 0.75)
J = 12.0                      # the cost of a new section (evidence units)
HALF_RESET = 96.0             # extra cost of a same-tempo half-beat reset (8 J)


class Octave:
    """The tempo octave the decoder works in, [lo, 2 lo) (from the user's BPM range setting)."""

    def __init__(self, lo: float):
        self.lo = float(lo)
        self.hi = 2.0 * float(lo)

    def fold(self, b: float) -> float:
        while b >= self.hi:
            b /= 2
        while b < self.lo:
            b *= 2
        return b


def pooled_contrast(act, hop):
    p = maximum_filter1d(np.asarray(act, np.float64), 3)
    return p - uniform_filter1d(p, max(1, int(round(4.0 / hop))))


def readers(act, hop, t0):
    """(val, fine): the pooled contrast at the frame nearest t, and the activation minus its 4 s mean read
    between frames (frame i is at t0 + i hop)."""
    a = np.asarray(act, np.float64)
    con = pooled_contrast(a, hop)
    last = con.size - 1
    raw = a - uniform_filter1d(a, max(1, int(round(4.0 / hop))))
    fi = np.arange(raw.size, dtype=np.float64)

    def val(t):
        return con[np.clip(np.round((np.asarray(t, np.float64) - t0) / hop).astype(np.int64), 0, last)]

    def fine(t):
        return np.interp((np.asarray(t, np.float64) - t0) / hop, fi, raw)

    return val, fine


def coarse_tempos(act, hop, octave: Octave, win=8.0, step=4.0, top=3, group=0.01):
    """The pulse periods the music shows locally: in 8 s windows (4 s hop) the autocorrelation of the
    onset contrast, its strongest peaks (parabolic sub-frame lag), folded into the octave and grouped
    within 1 %; a tempo heard in at least two windows or clearly in one is kept."""
    c = pooled_contrast(act, hop)
    W, S = int(round(win / hop)), int(round(step / hop))
    lag_lo, lag_hi = int(np.floor(60.0 / 330.0 / hop)), int(np.ceil(60.0 / 55.0 / hop))
    found = []
    for a in range(0, max(1, c.size - W + 1), S):
        x = c[a:a + W] - c[a:a + W].mean()
        e = float(np.dot(x, x))
        if e <= 1e-9:
            continue
        r = np.array([np.dot(x[:-l], x[l:]) / e for l in range(lag_lo, min(lag_hi, x.size - 1))])
        if r.size < 3:
            continue
        pk = np.flatnonzero((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:]) & (r[1:-1] > 0.05)) + 1
        for i in pk[np.argsort(r[pk])[::-1][:top]]:
            y0, y1, y2 = r[i - 1], r[i], r[i + 1]
            den = y0 - 2 * y1 + y2
            sub = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            found.append((octave.fold(60.0 / ((lag_lo + i + sub) * hop)), float(y1)))
    if not found:
        return np.array([octave.fold(120.0)])
    found.sort()
    groups = []
    for t, w in found:
        if groups and (t - groups[-1][0][-1]) / t <= group:
            groups[-1][0].append(t)
            groups[-1][1].append(w)
        else:
            groups.append(([t], [w]))
    best = max(sum(w) for _, w in groups)
    return np.asarray(sorted(float(np.average(t, weights=w)) for t, w in groups if len(t) >= 2 or sum(w) >= 0.5 * best))


def with_ratios(tempos, octave: Octave):
    """The tempos and their simple ratios (3:2, 2:3, 4:3, 3:4), folded, no duplicates within 0.2 %."""
    out = []
    for t in np.asarray(tempos, float):
        for r in RATIOS:
            b = t * r
            while b > octave.hi:
                b /= 2
            while b < octave.lo:
                b *= 2
            if all(abs(b - o) / o > 0.002 for o in out):
                out.append(b)
    return np.asarray(sorted(out))


def refit_tempos(rows, act, hop, off, tol=0.125):
    """Exact tempos of the decoded steady sections: least squares on the activation peaks within an
    eighth of a beat of the section's beats (at least 8, the period moving < 2 %)."""
    a = np.asarray(act, np.float64)
    m = (a >= maximum_filter1d(a, 5)) & (a > 0.3 * a.max())
    pt = off + hop * (np.flatnonzero(m) + 0.5)
    out = []
    for r in np.asarray(rows, float):
        P = 60.0 / r[2]
        k = np.arange(np.ceil((r[0] - r[3]) / P), np.floor((r[1] - r[3]) / P) + 1)
        g = r[3] + P * k
        if g.size < 8 or pt.size < 8:
            continue
        j = np.clip(np.searchsorted(pt, g), 1, pt.size - 1)
        near = np.where(np.abs(pt[j - 1] - g) <= np.abs(pt[j] - g), pt[j - 1], pt[j])
        ok = np.abs(near - g) <= tol * P
        if ok.sum() < 8:
            continue
        A = np.column_stack([np.ones(ok.sum()), k[ok]])
        (_, p2), *_ = np.linalg.lstsq(A, near[ok], rcond=None)
        if abs(p2 - P) / P < 0.02:
            out.append(60.0 / p2)
    return np.asarray(out)


CHUNK_SEC = 300.0             # long tracks (mixes) are decoded in pieces of this length ...
CHUNK_OVERLAP_SEC = 60.0      # ... overlapping by this much, stitched in the middle of the overlap
CHUNK_FROM_SEC = 480.0        # songs up to this length are decoded in one piece


def partition(act, hop, off, dur, tempos, J=J):
    """The optimal sequence of steady grids: rows [start, end, bpm, phase anchor, 4, bpm].

    The dynamic programme compares every section start with every end (quadratic in the length), so a
    long track (a mix) is decoded in overlapping pieces: which grid a stretch of music has is decided by
    the music around it, not minutes away. The pieces are joined in the middle of their overlap; where both
    sides chose the same grid it is one section, otherwise the join is an ordinary junction for the
    phase judgement, the refits, the ramps and the boundary refinement."""
    if dur <= CHUNK_FROM_SEC:
        return _partition(act, hop, off, dur, tempos, J)
    a = np.asarray(act, np.float64)
    rows_all = []
    step = CHUNK_SEC - CHUNK_OVERLAP_SEC
    starts = np.arange(0.0, max(dur - CHUNK_OVERLAP_SEC, 1e-9), step)
    for k, c0 in enumerate(starts):
        c1 = min(dur, c0 + CHUNK_SEC)
        i0 = max(0, int(np.floor((c0 - off) / hop)))
        i1 = min(a.size, int(np.ceil((c1 - off) / hop)) + 1)
        sub_off = off + i0 * hop
        rows = _partition(a[i0:i1], hop, sub_off - c0, c1 - c0, tempos, J)
        rows[:, [0, 1, 3]] += c0
        lo = c0 + 0.5 * CHUNK_OVERLAP_SEC if k > 0 else 0.0          # keep [lo, hi) of this piece
        hi = c0 + step + 0.5 * CHUNK_OVERLAP_SEC if c1 < dur else dur
        for r in rows:
            if r[1] <= lo or r[0] >= hi:
                continue
            r = r.copy()
            r[0] = max(r[0], lo)
            r[1] = min(r[1], hi)
            if rows_all:
                p = rows_all[-1]
                same = abs(r[2] - p[2]) < 1e-9 and abs(((r[3] - p[3]) / (60.0 / p[2]) + 1e-9) % 1.0) < 1e-6
                if same and abs(p[1] - r[0]) < 1e-6:
                    p[1] = r[1]
                    continue
            rows_all.append(r)
        if c1 >= dur:
            break
    return np.asarray(rows_all)


def _tables(act, hop, off, dur, tempos):
    """(positions, evidence table): the evidence of each (tempo, phase) grid's beats before each position."""
    con = pooled_contrast(act, hop)
    last = con.size - 1

    def read(t):
        return con[np.clip(np.round((t - off) / hop - 0.5).astype(np.int64), 0, last)]

    pos = np.r_[np.arange(0.0, dur, POS_SEC), dur]
    tables = []
    for bpm in tempos:
        P = 60.0 / bpm
        ph = (np.arange(PHASES) / PHASES) * P
        k = np.arange(int(np.ceil(dur / P)) + 2)
        beats = ph[:, None] + P * k[None, :]
        vals = read(beats) * (beats < dur)
        pref = np.concatenate([np.zeros((PHASES, 1)), np.cumsum(vals, axis=1)], axis=1)
        kpos = np.clip(np.ceil((pos[None, :] - ph[:, None]) / P).astype(int), 0, k.size)
        tables.append(np.take_along_axis(pref, kpos, axis=1))
    return pos, np.stack(tables)


def single_grid(act, hop, off, dur, tempos):
    """The one steady grid with the most evidence over the whole track (static analysis): a row
    [0, dur, bpm, phase anchor, 4, bpm]."""
    _, tab = _tables(act, hop, off, dur, tempos)
    ti, m = divmod(int(np.argmax(tab[:, :, -1] - tab[:, :, 0])), PHASES)
    bpm = tempos[ti]
    return np.array([[0.0, dur, bpm, (m / PHASES) * 60.0 / bpm, 4.0, bpm]])


def _partition(act, hop, off, dur, tempos, J=J):
    pos, tab = _tables(act, hop, off, dur, tempos)
    n = pos.size
    best = np.full(n, -np.inf)
    best[0] = 0.0
    back = np.zeros((n, 3), int)
    for j in range(1, n):
        s = tab[:, :, j][:, :, None] - tab[:, :, :j]
        flat = s.reshape(-1, j)
        arg = np.argmax(flat, axis=0)
        v = flat[arg, np.arange(j)] + best[:j] - J
        i = int(np.argmax(v))
        best[j] = v[i]
        back[j] = (i, *divmod(int(arg[i]), PHASES))
    rows = []
    j = n - 1
    while j > 0:
        i, ti, m = back[j]
        bpm = tempos[ti]
        P = 60.0 / bpm
        rows.append([pos[i], pos[j], bpm, (m / PHASES) * P, 4.0, bpm])
        j = i
    rows = np.asarray(rows[::-1])
    out = [rows[0]]
    for r in rows[1:]:
        p = out[-1]
        if abs(r[2] - p[2]) < 1e-9 and abs(((r[3] - p[3]) / (60.0 / p[2]) + 1e-9) % 1.0) < 1e-6:
            p[1] = r[1]
        else:
            out.append(r)
    return np.asarray(out)


def is_ramp(r) -> bool:
    return r.size > 5 and abs(r[5] - r[2]) > 1e-6


def steady_beats(row, t0, t1):
    """Beats of a steady row's grid in [t0, t1] (the grid extended beyond the row)."""
    P = 60.0 / row[2]
    k = np.arange(np.ceil((t0 - row[3]) / P - 1e-9), np.floor((t1 - row[3]) / P - 1e-9) + 1)
    return row[3] + P * k


def phase_merge(rows, val, extra=HALF_RESET):
    """Same-tempo junctions keep one phase unless the music clearly moves it: the reset as decoded, A's
    phase going on through B, or B's phase reaching back over A -- the most onset evidence wins, a reset
    by half a beat (+-1/8) charged ``extra`` (the off-beat is the classic confusion of an observation
    model and rarely the music)."""
    rows = [np.asarray(r, float).copy() for r in rows]
    out = [rows[0]]
    for B in rows[1:]:
        A = out[-1]
        if is_ramp(A) or is_ramp(B) or abs(B[2] / A[2] - 1.0) > 0.01:
            out.append(B)
            continue
        P = 60.0 / A[2]
        half = abs(((B[3] - A[3]) / P) % 1.0 - 0.5) <= 0.125
        keep = float(val(steady_beats(A, A[0], B[0] - 1e-6)).sum() + val(steady_beats(B, B[0], B[1])).sum())
        keep -= extra if half else 0.0
        a_on = float(val(steady_beats(A, A[0], B[1])).sum())
        b_back = float(val(steady_beats(B, A[0], B[1])).sum())
        best = max(keep, a_on, b_back)
        if best == keep:
            out.append(B)
        elif best == a_on:
            A[1] = B[1]
        else:
            A[1] = B[1]
            A[2] = A[5] = B[2]
            A[3] = B[3]
    return np.asarray(out)


def final_refit(rows, val, span=0.003, steps=301, fine=0.00005):
    """Each steady section's exact tempo and phase: the steady grid over the section that puts the most
    evidence on its beats (tempo within +-0.3 %, phase within a quarter beat), coarse then fine."""
    out = np.asarray(rows, float).copy()
    for r in out:
        if is_ramp(r):
            continue
        P0 = 60.0 / r[2]
        n = int((r[1] - r[0]) / P0)
        if n < 16:
            continue
        first = r[3] + np.ceil((r[0] - r[3]) / P0 - 1e-9) * P0
        k = np.arange(n)
        best = (float(val(first + P0 * k).sum()), r[2], first)
        for width, m in ((span, steps), (fine * 20, 41)):
            T = best[1] * (1 + np.linspace(-width, width, m))
            for ph in np.linspace(-0.25, 0.25, 33) * P0:
                g = (best[2] + ph) + (60.0 / T)[:, None] * k[None, :]
                s = val(g.ravel()).reshape(g.shape)
                tot = (s * (g < r[1])).sum(axis=1)
                i = int(np.argmax(tot))
                if tot[i] > best[0] + 1e-9:
                    best = (float(tot[i]), float(T[i]), float(best[2] + ph))
        r[2] = r[5] = best[1]
        r[3] = best[2]
    return out
