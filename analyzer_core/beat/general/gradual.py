"""Gradual tempo: ramps (accelerando / ritardando) and drifting sections.

Ramps. A decoded tempo change of 2 % or more (any change in the BPM octave) is either sudden or gradual.
Musical: in a gradual change the tempo moves along a straight line in time, starting on a beat of the old
grid; the new section continues from the ramp's last beat. The ramp either joins both tempos (continuous), or
one of its ends jumps: an accelerando that starts with a jump on a beat (a fill, a new phrase) and then keeps
speeding up to the new tempo, or a ritardando that ends with a jump into the new tempo.
Engineering: a continuous ramp has one free parameter (its length) besides where it starts; a jump adds one
(the tempo at the jumping end, a fraction of the change). Every candidate is scored over the whole new
section and kept only when it explains the onsets better than the step at its best place (the decoder's
junction is on a 0.5 s grid; the boundary refinement comes later) by one beat of evidence per added
parameter.

Drift. A live band without a click drifts: sections whose tempo moves linearly in time instead of a
staircase of steady sections with re-syncs. Same objective as the decoder (evidence minus J per section)
with one more kind of section (one more parameter, cost 1); neighbouring sections merge into one
steady or drifting section while that raises the objective, the best merge first. A slope that moves no
beat audibly (25 ms) from the steady fit is not a drift. Ramps from the ramp stage are never merged.
"""
from __future__ import annotations

import numpy as np

from core.tempo_rows import ramp_beat_times

from .decode import J as SECTION_COST, is_ramp, steady_beats

MAX_N = 32                 # a ramp is at most 32 beats long
MAX_CHANGE = 0.03          # neighbours further apart in tempo are a tempo change, not a drift
AUDIBLE = 0.025            # a slope must move some beat this far from the steady fit
JUMP_FRACS = np.linspace(0.1, 0.9, 9)   # a ramp's jumping end: its tempo at these fractions of the change
PHASE_STEP = 0.0005        # s: the new grid's evidence is tabulated per phase of its beat at this step


def ramp_fix(rows, val, cost=1.0):
    rows = [np.asarray(r, float).copy() for r in rows]
    out = [rows[0]]
    n_ramps = 0
    Ns = np.arange(2, MAX_N + 1, dtype=np.float64)              # ramp lengths in beats
    kk = np.arange(1, MAX_N, dtype=np.float64)                  # a ramp's inner beats 1 .. N-1
    for B in rows[1:]:
        A = out[-1]
        TA, TB = A[2], B[2]
        if is_ramp(A) or is_ramp(B) or abs(TB / TA - 1.0) < 0.02:
            out.append(B)
            continue
        PA, PB = 60.0 / TA, 60.0 / TB
        Jt = B[0]
        lo_w = max(A[0], Jt - (MAX_N + 2) * PA)
        a_all = steady_beats(A, lo_w, Jt - 1e-6)
        b_all = steady_beats(B, Jt, B[1])
        step = float(val(a_all).sum() + val(b_all).sum())
        # the step at its best place: each grid keeps its phase, any cut within the ramp search window
        hi_c = min(B[1] - PB, Jt + (MAX_N + 2) * PB)
        a_ext = steady_beats(A, lo_w, hi_c)
        b_ext = steady_beats(B, lo_w, B[1])
        cuts = np.unique(np.r_[a_ext, b_ext])
        cuts = cuts[(cuts > lo_w + PA) & (cuts <= hi_c)]
        if cuts.size:
            ca = np.cumsum(np.r_[0.0, val(a_ext)])
            cb = np.cumsum(np.r_[0.0, val(b_ext)])
            ia = np.searchsorted(a_ext, cuts - 1e-6)
            ib = np.searchsorted(b_ext, cuts - 1e-6)
            step = max(step, float(np.max(ca[ia] + (cb[-1] - cb[ib]))))
        va = np.cumsum(np.r_[0.0, val(a_all)])

        # the new grid's evidence from a ramp end e to the section end: per phase of B's beat, the
        # cumulative evidence of that comb, so after(e) = the comb's total - its beats before e
        ph = np.arange(0.0, PB, PHASE_STEP)
        nb = int((B[1] - Jt) / PB) + 2
        comb = Jt + ph[:, None] + PB * np.arange(nb)[None, :]
        cc = np.concatenate([np.zeros((ph.size, 1)), np.cumsum(np.where(comb < B[1], val(comb), 0.0), 1)], 1)

        def after(e):
            d = e - Jt
            j0 = np.floor(d / PB)
            ip = np.round((d - j0 * PB) / PHASE_STEP).astype(np.int64)
            j0 = j0.astype(np.int64) + (ip >= ph.size)
            ip = ip % ph.size
            return cc[ip, -1] - cc[ip, np.minimum(j0, nb)]

        # every (start beat, length) per shape: the ramp's beats under its tempo line T0 -> T1
        ias = np.arange(max(0, a_all.size - MAX_N), a_all.size)
        ias = ias[a_all[ias] > A[0] + PA]
        S = a_all[ias][:, None]
        before = (va[ias] + val(a_all[ias]))[:, None]           # A up to the ramp start, which is a ramp beat
        shapes = [('c', TA, TB)]
        shapes += [('s', TA + f * (TB - TA), TB) for f in JUMP_FRACS]
        shapes += [('e', TA, TA + f * (TB - TA)) for f in JUMP_FRACS]
        cands = []
        for kind, T0, T1 in shapes:
            e = S + 120.0 * Ns[None, :] / (T0 + T1)             # N beats under the line T0 -> T1
            ok = (e > Jt) & (e < B[1] - PB)
            if not ok.any():
                continue
            slope = (T1 - T0) / (e - S)                         # BPM per second
            flat = np.abs(slope[..., None]) < 1e-12
            x = np.where(flat, 60.0 * kk / T0, (-T0 + np.sqrt(np.maximum(T0 * T0 + 120.0 * slope[..., None] * kk, 0.0)))
                         / np.where(flat, 1.0, slope[..., None]))
            inner = np.where(kk < Ns[None, :, None], val(S[..., None] + x), 0.0).sum(-1)
            ev = np.where(ok, before + inner + after(np.where(ok, e, Jt + PB)), -np.inf)
            ev -= 0.0 if kind == 'c' else cost                    # a jump: one more parameter
            for t in np.argsort(ev, axis=None)[::-1][:3]:
                i, n = np.unravel_index(t, ev.shape)
                if np.isfinite(ev[i, n]):
                    cands.append((float(ev[i, n]), kind, int(ias[i]), int(Ns[n]), T0, T1))
        best = None
        # the new tempo refitted, anchored at the ramp end, for the 3 best candidates of each shape
        for kind in 'cse':
            for _, _, ia, N, T0, T1 in sorted((c for c in cands if c[1] == kind), reverse=True)[:3]:
                s = a_all[ia]
                for T2 in TB * (1.0 + np.linspace(-0.004, 0.004, 81)):
                    t0 = T0 * T2 / TB if kind == 's' else T0    # a jump tempo keeps its fraction of the change
                    t1 = T1 if kind == 'e' else T2
                    e = s + 120.0 * N / (t0 + t1)
                    if e <= Jt or e >= B[1] - PB:
                        continue
                    P2 = 60.0 / T2
                    nxt = e + P2 * np.arange(int((B[1] - e) / P2) + 1)
                    ev = float(va[ia] + val(ramp_beat_times(s, e, t0, t1, s, e)).sum() + val(nxt[nxt < B[1]]).sum())
                    ev -= 0.0 if kind == 'c' else cost
                    if best is None or ev > best[0]:
                        best = (ev, s, e, t0, t1, T2)
        if best is not None and best[0] - step > cost:
            _, s, e, t0, t1, T2 = best
            A[1] = s
            out.append(np.array([s, e, t0, s, A[4] if A.size > 4 else 4.0, t1]))
            B[0] = e - 1e-3
            B[2] = T2
            B[3] = e
            if B.size > 5:
                B[5] = T2
            n_ramps += 1
        out.append(B)
    return np.asarray(out), n_ramps


def row_beats(r):
    T0 = r[2]
    T1 = r[5] if r.size > 5 else r[2]
    if abs(T1 - T0) < 1e-9:
        P = 60.0 / T0
        k = np.arange(np.ceil((r[0] - r[3]) / P - 1e-9), np.floor((r[1] - r[3]) / P - 1e-9) + 1)
        return r[3] + P * k
    return ramp_beat_times(r[0], r[1], T0, T1, r[3], r[1])


def _fit(start, end, T0, T1, anchor, val, steady=False):
    """Coordinate search of (T0, T1, anchor) for a linear-tempo section over [start, end) (steady: T1 = T0)."""
    def ev(a, b, c):
        if abs(b - a) > 1e-9:
            g = ramp_beat_times(start, end, a, b, c, end)
        else:
            g = c + (60.0 / a) * np.arange(np.ceil((start - c) * a / 60.0 - 1e-9), np.floor((end - c) * a / 60.0 - 1e-9) + 1)
        return float(val(g).sum()) if g.size else -1e9

    best = (ev(T0, T1, anchor), T0, T1, anchor)
    for width in (0.008, 0.002, 0.0005):
        for _ in range(2):
            _, a, b, c = best
            for T in a * (1 + np.linspace(-width, width, 17)):
                e = ev(T, T if steady else b, c)
                if e > best[0]:
                    best = (e, T, T if steady else b, c)
            _, a, b, c = best
            for T in ([] if steady else b * (1 + np.linspace(-width, width, 17))):
                e = ev(a, T, c)
                if e > best[0]:
                    best = (e, a, T, c)
            _, a, b, c = best
            P = 60.0 / a
            for d in np.linspace(-width * 15, width * 15, 13) * P:
                e = ev(a, b, c + d)
                if e > best[0]:
                    best = (e, a, b, c + d)
    return best


def _max_dev(gs, gd):
    j = np.clip(np.searchsorted(gs, gd), 1, max(1, gs.size - 1))
    return float(np.minimum(np.abs(gs[j - 1] - gd), np.abs(gs[np.minimum(j, gs.size - 1)] - gd)).max())


def drift_merge(rows, val, J=SECTION_COST, cost=1.0):
    rows = [np.asarray(r, float).copy() for r in rows]
    rows = [np.r_[r, r[2]] if r.size == 5 else r for r in rows]
    free = [not is_ramp(r) for r in rows]
    n = 0
    while len(rows) > 1:
        best = None
        for i in range(len(rows) - 1):
            A, B = rows[i], rows[i + 1]
            if not (free[i] and free[i + 1]) or abs(B[2] / A[5] - 1.0) > MAX_CHANGE:
                continue
            cur = float(val(row_beats(A)).sum() + val(row_beats(B)).sum())
            st = None
            for steady in (True, False):
                e, T0, T1, c = _fit(A[0], B[1], A[2], A[2] if steady else B[5], A[3], val, steady)
                row = np.array([A[0], B[1], T0, c, A[4], T1])
                if steady:
                    st = row
                else:
                    gs, gd = row_beats(st), row_beats(row)
                    if gs.size and gd.size and _max_dev(gs, gd) < AUDIBLE:
                        continue
                gain = e - cur + J - (0.0 if steady else cost)
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, i, row)
        for i, A in enumerate(rows):                            # a single section may drift as well
            if not free[i] or is_ramp(A) or (A[1] - A[0]) * A[2] / 60.0 < 32:
                continue
            cur = float(val(row_beats(A)).sum())
            e, T0, T1, c = _fit(A[0], A[1], A[2], A[2], A[3], val)
            row = np.array([A[0], A[1], T0, c, A[4], T1])
            gs, gd = row_beats(A), row_beats(row)
            if not (gs.size and gd.size):
                continue
            gain = e - cur - cost
            if _max_dev(gs, gd) >= AUDIBLE and gain > 0 and (best is None or gain > best[0]):
                best = (gain, i, row, 1)
        if best is None:
            break
        i, m = best[1], best[2]
        k = 1 if len(best) > 3 else 2
        rows[i:i + k] = [m]
        free[i:i + k] = [True]
        n += 1
    return np.asarray(rows), n
