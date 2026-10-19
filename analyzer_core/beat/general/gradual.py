"""Gradual tempo: ramps (accelerando / ritardando) and drifting sections.

Ramps. A decoded tempo change (2 - 30 %) is either sudden or gradual -- one change event either way.
Musical: in a gradual change the tempo moves without a jump: the ramp starts on a beat of the old grid
at the old tempo, ends at the new tempo, and the new section continues from the ramp's last beat.
Engineering: with both end tempos tied to the sections and the new section's phase following the ramp,
a ramp has one free parameter (its length) besides where it starts; it is scored over the whole new
section and kept only when it explains the onsets better (by one beat of evidence) than the step at its
best place (the decoder's junction is on a 0.5 s grid; the boundary refinement comes later).

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


def ramp_fix(rows, val, cost=1.0):
    rows = [np.asarray(r, float).copy() for r in rows]
    out = [rows[0]]
    n_ramps = 0
    for B in rows[1:]:
        A = out[-1]
        TA, TB = A[2], B[2]
        if is_ramp(A) or is_ramp(B) or not (0.02 <= abs(TB / TA - 1.0) <= 0.30):
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
        cands = []
        for ia in range(max(0, a_all.size - MAX_N), a_all.size):
            s = a_all[ia]
            if s <= A[0] + PA:
                continue
            for N in range(2, MAX_N + 1):
                e = s + 120.0 * N / (TA + TB)                   # N beats under the linear tempo line TA -> TB
                if e <= Jt or e >= B[1] - PB:
                    continue
                inner = ramp_beat_times(s, e, TA, TB, s, e)
                after = e + PB * np.arange(int((B[1] - e) / PB) + 1)
                ev = float(va[ia] + val(inner).sum() + val(after[after < B[1]]).sum())
                cands.append((ev, ia, N))
        if not cands:
            out.append(B)
            continue
        cands.sort(reverse=True)
        best = None
        for _, ia, N in cands[:8]:                              # the new tempo refitted, anchored at the ramp end
            s = a_all[ia]
            for T2 in TB * (1.0 + np.linspace(-0.004, 0.004, 81)):
                e = s + 120.0 * N / (TA + T2)
                if e <= Jt or e >= B[1] - PB:
                    continue
                P2 = 60.0 / T2
                after = e + P2 * np.arange(int((B[1] - e) / P2) + 1)
                ev = float(va[ia] + val(ramp_beat_times(s, e, TA, T2, s, e)).sum() + val(after[after < B[1]]).sum())
                if best is None or ev > best[0]:
                    best = (ev, s, e, T2)
        if best is not None and best[0] - step > cost:
            _, s, e, T2 = best
            A[1] = s
            out.append(np.array([s, e, TA, s, A[4] if A.size > 4 else 4.0, T2]))
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
