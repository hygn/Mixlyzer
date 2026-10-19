"""Grid junctions: what the beat at a hand-over may look like, and the post-process that makes every
junction allowed.

The hand-over interval h runs from the last beat of the old grid to the first beat of the new one
(periods PA before, PB after):
  tempo change  h between the two periods, at most an eighth of a beat outside: the beat at a junction
                has a tempo between the two tempos;
  same tempo    h within an eighth of a beat of one period (the pulse goes on), or a half-beat reset
                (0.5 - 0.625 or 1.375 - 1.5 periods).
Fix: the boundary is fine-corrected (the hand-over pair of beats within two beats of the junction that
explains the onsets best); when the grids' phases cannot meet, either the new grid moves by at most an
eighth of a beat or a short bridge of in-between beats joins them -- whichever explains the onsets
better (a shift moves every beat of the new section, a bridge only the junction).

Also: tiny re-syncs (same tempo within 0.5 %, grids within 1/8 beat) become one steady section when no
beat moves by more than 25 ms -- a steady pulse does not jump, and every row is a marker a DJ may have
to delete.
"""
from __future__ import annotations

import numpy as np

from core.tempo_rows import rebuild_grid_from_segments

from .decode import is_ramp, steady_beats

SLIGHT = 0.125   # an eighth of a beat
SAME = 0.02      # periods this close are one tempo
MAX_MOVE = 0.025


def _beats_of(row, t0, t1):
    """Beats of a steady row's grid in [t0, t1] (the grid extended beyond the row; t1 inclusive)."""
    P = 60.0 / row[2]
    k = np.arange(np.ceil((t0 - row[3]) / P - 1e-9), np.floor((t1 - row[3]) / P + 1e-9) + 1)
    return row[3] + P * k


def _windows(PA, PB):
    lo, hi = min(PA, PB), max(PA, PB)
    if (hi - lo) / lo <= SAME:
        P = 0.5 * (PA + PB)
        return [(P * (1 - SLIGHT), P * (1 + SLIGHT), 'between'), (0.5 * P, 0.625 * P, 'half'), (1.375 * P, 1.5 * P, 'half')]
    return [(lo * (1 - SLIGHT), hi * (1 + SLIGHT), 'between')]


def classify(h, PA, PB):
    for a, b, kind in _windows(PA, PB):
        if a - 1e-9 <= h <= b + 1e-9:
            return kind
    lo, hi = min(PA, PB), max(PA, PB)
    if h < 0.5 * lo * 0.95 or h > 2.0 * hi * 1.05:
        return 'extreme'
    return 'off'


def _shift_to_allowed(h, PA, PB):
    best = None
    for a, b, _ in _windows(PA, PB):
        d = 0.0 if a <= h <= b else (a - h if h < a else b - h)
        if best is None or abs(d) < abs(best):
            best = d
    return best


def _bridge_ok(p, PA, PB):
    lo, hi = min(PA, PB), max(PA, PB)
    return lo * (1 - SLIGHT) <= p <= hi * (1 + SLIGHT)


def _cut_between(a, b, PA, PB):
    lo, hi = max(a, b - PB), min(b, a + PA)
    return 0.5 * (lo + hi) if hi > lo else 0.5 * (a + b)


def _row_beats(row):
    r = np.asarray(row, float).copy()
    if not is_ramp(r):
        return _beats_of(r, r[0], r[1])
    return rebuild_grid_from_segments(r[None, :])


def _bridge(la, fb, A, PA, PB, lo_w, val, steadyA, B, scored=False):
    """The bridge with the fewest skipped beats (scored: the most onset evidence over A, bridge and B)."""
    br = None
    for ka in range(0, min(5, la.size)):
        for kb in range(0, min(5, fb.size)):
            a, b = la[-1 - ka], fb[kb]
            g = b - a
            lo, hi = min(PA, PB) * (1 - SLIGHT), max(PA, PB) * (1 + SLIGHT)
            for n in range(max(1, int(np.ceil(g / hi - 1e-9))), int(np.floor(g / lo + 1e-9)) + 1):
                if not _bridge_ok(g / n, PA, PB):
                    continue
                inner = a + (g / n) * np.arange(1, n)
                if scored:
                    key = float(val(_beats_of(A, lo_w, a) if steadyA else la[la <= a + 1e-9]).sum()) + \
                        float(val(inner).sum()) + float(val(_beats_of(B, b, B[1])).sum())
                else:
                    key = (-(ka + kb), -n, float(val(inner).sum()) if inner.size else 0.0)
                if br is None or key > br[0]:
                    br = (key, a, b, g / n)
                break
    return br


def fix(rows, val):
    rows = [np.asarray(r, float).copy() for r in rows]
    out = [rows[0]]
    for B in rows[1:]:
        A = out[-1]
        PA = 60.0 / (A[5] if is_ramp(A) else A[2])
        PB = 60.0 / B[2]
        J = B[0]
        la = _row_beats(A)
        la = la[la <= A[1] + 1e-3] if not is_ramp(A) else la
        fb = _row_beats(B)
        fb = fb[fb >= B[0]] if not is_ramp(B) else fb
        if la.size and fb.size and fb[0] - la[-1] < 0.05:
            fb = fb[1:]
        if not la.size or not fb.size:
            out.append(B)
            continue
        if classify(fb[0] - la[-1], PA, PB) in ('between', 'half'):
            out.append(B)
            continue
        steadyA, steadyB = not is_ramp(A), not is_ramp(B)
        a_c = _beats_of(A, max(A[0], J - 2 * PA), min(B[1] - PB, J + 2 * PA)) if steadyA else la[-1:]
        b_c = _beats_of(B, max(A[0] + PA, J - 2 * PB), min(B[1], J + 2 * PB)) if steadyB else fb[:1]
        lo_w = J - 3 * max(PA, PB)
        best = None
        for a in a_c:
            ka = _beats_of(A, lo_w, a) if steadyA else la[-3:]
            sa = float(val(ka).sum())
            for b in b_c:
                if b <= a:
                    continue
                d = _shift_to_allowed(b - a, PA, PB)
                if abs(d) > SLIGHT * PB + 1e-9 or (d != 0.0 and not steadyB):
                    continue
                kb = _beats_of(B, b, B[1]) + d if steadyB else fb
                key = (round(sa + float(val(kb).sum()), 6), -abs(d), -abs(0.5 * (a + b) - J))
                if best is None or key > best[0]:
                    best = (key, a, b, d)
        if best is not None and best[3] != 0.0 and steadyB:
            # a shift moves ALL of the new grid; a bridge keeps both grids: the bridge when it explains the
            # onsets at least as well
            br = _bridge(la, fb, A, PA, PB, lo_w, val, steadyA, B, scored=True)
            _, a, b, d = best
            ev_m = float(val(_beats_of(A, lo_w, a) if steadyA else la[la <= a + 1e-9]).sum()) + \
                float(val(_beats_of(B, b, B[1]) + d).sum())
            if br is not None and br[0] >= ev_m:
                _, a2, b2, Pbr = br
                A[1] = a2 + 1e-3
                out.append(np.array([a2 + 1e-3, b2 - 1e-3, 60.0 / Pbr, a2, A[4] if A.size > 4 else 4.0, 60.0 / Pbr]))
                B[0] = b2 - 1e-3
                out.append(B)
                continue
        if best is not None:
            _, a, b, d = best
            B[3] += d
            A[1] = B[0] = _cut_between(a, b + d, PA, PB)
            out.append(B)
            continue
        br = _bridge(la, fb, A, PA, PB, lo_w, val, steadyA, B)            # last resort: a bridge
        if br is None:
            out.append(B)
            continue
        _, a, b, Pbr = br
        A[1] = a + 1e-3
        out.append(np.array([a + 1e-3, b - 1e-3, 60.0 / Pbr, a, A[4] if A.size > 4 else 4.0, 60.0 / Pbr]))
        B[0] = b - 1e-3
        out.append(B)
    return np.asarray(out)


def _try_merge(A, B):
    if is_ramp(A) or is_ramp(B) or abs(B[2] / A[2] - 1.0) > 0.005:
        return None
    P = 60.0 / A[2]
    sh = ((B[3] - A[3]) / P) % 1.0
    if min(sh, 1.0 - sh) >= 0.125:
        return None
    b = np.r_[steady_beats(A, A[0], B[0] - 1e-6), steady_beats(B, B[0], B[1])]
    if b.size < 8:
        return None
    k = np.round((b - b[0]) / P)
    Pm, t0 = np.polyfit(k, b, 1)
    if np.max(np.abs(t0 + Pm * k - b)) > MAX_MOVE:
        return None
    row = A.copy()
    row[1] = B[1]
    row[2] = 60.0 / Pm
    row[3] = t0
    if row.size > 5:
        row[5] = row[2]
    return row


def merge_resyncs(rows):
    res = [np.asarray(rows[0], float).copy()]
    for B in np.asarray(rows, float)[1:]:
        m = _try_merge(res[-1], B)
        if m is not None:
            res[-1] = m
        else:
            res.append(B.copy())
    return np.asarray(res)
