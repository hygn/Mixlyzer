"""Numba kernel for core.audio.dsp.read_at (the scratch reader).

Loaded by dsp.warm_scratch_numba() on the main thread, so the audio thread never
compiles or loads it. read_at's NumPy path computes the same output (see there for
the kernel); the constants are dsp's READ_* values.
"""
from __future__ import annotations

import numpy as np
from numba import njit

HALF = 16
REACH = 32
STRIDE_STRETCH = 4
SINC_RES = 512
WIN_RES = 4096


@njit(cache=True, nogil=True, inline="always")
def _lerp(table, a):
    i = int(a)
    if i >= table.shape[0] - 1:
        return table[table.shape[0] - 1]
    return table[i] + (a - i) * (table[i + 1] - table[i])


@njit(cache=True, nogil=True, inline="always")
def _clip(x, top):
    return min(max(x, 0.0), top)


@njit(cache=True, nogil=True)
def read(pcm, positions, previous, unit, sinc, win_table, stride_w, out):
    """out = pcm read at positions (``previous``: the position before the first,
    NaN = none)."""
    n = pcm.shape[0]
    ch = pcm.shape[1]
    top = float(n - 1)
    acc = np.zeros(ch)
    lo_f = np.inf
    hi_f = -np.inf
    d_max = 0.0
    before = _clip(positions[0], top) if np.isnan(previous) else _clip(previous, top)
    for m in range(positions.shape[0]):
        pm = _clip(positions[m], top)
        rm = max(abs(pm - before), 1.0)
        before = pm
        if rm > STRIDE_STRETCH:
            lo_f = min(lo_f, pm)
            hi_f = max(hi_f, pm)
            d_max = max(d_max, rm / STRIDE_STRETCH)
            continue
        reach = min(HALF * rm, float(REACH))
        sinc_scale = SINC_RES / rm
        win_scale = WIN_RES / reach
        acc[:] = 0.0
        wsum = 0.0
        for j in range(int(np.floor(pm - reach)) + 1, int(np.floor(pm + reach)) + 1):
            t = abs(j - pm)
            if t >= reach:
                continue
            if rm <= 2.0:
                w = _lerp(unit, t * sinc_scale)
            else:
                w = _lerp(sinc, t * sinc_scale) * _lerp(win_table, t * win_scale)
            wsum += w
            k = min(max(j, 0), n - 1)
            for c in range(ch):
                acc[c] += w * pcm[k, c]
        for c in range(ch):
            out[m, c] = acc[c] / wsum
    if d_max == 0.0:
        return

    # Strided taps. c2: second running integral of the input over the span (frame i
    # held over [i, i + 1) from lo); the first integral is its difference.
    span = REACH * d_max + 2.0
    lo = int(np.floor(lo_f - span))
    size = int(np.ceil(hi_f + span)) + 2 - lo
    c2 = np.zeros((size + 1, ch))
    c1 = np.zeros(ch)
    for i in range(size):
        k = min(max(lo + i, 0), n - 1)
        for c in range(ch):
            v = float(pcm[k, c])
            c2[i + 1, c] = c2[i, c] + c1[c] + 0.5 * v
            c1[c] += v
    taps = stride_w.shape[0]
    g0 = np.zeros(ch)
    g1 = np.zeros(ch)
    before = _clip(positions[0], top) if np.isnan(previous) else _clip(previous, top)
    for m in range(positions.shape[0]):
        pm = _clip(positions[m], top)
        rm = max(abs(pm - before), 1.0)
        before = pm
        if rm <= STRIDE_STRETCH:
            continue
        d = rm / STRIDE_STRETCH
        base = pm - lo + 0.5
        inv = 1.0 / (d * d)
        acc[:] = 0.0
        # the second integral on the grid pm + e * d, e = -REACH .. REACH; each tap is
        # the second difference around its point (a triangle mean over 2 d)
        for e in range(taps + 2):
            q = base + (e - REACH) * d
            k = int(q)
            f = q - k
            kk = min(max(lo + k, 0), n - 1)
            for c in range(ch):
                v = float(pcm[kk, c])
                c2k = c2[k, c]
                g = c2k + f * (c2[k + 1, c] - c2k - 0.5 * v + 0.5 * f * v)
                if e >= 2:
                    acc[c] += stride_w[e - 2] * (g - 2.0 * g1[c] + g0[c]) * inv
                g0[c] = g1[c]
                g1[c] = g
        for c in range(ch):
            out[m, c] = acc[c]


def warm() -> None:
    """Compile (first run) or load (disk cache) the kernel."""
    pcm = np.zeros((256, 2), np.float32)
    out = np.empty((2, 2), np.float32)
    table = np.ones(8)
    read(pcm, np.array([100.0, 120.0]), np.nan, table, table, table, np.ones(3), out)
