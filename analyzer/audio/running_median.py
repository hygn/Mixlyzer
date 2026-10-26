"""Median filter along one axis of a 2-D array, equal to ``scipy.ndimage.median_filter`` with a 1-D window
(``size=(1, k)`` / ``(k, 1)``, ``mode="reflect"``), several times faster: the window's values are kept sorted
as it slides (one value out, one in) instead of selected anew at each position. The median is the same
element of the same values, so the result is identical. Falls back to scipy without numba."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def _reflect(i, n):
    # scipy's "reflect": d c b a | a b c d | d c b a (the edge sample repeated), repeated for any distance
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        if i >= n:
            i = 2 * n - i - 1
    return i


def _rows(x, half):
    """Median of every row of ``x`` [R, N] over windows of 2 * half + 1."""
    rows, n = x.shape
    k = 2 * half + 1
    out = np.empty_like(x)
    window = np.empty(k, dtype=x.dtype)
    for r in range(rows):
        for t in range(k):
            window[t] = x[r, _reflect(t - half, n)]
        window.sort()
        out[r, 0] = window[half]
        for j in range(1, n):
            old = x[r, _reflect(j - half - 1, n)]
            new = x[r, _reflect(j + half, n)]
            if old == new:
                out[r, j] = window[half]
                continue
            # the leaving value's slot (one of them, if repeated: they are alike)
            lo, hi = 0, k
            while lo < hi:
                mid = (lo + hi) // 2
                if window[mid] < old:
                    lo = mid + 1
                else:
                    hi = mid
            p = lo
            if new > old:   # the entering value goes right of the slot: shift its lower neighbours left
                while p + 1 < k and window[p + 1] < new:
                    window[p] = window[p + 1]
                    p += 1
            else:           # left of it: shift the higher neighbours right
                while p > 0 and window[p - 1] > new:
                    window[p] = window[p - 1]
                    p -= 1
            window[p] = new
            out[r, j] = window[half]
    return out


try:
    from numba import njit

    _reflect = njit(cache=True, inline="always")(_reflect)
    _rows_numba = njit(cache=True, nogil=True)(_rows)
except Exception:  # pragma: no cover - numba is optional
    _rows_numba = None


def median_filter_1d(x: np.ndarray, size: int, axis: int) -> np.ndarray:
    """``median_filter(x, size=(1, size) or (size, 1), mode="reflect")`` of a 2-D array, along ``axis``."""
    x = np.asarray(x)
    if _rows_numba is None or x.ndim != 2 or size < 1 or size % 2 == 0 or not np.issubdtype(x.dtype, np.floating):
        shape = (1, size) if axis == 1 else (size, 1)
        return median_filter(x, size=shape, mode="reflect")
    if axis == 1:
        return _rows_numba(np.ascontiguousarray(x), size // 2)
    return np.ascontiguousarray(_rows_numba(np.ascontiguousarray(x.T), size // 2).T)
