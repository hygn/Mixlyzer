"""A recording of the song itself (a phone or a speaker, picked up by a microphone): spectral-peak landmarks after
Wang (2003). The peaks of a spectrogram survive noise, a room and a band-limited speaker; a pair of nearby peaks (both
frequencies and their time apart) is a hash; the song whose hashes come at one time offset from the query's is it.
A DJ's tempo with the key locked (master tempo) changes the times apart: search_tempo maps the query's peaks back to
the song's time at each of TEMPO_FACTORS. A query may keep more peaks than the index (a crowd's calls take a
second's strongest): paired with more of them, the index's pairs are among its own. Voted in NumPy on the index's
rows of the query's hashes (each looked up once); a pitch change without the key lock (varispeed) is not searched.

Measured (simulated speaker, room reverb RT60 0.3 - 0.8 s, noise 10 - 25 dB SNR, 12 s over 385 songs): the right song
first 95 %, about 30 ms; the melody search on the same recordings 75 %.
"""
from __future__ import annotations

import sqlite3

import numpy as np
from numba import njit
from scipy.ndimage import maximum_filter

SR = 8000
N_FFT, HOP = 1024, 256          # 128 ms windows, 32 ms frames
PEAK_T, PEAK_F = 11, 15         # a peak's neighbourhood (frames, bins)
PEAKS_PER_SEC = 10
FAN_OUT = 4                     # pairs per anchor peak
DT_MAX = 48                     # frames (1.5 s) ahead for a pair
DF_MAX = 96                     # bins apart at most
FRAME_SEC = HOP / SR
METHOD = "wang-1"
# a DJ's tempo with the key locked, the song's tempo times this: +-8 % (the CDJs' default range) in 1 % steps (a pair
# 1.5 s apart at most is then off by a quarter frame at most); 1 itself is the plain search
TEMPO_FACTORS = np.array([round(1 + k / 100, 2) for k in range(-8, 9) if k])
_D_ZERO = 1 << 23               # offsets (frames, either sign) packed in 24 bits: +-4.6 hours
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def to_sr(y: np.ndarray, sr: int) -> np.ndarray:
    import soxr
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y if sr == SR else soxr.resample(y, sr, SR).astype(np.float32)


def peaks(y: np.ndarray, per_sec: int = PEAKS_PER_SEC) -> np.ndarray:
    """(frame, bin) of the spectrogram's local maxima, the strongest ``per_sec`` of each second, in time order."""
    n = 1 + max(y.size - N_FFT, 0) // HOP
    if n < 2:
        return np.zeros((0, 2), np.int64)
    from scipy import fft
    frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(np.asarray(y, np.float32), (0, N_FFT)), N_FFT)[:n * HOP:HOP] * _WINDOW
    S = np.log(np.abs(fft.rfft(frames, axis=1))[:, 1:N_FFT // 2] + np.float32(1e-6))
    S -= np.median(S, axis=0, keepdims=True)                         # the spectral tilt flattened
    is_peak = (S == maximum_filter(S, size=(PEAK_T, PEAK_F))) & (S > 0)
    t, f = np.nonzero(is_peak)
    if t.size == 0:
        return np.zeros((0, 2), np.int64)
    v = S[t, f]
    sec = (t * FRAME_SEC).astype(np.int64)
    order = np.lexsort((-v, sec))                                    # each second's peaks, the strongest first
    first = np.searchsorted(sec[order], sec[order], "left")
    k = np.sort(order[np.arange(order.size) - first < per_sec])
    out = np.stack([t[k], f[k] + 1], axis=1).astype(np.int64)
    return out[np.lexsort((out[:, 1], out[:, 0]))]


@njit(cache=True, nogil=True)
def _pairs(t, f, dt_max, df_max, fan_out):
    n = t.size
    hs = np.empty(n * fan_out, np.int64)
    ts = np.empty(n * fan_out, np.int64)
    k = 0
    for i in range(n):
        c = 0
        for j in range(i + 1, n):
            dt = t[j] - t[i]
            if dt > dt_max:
                break
            if dt < 1 or abs(f[j] - f[i]) > df_max:
                continue
            hs[k] = (f[i] << 15) | (f[j] << 6) | dt
            ts[k] = t[i]
            k += 1
            c += 1
            if c == fan_out:
                break
    return hs[:k], ts[:k]


def hashes(p: np.ndarray, fan_out: int = FAN_OUT) -> tuple[np.ndarray, np.ndarray]:
    """(hash, anchor frame) of the peak pairs: f1 (9 bits) | f2 (9 bits) | dt (6 bits)."""
    p = np.asarray(p, dtype=np.int64)
    return _pairs(np.ascontiguousarray(p[:, 0]), np.ascontiguousarray(p[:, 1]), DT_MAX, DF_MAX, fan_out)


def audio_hashes(y: np.ndarray, sr: int, per_sec: int = PEAKS_PER_SEC,
                 fan_out: int = FAN_OUT) -> tuple[np.ndarray, np.ndarray]:
    return hashes(peaks(to_sr(y, sr), per_sec), fan_out)


def search(con: sqlite3.Connection, y: np.ndarray, sr: int, top: int = 10, per_sec: int = PEAKS_PER_SEC,
           fan_out: int = FAN_OUT) -> list[tuple[int, str, float, tuple[float, ...]]]:
    """[(votes, uid, start_sec, places)] best first: the songs whose hashes line up at one offset with the query's;
    places: every offset with half the best's votes or more (a song repeats itself: a chorus again), 3 s apart at
    least, best first."""
    return [m[:4] for m in _matches(con, [audio_hashes(y, sr, per_sec, fan_out)], top)]


def search_tempo(con: sqlite3.Connection, y: np.ndarray, sr: int, top: int = 10,
                 factors: np.ndarray = TEMPO_FACTORS, per_sec: int = PEAKS_PER_SEC,
                 fan_out: int = FAN_OUT) -> list[tuple[int, str, float, tuple[float, ...], float]]:
    """``search`` through a DJ's tempo change with the key locked (master tempo: the frequencies stay, the times
    shrink by the factor): the query's peaks at frame t are the song's at t * factor, so for each factor the peaks
    are moved to the song's time and paired again, and all factors are voted at once. [(votes, uid, start_sec,
    places, factor)] best first, each song at its best factor."""
    p = peaks(to_sr(y, sr), per_sec)
    if p.shape[0] == 0:
        return []
    sets = []
    for s in factors:
        q = p.copy()
        q[:, 0] = np.round(p[:, 0] * float(s)).astype(np.int64)
        sets.append(hashes(q, fan_out))
    return [(*m[:4], float(factors[m[4]])) for m in _matches(con, sets, top)]


def _matches(con: sqlite3.Connection, sets: list[tuple[np.ndarray, np.ndarray]], top: int) -> list[tuple]:
    """The songs whose hashes line up with one of the query's hash sets (hash, anchor frame in the song's time) at one
    offset: [(votes, uid, start_sec, places, set)] best first, each song at its best set."""
    qh = np.concatenate([h for h, _t in sets]) if sets else np.zeros(0, np.int64)
    if qh.size == 0:
        return []
    qt = np.concatenate([t for _h, t in sets])
    qk = np.concatenate([np.full(h.size, k, np.int64) for k, (h, _t) in enumerate(sets)])
    # the index's rows of the query's hashes (each hash looked up once, however many sets have it), joined here
    con.execute("CREATE TEMP TABLE IF NOT EXISTS qh (hash INTEGER PRIMARY KEY)")
    con.execute("DELETE FROM qh")
    con.executemany("INSERT INTO qh VALUES (?)", ((h,) for h in np.unique(qh).tolist()))
    # CROSS JOIN keeps this order (SQLite would otherwise scan the whole index against the query's few hashes)
    rows = np.array(con.execute("SELECT l.hash, l.song, l.t FROM qh CROSS JOIN landmark l ON l.hash = qh.hash")
                    .fetchall(), np.int64).reshape(-1, 3)
    if rows.shape[0] == 0:
        return []
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    lo = np.searchsorted(rows[:, 0], qh, "left")
    n = np.searchsorted(rows[:, 0], qh, "right") - lo
    qi = np.repeat(np.arange(qh.size), n)
    ri = lo[qi] + (np.arange(qi.size) - np.repeat(np.cumsum(n) - n, n))
    d = rows[ri, 2] - qt[qi]
    key = (rows[ri, 1] << 30) | (qk[qi] << 24) | (d + _D_ZERO)
    u, c = np.unique(key, return_counts=True)
    u, c = u[c >= 2], c[c >= 2]
    if u.size == 0:
        return []
    # an offset and its neighbours (the song's and the query's frame grids fall differently) are one match
    v = c.copy()
    for step in (-1, 1):
        j = np.clip(np.searchsorted(u, u + step), 0, u.size - 1)
        v += np.where(u[j] == u + step, c[j], 0)
    song_set = u >> 24
    order = np.lexsort((-v, u >> 30))                  # each song's offsets, the most votes first
    first = order[np.r_[True, (u[order] >> 30)[1:] != (u[order] >> 30)[:-1]]]
    first = first[np.argsort(-v[first], kind="stable")][:top]
    ids = dict(con.execute("SELECT id, uid FROM landmark_song"))
    gap = int(round(3.0 / FRAME_SEC))
    out = []
    for i in first:
        song, k = int(u[i] >> 30), int((u[i] >> 24) & 63)
        if song not in ids:
            continue
        same = np.flatnonzero(song_set == song_set[i])
        same = same[np.argsort(-v[same], kind="stable")]
        places: list[int] = []
        for j in same:
            if v[j] * 2 < v[i] or len(places) == 4:
                break
            dj = int(u[j] & 0xFFFFFF) - _D_ZERO
            if all(abs(dj - p) >= gap for p in places):
                places.append(dj)
        out.append((int(v[i]), ids[song], places[0] * FRAME_SEC, tuple(p * FRAME_SEC for p in places), k))
    return out
