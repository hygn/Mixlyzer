"""A recording of the song itself (a phone or a speaker, picked up by a microphone): spectral-peak landmarks after
Wang (2003). The peaks of a spectrogram survive noise, a room and a band-limited speaker; a pair of nearby peaks (both
frequencies and their time apart) is a hash; the song whose hashes come at one time offset from the query's is it.
Not robust to a changed tempo or pitch (a DJ's pitch fader): the frequencies and the time apart change with them.

Measured (simulated speaker, room reverb RT60 0.3 - 0.8 s, noise 10 - 25 dB SNR, 12 s over 385 songs): the right song
first 95 %, about 30 ms; the melody search on the same recordings 75 %.
"""
from __future__ import annotations

import sqlite3

import numpy as np
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


def to_sr(y: np.ndarray, sr: int) -> np.ndarray:
    import soxr
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y if sr == SR else soxr.resample(y, sr, SR).astype(np.float32)


def peaks(y: np.ndarray) -> np.ndarray:
    """(frame, bin) of the spectrogram's local maxima, the strongest PEAKS_PER_SEC of each second, in time order."""
    n = 1 + max(y.size - N_FFT, 0) // HOP
    if n < 2:
        return np.zeros((0, 2), np.int64)
    idx = np.arange(n)[:, None] * HOP + np.arange(N_FFT)[None, :]
    frames = np.pad(y, (0, max(0, idx.max() + 1 - y.size)))[idx] * np.hanning(N_FFT)[None, :]
    S = np.log(np.abs(np.fft.rfft(frames, axis=1))[:, 1:N_FFT // 2] + 1e-6)
    S -= np.median(S, axis=0, keepdims=True)                         # the spectral tilt flattened
    is_peak = (S == maximum_filter(S, size=(PEAK_T, PEAK_F))) & (S > 0)
    t, f = np.nonzero(is_peak)
    if t.size == 0:
        return np.zeros((0, 2), np.int64)
    v = S[t, f]
    sec = (t * FRAME_SEC).astype(np.int64)
    keep = [i[np.argsort(-v[i])[:PEAKS_PER_SEC]] for s in np.unique(sec) for i in [np.flatnonzero(sec == s)]]
    k = np.sort(np.concatenate(keep))
    out = np.stack([t[k], f[k] + 1], axis=1)
    return out[np.lexsort((out[:, 1], out[:, 0]))]


def hashes(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hash, anchor frame) of the peak pairs: f1 (9 bits) | f2 (9 bits) | dt (6 bits)."""
    hs, ts = [], []
    n = p.shape[0]
    for i in range(n):
        t1, f1 = p[i]
        c = 0
        for j in range(i + 1, n):
            dt = p[j, 0] - t1
            if dt > DT_MAX:
                break
            if dt < 1 or abs(p[j, 1] - f1) > DF_MAX:
                continue
            hs.append((int(f1) << 15) | (int(p[j, 1]) << 6) | int(dt))
            ts.append(int(t1))
            c += 1
            if c == FAN_OUT:
                break
    return np.array(hs, np.int64), np.array(ts, np.int64)


def audio_hashes(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    return hashes(peaks(to_sr(y, sr)))


def search(con: sqlite3.Connection, y: np.ndarray, sr: int,
           top: int = 10) -> list[tuple[int, str, float, tuple[float, ...]]]:
    """[(votes, uid, start_sec, places)] best first: the songs whose hashes line up at one offset with the query's;
    places: every offset with half the best's votes or more (a song repeats itself: a chorus again), 3 s apart at
    least, best first."""
    h, t = audio_hashes(y, sr)
    if h.size == 0:
        return []
    con.execute("CREATE TEMP TABLE IF NOT EXISTS q (hash INTEGER, t INTEGER)")
    con.execute("DELETE FROM q")
    con.executemany("INSERT INTO q VALUES (?, ?)", zip(h.tolist(), t.tolist()))
    rows = con.execute("""SELECT l.song, l.t - q.t AS d, count(*) AS c FROM q JOIN landmark l ON l.hash = q.hash
                          GROUP BY l.song, d HAVING c >= 2 ORDER BY c DESC LIMIT 5000""").fetchall()
    counts = {(song, d): c for song, d, c in rows}
    offsets: dict[int, list[tuple[int, int]]] = {}
    for (song, d), c in counts.items():
        # an offset and its neighbours (the song's and the query's frame grids fall differently) are one match
        offsets.setdefault(song, []).append((c + counts.get((song, d - 1), 0) + counts.get((song, d + 1), 0), d))
    ids = dict(con.execute("SELECT id, uid FROM landmark_song"))
    gap = int(round(3.0 / FRAME_SEC))
    out = []
    for song, found in offsets.items():
        if song not in ids:
            continue
        found.sort(reverse=True)
        places: list[int] = []
        for v, d in found:
            if v * 2 < found[0][0] or len(places) == 4:
                break
            if all(abs(d - p) >= gap for p in places):
                places.append(d)
        out.append((found[0][0], ids[song], found[0][1] * FRAME_SEC, tuple(p * FRAME_SEC for p in places)))
    return sorted(out, key=lambda x: -x[0])[:top]
