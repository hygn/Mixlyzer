"""The melody match: a voice's pitch against every song's reference lines (analyzer.search.music.melody).

Subsequence DTW (the query may start anywhere in the song), steps (1,1) / (1,2) / (2,1): the query at half to double
the song's tempo. A frame's cost is the distance to the nearer reference line, folded into one octave (a hummer an
octave off, a whistle octaves up), TOL semitones of it free (a voice is no stable oscillator: vibrato, scoops) and
capped at CAP (a wrong note costs no more than a missing one); UNVOICED where either side has no pitch. Every path
covers each query frame once ((2,1) adds its cost twice over two rows), so the score is the sum / the query length
and the recurrence is a plain min.

Transposition: a lane per third of a semitone within +-RANGE semitones of the song's key (people sing a song they
know near its key, Levitin 1994; a third of a semitone takes a hummer's own A, never 440 Hz), all lanes in one pass.
A cascade over all songs (SEARCH_STAGES: frame size, songs kept, lanes), coarse to fine: the key in whole semitones
first, every third next, then the thirds around the key; the songs split over threads,
without the interpreter lock (a search runs beside the GUI).

Measured (benchmark/qbh_research, synthetic queries): 10,000 songs about 0.5 s a query; hummed / whistled / played
vocal lines top-10 0.87 / 0.77 / 0.91.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from analyzer.search.music.melody import BASE, FRAMES, HOP, SR, STEPS, UNV, reduce_pitch

CAP = 2.0
UNVOICED = 1.0
TOL = 0.25
RANGE = 3.0
LANE_STEP = 1.0 / 3.0


@dataclass(frozen=True)
class MelodyMatch:
    uid: str
    cost: float            # mean frame cost (semitones, 0 best)
    transposition: float   # semitones the song is above the query (the voice sang it this much lower)
    start_sec: float       # where in the song the query matched
    end_sec: float
    places: tuple[float, ...] = ()   # every place it matched about as well (a song repeats itself), best first


# A song's other places: its matches within PLACE_COST of its best (a chorus sung again, a verse's second half), at
# least half the query apart, MAX_PLACES at most
PLACE_COST = 0.06
MAX_PLACES = 4


def lane_offsets(rng: float, step: float = LANE_STEP) -> np.ndarray:
    k = int(round(rng / step))
    return (np.arange(-k, k + 1) * step).astype(np.float32)


# The cascade, coarse to fine: (frame size, songs kept, lanes). The first stage, over all songs, tries the whole
# semitones of the range; the second every third of a semitone (the first's key is off by several semitones now and
# then: whistles); the third only the thirds within 2/3 of a semitone of the second's key, the costliest frames on the
# fewest lanes. The coarse stages rank the songs loosely (measured: keeping fewer than 300 / 60 loses songs), so
# they keep as many. Against every lane at every stage: the same songs found, 40 % less work.
SEARCH_STAGES = ((0.4, 300, "semitones"), (0.2, 60, "thirds"), (0.1, 20, 2.0 / 3.0))


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _frame_costs(qi, qok, rp, rvf, off, tol, cap, unv, c, h):
    """c[j]: the query frame against reference frame j-1 (its nearer line; rvf 1 where a line has a pitch). A line at a
    time along the song, so each pass vectorizes; h is scratch."""
    K, m = rp.shape
    twelve = np.float32(12.0); inv = np.float32(1.0 / 12.0); half = np.float32(0.5); one = np.float32(1.0)
    zero = np.float32(0.0); big = np.float32(1e6)
    qs = qi + off
    for j in range(1, m + 1):
        c[j] = big
        h[j] = zero
    for k in range(K):
        r = rp[k]; v = rvf[k]
        for j in range(1, m + 1):
            d = qs - r[j - 1]
            d = d - twelve * np.floor(d * inv + half)
            c[j] = min(c[j], min(max(abs(d) - tol, zero), cap) + (one - v[j - 1]) * big)
            h[j] = max(h[j], v[j - 1])
    for j in range(1, m + 1):
        w = qok * h[j]
        c[j] = w * min(c[j], cap) + (one - w) * unv


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _sdtw(q, qv, rp, rvf, offs, tol, cap, unvoiced, out_cost, out_end):
    n, m, L = q.size, rp.shape[1], offs.size
    INF = np.float32(1e30)
    d2 = np.full((L, m + 1), INF, np.float32)       # row -1: none
    d1 = np.zeros((L, m + 1), np.float32)           # row 0: the query may start anywhere
    d0 = np.empty((L, m + 1), np.float32)
    c = np.empty(m + 1, np.float32); h = np.empty(m + 1, np.float32)
    capf = np.float32(cap); unv = np.float32(unvoiced); tolf = np.float32(tol)
    for i in range(n):
        qok = np.float32(1.0) if qv[i] else np.float32(0.0)
        for l in range(L):
            _frame_costs(q[i], qok, rp, rvf, offs[l], tolf, capf, unv, c, h)
            a0 = d0[l]; a1 = d1[l]; a2 = d2[l]
            a0[0] = INF
            a0[1] = min(a1[0], a2[0] + c[1]) + c[1]
            for j in range(2, m + 1):
                a0[j] = min(min(a1[j - 1], a1[j - 2]), a2[j - 1] + c[j]) + c[j]
        d2, d1, d0 = d1, d0, d2
    for l in range(L):
        v = INF
        e = 0
        a1 = d1[l]
        for j in range(1, m + 1):
            if a1[j] < v:
                v = a1[j]
                e = j
        out_cost[l] = v / max(n, 1)
        out_end[l] = e


@njit(parallel=True, cache=True, nogil=True)
def _scan(q, qv, refs, starts, ends, songs, offs, tol, cap, unvoiced):
    """Per song (indices into starts / ends) and lane (offs: a row of transpositions per song) the cost and end."""
    cost = np.empty(offs.shape, np.float32)
    end = np.empty(offs.shape, np.int64)
    for k in prange(songs.size):
        s = songs[k]
        r = refs[:, starts[s]:ends[s]]
        rp = np.empty(r.shape, np.float32); rv = np.empty(r.shape, np.float32)
        for a in range(r.shape[0]):
            for j in range(r.shape[1]):
                rv[a, j] = 1.0 if r[a, j] != 255 else 0.0
                rp[a, j] = np.float32(BASE) + np.float32(r[a, j]) / np.float32(STEPS)
        _sdtw(q, qv, rp, rv, offs[k], tol, cap, unvoiced, cost[k], end[k])
    return cost, end


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _end_costs(q, qv, rp, rvf, off, tol, cap, unvoiced):
    """The mean cost of the best path ending at each reference frame (1-based; [0] unused), at one transposition."""
    n, m = q.size, rp.shape[1]
    INF = np.float32(1e30)
    d2 = np.full(m + 1, INF, np.float32)
    d1 = np.zeros(m + 1, np.float32)
    d0 = np.empty(m + 1, np.float32)
    c = np.empty(m + 1, np.float32); h = np.empty(m + 1, np.float32)
    for i in range(n):
        qok = np.float32(1.0) if qv[i] else np.float32(0.0)
        _frame_costs(q[i], qok, rp, rvf, off, np.float32(tol), np.float32(cap), np.float32(unvoiced), c, h)
        d0[0] = INF
        d0[1] = min(d1[0], d2[0] + c[1]) + c[1]
        for j in range(2, m + 1):
            d0[j] = min(min(d1[j - 1], d1[j - 2]), d2[j - 1] + c[j]) + c[j]
        d2, d1, d0 = d1, d0, d2
    out = np.empty(m + 1, np.float32)
    for j in range(m + 1):
        out[j] = d1[j] / max(n, 1)
    return out


def _places(ends: np.ndarray, best: float, gap: int) -> list[int]:
    """The ends within PLACE_COST of ``best``, the cheapest first, ``gap`` frames apart at least."""
    order = np.argsort(ends[1:], kind="stable") + 1
    picked: list[int] = []
    for j in order:
        if ends[j] > best + PLACE_COST or len(picked) == MAX_PLACES:
            break
        if all(abs(int(j) - p) >= gap for p in picked):
            picked.append(int(j))
    return picked


@njit(cache=True, nogil=True)
def _start_of(q, qv, rp, rvf, off, tol, cap, unvoiced, end):
    """Where the best path ending at ``end`` (1-based reference frame) starts: the DP again with start pointers."""
    n, m = q.size, end
    rp = rp[:, :m]; rvf = rvf[:, :m]
    INF = np.float32(1e30)
    d2 = np.full(m + 1, INF, np.float32); s2 = np.zeros(m + 1, np.int64)
    d1 = np.zeros(m + 1, np.float32); s1 = np.arange(m + 1)
    d0 = np.empty(m + 1, np.float32); s0 = np.zeros(m + 1, np.int64)
    c = np.empty(m + 1, np.float32); h = np.empty(m + 1, np.float32)
    for i in range(n):
        qok = np.float32(1.0) if qv[i] else np.float32(0.0)
        _frame_costs(q[i], qok, rp, rvf, off, np.float32(tol), np.float32(cap), np.float32(unvoiced), c, h)
        d0[0] = INF
        for j in range(1, m + 1):
            best = d1[j - 1]; bs = s1[j - 1]
            if j >= 2 and d1[j - 2] < best:
                best = d1[j - 2]; bs = s1[j - 2]
            if d2[j - 1] + c[j] < best:
                best = d2[j - 1] + c[j]; bs = s2[j - 1]
            d0[j] = best + c[j]
            s0[j] = bs
        d2, d1, d0 = d1, d0, d2
        s2, s1, s0 = s1, s0, s2
    return s1[m]


class MelodyIndex:
    """The songs' reference lines at each frame size, concatenated: (lines, frames of all songs) u8."""

    def __init__(self) -> None:
        self.uids: list[str] = []
        self._parts: dict[float, list[np.ndarray]] | None = {f: [] for f in FRAMES}
        self.refs: dict[float, np.ndarray] = {}
        self.starts: dict[float, np.ndarray] = {}
        self.ends: dict[float, np.ndarray] = {}

    def add(self, uid: str, lines: dict[float, np.ndarray]) -> None:
        self.uids.append(uid)
        for f in FRAMES:
            self._parts[f].append(np.asarray(lines[f], dtype=np.uint8))

    def freeze(self) -> "MelodyIndex":
        for f in FRAMES:
            parts = self._parts[f]
            K = max((p.shape[0] for p in parts), default=1)
            parts = [p if p.shape[0] == K else np.vstack([p, np.full((K - p.shape[0], p.shape[1]), UNV, np.uint8)])
                     for p in parts]
            lens = np.array([p.shape[1] for p in parts], dtype=np.int64)
            self.ends[f] = np.cumsum(lens)
            self.starts[f] = self.ends[f] - lens
            self.refs[f] = np.ascontiguousarray(np.concatenate(parts, axis=1)) if parts else np.zeros((1, 0), np.uint8)
        self._parts = None
        return self

    def __len__(self) -> int:
        return len(self.uids)

    def search(self, q_pitch: np.ndarray, top: int = 10, rng: float = RANGE,
               hop_sec: float = HOP / SR) -> list[MelodyMatch]:
        """The best songs for a query's pitch (MIDI per ``hop_sec``, NaN unvoiced)."""
        if not self.uids:
            return []
        songs = np.arange(len(self.uids))
        cost = end = q = offs = key = None
        f = 0.1
        for f, keep, around in SEARCH_STAGES:
            if around in ("semitones", "thirds"):
                offs = np.tile(lane_offsets(rng, 1.0 if around == "semitones" else LANE_STEP), (songs.size, 1))
            else:
                # the lanes within `around` of each song's key so far, a third of a semitone apart, kept in the range
                near = lane_offsets(around, LANE_STEP)
                offs = np.clip(key[:, None] + near[None, :], -rng, rng).astype(np.float32)
            q = reduce_pitch(q_pitch, hop_sec, f)
            per, ends = _scan(np.nan_to_num(q).astype(np.float32), np.isfinite(q), self.refs[f], self.starts[f],
                              self.ends[f], songs, np.ascontiguousarray(offs), TOL, CAP, UNVOICED)
            ln = per.argmin(axis=1)
            rows = np.arange(songs.size)
            cost = per[rows, ln]
            order = np.argsort(cost, kind="stable")[:max(keep, top)]
            songs, cost = songs[order], cost[order]
            key, end = offs[rows, ln][order], ends[rows, ln][order]
        out = []
        qf = np.nan_to_num(q).astype(np.float32)
        qv = np.isfinite(q)
        for k in range(min(top, songs.size)):
            s = songs[k]
            song = self.refs[f][:, self.starts[f][s]:self.ends[f][s]]
            song_p = np.where(song == UNV, 0.0, BASE + song / STEPS).astype(np.float32)
            song_v = (song != UNV).astype(np.float32)
            ends = _end_costs(qf, qv, song_p, song_v, np.float32(key[k]), TOL, CAP, UNVOICED)
            spans = []
            for e in _places(ends, float(cost[k]), max(1, qf.size // 2)) or [int(end[k])]:
                # a path spans at most 2 reference frames a query frame: only that much before its end matters
                lo = max(0, e - 2 * qf.size - 2)
                st = lo + _start_of(qf, qv, song_p[:, lo:e], song_v[:, lo:e], np.float32(key[k]), TOL, CAP, UNVOICED,
                                    e - lo)
                spans.append((st * f, e * f))
            out.append(MelodyMatch(self.uids[s], float(cost[k]), float(key[k]), spans[0][0], spans[0][1],
                                   tuple(st for st, _ in spans)))
        return out
