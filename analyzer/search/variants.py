"""The song fingerprint, from the analysis's stored features only: the beat grid and the overview's 12 log-spaced
bands (20 Hz - 20 kHz).

Frames: beat subdivisions putting the frame rate in FRAMES_PER_MIN, so the fingerprint follows the music's time
whatever the tempo or a start offset, and a grid read at half or double tempo gives the same frames (a query near the
octave's edge also tries the neighbouring one). Per frame, the mean log power of the 12 bands, and from it:
  code  u16  11 bits over all 12 bands (the mix) | 5 bits over the backing bands << 11 (20-200 Hz and 6.3-20 kHz:
             kick, bass, hats, cymbals; a voice barely reaches them): Philips bits, the sign of the change of each
             neighbouring band pair's difference from the previous frame
  rel   u16  the same bits' reliability (|change| > RELIABLE): an unreliable bit is not compared
  voc   i8   the vocal range's (200 Hz - 6.3 kHz) level over the backing bands', 0.5 dB steps: where two versions
             share a backing, which one carries more in the vocal range (an instrumental: less)
Index rows: the backing bits of HASH_FRAMES frames (30 bits), every DB_STRIDE-th frame; a query tries every frame
and each window with its PROBE weakest bits flipped, votes per (track, frame offset) (core.library.variants), and its
candidates are verified on the stored codes: the stretches of WINDOW frames whose reliable backing bits agree >= GOOD
over the offsets (an edit has several), their union covering each track -> the relation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from analyzer.beat.grid_version import BEAT_GRID_VERSION
from analyzer.audio.wave_overview import WAVE_BANDS_VERSION

# the fingerprint's method: this algorithm and the version of each stored feature it reads (core.library.variants keeps
# it with every fingerprint and compares only fingerprints of one method)
METHOD = f"fp1;hash=6;stride=8;beats={BEAT_GRID_VERSION};bands={WAVE_BANDS_VERSION}"
FEATURES = ("wave_bands_np", "wave_bands_ref", "beats_time_sec", "duration_sec")   # what it is made from

BACKING = (0, 1, 2, 3, 10, 11)
VOCAL = (4, 5, 6, 7, 8, 9)
RELIABLE = 0.3
HASH_FRAMES = 6
DB_STRIDE = 8
PROBE = 6
MIN_VOTES = 2
WINDOW = 32           # frames (16 beats, 4 bars at 2 a beat) of local agreement
GOOD = 0.92
SHARED_MIN = 0.5      # the shorter track (the larger cover) mostly made of the other's material: a version, not a shared loop
FULL_MIN = 0.8        # all bands on the covered frames: a shared backing also shares its vocal-free parts
SAME_MIX = 0.98       # all bands agreeing this well: the same mix
EDIT_COVER = 0.2      # one side covered this much more than the other: material cut or added (a TV size)
PARTLY_COVER = 0.85   # both covered alike but less than this: some sections differ
VOCAL_DB = 2.0        # a frame's vocal range this much lower: less vocal there
LESS_VOCAL = 0.25     # share of frames lower on one side (and the other side under a fifth of it): an instrumental
DB_PER_LN = 10.0 / np.log(10.0)
FRAMES_PER_MIN = (200.0, 400.0)   # the frame rate's octave: a grid read at half / double tempo gives the same frames
NEAR_EDGE = 0.06                  # a tempo this close to the octave's edge: the query also tries the neighbouring one

_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int32)


@dataclass(frozen=True)
class Fingerprint:
    code: np.ndarray            # u16 per frame
    rel: np.ndarray             # u16 per frame
    voc: np.ndarray             # i8 per frame
    back: np.ndarray | None = None     # the backing bits per frame (a query's)
    loud: np.ndarray | None = None     # frames not silent
    margin: np.ndarray | None = None   # (frames, 5) |change| of the backing bits: the weakest are probed


def subdivisions(features: Mapping, query: bool = False) -> list[float]:
    """Frames per beat: the one putting the frame rate in FRAMES_PER_MIN; for a query near the edge, also the
    neighbouring octave's (another analysis of the same audio may read the tempo just across it)."""
    b = np.asarray(features["beats_time_sec"], dtype=np.float64)
    if b.size < 3:
        return [1.0]
    bpm = 60.0 / float(np.median(np.diff(b)))
    sub = 1.0
    while bpm * sub < FRAMES_PER_MIN[0]:
        sub *= 2.0
    while bpm * sub >= FRAMES_PER_MIN[1]:
        sub *= 0.5
    out = [sub]
    if query:
        rate = bpm * sub
        if rate < FRAMES_PER_MIN[0] * (1.0 + NEAR_EDGE):
            out.append(sub * 2.0)
        elif rate > FRAMES_PER_MIN[1] * (1.0 - NEAR_EDGE):
            out.append(sub * 0.5)
    return out


def track_fingerprints(features: Mapping) -> list[Fingerprint]:
    """A track's fingerprints (``FEATURES`` of its analysis): the one stored first, then the queries' of the octave's
    neighbour when its frame rate is near the edge."""
    return [fingerprint(features)] + [fingerprint(features, sub) for sub in subdivisions(features, query=True)[1:]]


def _frames_energy(features: Mapping, sub: float) -> np.ndarray:
    """(frames x 12) mean log power per frame, from wave_bands_np / _ref and the beats."""
    lv = np.asarray(features["wave_bands_np"]).astype(np.float64) * (1.0 / 255.0)
    logp = np.log(np.power(lv, 4.0) * np.asarray(features["wave_bands_ref"]).astype(np.float64)[None, :] + 1e-12)
    n = logp.shape[0]
    b = np.asarray(features["beats_time_sec"], dtype=np.float64)
    if b.size < 3 or n == 0 or logp.shape[1] != 12:
        return np.zeros((0, 12))
    width = float(features["duration_sec"]) / n
    if sub >= 1.0:
        k = int(sub)
        edges = (b[:-1, None] + np.diff(b)[:, None] * (np.arange(k)[None, :] / k)).reshape(-1)
        edges = np.append(edges, b[-1])
    else:
        edges = b[::int(round(1.0 / sub))]
    cs = np.vstack([np.zeros((1, 12)), np.cumsum(logp, axis=0)])
    idx = np.clip(np.rint(edges / width).astype(np.int64), 0, n)
    lo, hi = idx[:-1], np.minimum(np.maximum(idx[1:], idx[:-1] + 1), n)
    lo = np.minimum(lo, hi - 1)
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def _changes(E: np.ndarray, bands) -> np.ndarray:
    """The change of each neighbouring band pair's difference from the previous frame (the first frame's: 0)."""
    X = E[:, list(bands)]
    d = X[:, :-1] - X[:, 1:]
    dd = np.zeros_like(d)
    dd[1:] = d[1:] - d[:-1]
    return dd


def fingerprint(features: Mapping, sub: float | None = None) -> Fingerprint:
    """One track's fingerprint at ``sub`` frames a beat (default: its own, `subdivisions`)."""
    E = _frames_energy(features, subdivisions(features)[0] if sub is None else sub)
    full, back = _changes(E, tuple(range(12))), _changes(E, BACKING)
    w11 = (1 << np.arange(11)).astype(np.uint16)
    w5 = (1 << np.arange(5)).astype(np.uint16)
    code = ((full > 0) @ w11).astype(np.uint16) | (((back > 0) @ w5).astype(np.uint16) << 11)
    rel = ((np.abs(full) > RELIABLE) @ w11).astype(np.uint16) | (((np.abs(back) > RELIABLE) @ w5).astype(np.uint16) << 11)
    voc_db = (E[:, list(VOCAL)].mean(axis=1) - E[:, list(BACKING)].mean(axis=1)) * DB_PER_LN
    voc = np.clip(np.rint(voc_db * 2.0), -127, 127).astype(np.int8)
    loud = E.mean(axis=1) > np.log(1e-7)
    return Fingerprint(code, rel, voc, (code >> 11).astype(np.int64), loud, np.abs(back))


def _windows(fp: Fingerprint, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(hash, start) of the windows of HASH_FRAMES loud frames starting at ``starts`` (from frame 1: frame 0 has no
    change)."""
    n = fp.back.size
    starts = starts[(starts >= 1) & (starts + HASH_FRAMES <= n)]
    quiet = np.concatenate([[0], np.cumsum(~fp.loud)])
    starts = starts[quiet[starts + HASH_FRAMES] == quiet[starts]]
    h = np.zeros(starts.size, dtype=np.int64)
    for j in range(HASH_FRAMES):
        h = (h << 5) | fp.back[starts + j]
    return h, starts


def index_rows(fp: Fingerprint) -> tuple[np.ndarray, np.ndarray]:
    """(hashes, frames) stored for a track."""
    return _windows(fp, np.arange(1, fp.back.size, DB_STRIDE))


def query_rows(fp: Fingerprint) -> tuple[np.ndarray, np.ndarray]:
    """(hashes, frames) a query looks up: every window, and the window with each of its PROBE weakest bits flipped."""
    h, starts = _windows(fp, np.arange(1, fp.back.size))
    if h.size == 0:
        return h, starts
    m = fp.margin[starts[:, None] + np.arange(HASH_FRAMES)[None, :]].reshape(h.size, -1)   # frame-major, bits 0..4
    frame, bit = np.divmod(np.argsort(m, axis=1)[:, :PROBE], 5)
    probes = h[:, None] ^ (np.int64(1) << (5 * (HASH_FRAMES - 1 - frame) + bit))
    return (np.concatenate([h, probes.reshape(-1)]),
            np.concatenate([starts, np.repeat(starts, probes.shape[1])]))


@dataclass(frozen=True)
class Verification:
    cover_a: float       # share of a's frames covered by agreeing stretches
    cover_b: float
    covered: int         # a's frames covered
    full: float          # all bands' agreement on them
    voc_lower_a: float   # share of them where a's vocal range is VOCAL_DB lower than b's
    voc_lower_b: float


def verify(a: Fingerprint, b: Fingerprint, offsets) -> Verification:
    """The frames of a and b covered by agreeing stretches (backing bits) over the offsets (b's frame - a's), and on
    them the agreement of all bands and the vocal range's difference."""
    na, nb = a.code.size, b.code.size
    cov_a, cov_b = np.zeros(na, bool), np.zeros(nb, bool)
    full_eq = full_n = 0
    voc_diff = []
    for off in offsets:
        lo, hi = max(1, -off), min(na, nb - off)
        if hi - lo < WINDOW:
            continue
        ca, cb = a.code[lo:hi], b.code[lo + off:hi + off]
        both = a.rel[lo:hi] & b.rel[lo + off:hi + off]
        same = ~(ca ^ cb) & both
        back_n = _POP[both >> 11]
        back_eq = _POP[same >> 11]
        ce = np.convolve(back_eq, np.ones(WINDOW), "valid")
        cw = np.convolve(back_n, np.ones(WINDOW), "valid")
        good = np.flatnonzero(ce / np.maximum(cw, 1) >= GOOD)
        if good.size == 0:
            continue
        mask = np.zeros(hi - lo, bool)
        for i in good:
            mask[i:i + WINDOW] = True
        new = mask & ~cov_a[lo:hi]
        if new.sum() < WINDOW // 2:
            continue
        cov_a[lo:hi] |= mask
        cov_b[lo + off:hi + off] |= mask
        full_eq += int(_POP[(same & 0x7FF)[new]].sum())
        full_n += int(_POP[(both & 0x7FF)[new]].sum())
        voc_diff.append((a.voc[lo:hi][new].astype(float) - b.voc[lo + off:hi + off][new]) * 0.5)
    vd = np.concatenate(voc_diff) if voc_diff else np.zeros(0)
    return Verification(
        cover_a=float(cov_a.mean()) if na else 0.0, cover_b=float(cov_b.mean()) if nb else 0.0,
        covered=int(cov_a.sum()), full=full_eq / max(1, full_n),
        voc_lower_a=float((vd < -VOCAL_DB).mean()) if vd.size else 0.0,
        voc_lower_b=float((vd > VOCAL_DB).mean()) if vd.size else 0.0)


# the relations: of b to a, as stored (core.library.variants); RELATED the ones that are variants
SAME_MIX_KIND = "same"            # the same mix (the same audio, or its edit)
A_INSTRUMENTAL = "a_instrumental"
B_INSTRUMENTAL = "b_instrumental"
OTHER_VOCAL = "vocal"             # another vocal (another singer, take or mix)


def relation(v: Verification) -> str | None:
    """The kind of version a and b are of one song, or None when they are not."""
    if v.covered < 2 * WINDOW or max(v.cover_a, v.cover_b) < SHARED_MIN or v.full < FULL_MIN:
        return None
    if v.full >= SAME_MIX:
        return SAME_MIX_KIND
    if v.voc_lower_a >= LESS_VOCAL and v.voc_lower_a > 5 * v.voc_lower_b:
        return A_INSTRUMENTAL
    if v.voc_lower_b >= LESS_VOCAL and v.voc_lower_b > 5 * v.voc_lower_a:
        return B_INSTRUMENTAL
    return OTHER_VOCAL


def length(cover_self: float, cover_other: float) -> str:
    """How a track's material compares to the other's: "edit" (a shorter edit of it: all of it in the other, the
    other only partly in it), "longer" (the other is), "partly" (some sections differ on both), "" (the same)."""
    if cover_self - cover_other > EDIT_COVER:
        return "edit"
    if cover_other - cover_self > EDIT_COVER:
        return "longer"
    if min(cover_self, cover_other) < PARTLY_COVER:
        return "partly"
    return ""
