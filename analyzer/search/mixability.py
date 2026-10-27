"""Mix candidates: where the end of one track (A) and the start of another (B) sound alike enough to blend, from the
analysis's results (barmix; research/benchmark/fingerprint_research, not committed: listened to as rendered
blends, all but two judged playable on a floor).

Per bar (from one downbeat to the next) a small description of how it sounds:
  energy   the bar's level and its 12 overview bands' balance (dB from their mean)
  rhythm   the percussive part's onsets in three bands (kick / snare-body / hats) at POSITIONS places a beat, the
           strongest in each, averaged over the bar's beats: straight and swung (triplet) feels apart
  harmony  the bar's chroma in the Tonal Interval Space (Bernardes et al. 2016: the chroma's DFT, weighted; near
           points are related pitch sets, so near = consonant together)
Each component's distance is in its unit (SCALE: the typical distance between bars of two random tracks), weighted.
Checked on the user's playlists (121 tracks played one after the other against random pairs): each component's bars
are nearer on the tracks played in a row (AUC 0.59 - 0.61), their sum more so (0.63). A penalty where both bars sing
(the overview's vocal range over its backing) was left out: those played in a row sang together more often than
random pairs (the band holds synths and guitars too), and it took the sum down to 0.58.

The blend (an anison DJ's, from the user's listening and their recorded sets): A plays at least to the end of its
first chorus and leaves where a whole INTERLUDE / OUTRO run of phrases ends; B comes in from its top (its first
phrase after a lead-in; a later phrase head before its first chorus only when none fits from the top, each bar later
costing LATE_ENTRY_PER_BAR); the overlap starts on a phrase head of both and ends on a phrase boundary of both, as
many bars on each side (MIN_BARS .. MAX_BARS), over a steady grid on both (every bar of its beats, none off the
median length by more than STEADY). Among those, the bars most alike (their mean distance, a longer overlap a little
better)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyzer.audio.wave_overview import WAVE_BANDS_VERSION, WaveOverview
from analyzer.beat.grid_version import BEAT_GRID_VERSION
from analyzer.music.beat_geometry import downbeat_beat_indices

# the bars' method: this algorithm and the versions of the results it reads (core.library.mixability keeps it with
# every track's bars; bars of another method are made again)
METHOD = f"mx1;beats={BEAT_GRID_VERSION};bands={WAVE_BANDS_VERSION}"

POSITIONS = 12
TIS_WEIGHTS = np.array([3.0, 8.0, 11.5, 15.0, 14.5, 7.5])     # Bernardes et al. 2016, for audio chroma
WEIGHTS = {"energy": 1.0, "rhythm": 0.8, "harmony": 1.2}
# each component's unit: the median distance between bars of two random tracks (a 420-track library)
SCALE = {"energy": 18.24, "rhythm": 0.668, "harmony": 2.82}
MIN_BARS, MAX_BARS = 2, 16
LENGTH_BONUS = 0.02      # per bar of the overlap
LATE_ENTRY_PER_BAR = 0.05
STEADY = 0.03
DB = 10.0 / np.log(10.0)

LEAD = ("LEAD_IN", "SILENCE")
EXITS = ("INTERLUDE", "OUTRO")
CHORUS = "CHORUS"

# a bar's row: energy (1 + 12), rhythm (3 x POSITIONS), harmony (12)
ENERGY = slice(0, 13)
RHYTHM = slice(13, 13 + 3 * POSITIONS)
HARMONY = slice(13 + 3 * POSITIONS, 25 + 3 * POSITIONS)
COLUMNS = 25 + 3 * POSITIONS


@dataclass(frozen=True)
class BeatFeatures:
    """Per beat (from one beat to the next) what the bars read of the audio."""
    chroma: np.ndarray   # [beats - 1, 12] the mix's CQT chroma, its mean over the beat
    onsets: np.ndarray   # [beats - 1, 3, POSITIONS] the percussive flux's maximum in each part of the beat


@dataclass(frozen=True)
class Bars:
    """A track's bars: start / end (s), beats in each, and their rows (COLUMNS)."""
    start: np.ndarray
    end: np.ndarray
    beats: np.ndarray
    rows: np.ndarray

    @property
    def size(self) -> int:
        return int(self.start.size)

    def bpm(self, i: int) -> float:
        return 60.0 * float(self.beats[i]) / max(float(self.end[i] - self.start[i]), 1e-6)


@dataclass(frozen=True)
class Unit:
    """A phrase: start / end (s) and its label."""
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class Overlap:
    """The best blend found: its score (lower: more alike), A's and B's first bar and the bars it lasts."""
    score: float
    a_bar: int
    b_bar: int
    bars: int


def beat_features(beats, frame_times, chroma, percussive_bands) -> BeatFeatures:
    """Each beat's chroma and onsets, from frame features (analyzer.beat.frame_features: ``chroma``,
    ``percussive_bands`` at ``frame_times``)."""
    b = np.asarray(beats, dtype=np.float64)
    ft = np.asarray(frame_times, dtype=np.float64)
    if b.size < 2 or ft.size == 0:
        return BeatFeatures(np.zeros((0, 12), np.float32), np.zeros((0, 3, POSITIONS), np.float32))
    last = ft.size - 1
    # chroma: the mean of each beat's frames (a beat shorter than a frame: the frame it starts in); reduceat's last
    # part runs to the array's end, so the last beat's end is a part of its own, left out
    idx = np.minimum(np.searchsorted(ft, b), last)
    sums = np.add.reduceat(np.asarray(chroma, dtype=np.float64), idx, axis=1)[:, :-1]
    count = np.maximum(np.diff(idx), 1)
    beat_chroma = (sums / count[None, :]).T
    # onsets: the maximum in each POSITIONS-th of the beat
    edges = (b[:-1, None] + np.diff(b)[:, None] * (np.arange(POSITIONS)[None, :] / POSITIONS)).reshape(-1)
    pidx = np.minimum(np.searchsorted(ft, np.append(edges, b[-1])), last)
    peaks = np.maximum.reduceat(np.asarray(percussive_bands, dtype=np.float64), pidx, axis=1)[:, :-1]
    onsets = peaks.reshape(3, b.size - 1, POSITIONS).transpose(1, 0, 2)
    return BeatFeatures(beat_chroma.astype(np.float32), onsets.astype(np.float32))


def beat_features_of_audio(mono: np.ndarray, sample_rate: int, beats) -> BeatFeatures:
    """Each beat's chroma and onsets from the track's audio (the analysis's mono mix at its rate): a track analyzed
    before the mix candidates existed. The same features as the analysis's."""
    from analyzer.audio.hpss import hpss_audio_with_spectra
    from analyzer.beat.frame_features import chroma_and_percussive_bands

    _harmonic, percussive, _spectra = hpss_audio_with_spectra(mono, int(sample_rate))
    return beat_features(beats, *chroma_and_percussive_bands(mono, percussive, int(sample_rate)))


def tis(chroma: np.ndarray) -> np.ndarray:
    """A chroma's point in the Tonal Interval Space (12 numbers: the weighted DFT's coefficients 1 - 6)."""
    c = chroma / max(float(chroma.sum()), 1e-9)
    f = np.fft.fft(c)[1:7] * TIS_WEIGHTS
    return np.concatenate([f.real, f.imag])


def bars(beats, tempo_segments, wave: WaveOverview, duration_sec: float, features: BeatFeatures) -> Bars:
    """A track's bars, from its beats and tempo rows (the downbeats), its overview and its beats' features."""
    b = np.asarray(beats, dtype=np.float64)
    down = downbeat_beat_indices(b, tempo_segments) if b.size else np.zeros(0, int)
    down = down[down < features.chroma.shape[0] + 1]
    if down.size < 3:
        return Bars(np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int16),
                    np.zeros((0, COLUMNS), np.float32))
    lv = np.asarray(wave.bands, dtype=np.float64) / 255.0
    power_db = DB * np.log(np.power(lv, 4.0) * np.asarray(wave.bands_ref, dtype=np.float64)[None, :] + 1e-12)
    width = float(duration_sec) / max(power_db.shape[0], 1)
    rows = np.zeros((down.size - 1, COLUMNS))
    for k, (i0, i1) in enumerate(zip(down[:-1], down[1:])):
        c0 = min(int(b[i0] / width), power_db.shape[0] - 1)
        c1 = max(int(b[i1] / width), c0 + 1)
        seg = power_db[c0:c1]
        mean = seg.mean(axis=0)
        level = DB * np.log(np.mean(np.power(10.0, mean / 10.0)) + 1e-12)
        rows[k, ENERGY] = np.concatenate([[level], mean - mean.mean()])
        onsets = features.onsets[i0:i1].astype(np.float64).mean(axis=0).reshape(-1)
        rows[k, RHYTHM] = onsets / max(float(np.linalg.norm(onsets)), 1e-9)
        rows[k, HARMONY] = tis(features.chroma[i0:i1].astype(np.float64).mean(axis=0))
    return Bars(b[down[:-1]].astype(np.float32), b[down[1:]].astype(np.float32),
                np.diff(down).astype(np.int16), rows.astype(np.float32))


def distances(A: Bars, B: Bars, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """(len(ia), len(ib)) how unlike A's bars ``ia`` and B's ``ib`` sound: the components' weighted mean distance."""
    X, Y = A.rows[ia].astype(np.float64), B.rows[ib].astype(np.float64)
    D = np.zeros((ia.size, ib.size))
    for name, part in (("energy", ENERGY), ("rhythm", RHYTHM), ("harmony", HARMONY)):
        D += WEIGHTS[name] * np.linalg.norm(X[:, None, part] - Y[None, :, part], axis=2) / SCALE[name]
    return D / sum(WEIGHTS.values())


def bar_at(bars_: Bars, t: float) -> int:
    """The bar starting nearest ``t`` (the track's end: one past its last bar)."""
    if t >= float(bars_.end[-1]):
        return bars_.size
    return int(np.argmin(np.abs(bars_.start - t)))


def steady(bars_: Bars, first: int, n: int) -> bool:
    """Whether bars ``first`` .. ``first + n`` keep one grid: as many beats each, none off their median length by
    more than STEADY."""
    if first + n > bars_.size:
        return False
    beats = bars_.beats[first:first + n]
    length = (bars_.end[first:first + n] - bars_.start[first:first + n]).astype(np.float64)
    median = float(np.median(length))
    return bool(np.all(beats == beats[0]) and np.all(np.abs(length - median) <= STEADY * median))


def exit_windows(A: Bars, units: list[Unit]) -> list[tuple[int, int]]:
    """(first bar, bars) where A can be blended out: from a phrase head inside an INTERLUDE / OUTRO run after its
    first chorus to the run's end (A leaves where the whole run ends, never inside it)."""
    chorus = next((i for i, u in enumerate(units) if u.label == CHORUS), None)
    if chorus is None or A.size == 0:
        return []
    i = next((k for k in range(chorus, len(units)) if units[k].label != CHORUS), len(units))
    out = []
    while i < len(units):
        if units[i].label not in EXITS:
            i += 1
            continue
        j = i
        while j < len(units) and units[j].label in EXITS:
            j += 1
        end_bar = bar_at(A, units[j - 1].end)
        for k in range(i, j):
            first = bar_at(A, units[k].start)
            n = end_bar - first
            if MIN_BARS <= n <= MAX_BARS:
                out.append((first, n))
        i = j
    return out


def entry_head(units: list[Unit]) -> int:
    """B's top: its first phrase after a lead-in."""
    return next((i for i, u in enumerate(units) if u.label not in LEAD), 0)


def entry_windows(B: Bars, units: list[Unit]) -> list[tuple[int, int]]:
    """(first bar, bars) where B can come in: runs of whole phrases before its first chorus, from a phrase head (its
    next phrase starts as A leaves)."""
    chorus = next((i for i, u in enumerate(units) if u.label == CHORUS), None)
    if chorus is None or B.size == 0:
        return []
    out = []
    for i in range(entry_head(units), chorus):
        first = bar_at(B, units[i].start)
        for k in range(i, chorus):
            if k + 1 >= len(units):
                break
            n = bar_at(B, units[k].end) - first
            if n > MAX_BARS:
                break
            if n >= 1:
                out.append((first, n))
    return out


def best_overlap(A: Bars, B: Bars, a_units: list[Unit], b_units: list[Unit]) -> Overlap | None:
    """The best blend of A out into B (the module's note), None when the phrases leave none."""
    a_wins = [w for w in exit_windows(A, a_units) if steady(A, *w)]
    b_wins = [w for w in entry_windows(B, b_units) if steady(B, *w)]
    if not a_wins or not b_wins:
        return None
    head = bar_at(B, b_units[entry_head(b_units)].start)
    ia = np.arange(min(f for f, _ in a_wins), max(f + n for f, n in a_wins))
    ib = np.arange(min(f for f, _ in b_wins), max(f + n for f, n in b_wins))
    D = distances(A, B, ia, ib)
    best = None
    for from_top in (True, False):
        for a0, n in a_wins:
            for b0, m in b_wins:
                if m != n or (from_top and b0 != head):
                    continue
                diag = np.diagonal(D[a0 - ia[0]:a0 - ia[0] + n, b0 - ib[0]:b0 - ib[0] + n])
                s = float(diag.mean()) - LENGTH_BONUS * n + LATE_ENTRY_PER_BAR * max(0, b0 - head)
                if best is None or s < best.score:
                    best = Overlap(s, a0, b0, n)
        if best is not None:
            break
    return best
