"""Chord-aware key analysis (NumPy only).

A small root-equivariant chord CNN reads every beat (bass / treble CQT chroma and STFT chroma of the harmonic part)
as P(chord | beat) over 24 triads; the key evidence of a beat is log sum_c P(c | beat) P(c | key, phrase position),
the chords taken relative to each of the 24 keys (P(c | key, position): the diatonic functions of a phrase's first,
last and other bars). The chord layer was trained through the key loss only (no chord labels), on keys in the DJ /
DAW sense: the scale (key signature, Alphanumeric number) with major / minor by the tonic chord.

The keys follow the bars: a bar Viterbi over the 24 keys where a key change costs J at a phrase start and J + JX
elsewhere (half of it between relative keys, which share the scale), plus an interval prior (a common modulation --
a minor third, a half step up, a whole step -- costs less than a fifth, which is rare as a real modulation and the
commonest misreading). The output signature runs are named major or minor by the song's majority mode.

Weights: ``assets/weights/key_chord_cnn.npz`` (the chord CNN, the chord-in-key tables and the decoder settings);
a missing file means the shipped ``default_key_chord_cnn.npz`` (``foundation/model_files.py``). The models were trained
on 22050 Hz HPSS harmonic audio; other rates are resampled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foundation.model_files import resolve_model

SR = 22050
HOP = 512
MODEL_PATH = "assets/weights/key_chord_cnn.npz"
_SIG_OF = np.r_[np.arange(12), (np.arange(12) + 3) % 12]       # key index (pc + 12 * minor) -> its signature (major tonic)
_REL = _SIG_OF[:, None] == _SIG_OF[None, :]


# ---------------------------------------------------------------- features
def _gated_cqt_chroma(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Bass (C1-C3) and treble (C3-C7) CQT chroma, bins within 20 cents of a note only. -> 2 x [12, T]"""
    import librosa
    from analyzer.audio.conversion import cqt_frequencies, hz_to_midi, note_to_hz

    bpo, n_oct = 36, 6
    fmin = note_to_hz("C1")
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=HOP, bins_per_octave=bpo, n_bins=bpo * n_oct, fmin=fmin)).astype(np.float32)
    midi = hz_to_midi(cqt_frequencies(bpo * n_oct, fmin=fmin, bins_per_octave=bpo))
    near = np.rint(midi)
    ok = 100.0 * np.abs(midi - near) <= 20.0
    pc = near.astype(int) % 12
    bass_band = midi < 48
    out = []
    for band in (bass_band, ~bass_band):
        ch = np.zeros((12, C.shape[1]), np.float32)
        for p in range(12):
            idx = np.flatnonzero(ok & band & (pc == p))
            if idx.size:
                ch[p] = C[idx].sum(axis=0)
        out.append(ch)
    return out[0], out[1]


def _stft_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    from analyzer.audio.spectral import stft

    S = np.abs(stft(y, n_fft=4096, hop_length=HOP, center=True))
    return librosa.feature.chroma_stft(S=np.square(S), sr=sr, hop_length=HOP, n_fft=4096, tuning=None).astype(np.float32)


def _pool(chroma: np.ndarray, ftimes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ftimes, edges)
    out = np.zeros((12, len(edges) - 1), np.float32)
    for i in range(len(edges) - 1):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        if b <= chroma.shape[1]:
            out[:, i] = chroma[:, a:b].mean(axis=1)
    return out


@dataclass
class BeatGrid:
    beat_t: np.ndarray        # beat start times [nb]
    bar: np.ndarray           # bar index of every beat [nb]
    phrase_first: np.ndarray  # per bar: the bar a phrase starts on [nbar]
    pos: np.ndarray           # per beat: 0 phrase-first bar, 1 phrase-last bar, 2 elsewhere


def beat_grid(beats: np.ndarray, downbeats: np.ndarray, phrase_starts: np.ndarray, duration: float) -> BeatGrid:
    b = np.asarray(beats, float)
    b = b[(b >= 0) & (b < duration)]
    nb = b.size
    db = np.asarray(downbeats if downbeats is not None else [], float).ravel()
    if db.size == 0:
        db = b[::4]                                   # no bar information: bars of four beats
    starts = np.unique(np.r_[0, np.searchsorted(b, db - 1e-3)])
    starts = starts[starts < nb]
    bar_t = b[starts]
    first = np.zeros(starts.size, bool)
    for t in np.asarray(phrase_starts, float):
        i = int(np.argmin(np.abs(bar_t - t)))
        if abs(bar_t[i] - t) < 1.0:
            first[i] = True
    last = np.r_[first[1:], True]
    bar = np.searchsorted(starts, np.arange(nb), side="right") - 1
    pos_bar = np.where(first, 0, np.where(last, 1, 2))
    return BeatGrid(beat_t=b, bar=bar.astype(np.int64), phrase_first=first, pos=pos_bar[bar])


def frame_chroma(y_harm: np.ndarray, sr: int) -> np.ndarray:
    """[36, T] at 22050 Hz / hop 512: bass CQT, treble CQT and STFT chroma of the harmonic part (the frames
    the beat inputs are pooled from; the key optimizer caches them)."""
    import librosa

    y = np.asarray(y_harm, np.float32)
    if int(sr) != SR:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=SR).astype(np.float32)
    bass, treble = _gated_cqt_chroma(y, SR)
    st = _stft_chroma(y, SR)
    n = min(bass.shape[1], treble.shape[1], st.shape[1])
    return np.concatenate([bass[:, :n], treble[:, :n], st[:, :n]], axis=0)


def pool_beats(frames: np.ndarray, grid: BeatGrid, duration: float) -> np.ndarray:
    """[nb, 36]: the frame chroma per beat (sqrt, each block over its max)."""
    frames = np.asarray(frames, np.float32)
    ft = np.arange(frames.shape[1]) * HOP / float(SR)
    b = grid.beat_t
    step = float(np.median(np.diff(b))) if b.size > 1 else 0.5
    edges = np.r_[b, min(duration, b[-1] + step)]
    blocks = []
    for blk in range(3):
        c = np.sqrt(np.maximum(_pool(frames[12 * blk:12 * (blk + 1)], ft, edges), 0)).T
        blocks.append(c / (c.max(axis=1, keepdims=True) + 1e-9))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def beat_inputs(y_harm: np.ndarray, sr: int, grid: BeatGrid, duration: float) -> np.ndarray:
    """[nb, 36]: bass CQT, treble CQT and STFT chroma per beat (sqrt, each block over its max)."""
    return pool_beats(frame_chroma(y_harm, sr), grid, duration)


# ---------------------------------------------------------------- the chord CNN
def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x [n, cin], w [cout, cin, k] ('same' padding) -> [n, cout]"""
    k = w.shape[2]
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)))
    n = x.shape[0]
    out = np.zeros((n, w.shape[0]), np.float32) + b[None, :]
    for j in range(k):
        out += xp[j:j + n] @ w[:, :, j].T
    return out


def _log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = z.max(axis=axis, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=axis, keepdims=True))


WEIGHTS = ("c1_w", "c1_b", "c2_w", "c2_b", "out_w", "out_b", "tab")
SETTINGS = ("J", "JX", "J_REL", "LAM", "MINEX", "iv_cost")


def read_model_arrays(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Every array of a key model file (the user's file when it exists, else the shipped one)."""
    with np.load(resolve_model(path or MODEL_PATH), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in WEIGHTS + SETTINGS}


class ChordKeyModel:
    def __init__(self, path: str | Path | None = None, arrays: dict | None = None):
        z = arrays if arrays is not None else read_model_arrays(path)
        self.w = {k: np.asarray(z[k], np.float32) for k in WEIGHTS}
        self.J = float(z["J"]); self.JX = float(z["JX"]); self.J_REL = float(z["J_REL"])
        self.LAM = float(z["LAM"]); self.MINEX = int(z["MINEX"])
        self.iv_cost = np.asarray(z["iv_cost"], float)

    def chords(self, x: np.ndarray) -> np.ndarray:
        """[nb, 36] -> log P(chord | beat) [nb, 24] (roots 0-11 major, 12-23 minor)."""
        nb = x.shape[0]
        logits = np.zeros((nb, 2, 12), np.float32)
        for r in range(12):
            xr = np.concatenate([np.roll(x[:, 12 * blk:12 * (blk + 1)], -r, axis=1) for blk in range(3)], axis=1)
            h = np.maximum(_conv1d(xr, self.w["c1_w"], self.w["c1_b"]), 0)
            h = np.maximum(_conv1d(h, self.w["c2_w"], self.w["c2_b"]), 0) + h
            logits[:, :, r] = _conv1d(h, self.w["out_w"], self.w["out_b"])
        return _log_softmax(logits.reshape(nb, 24), -1)

    def key_loglik(self, logpc: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """-> [nb, 24 keys] (key index pc + 12 * minor): log sum_c P(c | beat) P(c | key, phrase position)."""
        R = _log_softmax(self.w["tab"], -1)                   # [2 modes, 3 positions, 24 relative chords]
        out = np.zeros((logpc.shape[0], 24), np.float32)
        for m in range(2):
            Rb = R[m][pos]                                     # [nb, 24]
            for k in range(12):
                lp = np.concatenate([np.roll(logpc[:, :12], -k, axis=1), np.roll(logpc[:, 12:], -k, axis=1)], axis=1)
                z = lp + Rb
                zm = z.max(axis=1, keepdims=True)
                out[:, m * 12 + k] = (zm[:, 0] + np.log(np.exp(z - zm).sum(axis=1)))
        return out

    # ------------------------------------------------------------ decoder
    def decode(self, L: np.ndarray, grid: BeatGrid, duration: float) -> tuple[np.ndarray, np.ndarray]:
        """Bar Viterbi -> (key_segments [n, 4] rows (pc, mode, t0, t1), key index per bar)."""
        bar = grid.bar
        nbar = int(bar.max()) + 1
        E = np.zeros((nbar, 24))
        for k in range(24):
            E[:, k] = np.bincount(bar, weights=L[:, k], minlength=nbar)
        cost = np.where(grid.phrase_first[:nbar], self.J, self.J + self.JX)
        C = self.LAM * self.iv_cost[(_SIG_OF[None, :] - _SIG_OF[:, None]) % 12]
        score = E[0].copy()
        back = np.zeros((nbar, 24), int)
        for i in range(1, nbar):
            trans = score[:, None] - (np.where(_REL, self.J_REL * cost[i], cost[i]) + C)
            np.fill_diagonal(trans, score)
            back[i] = np.argmax(trans, axis=0)
            score = trans[back[i], np.arange(24)] + E[i]
        path = np.empty(nbar, int)
        path[-1] = int(np.argmax(score))
        for i in range(nbar - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        if self.MINEX > 0:
            changed = True
            while changed:
                changed = False
                starts = np.r_[0, np.flatnonzero(np.diff(path)) + 1]
                ends = np.r_[starts[1:], nbar]
                for j in range(1, len(starts) - 1):
                    if path[starts[j - 1]] == path[starts[j + 1]] and ends[j] - starts[j] < self.MINEX:
                        path[starts[j]:ends[j]] = path[starts[j - 1]]
                        changed = True
                        break
        bar_t = np.array([grid.beat_t[np.flatnonzero(bar == i)[0]] for i in range(nbar)])
        bar_len = np.diff(np.r_[bar_t, duration])
        minor = float(np.sum(bar_len[path >= 12])) > 0.5 * float(np.sum(bar_len))
        sig = _SIG_OF[path]
        rows: list[list[float]] = []
        for i in range(nbar):
            t0 = 0.0 if i == 0 else float(bar_t[i])
            if rows and rows[-1][0] == sig[i]:
                continue
            if rows:
                rows[-1][3] = t0
            rows.append([float(sig[i]), 0.0, t0, float(duration)])
        out = np.asarray(rows, float)
        if minor:
            out[:, 0] = (out[:, 0] + 9) % 12
            out[:, 1] = 1.0
        key_per_bar = np.where(minor, (sig + 9) % 12 + 12, sig)
        return out, key_per_bar


_MODEL_CACHE: dict[str, ChordKeyModel] = {}


def load_model(path: str | Path | None = None) -> ChordKeyModel:
    key = str(resolve_model(path or MODEL_PATH))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = ChordKeyModel(key)
    return _MODEL_CACHE[key]


def key_evidence(y_harm: np.ndarray, sr: int, beats: np.ndarray, downbeats, phrase_starts, duration: float,
                 model_path: str | Path | None = None) -> tuple[ChordKeyModel, BeatGrid, np.ndarray]:
    """(model, beat grid, per-beat key log-likelihoods [nb, 24])."""
    model = load_model(model_path)
    grid = beat_grid(beats, downbeats, np.asarray(phrase_starts if phrase_starts is not None else [], float), duration)
    if grid.beat_t.size < 8:
        raise ValueError("too few beats for the key analysis")
    x = beat_inputs(y_harm, sr, grid, duration)
    return model, grid, model.key_loglik(model.chords(x), grid.pos)


# The phrases where tracks are mixed: the static analysis's key is the dynamic analysis's key there.
MIX_LABELS = ("INTRO", "OUTRO", "INTERLUDE")


def static_key(segments: np.ndarray, phrases, duration: float) -> tuple[int, int]:
    """One key for the whole track from the dynamic analysis's key sections: the key that holds the most time in
    the intro, outro and interlude phrases; without them, the key that holds the most time. -> (pc, mode)"""
    spans = []
    for p in phrases or ():
        label = str(p.get("label", "") if isinstance(p, dict) else p[2]).upper()
        if label in MIX_LABELS:
            a = float(p["start"] if isinstance(p, dict) else p[0])
            b = float(p["end"] if isinstance(p, dict) else p[1])
            spans.append((a, b))
    if not spans:
        spans = [(0.0, float(duration))]
    held: dict[tuple[int, int], float] = {}
    for pc, mode, t0, t1 in np.asarray(segments, float):
        overlap = sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in spans)
        key = (int(pc), int(mode))
        held[key] = held.get(key, 0.0) + overlap
    if not any(v > 0 for v in held.values()):
        for pc, mode, t0, t1 in np.asarray(segments, float):
            held[(int(pc), int(mode))] = held.get((int(pc), int(mode)), 0.0) + (t1 - t0)
    return max(held, key=held.get)


def track_keys(model: ChordKeyModel, grid: BeatGrid, L: np.ndarray, duration: float, dynamic: bool = True,
               phrases=None) -> tuple[np.ndarray, np.ndarray]:
    """The keys from the evidence: the decoder's key sections (dynamic), or one key for the whole track (static):
    the dynamic analysis's key in the intro / outro / interludes (``phrases``: dicts or (start, end, label))."""
    segments, key_per_bar = model.decode(L, grid, duration)
    if dynamic:
        return np.asarray(key_per_bar)[grid.bar], segments
    pc, mode = static_key(segments, phrases, duration)
    return np.full(grid.beat_t.size, pc + 12 * mode), np.asarray([[pc, mode, 0.0, duration]], float)


def analyze_keys(y_harm: np.ndarray, sr: int, beats: np.ndarray, downbeats, phrase_starts,
                 duration: float, model_path: str | Path | None = None,
                 dynamic: bool = True, phrases=None) -> tuple[np.ndarray, np.ndarray]:
    """Key segments of a track (dynamic: key changes; static: one key, the dynamic analysis's key in the intro /
    outro / interludes, given ``phrases`` with their labels).
    -> (key per beat [nb] as key index pc + 12 * minor,
    key_segments [n, 4] rows (pitch class, mode 0 major / 1 minor, start s, end s) covering 0..duration)."""
    model, grid, L = key_evidence(y_harm, sr, beats, downbeats, phrase_starts, duration, model_path)
    return track_keys(model, grid, L, duration, dynamic, phrases)


def selection_key(grid: BeatGrid, L: np.ndarray, start: float, end: float) -> tuple[int, int]:
    """The one key of a time span: the most evidence summed over its beats. -> (pitch class, mode)"""
    sel = (grid.beat_t >= start - 1e-6) & (grid.beat_t < end - 1e-6)
    if not sel.any():
        raise ValueError("no beat in the selection")
    k = int(np.argmax(L[sel].sum(axis=0)))
    return k % 12, k // 12
