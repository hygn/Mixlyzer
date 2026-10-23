"""Every song's reference lines for the melody search, made from its audio.

Two lines a frame, because a hummed melody is not always the line a melody extractor calls the melody:
- the predominant melody (analyzer_core.beat.melody_contour) of the stereo centre: the lead vocal is mixed to the
  centre, much of a dense backing wide (each STFT bin weighted by how alike left and right are, Avendano 2003);
  measured against sung lines: right on 55 % of the sung frames, 48 % from the mix;
- the mix's top line: the highest of its TOP_OF strongest contours a frame: of several lines at once a listener takes
  the highest for the melody (the high-voice superiority effect, Trainor et al. 2014), as the riff or lead of an
  instrumental intro, which is often not the strongest contour.

A line is stored per FRAMES frame size (the median of the voiced analysis frames; unvoiced when under half are
voiced) as u8 in thirds of a semitone from MIDI BASE, UNV unvoiced: about 7 KB a song for both lines.
"""
from __future__ import annotations

import numpy as np

SR, HOP = 22050, 256                # the analyzer's rate and hop: its melody extractor's frames
FRAMES = (0.4, 0.2, 0.1)            # the search cascade's frame sizes (s)
UNV = 255
BASE, STEPS = 20.0, 3.0             # u8 = (MIDI - BASE) * STEPS: thirds of a semitone, MIDI 20 - 104.6
LINES = 2
TOP_OF = 3
METHOD = "center+mixtop-1"          # stored with the index: another method re-extracts every song


def reduce_pitch(pitch: np.ndarray, hop_sec: float, frame_sec: float) -> np.ndarray:
    """MIDI per frame_sec frame: the median of the voiced values, NaN when under half are voiced."""
    pitch = np.asarray(pitch, dtype=np.float64)
    per = frame_sec / hop_sec
    n = int(pitch.size / per)
    out = np.full(n, np.nan)
    for i in range(n):
        seg = pitch[int(i * per):int((i + 1) * per)]
        v = seg[np.isfinite(seg)]
        if v.size * 2 >= max(seg.size, 1):
            out[i] = np.median(v)
    return out


def pack(p: np.ndarray) -> np.ndarray:
    out = np.full(p.size, UNV, dtype=np.uint8)
    ok = np.isfinite(p)
    out[ok] = np.clip(np.round((p[ok] - BASE) * STEPS), 0, 254).astype(np.uint8)
    return out


def center(stereo: np.ndarray) -> np.ndarray:
    """The centre of a stereo signal (n, 2): each bin weighted by 2|L R*| / (|L|^2 + |R|^2) to the 8th power."""
    import librosa
    L = librosa.stft(stereo[:, 0], n_fft=2048, hop_length=512)
    R = librosa.stft(stereo[:, 1], n_fft=2048, hop_length=512)
    psi = 2 * np.abs(L * np.conj(R)) / np.maximum(np.abs(L) ** 2 + np.abs(R) ** 2, 1e-12)
    return librosa.istft((psi ** 8) * 0.5 * (L + R), hop_length=512, length=stereo.shape[0]).astype(np.float32)


def candidates(harmonic: np.ndarray, k: int) -> np.ndarray:
    """(k, frames) MIDI (NaN where fewer) of an HPSS harmonic part at SR: the melody extractor's contours (its voicing
    filter kept), the k strongest present at each frame by rank; rank 0 is the extractor's own melody."""
    from analyzer_core.beat import melody_contour as MC
    harmonic = np.ascontiguousarray(harmonic, dtype=np.float32)
    n_frames = 1 + harmonic.size // HOP
    salience, f0s = MC._salience(harmonic, SR, HOP)
    contours = MC._contours(salience, HOP / SR)
    if contours:
        means = np.array([np.mean(c[1]) for c in contours])
        keep = means >= means.mean() - MC.VOICING_NU * means.std()
        contours = [c for c, kk in zip(contours, keep) if kk]
    T = salience.shape[1]
    out = np.full((k, T), np.nan)
    count = np.zeros(T, np.int64)
    for bins, _sal, start in sorted(contours, key=lambda c: -float(np.mean(c[1]))):
        frames = np.arange(start, min(start + len(bins), T))
        b = np.asarray(bins)[: frames.size]
        free = count[frames] < k
        fr = frames[free]
        out[count[fr], fr] = 12.0 * np.log2(f0s[b[free]] / 440.0) + 69.0
        count[fr] += 1
    if T >= n_frames:
        return out[:, :n_frames]
    return np.concatenate([out, np.full((k, n_frames - T), np.nan)], axis=1)


def harmonic_part(y: np.ndarray) -> np.ndarray:
    from analyzer_core.hpss import hpss_audio_with_spectra
    return hpss_audio_with_spectra(np.ascontiguousarray(y, dtype=np.float32), SR)[0]


def reference_lines(stereo: np.ndarray, mix_harmonic: np.ndarray | None = None) -> np.ndarray:
    """(LINES, frames) MIDI per HOP (NaN unvoiced) of a signal at SR, (n,) or (n, 2); ``mix_harmonic``: the mono
    mix's HPSS harmonic part when already made (the analysis's)."""
    y = np.asarray(stereo, dtype=np.float32)
    mono = y.reshape(-1) if y.ndim == 1 else y.mean(axis=1)
    cen = center(y) if y.ndim > 1 and y.shape[1] == 2 else mono
    lead = candidates(harmonic_part(cen), 1)[0]
    mix = candidates(harmonic_part(mono) if mix_harmonic is None else mix_harmonic, TOP_OF)
    n = min(lead.size, mix.shape[1])
    top = np.where(np.isfinite(mix[:, :n]), mix[:, :n], -np.inf).max(axis=0)
    return np.stack([lead[:n], np.where(np.isfinite(top), top, np.nan)])


def packed_lines(lines: np.ndarray) -> dict[float, np.ndarray]:
    """Per frame size the (LINES, frames) u8 lines."""
    out = {}
    for f in FRAMES:
        rows = [pack(reduce_pitch(x, HOP / SR, f)) for x in lines]
        n = min(r.size for r in rows)
        out[f] = np.ascontiguousarray(np.stack([r[:n] for r in rows]))
    return out
