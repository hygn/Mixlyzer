"""Every song's reference lines for the melody search, made from its audio.

Two lines a frame, because a hummed melody is not always the line a melody extractor calls the melody:
- the predominant melody (analyzer.beat.melody_contour) of the stereo centre: the lead vocal is mixed to the
  centre, much of a dense backing wide (each STFT bin weighted by how alike left and right are, Avendano 2003);
  measured against sung lines: right on 55 % of the sung frames, 48 % from the mix; its harmonic part by the mix's
  HPSS mask in the same STFT (harmonic_parts: one STFT a channel, one HPSS, none when the analysis made it);
- the mix's top line: the highest of its TOP_OF strongest contours a frame: of several lines at once a listener takes
  the highest for the melody (the high-voice superiority effect, Trainor et al. 2014), as the riff or lead of an
  instrumental intro, which is often not the strongest contour. In the analysis the mix's contours are the ones its
  frame features found in the same harmonic part (melody_contour.take_contours): not found twice.

The extractor's CQT salience is kept (a salience from the STFT itself, as MELODIA's, took half the time but found
3 - 4 points fewer songs over 416 queries). A line is stored per FRAMES frame size (the median of the voiced
analysis frames; unvoiced when under half are voiced) as u8 in thirds of a semitone from MIDI BASE, UNV unvoiced:
about 7 KB a song for both lines.
"""
from __future__ import annotations

import numpy as np
from numba import njit

SR, HOP = 22050, 256                # the analyzer's rate and hop: its melody extractor's frames
FRAMES = (0.4, 0.2, 0.1)            # the search cascade's frame sizes (s)
UNV = 255
BASE, STEPS = 20.0, 3.0             # u8 = (MIDI - BASE) * STEPS: thirds of a semitone, MIDI 20 - 104.6
LINES = 2
TOP_OF = 3
METHOD = "center+mixtop-2"          # stored with the index: another method re-extracts every song


@njit(cache=True, nogil=True)
def _reduce(pitch, per):
    n = int(pitch.size / per)
    out = np.full(n, np.nan)
    for i in range(n):
        seg = pitch[int(i * per):int((i + 1) * per)]
        v = seg[np.isfinite(seg)]
        if v.size * 2 >= max(seg.size, 1):
            out[i] = np.median(v)
    return out


def reduce_pitch(pitch: np.ndarray, hop_sec: float, frame_sec: float) -> np.ndarray:
    """MIDI per frame_sec frame: the median of the voiced values, NaN when under half are voiced."""
    return _reduce(np.ascontiguousarray(pitch, dtype=np.float64), frame_sec / hop_sec)


def pack(p: np.ndarray) -> np.ndarray:
    out = np.full(p.size, UNV, dtype=np.uint8)
    ok = np.isfinite(p)
    out[ok] = np.clip(np.round((p[ok] - BASE) * STEPS), 0, 254).astype(np.uint8)
    return out


def _frame_peaks(salience: np.ndarray) -> np.ndarray:
    """(frames, PEAKS_PER_FRAME) bins, -1 padded: melody_contour._frame_peaks' peaks for all frames at once (the
    strongest local maxima within its floor, strongest first)."""
    from analyzer.beat import melody_contour as MC
    is_peak = (salience >= np.roll(salience, 1, axis=0)) & (salience >= np.roll(salience, -1, axis=0)) & (salience > 0)
    floor = (1.0 - MC.PEAK_RELATIVE_FLOOR) * salience.max(axis=0)
    cand = np.where(is_peak & (salience >= floor[None, :]), salience, -np.inf)
    k = min(MC.PEAKS_PER_FRAME, cand.shape[0])
    part = np.argpartition(-cand, k - 1, axis=0)[:k]
    order = np.argsort(-np.take_along_axis(cand, part, axis=0), axis=0, kind="stable")
    bins = np.take_along_axis(part, order, axis=0)
    out = np.where(np.isfinite(np.take_along_axis(cand, bins, axis=0)), bins, -1).T
    return np.ascontiguousarray(np.pad(out, ((0, 0), (0, MC.PEAKS_PER_FRAME - k)), constant_values=-1), np.int64)


def _contours(salience: np.ndarray) -> list:
    """melody_contour._contours with the peaks of every frame found at once."""
    from analyzer.beat import melody_contour as MC
    if MC._link_peaks_numba is None:
        return MC._contours(salience, HOP / SR)
    gap = int(round(MC.GAP_SEC / (HOP / SR)))
    entry_contour, entry_bin, entry_salience, start, finish_order = MC._link_peaks_numba(
        _frame_peaks(salience), np.ascontiguousarray(salience, dtype=np.float64), gap, MC.LINK_BINS)
    order = np.argsort(entry_contour, kind="stable")
    bounds = np.searchsorted(entry_contour[order], np.arange(start.size + 1))
    min_len = int(round(MC.MIN_CONTOUR_SEC / (HOP / SR)))
    out = []
    for contour in finish_order:
        entries = order[bounds[contour]:bounds[contour + 1]]
        if entries.size >= min_len:
            out.append((entry_bin[entries], entry_salience[entries], int(start[contour])))
    return out


def _stft(y: np.ndarray) -> np.ndarray:
    from analyzer.audio.spectral import stft
    from analyzer.audio.hpss import HPSS_N_FFT
    return stft(y, n_fft=HPSS_N_FFT, hop_length=HPSS_N_FFT // 4, center=True, pad_mode="constant")


def _istft(spectrum: np.ndarray, length: int) -> np.ndarray:
    from analyzer.audio.spectral import istft
    from analyzer.audio.hpss import HPSS_N_FFT
    return istft(spectrum, dtype=np.float32, n_fft=HPSS_N_FFT, hop_length=HPSS_N_FFT // 4, length=length)


def harmonic_parts(stereo: np.ndarray, mix_mask: np.ndarray | None = None,
                   need_mix: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """(the centre's, the mix's) HPSS harmonic parts of a signal at SR, (n,) or (n, 2), from one STFT a channel: the
    centre is the mix's spectrum with each bin weighted by 2|L R*| / (|L|^2 + |R|^2) to the 8th power (alike in left
    and right), and both take the mix's harmonic mask (``mix_mask``: the analysis's HPSS's, the same STFT frames,
    when made; else made here) -- the centre's own HPSS found the same melodies (quick eval: hum 0.82 / 0.82,
    whistle 0.88 / 0.84) at a second HPSS's cost."""
    from analyzer.audio.hpss import hpss_masks
    y = np.asarray(stereo, dtype=np.float32)
    if y.ndim == 1 or y.shape[1] == 1:
        X = _stft(y.reshape(-1))
        weight = None
    else:
        L, R = _stft(y[:, 0]), _stft(y[:, 1])
        # (2 |L R*| / max(|L|^2 + |R|^2, 1e-12)) ** 8, the same operations in place
        weight = np.abs(L * np.conj(R))
        weight *= 2
        power = np.abs(L)
        power **= 2
        other = np.abs(R)
        other **= 2
        power += other
        del other
        np.maximum(power, 1e-12, out=power)
        weight /= power
        del power
        weight **= 8
        X = L
        X += R
        X *= 0.5
        del R
    mask = mix_mask if mix_mask is not None and mix_mask.shape == X.shape else hpss_masks(np.abs(X))[0]
    n = y.shape[0]
    mix = _istft(X * mask, n) if need_mix or weight is None else None
    if weight is None:
        return mix, mix
    weight *= mask
    X *= weight
    return _istft(X, n), mix


def candidates(harmonic: np.ndarray, k: int) -> np.ndarray:
    """(k, frames) MIDI (NaN where fewer) of an HPSS harmonic part at SR: the melody extractor's contours (its voicing
    filter kept), the k strongest present at each frame by rank; rank 0 is the extractor's own melody. Its contours
    are the analysis's when its frame features found them in this very harmonic part."""
    from analyzer.beat import melody_contour as MC
    harmonic = np.ascontiguousarray(harmonic, dtype=np.float32)
    n_frames = 1 + harmonic.size // HOP
    found = MC.take_contours(harmonic, SR, HOP)
    if found is None:
        salience, f0s = MC._salience(harmonic, SR, HOP)
        contours, T = _contours(salience), salience.shape[1]
    else:
        contours, T, f0s = found
    if contours:
        means = np.array([np.mean(c[1]) for c in contours])
        keep = means >= means.mean() - MC.VOICING_NU * means.std()
        contours = [c for c, kk in zip(contours, keep) if kk]
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
    from analyzer.audio.hpss import hpss_audio_with_spectra
    return hpss_audio_with_spectra(np.ascontiguousarray(y, dtype=np.float32), SR)[0]


def reference_lines(stereo: np.ndarray, mix_harmonic: np.ndarray | None = None,
                    mix_mask: np.ndarray | None = None) -> np.ndarray:
    """(LINES, frames) MIDI per HOP (NaN unvoiced) of a signal at SR, (n,) or (n, 2); ``mix_harmonic`` and
    ``mix_mask``: the mono mix's HPSS harmonic part and harmonic mask when already made (the analysis's)."""
    from concurrent.futures import ThreadPoolExecutor
    cen, mix_h = harmonic_parts(stereo, mix_mask, need_mix=mix_harmonic is None)
    if mix_harmonic is not None:
        mix_h = mix_harmonic
    with ThreadPoolExecutor(2) as pool:             # the two lines' CQTs release the GIL for much of their time
        lead_job = pool.submit(candidates, cen, 1)
        mix = candidates(mix_h, TOP_OF)
        lead = lead_job.result()[0]
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
