"""A voice's pitch: YIN (de Cheveigne & Kawahara 2002) computed for every frame at once from FFT cross-correlations
(about 0.1 s for 12 s; pYIN took 18 s). A hum and a whistle are searched without a mode: a whistle is told by its
nearly pure, high tone (is_whistle) and tracked in its own range (in the voice's range it is found at a sub-octave:
folded in the match, but less cleanly)."""
from __future__ import annotations

import numpy as np

from analyzer.search.music.melody import HOP, SR

YIN_THRESHOLD = 0.15     # the first dip of the normalized difference below this is the period
YIN_VOICED = 0.3         # a frame whose best dip stays above this has no clear period: unvoiced
YIN_FLOOR_DB = -40.0     # frames this far under the query's loudest are silence
HUM = (65.0, 1000.0)
WHISTLE = (400.0, 4000.0)


def yin(y: np.ndarray, sr: int, fmin: float, fmax: float, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    """(MIDI per hop, NaN unvoiced; the dip of each frame's best period, lower = clearer)."""
    y = np.asarray(y, dtype=np.float64)
    tau_min, tau_max = max(2, int(sr / fmax)), int(np.ceil(sr / fmin))
    w = max(2 * tau_max, 512)                        # the integration window: two periods of the lowest pitch
    length = w + tau_max + 1
    pad = np.concatenate([np.zeros(length // 2), y, np.zeros(length)])
    n_frames = 1 + y.size // hop
    idx = np.arange(n_frames)[:, None] * hop + np.arange(length)[None, :]
    frames = pad[idx]
    nfft = 1 << int(np.ceil(np.log2(2 * length)))
    head = np.zeros_like(frames)
    head[:, :w] = frames[:, :w]
    r = np.fft.irfft(np.conj(np.fft.rfft(head, nfft)) * np.fft.rfft(frames, nfft), nfft)[:, :tau_max + 1]
    sq = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    e0 = sq[:, w][:, None]
    et = sq[:, taus + w] - sq[:, taus]
    d = np.maximum(e0 + et - 2.0 * r, 0.0)
    cm = np.cumsum(d[:, 1:], axis=1)
    cmndf = np.ones_like(d)
    cmndf[:, 1:] = d[:, 1:] * taus[1:][None, :] / np.maximum(cm, 1e-12)
    band = cmndf[:, tau_min:tau_max]
    local_min = np.zeros_like(band, dtype=bool)
    local_min[:, 1:-1] = (band[:, 1:-1] <= band[:, :-2]) & (band[:, 1:-1] <= band[:, 2:])
    dip = local_min & (band < YIN_THRESHOLD)
    first = np.where(dip.any(axis=1), dip.argmax(axis=1), band.argmin(axis=1))
    rows = np.arange(n_frames)
    k = np.clip(first, 1, band.shape[1] - 2)
    a, b, c = band[rows, k - 1], band[rows, k], band[rows, k + 1]
    den = a - 2 * b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    tau = tau_min + k + np.clip(shift, -1, 1)
    rms = np.sqrt(e0[:, 0] / w)
    loud = rms > max(rms.max(), 1e-12) * 10 ** (YIN_FLOOR_DB / 20)
    clarity = band[rows, first]
    voiced = (clarity < YIN_VOICED) & loud
    midi = 69.0 + 12.0 * np.log2(sr / tau / 440.0)
    return np.where(voiced, midi, np.nan), np.where(loud, clarity, 1.0)


WHISTLE_PURITY = 0.6     # a whistle is a nearly pure tone: most of a frame's power in its strongest partial
WHISTLE_MIN_HZ = 950.0   # ... and above a voice's range (HUM's top): below it the voice's range tracks a pure tone
                         # as well (a high hummed line is near pure too); above it only the whistle's range does
                         # (whistles at 1 - 2.5 kHz: the right song first 0.87 against 0.47)


def is_whistle(y: np.ndarray, sr: int = SR) -> bool:
    """Whether a query is whistled: over its loud frames, the median share of the power (80 Hz - 4 kHz) within
    +-40 Hz of the frame's strongest frequency (a voice spreads it over its harmonics) and the median of that
    frequency."""
    y = np.asarray(y, dtype=np.float64)
    n = 2048
    if y.size < n:
        return False
    starts = np.arange(0, y.size - n, n // 2)
    spec = np.abs(np.fft.rfft(y[starts[:, None] + np.arange(n)[None, :]] * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= 80.0) & (freqs <= 4000.0)
    power = spec[:, band].sum(axis=1)
    loud = power > power.max() * 10 ** (YIN_FLOOR_DB / 10)
    if not loud.any():
        return False
    s = spec[loud][:, band]
    peak = s.argmax(axis=1)
    near = np.abs(np.arange(s.shape[1])[None, :] - peak[:, None]) * (freqs[1] - freqs[0]) <= 40.0
    purity = (s * near).sum(axis=1) / np.maximum(s.sum(axis=1), 1e-30)
    return float(np.median(purity)) > WHISTLE_PURITY and float(np.median(freqs[band][peak])) > WHISTLE_MIN_HZ
