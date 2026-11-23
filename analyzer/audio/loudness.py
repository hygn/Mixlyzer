"""A track's loudness and peak as ReplayGain 2.0 takes them: the integrated loudness of ITU-R BS.1770-4 (EBU R 128) and
the peak, measured on the stereo decode the overview waveform is made from (48 kHz: analyzer.pipeline.track; FFmpeg
puts a mono file on both channels at -3 dB, so it reads as its one channel would).

    integrated_loudness(pcm, rate)   LUFS of [N, channels] float PCM (None: nothing above the absolute gate)
    measure(stereo, rate)            Loudness(lufs, peak) of decoded PCM
    measure_track(path)              the same of an audio file, decoded as the analysis decodes it
    replaygain_db(lufs)              the gain to REPLAYGAIN_REFERENCE_LUFS

BS.1770-4: each channel K-weighted (a high shelf, +4 dB above about 1.7 kHz, then a high-pass at about 38 Hz: the
standard's two biquads, made for any rate from their analogue design as the standard's 48 kHz coefficients are), its
mean square over 400 ms blocks every 100 ms, the channels summed (left and right weighted 1), a block's loudness
-0.691 + 10 log10 of that; the blocks under -70 LUFS left out, then those 10 LU under the mean of the rest; the
loudness of the mean of those left. Checked against EBU Tech 3341's cases (tests/test_loudness.py)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt

REPLAYGAIN_REFERENCE_LUFS = -18.0     # ReplayGain 2.0's reference loudness
BLOCK_SEC, STEP_SEC = 0.4, 0.1        # the gating blocks: 400 ms, 75 % overlap
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
LOUDNESS_OFFSET = -0.691              # BS.1770's: a 997 Hz sine at full scale in one channel reads -3.01 LUFS
# the K-weighting's two stages, the parameters BS.1770-4's 48 kHz coefficients come from (libebur128's design, which
# gives the standard's coefficients at 48 kHz and the same filter at any other rate)
SHELF_HZ, SHELF_DB, SHELF_Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
SHELF_VB_EXPONENT = 0.4996667741545416
HIGH_PASS_HZ, HIGH_PASS_Q = 38.13547087602444, 0.5003270373238773


@dataclass(frozen=True)
class Loudness:
    lufs: float | None     # integrated loudness (None: silent, nothing above the absolute gate)
    peak: float            # the largest |sample| of the decode (1.0 full scale; a lossy file's may go past it)


def _k_weighting(rate: int) -> np.ndarray:
    """The K-weighting's second-order sections at ``rate`` (the shelf, then the high-pass)."""
    k = np.tan(np.pi * SHELF_HZ / float(rate))
    vh = 10.0 ** (SHELF_DB / 20.0)
    vb = vh ** SHELF_VB_EXPONENT
    a0 = 1.0 + k / SHELF_Q + k * k
    shelf = [(vh + vb * k / SHELF_Q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / SHELF_Q + k * k) / a0,
             1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / SHELF_Q + k * k) / a0]
    k = np.tan(np.pi * HIGH_PASS_HZ / float(rate))
    a0 = 1.0 + k / HIGH_PASS_Q + k * k
    high_pass = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / HIGH_PASS_Q + k * k) / a0]
    return np.array([shelf, high_pass])


def integrated_loudness(pcm: np.ndarray, rate: int) -> float | None:
    """The integrated loudness (LUFS) of ``pcm`` [N, channels] at ``rate``; None when no block is above the absolute
    gate (silence, or shorter than one block)."""
    x = np.asarray(pcm)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    step = int(round(STEP_SEC * rate))
    per_block = int(round(BLOCK_SEC / STEP_SEC))       # a block is 4 steps
    if step <= 0 or x.shape[0] < step * per_block:
        return None
    sos = _k_weighting(int(rate))
    n_steps = x.shape[0] // step
    power = np.zeros(n_steps - per_block + 1)
    for ch in range(x.shape[1]):
        y = sosfilt(sos, x[:, ch])                    # float64 out of a float32 channel: no copy of the whole track
        y = y[: n_steps * step]
        energy = np.einsum("ij,ij->i", y.reshape(n_steps, step), y.reshape(n_steps, step))   # each step's sum
        sums = np.concatenate(([0.0], np.cumsum(energy)))
        power += (sums[per_block:] - sums[:-per_block]) / (step * per_block)   # channels weighted 1 (left, right)
    with np.errstate(divide="ignore"):
        loud = LOUDNESS_OFFSET + 10.0 * np.log10(power)
    kept = power[loud > ABSOLUTE_GATE_LUFS]
    if kept.size == 0:
        return None
    relative = LOUDNESS_OFFSET + 10.0 * np.log10(kept.mean()) + RELATIVE_GATE_LU
    kept = kept[LOUDNESS_OFFSET + 10.0 * np.log10(kept) > relative]
    return float(LOUDNESS_OFFSET + 10.0 * np.log10(kept.mean()))


def replaygain_db(lufs: float | None) -> float | None:
    """ReplayGain 2.0's track gain (dB) of an integrated loudness."""
    return None if lufs is None else REPLAYGAIN_REFERENCE_LUFS - float(lufs)


def measure(stereo: np.ndarray, rate: int) -> Loudness:
    """The loudness and peak of decoded ``stereo`` [N, 2] at ``rate``."""
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    return Loudness(integrated_loudness(stereo, rate), peak)


def measure_track(path: str, rate: int = 48000) -> Loudness:
    """The file's loudness and peak, decoded as the analysis decodes it for the overview (stereo at ``rate``: the
    overview's, WAVE_OVERVIEW_SAMPLE_RATE)."""
    from analyzer.audio.load import fast_load
    return measure(fast_load(path, rate, stereo=True), rate)
