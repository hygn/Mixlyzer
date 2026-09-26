"""Numba kernels for core.audio.timestretch.

Imported only by the time stretcher's warm-up thread (compiling here keeps numba out
of the audio thread). The phase kernel is ported from the SELEBI tuning prototype
(v3.1): discrete-edge RTPGHI (phase vocoder done right) with a heap shared by all
channels, identity phase locking around joint spectral peaks and a shared per-bin
stereo rotation. The other kernels fuse the per-frame polar/rectangular conversions
and the percussive-ratio detector.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_TWO_PI = 2.0 * np.pi
_EPS = 1e-12


@njit(cache=True, inline="always")
def _wrap(value):
    return value - _TWO_PI * np.rint(value / _TWO_PI)


# Min-heap of (key, item): key = -magnitude, item = 2 * bin + flag (flag 0: reached
# in time, 1: reached in frequency). Ties sort by item, as the prototype's
# (magnitude, flag, bin) tuples do for a given bin.
@njit(cache=True, inline="always")
def _sift_down(keys, items, start, size):
    key = keys[start]
    item = items[start]
    parent = start
    while True:
        child = 2 * parent + 1
        if child >= size:
            break
        right = child + 1
        if right < size and (
            keys[right] < keys[child] or (keys[right] == keys[child] and items[right] < items[child])
        ):
            child = right
        if keys[child] < key or (keys[child] == key and items[child] < item):
            keys[parent] = keys[child]
            items[parent] = items[child]
            parent = child
        else:
            break
    keys[parent] = key
    items[parent] = item


@njit(cache=True, inline="always")
def _push(keys, items, size, key, item):
    child = size
    while child > 0:
        parent = (child - 1) // 2
        if key < keys[parent] or (key == keys[parent] and item < items[parent]):
            keys[child] = keys[parent]
            items[child] = items[parent]
            child = parent
        else:
            break
    keys[child] = key
    items[child] = item
    return size + 1


@njit(cache=True, nogil=True)
def rtpghi_frame(
    analysis,             # [channels, bins] float64, centre-referenced phase
    magnitude,            # [channels, bins] float32
    joint,                # [bins] float32, sqrt(sum over channels of mag^2)
    previous_analysis,
    previous_synthesis,
    previous_joint,
    analysis_hop,
    synthesis_hop,
    n_fft,
    tolerance,
    lock_strength,
    stereo_coherence,
):
    """Synthesis phases of one frame.

    Bins are integrated in order of magnitude: a bin whose previous-frame coordinate
    is the strongest remaining one is integrated in time (heterodyned phase
    advance scaled by the local stretch); a bin reached from an already integrated
    neighbour in the same frame is integrated in frequency (the analysis phase
    difference scaled by the local stretch). With equal hops both reproduce the
    analysis phases exactly. Bins below ``tolerance`` keep their analysis phase.
    """
    channels, bins = analysis.shape
    out = analysis.copy()
    ha = max(1, analysis_hop)
    hs = max(1, synthesis_hop)
    local_stretch = hs / ha

    max_magnitude = _EPS
    for m in range(bins):
        max_magnitude = max(max_magnitude, previous_joint[m], joint[m])
    floor = tolerance * max_magnitude

    remaining = np.zeros(bins, dtype=np.bool_)
    remaining_count = 0
    keys = np.empty(2 * bins + 4, dtype=np.float64)
    items = np.empty(2 * bins + 4, dtype=np.int64)
    size = 0
    for m in range(bins):
        if joint[m] > floor:
            remaining[m] = True
            remaining_count += 1
            keys[size] = -previous_joint[m]
            items[size] = 2 * m
            size += 1
    for index in range(size // 2 - 1, -1, -1):
        _sift_down(keys, items, index, size)

    while remaining_count > 0:
        if size == 0:
            # Numerical safety only: every active bin was seeded above.
            m = 0
            while m < bins and not remaining[m]:
                m += 1
            omega = _TWO_PI * m / n_fft
            for c in range(channels):
                residual = _wrap(analysis[c, m] - previous_analysis[c, m] - omega * ha)
                out[c, m] = previous_synthesis[c, m] + omega * hs + local_stretch * residual
            remaining[m] = False
            remaining_count -= 1
            size = _push(keys, items, size, -joint[m], 2 * m + 1)

        item = items[0]
        size -= 1
        if size > 0:
            keys[0] = keys[size]
            items[0] = items[size]
            _sift_down(keys, items, 0, size)
        m = item >> 1

        if (item & 1) == 0:
            if remaining[m]:
                omega = _TWO_PI * m / n_fft
                for c in range(channels):
                    residual = _wrap(analysis[c, m] - previous_analysis[c, m] - omega * ha)
                    out[c, m] = previous_synthesis[c, m] + omega * hs + local_stretch * residual
                remaining[m] = False
                remaining_count -= 1
                size = _push(keys, items, size, -joint[m], 2 * m + 1)
            continue

        up = m + 1
        if up < bins and remaining[up]:
            for c in range(channels):
                out[c, up] = out[c, m] + local_stretch * _wrap(analysis[c, up] - analysis[c, m])
            remaining[up] = False
            remaining_count -= 1
            size = _push(keys, items, size, -joint[up], 2 * up + 1)

        down = m - 1
        if down >= 0 and remaining[down]:
            for c in range(channels):
                out[c, down] = out[c, m] - local_stretch * _wrap(analysis[c, m] - analysis[c, down])
            remaining[down] = False
            remaining_count -= 1
            size = _push(keys, items, size, -joint[down], 2 * down + 1)

    # Identity phase locking: bins follow the phase of their region's peak.
    if lock_strength > 0.0 and bins >= 3:
        peak_at = 0
        peak_value = _EPS
        for m in range(bins):
            if joint[m] > peak_value:
                peak_value = joint[m]
                peak_at = m
        threshold = 1e-3 * peak_value
        peaks = np.empty(bins, dtype=np.int64)
        peak_count = 0
        for m in range(1, bins - 1):
            if joint[m] >= joint[m - 1] and joint[m] > joint[m + 1] and joint[m] > threshold:
                peaks[peak_count] = m
                peak_count += 1
        if peak_count == 0:
            peaks[0] = peak_at
            peak_count = 1
        start = 0
        for index in range(peak_count):
            peak = peaks[index]
            stop = bins
            if index + 1 < peak_count:
                stop = (peak + peaks[index + 1]) // 2 + 1
            for m in range(start, stop):
                for c in range(channels):
                    locked = out[c, peak] + _wrap(analysis[c, m] - analysis[c, peak])
                    out[c, m] += lock_strength * _wrap(locked - out[c, m])
            start = stop

    # One rotation per bin shared by all channels keeps the input's L/R phase relation.
    if stereo_coherence and channels > 1:
        for m in range(bins):
            vector_real = 0.0
            vector_imag = 0.0
            for c in range(channels):
                rotation = _wrap(out[c, m] - analysis[c, m])
                vector_real += magnitude[c, m] * np.cos(rotation)
                vector_imag += magnitude[c, m] * np.sin(rotation)
            common = np.arctan2(vector_imag, vector_real)
            for c in range(channels):
                out[c, m] = analysis[c, m] + common

    for c in range(channels):
        for m in range(bins):
            out[c, m] = _wrap(out[c, m])
    return out


@njit(cache=True, nogil=True)
def polar(spectrum):
    """[channels, bins] complex64 -> phase (float64), magnitude (float32), joint."""
    channels, bins = spectrum.shape
    phase = np.empty((channels, bins), dtype=np.float64)
    magnitude = np.empty((channels, bins), dtype=np.float32)
    joint = np.empty(bins, dtype=np.float32)
    for m in range(bins):
        power = 0.0
        for c in range(channels):
            value = spectrum[c, m]
            re = np.float64(value.real)
            im = np.float64(value.imag)
            mag = np.sqrt(re * re + im * im)
            magnitude[c, m] = mag
            phase[c, m] = np.arctan2(im, re)
            power += mag * mag
        joint[m] = np.sqrt(power)
    return phase, magnitude, joint


@njit(cache=True, nogil=True)
def rect(magnitude, phase):
    """magnitude * exp(1j * phase) as complex64."""
    channels, bins = phase.shape
    out = np.empty((channels, bins), dtype=np.complex64)
    for c in range(channels):
        for m in range(bins):
            mag = np.float64(magnitude[c, m])
            out[c, m] = complex(mag * np.cos(phase[c, m]), mag * np.sin(phase[c, m]))
    return out


@njit(cache=True, nogil=True)
def detector_frame(
    spectrum,            # [bins] complex, centre-referenced
    previous_phase,      # [bins] float64 (ignored when has_previous is False)
    has_previous,
    hop,
    n_fft,
    magnitude_floor,
    level,               # running peak level before this frame
    level_decay,
    mpd_lower,
    mpd_upper,
    kick_bin,
    wide_bin,
):
    """MPD percussive ratio, low/wide power ratio, new level and phase of one frame."""
    bins = spectrum.shape[0]
    phase = np.empty(bins, dtype=np.float64)
    magnitude = np.empty(bins, dtype=np.float64)
    peak = 0.0
    for m in range(bins):
        re = np.float64(spectrum[m].real)
        im = np.float64(spectrum[m].imag)
        magnitude[m] = np.sqrt(re * re + im * im)
        phase[m] = np.arctan2(im, re)
        peak = max(peak, magnitude[m])
    level = max(peak, level * level_decay)

    ratio = 0.0
    if has_previous and bins > 1:
        bin_width = _TWO_PI / n_fft
        floor = magnitude_floor * level
        lower = 1.0 - mpd_lower
        upper = 1.0 + mpd_upper
        total = 0.0
        percussive = 0.0
        dt = _wrap(phase[0] - previous_phase[0]) / hop
        mixed = 0.0
        for m in range(bins):
            if m + 1 < bins:
                dt_next = _wrap(phase[m + 1] - previous_phase[m + 1]) / hop
                mixed = _wrap(dt_next - dt) / bin_width
                dt = dt_next
            # the last bin repeats the previous edge's value
            total += magnitude[m]
            if magnitude[m] > floor and mixed > lower and mixed < upper:
                percussive += magnitude[m]
        ratio = percussive / (total + _EPS)

    low = 0.0
    wide = 0.0
    for m in range(1, wide_bin):
        power = magnitude[m] * magnitude[m]
        wide += power
        if m < kick_bin:
            low += power
    return ratio, low / (wide + _EPS), level, phase


def warm() -> None:
    """Compile (or load from the disk cache) the specializations the stretcher uses."""
    bins = 33
    spectrum = np.ones((2, bins), dtype=np.complex64)
    phase, magnitude, joint = polar(spectrum)
    rtpghi_frame(phase, magnitude, joint, phase, phase, joint, 16, 16, 64, 1e-8, 0.5, True)
    rect(magnitude, phase)
    detector_frame(np.ones(bins, dtype=np.complex64), np.zeros(bins), True, 16, 64,
                   0.01, 0.0, 0.99, 0.5, 0.75, 3, 20)
