"""Time, frequency and decibel conversions used by the analyzer.

These functions intentionally follow librosa's scalar and array conventions so
feature models keep receiving the same coordinates and log-scaled values.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np


def frames_to_time(
    frames: np.ndarray | float | int,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> np.ndarray:
    samples = np.asarray(frames) * int(hop_length)
    if n_fft is not None:
        samples = samples + int(n_fft) // 2
    return samples / float(sr)


def time_to_frames(
    times: np.ndarray | float,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
) -> np.ndarray:
    samples = np.asarray(times) * float(sr)
    if n_fft is not None:
        samples = samples - int(n_fft) // 2
    return np.floor(samples / int(hop_length)).astype(np.int64)


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    return np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))


def hz_to_mel(frequencies: np.ndarray | float, *, htk: bool = False) -> np.ndarray:
    frequencies = np.asanyarray(frequencies, dtype=float)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_min = 0.0
    f_sp = 200.0 / 3.0
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = frequencies >= min_log_hz
    if frequencies.ndim:
        mels[log_region] = min_log_mel + np.log(frequencies[log_region] / min_log_hz) / logstep
    elif log_region:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels: np.ndarray | float, *, htk: bool = False) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=float)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3.0
    frequencies = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    if mels.ndim:
        frequencies[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif log_region:
        frequencies = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return frequencies


def mel_frequencies(
    n_mels: int = 128,
    *,
    fmin: float = 0.0,
    fmax: float = 11025.0,
    htk: bool = False,
) -> np.ndarray:
    min_mel = hz_to_mel(float(fmin), htk=htk)
    max_mel = hz_to_mel(float(fmax), htk=htk)
    return mel_to_hz(np.linspace(min_mel, max_mel, int(n_mels)), htk=htk)


def midi_to_hz(notes: np.ndarray | float) -> np.ndarray:
    return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))


def hz_to_midi(frequencies: np.ndarray | float) -> np.ndarray:
    return 12.0 * (np.log2(np.asanyarray(frequencies)) - np.log2(440.0)) + 69.0


_NOTE_RE = re.compile(r"^\s*([A-Ga-g])([#\u266fb!\u266d]*)(-?\d+)\s*$")
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_hz(note: str) -> float:
    match = _NOTE_RE.match(str(note))
    if match is None:
        raise ValueError(f"Unsupported note spelling: {note!r}")
    name, accidentals, octave_text = match.groups()
    pitch_class = _NOTE_PC[name.upper()]
    for accidental in accidentals:
        pitch_class += 1 if accidental in ("#", "\u266f") else -1
    midi = 12 * (int(octave_text) + 1) + pitch_class
    return float(midi_to_hz(midi))


def cqt_frequencies(
    n_bins: int,
    *,
    fmin: float,
    bins_per_octave: int = 12,
    tuning: float = 0.0,
) -> np.ndarray:
    correction = 2.0 ** (float(tuning) / int(bins_per_octave))
    return correction * float(fmin) * 2.0 ** (np.arange(int(n_bins)) / float(bins_per_octave))


def power_to_db(
    power: np.ndarray,
    *,
    ref: float | Callable[[np.ndarray], float] = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> np.ndarray:
    if amin <= 0:
        raise ValueError("amin must be strictly positive")
    if top_db is not None and top_db < 0:
        raise ValueError("top_db must be non-negative")

    magnitude = np.abs(power)
    ref_value = ref(magnitude) if callable(ref) else abs(ref)
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(max(amin, float(ref_value)))
    if top_db is not None:
        log_spec = np.maximum(log_spec, np.max(log_spec) - float(top_db))
    return log_spec


def amplitude_to_db(
    amplitude: np.ndarray,
    *,
    ref: float | Callable[[np.ndarray], float] = 1.0,
    amin: float = 1e-5,
    top_db: float | None = 80.0,
) -> np.ndarray:
    magnitude = np.abs(amplitude)
    ref_value = ref(magnitude) if callable(ref) else abs(ref)
    return power_to_db(
        np.square(magnitude),
        ref=float(ref_value) ** 2,
        amin=float(amin) ** 2,
        top_db=top_db,
    )
