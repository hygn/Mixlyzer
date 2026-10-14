"""Small, dependency-light DSP primitives shared by the analyzers."""

from .conversion import (
    amplitude_to_db,
    cqt_frequencies,
    fft_frequencies,
    frames_to_time,
    hz_to_midi,
    mel_frequencies,
    midi_to_hz,
    note_to_hz,
    power_to_db,
    time_to_frames,
)
from .resample import resample_audio
from .spectral import istft, softmask, stft

__all__ = [
    "amplitude_to_db",
    "cqt_frequencies",
    "fft_frequencies",
    "frames_to_time",
    "hz_to_midi",
    "istft",
    "mel_frequencies",
    "midi_to_hz",
    "note_to_hz",
    "power_to_db",
    "resample_audio",
    "softmask",
    "stft",
    "time_to_frames",
]
