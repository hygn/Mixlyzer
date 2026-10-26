"""One song's index entry from its audio: the reference lines (analyzer.search.music.melody) and the landmarks
(analyzer.search.music.landmarks). Made by the analysis (which passes its decoded stereo audio, its HPSS's harmonic part
and mask) and by Index Now (core.workers.music_index) for tracks analyzed before."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyzer.search.music import landmarks, melody

SR = melody.SR


@dataclass(frozen=True)
class IndexEntry:
    lines: dict[float, np.ndarray]   # frame size -> (lines, frames) u8
    hashes: np.ndarray
    times: np.ndarray


def index_audio(stereo: np.ndarray, mix_harmonic: np.ndarray | None = None,
                mix_mask: np.ndarray | None = None) -> IndexEntry:
    """The entry of a song decoded at SR ((n, 2) or (n,)); ``mix_harmonic`` and ``mix_mask``: its mono mix's HPSS
    harmonic part and harmonic mask when already made (the analysis's)."""
    y = np.asarray(stereo, dtype=np.float32)
    lines = melody.packed_lines(melody.reference_lines(y, mix_harmonic, mix_mask))
    hashes, times = landmarks.audio_hashes(y if y.ndim == 1 else y.mean(axis=1), SR)
    return IndexEntry(lines, hashes, times)
