"""Production phrase detection.

The production phrase analyzer is fixed to the current NPZ pipeline:

1. beat-level current production features,
2. whole-song repetition/self-similarity features plus a boundary GBM,
3. probability-peak selection and DP boundary drift correction,
4. segment label GBM probabilities,
5. fixed-boundary label DP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


from analyzer.audio.hpss import HpssSpectra
from analyzer.music.marks import Phrase

from analyzer.phrase.phrase_analyzer import (
    detect_two_stage_phrase_segments,
)


def detect_phrase_segments(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    tempo_segments: np.ndarray,
    *,
    model_path: str | Path,
    hpss: HpssSpectra | None = None,
    frames=None,
    cues=None,
):
    """Detect phrase segments with the current NPZ two-stage GBM pipeline.

    ``hpss``: HPSS spectra of the mono mix, reused by the feature extraction. ``frames``: the
    audio's frame-level features when already made (phrase_analyzer.phrase_song_frames); ``cues``: a stacked
    model's song cues when already made (phrase_analyzer.phrase_cue_frames).
    The phrases are analyzer.music.marks.Phrase.
    """

    return _as_phrases(detect_two_stage_phrase_segments(
        audio,
        int(sample_rate),
        beat_times_sec,
        tempo_segments,
        model_path=model_path,
        hpss=hpss,
        frames=frames,
        cues=cues,
    ))


def _as_phrases(segments) -> list[Phrase]:
    return [Phrase(float(s["start"]), float(s["end"]), str(s["label"])) for s in segments]
