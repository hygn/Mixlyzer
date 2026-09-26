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

from core.resource_paths import resource_path

from analyzer_core.hpss import HpssSpectra

from analyzer_core.cue_and_phrase.phrase_analyzer import (
    detect_two_stage_phrase_segments,
)


_ASSET_MODEL = Path(resource_path("assets/weights/phrase_analyzer.npz"))


def detect_phrase_segments(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    tempo_segments: np.ndarray,
    *,
    model_path: str | Path | None = None,
    hpss: HpssSpectra | None = None,
) -> list[dict[str, object]]:
    """Detect phrase segments with the current NPZ two-stage GBM pipeline.

    ``hpss``: HPSS spectra of the mono mix, reused by the feature extraction.
    """

    return detect_two_stage_phrase_segments(
        audio,
        int(sample_rate),
        beat_times_sec,
        tempo_segments,
        model_path=Path(model_path) if model_path is not None else _ASSET_MODEL,
        hpss=hpss,
    )
