"""Production phrase detection.

The production phrase analyzer is fixed to the benchmark-winning pipeline:

1. beat-level current production features,
2. hard boundary GBM via ``predict()``,
3. DP boundary drift correction,
4. segment label GBM probabilities,
5. fixed-boundary label DP.

Legacy bar-level joint phrase detection code is archived under
``benchmark/old/production_legacy_phrase`` and is not used by production.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.resource_paths import resource_path

from analyzer_core.cue_and_phrase.phrase_analyzer import (
    detect_two_stage_phrase_segments,
)


_ASSET_MODEL = Path(resource_path("assets/weights/phrase_analyzer.npz"))
_DEV_MODEL = Path(__file__).resolve().parents[2] / "benchmark" / "phrase_analyzer.npz"


def _resolve_model(model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        return Path(model_path)
    if _DEV_MODEL.exists():
        return _DEV_MODEL
    return _ASSET_MODEL


def detect_phrase_segments(
    audio: np.ndarray,
    sample_rate: int,
    beat_times_sec: np.ndarray,
    tempo_segments: np.ndarray,
    *,
    prod_beat_probability: np.ndarray | None = None,
    model_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Detect phrase segments with the hard-boundary two-stage GBM pipeline.

    ``prod_beat_probability`` is accepted only for API compatibility with older
    call sites; production boundary detection is hard GBM only.
    """

    _ = prod_beat_probability
    return detect_two_stage_phrase_segments(
        audio,
        int(sample_rate),
        beat_times_sec,
        tempo_segments,
        model_path=_resolve_model(model_path),
    )
