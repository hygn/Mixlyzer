from typing import Callable, Optional

import numpy as np

from analyzer_core.cue_and_phrase import detect_phrase_segments
from analyzer_core.global_analyzer import fast_load
from analyzer_core.hpss import hpss_audio_with_spectra
from core.config import ANALYSIS_SAMPLE_RATE, config

# the full analysis's floor: fewer beats are too short for the phrase model
MIN_BEATS = 17


def reanalyze_phrases(path: str, cfg: config, beats_time_arr: np.ndarray, *,
    tempo_segments: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,) -> list[dict]:
    """Phrases on the current beat grid and tempo segments, detected as in the full
    analysis (same decode, HPSS and model). Returns [{start, end, label}, ...]."""
    gcf = cfg.analysisconfig
    beats_time_arr = np.asarray(beats_time_arr, dtype=float)
    if beats_time_arr.size < MIN_BEATS:
        raise ValueError(f"the phrase analysis needs at least {MIN_BEATS} beats")
    if tempo_segments is None or np.asarray(tempo_segments).size == 0:
        raise ValueError("tempo segments are required for phrase reanalysis")
    tempo_segments = np.asarray(tempo_segments, dtype=float)
    if progress_cb:
        progress_cb("Loading audio", 0.0)
    sr = ANALYSIS_SAMPLE_RATE
    audio_stereo = fast_load(path, sr, stereo=True)
    if progress_cb:
        progress_cb("Processing HPSS", 0.2)
    audio_full = np.ascontiguousarray(audio_stereo.mean(axis=1), dtype=np.float32)
    _y_harm, _y_perc, hpss_spectra = hpss_audio_with_spectra(audio_full, sr)
    if progress_cb:
        progress_cb("Analyzing phrases", 0.5)
    segments = detect_phrase_segments(
        audio_stereo,
        sr,
        beats_time_arr,
        tempo_segments,
        model_path=str(gcf.phrase_parameter_path).strip(),
        hpss=hpss_spectra,
    )
    rows = [
        {"start": float(s["start"]), "end": float(s["end"]), "label": str(s["label"])}
        for s in segments
    ]
    print(f"[Phrase] reanalysis segments={len(rows)} labels=[{', '.join(r['label'] for r in rows)}]")
    if progress_cb:
        progress_cb("Phrases Updated", 0.9)
    return rows
