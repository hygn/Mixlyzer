"""The editor's phrase reanalysis: phrases on the current beat grid and tempo segments, detected as in the full
analysis (the same decode, HPSS and model)."""
from typing import Optional

import numpy as np

from analyzer.audio.hpss import hpss_audio_with_spectra
from analyzer.audio.load import fast_load
from analyzer.errors import PHRASE_NEEDS_TEMPO, PHRASE_TOO_FEW_BEATS, AnalysisError
from analyzer.music.marks import Phrase
from analyzer.phrase import detect_phrase_segments
from analyzer.pipeline import stages
from analyzer.pipeline.stages import Progress, report
from analyzer.settings import ANALYSIS_SAMPLE_RATE, AnalysisSettings

# the full analysis's floor: fewer beats are too short for the phrase model
MIN_BEATS = 17


def reanalyze_phrases(path: str, settings: AnalysisSettings, beats_time_arr: np.ndarray, *,
                      tempo_segments: Optional[np.ndarray] = None,
                      progress: Progress = None) -> list[Phrase]:
    """The phrases (analyzer.music.marks.Phrase)."""
    beats_time_arr = np.asarray(beats_time_arr, dtype=float)
    if beats_time_arr.size < MIN_BEATS:
        raise AnalysisError(PHRASE_TOO_FEW_BEATS, count=MIN_BEATS)
    if tempo_segments is None or np.asarray(tempo_segments).size == 0:
        raise AnalysisError(PHRASE_NEEDS_TEMPO)
    tempo_segments = np.asarray(tempo_segments, dtype=float)
    report(progress, stages.LOAD, 0.0)
    sr = ANALYSIS_SAMPLE_RATE
    audio_stereo = fast_load(path, sr, stereo=True)
    report(progress, stages.HPSS, 0.2)
    audio_full = np.ascontiguousarray(audio_stereo.mean(axis=1), dtype=np.float32)
    _y_harm, _y_perc, hpss_spectra = hpss_audio_with_spectra(audio_full, sr)
    report(progress, stages.PHRASE, 0.5)
    segments = detect_phrase_segments(
        audio_stereo,
        sr,
        beats_time_arr,
        tempo_segments,
        model_path=settings.models.phrase,
        hpss=hpss_spectra,
    )
    rows = list(segments)
    print(f"[Phrase] reanalysis segments={len(rows)} labels=[{', '.join(r.label for r in rows)}]")
    return rows
