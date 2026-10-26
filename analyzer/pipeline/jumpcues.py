"""The editor's JumpCUE reanalysis: JumpCUEs on the current beat grid, computed as in the full analysis (the parts it
needs, HPSS, the frame features and the phrase boundary probability, analyzed first)."""
from dataclasses import replace
from typing import Optional

import numpy as np

from analyzer.audio.hpss import hpss_audio_with_spectra
from analyzer.audio.load import fast_load
from analyzer.beat.frame_features import FRAME_HOP_LENGTH, extract_frame_features
from analyzer.errors import JUMPCUE_NEEDS_TEMPO, JUMPCUE_TOO_FEW_BEATS, AnalysisError
from analyzer.jumpcue.JumpCUE import JumpCueConfig, JumpCueEngine
from analyzer.jumpcue.autocue import autocue_jump_pair
from analyzer.jumpcue.jump_point_model import jump_point_beat_features
from analyzer.phrase import detect_phrase_segments
from analyzer.phrase.cue_points import autocue_time_sec
from analyzer.pipeline import stages
from analyzer.pipeline.stages import Progress, report
from analyzer.settings import ANALYSIS_SAMPLE_RATE, AnalysisSettings


def reanalyze_jumpcues(path: str, settings: AnalysisSettings, beats_time_arr: np.ndarray, *,
                       tempo_segments: Optional[np.ndarray] = None,
                       phrases: Optional[list] = None,
                       progress: Progress = None) -> list[dict]:
    """The JumpCUE pairs (AutoCUE's last, with AutoCUE on JumpCUEs).

    ``phrases``: the editor's current phrases. With AutoCUE on JumpCUEs, its position follows the current lead-in
    and beat grid (one hot cue slot is kept for it), so it follows phrase and beat-grid edits and reanalyses."""
    if tempo_segments is None or np.asarray(tempo_segments).size == 0:
        raise AnalysisError(JUMPCUE_NEEDS_TEMPO)
    tempo_segments = np.asarray(tempo_segments, dtype=float)
    report(progress, stages.LOAD, 0.0)
    sr = ANALYSIS_SAMPLE_RATE
    audio_stereo = fast_load(path, sr, stereo=True)
    audio_full = np.ascontiguousarray(audio_stereo.mean(axis=1), dtype=np.float32)
    report(progress, stages.HPSS, 0.1)
    y_harm, y_perc, hpss_spectra = hpss_audio_with_spectra(audio_full, sr)
    report(progress, stages.FRAME_FEATURES, 0.25)
    frames = extract_frame_features(audio_full, sr, y_perc, y_harm, hop_length=FRAME_HOP_LENGTH)
    report(progress, stages.PHRASE_BOUNDARIES, 0.4)
    _segments, boundary_prob, phrase_feature_z = detect_phrase_segments(
        audio_stereo,
        sr,
        beats_time_arr,
        tempo_segments,
        model_path=settings.models.phrase,
        hpss=hpss_spectra,
        return_boundary_probability=True,
        return_feature_z=True,
    )
    if boundary_prob is None:
        raise AnalysisError(JUMPCUE_TOO_FEW_BEATS)
    report(progress, stages.JUMPCUE, 0.55)
    autocue_target = (
        autocue_time_sec(phrases, beats_time_arr, tempo_segments)
        if settings.autocue == "jumpcue"
        else None
    )
    jump_config = JumpCueConfig(jump_point_model_path=settings.models.jumpcue)
    if autocue_target is not None:
        # one hot cue slot stays free for the AutoCUE JumpCUE (below)
        jump_config = replace(jump_config, hot_cue_slots=jump_config.hot_cue_slots - 1)
    jump_engine = JumpCueEngine(jump_config)
    jump_result = jump_engine.run(
        y_harm=y_harm,
        sr=sr,
        beats_time=beats_time_arr,
        y_mix=audio_full,
        tempo_segments=tempo_segments,
        mix_magnitude=(
            hpss_spectra.magnitude
            if hpss_spectra is not None
            and (hpss_spectra.n_fft, hpss_spectra.hop_length)
            == (jump_engine.config.vocal.n_fft, jump_engine.config.vocal.hop)
            else None
        ),
        jump_point_features=jump_point_beat_features(
            frames, beats_time_arr, tempo_segments, boundary_prob, audio=audio_stereo,
            phrase_feature_z=phrase_feature_z,
        ),
    )
    jump_pairs = [pair.as_dict() for pair in jump_result.pairs]
    if jump_pairs:
        print("[JumpCUE] pairs", jump_pairs)
    else:
        print("[JumpCUE] no jump-compatible pairs detected")
    if autocue_target is not None:
        jump_pairs = [*jump_pairs, autocue_jump_pair(autocue_target)]
    return jump_pairs
