"""The editor's JumpCUE reanalysis: JumpCUEs on the current beat grid and the editor's phrases, computed as in the full
analysis (the parts it needs, HPSS, the frame features and the keys' frame chroma, analyzed first: ``jumpcue_inputs``,
which the training reads too)."""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from analyzer.audio.hpss import hpss_audio_with_spectra
from analyzer.audio.load import fast_load
from analyzer.beat import melody_contour
from analyzer.beat.frame_features import FRAME_HOP_LENGTH, extract_frame_features
from analyzer.errors import JUMPCUE_NEEDS_TEMPO, AnalysisError
from analyzer.jumpcue.JumpCUE import JumpContext, audio_context, jump_cue_pairs
from analyzer.jumpcue.autocue import autocue_jump_pair
from analyzer.key.chord_key import frame_chroma as key_frame_chroma
from analyzer.phrase.cue_points import autocue_time_sec, phrase_cue_points
from analyzer.pipeline import stages
from analyzer.pipeline.stages import Progress, report
from analyzer.settings import ANALYSIS_SAMPLE_RATE, AnalysisSettings


@dataclass(frozen=True)
class JumpCueInputs:
    """What the JumpCUE pairs are made from beside the beats and the phrases, as the full analysis makes it."""
    context: JumpContext
    frames: Any                        # the frame features (analyzer.beat.frame_features)
    key_frames: np.ndarray             # the keys' frame chroma


def jumpcue_inputs(path: str, beats_time: np.ndarray, progress: Progress = None) -> JumpCueInputs:
    """The audio decoded and analyzed up to the JumpCUEs on this beat grid."""
    report(progress, stages.LOAD, 0.0)
    sr = ANALYSIS_SAMPLE_RATE
    audio_stereo = fast_load(path, sr, stereo=True)
    audio_full = np.ascontiguousarray(audio_stereo.mean(axis=1), dtype=np.float32)
    report(progress, stages.HPSS, 0.1)
    y_harm, y_perc, hpss_spectra = hpss_audio_with_spectra(audio_full, sr)
    report(progress, stages.FRAME_FEATURES, 0.25)
    frames = extract_frame_features(audio_full, sr, y_perc, y_harm, hop_length=FRAME_HOP_LENGTH)
    # the melody contours the frame features keep for the full analysis's index (none here): let go, not held by
    # the resident reanalysis process with the track's harmonic part until its next job
    melody_contour.forget_contours()
    return JumpCueInputs(audio_context(y_harm, audio_full, sr, beats_time, spectra=hpss_spectra), frames,
                         key_frame_chroma(y_harm, sr))


def reanalyze_jumpcues(path: str, settings: AnalysisSettings, beats_time_arr: np.ndarray, *,
                       tempo_segments: Optional[np.ndarray] = None,
                       phrases: Optional[list] = None,
                       progress: Progress = None) -> list[dict]:
    """The JumpCUE pairs (AutoCUE's last, with AutoCUE on JumpCUEs), on ``phrases``, the editor's current ones (none: no
    learned pair), which also give the CUE points the pair may not wholly sit on. With AutoCUE on JumpCUEs, its position
    follows the current lead-in and beat grid, so it follows phrase and beat-grid edits and reanalyses."""
    if tempo_segments is None or np.asarray(tempo_segments).size == 0:
        raise AnalysisError(JUMPCUE_NEEDS_TEMPO)
    tempo_segments = np.asarray(tempo_segments, dtype=float)
    inputs = jumpcue_inputs(path, beats_time_arr, progress)
    report(progress, stages.JUMPCUE, 0.55)
    autocue_target = (
        autocue_time_sec(phrases, beats_time_arr, tempo_segments)
        if settings.autocue == "jumpcue"
        else None
    )
    phrases = list(phrases or [])
    cue_points = phrase_cue_points(phrases, beats_time_arr, tempo_segments, settings.cue_rules, settings.autocue)
    jump_pairs = jump_cue_pairs(
        inputs.context, model_path=settings.models.jumpcue, beats_time=beats_time_arr, tempo_segments=tempo_segments,
        phrases=[(s.start, s.end, s.label) for s in phrases], frames=inputs.frames, key_frames=inputs.key_frames,
        rules=settings.jumpcue_rules, cue_times=[float(c.time_sec) for c in cue_points],
    ) if phrases else []
    if jump_pairs:
        print("[JumpCUE] pairs", jump_pairs)
    else:
        print("[JumpCUE] no jump-compatible pairs detected")
    if autocue_target is not None:
        jump_pairs = [*jump_pairs, autocue_jump_pair(autocue_target)]
    return jump_pairs
