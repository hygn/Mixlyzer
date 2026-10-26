"""The analyzer's stages and refusals in the UI's words (the analyzer reports ids: analyzer.pipeline.stages,
analyzer.errors)."""

from __future__ import annotations

from analyzer import errors
from analyzer.errors import AnalysisError
from analyzer.pipeline import stages
from foundation.i18n import tr, tr_mark

# the full analysis's stages: (the status line's word, the task's)
TRACK_STAGES = {
    stages.LOAD: (tr_mark("Loading"), tr_mark("Loading")),
    stages.HPSS: ("HPSS", tr_mark("Processing HPSS")),
    stages.WAVEFORM: (tr_mark("Waveform"), tr_mark("Processing Waveform")),
    stages.TEMPO: (tr_mark("Tempo"), tr_mark("Tempo Analyzing")),
    stages.PHRASE: (tr_mark("Phrase"), tr_mark("Phrase Boundary Analyzing")),
    stages.JUMPCUE: ("JumpCUE", tr_mark("JumpCUE Analyzing")),
    stages.KEY: (tr_mark("Key"), tr_mark("Analyzing Keys")),
    stages.MUSIC_INDEX: (tr_mark("Search by Music"), tr_mark("Indexing for Search by Music")),
}

# the editor's reanalyses' stages
REANALYSIS_STAGES = {
    stages.LOAD: tr_mark("Loading audio"),
    stages.HPSS: tr_mark("Processing HPSS"),
    stages.FRAME_FEATURES: tr_mark("Frame features"),
    stages.PHRASE_BOUNDARIES: tr_mark("Phrase boundaries"),
    stages.PHRASE: tr_mark("Analyzing phrases"),
    stages.JUMPCUE: tr_mark("Analyzing JumpCUE"),
    stages.KEY: tr_mark("Analyzing key"),
    stages.BEAT_ACTIVATIONS: tr_mark("Beat activations"),
    stages.TEMPO_ESTIMATE: tr_mark("Estimating BPM"),
    stages.DOWNBEATS: tr_mark("Downbeat analysis"),
    stages.GRID_UPDATE: tr_mark("Updating beat grid"),
}

ERRORS = {
    errors.DECODE_FAILED: tr_mark("Audio decode failed via FFmpeg and SoundFile: {path}"),
    errors.SEGMENT_TOO_SHORT: tr_mark("segment too short for reanalysis"),
    errors.SELECTION_TOO_SHORT: tr_mark("selection duration too small"),
    errors.PHRASE_TOO_FEW_BEATS: tr_mark("the phrase analysis needs at least {count} beats"),
    errors.PHRASE_NEEDS_TEMPO: tr_mark("tempo segments are required for phrase reanalysis"),
    errors.JUMPCUE_TOO_FEW_BEATS: tr_mark("too few beats for the phrase analysis JumpCUE needs"),
    errors.JUMPCUE_NEEDS_TEMPO: tr_mark("tempo segments are required for JumpCUE reanalysis"),
}


def track_stage(stage: str) -> tuple[str, str]:
    status, task = TRACK_STAGES.get(stage, (stage, stage))
    return tr(status), tr(task)


def reanalysis_stage(stage: str) -> str:
    return tr(REANALYSIS_STAGES.get(stage, stage))


def error_text(exc: BaseException) -> str:
    """What went wrong, in words (an analyzer's refusal translated, any other error as it reads)."""
    if isinstance(exc, AnalysisError) and exc.code in ERRORS:
        return tr(ERRORS[exc.code]).format(**exc.params)
    return str(exc)
