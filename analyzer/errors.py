"""The analyses' refusals: what cannot be analyzed from what was given. Each case has its code and values; the app
words it (the analyzer has no UI language)."""

from __future__ import annotations

# the codes and their values
DECODE_FAILED = "decode_failed"                    # path: neither FFmpeg nor SoundFile read the audio
SEGMENT_TOO_SHORT = "segment_too_short"            # a tempo segment under 2 s to reanalyze
SELECTION_TOO_SHORT = "selection_too_short"        # a key selection of no length
PHRASE_TOO_FEW_BEATS = "phrase_too_few_beats"      # count: the beats the phrase model needs at least
PHRASE_NEEDS_TEMPO = "phrase_needs_tempo"          # no tempo segments for the phrases
JUMPCUE_TOO_FEW_BEATS = "jumpcue_too_few_beats"    # too few beats for the phrase analysis JumpCUE reads
JUMPCUE_NEEDS_TEMPO = "jumpcue_needs_tempo"        # no tempo segments for JumpCUE


class AnalysisError(Exception):
    def __init__(self, code: str, **params):
        self.code = code
        self.params = params
        super().__init__(code + (f" {params}" if params else ""))
