"""The stages the pipelines report, ``progress(stage, fraction)``: ids the app words (the analyzer has no UI
language), the fraction the pipeline's own share done."""

from typing import Callable, Optional

Progress = Optional[Callable[[str, float], None]]

LOAD = "load"                            # decoding the audio
HPSS = "hpss"                            # the harmonic / percussive split
WAVEFORM = "waveform"                    # the overview waveform
TEMPO = "tempo"                          # the beat tracker
PHRASE = "phrase"                        # phrases and their labels
JUMPCUE = "jumpcue"                      # JumpCUE detection
KEY = "key"                              # the keys
MUSIC_INDEX = "music_index"              # Search by Music's index
FRAME_FEATURES = "frame_features"        # the frame features (JumpCUE's reanalysis)
PHRASE_BOUNDARIES = "phrase_boundaries"  # the phrase boundary probability (JumpCUE's reanalysis)
BEAT_ACTIVATIONS = "beat_activations"    # the frame TCN (a tempo segment's reanalysis), then:
TEMPO_ESTIMATE = "tempo_estimate"        # its tempo,
DOWNBEATS = "downbeats"                  # its bars,
GRID_UPDATE = "grid_update"              # the grid with it


def report(progress: Progress, stage: str, fraction: float) -> None:
    if progress is not None:
        progress(stage, fraction)
