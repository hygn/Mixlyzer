"""What an analysis is asked for: the settings the app hands the analyzer (where they are kept and edited, and the
defaults, are the app's). Every model file is given as the file to read: the analyzer looks for none itself."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

# every analysis decodes at this rate: the learned models were trained on frames of it (the beat tracker's, the key
# model's, the phrase models')
ANALYSIS_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class ModelPaths:
    """The learned models' files, as read."""
    beat_tcn: str            # the beat tracker's frame TCN (its trainings in one file)
    beat_critic: str         # its metronome critic
    beat_decoder: str        # its decoder's costs (JSON)
    downbeat_ctx: str        # the contextual downbeat GBM
    refine_onset: str        # the tempo-change onset GBM
    phrase: str              # the phrase boundary and label models
    jumpcue: str             # the JumpCUE jump point model (JSON)
    key: str                 # the chord-aware key model


@dataclass(frozen=True)
class CueRule:
    """Where a phrase of a label gets a CUE point."""
    at_start: bool               # where it starts
    at_end: bool                 # where it ends
    same_label_boundaries: bool  # also between two phrases of the label (else only next to another label)


@dataclass(frozen=True)
class AnalysisSettings:
    models: ModelPaths
    bpm_min: float                       # every tempo is read in the octave from here (bpm_min .. 2 x bpm_min)
    # the detail waveform's low / mid / high ranges (Hz): the overview stores their scale
    wave_bands: tuple[tuple[float, float], ...]
    bpm_dynamic: bool = True             # tempo changes within a track (off: one steady grid)
    dynamic_downbeat: bool = True        # odd bars and meter changes of a steady grid (always with bpm_dynamic)
    key_dynamic: bool = True             # key changes within a track (off: one key)
    # AutoCUE: what goes where the LEAD_IN ends (an early first downbeat, or 0:00): a CUE point, a JumpCUE, nothing
    autocue: Literal["off", "cue_point", "jumpcue"] = "off"
    cue_rules: Mapping[str, CueRule] = field(default_factory=dict)    # by phrase label (uppercase)
