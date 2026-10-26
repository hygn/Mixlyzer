"""The main view's and the overview's plugins: each module registers its view in views.base.REGISTRY when imported,
in this order, so importing the package registers them all."""
from . import waveform, cue_points, keystrip, playhead, beatgrid, JumpCUE, phrase, tempo_label, deck_cue

__all__ = ["waveform", "cue_points", "keystrip", "playhead", "beatgrid", "JumpCUE", "phrase", "tempo_label", "deck_cue"]
