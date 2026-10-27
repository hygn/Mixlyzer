"""The mix candidates' beats' features of a track analyzed before they existed (Index Now): its audio decoded as the
full analysis decodes it and read on its stored beats (the full analysis pools its own frame features instead)."""

import numpy as np

from analyzer.audio.load import fast_load
from analyzer.search.mixability import BeatFeatures, beat_features_of_audio
from analyzer.settings import ANALYSIS_SAMPLE_RATE


def mix_features_of_track(path: str, beats) -> BeatFeatures:
    stereo = fast_load(path, target_sr=ANALYSIS_SAMPLE_RATE, stereo=True)
    mono = np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32)
    return beat_features_of_audio(mono, ANALYSIS_SAMPLE_RATE, np.asarray(beats, dtype=np.float64))
