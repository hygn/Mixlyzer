"""Phrases and their labels. The detector is imported where it is first asked for: the package also holds what the app's
GUI process reads (``cue_points``), which should not load what the detector needs (sklearn, librosa, numba)."""

__all__ = [
    "detect_phrase_segments",
]


def __getattr__(name: str):
    if name == "detect_phrase_segments":
        from analyzer.phrase.phrase import detect_phrase_segments
        return detect_phrase_segments
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
