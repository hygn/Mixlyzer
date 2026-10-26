"""AutoCUE as a JumpCUE: the analysis's start cue (analyzer.phrase.cue_points.autocue_time_sec) given its own group."""


def autocue_jump_pair(time_sec: float) -> dict:
    """AutoCUE as a one-cue JumpCUE group, lettered with the other JumpCUEs."""
    cue = {"label": "", "start": float(time_sec), "end": float(time_sec), "point": float(time_sec)}
    return {"forward": dict(cue), "backward": dict(cue), "lag_beats": 0.0, "lag_sec": 0.0,
            "score": 1.0, "confidence": 1.0}
