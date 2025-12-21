from __future__ import annotations

from typing import Any, Dict, List


def extract_jump_cue_pairs(features: Dict[str, Any]) -> List[dict]:
    """
    Normalize Jump CUE analysis output into a list of dicts with forward/backward segments.
    Supports both flattened npz storage (jump_cues_np.*) and legacy list formats.
    """
    if not isinstance(features, dict):
        return []
    block = features.get("jump_cues_np")
    if not isinstance(block, dict) or not block:
        prefix = "jump_cues_np."
        block = {
            key[len(prefix) :]: value
            for key, value in features.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
    if not block:
        return []

    required = [
        "forward_label",
        "forward_start",
        "forward_end",
        "forward_point",
        "backward_label",
        "backward_start",
        "backward_end",
        "backward_point",
        "lag_beats",
        "lag_sec",
        "score",
        "confidence",
    ]
    if any(key not in block for key in required):
        return []

    length = min(len(block[key]) for key in required)
    if length <= 0:
        return []

    pairs: List[dict] = []
    for idx in range(length):
        pairs.append(
            {
                "forward": {
                    "label": str(block["forward_label"][idx]),
                    "start": float(block["forward_start"][idx]),
                    "end": float(block["forward_end"][idx]),
                    "point": float(block["forward_point"][idx]),
                },
                "backward": {
                    "label": str(block["backward_label"][idx]),
                    "start": float(block["backward_start"][idx]),
                    "end": float(block["backward_end"][idx]),
                    "point": float(block["backward_point"][idx]),
                },
                "lag_beats": float(block["lag_beats"][idx]),
                "lag_sec": float(block["lag_sec"][idx]),
                "score": float(block["score"][idx]),
                "confidence": float(block["confidence"][idx]),
            }
        )
    return pairs


__all__ = ["extract_jump_cue_pairs"]

