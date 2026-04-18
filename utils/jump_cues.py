from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


_VALID_JUMPCUE_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _cue_sort_key(cue: dict[str, float | str]) -> tuple[float, float, float, str]:
    point = float(cue.get("point", cue.get("start", 0.0)) or 0.0)
    start = float(cue.get("start", 0.0) or 0.0)
    end = float(cue.get("end", start) or start)
    label = str(cue.get("label", "") or "")
    return (point, start, end, label)


def _normalized_cue(cue: dict | None) -> dict[str, float | str]:
    cue = cue or {}
    start = float(cue.get("start", 0.0) or 0.0)
    label = str(cue.get("label", "") or "")
    return {
        "label": label,
        "comment": str(cue.get("comment", label) or label),
        "start": start,
        "end": float(cue.get("end", start) or start),
        "point": float(cue.get("point", start) or start),
    }


def _label_from_index(idx: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = len(alphabet)
    if 0 <= idx < n:
        return alphabet[idx]
    raise ValueError("JumpCUE labels are limited to unique A-Z.")


def _validate_edit_labels(sorted_cues: list[dict[str, float | str]]) -> None:
    labels: list[str] = []
    for cue in sorted_cues:
        label = str(cue.get("label", "") or "").strip().upper()
        if label not in _VALID_JUMPCUE_LABELS:
            raise ValueError("JumpCUE labels must be unique single letters A-Z.")
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError("JumpCUE labels must be unique single letters A-Z.")


def build_jump_cues_np(
    jump_pairs: List[dict] | None,
    *,
    canonicalize_labels: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Store JumpCUE in 0.2.0 format:
    - cue_* arrays describe arbitrary cue points
    - pair_src_label / pair_dst_label describe arbitrary directed pair links
    """
    pairs = jump_pairs if isinstance(jump_pairs, list) else []
    if not pairs:
        return {
            "cue_label": np.asarray([], dtype="U1"),
            "cue_comment": np.asarray([], dtype="U1"),
            "cue_start": np.asarray([], dtype=np.float32),
            "cue_end": np.asarray([], dtype=np.float32),
            "cue_point": np.asarray([], dtype=np.float32),
            "pair_src_label": np.asarray([], dtype="U1"),
            "pair_dst_label": np.asarray([], dtype="U1"),
            "pair_lag_beats": np.asarray([], dtype=np.float32),
            "pair_lag_sec": np.asarray([], dtype=np.float32),
            "pair_score": np.asarray([], dtype=np.float32),
            "pair_confidence": np.asarray([], dtype=np.float32),
        }

    cue_map: dict[tuple[str, float, float, float], dict[str, float | str]] = {}
    raw_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        forward = _normalized_cue(pair.get("forward", {}))
        backward = _normalized_cue(pair.get("backward", {}))
        cue_map[(str(forward["label"]), float(forward["start"]), float(forward["end"]), float(forward["point"]))] = forward
        cue_map[(str(backward["label"]), float(backward["start"]), float(backward["end"]), float(backward["point"]))] = backward
        raw_pairs.append(
            {
                "forward": forward,
                "backward": backward,
                "lag_beats": float(pair.get("lag_beats", 0.0)),
                "lag_sec": float(pair.get("lag_sec", 0.0)),
                "score": float(pair.get("score", 0.0)),
                "confidence": float(pair.get("confidence", 0.0)),
            }
        )

    sorted_cues = (
        sorted(cue_map.values(), key=_cue_sort_key)
        if canonicalize_labels
        else list(cue_map.values())
    )
    if not canonicalize_labels:
        _validate_edit_labels(sorted_cues)
    cue_labels: set[str] = set()
    cues: list[dict[str, float | str]] = []
    for idx, cue in enumerate(sorted_cues):
        label = _label_from_index(idx) if canonicalize_labels else str(cue["label"]).strip().upper()
        if not label:
            label = _label_from_index(idx)
        if label in cue_labels:
            raise ValueError("JumpCUE labels must be unique single letters A-Z.")
        cue_labels.add(label)
        cues.append(
            {
                "label": label,
                "comment": str(cue.get("comment", label) or label),
                "start": float(cue["start"]),
                "end": float(cue["end"]),
                "point": float(cue["point"]),
            }
        )

    cue_label_map = {
        (str(src["label"]), float(src["start"]), float(src["end"]), float(src["point"])): str(dst["label"])
        for src, dst in zip(sorted_cues, cues)
    }
    pair_records: list[dict[str, float | int]] = []
    for pair in raw_pairs:
        forward = pair["forward"]
        backward = pair["backward"]
        src_label = cue_label_map[(str(forward["label"]), float(forward["start"]), float(forward["end"]), float(forward["point"]))]
        dst_label = cue_label_map[(str(backward["label"]), float(backward["start"]), float(backward["end"]), float(backward["point"]))]
        pair_records.append(
            {
                "src_label": src_label,
                "dst_label": dst_label,
                "lag_beats": float(pair["lag_beats"]),
                "lag_sec": float(pair["lag_sec"]),
                "score": float(pair["score"]),
                "confidence": float(pair["confidence"]),
            }
        )
    pair_records.sort(
        key=lambda p: (
            str(p["src_label"]),
            str(p["dst_label"]),
            float(p["lag_sec"]),
            float(p["score"]),
            float(p["confidence"]),
        )
    )

    labels = [str(c.get("label", "")) for c in cues]
    comments = [str(c.get("comment", c.get("label", "")) or c.get("label", "")) for c in cues]
    label_width = max(1, max((len(s) for s in labels), default=1))
    comment_width = max(1, max((len(s) for s in comments), default=1))
    return {
        "cue_label": np.asarray(labels, dtype=f"U{label_width}"),
        "cue_comment": np.asarray(comments, dtype=f"U{comment_width}"),
        "cue_start": np.asarray([float(c.get("start", 0.0)) for c in cues], dtype=np.float32),
        "cue_end": np.asarray([float(c.get("end", 0.0)) for c in cues], dtype=np.float32),
        "cue_point": np.asarray([float(c.get("point", 0.0)) for c in cues], dtype=np.float32),
        "pair_src_label": np.asarray([str(p["src_label"]) for p in pair_records], dtype=f"U{label_width}"),
        "pair_dst_label": np.asarray([str(p["dst_label"]) for p in pair_records], dtype=f"U{label_width}"),
        "pair_lag_beats": np.asarray([float(p["lag_beats"]) for p in pair_records], dtype=np.float32),
        "pair_lag_sec": np.asarray([float(p["lag_sec"]) for p in pair_records], dtype=np.float32),
        "pair_score": np.asarray([float(p["score"]) for p in pair_records], dtype=np.float32),
        "pair_confidence": np.asarray([float(p["confidence"]) for p in pair_records], dtype=np.float32),
    }


def extract_jump_cue_pairs(features: Dict[str, Any]) -> List[dict]:
    """
    Normalize Jump CUE analysis output into a list of dicts with forward/backward segments.
    Supports 0.2.0 cue/pair storage as well as legacy flattened pair storage.
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

    cue_required = [
        "cue_label",
        "cue_comment",
        "cue_start",
        "cue_end",
        "cue_point",
        "pair_src_label",
        "pair_dst_label",
        "pair_lag_beats",
        "pair_lag_sec",
        "pair_score",
        "pair_confidence",
    ]
    if all(key in block for key in cue_required):
        cue_length = min(len(block[key]) for key in ("cue_label", "cue_comment", "cue_start", "cue_end", "cue_point"))
        pair_length = min(len(block[key]) for key in (
            "pair_src_label",
            "pair_dst_label",
            "pair_lag_beats",
            "pair_lag_sec",
            "pair_score",
            "pair_confidence",
        ))
        if cue_length <= 0 or pair_length <= 0:
            return []
        cue_lookup = {
            str(block["cue_label"][idx]): {
                "label": str(block["cue_label"][idx]),
                "comment": str(block["cue_comment"][idx]),
                "start": float(block["cue_start"][idx]),
                "end": float(block["cue_end"][idx]),
                "point": float(block["cue_point"][idx]),
            }
            for idx in range(cue_length)
        }
        pairs: List[dict] = []
        for idx in range(pair_length):
            src_label = str(block["pair_src_label"][idx])
            dst_label = str(block["pair_dst_label"][idx])
            if src_label not in cue_lookup or dst_label not in cue_lookup:
                continue
            pairs.append(
                {
                    "forward": dict(cue_lookup[src_label]),
                    "backward": dict(cue_lookup[dst_label]),
                    "lag_beats": float(block["pair_lag_beats"][idx]),
                    "lag_sec": float(block["pair_lag_sec"][idx]),
                    "score": float(block["pair_score"][idx]),
                    "confidence": float(block["pair_confidence"][idx]),
                }
            )
        return pairs

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
                    "comment": str(block["forward_label"][idx]),
                    "start": float(block["forward_start"][idx]),
                    "end": float(block["forward_end"][idx]),
                    "point": float(block["forward_point"][idx]),
                },
                "backward": {
                    "label": str(block["backward_label"][idx]),
                    "comment": str(block["backward_label"][idx]),
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


__all__ = ["build_jump_cues_np", "extract_jump_cue_pairs"]
