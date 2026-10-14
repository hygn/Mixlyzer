from __future__ import annotations

import math
from typing import Any

import numpy as np


CUE_POINT_PREFIX = "cue_points_np."
CUE_POINT_KEYS = (
    "cue_id",
    "cue_time_sec",
    "cue_label",
    "cue_comment",
    "cue_color_idx",  # colour preset (utils.jumpcue_colors.CUE_POINT_COLORS)
)
_DEDUP_TIME_EPSILON_SEC = 1e-3


def empty_cue_points_np() -> dict[str, np.ndarray]:
    return {
        "cue_id": np.asarray([], dtype=np.int32),
        "cue_time_sec": np.asarray([], dtype=np.float32),
        "cue_label": np.asarray([], dtype="U1"),
        "cue_comment": np.asarray([], dtype="U1"),
        "cue_color_idx": np.asarray([], dtype=np.int32),
    }


def _append_unique(parts: list[str], value: object) -> None:
    text = str(value or "").strip()
    if text and text not in parts:
        parts.append(text)


def _dedupe_cue_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        return []
    rows = sorted(points, key=lambda point: (float(point["time_sec"]), int(point["id"])))
    merged: list[dict[str, Any]] = []
    for point in rows:
        time_sec = float(point["time_sec"])
        if (
            merged
            and abs(float(merged[-1]["time_sec"]) - time_sec) <= _DEDUP_TIME_EPSILON_SEC
        ):
            _append_unique(merged[-1]["_labels"], point.get("label"))
            _append_unique(merged[-1]["_comments"], point.get("comment"))
            continue
        labels: list[str] = []
        comments: list[str] = []
        _append_unique(labels, point.get("label"))
        _append_unique(comments, point.get("comment"))
        merged.append(
            {
                "id": len(merged),
                "time_sec": time_sec,
                "_labels": labels,
                "_comments": comments,
                "color_idx": int(point.get("color_idx", 0) or 0),
            }
        )
    return [
        {
            "id": index,
            "time_sec": float(point["time_sec"]),
            "label": "/".join(point["_labels"]),
            "comment": " / ".join(point["_comments"]),
            "color_idx": int(point["color_idx"]),
        }
        for index, point in enumerate(merged)
    ]


def build_cue_points_np(points: list[dict[str, Any]] | None) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points or []):
        try:
            time_sec = float(point.get("time_sec"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(time_sec) or time_sec < 0.0:
            continue
        rows.append(
            {
                "id": int(point.get("id", index)),
                "time_sec": time_sec,
                "label": str(point.get("label", "") or ""),
                "comment": str(point.get("comment", "") or ""),
                "color_idx": int(point.get("color_idx", 0) or 0),
            }
        )
    rows = _dedupe_cue_points(rows)
    if not rows:
        return empty_cue_points_np()
    rows.sort(key=lambda point: (float(point["time_sec"]), int(point["id"])))
    label_width = max(1, max(len(str(point["label"])) for point in rows))
    comment_width = max(1, max(len(str(point["comment"])) for point in rows))
    return {
        "cue_id": np.asarray([point["id"] for point in rows], dtype=np.int32),
        "cue_time_sec": np.asarray(
            [point["time_sec"] for point in rows],
            dtype=np.float32,
        ),
        "cue_label": np.asarray(
            [point["label"] for point in rows],
            dtype=f"U{label_width}",
        ),
        "cue_comment": np.asarray(
            [point["comment"] for point in rows],
            dtype=f"U{comment_width}",
        ),
        "cue_color_idx": np.asarray([point["color_idx"] for point in rows], dtype=np.int32),
    }


def build_phrase_cue_points(phrases: list[dict[str, Any]] | None, rules: dict[str, Any]) -> list[dict[str, Any]]:
    """CUE points at phrase starts / ends by the rule of each phrase's label
    (``rules``: uppercase label -> object with ``at_start``, ``at_end`` and
    ``same_label_boundaries``, see core.config.cuepointconfig). A start or end
    next to a phrase of the same label gets one only with ``same_label_boundaries``.
    Named after the label (the end: ``<LABEL>_OUT``)."""
    rows: list[dict[str, Any]] = []
    for phrase in phrases or []:
        try:
            start = float(phrase.get("start"))
            end = float(phrase.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        rows.append({"start": start, "end": end, "label": str(phrase.get("label", "") or "").strip().upper()})
    rows.sort(key=lambda phrase: phrase["start"])

    points: list[dict[str, Any]] = []
    for i, phrase in enumerate(rows):
        label = phrase["label"]
        rule = rules.get(label)
        if rule is None:
            continue
        prev_same = i > 0 and rows[i - 1]["label"] == label
        next_same = i + 1 < len(rows) and rows[i + 1]["label"] == label
        if rule.at_start and (rule.same_label_boundaries or not prev_same):
            points.append({"id": len(points), "time_sec": phrase["start"], "label": label,
                           "comment": f"{label.title()} start"})
        if rule.at_end and (rule.same_label_boundaries or not next_same):
            points.append({"id": len(points), "time_sec": phrase["end"], "label": f"{label}_OUT",
                           "comment": f"{label.title()} end"})
    return _dedupe_cue_points(points)


def extract_cue_points(features: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not features:
        return []
    block = features.get("cue_points_np")
    if isinstance(block, dict) and all(key in block for key in CUE_POINT_KEYS):
        arrays = {
            key: np.asarray(block[key])
            for key in CUE_POINT_KEYS
        }
    else:
        try:
            arrays = {
                key: np.asarray(features[f"{CUE_POINT_PREFIX}{key}"])
                for key in CUE_POINT_KEYS
            }
        except KeyError:
            return []
    count = min(array.size for array in arrays.values())
    points: list[dict[str, Any]] = []
    for index in range(count):
        time_sec = float(arrays["cue_time_sec"][index])
        if not math.isfinite(time_sec) or time_sec < 0.0:
            continue
        points.append(
            {
                "id": int(arrays["cue_id"][index]),
                "time_sec": time_sec,
                "label": str(arrays["cue_label"][index]),
                "comment": str(arrays["cue_comment"][index]),
                "color_idx": int(arrays["cue_color_idx"][index]),
            }
        )
    points.sort(key=lambda point: (float(point["time_sec"]), int(point["id"])))
    return points


__all__ = [
    "CUE_POINT_KEYS",
    "CUE_POINT_PREFIX",
    "build_phrase_cue_points",
    "build_cue_points_np",
    "empty_cue_points_np",
    "extract_cue_points",
]
