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
)
_DEDUP_TIME_EPSILON_SEC = 1e-3


def empty_cue_points_np() -> dict[str, np.ndarray]:
    return {
        "cue_id": np.asarray([], dtype=np.int32),
        "cue_time_sec": np.asarray([], dtype=np.float32),
        "cue_label": np.asarray([], dtype="U1"),
        "cue_comment": np.asarray([], dtype="U1"),
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
            }
        )
    return [
        {
            "id": index,
            "time_sec": float(point["time_sec"]),
            "label": "/".join(point["_labels"]),
            "comment": " / ".join(point["_comments"]),
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
    }


def build_phrase_cue_points(phrases: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phrase in phrases or []:
        try:
            start = float(phrase.get("start"))
            end = float(phrase.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        rows.append(
            {
                "start": start,
                "end": end,
                "label": str(phrase.get("label", "") or "").strip(),
            }
        )
    rows.sort(key=lambda phrase: phrase["start"])

    points: list[dict[str, Any]] = []

    def add(time_sec: float, label: str, comment: str) -> None:
        points.append(
            {
                "id": len(points),
                "time_sec": float(time_sec),
                "label": label,
                "comment": comment,
            }
        )

    def normalized_label(index: int) -> str:
        return str(rows[index].get("label", "") or "").strip().upper()

    index = 0
    while index < len(rows):
        label = normalized_label(index)
        phrase = rows[index]
        start = float(phrase["start"])
        if label == "INTERLUDE":
            add(start, "INTERLUDE", "Interlude start")
        elif label == "OUTRO":
            add(start, "OUTRO", "Outro start")

        if label != "CHORUS":
            index += 1
            continue

        group_start = index
        group_end = index + 1
        while group_end < len(rows) and normalized_label(group_end) == "CHORUS":
            group_end += 1

        chorus_group = rows[group_start:group_end]
        add(float(chorus_group[0]["start"]), "CHORUS_IN", "Chorus start")
        if len(chorus_group) >= 2:
            add(float(chorus_group[1]["start"]), "CHORUS_NEXT", "Chorus internal boundary")
            add(float(chorus_group[-1]["start"]), "CHORUS_PRE_OUT", "Chorus pre-exit boundary")
        if group_end < len(rows):
            add(float(chorus_group[-1]["end"]), "CHORUS_OUT", "Chorus exit")
        index = group_end

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
