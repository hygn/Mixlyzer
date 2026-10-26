from __future__ import annotations

import math
from typing import Any

import numpy as np

from analyzer.phrase.cue_points import merge_coincident


CUE_POINT_PREFIX = "cue_points_np."
CUE_POINT_KEYS = (
    "cue_id",
    "cue_time_sec",
    "cue_label",
    "cue_comment",
    "cue_color_idx",  # colour preset (utils.jumpcue_colors.CUE_POINT_COLORS)
)


def empty_cue_points_np() -> dict[str, np.ndarray]:
    return {
        "cue_id": np.asarray([], dtype=np.int32),
        "cue_time_sec": np.asarray([], dtype=np.float32),
        "cue_label": np.asarray([], dtype="U1"),
        "cue_comment": np.asarray([], dtype="U1"),
        "cue_color_idx": np.asarray([], dtype=np.int32),
    }


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
    rows = merge_coincident(rows)
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
    "build_cue_points_np",
    "empty_cue_points_np",
    "extract_cue_points",
]
