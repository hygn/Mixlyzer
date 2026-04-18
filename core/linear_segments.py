from __future__ import annotations

from typing import Any

import numpy as np

from utils.labels import idx_to_labels


def key_value_to_label(key_value: int | None) -> str:
    if key_value is None:
        return ""
    try:
        if 0 <= int(key_value) <= 23:
            return idx_to_labels(int(key_value))[0]
    except Exception:
        return ""
    return ""


def harmonic_compatible_keys(anchor_key: int | None) -> set[int]:
    if anchor_key is None:
        return set()
    key = int(anchor_key) % 24
    mode = "A" if key >= 12 else "B"
    camelot, _ = idx_to_labels(key)
    number = int(camelot[:-1])
    other_mode = "B" if mode == "A" else "A"
    candidates = {
        f"{number}{mode}",
        f"{((number - 2) % 12) + 1}{mode}",
        f"{(number % 12) + 1}{mode}",
        f"{number}{other_mode}",
    }
    out: set[int] = set()
    for idx in range(24):
        label, _ = idx_to_labels(idx)
        if label in candidates:
            out.add(idx)
    return out


def build_bpm_segments(tempo_segments) -> list[dict[str, Any]]:
    arr = _normalize_tempo_segments(tempo_segments)
    rows: list[dict[str, Any]] = []
    for start, end, bpm in arr:
        if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(bpm)):
            continue
        duration = float(end - start)
        if duration <= 1e-6:
            continue
        row = {
            "seq_index": len(rows),
            "start_sec": float(start),
            "end_sec": float(end),
            "duration_sec": duration,
            "bpm": float(bpm),
            "bpm_rounded": int(round(float(bpm))),
        }
        if rows and _can_merge_bpm_rows(rows[-1], row):
            rows[-1]["end_sec"] = row["end_sec"]
            rows[-1]["duration_sec"] = float(rows[-1]["end_sec"] - rows[-1]["start_sec"])
            continue
        rows.append(row)
    for idx, row in enumerate(rows):
        row["seq_index"] = idx
    return rows


def build_key_segments(key_segments) -> list[dict[str, Any]]:
    arr = _normalize_key_segments(key_segments)
    rows: list[dict[str, Any]] = []
    for pitch, mode_flag, start, end in arr:
        if not (np.isfinite(pitch) and np.isfinite(start) and np.isfinite(end)):
            continue
        duration = float(end - start)
        if duration <= 1e-6:
            continue
        key_value = int(pitch)
        if np.isfinite(mode_flag) and int(mode_flag) == 1:
            key_value += 12
        row = {
            "seq_index": len(rows),
            "start_sec": float(start),
            "end_sec": float(end),
            "duration_sec": duration,
            "key_value": key_value,
            "key_label": key_value_to_label(key_value),
        }
        if rows and _can_merge_key_rows(rows[-1], row):
            rows[-1]["end_sec"] = row["end_sec"]
            rows[-1]["duration_sec"] = float(rows[-1]["end_sec"] - rows[-1]["start_sec"])
            continue
        rows.append(row)
    for idx, row in enumerate(rows):
        row["seq_index"] = idx
    return rows


def _normalize_tempo_segments(tempo_segments) -> np.ndarray:
    if tempo_segments is None:
        return np.empty((0, 3), dtype=float)
    arr = np.asarray(tempo_segments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=float)
    return arr[:, :3].astype(float, copy=False)


def _normalize_key_segments(key_segments) -> np.ndarray:
    if key_segments is None:
        return np.empty((0, 4), dtype=float)
    arr = np.asarray(key_segments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.empty((0, 4), dtype=float)
    return arr[:, :4].astype(float, copy=False)


def _can_merge_bpm_rows(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return abs(float(left["bpm"]) - float(right["bpm"])) < 1e-6


def _can_merge_key_rows(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["key_value"]) == int(right["key_value"])
