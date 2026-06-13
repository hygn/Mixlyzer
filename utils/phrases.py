from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


# Predefined phrase labels. Custom labels (free text) are also allowed.
PHRASE_LABELS: list[str] = [
    "INTRO",
    "VERSE",
    "CHORUS",
    "BRIDGE",
    "OUTRO",
    "INTERLUDE",
    "SILENCE",
    "FILL",
]

# Stable colors for the predefined labels; custom labels fall back to a hash.
_PHRASE_COLORS: dict[str, tuple[int, int, int]] = {
    "INTRO": (70, 130, 180),
    "VERSE": (46, 139, 87),
    "CHORUS": (220, 60, 90),
    "BRIDGE": (148, 0, 211),
    "OUTRO": (95, 95, 110),
    "INTERLUDE": (218, 165, 32),
    "SILENCE": (110, 110, 110),
    "FILL": (0, 153, 204),
}

_FALLBACK_PALETTE: list[tuple[int, int, int]] = [
    (200, 80, 40),
    (40, 160, 160),
    (160, 100, 200),
    (120, 160, 40),
    (200, 120, 160),
    (80, 120, 200),
]


def normalize_base_label(label: object) -> str:
    """Normalize a user-entered label to its base form (no trailing digits)."""
    text = str(label or "").strip()
    return text


def phrase_color(base_label: str) -> tuple[int, int, int]:
    key = str(base_label or "").strip().upper()
    if key in _PHRASE_COLORS:
        return _PHRASE_COLORS[key]
    if not key:
        return (110, 110, 110)
    return _FALLBACK_PALETTE[hash(key) % len(_FALLBACK_PALETTE)]


def _sorted_phrases(phrases) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in phrases or []:
        try:
            start = float(p.get("start"))
            end = float(p.get("end"))
        except Exception:
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        label = normalize_base_label(p.get("label"))
        rows.append({"start": start, "end": end, "label": label})
    rows.sort(key=lambda r: r["start"])
    return rows


def number_phrase_labels(phrases) -> list[str]:
    """Return display labels: when a base label occurs >=2x, number all from 1.

    Order is by start time. Single-occurrence labels are returned unchanged.
    """
    rows = _sorted_phrases(phrases)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    running: dict[str, int] = {}
    out: list[str] = []
    for r in rows:
        base = r["label"]
        if counts.get(base, 0) >= 2:
            running[base] = running.get(base, 0) + 1
            out.append(f"{base}{running[base]}")
        else:
            out.append(base)
    return out


def abbreviate_phrase_labels(phrases) -> list[str]:
    """Overview labels: predefined phrases shown as first-letter + number (e.g.
    CHORUS2 -> "C2", VERSE -> "V"); custom labels keep their full numbered form.
    """
    rows = _sorted_phrases(phrases)
    numbered = number_phrase_labels(rows)
    predefined = {label.upper() for label in PHRASE_LABELS}
    out: list[str] = []
    for row, disp in zip(rows, numbered):
        base = row["label"]
        if base and base.upper() in predefined:
            suffix = disp[len(base):]  # trailing number, or "" for a lone occurrence
            out.append(base[0].upper() + suffix)
        else:
            out.append(disp)
    return out


def assign_phrase_to_selection(phrases, start: float, end: float, base_label: str) -> list[dict[str, Any]]:
    """Overwrite [start, end) with base_label, then merge adjacent same-label spans."""
    start = float(start)
    end = float(end)
    if end <= start:
        raise ValueError("selection end must be greater than start")
    base = normalize_base_label(base_label)
    if not base:
        raise ValueError("phrase label must not be empty")

    new_seg = {"start": start, "end": end, "label": base}
    result: list[dict[str, Any]] = []
    inserted = False
    for seg in _sorted_phrases(phrases):
        s, e = seg["start"], seg["end"]
        if e <= start:
            result.append(dict(seg))
            continue
        if s >= end:
            if not inserted:
                result.append(dict(new_seg))
                inserted = True
            result.append(dict(seg))
            continue
        # overlaps the new span
        if s < start:
            left = dict(seg)
            left["end"] = start
            if left["end"] - left["start"] > 1e-9:
                result.append(left)
        if not inserted:
            result.append(dict(new_seg))
            inserted = True
        if e > end:
            right = dict(seg)
            right["start"] = end
            if right["end"] - right["start"] > 1e-9:
                result.append(right)
    if not inserted:
        result.append(dict(new_seg))

    # Adjacent segments that share a label are kept separate (not merged), so
    # repeated sections stay individually numbered (e.g. VERSE1, VERSE2).
    result.sort(key=lambda r: r["start"])
    return result


def clear_phrase_in_selection(phrases, start: float, end: float) -> list[dict[str, Any]]:
    """Remove any phrase coverage within [start, end), splitting overlaps."""
    start = float(start)
    end = float(end)
    if end <= start:
        return _sorted_phrases(phrases)
    result: list[dict[str, Any]] = []
    for seg in _sorted_phrases(phrases):
        s, e = seg["start"], seg["end"]
        if e <= start or s >= end:
            result.append(dict(seg))
            continue
        if s < start:
            left = dict(seg)
            left["end"] = start
            if left["end"] - left["start"] > 1e-9:
                result.append(left)
        if e > end:
            right = dict(seg)
            right["start"] = end
            if right["end"] - right["start"] > 1e-9:
                result.append(right)
    return result


def build_phrase_segments_np(phrases) -> Dict[str, np.ndarray]:
    """Serialize phrases (base labels) into an NPZ-friendly block."""
    rows = _sorted_phrases(phrases)
    if not rows:
        return {
            "start": np.asarray([], dtype=np.float32),
            "end": np.asarray([], dtype=np.float32),
            "label": np.asarray([], dtype="U1"),
        }
    label_width = max(1, max(len(r["label"]) for r in rows))
    return {
        "start": np.asarray([r["start"] for r in rows], dtype=np.float32),
        "end": np.asarray([r["end"] for r in rows], dtype=np.float32),
        "label": np.asarray([r["label"] for r in rows], dtype=f"U{label_width}"),
    }


def _extract_phrase_block(features: Dict[str, Any]) -> dict[str, Any]:
    if not isinstance(features, dict):
        return {}
    block = features.get("phrase_segments_np")
    if isinstance(block, dict) and block:
        return block
    prefix = "phrase_segments_np."
    return {
        key[len(prefix):]: value
        for key, value in features.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def extract_phrase_segments(features: Dict[str, Any]) -> List[dict[str, Any]]:
    """Read phrases (base labels) from a features dict; returns sorted list."""
    block = _extract_phrase_block(features)
    if not block:
        return []
    try:
        starts = np.asarray(block["start"], dtype=float).ravel()
        ends = np.asarray(block["end"], dtype=float).ravel()
        labels = np.asarray(block["label"]).ravel()
    except Exception:
        return []
    n = min(len(starts), len(ends), len(labels))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append({
            "start": float(starts[i]),
            "end": float(ends[i]),
            "label": normalize_base_label(str(labels[i])),
        })
    return _sorted_phrases(rows)


def build_phrase_strip_buffer(phrases, duration_sec: float, *, width: int = 4096) -> np.ndarray | None:
    """Return a (width, 1, 3) uint8 color band for the phrase overlay strip."""
    duration = float(duration_sec or 0.0)
    if duration <= 0:
        return None
    rows = _sorted_phrases(phrases)
    if not rows:
        return None
    strip = np.zeros((1, width, 3), dtype=np.uint8)
    for seg in rows:
        start = seg["start"]
        end = seg["end"]
        start_px = int(np.clip(np.floor(start / duration * width), 0, width - 1))
        end_px = int(np.clip(math.ceil(end / duration * width), 0, width))
        if end_px <= start_px:
            end_px = min(width, start_px + 1)
        strip[:, start_px:end_px, :] = phrase_color(seg["label"])
    return strip.swapaxes(0, 1)
