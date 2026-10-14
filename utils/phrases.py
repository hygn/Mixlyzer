from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


# Predefined phrase labels (always uppercase). Custom labels (free text) are
# also allowed and are kept exactly as the user types them.
PHRASE_LABELS: list[str] = [
    "INTRO",
    "VERSE",
    "CHORUS",
    "BREAK_CHORUS",
    "BRIDGE",
    "OUTRO",
    "INTERLUDE",
    "SILENCE",
    "FILL_IN",
    "FILL_OUT",
]

# Stable colors for the predefined labels; custom labels fall back to a hash.
_PHRASE_COLORS: dict[str, tuple[int, int, int]] = {
    "INTRO": (70, 130, 180),
    "VERSE": (46, 139, 87),
    "CHORUS": (220, 60, 90),
    "BREAK_CHORUS": (235, 140, 120),
    "BRIDGE": (148, 0, 211),
    "OUTRO": (95, 95, 110),
    "INTERLUDE": (218, 165, 32),
    "SILENCE": (110, 110, 110),
    "FILL_IN": (0, 180, 200),
    "FILL_OUT": (0, 120, 150),
}

_FALLBACK_PALETTE: list[tuple[int, int, int]] = [
    (200, 80, 40),
    (40, 160, 160),
    (160, 100, 200),
    (120, 160, 40),
    (200, 120, 160),
    (80, 120, 200),
]


# Phrase boundaries are persisted as float32.  At track-scale timestamps its
# resolution is around 10-30 microseconds, so sub-0.1 ms fragments can collapse
# to zero on save and become impossible to select in the editor.
MIN_PHRASE_DUR = 1e-4
# Shorter than any real phrase (a beat at 300 BPM is 0.2 s): leftovers of a
# boundary placed on an older beatgrid. Dropped, and gaps this small closed.
PHRASE_SLIVER_SEC = 0.05


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


def normalize_phrase_segments(
    phrases,
    *,
    min_time: float | None = None,
    max_time: float | None = None,
) -> list[dict[str, Any]]:
    """Return finite, ordered, non-overlapping Phrase segments.

    Optional time bounds clip stale segments left behind after beatgrid or
    duration edits.  Tiny float precision slivers are removed before they can
    collapse into zero-length float32 records.
    """

    lower = float(min_time) if min_time is not None and math.isfinite(min_time) else None
    upper = float(max_time) if max_time is not None and math.isfinite(max_time) else None
    if lower is not None and upper is not None and upper < lower:
        lower, upper = upper, lower

    rows: list[dict[str, Any]] = []
    for p in phrases or []:
        try:
            start = float(p.get("start"))
            end = float(p.get("end"))
        except Exception:
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if lower is not None:
            start = max(start, lower)
            end = max(end, lower)
        if upper is not None:
            start = min(start, upper)
            end = min(end, upper)
        if end - start <= MIN_PHRASE_DUR:
            continue
        label = normalize_base_label(p.get("label"))
        rows.append({"start": start, "end": end, "label": label})
    rows.sort(key=lambda r: r["start"])

    normalized: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        if normalized:
            previous_end = float(normalized[-1]["end"])
            if float(current["start"]) < previous_end + MIN_PHRASE_DUR:
                current["start"] = previous_end
            elif float(current["start"]) - previous_end < PHRASE_SLIVER_SEC:
                # a hairline gap left by a boundary from an older beatgrid
                normalized[-1]["end"] = float(current["start"])
        if float(current["end"]) - float(current["start"]) < PHRASE_SLIVER_SEC:
            continue
        normalized.append(current)
    return normalized


def _sorted_phrases(phrases) -> list[dict[str, Any]]:
    return normalize_phrase_segments(phrases)


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
            result.append(left)
        if not inserted:
            result.append(dict(new_seg))
            inserted = True
        if e > end:
            right = dict(seg)
            right["start"] = end
            result.append(right)
    if not inserted:
        result.append(dict(new_seg))

    # Adjacent segments that share a label are kept separate (not merged), so
    # repeated sections stay individually numbered (e.g. VERSE1, VERSE2).
    return normalize_phrase_segments(result)


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
            result.append(left)
        if e > end:
            right = dict(seg)
            right["start"] = end
            result.append(right)
    return normalize_phrase_segments(result)


def build_phrase_segments_np(phrases) -> Dict[str, np.ndarray]:
    """Serialize phrases (base labels) into an NPZ-friendly block."""
    rows = _sorted_phrases(phrases)
    if not rows:
        return {
            "start": np.asarray([], dtype=np.float32),
            "end": np.asarray([], dtype=np.float32),
            "label": np.asarray([], dtype="U1"),
        }
    starts = np.asarray([r["start"] for r in rows], dtype=np.float32)
    ends = np.asarray([r["end"] for r in rows], dtype=np.float32)
    # Validate after quantization too: distinct float64 boundaries can map to
    # the same float32 timestamp at the end of a long track.
    valid = (ends.astype(np.float64) - starts.astype(np.float64)) > MIN_PHRASE_DUR
    rows = [row for row, keep in zip(rows, valid) if bool(keep)]
    starts = starts[valid]
    ends = ends[valid]
    if not rows:
        return {
            "start": np.asarray([], dtype=np.float32),
            "end": np.asarray([], dtype=np.float32),
            "label": np.asarray([], dtype="U1"),
        }
    label_width = max(1, max(len(r["label"]) for r in rows))
    return {
        "start": starts,
        "end": ends,
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


def merge_fill_phrases(
    phrases,
    *,
    orphan_fallback: bool = False,
) -> List[dict[str, Any]]:
    """Absorb fill ranges into their adjacent non-fill phrase.

    FILL_IN belongs to the following phrase and FILL_OUT belongs to the
    preceding phrase. ``orphan_fallback`` permits training data at a track edge
    to use the opposite neighbour.
    """

    rows = _sorted_phrases(phrases)
    if not rows:
        return []
    fill_labels = {"FILL_IN", "FILL_OUT"}

    def nearest_nonfill(index: int, step: int) -> int | None:
        position = index + step
        while 0 <= position < len(rows):
            if str(rows[position]["label"]).strip().upper() not in fill_labels:
                return position
            position += step
        return None

    labeled: list[tuple[dict[str, Any], bool]] = []
    for index, row in enumerate(rows):
        label = str(row["label"]).strip().upper()
        if label in fill_labels:
            preferred_step = 1 if label == "FILL_IN" else -1
            target = nearest_nonfill(index, preferred_step)
            if target is None and orphan_fallback:
                target = nearest_nonfill(index, -preferred_step)
            if target is None:
                continue
            current = {**row, "label": rows[target]["label"]}
            labeled.append((current, True))
        else:
            labeled.append((dict(row), False))

    merged: list[dict[str, Any]] = []
    merged_absorbed: list[bool] = []
    for current, from_fill in labeled:
        if (
            merged
            and merged[-1]["label"] == current["label"]
            and abs(float(merged[-1]["end"]) - float(current["start"]))
            <= MIN_PHRASE_DUR
            and (merged_absorbed[-1] or from_fill)
        ):
            merged[-1]["end"] = max(merged[-1]["end"], current["end"])
            merged_absorbed[-1] = merged_absorbed[-1] or from_fill
        else:
            merged.append(current)
            merged_absorbed.append(from_fill)
    return normalize_phrase_segments(merged)


def merge_fill_phrases_for_display(phrases) -> List[dict[str, Any]]:
    """Return display-only phrases with FILL_IN/OUT absorbed into neighbours.

    FILL_IN is shown as part of the following non-fill phrase. FILL_OUT is shown
    as part of the previous non-fill phrase. Orphan fills are omitted.
    """
    return merge_fill_phrases(phrases, orphan_fallback=False)


def extract_fill_phrase_markers(phrases) -> List[dict[str, Any]]:
    """Return display marker ranges for fill phrases.

    direction is "in" for FILL_IN and "out" for FILL_OUT.
    """
    rows = _sorted_phrases(phrases)
    markers: list[dict[str, Any]] = []
    for seg in rows:
        label = str(seg.get("label", "") or "").strip().upper()
        if label == "FILL_IN":
            direction = "in"
        elif label == "FILL_OUT":
            direction = "out"
        else:
            continue
        markers.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "direction": direction,
            }
        )
    return markers


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
