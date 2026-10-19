from __future__ import annotations

from typing import Any

import numpy as np

from core.tempo_rows import END_BPM_COLUMN, segment_end_bpm
from utils.labels import idx_to_labels
from utils.phrases import assigned_phrases


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
    for start, end, bpm, ts_num, bpm_end in arr:
        if not (np.isfinite(start) and np.isfinite(end) and np.isfinite(bpm)):
            continue
        duration = float(end - start)
        if duration <= 1e-6:
            continue
        time_signature = int(round(float(ts_num))) if np.isfinite(ts_num) and ts_num >= 1 else 4
        row = {
            "seq_index": len(rows),
            "start_sec": float(start),
            "end_sec": float(end),
            "duration_sec": duration,
            "bpm": float(bpm),
            "bpm_end": float(bpm_end) if abs(float(bpm_end) - float(bpm)) > 1e-6 else None,
            "bpm_rounded": int(round(float(bpm))),
            "time_signature": time_signature,
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


def build_phrase_index_rows(
    phrases,
    bpm_segments: list[dict[str, Any]],
    key_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map every stored phrase to its BPM/key timeline without losing changes.

    A phrase becomes one or more sections, split only where an overlapping BPM
    or key segment changes.  ``phrase_seq_index`` identifies the phrase and
    ``section_index`` orders its constant BPM/key portions.  Missing BPM or key
    coverage is retained as ``None`` so the phrase itself remains indexable.
    """
    phrase_rows = assigned_phrases(phrases)
    bpm_rows = _valid_index_segments(bpm_segments)
    key_rows = _valid_index_segments(key_segments)
    out: list[dict[str, Any]] = []

    for phrase_seq, phrase in enumerate(phrase_rows):
        phrase_start = float(phrase["start"])
        phrase_end = float(phrase["end"])
        phrase_duration = phrase_end - phrase_start
        boundaries = {phrase_start, phrase_end}
        for segment in (*bpm_rows, *key_rows):
            start = float(segment["start_sec"])
            end = float(segment["end_sec"])
            if phrase_start < start < phrase_end:
                boundaries.add(start)
            if phrase_start < end < phrase_end:
                boundaries.add(end)
        ordered = sorted(boundaries)

        for section_index, (start, end) in enumerate(zip(ordered, ordered[1:])):
            if end - start <= 1e-6:
                continue
            midpoint = 0.5 * (start + end)
            bpm_row = _segment_at(bpm_rows, midpoint)
            key_row = _segment_at(key_rows, midpoint)
            bpm = _bpm_at(bpm_row, midpoint) if bpm_row else None
            key_value = key_row.get("key_value") if key_row else None
            out.append(
                {
                    "phrase_seq_index": phrase_seq,
                    "section_index": section_index,
                    "phrase_label": str(phrase.get("label") or "").strip().upper(),
                    "phrase_start_sec": phrase_start,
                    "phrase_end_sec": phrase_end,
                    "phrase_duration_sec": phrase_duration,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": end - start,
                    "bpm_seq_index": None if bpm_row is None else int(bpm_row["seq_index"]),
                    "bpm": None if bpm is None else float(bpm),
                    "bpm_rounded": None if bpm is None else int(round(float(bpm))),
                    "time_signature": None if bpm_row is None else int(bpm_row.get("time_signature", 4)),
                    "key_seq_index": None if key_row is None else int(key_row["seq_index"]),
                    "key_value": None if key_value is None else int(key_value),
                    "key_label": "" if key_row is None else str(key_row.get("key_label") or ""),
                }
            )
    return out


def _valid_index_segments(segments) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in segments or []:
        try:
            row = dict(raw)
            start = float(row["start_sec"])
            end = float(row["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(start) and np.isfinite(end)) or end - start <= 1e-6:
            continue
        row["start_sec"] = start
        row["end_sec"] = end
        row["seq_index"] = int(row.get("seq_index", len(rows)))
        rows.append(row)
    return sorted(rows, key=lambda row: (float(row["start_sec"]), int(row["seq_index"])))


def _segment_at(rows: list[dict[str, Any]], time_sec: float) -> dict[str, Any] | None:
    for row in rows:
        if float(row["start_sec"]) <= time_sec < float(row["end_sec"]):
            return row
    return None


def _normalize_tempo_segments(tempo_segments) -> np.ndarray:
    """Return (N,5) rows of [start, end, bpm, ts_num, bpm_end].

    Source tempo_segments are [start, end, bpm, inizio, ts_num, bpm_end]; inizio is not
    needed for DB segment rows. Legacy arrays without a ts_num column default
    to 4; without a bpm_end column (or with bpm_end == bpm) the row is steady.
    """
    if tempo_segments is None:
        return np.empty((0, 5), dtype=float)
    arr = np.asarray(tempo_segments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.empty((0, 5), dtype=float)
    n = arr.shape[0]
    out = np.empty((n, 5), dtype=float)
    out[:, :3] = arr[:, :3]
    out[:, 3] = arr[:, 4] if arr.shape[1] >= 5 else 4.0
    out[:, 4] = [segment_end_bpm(row) for row in arr] if arr.shape[1] > END_BPM_COLUMN else arr[:, 2]
    return out


def _bpm_at(row: dict[str, Any], time_sec: float) -> float | None:
    """Tempo of a BPM segment row at ``time_sec`` (linear across a ramp row)."""
    bpm = row.get("bpm")
    if bpm is None:
        return None
    end_bpm = row.get("bpm_end")
    start, end = float(row["start_sec"]), float(row["end_sec"])
    if end_bpm is None or end <= start:
        return float(bpm)
    return float(bpm) + (float(end_bpm) - float(bpm)) * (float(time_sec) - start) / (end - start)


def _normalize_key_segments(key_segments) -> np.ndarray:
    if key_segments is None:
        return np.empty((0, 4), dtype=float)
    arr = np.asarray(key_segments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return np.empty((0, 4), dtype=float)
    return arr[:, :4].astype(float, copy=False)


def _can_merge_bpm_rows(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("bpm_end") is None
        and right.get("bpm_end") is None
        and abs(float(left["bpm"]) - float(right["bpm"])) < 1e-6
        and int(left.get("time_signature", 4)) == int(right.get("time_signature", 4))
    )


def _can_merge_key_rows(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["key_value"]) == int(right["key_value"])
