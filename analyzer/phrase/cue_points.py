"""The CUE points the analysis sets: at phrase starts / ends by each label's rule, and AutoCUE's (where the LEAD_IN
ends, an early first downbeat, or 0:00), as analyzer.music.marks.CuePoint."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from analyzer.music.beat_geometry import downbeat_times
from analyzer.music.marks import CuePoint
from analyzer.settings import CueRule

_DEDUP_TIME_EPSILON_SEC = 1e-3


def phrase_cue_points(phrases: list[dict[str, Any]] | None, beat_times: Any, tempo_segments: Any,
                      rules: Mapping[str, CueRule], autocue: str) -> list[CuePoint]:
    """The CUE points of the phrases: each label's rule, and with AutoCUE on CUE points (``autocue`` "cue_point") its
    start."""
    points = build_phrase_cue_points(phrases, rules)
    if autocue == "cue_point":
        points = add_autocue_cue_point(points, autocue_time_sec(phrases, beat_times, tempo_segments),
                                       lead_in_end=lead_in_end_sec(phrases) is not None)
    return points


def _append_unique(parts: list[str], value: object) -> None:
    text = str(value or "").strip()
    if text and text not in parts:
        parts.append(text)


def merge_coincident(points: list[CuePoint]) -> list[CuePoint]:
    """The points in time order (``id`` orders points at one time; None: their place in ``points``), those within a
    millisecond of each other one (their labels and comments joined, the first one's colour), renumbered."""
    if not points:
        return []
    rows = sorted(enumerate(points), key=lambda item: (float(item[1].time_sec),
                                                       item[0] if item[1].id is None else int(item[1].id)))
    merged: list[tuple[float, list[str], list[str], int]] = []
    for _index, point in rows:
        time_sec = float(point.time_sec)
        if merged and abs(merged[-1][0] - time_sec) <= _DEDUP_TIME_EPSILON_SEC:
            _append_unique(merged[-1][1], point.label)
            _append_unique(merged[-1][2], point.comment)
            continue
        labels: list[str] = []
        comments: list[str] = []
        _append_unique(labels, point.label)
        _append_unique(comments, point.comment)
        merged.append((time_sec, labels, comments, int(point.color_idx or 0)))
    return [CuePoint(time_sec=time_sec, label="/".join(labels), comment=" / ".join(comments), color_idx=color,
                     id=index)
            for index, (time_sec, labels, comments, color) in enumerate(merged)]


def build_phrase_cue_points(phrases: list[dict[str, Any]] | None, rules: Mapping[str, CueRule]) -> list[CuePoint]:
    """CUE points at phrase starts / ends by the rule of each phrase's label
    (``rules``: uppercase label -> object with ``at_start``, ``at_end`` and
    ``same_label_boundaries``, analyzer.settings.CueRule). A start or end
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

    points: list[CuePoint] = []
    for i, phrase in enumerate(rows):
        label = phrase["label"]
        rule = rules.get(label)
        if rule is None:
            continue
        prev_same = i > 0 and rows[i - 1]["label"] == label
        next_same = i + 1 < len(rows) and rows[i + 1]["label"] == label
        if rule.at_start and (rule.same_label_boundaries or not prev_same):
            points.append(CuePoint(id=len(points), time_sec=phrase["start"], label=label,
                                   comment=f"{label.title()} start"))
        if rule.at_end and (rule.same_label_boundaries or not next_same):
            points.append(CuePoint(id=len(points), time_sec=phrase["end"], label=f"{label}_OUT",
                                   comment=f"{label.title()} end"))
    return merge_coincident(points)


def lead_in_end_sec(phrases: list[dict[str, Any]] | None) -> float | None:
    """Where the song's LEAD_IN (its first phrase, when labelled so) ends."""
    rows = []
    for phrase in phrases or []:
        try:
            rows.append((float(phrase.get("start")), float(phrase.get("end")),
                         str(phrase.get("label", "") or "").strip().upper()))
        except (AttributeError, TypeError, ValueError):
            continue
    if not rows:
        return None
    start, end, label = min(rows)
    if label != "LEAD_IN" or not (math.isfinite(end) and end > start):
        return None
    return end


def autocue_time_sec(
    phrases: list[dict[str, Any]] | None,
    beat_times: Any = None,
    tempo_segments: Any = None,
) -> float:
    """Choose the automatic start cue, even when there is no ``LEAD_IN``.

    A detected lead-in still wins.  Without one, a downbeat less than one beat
    from the start is treated as a short pre-roll and used as the cue.  If the
    first available downbeat is later (or cannot be found), the track has
    effectively started before the grid's first bar and 0:00 is the safest cue.
    """
    lead_in_end = lead_in_end_sec(phrases)
    if lead_in_end is not None:
        return float(lead_in_end)

    try:
        beats = np.asarray(beat_times, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        beats = np.empty(0, dtype=float)
    beats = np.sort(beats[np.isfinite(beats) & (beats >= 0.0)])

    downbeats = downbeat_times(beats, tempo_segments)
    if downbeats is None:
        return 0.0
    downbeats = np.asarray(downbeats, dtype=float).reshape(-1)
    downbeats = np.sort(downbeats[np.isfinite(downbeats) & (downbeats >= 0.0)])
    if downbeats.size == 0:
        return 0.0
    first_downbeat = float(downbeats[0])
    if first_downbeat <= _DEDUP_TIME_EPSILON_SEC:
        return first_downbeat

    # The early grid is the most faithful measure for a possible start pre-roll.
    # Fall back to the tempo segment when too few beat marks are available.
    positive_diffs = np.diff(beats[:9])
    positive_diffs = positive_diffs[np.isfinite(positive_diffs) & (positive_diffs > 0.0)]
    beat_duration = float(np.median(positive_diffs)) if positive_diffs.size else math.nan
    if not math.isfinite(beat_duration):
        try:
            segments = np.asarray(tempo_segments, dtype=float)
            if segments.ndim == 1:
                segments = segments.reshape(1, -1)
            candidates = [
                row for row in segments
                if row.size >= 3 and row[0] <= first_downbeat < row[1]
                and math.isfinite(float(row[2])) and float(row[2]) > 0.0
            ]
            if candidates:
                beat_duration = 60.0 / float(candidates[0][2])
        except (TypeError, ValueError):
            pass

    if math.isfinite(beat_duration) and first_downbeat < beat_duration:
        return first_downbeat
    return 0.0


def add_autocue_cue_point(points: list[CuePoint], time_sec: float, *, lead_in_end: bool = False) -> list[CuePoint]:
    """Add the selected AutoCUE position, merging a coincident phrase cue."""
    point = CuePoint(id=len(points), time_sec=float(time_sec), label="LEAD_IN_OUT" if lead_in_end else "AUTOCUE",
                     comment="Lead-in end" if lead_in_end else "AutoCUE")
    return merge_coincident([*points, point])
