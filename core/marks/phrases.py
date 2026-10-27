"""What a phrase is to the app and how it is shown: the labels (no others), UNASSIGNED where none is, the phrases
ordered and cleaned, the track covered, the labels numbered, named (the settings' aliases) and coloured. No storage:
the stored block is core.analysis_library.phrases'."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, List

import numpy as np

from analyzer.music.marks import Phrase


# The phrase labels (uppercase); no others.
PHRASE_LABELS: list[str] = [
    "LEAD_IN",
    "INTRO",
    "VERSE",
    "CHORUS",
    "BREAK_CHORUS",
    "BRIDGE",
    "OUTRO",
    "INTERLUDE",
    "SILENCE",
]

# Where no phrase is assigned (gaps, or a track without phrase analysis). Shown
# and edited like a label, never stored: stored phrases leave it as a gap.
UNASSIGNED = "UNASSIGNED"

_PHRASE_COLORS: dict[str, tuple[int, int, int]] = {
    "LEAD_IN": (0, 180, 200),
    "INTRO": (70, 130, 180),
    "VERSE": (46, 139, 87),
    "CHORUS": (220, 60, 90),
    "BREAK_CHORUS": (235, 140, 120),
    "BRIDGE": (148, 0, 211),
    "OUTRO": (95, 95, 110),
    "INTERLUDE": (218, 165, 32),
    "SILENCE": (110, 110, 110),
    UNASSIGNED: (52, 52, 58),
}


# How labels are shown (config viewconfig.phrase_labels, set by the app): label ->
# (alias, overview abbreviation), "" for the defaults.
_NAMES: dict[str, tuple[str, str]] = {}


def set_phrase_label_names(names: dict[str, Any]) -> None:
    """Set the labels' aliases / overview abbreviations (label -> object with
    ``alias`` and ``abbrev``, see core.settings.config.phraselabelnames)."""
    _NAMES.clear()
    for label, name in names.items():
        _NAMES[str(label)] = (str(name.alias or "").strip(), str(name.abbrev or "").strip())


def phrase_display_name(label: str) -> str:
    """The label as shown: its alias, else itself."""
    return _NAMES.get(label, ("", ""))[0] or label


def phrase_abbrev(label: str) -> str:
    """The label in the overview: its abbreviation, else the first letter of its name."""
    return _NAMES.get(label, ("", ""))[1] or phrase_display_name(label)[:1]


# Phrase boundaries are persisted as float32.  At track-scale timestamps its
# resolution is around 10-30 microseconds, so sub-0.1 ms fragments can collapse
# to zero on save and become impossible to select in the editor.
MIN_PHRASE_DUR = 1e-4
# Shorter than any real phrase (a beat at 300 BPM is 0.2 s): leftovers of a
# boundary placed on an older beatgrid. Dropped, and gaps this small closed.
PHRASE_SLIVER_SEC = 0.05


def phrase_color(label: str) -> tuple[int, int, int]:
    return _PHRASE_COLORS.get(str(label), _PHRASE_COLORS[UNASSIGNED])


def normalize_phrase_segments(
    phrases,
    *,
    min_time: float | None = None,
    max_time: float | None = None,
) -> list[Phrase]:
    """Return finite, ordered, non-overlapping Phrase segments.

    Optional time bounds clip stale segments left behind after beatgrid or
    duration edits.  Tiny float precision slivers are removed before they can
    collapse into zero-length float32 records.
    """

    lower = float(min_time) if min_time is not None and math.isfinite(min_time) else None
    upper = float(max_time) if max_time is not None and math.isfinite(max_time) else None
    if lower is not None and upper is not None and upper < lower:
        lower, upper = upper, lower

    rows: list[Phrase] = []
    for p in phrases or []:
        try:
            start = float(p.start)       # not a Phrase: an error, never left out quietly
            end = float(p.end)
        except (TypeError, ValueError):
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
        label = str(p.label or "").strip()
        rows.append(Phrase(start, end, label))
    rows.sort(key=lambda r: r.start)

    normalized: list[Phrase] = []
    for current in rows:
        if normalized:
            previous_end = float(normalized[-1].end)
            if float(current.start) < previous_end + MIN_PHRASE_DUR:
                current = replace(current, start=previous_end)
            elif float(current.start) - previous_end < PHRASE_SLIVER_SEC:
                # a hairline gap left by a boundary from an older beatgrid
                normalized[-1] = replace(normalized[-1], end=float(current.start))
        if float(current.end) - float(current.start) < PHRASE_SLIVER_SEC:
            continue
        normalized.append(current)
    return normalized


def _sorted_phrases(phrases) -> list[Phrase]:
    return normalize_phrase_segments(phrases)


def _phrase_numbers(rows) -> list[str]:
    """Each phrase's number among those of its label ("" for a label that occurs
    once, and for UNASSIGNED)."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.label] = counts.get(r.label, 0) + 1
    running: dict[str, int] = {}
    out: list[str] = []
    for r in rows:
        base = r.label
        if counts[base] >= 2 and base != UNASSIGNED:
            running[base] = running.get(base, 0) + 1
            out.append(str(running[base]))
        else:
            out.append("")
    return out


def number_phrase_labels(phrases) -> list[str]:
    """Display names in time order: the label's name, numbered from 1 when the
    label occurs more than once (e.g. CHORUS1, CHORUS2)."""
    rows = _sorted_phrases(phrases)
    return [phrase_display_name(r.label) + n for r, n in zip(rows, _phrase_numbers(rows))]


def abbreviate_phrase_labels(phrases) -> list[str]:
    """Overview labels: the abbreviation + number (e.g. CHORUS2 -> "C2", VERSE -> "V")."""
    rows = _sorted_phrases(phrases)
    return [phrase_abbrev(r.label) + n for r, n in zip(rows, _phrase_numbers(rows))]


def fill_unassigned(phrases, duration_sec) -> List[Phrase]:
    """The phrases with every stretch of [0, duration] they leave uncovered as an
    UNASSIGNED phrase (the whole track when there are none); phrases past the
    end are cut there."""
    try:
        duration = float(np.asarray(duration_sec, dtype=float).ravel()[0])
    except Exception:
        duration = 0.0
    if not (math.isfinite(duration) and duration > 0.0):
        return _sorted_phrases(phrases)
    rows = normalize_phrase_segments(phrases, min_time=0.0, max_time=duration)
    if rows and rows[0].start < PHRASE_SLIVER_SEC:            # a hairline at either end closed, as between phrases
        rows[0] = replace(rows[0], start=0.0)
    if rows and duration - rows[-1].end < PHRASE_SLIVER_SEC:
        rows[-1] = replace(rows[-1], end=duration)
    out: list[Phrase] = []
    cursor = 0.0
    for row in rows:
        if row.start - cursor >= PHRASE_SLIVER_SEC:
            out.append(Phrase(cursor, row.start, UNASSIGNED))
        out.append(row)
        cursor = max(cursor, row.end)
    if duration - cursor >= PHRASE_SLIVER_SEC:
        out.append(Phrase(cursor, duration, UNASSIGNED))
    return out


def assigned_phrases(phrases) -> List[Phrase]:
    """The phrases to store: UNASSIGNED ones left out (gaps)."""
    return [row for row in _sorted_phrases(phrases) if row.label != UNASSIGNED]


def phrase_strip_segments(phrases) -> list[tuple[float, float, tuple[int, int, int]]]:
    """(start, end, colour) of each phrase, for a SegmentStrip."""
    return [(seg.start, seg.end, phrase_color(seg.label)) for seg in _sorted_phrases(phrases)]
