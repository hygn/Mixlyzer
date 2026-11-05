"""Where a change of the loaded track's marks is (core.track.model): the kinds of marks it changed and, for each, the
stretch of the track it changed (pure functions on two snapshots, core.track.history.Marks). The views blink it for a
moment (an edit, undo / redo, a reanalysis's result), so a change off the screen is seen too."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from core.track.history import Marks

# the kinds of marks: the editor's pages (views.base.EDIT_PAGES), the CUE page's two apart, each drawn in its own shape
BEATGRID, KEY, PHRASE, JUMPCUE, CUE = "beatgrid", "key", "phrase", "jumpcue", "cue"


@dataclass(frozen=True)
class Change:
    """One kind of marks changed between ``start`` and ``end`` (track seconds; equal for a single point)."""
    kind: str
    start: float
    end: float


Rows = Sequence[tuple[float, float, object]]


def _rows_alike(a: Rows, b: Rows) -> int:
    """How many rows from the first on are alike in both."""
    alike = 0
    for row_a, row_b in zip(a, b):
        if row_a != row_b:
            break
        alike += 1
    return alike


def _piecewise_span(a: Rows, b: Rows) -> Optional[tuple[float, float]]:
    """Where two coverings of the track by (start, end, value) rows differ: all of the rows between those alike from the
    first on and from the last back (a split, a merge, a value changed, both rows of a boundary moved). None: alike."""
    front = _rows_alike(a, b)
    back = min(_rows_alike(a[::-1], b[::-1]), min(len(a), len(b)) - front)
    rows = [*a[front:len(a) - back], *b[front:len(b) - back]]
    return (min(float(r[0]) for r in rows), max(float(r[1]) for r in rows)) if rows else None


def _alike_run(a: np.ndarray, b: np.ndarray) -> int:
    """How many from the first on are alike in both."""
    m = min(len(a), len(b))
    alike = a[:m] == b[:m]
    return int(m if alike.all() else np.argmin(alike))


def _beat_span(a: np.ndarray, b: np.ndarray) -> Optional[tuple[float, float]]:
    """The beats that moved, came or went: between those alike from the first on and those alike from the last
    back (compared as whole arrays: a track's beats)."""
    a, b = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    front = _alike_run(a, b)
    if front == a.size == b.size:
        return None
    back = min(_alike_run(a[::-1], b[::-1]), min(a.size, b.size) - front)
    # never empty: they differ, so one of them has a beat left between the ends alike
    changed = np.concatenate([a[front:a.size - back], b[front:b.size - back]])
    return float(changed.min()), float(changed.max())


def _join(*spans) -> Optional[tuple[float, float]]:
    spans = [s for s in spans if s is not None]
    return (min(s for s, _e in spans), max(e for _s, e in spans)) if spans else None


def _points_span(times) -> Optional[tuple[float, float]]:
    times = [float(t) for t in times]
    return (min(times), max(times)) if times else None


def _jump_cue_span(before: Marks, after: Marks) -> Optional[tuple[float, float]]:
    """The JumpCUEs added, taken out or changed (a cue's group too): their points."""
    def jump(m: Marks) -> set:
        return {(c, g.color_idx) for g in m.jump_cues for c in g.cues}
    return _points_span(c.point for c, _g in jump(before) ^ jump(after))


def _cue_point_span(before: Marks, after: Marks) -> Optional[tuple[float, float]]:
    """The CUE points added, taken out or changed: their times."""
    return _points_span(p.time_sec for p in set(before.cue_points) ^ set(after.cue_points))


def changes(before: Marks, after: Marks) -> list[Change]:
    """What changed from ``before`` to ``after``: one Change per kind of marks changed (none: alike)."""
    def tempo(m: Marks) -> Rows:      # a row's tempo, downbeat and bar length: its beats and bars
        return [(r[0], r[1], tuple(r[2:])) for r in np.asarray(m.tempo_segments, dtype=float).tolist()]

    def keys(m: Marks) -> Rows:
        return [(r[2], r[3], (r[0], r[1])) for r in np.asarray(m.key_segments, dtype=float).tolist()]

    def phrases(m: Marks) -> Rows:
        return [(p.start, p.end, p.label) for p in m.phrases]

    def same(a, b) -> bool:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))

    found = []
    beatgrid = None if same(before.beats, after.beats) and same(before.tempo_segments, after.tempo_segments) else \
        _join(_beat_span(before.beats, after.beats), _piecewise_span(tempo(before), tempo(after)))
    key = None if same(before.key_segments, after.key_segments) else _piecewise_span(keys(before), keys(after))
    phrase = None if before.phrases == after.phrases else _piecewise_span(phrases(before), phrases(after))
    for kind, span in ((BEATGRID, beatgrid), (KEY, key), (PHRASE, phrase), (JUMPCUE, _jump_cue_span(before, after)),
                       (CUE, _cue_point_span(before, after))):
        if span is not None:
            found.append(Change(kind, *span))
    return found
