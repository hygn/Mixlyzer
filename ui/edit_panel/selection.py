from __future__ import annotations

from typing import Optional

import numpy as np


class SelectionMixin:
    """Beat-snapped time selection shared by the Key and Phrase sections."""

    def _key_selection_set_start(self) -> None:
        # Always reset the selection to just the start point.
        anchor = self._selection_anchor_time()
        self._selection_start = anchor
        self._selection_end = None
        self._emit_selection_range()

    def _key_selection_set_end(self) -> None:
        anchor = self._selection_anchor_time()
        start = self._selection_start
        end = self._selection_end
        if start is None:
            # No start point yet: nothing to anchor an end against.
            return
        if end is None:
            # Only a start point exists -> set the end.
            self._selection_end = anchor
            self._emit_selection_range()
            return
        # A full range exists; reassign based on the playhead position.
        range_start = min(float(start), float(end))
        range_end = max(float(start), float(end))
        if anchor > range_end:
            # After the range end: old end becomes the new start, click is the end.
            self._selection_start = range_end
            self._selection_end = anchor
        elif anchor < range_start:
            # Before the range start: old start becomes the end, click is the start.
            self._selection_start = anchor
            self._selection_end = range_start
        else:
            # Inside the range: drop the old end, click becomes the new end.
            self._selection_start = range_start
            self._selection_end = anchor
        self._emit_selection_range()

    def _key_selection_clear(self) -> None:
        self._clear_selection()

    def _key_selection_select_all(self) -> None:
        start, end = self._track_time_bounds()
        if end - start <= 1e-6:
            return
        self._selection_start = start
        self._selection_end = end
        self._emit_selection_range()

    def _clear_selection(self, notify: bool = True) -> None:
        self._selection_start = None
        self._selection_end = None
        if notify:
            self.bus.sig_key_selection_changed.emit(None)

    def _selection_anchor_time(self, t: float | None = None) -> float:
        cur = self._current_time if t is None else float(t)
        start, end = self._track_time_bounds()
        cur = float(np.clip(cur, start, end))
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size == 0:
            return cur
        idx = int(np.searchsorted(beats, cur))
        if idx <= 0:
            return start
        if idx >= len(beats):
            return end
        prev_val = float(beats[idx - 1])
        next_val = float(beats[idx])
        return prev_val if (cur - prev_val) <= (next_val - cur) else next_val

    def _track_time_bounds(self) -> tuple[float, float]:
        if self.duration:
            start = 0
            end = self.duration
        elif self._segments.size > 0:
            start = float(self._segments[0, 0])
            end = float(self._segments[-1, 1])
        elif self._beatgrid is not None and len(self._beatgrid) > 0:
            start = float(self._beatgrid[0])
            end = float(self._beatgrid[-1])
        else:
            start = 0.0
            end = max(self._current_time, 0.0)
        return (start, end)

    def _emit_selection_range(self) -> None:
        payload = self._current_selection_payload()
        self.bus.sig_key_selection_changed.emit(payload)

    def _current_selection_payload(self):
        start = self._selection_start
        end = self._selection_end
        if start is None and end is None:
            return None
        s_val = float(start) if start is not None and np.isfinite(start) else None
        e_val = float(end) if end is not None and np.isfinite(end) else None
        if s_val is not None and e_val is not None:
            if e_val < s_val:
                s_val, e_val = e_val, s_val
            if abs(e_val - s_val) <= 1e-6:
                e_val = None
        return (s_val, e_val)

    def _selection_bounds(self) -> Optional[tuple[float, float]]:
        sel = self._current_selection_payload()
        if sel is None:
            return None
        start, end = sel
        if start is None or end is None:
            return None
        return float(start), float(end)
