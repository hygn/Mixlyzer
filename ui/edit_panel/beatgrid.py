from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from analyzer_core.editor.beatgrid import canonical_inizio, shift_grid_in_seg
from foundation.tempo_rows import (bpm_at, is_ramp_row, rebuild_grid_from_segments, retempo_row, segment_end_bpm, split_row,
                             with_end_bpm)
from foundation.beat_geometry import MAX_BEATS_PER_BAR
from ui.edit_panel.blocks import mode_column, rule_cell
from ui.edit_panel.icons import badged_icon, icon
from ui.theme import current_theme
from ui.edit_panel.styles import BUTTON_STYLESHEET, WARNING_BUTTON_STYLESHEET
from foundation.i18n import tr, tr_mark
from ui.live_label import LiveLabel

# time signatures with their own button; the rest are under Custom
TS_PRESETS = (3, 4, 5, 7)

# the tools whose target follows the edit scope: icon name -> (tooltip with the whole grid, with the current
# segment), translated where shown
SCOPED_TOOLS = {
    "shift_half_back": (tr_mark("Shift the whole beatgrid (every segment) half a beat earlier"),
                        tr_mark("Shift the current segment half a beat earlier")),
    "shift_coarse_back": (tr_mark("Shift the whole beatgrid (every segment) 10 ms earlier"),
                          tr_mark("Shift the current segment 10 ms earlier")),
    "shift_fine_back": (tr_mark("Shift the whole beatgrid (every segment) 1 ms earlier"),
                        tr_mark("Shift the current segment 1 ms earlier")),
    "shift_fine_fwd": (tr_mark("Shift the whole beatgrid (every segment) 1 ms later"),
                       tr_mark("Shift the current segment 1 ms later")),
    "shift_coarse_fwd": (tr_mark("Shift the whole beatgrid (every segment) 10 ms later"),
                         tr_mark("Shift the current segment 10 ms later")),
    "shift_half_fwd": (tr_mark("Shift the whole beatgrid (every segment) half a beat later"),
                       tr_mark("Shift the current segment half a beat later")),
    "bpm_half": (tr_mark("Halve the BPM of the whole beatgrid (every segment); the downbeats stay"),
                 tr_mark("Halve the BPM of the current segment; the downbeats stay")),
    "bpm_double": (tr_mark("Double the BPM of the whole beatgrid (every segment); the downbeats stay"),
                   tr_mark("Double the BPM of the current segment; the downbeats stay")),
}

# the TAP icon is a word: no padding, so it is not squeezed in its narrow button
_TAP_STYLE = "QToolButton { padding: 0; }"


class BeatgridEditMixin:
    """Beatgrid section: shift, reference BPM (tap / assign / reanalysis), BPM x2 / x1/2, merge / split,
    downbeat, edit scope (whole grid or current segment), time signature."""

    def _build_beatgrid_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """Beatgrid widgets on the Beatgrid page, three button rows high: the edit scope
        (whole grid / current segment), the edit tools (6x3) and the time signature (2x3),
        a rule between them; returns the edit buttons."""

        def _btn(icon_name: str, tip: str, *, checkable: bool = False) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setIcon(icon(f"beatgrid_{icon_name}"))
            btn.setIconSize(QtCore.QSize(19, 19))
            btn.setToolTip(tip)
            btn.setCheckable(checkable)
            return btn

        # Row 1: shift (icons and tooltips follow the edit scope: _refresh_scoped_tools)
        self.btn_shift_half_back = _btn("shift_half_back", "")
        self.btn_shift_coarse_back = _btn("shift_coarse_back", "")
        self.btn_shift_fine_back = _btn("shift_fine_back", "")
        self.btn_shift_fine_fwd = _btn("shift_fine_fwd", "")
        self.btn_shift_coarse_fwd = _btn("shift_coarse_fwd", "")
        self.btn_shift_half_fwd = _btn("shift_half_fwd", "")

        # Row 2: reference BPM and tempo
        self.btn_zeroBPM = _btn("zero", tr("Clear the reference BPM and the tap history"))
        # ZERO and TAP share one cell; the reference BPM (tapped or typed) has its own
        self.btn_tapBPM = QtWidgets.QToolButton()
        self.btn_tapBPM.setIcon(icon("beatgrid_tap"))
        self.btn_tapBPM.setIconSize(QtCore.QSize(24, 12))
        self.btn_tapBPM.setStyleSheet(_TAP_STYLE)
        self.btn_tapBPM.setToolTip(tr("Tap the beat to set the reference BPM"))
        self.zero_tap_cell = QtWidgets.QWidget()
        zt_lay = QtWidgets.QHBoxLayout(self.zero_tap_cell)
        zt_lay.setContentsMargins(0, 0, 0, 0)
        zt_lay.setSpacing(2)
        half_w = (self.BTN_WIDTH - 2) // 2
        for b in (self.btn_zeroBPM, self.btn_tapBPM):
            b.setFixedSize(half_w, self.BTN_HEIGHT)
            zt_lay.addWidget(b)
        self.inp_refBPM = QtWidgets.QLineEdit()
        ref_bpm_regex = QtCore.QRegularExpression(r"^$|^(?:\d+\.?\d*|\.\d+)$")
        self._ref_bpm_validator = QtGui.QRegularExpressionValidator(
            ref_bpm_regex, self.inp_refBPM
        )
        self.inp_refBPM.setValidator(self._ref_bpm_validator)
        self.inp_refBPM.setPlaceholderText(tr("Ref"))
        self.inp_refBPM.setAlignment(QtCore.Qt.AlignCenter)
        self.inp_refBPM.setStyleSheet("QLineEdit { padding: 0 2px; }")
        self.inp_refBPM.setToolTip(tr("Reference BPM (tapped or typed)"))
        self.btn_setBPM = _btn("assign", tr("Assign the reference BPM to the current segment (downbeat at its start)"))
        self.btn_ramp_start = _btn("ramp_start", tr("Make the reference BPM the current segment's starting tempo (a ramp to its end tempo; its first beat stays)"))
        self.btn_ramp_end = _btn("ramp_end", tr("Make the reference BPM the current segment's ending tempo (a ramp from its start tempo; its first beat stays)"))
        self.btn_reanalyze_segment = _btn("reanalyze", "")

        # Row 3: split / merge, downbeat, BPM x1/2 / x2 (these two follow the edit scope)
        self.btn_merge_prev = _btn("merge_prev", tr("Merge the previous segment into the current one (keeps the current tempo / phase)"))
        self.btn_merge_next = _btn("merge_next", tr("Merge the next segment into the current one (keeps the current tempo / phase)"))
        self.btn_cutseg = _btn("split", tr("Split the current segment at the playhead"))
        self.btn_set_downbeat = _btn("downbeat", tr("Make the beat nearest the playhead the current segment's downbeat"))
        self.btn_bpm_half = _btn("bpm_half", "")
        self.btn_bpm_double = _btn("bpm_double", "")

        # The edit scope, apart from the tools: what shift and BPM x2 / x1/2 apply to
        self.btn_scope_all = _btn("scope_all", tr("Edit the whole beatgrid: shift and BPM x2 / x1/2 apply to every segment"), checkable=True)
        self.btn_scope_segment = _btn("scope_segment", tr("Edit the current segment: shift and BPM x2 / x1/2 apply to it only"), checkable=True)
        self.btn_scope_segment.setChecked(True)
        self._beatgrid_scope_group = QtWidgets.QButtonGroup(self)
        self._beatgrid_scope_group.setExclusive(True)
        self._beatgrid_scope_group.addButton(self.btn_scope_all)
        self._beatgrid_scope_group.addButton(self.btn_scope_segment)
        self._scoped_tools = {
            "shift_half_back": self.btn_shift_half_back, "shift_coarse_back": self.btn_shift_coarse_back,
            "shift_fine_back": self.btn_shift_fine_back, "shift_fine_fwd": self.btn_shift_fine_fwd,
            "shift_coarse_fwd": self.btn_shift_coarse_fwd, "shift_half_fwd": self.btn_shift_half_fwd,
            "bpm_half": self.btn_bpm_half, "bpm_double": self.btn_bpm_double,
        }

        # Per-segment time signature (n/4): presets and Custom (1-8 from a menu), applied
        # to the current segment on click; the segment's own is filled blue.
        self.ts_buttons: dict[int, QtWidgets.QToolButton] = {}
        for n in TS_PRESETS:
            b = _btn(f"ts_{n}", tr("Set the current segment to {meter}").format(meter=f"{n}/4"))
            b.setIconSize(QtCore.QSize(36, 18))
            b.clicked.connect(lambda _=False, n=n: self._apply_time_signature(n))
            self.ts_buttons[n] = b
        self.btn_ts_custom = _btn("ts_custom", tr("Set the current segment to another time signature (1/4 - 8/4)"))
        self.btn_ts_custom.setIconSize(QtCore.QSize(36, 18))
        custom_menu = QtWidgets.QMenu(self.btn_ts_custom)
        for n in range(1, MAX_BEATS_PER_BAR + 1):
            custom_menu.addAction(f"{n}/4", lambda n=n: self._apply_time_signature(n))
        self.btn_ts_custom.clicked.connect(
            lambda: custom_menu.exec(self.btn_ts_custom.mapToGlobal(QtCore.QPoint(0, self.btn_ts_custom.height())))
        )
        self._ts_current = 4
        # metronome on / off, the same setting as in the settings (saved); stays usable
        # while editing is locked
        self.btn_metronome = _btn("metronome", tr("Metronome on / off"), checkable=True)
        self.btn_metronome.toggled.connect(self.bus.sig_metronome_toggle_requested.emit)

        self.beatgrid_status = LiveLabel(tr("Seg -- | Elapsed: 00:00.00 | Left: 00:00.00"))
        self.beatgrid_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        rows = [
            [self.btn_shift_half_back, self.btn_shift_coarse_back, self.btn_shift_fine_back,
             self.btn_shift_fine_fwd, self.btn_shift_coarse_fwd, self.btn_shift_half_fwd],
            [self.zero_tap_cell, self.inp_refBPM, self.btn_setBPM, self.btn_ramp_start, self.btn_ramp_end,
             self.btn_reanalyze_segment],
            [self.btn_cutseg, self.btn_merge_prev, self.btn_merge_next, self.btn_set_downbeat,
             self.btn_bpm_half, self.btn_bpm_double],
        ]
        ts_cells = [
            [self.ts_buttons[3], self.ts_buttons[4]],
            [self.ts_buttons[5], self.ts_buttons[7]],
            [self.btn_metronome, self.btn_ts_custom],
        ]

        # [scope] | [tools 6x3] | [time signature 2x3]: each block three button rows high. The
        # tools as on every page (blocks.py): moves in time on top, values in the middle, split /
        # merge first in the last row
        for btn in (self.btn_scope_all, self.btn_scope_segment):
            btn.setIconSize(QtCore.QSize(26, 26))
        self._mode_pairs.append((self.btn_scope_all, self.btn_scope_segment))
        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        grid.addWidget(mode_column(self.btn_scope_all, self.btn_scope_segment), 0, 0, 3, 1)
        grid.addWidget(rule_cell(), 0, 1, 3, 1)
        tools_col = 2
        for r, row in enumerate(rows):
            for c, w in enumerate(row):
                grid.addWidget(w, r, tools_col + c, _AL)
        ts_col = tools_col + max(len(row) for row in rows)
        grid.addWidget(rule_cell(), 0, ts_col, 3, 1)
        for r, row in enumerate(ts_cells):
            for c, w in enumerate(row):
                grid.addWidget(w, r, ts_col + 1 + c, _AL)
        grid.addWidget(self.beatgrid_status, 3, 0, 1, ts_col + 3, _AL)
        self._beatgrid_scope_group.buttonToggled.connect(self._refresh_scoped_tools)
        self._refresh_scoped_tools()

        self.btn_shift_half_back.clicked.connect(lambda: self._shift_beatgrid_by_beats(-0.5))
        self.btn_shift_coarse_back.clicked.connect(lambda: self._shift_beatgrid_by(-0.010))
        self.btn_shift_fine_back.clicked.connect(lambda: self._shift_beatgrid_by(-0.001))
        self.btn_shift_fine_fwd.clicked.connect(lambda: self._shift_beatgrid_by(0.001))
        self.btn_shift_coarse_fwd.clicked.connect(lambda: self._shift_beatgrid_by(0.010))
        self.btn_shift_half_fwd.clicked.connect(lambda: self._shift_beatgrid_by_beats(0.5))
        self.btn_zeroBPM.clicked.connect(self._zeroBPM)
        self.btn_tapBPM.clicked.connect(self._tapBPM_set)
        self.btn_setBPM.clicked.connect(self._setBPM_set)
        self.btn_ramp_start.clicked.connect(lambda: self._set_ramp_bpm(at_end=False))
        self.btn_ramp_end.clicked.connect(lambda: self._set_ramp_bpm(at_end=True))
        self.btn_reanalyze_segment.clicked.connect(self._reanalyze_segment_auto)
        self.btn_bpm_half.clicked.connect(lambda: self._scale_bpm(0.5))
        self.btn_bpm_double.clicked.connect(lambda: self._scale_bpm(2.0))
        self.btn_merge_prev.clicked.connect(lambda: self._merge_segments(direction="prev", keep="current"))
        self.btn_merge_next.clicked.connect(lambda: self._merge_segments(direction="next", keep="current"))
        self.btn_cutseg.clicked.connect(self._cut_current_segment)
        self.btn_set_downbeat.clicked.connect(self._set_downbeat_at_playhead)
        self.inp_refBPM.textChanged.connect(self._refresh_reanalyze_button)
        self._refresh_reanalyze_button()

        self._refresh_time_sig_buttons()
        return [w for row in rows + ts_cells for w in row if w is not self.btn_metronome] + [
            self.btn_scope_all, self.btn_scope_segment]

    def _refresh_scoped_tools(self, *_args) -> None:
        """Shift and BPM x2 / x1/2 show what they apply to: the icon's strip and the tooltip follow
        the edit scope."""
        whole = self._edit_whole_grid()
        for name, btn in self._scoped_tools.items():
            btn.setIcon(badged_icon(f"beatgrid_{name}", "scope_all" if whole else "scope_segment"))
            btn.setToolTip(tr(SCOPED_TOOLS[name][0 if whole else 1]))

    def set_metronome_checked(self, on: bool) -> None:
        """Show the metronome setting (changed elsewhere, e.g. in the settings)."""
        self.btn_metronome.blockSignals(True)
        self.btn_metronome.setChecked(bool(on))
        self.btn_metronome.blockSignals(False)

    def _reference_bpm(self) -> float | None:
        try:
            bpm = float(self.inp_refBPM.text().strip())
        except (TypeError, ValueError):
            return None
        return bpm if np.isfinite(bpm) and bpm > 0 else None

    def _refresh_reanalyze_button(self, *_args) -> None:
        """Segment reanalysis follows the reference BPM when there is one."""
        if self._reference_bpm() is None:
            self.btn_reanalyze_segment.setIcon(icon("beatgrid_reanalyze"))
            self.btn_reanalyze_segment.setToolTip(tr("Reanalyze the current segment's beatgrid from the audio"))
        else:
            self.btn_reanalyze_segment.setIcon(icon("beatgrid_reanalyze_ref"))
            self.btn_reanalyze_segment.setToolTip(tr("Reanalyze the current segment's beatgrid at the reference BPM"))

    def _reanalyze_segment_auto(self) -> None:
        if self._reference_bpm() is None:
            self._reanalyze_current_segment()
        else:
            self._reanalyze_with_reference_bpm()

    def _edit_whole_grid(self) -> bool:
        return self.btn_scope_all.isChecked()

    def _scope_indices(self) -> list[int]:
        """Segments a shift / BPM scale applies to under the selected scope."""
        if self._edit_whole_grid():
            return list(range(len(self._segments)))
        return [self._cur_idx] if self._has_active_segment() else []

    def _set_status(self, idx: Optional[int], elapsed: float, left: float) -> None:
        def fmt(sec: float) -> str:
            m = int(sec // 60)
            s = int(sec - 60 * m)
            cs = int((sec - 60 * m - s) * 100)
            return f"{m:02d}:{s:02d}.{cs:02d}"
        if idx is None:
            self.beatgrid_status.setText(tr("Seg -- | Elapsed: 00:00.00 | Left: 00:00.00"))
        else:
            self.beatgrid_status.setText(tr("Seg {index} | Elapsed: {elapsed} | Left: {left}").format(
                index=idx + 1, elapsed=fmt(elapsed), left=fmt(left)))
        self._refresh_time_sig_selector()
        self._update_phrase_status()

    @staticmethod
    def _ensure_ts_column(segments: np.ndarray) -> np.ndarray:
        """Return an (N,6) copy ``[start, end, bpm, inizio, ts_num, bpm_end]`` (ts_num 4 and a
        steady bpm_end backfilled when absent)."""
        return with_end_bpm(segments)

    def _current_segment_time_sig(self) -> int:
        if not self._has_active_segment() or self._segments.shape[1] < 5:
            return 4
        value = self._segments[self._cur_idx, 4]
        return int(round(float(value))) if np.isfinite(value) and value >= 1 else 4

    def _refresh_time_sig_selector(self) -> None:
        """Follow the active segment: its time signature is the one lit."""
        ts = self._current_segment_time_sig()
        if ts == self._ts_current:
            return
        self._ts_current = ts
        self._refresh_time_sig_buttons()

    def _refresh_time_sig_buttons(self) -> None:
        theme = current_theme()
        custom = self._ts_current not in TS_PRESETS
        self.btn_ts_custom.setIcon(icon(f"beatgrid_ts_{self._ts_current}" if custom else "beatgrid_ts_custom"))
        lit = self.btn_ts_custom if custom else self.ts_buttons[self._ts_current]
        for btn in (*self.ts_buttons.values(), self.btn_ts_custom):
            btn.setStyleSheet(
                f"QToolButton {{ background-color: {theme.accent}; border: 1px solid {theme.accent}; }}"
                if btn is lit else ""
            )

    def _apply_time_signature(self, value: int) -> None:
        """Set the current segment's time signature."""
        if not self._has_active_segment():
            return
        value = int(value)
        if value < 1:
            return
        segments = self._ensure_ts_column(self._segments)
        idx = self._cur_idx
        if int(round(float(segments[idx, 4]))) == value:
            return
        segments[idx, 4] = float(value)
        self.model.set_beatgrid(self._beatgrid, segments)

    def _has_active_segment(self) -> bool:
        """Return True when a valid editing target exists."""
        return self._segments.size > 0 and 0 <= self._cur_idx < len(self._segments)

    def _shift_beatgrid_by(self, offset: float) -> None:
        """Shift the beats of the scope's segments by ``offset`` seconds."""
        targets = self._scope_indices()
        if not targets or len(self._beatgrid) == 0:
            return
        beatgrid, segments = self._beatgrid, self._segments
        for idx in targets:
            bpm = float(segments[idx][2])
            if not (np.isfinite(bpm) and bpm > 0):
                continue
            beatgrid, segments = shift_grid_in_seg(beatgrid, segments, idx, offset)
        self.model.set_beatgrid(beatgrid, segments)

    def _shift_beatgrid_by_beats(self, beats: float) -> None:
        """Shift each of the scope's segments by ``beats`` of its own beat period."""
        targets = self._scope_indices()
        if not targets or len(self._beatgrid) == 0:
            return
        beatgrid, segments = self._beatgrid, self._segments
        for idx in targets:
            # A beat of the segment at its downbeat (a ramp's tempo there).
            bpm = bpm_at(segments[idx], float(segments[idx][3]))
            if not (np.isfinite(bpm) and bpm > 0):
                continue
            beatgrid, segments = shift_grid_in_seg(beatgrid, segments, idx, beats * 60.0 / bpm)
        self.model.set_beatgrid(beatgrid, segments)

    def _scale_bpm(self, factor: float) -> None:
        """BPM x2 / x1/2 over the scope; each segment keeps its downbeat (inizio)."""
        targets = self._scope_indices()
        if not targets:
            return
        segments = self._ensure_ts_column(self._segments)
        changed = False
        for idx in targets:
            st, _ed, bpm, inz, ts_num = (float(v) for v in segments[idx][:5])
            new_bpm = bpm * factor
            if not (np.isfinite(new_bpm) and new_bpm > 0):
                continue
            ramp = is_ramp_row(segments[idx])
            segments[idx, 2] = new_bpm
            segments[idx, 5] *= factor
            if not ramp:
                segments[idx, 3] = canonical_inizio(inz, st, new_bpm, ts_num)
            changed = True
        if not changed:
            return
        self.model.set_beatgrid(rebuild_grid_from_segments(segments), segments)

    def _set_downbeat_at_playhead(self) -> None:
        """The beat nearest the playhead (inside the current segment) becomes its downbeat."""
        if not self._has_active_segment() or len(self._beatgrid) == 0:
            return
        segments = self._ensure_ts_column(self._segments)
        st, ed, bpm, _inz, ts_num = (float(v) for v in segments[self._cur_idx][:5])
        beats = np.asarray(self._beatgrid, dtype=float)
        inside = beats[(beats >= st - 1e-3) & (beats < ed)]
        if inside.size == 0:
            return
        beat = float(inside[np.argmin(np.abs(inside - float(self._current_time)))])
        # A ramp's bars have no single length: its downbeat stays on the chosen beat.
        if is_ramp_row(segments[self._cur_idx]):
            segments[self._cur_idx, 3] = beat
        else:
            segments[self._cur_idx, 3] = canonical_inizio(beat, st, bpm, ts_num)
        self.model.set_beatgrid(rebuild_grid_from_segments(segments), segments)

    def _cut_current_segment(self) -> None:
        """Split the active segment at the playhead while keeping absolute downbeats."""
        if not self._has_active_segment():
            return
        segs = self._ensure_ts_column(self._segments)
        cut_t = float(self._current_time)
        st, ed, bpm, inz = segs[self._cur_idx][:4]
        ts_num = float(segs[self._cur_idx, 4])
        if not (np.isfinite(st) and np.isfinite(ed) and ed > st):
            return
        eps = 1e-3
        if cut_t <= st + eps or cut_t >= ed - eps:
            return

        new_segments = np.asarray(segs, dtype=float).copy()
        idx = self._cur_idx
        if is_ramp_row(new_segments[idx]):
            # Both parts keep the tempo line and the downbeat reference: the beats stay.
            left, right = split_row(new_segments[idx], cut_t)
            new_segments[idx] = left
            new_segments = np.insert(new_segments, idx + 1, np.asarray(right, dtype=float), axis=0)
        else:
            new_segments[idx, 1] = cut_t
            new_segments[idx, 3] = inz
            next_inz = self._next_downbeat_after(cut_t, inz, bpm, st, ts_num)
            # New split half inherits the source segment's time signature.
            next_seg = np.array([cut_t, ed, bpm, next_inz, ts_num, bpm], dtype=float)
            new_segments = np.insert(new_segments, idx + 1, next_seg, axis=0)

        self.model.set_beatgrid(rebuild_grid_from_segments(new_segments), new_segments)

    def _merge_segments(self, *, direction: str, keep: str) -> None:
        """Merge the active segment with its neighbor, preserving downbeat alignment."""
        if not self._has_active_segment():
            return
        segments = self._ensure_ts_column(self._segments)
        count = len(segments)
        idx = self._cur_idx

        if direction == "prev":
            neighbor_idx = idx - 1
            if neighbor_idx < 0:
                return
        elif direction == "next":
            neighbor_idx = idx + 1
            if neighbor_idx >= count:
                return
        else:
            return

        if keep not in {"neighbor", "current"}:
            return

        current_idx = idx
        target_idx = neighbor_idx if keep == "neighbor" else current_idx
        donor_idx = current_idx if keep == "neighbor" else neighbor_idx

        idx_a = min(current_idx, neighbor_idx)
        idx_b = max(current_idx, neighbor_idx)
        seg_start = float(min(segments[idx_a][0], segments[idx_b][0]))
        seg_end = float(max(segments[idx_a][1], segments[idx_b][1]))
        if seg_end - seg_start <= 1e-6:
            return

        updated = segments.copy()
        target = updated[target_idx].copy()
        bpm = float(target[2])
        inz = float(target[3])
        ts_num = float(target[4])
        updated[target_idx, 0] = seg_start
        updated[target_idx, 1] = seg_end
        if is_ramp_row(target):
            # The ramp's tempo line runs on over the merged span (steady where it would turn
            # non-positive).
            b0, b1 = bpm_at(target, seg_start), bpm_at(target, seg_end)
            if b0 > 0 and b1 > 0:
                updated[target_idx, 2], updated[target_idx, 5] = b0, b1
            else:
                updated[target_idx, 2] = updated[target_idx, 5] = bpm_at(target, inz)
        else:
            updated[target_idx, 2] = updated[target_idx, 5] = bpm
            updated[target_idx, 3] = canonical_inizio(inz, seg_start, bpm, ts_num)

        updated = np.delete(updated, donor_idx, axis=0)
        if donor_idx < target_idx:
            target_idx -= 1

        self._cur_idx = target_idx
        self.model.set_beatgrid(rebuild_grid_from_segments(updated), updated)

    def _reanalyze_current_segment(self) -> None:
        self._request_segment_reanalysis()

    def _reanalyze_with_reference_bpm(self) -> None:
        text = self.inp_refBPM.text().strip()
        try:
            bpm = float(text)
        except (TypeError, ValueError):
            return
        if not (np.isfinite(bpm) and bpm > 0):
            return
        self._request_segment_reanalysis(prev_bpm=bpm, use_only_prev_bpm=True)

    def _request_segment_reanalysis(self, prev_bpm: float | None = None, use_only_prev_bpm: bool = False) -> None:
        if not self._has_active_segment():
            return
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        payload = {
            "path": self._current_track_path(),
            "segment_index": int(self._cur_idx),
            "segments": np.asarray(self._segments, dtype=float).copy(),
            "beats_time_sec": np.asarray(self._beatgrid, dtype=float).copy(),
        }
        payload["analyze_type"] = "beat"
        if prev_bpm is not None:
            payload["prev_bpm"] = float(prev_bpm)
            payload["use_only_prev_bpm"] = bool(use_only_prev_bpm)
        self.bus.sig_segment_reanalyze_requested.emit(payload)

    def _on_playback_tempo_changed(self, tempofactor):
        if abs(tempofactor - 1) > 0.0001:
            self.btn_tapBPM.setToolTip(tr("Tap to set reference BPM (WARNING! Tempo factor is not 1. RefBPM Will be scaled accordingly)"))
            self.inp_refBPM.setToolTip(tr("Reference BPM for Analysis and BPM Assigning (WARNING! Tempo factor is not 1. RefBPM Will be scaled accordingly)"))
            self.btn_tapBPM.setStyleSheet(WARNING_BUTTON_STYLESHEET + _TAP_STYLE)
            self.btn_tapBPM.setIcon(icon("beatgrid_tap", inverted=True))  # dark on yellow
        else:
            self.btn_tapBPM.setToolTip(tr("Tap to set reference BPM"))
            self.inp_refBPM.setToolTip(tr("Reference BPM for Analysis and BPM Assigning"))
            self.btn_tapBPM.setStyleSheet(BUTTON_STYLESHEET + _TAP_STYLE)
            self.btn_tapBPM.setIcon(icon("beatgrid_tap"))
        self.tempo_multiplier = tempofactor
        self._zeroBPM()
    
    def _tapBPM_set(self):
        now = time.perf_counter()
        if self.prev_tap_time is None:
            # Need at least two taps to compute BPM.
            self.prev_tap_time = now
            self.inp_refBPM.setText("0")
            return
        dT = now - self.prev_tap_time
        self.prev_tap_time = now
        if dT <= 1e-6:
            return
        inst_bpm = 60.0 / dT
        if not np.isfinite(inst_bpm):
            return
        alpha = 0.1
        if self.prev_tap_BPM <= 0:
            filtered = inst_bpm
        else:
            filtered = self.prev_tap_BPM + alpha * (inst_bpm - self.prev_tap_BPM)
        self.prev_tap_BPM = filtered
        if self.tempo_multiplier <= 0:
            return  # tempo 0 (deck halted): no tempo to scale the tap back by
        self.inp_refBPM.setText(f"{filtered/self.tempo_multiplier:.2f}")
    
    def _setBPM_set(self):
        if not self._has_active_segment():
            return
        text = self.inp_refBPM.text().strip()
        try:
            bpm = float(text)
        except (TypeError, ValueError):
            return
        if not (np.isfinite(bpm) and bpm > 0):
            return
        segments = self._ensure_ts_column(self._segments)
        idx = self._cur_idx
        seg_start = float(segments[idx, 0])
        segments[idx, 2] = segments[idx, 5] = bpm   # a typed BPM makes the segment steady
        segments[idx, 3] = float(seg_start)
        self.model.set_beatgrid(rebuild_grid_from_segments(segments), segments)

    def _set_ramp_bpm(self, *, at_end: bool) -> None:
        """The reference BPM becomes the current segment's tempo at its start (or end); the other end
        keeps its tempo, so the segment becomes a linear ramp. Its first beat stays where it is."""
        if not self._has_active_segment():
            return
        bpm = self._reference_bpm()
        if bpm is None:
            return
        segments = self._ensure_ts_column(self._segments)
        idx = self._cur_idx
        row = segments[idx]
        st, ed = float(row[0]), float(row[1])
        b0, b1 = float(row[2]), segment_end_bpm(row)
        if at_end:
            b1 = bpm
        else:
            b0 = bpm
        beats = np.asarray(self._beatgrid, dtype=float)
        inside = beats[(beats >= st - 1e-3) & (beats < ed)]
        keep = float(inside[0]) if inside.size else float(row[3])
        segments[idx] = retempo_row(row, b0, b1, keep)
        self.model.set_beatgrid(rebuild_grid_from_segments(segments), segments)

    def _zeroBPM(self):
        self.prev_tap_time = None
        self.inp_refBPM.setText("")
        self.prev_tap_BPM = 0

    @staticmethod
    def _next_downbeat_after(t: float, inizio: float, bpm: float, fallback_start: float,
                             beats_per_bar: float = 4.0) -> float:
        """Return the absolute time of the first downbeat >= t (bars of ``beats_per_bar`` beats)."""
        target = float(t)
        ref = float(inizio) if np.isfinite(inizio) else float(fallback_start)
        if target <= ref + 1e-9:
            return ref
        if not (np.isfinite(bpm) and bpm > 0):
            return target
        meter = round(float(beats_per_bar)) if np.isfinite(beats_per_bar) and beats_per_bar >= 1 else 4
        bar = meter * (60.0 / float(bpm))
        if not np.isfinite(bar) or bar <= 0:
            return target
        steps = math.ceil((target - ref) / bar)
        return float(ref + steps * bar)
