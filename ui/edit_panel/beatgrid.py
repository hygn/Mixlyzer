from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from analyzer_core.editor.beatgrid import canonical_inizio, rebuild_grid_from_segments, shift_grid_in_seg
from core.beat_geometry import MAX_BEATS_PER_BAR
from ui.edit_panel.icons import icon
from ui.theme import current_theme
from ui.edit_panel.styles import BUTTON_STYLESHEET, WARNING_BUTTON_STYLESHEET

# time signatures with their own button; the rest are under Custom
TS_PRESETS = (3, 4, 5, 7)

# the TAP icon is a word: no padding, so it is not squeezed in its narrow button
_TAP_STYLE = "QToolButton { padding: 0; }"


class BeatgridEditMixin:
    """Beatgrid section: shift, reference BPM (tap / assign / reanalysis), BPM x2 / x1/2, merge / split,
    downbeat, edit scope (whole grid or current segment), time signature."""

    def _build_beatgrid_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """Beatgrid widgets on the Beat/Key page (a 6x3 block of buttons plus the
        time signature); returns the edit buttons."""

        def _btn(icon_name: str, tip: str, *, checkable: bool = False) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setIcon(icon(f"beatgrid_{icon_name}"))
            btn.setIconSize(QtCore.QSize(19, 19))
            btn.setToolTip(tip)
            btn.setCheckable(checkable)
            return btn

        # Row 1: shift (scope: whole grid or current segment)
        self.btn_shift_half_back = _btn("shift_half_back", "Shift the beatgrid half a beat earlier")
        self.btn_shift_coarse_back = _btn("shift_coarse_back", "Shift the beatgrid 10 ms earlier")
        self.btn_shift_fine_back = _btn("shift_fine_back", "Shift the beatgrid 1 ms earlier")
        self.btn_shift_fine_fwd = _btn("shift_fine_fwd", "Shift the beatgrid 1 ms later")
        self.btn_shift_coarse_fwd = _btn("shift_coarse_fwd", "Shift the beatgrid 10 ms later")
        self.btn_shift_half_fwd = _btn("shift_half_fwd", "Shift the beatgrid half a beat later")

        # Row 2: reference BPM and tempo
        self.btn_zeroBPM = _btn("zero", "Clear the reference BPM and the tap history")
        # ZERO and TAP share one cell; the reference BPM (tapped or typed) has its own
        self.btn_tapBPM = QtWidgets.QToolButton()
        self.btn_tapBPM.setIcon(icon("beatgrid_tap"))
        self.btn_tapBPM.setIconSize(QtCore.QSize(24, 12))
        self.btn_tapBPM.setStyleSheet(_TAP_STYLE)
        self.btn_tapBPM.setToolTip("Tap the beat to set the reference BPM")
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
        self.inp_refBPM.setPlaceholderText("Ref")
        self.inp_refBPM.setAlignment(QtCore.Qt.AlignCenter)
        self.inp_refBPM.setStyleSheet("QLineEdit { padding: 0 2px; }")
        self.inp_refBPM.setToolTip("Reference BPM (tapped or typed)")
        self.btn_setBPM = _btn("assign", "Assign the reference BPM to the current segment (downbeat at its start)")
        self.btn_reanalyze_segment = _btn("reanalyze", "")
        self.btn_bpm_half = _btn("bpm_half", "Halve the BPM")
        self.btn_bpm_double = _btn("bpm_double", "Double the BPM")

        # Row 3: segments, edit scope, downbeat
        self.btn_merge_prev = _btn("merge_prev", "Merge the previous segment into the current one (keeps the current tempo / phase)")
        self.btn_merge_next = _btn("merge_next", "Merge the next segment into the current one (keeps the current tempo / phase)")
        self.btn_cutseg = _btn("split", "Split the current segment at the playhead")
        self.btn_scope_all = _btn("scope_all", "Edit the whole beatgrid: shift and BPM x2 / x1/2 apply to every segment", checkable=True)
        self.btn_scope_segment = _btn("scope_segment", "Edit the current segment: shift and BPM x2 / x1/2 apply to it only", checkable=True)
        self.btn_scope_segment.setChecked(True)
        self._beatgrid_scope_group = QtWidgets.QButtonGroup(self)
        self._beatgrid_scope_group.setExclusive(True)
        self._beatgrid_scope_group.addButton(self.btn_scope_all)
        self._beatgrid_scope_group.addButton(self.btn_scope_segment)
        self.btn_set_downbeat = _btn("downbeat", "Make the beat nearest the playhead the current segment's downbeat")

        # Per-segment time signature (n/4): presets, Custom (1-8 from a menu) and
        # Assign. The pick to assign is filled blue; the segment's current one is outlined.
        self.ts_buttons: dict[int, QtWidgets.QToolButton] = {}
        for n in TS_PRESETS:
            b = _btn(f"ts_{n}", f"Select {n}/4 (Assign applies it to the current segment)")
            b.setIconSize(QtCore.QSize(36, 18))
            b.clicked.connect(lambda _=False, n=n: self._select_time_signature(n))
            self.ts_buttons[n] = b
        self.btn_ts_custom = _btn("ts_custom", "Select another time signature (1/4 - 8/4)")
        self.btn_ts_custom.setIconSize(QtCore.QSize(36, 18))
        custom_menu = QtWidgets.QMenu(self.btn_ts_custom)
        for n in range(1, MAX_BEATS_PER_BAR + 1):
            custom_menu.addAction(f"{n}/4", lambda n=n: self._select_time_signature(n))
        self.btn_ts_custom.clicked.connect(
            lambda: custom_menu.exec(self.btn_ts_custom.mapToGlobal(QtCore.QPoint(0, self.btn_ts_custom.height())))
        )
        self.btn_apply_time_sig = _btn("ts_apply", "Apply the selected time signature to the current segment")
        self._ts_current = 4
        self._ts_selected = 4

        self.beatgrid_status = QtWidgets.QLabel("Seg -- | Elapsed: 00:00.00 | Left: 00:00.00")
        self.beatgrid_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        rows = [
            [self.btn_shift_half_back, self.btn_shift_coarse_back, self.btn_shift_fine_back,
             self.btn_shift_fine_fwd, self.btn_shift_coarse_fwd, self.btn_shift_half_fwd],
            [self.zero_tap_cell, self.inp_refBPM, self.btn_setBPM,
             self.btn_reanalyze_segment, self.btn_bpm_half, self.btn_bpm_double],
            [self.btn_merge_prev, self.btn_merge_next, self.btn_cutseg,
             self.btn_scope_all, self.btn_scope_segment, self.btn_set_downbeat],
        ]
        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        for r, row in enumerate(rows):
            for c, w in enumerate(row):
                grid.addWidget(w, r, c, _AL)
        ts_cells = [
            [self.ts_buttons[3], self.ts_buttons[4]],
            [self.ts_buttons[5], self.ts_buttons[7]],
            [self.btn_apply_time_sig, self.btn_ts_custom],
        ]
        # one rule between the beatgrid block and the time signatures
        ts_rule = QtWidgets.QFrame()
        ts_rule.setFixedWidth(1)
        ts_rule.setStyleSheet(f"background: {current_theme().border};")
        ts_rule_cell = QtWidgets.QWidget()
        rule_lay = QtWidgets.QHBoxLayout(ts_rule_cell)
        rule_lay.setContentsMargins(4, 2, 4, 2)
        rule_lay.addWidget(ts_rule)
        grid.addWidget(ts_rule_cell, 0, 6, 3, 1)
        for r, row in enumerate(ts_cells):
            for c, w in enumerate(row):
                grid.addWidget(w, r, 7 + c, _AL)
        grid.addWidget(self.beatgrid_status, 3, 0, 1, 9, _AL)

        self.btn_shift_half_back.clicked.connect(lambda: self._shift_beatgrid_by_beats(-0.5))
        self.btn_shift_coarse_back.clicked.connect(lambda: self._shift_beatgrid_by(-0.010))
        self.btn_shift_fine_back.clicked.connect(lambda: self._shift_beatgrid_by(-0.001))
        self.btn_shift_fine_fwd.clicked.connect(lambda: self._shift_beatgrid_by(0.001))
        self.btn_shift_coarse_fwd.clicked.connect(lambda: self._shift_beatgrid_by(0.010))
        self.btn_shift_half_fwd.clicked.connect(lambda: self._shift_beatgrid_by_beats(0.5))
        self.btn_zeroBPM.clicked.connect(self._zeroBPM)
        self.btn_tapBPM.clicked.connect(self._tapBPM_set)
        self.btn_setBPM.clicked.connect(self._setBPM_set)
        self.btn_reanalyze_segment.clicked.connect(self._reanalyze_segment_auto)
        self.btn_bpm_half.clicked.connect(lambda: self._scale_bpm(0.5))
        self.btn_bpm_double.clicked.connect(lambda: self._scale_bpm(2.0))
        self.btn_merge_prev.clicked.connect(lambda: self._merge_segments(direction="prev", keep="current"))
        self.btn_merge_next.clicked.connect(lambda: self._merge_segments(direction="next", keep="current"))
        self.btn_cutseg.clicked.connect(self._cut_current_segment)
        self.btn_set_downbeat.clicked.connect(self._set_downbeat_at_playhead)
        self.btn_apply_time_sig.clicked.connect(self._apply_time_signature)
        self.inp_refBPM.textChanged.connect(self._refresh_reanalyze_button)
        self._refresh_reanalyze_button()

        self._refresh_time_sig_buttons()
        return [w for row in rows + ts_cells for w in row]

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
            self.btn_reanalyze_segment.setToolTip("Reanalyze the current segment's beatgrid from the audio")
        else:
            self.btn_reanalyze_segment.setIcon(icon("beatgrid_reanalyze_ref"))
            self.btn_reanalyze_segment.setToolTip("Reanalyze the current segment's beatgrid at the reference BPM")

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
            self.beatgrid_status.setText("Seg -- | Elapsed: 00:00.00 | Left: 00:00.00")
        else:
            self.beatgrid_status.setText(f"Seg {idx+1} | Elapsed: {fmt(elapsed)} | Left: {fmt(left)}")
        self._refresh_time_sig_selector()
        self._update_phrase_status()

    @staticmethod
    def _ensure_ts_column(segments: np.ndarray) -> np.ndarray:
        """Return an (N,5) copy of segments, backfilling ts_num=4 when absent."""
        arr = np.asarray(segments, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return np.empty((0, 5), dtype=float)
        if arr.shape[1] >= 5:
            return arr[:, :5].copy()
        out = np.empty((arr.shape[0], 5), dtype=float)
        out[:, :4] = arr[:, :4]
        out[:, 4] = 4.0
        return out

    def _current_segment_time_sig(self) -> int:
        if not self._has_active_segment() or self._segments.shape[1] < 5:
            return 4
        value = self._segments[self._cur_idx, 4]
        return int(round(float(value))) if np.isfinite(value) and value >= 1 else 4

    def _refresh_time_sig_selector(self) -> None:
        """Follow the active segment: outline its time signature; moving to another
        segment (or its time signature changing) resets the pick to it."""
        ts = self._current_segment_time_sig()
        if ts == self._ts_current:
            return
        self._ts_current = ts
        self._ts_selected = ts
        self._refresh_time_sig_buttons()

    def _select_time_signature(self, n: int) -> None:
        self._ts_selected = int(n)
        self._refresh_time_sig_buttons()

    def _refresh_time_sig_buttons(self) -> None:
        theme = current_theme()
        custom_n = next((n for n in (self._ts_selected, self._ts_current) if n not in TS_PRESETS), None)
        self.btn_ts_custom.setIcon(icon(f"beatgrid_ts_{custom_n}" if custom_n else "beatgrid_ts_custom"))
        cells = {**self.ts_buttons}
        for n in (self._ts_selected, self._ts_current):
            if n not in TS_PRESETS:
                cells[n] = self.btn_ts_custom
        for btn in {*self.ts_buttons.values(), self.btn_ts_custom}:
            selected = cells.get(self._ts_selected) is btn
            current = cells.get(self._ts_current) is btn
            if selected:
                # to be assigned: filled
                style = (f"QToolButton {{ background-color: {theme.accent}; border: 1px solid {theme.accent}; }}"
                         + (f"QToolButton {{ border: 2px solid {theme.on_accent}; }}" if current else ""))
            elif current:
                # the segment's own: outlined
                style = f"QToolButton {{ border: 2px solid {theme.accent}; }}"
            else:
                style = ""
            btn.setStyleSheet(style)

    def _apply_time_signature(self) -> None:
        """Apply the selected time signature to the current segment."""
        if not self._has_active_segment():
            return
        value = int(self._ts_selected)
        if value < 1:
            return
        self._segments = self._ensure_ts_column(self._segments)
        idx = self._cur_idx
        if int(round(float(self._segments[idx, 4]))) == value:
            return
        self._segments[idx, 4] = float(value)
        self._emit_edit_update()
        self._refresh_time_sig_selector()

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
        self._beatgrid, self._segments = beatgrid, segments
        self._emit_edit_update()

    def _shift_beatgrid_by_beats(self, beats: float) -> None:
        """Shift each of the scope's segments by ``beats`` of its own beat period."""
        targets = self._scope_indices()
        if not targets or len(self._beatgrid) == 0:
            return
        beatgrid, segments = self._beatgrid, self._segments
        for idx in targets:
            bpm = float(segments[idx][2])
            if not (np.isfinite(bpm) and bpm > 0):
                continue
            beatgrid, segments = shift_grid_in_seg(beatgrid, segments, idx, beats * 60.0 / bpm)
        self._beatgrid, self._segments = beatgrid, segments
        self._emit_edit_update()

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
            segments[idx, 2] = new_bpm
            segments[idx, 3] = canonical_inizio(inz, st, new_bpm, ts_num)
            changed = True
        if not changed:
            return
        self._segments = segments
        self._beatgrid = rebuild_grid_from_segments(self._segments)
        self._emit_edit_update()
        self.update_time(self._current_time)

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
        segments[self._cur_idx, 3] = canonical_inizio(beat, st, bpm, ts_num)
        self._segments = segments
        self._beatgrid = rebuild_grid_from_segments(self._segments)
        self._emit_edit_update()
        self.update_time(self._current_time)

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
        new_segments[idx, 1] = cut_t
        new_segments[idx, 3] = inz

        next_inz = self._next_downbeat_after(cut_t, inz, bpm, st, ts_num)
        # New split half inherits the source segment's time signature.
        next_seg = np.array([cut_t, ed, bpm, next_inz, ts_num], dtype=float)
        new_segments = np.insert(new_segments, idx + 1, next_seg, axis=0)

        self._segments = new_segments
        self._beatgrid = rebuild_grid_from_segments(self._segments)
        self._emit_edit_update()
        self.update_time(self._current_time)

    def _merge_segments(self, *, direction: str, keep: str) -> None:
        """Merge the active segment with its neighbor, preserving downbeat alignment."""
        if not self._has_active_segment():
            return
        segments = np.asarray(self._segments, dtype=float)
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
        bpm = float(updated[target_idx][2])
        inz = float(updated[target_idx][3])
        updated[target_idx, 0] = seg_start
        updated[target_idx, 1] = seg_end
        updated[target_idx, 2] = bpm
        ts_num = float(updated[target_idx, 4]) if updated.shape[1] >= 5 else 4.0
        updated[target_idx, 3] = canonical_inizio(inz, seg_start, bpm, ts_num)

        updated = np.delete(updated, donor_idx, axis=0)
        if donor_idx < target_idx:
            target_idx -= 1

        self._segments = updated
        self._beatgrid = rebuild_grid_from_segments(self._segments)
        self._cur_idx = target_idx
        self._emit_edit_update()
        self.update_time(self._current_time)

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
            self.btn_tapBPM.setToolTip("Tap to set reference BPM (WARNING! Tempo factor is not 1. RefBPM Will be scaled accordingly)")
            self.inp_refBPM.setToolTip("Reference BPM for Analysis and BPM Assigning (WARNING! Tempo factor is not 1. RefBPM Will be scaled accordingly)")
            self.btn_tapBPM.setStyleSheet(WARNING_BUTTON_STYLESHEET + _TAP_STYLE)
            self.btn_tapBPM.setIcon(icon("beatgrid_tap", inverted=True))  # dark on yellow
        else:
            self.btn_tapBPM.setToolTip("Tap to set reference BPM")
            self.inp_refBPM.setToolTip("Reference BPM for Analysis and BPM Assigning")
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
        segments = np.asarray(self._segments, dtype=float).copy()
        idx = self._cur_idx
        seg_start = float(segments[idx, 0])
        segments[idx, 2] = bpm
        segments[idx, 3] = float(seg_start)
        self._segments = segments
        self._beatgrid = rebuild_grid_from_segments(self._segments)
        self._emit_edit_update()
        self.update_time(self._current_time)

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
