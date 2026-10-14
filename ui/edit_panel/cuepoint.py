from __future__ import annotations

import copy

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ui.edit_panel.cue_slots import SLOTS_PER_PAGE, CuePageWidget, color_swatch_icon
from ui.edit_panel.icons import icon
from ui.edit_panel.jumpcue import format_cue_time
from utils.cue_points import build_cue_points_np
from utils.jumpcue_colors import CUE_POINT_COLORS, get_cue_point_color


class CuePointEditMixin:
    """CUE point (memory cue) list on the CUE page, as rekordbox's memory cues: kept
    in time order, set with MEMORY at the playhead (on the nearest beat), each row
    colour | time | name (double-click to rename) | DEL | CALL. CALL moves the
    playhead to the cue while stopped (not while playing). CUE points are the
    user's: made by the full analysis from the phrases, then only edited here."""

    def _build_cuepoint_ui(self, page: CuePageWidget) -> list[QtWidgets.QWidget]:
        self.cp_memory = QtWidgets.QToolButton()
        self.cp_memory.setIcon(icon("cue_memory"))
        self.cp_memory.setIconSize(QtCore.QSize(14, 14))
        self.cp_memory.setToolTip("MEMORY: set a CUE point at the playhead (on the nearest beat)")
        self.cp_memory.clicked.connect(self._add_memory_cue)
        page.tool_slot.addWidget(self.cp_memory)

        self.cp_slots = page.slots[1]
        time_w = QtGui.QFontMetrics(self.cp_slots[0].letter.font()).horizontalAdvance("00:00.000") + 2
        for i, slot in enumerate(self.cp_slots):
            slot.letter.setFixedWidth(time_w)  # the time where the JumpCUE list has its letter
            for b in (slot.btn_set, slot.btn_action):
                policy = b.sizePolicy()
                policy.setRetainSizeWhenHidden(True)
                b.setSizePolicy(policy)
            slot.btn_set.setToolTip("Delete this CUE")
            slot.btn_action.setToolTip("Move the playhead to this CUE (not while playing)")
            slot.squircle.setToolTip("CUE colour")
            slot.squircle_clicked.connect(lambda i=i: self._cp_color_menu(i))
            slot.set_del_clicked.connect(lambda i=i: self._delete_memory_cue(i))
            slot.action_clicked.connect(lambda i=i: self._cp_call(i))
            slot.renamed.connect(lambda name, i=i: self._rename_cue_point(i, name))
        page.page_changed.connect(self._refresh_cue_point_slots)
        return [self.cp_memory, *(s.btn_set for s in self.cp_slots)]

    def set_cue_points(self, points) -> None:
        self._cue_points = sorted(
            (dict(p) for p in (points or [])), key=lambda p: float(p.get("time_sec", 0.0))
        )
        self._refresh_cue_point_slots()

    def _cp_at(self, slot: int) -> dict | None:
        i = self.cue_page.page_start(1) + int(slot)
        return self._cue_points[i] if 0 <= i < len(self._cue_points) else None

    def _refresh_cue_point_slots(self) -> None:
        if not hasattr(self, "cp_slots"):
            return
        n = len(self._cue_points)
        # no spare page: MEMORY adds a cue whatever page is shown
        self.cue_page.update_pager(1, max(0, n - 1))
        playing = bool(getattr(self, "_playing", False))
        for i, slot in enumerate(self.cp_slots):
            cue = self._cp_at(i)
            slot.btn_set.setVisible(cue is not None)
            slot.btn_action.setVisible(cue is not None)
            if cue is None:
                slot.show_cue(None)
                continue
            t = format_cue_time(float(cue.get("time_sec", 0.0)))
            slot.show_cue(get_cue_point_color(int(cue.get("color_idx", 0) or 0)), t, str(cue.get("label", "") or ""))
            slot.name.setToolTip(f"{t} - double-click to rename")
            slot.btn_action.setEnabled(not playing)

    def _add_memory_cue(self) -> None:
        """MEMORY: a CUE point at the beat nearest the playhead (at the playhead
        without a beatgrid); the list shows the page it lands on."""
        t = float(self._current_time)
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size:
            t = float(beats[int(np.argmin(np.abs(beats - t)))])
        if any(abs(float(p["time_sec"]) - t) < 1e-3 for p in self._cue_points):
            return
        points = [dict(p) for p in self._cue_points]
        points.append({"time_sec": t, "label": "", "comment": "", "color_idx": 0})
        self.set_cue_points(points)
        index = next(i for i, p in enumerate(self._cue_points) if abs(float(p["time_sec"]) - t) < 1e-3)
        self.cue_page.pages[1] = index // SLOTS_PER_PAGE
        self._refresh_cue_point_slots()
        self._emit_cue_points_update()

    def _delete_memory_cue(self, slot: int) -> None:
        cue = self._cp_at(slot)
        if cue is None:
            return
        t = float(cue["time_sec"])
        self.set_cue_points([p for p in self._cue_points if abs(float(p["time_sec"]) - t) >= 1e-3])
        self._emit_cue_points_update()

    def _rename_cue_point(self, slot: int, name: str) -> None:
        cue = self._cp_at(slot)
        if cue is None:
            return
        self._update_cue_point(cue, label=name)

    def _cp_call(self, slot: int) -> None:
        cue = self._cp_at(slot)
        if cue is None or getattr(self, "_playing", False):
            return
        self.bus.sig_seek_requested.emit(float(cue.get("time_sec", 0.0)))

    def _cp_color_menu(self, slot: int) -> None:
        cue = self._cp_at(slot)
        if cue is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addSection("CUE colour")
        for ci, rgb in enumerate(CUE_POINT_COLORS):
            act = menu.addAction(color_swatch_icon(rgb), f"Colour {ci + 1}")
            act.setCheckable(True)
            act.setChecked(int(cue.get("color_idx", 0) or 0) == ci)
            act.triggered.connect(lambda _=False, ci=ci: self._update_cue_point(cue, color_idx=ci))
        btn = self.cp_slots[slot].squircle
        menu.exec(btn.mapToGlobal(QtCore.QPoint(0, btn.height())))

    def _update_cue_point(self, cue: dict, **fields) -> None:
        points = [dict(p) for p in self._cue_points]
        for p in points:
            if abs(float(p["time_sec"]) - float(cue["time_sec"])) < 1e-3:
                p.update(fields)
        self.set_cue_points(points)
        self._emit_cue_points_update()

    def _emit_cue_points_update(self) -> None:
        self.edit_log.push(self._log_element())
        self._mark_unsaved(True)
        self._broadcast_cue_points()
        self._update_undo_redo_buttons()

    def _broadcast_cue_points(self) -> None:
        if self.model is not None:
            try:
                self.model.features["cue_points_np"] = build_cue_points_np(copy.deepcopy(self._cue_points))
            except Exception:
                pass
        self.bus.sig_cue_points_updated.emit()
