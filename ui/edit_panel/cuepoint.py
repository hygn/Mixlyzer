from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ui.edit_panel.cue_slots import SLOTS_PER_PAGE, CuePageWidget, color_swatch_icon
from ui.edit_panel.icons import icon
from ui.edit_panel.jumpcue import format_cue_time
from ui.edit_panel.quantize import quantize_to_beat_or_edge
from utils.jumpcue_colors import CUE_POINT_COLORS, get_cue_point_color
from core.i18n import tr

# the playhead is on a CUE point this close to it (CALL puts it exactly there)
CUE_AT_PLAYHEAD_SEC = 0.05


class CuePointEditMixin:
    """CUE point (memory cue) list on the CUE page, as rekordbox's memory cues: kept
    in time order, set with MEMORY at the playhead (on the nearest beat), each row
    colour (its menu: the colour, delete) | time | name (double-click to rename) | CALL.
    CALL moves the playhead to the cue while stopped (not while playing). CUE points
    are the user's: made by the full analysis from the phrases, then only edited here."""

    def _build_cuepoint_ui(self, page: CuePageWidget) -> list[QtWidgets.QWidget]:
        self.cp_memory = QtWidgets.QToolButton()
        self.cp_memory.setIcon(icon("cue_memory"))
        self.cp_memory.setIconSize(QtCore.QSize(14, 14))
        self.cp_memory.setToolTip(tr("MEMORY: set a CUE point at the nearest beat or track edge"))
        self.cp_memory.clicked.connect(self._add_memory_cue)
        page.tool_slot.addWidget(self.cp_memory)

        self.cp_slots = page.slots[1]
        time_w = QtGui.QFontMetrics(self.cp_slots[0].letter.font()).horizontalAdvance("00:00.000") + 2
        for i, slot in enumerate(self.cp_slots):
            slot.letter.setFixedWidth(time_w)  # the time where the JumpCUE list has its letter
            policy = slot.btn_action.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            slot.btn_action.setSizePolicy(policy)
            slot.btn_action.setToolTip(tr("Move the playhead to this CUE (not while playing)"))
            slot.squircle.setToolTip(tr("CUE color / delete"))
            slot.squircle_clicked.connect(lambda i=i: self._cp_color_menu(i))
            slot.action_clicked.connect(lambda i=i: self._cp_call(i))
            slot.renamed.connect(lambda name, i=i: self._rename_cue_point(i, name))
        page.page_changed.connect(self._refresh_cue_point_slots)
        return [self.cp_memory]

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
            slot.btn_action.setVisible(cue is not None)
            if cue is None:
                slot.show_cue(None)
                continue
            t = format_cue_time(float(cue.get("time_sec", 0.0)))
            slot.show_cue(get_cue_point_color(int(cue.get("color_idx", 0) or 0)), t, str(cue.get("label", "") or ""))
            slot.name.setToolTip(tr("{time} - double-click to rename").format(time=t))
            slot.btn_action.setEnabled(not playing)

    def _add_memory_cue(self) -> None:
        """MEMORY: a CUE point at the nearest beat or track edge (at the
        playhead without a beatgrid); the list shows the page it lands on."""
        t = float(self._current_time)
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size:
            t = quantize_to_beat_or_edge(t, beats, self.duration)
        if any(abs(float(p["time_sec"]) - t) < 1e-3 for p in self._cue_points):
            return
        points = [dict(p) for p in self._cue_points]
        points.append({"time_sec": t, "label": "", "comment": "", "color_idx": 0})
        self.model.set_cue_points(points)
        index = next(i for i, p in enumerate(self._cue_points) if abs(float(p["time_sec"]) - t) < 1e-3)
        self.cue_page.pages[1] = index // SLOTS_PER_PAGE
        self._refresh_cue_point_slots()

    def _delete_memory_cue(self, slot: int) -> None:
        cue = self._cp_at(slot)
        if cue is not None:
            self._delete_cue_point(cue)

    def _cue_point_at_playhead(self) -> dict | None:
        """The CUE point the playhead is on (within CUE_AT_PLAYHEAD_SEC), as after CALL."""
        t = float(self._current_time)
        near = [p for p in self._cue_points if abs(float(p["time_sec"]) - t) <= CUE_AT_PLAYHEAD_SEC]
        return min(near, key=lambda p: abs(float(p["time_sec"]) - t), default=None)

    def _cp_call_target(self, step: int) -> dict | None:
        """The CUE point before (``step`` < 0) / after the playhead."""
        t = float(self._current_time)
        if step > 0:
            return next((p for p in self._cue_points if float(p["time_sec"]) > t + 1e-3), None)
        return next((p for p in reversed(self._cue_points) if float(p["time_sec"]) < t - 1e-3), None)

    def _show_cue_point_slot(self, cue: dict):
        """The CUE point list at the page holding ``cue``; its slot."""
        index = next(i for i, p in enumerate(self._cue_points) if p is cue)
        self.cue_page.show_list(1, index // SLOTS_PER_PAGE)
        return self.cp_slots[index % SLOTS_PER_PAGE]

    def _delete_cue_point(self, cue: dict) -> None:
        t = float(cue["time_sec"])
        self.model.set_cue_points([p for p in self._cue_points if abs(float(p["time_sec"]) - t) >= 1e-3])

    def _rename_cue_point(self, slot: int, name: str) -> None:
        cue = self._cp_at(slot)
        if cue is None or not getattr(self, "_edit_enabled", True):
            return
        self._update_cue_point(cue, label=name)

    def _cp_call(self, slot: int) -> None:
        """CALL (stopped): the player's deck moves the playhead and its cue point there."""
        cue = self._cp_at(slot)
        if cue is None or getattr(self, "_playing", False):
            return
        self.bus.sig_call_requested.emit(float(cue.get("time_sec", 0.0)))

    def _cp_color_menu(self, slot: int) -> None:
        cue = self._cp_at(slot)
        if cue is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addSection(tr("CUE color"))
        for ci, rgb in enumerate(CUE_POINT_COLORS):
            act = menu.addAction(color_swatch_icon(rgb), tr("Color {number}").format(number=ci + 1))
            act.setCheckable(True)
            act.setChecked(int(cue.get("color_idx", 0) or 0) == ci)
            act.triggered.connect(lambda _=False, ci=ci: self._update_cue_point(cue, color_idx=ci))
        menu.addSeparator()
        menu.addAction(tr("Delete CUE point {time}").format(time=format_cue_time(float(cue.get("time_sec", 0.0)))),
                       lambda: self._delete_memory_cue(slot))
        if not getattr(self, "_edit_enabled", True):
            # every edit in this menu follows the edit lock, as the JumpCUE menu
            for a in menu.actions():
                a.setEnabled(False)
        btn = self.cp_slots[slot].squircle
        menu.exec(btn.mapToGlobal(QtCore.QPoint(0, btn.height())))

    def _update_cue_point(self, cue: dict, **fields) -> None:
        points = [dict(p) for p in self._cue_points]
        for p in points:
            if abs(float(p["time_sec"]) - float(cue["time_sec"])) < 1e-3:
                p.update(fields)
        self.model.set_cue_points(points)
