from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from PySide6 import QtCore, QtWidgets

from analyzer.music.marks import JumpCue, JumpCueGroup, cues_by_point
from ui.edit_panel.cue_slots import CuePageWidget, color_swatch_icon, slot_letter
from ui.edit_panel.icons import icon
from ui.edit_panel.quantize import beat_quantize_grid
from core.marks.colors import JUMPCUE_PAIR_COLORS, get_jumpcue_pair_color
from core.marks.phrases import assigned_phrases
from core.track.reanalysis import JUMPCUE, ReanalysisRequest
from system.i18n import tr

# JumpCUE slots are the letters A-Z (rekordbox hot cue letters).
JUMPCUE_SLOTS = 26

# the editor's working form of the JumpCUEs: each group's cues and its colour preset
Groups = list[tuple[list[JumpCue], int]]


def _jumpcue_name(cue: JumpCue) -> str:
    """A JumpCUE's user name: its comment, unless that is just its letter."""
    comment = str(cue.comment or "").strip()
    return "" if comment == str(cue.label or "").strip() else comment


def format_cue_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:06.3f}"


class JumpCUEEditMixin:
    """JumpCUE list on the CUE page: 4 x 2 slots per page (A-D, E-H, then I-P ...),
    each colour | letter | name (double-click to rename) | SET / JUMP (SET on an
    empty slot). A colour is a group (the cues of one colour jump between each
    other): the squircle's menu picks the cue's colour, which assigns its group,
    shifts that colour's cues by a beat, moves slots and deletes the cue (the only
    way to delete one)."""

    def _build_jumpcue_ui(self, page: CuePageWidget) -> list[QtWidgets.QWidget]:
        """JumpCUE slots of the CUE page; returns the buttons locked while editing is."""
        self.cue_page = page
        self.jc_reanalyze = QtWidgets.QToolButton()
        self.jc_reanalyze.setIcon(icon("beatgrid_reanalyze"))
        self.jc_reanalyze.setIconSize(QtCore.QSize(14, 14))
        self.jc_reanalyze.setToolTip(tr("Reanalyze the JumpCUEs with the current beatgrid"))
        page.tool_slot.addWidget(self.jc_reanalyze)
        self.jc_reanalyze.clicked.connect(self._reanalyze_jumpCUE)

        self.jc_slots = page.slots[0]
        for i, slot in enumerate(self.jc_slots):
            slot.squircle.setToolTip(tr("Color (its group), beat moves, slots, delete"))
            slot.squircle_clicked.connect(lambda i=i: self._jc_group_menu(i))
            slot.action_clicked.connect(lambda i=i: self._jc_set_or_jump(i))
            slot.renamed.connect(lambda name, i=i: self._rename_jumpcue(i, name))
        page.page_changed.connect(self._refresh_jumpcue_slots)
        # the JumpCUE armed, as the player's deck reports it (its destination, s; NaN: none)
        self._jc_armed_dest = math.nan
        self.bus.sig_deck_state.connect(self._on_deck_jump)
        # SET (an empty slot's button) and the menu's edits follow the edit lock in
        # _refresh_jumpcue_slots / _jc_group_menu; JUMP stays usable
        return [self.jc_reanalyze]

    # ---- slots ----

    def _jc_cues(self) -> list[tuple[JumpCue, int]]:
        """Every cue with its group's index, by point: made once per list of groups the model holds (it replaces the
        list on every edit), however many slots and colours ask."""
        groups = self._jump_cue_groups
        if getattr(self, "_jc_cues_of", None) is not groups:
            self._jc_cues_of, self._jc_cues_list = groups, cues_by_point(groups)
        return self._jc_cues_list

    def _jc_slot_letter(self, slot: int) -> str | None:
        i = self.cue_page.page_start(0) + int(slot)
        return slot_letter(i) if 0 <= i < JUMPCUE_SLOTS else None

    def _jc_cue_at(self, slot: int) -> JumpCue | None:
        """The cue in a slot: the one with the slot's letter."""
        return self._jc_cue_by_letter(self._jc_slot_letter(slot))

    def _jc_cue_by_letter(self, letter: str | None) -> JumpCue | None:
        return next((c for c, _gi in self._jc_cues() if c.label == letter), None)

    def _jc_group_of(self, cue: JumpCue) -> int:
        return next(gi for gi, group in enumerate(self._jump_cue_groups) if cue in group.cues)

    def _refresh_jumpcue_slots(self) -> None:
        if not hasattr(self, "cue_page"):
            return
        used = [ord(c.label[:1] or "A") - ord("A") for c, _gi in self._jc_cues()]
        self.cue_page.update_pager(0, max(used, default=-1) + 1, max_items=JUMPCUE_SLOTS)
        armed_point = self._jc_armed_dest if math.isfinite(self._jc_armed_dest) else None
        editable = bool(getattr(self, "_edit_enabled", True))
        for i, slot in enumerate(self.jc_slots):
            cue = self._jc_cue_at(i)
            if cue is None:
                slot.show_cue(None)
                slot.setEnabled(self._jc_slot_letter(i) is not None)
                slot.btn_action.setEnabled(editable)
                slot.btn_action.setToolTip(tr("SET: a JumpCUE in this slot at the nearest beat or track edge"))
                slot.set_action_lit(False)
                continue
            slot.setEnabled(True)
            slot.btn_action.setEnabled(True)
            slot.btn_action.setToolTip(tr("Jump here on the next beat (when stopped: move the playhead here)"))
            point = float(cue.point)
            color = get_jumpcue_pair_color(self._jump_cue_groups[self._jc_group_of(cue)].color_idx)
            slot.show_cue(tuple(color), cue.label, _jumpcue_name(cue), format_cue_time(point))
            slot.name.setToolTip(tr("{time} - double-click to rename").format(time=format_cue_time(point)))
            slot.set_action_lit(armed_point is not None and abs(point - armed_point) < 1e-3)

    def _rename_jumpcue(self, slot: int, name: str) -> None:
        """The name is the cue's comment; no name keeps the comment at its letter."""
        cue = self._jc_cue_at(slot)
        if cue is None or not getattr(self, "_edit_enabled", True):
            return
        self._set_jumpcue_groups([([replace(c, comment=name or c.label) if abs(c.point - cue.point) < 1e-3 else c
                                    for c in cues], color) for cues, color in self._jumpcue_groups()])

    def _jc_set_or_jump(self, slot: int) -> None:
        """The slot's button: SET on an empty slot (an edit), else JUMP."""
        if self._jc_cue_at(slot) is not None:
            self._jc_jump(slot)
            return
        letter = self._jc_slot_letter(slot)
        if letter is not None and getattr(self, "_edit_enabled", True):
            self._new_jumpcue_group(letter)

    # ---- jump ----

    def _jc_jump(self, slot: int) -> None:
        cue = self._jc_cue_at(slot)
        if cue is not None:
            self._jc_jump_cue(cue)

    def _jc_jump_cue(self, cue: JumpCue) -> None:
        """JUMP to the cue: the player's deck does it (playing: on the next beat, pressed again before it the
        jump stays as it is; stopped: the playhead there) and reports the jump armed (_on_deck_jump)."""
        self.bus.sig_jump_to_requested.emit(float(cue.point))

    def _on_deck_jump(self, report) -> None:
        dest = float(report.jump_dest)
        if dest != self._jc_armed_dest and not (math.isnan(dest) and math.isnan(self._jc_armed_dest)):
            self._jc_armed_dest = dest
            self._refresh_jumpcue_slots()

    # ---- group menu ----

    def _jc_group_menu(self, slot: int) -> None:
        cue = self._jc_cue_at(slot)
        if cue is None:
            return
        gidx = self._jc_group_of(cue)
        point = float(cue.point)
        current = int(self._jump_cue_groups[gidx].color_idx)
        menu = QtWidgets.QMenu(self)
        menu.addSection(tr("Color (cues of one color jump between each other)"))
        for ci, rgb in enumerate(JUMPCUE_PAIR_COLORS):
            members = [c.label for c, gi in self._jc_cues() if int(self._jump_cue_groups[gi].color_idx) == ci]
            act = menu.addAction(color_swatch_icon(rgb), " ".join(members) if members else tr("(unused)"))
            act.setCheckable(True)
            act.setChecked(current == ci)
            act.triggered.connect(lambda _=False, ci=ci: self._set_jumpcue_color(point, ci))
        menu.addSection(tr("This color"))
        menu.addAction(tr("One beat earlier"), lambda: self._shift_jumpcue_group(gidx, -1))
        menu.addAction(tr("One beat later"), lambda: self._shift_jumpcue_group(gidx, 1))
        menu.addSection(tr("Slots"))
        letter = cue.label
        act = menu.addAction(tr("Insert a slot at {letter} (this and later cues move one slot on)").format(letter=letter),
                             lambda: self._insert_jumpcue_slot(letter))
        act.setEnabled(self._can_insert_jumpcue_slot(letter))
        prev = chr(ord(letter) - 1) if letter > "A" else letter
        act = menu.addAction(tr("Move into the empty slot {letter} (this and later cues move one slot back)").format(letter=prev),
                             lambda: self._shift_jumpcue_slots(letter, -1))
        act.setEnabled(self._can_close_jumpcue_slot(letter))
        menu.addSeparator()
        menu.addAction(tr("Delete JumpCUE {letter}").format(letter=letter), lambda: self._remove_jumpcue_member(point))
        if not getattr(self, "_edit_enabled", True):
            # every edit in this menu follows the edit lock
            for a in menu.actions():
                a.setEnabled(False)
        menu.exec(self.jc_slots[slot].squircle.mapToGlobal(QtCore.QPoint(0, self.jc_slots[slot].squircle.height())))

    # ---- JumpCUE group edits ----
    # An edit takes the groups (_jumpcue_groups: each one's cues and colour preset), changes them and sets them on
    # the model (_set_jumpcue_groups); cues keep their letters (slots). The squircle keeps one group per colour.

    def _jumpcue_groups(self) -> Groups:
        """The current JumpCUE groups in the editor's working form (in the model's order)."""
        return [(list(group.cues), int(group.color_idx)) for group in self._jump_cue_groups]

    def _unused_jumpcue_color(self, groups: Groups) -> int | None:
        used = {int(color) for cues, color in groups if cues}
        return next((ci for ci in range(len(JUMPCUE_PAIR_COLORS)) if ci not in used), None)

    def _nearest_beat_index(self, t: float) -> int:
        grid = beat_quantize_grid(self._beatgrid, self.duration)
        return int(np.argmin(np.abs(grid - float(t))))

    def _beat_at(self, idx: int) -> float:
        grid = beat_quantize_grid(self._beatgrid, self.duration)
        return float(grid[int(np.clip(idx, 0, grid.size - 1))])

    def _set_jumpcue_groups(self, groups: Groups) -> bool:
        """Replace the JumpCUEs with ``groups`` (empty groups are dropped) and record the edit; cues keep their
        letters (slots). False (nothing changed) when two cues share a point."""
        groups = [(sorted(cues, key=lambda c: c.point), color) for cues, color in groups if cues]
        points = np.asarray(sorted(c.point for cues, _color in groups for c in cues), dtype=float)
        if points.size > 1 and np.min(np.diff(points)) < 1e-3:
            QtWidgets.QMessageBox.warning(self, "JumpCUE", tr("Two JumpCUEs cannot share a point."))
            return False
        try:
            self.model.set_jump_cues([JumpCueGroup(tuple(replace(c, comment=c.comment or c.label) for c in cues),
                                                   int(color)) for cues, color in groups])
        except ValueError as exc:      # a letter that is not a slot's
            QtWidgets.QMessageBox.warning(self, tr("Invalid JumpCUE Label"), str(exc))
            return False
        return True

    def _set_jumpcue_color(self, point: float, color_idx: int) -> None:
        """Give the cue at ``point`` a colour; the colour is its group: it joins the
        group of that colour, or becomes one of its own."""
        groups = self._jumpcue_groups()
        cue = next((c for cues, _color in groups for c in cues if abs(c.point - point) < 1e-3), None)
        if cue is None:
            return
        groups = [([c for c in cues if abs(c.point - point) >= 1e-3], color) for cues, color in groups]
        self._set_jumpcue_groups(self._joined(groups, cue, color_idx))

    @staticmethod
    def _joined(groups: Groups, cue: JumpCue, color_idx: int) -> Groups:
        """``groups`` with ``cue`` in the group of ``color_idx`` (the colour is the group), or in one of its own."""
        target = next((k for k, (cues, color) in enumerate(groups) if cues and int(color) == int(color_idx)), None)
        if target is None:
            groups.append(([cue], int(color_idx)))
        else:
            groups[target][0].append(cue)
        return groups

    def _shift_jumpcue_group(self, gidx: int, beats: int) -> None:
        """Move a group (points and passages) by ``beats`` beats."""
        groups = self._jumpcue_groups()
        if not 0 <= gidx < len(groups) or len(self._beatgrid) == 0:
            return
        n = beat_quantize_grid(self._beatgrid, self.duration).size
        moved = []
        for cue in groups[gidx][0]:
            k = self._nearest_beat_index(cue.point) + int(beats)
            if not 0 <= k < n:
                return
            moved.append(replace(cue, start=self._beat_at(self._nearest_beat_index(cue.start) + int(beats)),
                                 end=self._beat_at(self._nearest_beat_index(cue.end) + int(beats)),
                                 point=self._beat_at(k)))
        groups[gidx] = (moved, groups[gidx][1])
        self._set_jumpcue_groups(groups)

    def _can_insert_jumpcue_slot(self, letter: str) -> bool:
        """The cues from ``letter`` on can move one slot on: the last slot (Z) is free
        or before ``letter``."""
        last = slot_letter(JUMPCUE_SLOTS - 1)
        return letter < last and not any(c.label == last for c, _gi in self._jc_cues())

    def _can_close_jumpcue_slot(self, letter: str) -> bool:
        """The slot before ``letter`` exists and is empty."""
        if letter <= "A":
            return False
        prev = chr(ord(letter) - 1)
        return not any(c.label == prev for c, _gi in self._jc_cues())

    def _insert_jumpcue_slot(self, letter: str) -> None:
        self._shift_jumpcue_slots(letter, 1)

    def _shift_jumpcue_slots(self, letter: str, step: int) -> None:
        """Move the cue in slot ``letter`` and every cue in a later slot one slot on
        (``step`` 1, leaving ``letter`` empty) or back into the empty slot before it
        (``step`` -1). Names that are just the letter follow it."""
        ok = self._can_insert_jumpcue_slot(letter) if step > 0 else self._can_close_jumpcue_slot(letter)
        if not ok:
            return

        def moved(c: JumpCue) -> JumpCue:
            if c.label < letter:
                return c
            nxt = chr(ord(c.label) + (1 if step > 0 else -1))
            return replace(c, label=nxt, comment=nxt if c.comment == c.label else c.comment)

        self._set_jumpcue_groups([([moved(c) for c in cues], color) for cues, color in self._jumpcue_groups()])

    def _remove_jumpcue_member(self, point: float) -> None:
        self._set_jumpcue_groups([([c for c in cues if abs(c.point - point) >= 1e-3], color)
                                  for cues, color in self._jumpcue_groups()])

    def _new_jumpcue_group(self, letter: str) -> None:
        """A cue in slot ``letter`` at the nearest beat or track edge, in an
        unused colour (a group of its own); with every colour in use, in the
        preceding cue's colour (its group)."""
        if len(self._beatgrid) == 0:
            return
        p = self._beat_at(self._nearest_beat_index(self._current_time))
        color = self._unused_jumpcue_color(self._jumpcue_groups())
        if color is None:
            cues = self._jc_cues()
            before = [item for item in cues if item[0].point < p]
            _cue, gi = before[-1] if before else cues[0]
            color = int(self._jump_cue_groups[gi].color_idx)
        self._insert_jumpcue(p, color, letter)

    def _insert_jumpcue(self, point: float, color_idx: int, letter: str) -> None:
        """A new cue in slot ``letter`` at ``point``, in the group of ``color_idx`` (or a group of its own); where a
        cue already is, _set_jumpcue_groups refuses it and says so (two cues never share a point: each is a slot)."""
        cue = JumpCue(label=letter, comment=letter, start=point, end=point, point=point)
        self._set_jumpcue_groups(self._joined(self._jumpcue_groups(), cue, color_idx))

    def _reanalyze_jumpCUE(self) -> None:
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        # AutoCUE follows the current phrases and beat grid.
        self.bus.sig_segment_reanalyze_requested.emit(ReanalysisRequest(
            kind=JUMPCUE, path=self._current_track_path(), beats=np.asarray(self._beatgrid, dtype=float).copy(),
            tempo_segments=np.asarray(self._segments if self._segments is not None else np.empty((0, 6)), dtype=float).copy(), phrases=tuple(assigned_phrases(self._phrase_segments))))
