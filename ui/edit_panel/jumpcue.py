from __future__ import annotations

import copy

import numpy as np
from PySide6 import QtCore, QtWidgets

from ui.edit_panel.cue_slots import CuePageWidget, color_swatch_icon, slot_letter
from ui.edit_panel.history import logElement
from ui.edit_panel.icons import icon
from ui.edit_panel.styles import WARNING_BUTTON_STYLESHEET
from utils.jump_cues import build_jump_cue_graph, build_jump_cues_np
from utils.jumpcue_colors import JUMPCUE_PAIR_COLORS

# JumpCUE slots are the letters A-Z (rekordbox hot cue letters).
JUMPCUE_SLOTS = 26

# A jump is offered the next beats; the player jumps on the first one its render
# head (ahead of what is heard) has not passed yet.
JUMP_CANDIDATE_BEATS = 8


def _jumpcue_name(cue: dict) -> str:
    """A JumpCUE's user name: its comment, unless that is just its letter."""
    comment = str(cue.get("comment", "") or "").strip()
    return "" if comment == str(cue.get("label", "") or "").strip() else comment


def format_cue_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:06.3f}"


class JumpCUEEditMixin:
    """JumpCUE list on the CUE page: 4 x 2 slots per page (A-D, E-H, then I-P ...),
    each colour | letter | name (double-click to rename) | SET / DEL | JUMP. A colour is a group (the cues of one
    colour jump between each other): the squircle picks the cue's colour, which
    assigns its group, and shifts that colour's cues by a beat."""

    def _build_jumpcue_ui(self, page: CuePageWidget) -> list[QtWidgets.QWidget]:
        """JumpCUE slots of the CUE page; returns the buttons locked while editing is."""
        self.cue_page = page
        self.jc_reanalyze = QtWidgets.QToolButton()
        self.jc_reanalyze.setIcon(icon("beatgrid_reanalyze"))
        self.jc_reanalyze.setIconSize(QtCore.QSize(14, 14))
        self.jc_reanalyze.setToolTip("Reanalyze the JumpCUEs with the current beatgrid")
        page.tool_slot.addWidget(self.jc_reanalyze)
        self.jc_reanalyze.clicked.connect(self._reanalyze_jumpCUE)

        self.jc_slots = page.slots[0]
        for i, slot in enumerate(self.jc_slots):
            slot.btn_action.setToolTip("Jump here on the next beat (when stopped: move the playhead here)")
            slot.squircle.setToolTip("Colour (its group); move this colour's cues by a beat")
            slot.squircle_clicked.connect(lambda i=i: self._jc_group_menu(i))
            slot.set_del_clicked.connect(lambda i=i: self._jc_set_del(i))
            slot.action_clicked.connect(lambda i=i: self._jc_jump(i))
            slot.renamed.connect(lambda name, i=i: self._rename_jumpcue(i, name))
        page.page_changed.connect(self._refresh_jumpcue_slots)
        self.bus.sig_jump_fired.connect(self._on_jump_fired)
        self._jc_armed: dict | None = None
        return [self.jc_reanalyze, *(s.btn_set for s in self.jc_slots)]

    def set_JumpCUE(self, JumpCUE) -> None:
        if JumpCUE is None:
            JumpCUE = []
        self._JumpCUE = JumpCUE
        # cues keep their letters (slots): lettered by point only by the analysis
        self._jumpcue_graph_cues, _links = build_jump_cue_graph(self._JumpCUE, keep_labels=True)
        self._disarm_jumpcue()
        self._refresh_jumpcue_slots()

    # ---- slots ----

    def _jc_slot_letter(self, slot: int) -> str | None:
        i = self.cue_page.page_start(0) + int(slot)
        return slot_letter(i) if 0 <= i < JUMPCUE_SLOTS else None

    def _jc_cue_at(self, slot: int) -> dict | None:
        """The cue in a slot: the one with the slot's letter."""
        letter = self._jc_slot_letter(slot)
        return next((c for c in self._jumpcue_graph_cues if str(c.get("label", "")) == letter), None)

    def _refresh_jumpcue_slots(self) -> None:
        if not hasattr(self, "cue_page"):
            return
        used = [ord(str(c.get("label", "A"))[:1]) - ord("A") for c in self._jumpcue_graph_cues]
        self.cue_page.update_pager(0, max(used, default=-1) + 1, max_items=JUMPCUE_SLOTS)
        armed_point = self._jc_armed["dest"] if self._jc_armed else None
        for i, slot in enumerate(self.jc_slots):
            cue = self._jc_cue_at(i)
            if cue is None:
                slot.show_cue(None)
                slot.setEnabled(self._jc_slot_letter(i) is not None)
                slot.btn_set.setToolTip("Set a JumpCUE in this slot at the beat nearest the playhead")
                slot.set_action_lit(False)
                continue
            slot.setEnabled(True)
            point = float(cue.get("point", 0.0))
            slot.show_cue(tuple(cue["color"]), str(cue.get("label", "")),
                          _jumpcue_name(cue), format_cue_time(point))
            slot.name.setToolTip(f"{format_cue_time(point)} - double-click to rename")
            slot.btn_set.setToolTip("Delete this JumpCUE")
            slot.set_action_lit(armed_point is not None and abs(point - armed_point) < 1e-3)

    def _rename_jumpcue(self, slot: int, name: str) -> None:
        """The name is the cue's comment; no name keeps the comment at its letter."""
        cue = self._jc_cue_at(slot)
        if cue is None:
            return
        point = float(cue.get("point", 0.0))
        groups = self._jumpcue_groups()
        for g in groups.values():
            for c in g:
                if abs(c["point"] - point) < 1e-3:
                    c["comment"] = name or c["label"]
        self._set_jumpcue_groups(groups.values())

    def _jc_set_del(self, slot: int) -> None:
        cue = self._jc_cue_at(slot)
        if cue is None:
            letter = self._jc_slot_letter(slot)
            if letter is not None:
                self._new_jumpcue_group(letter)
        else:
            self._remove_jumpcue_member(float(cue.get("point", 0.0)))

    # ---- jump ----

    def _jc_jump(self, slot: int) -> None:
        """Playing: jump to the cue on the next beat (again: cancel). Stopped: move
        the playhead to it."""
        cue = self._jc_cue_at(slot)
        if cue is None:
            return
        dest = float(cue.get("point", 0.0))
        if not getattr(self, "_playing", False):
            self.bus.sig_seek_requested.emit(dest)
            return
        if self._jc_armed is not None and abs(self._jc_armed["dest"] - dest) < 1e-3:
            self._disarm_jumpcue()
            self._refresh_jumpcue_slots()
            return
        beats = np.asarray(self._beatgrid, dtype=float)
        now = float(self._current_time)
        ahead = beats[beats > now][:JUMP_CANDIDATE_BEATS]
        if ahead.size == 0:
            return
        self.bus.sig_jump_arm.emit({"source": {"point": float(ahead[0])}, "target": {"point": dest},
                                    "starts": ahead.tolist(), "once": True})
        self._jc_armed = {"src": float(ahead[-1]), "dest": dest, "from": now}
        self._refresh_jumpcue_slots()

    def _on_jump_fired(self) -> None:
        if getattr(self, "_jc_armed", None) is not None:
            self._jc_armed = None
            self._refresh_jumpcue_slots()

    def _disarm_jumpcue(self) -> None:
        if getattr(self, "_jc_armed", None) is not None:
            self._jc_armed = None
            self.bus.sig_jump_disarm.emit()

    def _follow_jumpcue_arm(self, t: float) -> None:
        """The armed jump is moot once the playhead leaves the stretch between arming
        and the last candidate beat without jumping: it was moved elsewhere."""
        armed = getattr(self, "_jc_armed", None)
        if armed is None:
            return
        if armed["from"] - 0.1 <= t <= armed["src"] + 0.02:
            return
        self._disarm_jumpcue()
        self._refresh_jumpcue_slots()

    # ---- group menu ----

    def _jc_group_menu(self, slot: int) -> None:
        cue = self._jc_cue_at(slot)
        if cue is None:
            return
        gidx = int(cue.get("graph_idx", 0) or 0)
        point = float(cue.get("point", 0.0))
        current = int(cue.get("color_idx", 0) or 0)
        menu = QtWidgets.QMenu(self)
        menu.addSection("Colour (cues of one colour jump between each other)")
        for ci, rgb in enumerate(JUMPCUE_PAIR_COLORS):
            members = [c["label"] for c in self._jumpcue_graph_cues if int(c.get("color_idx", -1)) == ci]
            act = menu.addAction(color_swatch_icon(rgb), " ".join(members) if members else "(unused)")
            act.setCheckable(True)
            act.setChecked(current == ci)
            act.triggered.connect(lambda _=False, ci=ci: self._set_jumpcue_color(point, ci))
        menu.addSection("This colour")
        menu.addAction("One beat earlier", lambda: self._shift_jumpcue_group(gidx, -1))
        menu.addAction("One beat later", lambda: self._shift_jumpcue_group(gidx, 1))
        menu.addSection("Slots")
        letter = str(cue.get("label", ""))
        act = menu.addAction(f"Insert a slot at {letter} (this and later cues move one slot on)",
                             lambda: self._insert_jumpcue_slot(letter))
        act.setEnabled(self._can_insert_jumpcue_slot(letter))
        prev = chr(ord(letter) - 1) if letter > "A" else letter
        act = menu.addAction(f"Move into the empty slot {prev} (this and later cues move one slot back)",
                             lambda: self._shift_jumpcue_slots(letter, -1))
        act.setEnabled(self._can_close_jumpcue_slot(letter))
        menu.exec(self.jc_slots[slot].squircle.mapToGlobal(QtCore.QPoint(0, self.jc_slots[slot].squircle.height())))

    # ---- JumpCUE group edits ----
    # Edits rebuild the pair list from groups of cues: every pair of members of
    # a group (a one-cue group as a pair of the cue with itself), labels A-Z by
    # point. A group's colour is a preset carried by its cues (color_idx); the
    # squircle keeps one group per colour.

    def _jumpcue_groups(self) -> dict[int, list[dict]]:
        """Cues of the current JumpCUEs by graph index."""
        groups: dict[int, list[dict]] = {}
        for cue in self._jumpcue_graph_cues:
            groups.setdefault(int(cue.get("graph_idx", 0) or 0), []).append(
                {
                    "label": str(cue.get("label", "") or ""),
                    "comment": str(cue.get("comment", "") or ""),
                    "start": float(cue.get("start", 0.0) or 0.0),
                    "end": float(cue.get("end", 0.0) or 0.0),
                    "point": float(cue.get("point", 0.0) or 0.0),
                    "color_idx": int(cue.get("color_idx", 0) or 0),
                }
            )
        return groups

    def _unused_jumpcue_color(self, groups) -> int | None:
        used = {int(g[0].get("color_idx", 0)) for g in groups if g}
        return next((ci for ci in range(len(JUMPCUE_PAIR_COLORS)) if ci not in used), None)

    def _nearest_beat_index(self, t: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self._beatgrid, dtype=float) - float(t))))

    def _beat_at(self, idx: int) -> float:
        beats = np.asarray(self._beatgrid, dtype=float)
        return float(beats[int(np.clip(idx, 0, beats.size - 1))])

    def _set_jumpcue_groups(self, groups) -> bool:
        """Replace the JumpCUEs with ``groups`` (lists of cues; empty groups are
        dropped) and record the edit; cues keep their letters (slots). False
        (nothing changed) when two cues share a point."""
        groups = [sorted(g, key=lambda c: c["point"]) for g in groups if g]
        cues = sorted((c for g in groups for c in g), key=lambda c: c["point"])
        points = np.asarray([c["point"] for c in cues], dtype=float)
        if points.size > 1 and np.min(np.diff(points)) < 1e-3:
            QtWidgets.QMessageBox.warning(self, "JumpCUE", "Two JumpCUEs cannot share a point.")
            return False
        for group in groups:
            # one colour per group: its first cue's
            color = next((c["color_idx"] for c in group if c.get("color_idx") is not None), 0)
            for c in group:
                c["color_idx"] = int(color)
        for cue in cues:
            if not cue["comment"]:
                cue["comment"] = cue["label"]  # no name
        pairs = []
        for group in groups:
            if len(group) == 1:
                pairs.append({"forward": dict(group[0]), "backward": dict(group[0]), "lag_beats": 0.0,
                              "lag_sec": 0.0, "score": 1.0, "confidence": 1.0})
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = dict(group[i]), dict(group[j])
                    pairs.append(
                        {
                            "forward": a,
                            "backward": b,
                            "lag_beats": 0.0,
                            "lag_sec": abs(b["point"] - a["point"]),
                            "score": 1.0,
                            "confidence": 1.0,
                        }
                    )
        self.set_JumpCUE(pairs)
        self._emit_jumpcue_update()
        return True

    def _set_jumpcue_color(self, point: float, color_idx: int, *, groups=None) -> None:
        """Give the cue at ``point`` a colour; the colour is its group: it joins the
        group of that colour, or becomes one of its own."""
        groups = self._jumpcue_groups() if groups is None else groups
        cue = None
        for g in groups.values():
            for c in g:
                if abs(c["point"] - point) < 1e-3:
                    cue = c
            g[:] = [c for c in g if abs(c["point"] - point) >= 1e-3]
        if cue is None:
            return
        cue["color_idx"] = int(color_idx)
        target = next((k for k, g in groups.items() if g and int(g[0]["color_idx"]) == int(color_idx)), None)
        if target is None:
            groups[max(groups, default=-1) + 1] = [cue]
        else:
            groups[target].append(cue)
        self._set_jumpcue_groups(groups.values())

    def _shift_jumpcue_group(self, gidx: int, beats: int) -> None:
        """Move a group (points and passages) by ``beats`` beats."""
        groups = self._jumpcue_groups()
        if gidx not in groups or len(self._beatgrid) == 0:
            return
        n = len(self._beatgrid)
        moved = []
        for cue in groups[gidx]:
            k = self._nearest_beat_index(cue["point"]) + int(beats)
            if not 0 <= k < n:
                return
            moved.append(
                dict(
                    cue,
                    start=self._beat_at(self._nearest_beat_index(cue["start"]) + int(beats)),
                    end=self._beat_at(self._nearest_beat_index(cue["end"]) + int(beats)),
                    point=self._beat_at(k),
                )
            )
        groups[gidx] = moved
        self._set_jumpcue_groups(groups.values())

    def _can_insert_jumpcue_slot(self, letter: str) -> bool:
        """The cues from ``letter`` on can move one slot on: the last slot (Z) is free
        or before ``letter``."""
        last = slot_letter(JUMPCUE_SLOTS - 1)
        return letter < last and not any(str(c.get("label", "")) == last for c in self._jumpcue_graph_cues)

    def _can_close_jumpcue_slot(self, letter: str) -> bool:
        """The slot before ``letter`` exists and is empty."""
        if letter <= "A":
            return False
        prev = chr(ord(letter) - 1)
        return not any(str(c.get("label", "")) == prev for c in self._jumpcue_graph_cues)

    def _insert_jumpcue_slot(self, letter: str) -> None:
        self._shift_jumpcue_slots(letter, 1)

    def _shift_jumpcue_slots(self, letter: str, step: int) -> None:
        """Move the cue in slot ``letter`` and every cue in a later slot one slot on
        (``step`` 1, leaving ``letter`` empty) or back into the empty slot before it
        (``step`` -1). Names that are just the letter follow it."""
        ok = self._can_insert_jumpcue_slot(letter) if step > 0 else self._can_close_jumpcue_slot(letter)
        if not ok:
            return
        groups = self._jumpcue_groups()
        for g in groups.values():
            for c in g:
                if c["label"] >= letter:
                    nxt = chr(ord(c["label"]) + (1 if step > 0 else -1))
                    if c["comment"] == c["label"]:
                        c["comment"] = nxt
                    c["label"] = nxt
        self._set_jumpcue_groups(groups.values())

    def _remove_jumpcue_member(self, point: float) -> None:
        groups = self._jumpcue_groups()
        for g in groups.values():
            g[:] = [c for c in g if abs(c["point"] - point) >= 1e-3]
        self._set_jumpcue_groups(groups.values())

    def _new_jumpcue_group(self, letter: str) -> None:
        """A cue in slot ``letter`` at the beat nearest the playhead, in an unused
        colour (a group of its own); with every colour in use, in the preceding
        cue's colour (its group)."""
        if len(self._beatgrid) == 0:
            return
        p = self._beat_at(self._nearest_beat_index(self._current_time))
        color = self._unused_jumpcue_color(list(self._jumpcue_groups().values()))
        if color is None:
            before = [c for c in self._jumpcue_graph_cues if float(c.get("point", 0.0)) < p]
            color = int((before[-1] if before else self._jumpcue_graph_cues[0]).get("color_idx", 0))
        self._insert_jumpcue(p, color, letter)

    def _insert_jumpcue(self, point: float, color_idx: int, letter: str) -> None:
        groups = self._jumpcue_groups()
        groups[max(groups, default=-1) + 1] = [{"label": letter, "comment": letter, "start": point, "end": point,
                                                "point": point, "color_idx": int(color_idx)}]
        self._set_jumpcue_color(point, color_idx, groups=groups)

    def _emit_jumpcue_update(self) -> None:
        self.edit_log.push(self._log_element(jump_cues=self._JumpCUE))
        self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        self._broadcast_jumpcue()
        self._update_undo_redo_buttons()

    def _broadcast_jumpcue(self) -> None:
        if self.model is not None:
            try:
                self.model.features["jump_cues_np"] = self._jumpcues_to_np(self._JumpCUE)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid JumpCUE Label", str(exc))
            except Exception:
                pass
        self.bus.sig_jumpcue_updated.emit()

    @staticmethod
    def _jumpcues_to_np(jump_cues: list[dict]):
        if jump_cues is not None and not isinstance(jump_cues, list):
            return None
        return build_jump_cues_np(jump_cues or [])

    def apply_jc_update_from_external(self, *, jump_cues=None, record: bool = True, broadcast: bool = True) -> None:
        if jump_cues is None:
            return
        cues_copy = copy.deepcopy(jump_cues)
        self.set_JumpCUE(cues_copy)
        if record:
            self.edit_log.push(self._log_element(jump_cues=cues_copy))
            self._update_undo_redo_buttons()
            self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        if broadcast:
            self._broadcast_jumpcue()

    def _reanalyze_jumpCUE(self) -> None:
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        payload = {
            "path": self._current_track_path(),
            "beats_time_sec": np.asarray(self._beatgrid, dtype=float).copy(),
        }
        if self._segments is not None:
            payload["segments"] = np.asarray(self._segments, dtype=float).copy()
        payload["analyze_type"] = "jumpcue"
        self.bus.sig_segment_reanalyze_requested.emit(payload)

    def _log_element(self, **override) -> logElement:
        """Snapshot of every edited state for the undo log (``override`` replaces fields)."""
        fields = dict(
            beats_time=self._beatgrid,
            beat_segments=self._segments,
            key_segments=self._key_segments,
            jump_cues=self._JumpCUE,
            phrases=self._phrase_segments,
            cue_points=self._cue_points,
        )
        fields.update(override)
        return logElement(**fields)
