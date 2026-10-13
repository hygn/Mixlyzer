from __future__ import annotations

import copy

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ui.edit_panel.history import logElement
from ui.edit_panel.styles import WARNING_BUTTON_STYLESHEET
from utils.jump_cues import build_jump_cue_graph, build_jump_cues_np
from utils.jumpcue_colors import get_jumpcue_pair_color

class JumpCUEEditMixin:
    """JumpCUE section: jump arm test, group edits, JumpCUE reanalysis."""

    def _build_jumpcue_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """JumpCUE widgets on the Phrase/JumpCUE page; returns the edit buttons."""

        self.jc_reanalyze = QtWidgets.QToolButton()
        self.jc_reanalyze.setText("RA")
        self.jc_reanalyze.setToolTip("Recompute JumpCUE pairs using current beatgrid")

        self.jc_source_selection = QtWidgets.QComboBox()
        self.jc_source_selection.setToolTip("Select source JumpCUE")
        self.jc_source_selection.setFixedSize(self.BTN_WIDTH // 2, self.BTN_HEIGHT)
        self.jc_source_selection.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.jc_source_selection.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.jc_target_selection = QtWidgets.QComboBox()
        self.jc_target_selection.setToolTip("Select target JumpCUE")
        self.jc_target_selection.setFixedSize(self.BTN_WIDTH // 2, self.BTN_HEIGHT)
        self.jc_target_selection.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.jc_target_selection.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.jc_selection_wrap = QtWidgets.QWidget()
        self.jc_selection_wrap.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)
        jc_layout = QtWidgets.QHBoxLayout(self.jc_selection_wrap)
        jc_layout.setContentsMargins(0, 0, 0, 0)
        jc_layout.setSpacing(0)
        jc_layout.addWidget(self.jc_source_selection)
        jc_layout.addWidget(self.jc_target_selection)

        self.jc_jumptest = QtWidgets.QToolButton()
        self.jc_jumptest.setText("ARM")
        self.jc_jumptest.setToolTip("Arm jump: transition from the selected source cue to the selected target cue")

        # Group edits act on the group of the selected source cue.
        self.jc_group_back = QtWidgets.QToolButton()
        self.jc_group_back.setText("Grp-1")
        self.jc_group_back.setToolTip("Move every cue of the source cue's group one beat earlier")

        self.jc_group_fwd = QtWidgets.QToolButton()
        self.jc_group_fwd.setText("Grp+1")
        self.jc_group_fwd.setToolTip("Move every cue of the source cue's group one beat later")

        self.jc_group_delete = QtWidgets.QToolButton()
        self.jc_group_delete.setText("GrpDel")
        self.jc_group_delete.setToolTip("Delete the source cue's group")

        self.jc_member_add = QtWidgets.QToolButton()
        self.jc_member_add.setText("Add")
        self.jc_member_add.setToolTip("Add a cue at the beat nearest the playhead to the source cue's group")

        self.jc_member_remove = QtWidgets.QToolButton()
        self.jc_member_remove.setText("Remove")
        self.jc_member_remove.setToolTip("Remove the source cue from its group")

        self.jc_group_new = QtWidgets.QToolButton()
        self.jc_group_new.setText("NewGrp")
        self.jc_group_new.setToolTip("Create a new group with one cue at the beat nearest the playhead")

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        grid.addWidget(self.jc_reanalyze, 0, 0, _AL)
        grid.addWidget(self.jc_selection_wrap, 1, 0, _AL)
        grid.addWidget(self.jc_jumptest, 2, 0, _AL)
        grid.addWidget(self.jc_group_back, 0, 1, _AL)
        grid.addWidget(self.jc_group_fwd, 1, 1, _AL)
        grid.addWidget(self.jc_group_delete, 2, 1, _AL)
        grid.addWidget(self.jc_member_add, 0, 2, _AL)
        grid.addWidget(self.jc_member_remove, 1, 2, _AL)
        grid.addWidget(self.jc_group_new, 2, 2, _AL)

        self.jc_jumptest.clicked.connect(self._arm_next_jumpCUE_jump)
        self.jc_reanalyze.clicked.connect(self._reanalyze_jumpCUE)
        self.jc_group_back.clicked.connect(lambda: self._shift_jumpcue_group(-1))
        self.jc_group_fwd.clicked.connect(lambda: self._shift_jumpcue_group(1))
        self.jc_group_delete.clicked.connect(self._delete_jumpcue_group)
        self.jc_member_add.clicked.connect(self._add_jumpcue_member)
        self.jc_member_remove.clicked.connect(self._remove_jumpcue_member)
        self.jc_group_new.clicked.connect(self._new_jumpcue_group)
        self.jc_source_selection.currentIndexChanged.connect(self._jc_source_selection_changed)
        self.jc_target_selection.currentIndexChanged.connect(self._jc_target_selection_changed)

        return [
            self.jc_reanalyze,
            self.jc_jumptest,
            self.jc_group_back,
            self.jc_group_fwd,
            self.jc_group_delete,
            self.jc_member_add,
            self.jc_member_remove,
            self.jc_group_new,
        ]

    def set_JumpCUE(self, JumpCUE) -> None:
        if JumpCUE == None:
            JumpCUE = []
        self._JumpCUE = JumpCUE
        self._jumpcue_graph_cues, _links = build_jump_cue_graph(self._JumpCUE)
        prev_source_id = self._current_jumpcue_source_id()
        prev_target_id = self._current_jumpcue_target_id()
        self._populate_jumpcue_combo(self.jc_source_selection, self._jumpcue_graph_cues, preferred_cue_id=prev_source_id)
        self._refresh_jumpcue_target_combo(preferred_target_id=prev_target_id)
        self._reset_jumpcue_arm_state()

    def _arm_next_jumpCUE_jump(self):
        if not self._jump_armed:
            payload = self._current_jumpcue_payload()
            if isinstance(payload, dict):
                self.bus.sig_jump_arm.emit(payload)
                self.jc_jumptest.setText("DISARM")
                self._jump_armed = True
        else:
            self._reset_jumpcue_arm_state()

    def _jc_source_selection_changed(self):
        self._refresh_jumpcue_target_combo(preferred_target_id=self._current_jumpcue_target_id())
        self._apply_jumpcue_combo_style()
        self._reset_jumpcue_arm_state()

    def _jc_target_selection_changed(self):
        self._apply_jumpcue_combo_style()
        self._reset_jumpcue_arm_state()

    @staticmethod
    def _jumpcue_color_for_component(color_or_index) -> tuple[int, int, int]:
        if isinstance(color_or_index, tuple) and len(color_or_index) == 3:
            try:
                return (int(color_or_index[0]), int(color_or_index[1]), int(color_or_index[2]))
            except Exception:
                pass
        return get_jumpcue_pair_color(int(color_or_index or 0))

    @classmethod
    def _jumpcue_item_brushes(cls, color) -> tuple[QtGui.QBrush, QtGui.QBrush]:
        sr, sg, sb = cls._jumpcue_color_for_component(color)
        bg = QtGui.QBrush(QtGui.QColor(sr, sg, sb, 96))
        luminance = (0.299 * sr) + (0.587 * sg) + (0.114 * sb)
        fg = QtGui.QBrush(QtGui.QColor(0, 0, 0) if luminance >= 140 else QtGui.QColor(255, 255, 255))
        return bg, fg

    def _apply_jumpcue_combo_style(self) -> None:
        self._apply_jumpcue_combo_style_for(self.jc_source_selection)
        self._apply_jumpcue_combo_style_for(self.jc_target_selection)

    def _apply_jumpcue_combo_style_for(self, combo: QtWidgets.QComboBox) -> None:
        payload = self._current_jumpcue_cue(combo)
        if not isinstance(payload, dict):
            self._reset_jumpcue_combo_palette(combo)
            return
        color = payload.get("color", payload.get("color_index", payload.get("component_index", 0)))
        sr, sg, sb = self._jumpcue_color_for_component(color)
        luminance = (0.299 * sr) + (0.587 * sg) + (0.114 * sb)
        text_color = "#000000" if luminance >= 140 else "#FFFFFF"
        base_color = QtGui.QColor(sr, sg, sb, 96)
        solid_color = QtGui.QColor(sr, sg, sb)
        fg_qcolor = QtGui.QColor(text_color)
        palette = combo.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Base, base_color)
        palette.setColor(QtGui.QPalette.ColorRole.Button, base_color)
        palette.setColor(QtGui.QPalette.ColorRole.Text, fg_qcolor)
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, fg_qcolor)
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, solid_color.darker(110))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, fg_qcolor)
        combo.setPalette(palette)
        combo.setStyleSheet("")
        view = combo.view()
        if view is not None:
            view_palette = view.palette()
            view_palette.setColor(QtGui.QPalette.ColorRole.Base, base_color)
            view_palette.setColor(QtGui.QPalette.ColorRole.Text, fg_qcolor)
            view_palette.setColor(QtGui.QPalette.ColorRole.Highlight, solid_color.darker(110))
            view_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, fg_qcolor)
            view.setPalette(view_palette)
            view.setStyleSheet("")

    def _reset_jumpcue_combo_palette(self, combo: QtWidgets.QComboBox) -> None:
        standard_palette = self.style().standardPalette()
        combo.setPalette(standard_palette)
        combo.setStyleSheet("")
        view = combo.view()
        if view is not None:
            view.setPalette(standard_palette)
            view.setStyleSheet("")

    @staticmethod
    def _find_jumpcue_combo_index(combo: QtWidgets.QComboBox, cue_id: int | None) -> int:
        if cue_id is None:
            return -1
        for idx in range(combo.count()):
            payload = combo.itemData(idx)
            if isinstance(payload, dict) and int(payload.get("id", -1)) == int(cue_id):
                return idx
        return -1

    def _populate_jumpcue_combo(
        self,
        combo: QtWidgets.QComboBox,
        cues: list[dict],
        *,
        preferred_cue_id: int | None = None,
        disabled_cue_id: int | None = None,
    ) -> None:
        with QtCore.QSignalBlocker(combo):
            combo.clear()
            for cue in cues:
                label = str(cue.get("label", "") or "").strip()
                if not label:
                    continue
                combo.addItem(label, cue)
                idx = combo.count() - 1
                color = cue.get("color", cue.get("color_index", cue.get("component_index", 0)))
                bg_brush, fg_brush = self._jumpcue_item_brushes(color)
                combo.setItemData(idx, bg_brush, QtCore.Qt.BackgroundRole)
                combo.setItemData(idx, fg_brush, QtCore.Qt.ForegroundRole)
                if disabled_cue_id is not None and int(cue.get("id", -1)) == int(disabled_cue_id):
                    model = combo.model()
                    item = model.item(idx) if hasattr(model, "item") else None
                    if item is not None:
                        item.setEnabled(False)
            if combo.count() > 0:
                cue_idx = self._find_jumpcue_combo_index(combo, preferred_cue_id)
                if cue_idx < 0 or not self._is_jumpcue_combo_index_enabled(combo, cue_idx):
                    cue_idx = self._first_enabled_jumpcue_combo_index(combo)
                combo.setCurrentIndex(cue_idx if cue_idx >= 0 else 0)

    @staticmethod
    def _is_jumpcue_combo_index_enabled(combo: QtWidgets.QComboBox, idx: int) -> bool:
        if idx < 0 or idx >= combo.count():
            return False
        model = combo.model()
        item = model.item(idx) if hasattr(model, "item") else None
        if item is None:
            return True
        return bool(item.isEnabled())

    @classmethod
    def _first_enabled_jumpcue_combo_index(cls, combo: QtWidgets.QComboBox) -> int:
        for idx in range(combo.count()):
            if cls._is_jumpcue_combo_index_enabled(combo, idx):
                return idx
        return -1

    def _refresh_jumpcue_target_combo(self, *, preferred_target_id: int | None = None) -> None:
        self._populate_jumpcue_combo(
            self.jc_target_selection,
            self._jumpcue_graph_cues,
            preferred_cue_id=preferred_target_id,
            disabled_cue_id=self._current_jumpcue_source_id(),
        )
        self._apply_jumpcue_combo_style()

    def _current_jumpcue_source_id(self) -> int | None:
        return self._current_jumpcue_cue_id(self.jc_source_selection)

    def _current_jumpcue_target_id(self) -> int | None:
        return self._current_jumpcue_cue_id(self.jc_target_selection)

    @staticmethod
    def _current_jumpcue_cue(combo: QtWidgets.QComboBox) -> dict | None:
        payload = combo.currentData()
        if isinstance(payload, dict):
            return payload
        return None

    def _current_jumpcue_cue_id(self, combo: QtWidgets.QComboBox) -> int | None:
        payload = self._current_jumpcue_cue(combo)
        if isinstance(payload, dict):
            return int(payload.get("id", -1))
        return None

    def _current_jumpcue_payload(self) -> dict | None:
        source = self._current_jumpcue_cue(self.jc_source_selection)
        target = self._current_jumpcue_cue(self.jc_target_selection)
        if not isinstance(source, dict) or not isinstance(target, dict):
            return None
        source_id = int(source.get("id", -1))
        target_id = int(target.get("id", -1))
        if source_id < 0 or target_id < 0:
            return None
        source_graph_idx = int(source.get("graph_idx", source.get("component_index", 0)) or 0)
        color = source.get("color", source.get("color_index", source.get("component_index", 0)))
        return {
            "kind": "F",
            "source_id": source_id,
            "target_id": target_id,
            "source_label": str(source.get("label", "") or ""),
            "target_label": str(target.get("label", "") or ""),
            "source": dict(source),
            "target": dict(target),
            "source_point": float(source.get("point", source.get("start", 0.0)) or 0.0),
            "target_point": float(target.get("point", target.get("start", 0.0)) or 0.0),
            "component_index": source_graph_idx,
            "graph_idx": source_graph_idx,
            "color_index": source_graph_idx,
            "color": color,
            "lag_beats": 0.0,
            "lag_sec": abs(float(target.get("point", 0.0) or 0.0) - float(source.get("point", 0.0) or 0.0)),
            "score": 1.0,
            "confidence": 1.0,
        }

    def _reset_jumpcue_arm_state(self) -> None:
        self.jc_jumptest.setText("ARM")
        self.bus.sig_jump_disarm.emit()
        self._jump_armed = False


    # ---- JumpCUE group edits ----
    # Edits rebuild the pair list from groups of cues: every pair of members of
    # a group (a one-cue group as a pair of the cue with itself), labels A-Z by
    # point (the graph the combos show).

    def _jumpcue_groups(self) -> dict[int, list[dict]]:
        """Cues of the current JumpCUEs by graph index (the combos' graph_idx)."""
        cues, _links = build_jump_cue_graph(self._JumpCUE)
        groups: dict[int, list[dict]] = {}
        for cue in cues:
            groups.setdefault(int(cue.get("graph_idx", 0) or 0), []).append(
                {
                    "label": str(cue.get("label", "") or ""),
                    "comment": str(cue.get("comment", "") or ""),
                    "start": float(cue.get("start", 0.0) or 0.0),
                    "end": float(cue.get("end", 0.0) or 0.0),
                    "point": float(cue.get("point", 0.0) or 0.0),
                }
            )
        return groups

    def _source_jumpcue_group(self) -> tuple[dict | None, dict[int, list[dict]], int | None]:
        """The selected source cue, all groups, and the source cue's graph index."""
        source = self._current_jumpcue_cue(self.jc_source_selection)
        if not isinstance(source, dict):
            return None, {}, None
        return source, self._jumpcue_groups(), int(source.get("graph_idx", 0) or 0)

    def _nearest_beat_index(self, t: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self._beatgrid, dtype=float) - float(t))))

    def _beat_at(self, idx: int) -> float:
        beats = np.asarray(self._beatgrid, dtype=float)
        return float(beats[int(np.clip(idx, 0, beats.size - 1))])

    def _set_jumpcue_groups(self, groups, *, select_point: float | None = None) -> bool:
        """Replace the JumpCUEs with ``groups`` (lists of cues; empty groups are
        dropped) and record the edit. False (nothing changed) when two cues
        share a point or there are more cues than labels A-Z."""
        groups = [sorted(g, key=lambda c: c["point"]) for g in groups if g]
        cues = sorted((c for g in groups for c in g), key=lambda c: c["point"])
        points = np.asarray([c["point"] for c in cues], dtype=float)
        if points.size > 1 and np.min(np.diff(points)) < 1e-3:
            QtWidgets.QMessageBox.warning(self, "JumpCUE", "Two JumpCUEs cannot share a point.")
            return False
        if len(cues) > 26:
            QtWidgets.QMessageBox.warning(self, "JumpCUE", "JumpCUEs are limited to 26 cues (A-Z).")
            return False
        for idx, cue in enumerate(cues):
            label = chr(ord("A") + idx)
            if cue["comment"] in ("", cue["label"]):
                cue["comment"] = label
            cue["label"] = label
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
        if select_point is not None:
            combo = self.jc_source_selection
            for idx in range(combo.count()):
                payload = combo.itemData(idx)
                if isinstance(payload, dict) and abs(float(payload.get("point", -1.0)) - select_point) < 1e-3:
                    combo.setCurrentIndex(idx)
                    break
        self._emit_jumpcue_update()
        return True

    def _shift_jumpcue_group(self, beats: int) -> None:
        """Move the source cue's group (points and passages) by ``beats`` beats."""
        source, groups, gidx = self._source_jumpcue_group()
        if source is None or gidx not in groups or len(self._beatgrid) == 0:
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
        select = self._beat_at(self._nearest_beat_index(float(source.get("point", 0.0))) + int(beats))
        self._set_jumpcue_groups(groups.values(), select_point=select)

    def _add_jumpcue_member(self) -> None:
        """Add a cue at the beat nearest the playhead to the source cue's group,
        its passage placed like the source cue's (same beats before and after)."""
        source, groups, gidx = self._source_jumpcue_group()
        if source is None or gidx not in groups or len(self._beatgrid) == 0:
            return
        k = self._nearest_beat_index(self._current_time)
        d = k - self._nearest_beat_index(float(source.get("point", 0.0)))
        cue = {
            "label": "",
            "comment": "",
            "start": self._beat_at(self._nearest_beat_index(float(source.get("start", 0.0))) + d),
            "end": self._beat_at(self._nearest_beat_index(float(source.get("end", 0.0))) + d),
            "point": self._beat_at(k),
        }
        groups[gidx] = groups[gidx] + [cue]
        self._set_jumpcue_groups(groups.values(), select_point=cue["point"])

    def _remove_jumpcue_member(self) -> None:
        source, groups, gidx = self._source_jumpcue_group()
        if source is None or gidx not in groups:
            return
        point = float(source.get("point", 0.0))
        groups[gidx] = [c for c in groups[gidx] if abs(c["point"] - point) >= 1e-3]
        self._set_jumpcue_groups(groups.values())

    def _delete_jumpcue_group(self) -> None:
        source, groups, gidx = self._source_jumpcue_group()
        if source is None or gidx not in groups:
            return
        del groups[gidx]
        self._set_jumpcue_groups(groups.values())

    def _new_jumpcue_group(self) -> None:
        """New group of one cue at the beat nearest the playhead."""
        if len(self._beatgrid) == 0:
            return
        p = self._beat_at(self._nearest_beat_index(self._current_time))
        new = [{"label": "", "comment": "", "start": p, "end": p, "point": p}]
        self._set_jumpcue_groups(list(self._jumpcue_groups().values()) + [new], select_point=p)

    def _emit_jumpcue_update(self) -> None:
        self.edit_log.push(
            logElement(
                beats_time=self._beatgrid,
                beat_segments=self._segments,
                key_segments=self._key_segments,
                jump_cues=self._JumpCUE,
                phrases=self._phrase_segments,
            )
        )
        self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        self._broadcast_jumpcue()
        self._update_undo_redo_buttons()

    def _broadcast_jumpcue(self) -> None:
        if self.model is not None:
            try:
                self.model.features["jump_cues_extracted"] = copy.deepcopy(self._JumpCUE)
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
            self.edit_log.push(
                logElement(
                    beats_time=self._beatgrid,
                    beat_segments=self._segments,
                    key_segments=self._key_segments,
                    jump_cues=cues_copy,
                    phrases=self._phrase_segments,
                )
            )
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
