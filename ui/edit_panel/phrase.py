from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ui.edit_panel.history import logElement
from ui.edit_panel.icons import icon
from ui.theme import current_theme
from utils.phrases import (
    PHRASE_LABELS,
    UNASSIGNED,
    build_phrase_segments_np,
    fill_unassigned,
    normalize_phrase_segments,
    number_phrase_labels,
    phrase_color,
    phrase_display_name,
)

# Label buttons (every label), applied to the current phrase on click like the
# time signatures.
_BUTTON_TEXT = {"BREAK_CHORUS": "BRK CHO", "INTERLUDE": "INTERLD"}


class PhraseEditMixin:
    """Phrase section, edited by segment like the key: the phrase under the playhead
    is the one edited (its label, its boundaries, split / merge). The track is
    covered end to end; stretches without a phrase are UNASSIGNED."""

    def _build_phrase_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """Phrase widgets on the Phrase page; returns the edit buttons."""

        def _btn(icon_name: str, tip: str, *, checkable: bool = False, wide: bool = False) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setIcon(icon(icon_name))
            btn.setIconSize(QtCore.QSize(36, 18) if wide else QtCore.QSize(19, 19))
            btn.setToolTip(tip)
            btn.setCheckable(checkable)
            return btn

        # boundaries, by a beat or a bar
        self.phrase_start_back = _btn("key_start_back", "Move the current phrase's start earlier")
        self.phrase_start_fwd = _btn("key_start_fwd", "Move the current phrase's start later")
        self.phrase_end_back = _btn("key_end_back", "Move the current phrase's end earlier")
        self.phrase_end_fwd = _btn("key_end_fwd", "Move the current phrase's end later")
        self.phrase_step_beat = _btn("key_step_beat", "Boundaries move by one beat", checkable=True, wide=True)
        self.phrase_step_bar = _btn("key_step_bar", "Boundaries move by one bar (to the neighbouring downbeat)", checkable=True, wide=True)
        self.phrase_step_beat.setChecked(True)
        self._phrase_step_group = QtWidgets.QButtonGroup(self)
        self._phrase_step_group.setExclusive(True)
        self._phrase_step_group.addButton(self.phrase_step_beat)
        self._phrase_step_group.addButton(self.phrase_step_bar)

        # segments
        self.phrase_split = _btn("beatgrid_split", "Split the current phrase at the beat nearest the playhead")
        self.phrase_merge_prev = _btn("beatgrid_merge_prev", "Merge the previous phrase into the current one (keeps the current label)")
        self.phrase_merge_next = _btn("beatgrid_merge_next", "Merge the next phrase into the current one (keeps the current label)")
        self.phrase_unassign = _btn("phrase_clear", "Unassign the current phrase (UNASSIGNED)", wide=True)

        # labels: applied to the current phrase on click; its own is filled in its colour
        label_font = QtGui.QFont(self.font())
        label_font.setPointSize(8)
        label_font.setBold(True)
        self.phrase_label_buttons: dict[str, QtWidgets.QToolButton] = {}
        for label in PHRASE_LABELS:
            b = QtWidgets.QToolButton()
            b.setFont(label_font)
            b.clicked.connect(lambda _=False, label=label: self._assign_phrase_label(label))
            self.phrase_label_buttons[label] = b

        self.phrase_status = QtWidgets.QLabel("-- | Seg -- | +0 beats | Left: 00:00.00")
        self.phrase_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        # Boundaries on top (start pair, end pair, step), the segment structure
        # under them, the beat counter in the last row; the labels stand on their
        # own after a rule, as the time signatures on the beatgrid page.
        cells = {
            (0, 0): self.phrase_start_back, (0, 1): self.phrase_start_fwd, (0, 2): self.phrase_end_back,
            (0, 3): self.phrase_end_fwd, (0, 4): self.phrase_step_beat,
            (1, 0): self.phrase_split, (1, 1): self.phrase_merge_prev, (1, 2): self.phrase_merge_next,
            (1, 4): self.phrase_step_bar,
            (1, 3): self.phrase_unassign,
        }
        for (r, c), w in cells.items():
            grid.addWidget(w, r, c, _AL)
        label_rule = QtWidgets.QFrame()
        label_rule.setFixedWidth(1)
        label_rule.setStyleSheet(f"background: {current_theme().border};")
        label_rule_cell = QtWidgets.QWidget()
        rule_lay = QtWidgets.QHBoxLayout(label_rule_cell)
        rule_lay.setContentsMargins(4, 2, 4, 2)
        rule_lay.addWidget(label_rule)
        grid.addWidget(label_rule_cell, 0, 5, 3, 1)
        label_cells = list(self.phrase_label_buttons.values())
        for i, w in enumerate(label_cells):
            grid.addWidget(w, i // 3, 6 + i % 3, _AL)
        grid.addWidget(self.phrase_status, 3, 0, 1, 9, _AL)

        self.phrase_start_back.clicked.connect(lambda: self._nudge_phrase_boundary("start", -1))
        self.phrase_start_fwd.clicked.connect(lambda: self._nudge_phrase_boundary("start", 1))
        self.phrase_end_back.clicked.connect(lambda: self._nudge_phrase_boundary("end", -1))
        self.phrase_end_fwd.clicked.connect(lambda: self._nudge_phrase_boundary("end", 1))
        self.phrase_split.clicked.connect(self._split_phrase)
        self.phrase_merge_prev.clicked.connect(lambda: self._merge_phrase(-1))
        self.phrase_merge_next.clicked.connect(lambda: self._merge_phrase(1))
        self.phrase_unassign.clicked.connect(lambda: self._assign_phrase_label(UNASSIGNED))

        self._phrase_cur = -1
        self._phrase_lit: str | None = ""
        self.refresh_phrase_label_names()
        return list(cells.values()) + label_cells

    def refresh_phrase_label_names(self) -> None:
        """Show the labels by their current names (aliases, set in the settings)."""
        for label, btn in self.phrase_label_buttons.items():
            btn.setText(_button_text(label))
            btn.setToolTip(f"Label the current phrase {phrase_display_name(label)}")
        lit = self._phrase_lit
        self._phrase_lit = ""
        self._refresh_phrase_label_buttons(lit)
        self._phrase_cur = -1
        self._update_phrase_status()

    # ---- data in / out ----

    def set_phrase_segments(self, phrases, *, initialize: bool = True) -> None:
        """Load phrases (list of {start, end, label}, covering the track: read with
        extract_phrase_segments). Default does not re-broadcast."""
        self._phrase_segments = normalize_phrase_segments(phrases or [])
        self._phrase_cur = -1
        self._update_phrase_status()
        if not initialize:
            self._broadcast_phrase()

    def _phrase_rows(self, phrases) -> list[dict]:
        """The phrases covering the track, the gaps UNASSIGNED."""
        return fill_unassigned(phrases, self.duration if self.duration > 0.0 else None)

    def _broadcast_phrase(self, *, record: bool = False) -> None:
        if record:
            self.edit_log.push(
                logElement(
                    beats_time=self._beatgrid,
                    beat_segments=self._segments,
                    key_segments=self._key_segments,
                    jump_cues=self._JumpCUE,
                    phrases=self._phrase_segments,
                    cue_points=self._cue_points,
                )
            )
            self._update_undo_redo_buttons()
            self._mark_unsaved(True)
        if self.model is not None:
            try:
                self.model.features["phrase_segments_np"] = build_phrase_segments_np(self._phrase_segments)
            except Exception:
                pass
        self._phrase_cur = -1
        self._update_phrase_status()
        self.bus.sig_phrase_segments_updated.emit()

    def _set_phrases_edited(self, rows: list[dict]) -> None:
        self._phrase_segments = self._phrase_rows(rows)
        self._broadcast_phrase(record=True)

    # ---- the current phrase ----

    def _phrase_index_at(self, t: float) -> int:
        rows = self._phrase_segments
        if not rows:
            return -1
        for i, row in enumerate(rows):
            if row["start"] <= t < row["end"]:
                return i
        return 0 if t < rows[0]["start"] else len(rows) - 1

    def _current_phrase_row(self) -> int | None:
        idx = self._phrase_index_at(float(self._current_time))
        return idx if idx >= 0 else None

    def _update_phrase_status(self) -> None:
        """Follow the playhead: status line and the lit label show the phrase under it."""
        if not hasattr(self, "phrase_status"):
            return
        t = float(self._current_time)
        idx = self._phrase_index_at(t)
        if idx < 0:
            self._phrase_cur = -1
            self.phrase_status.setText("-- | Seg -- | +0 beats | Left: 00:00.00")
            self._refresh_phrase_label_buttons(None)
            return
        row = self._phrase_segments[idx]
        left = max(0.0, float(row["end"]) - t)
        m = int(left // 60)
        name = number_phrase_labels(self._phrase_segments)[idx]
        self.phrase_status.setText(
            f"{name} | Seg {idx + 1} | +{self._phrase_beats_elapsed(t, float(row['start']))} beats"
            f" | Left: {m:02d}:{left - 60 * m:05.2f}"
        )
        if idx != self._phrase_cur:
            self._phrase_cur = idx
            self._refresh_phrase_label_buttons(row["label"])

    def _phrase_beats_elapsed(self, t: float, start: float) -> int:
        """Beats from the phrase's first beat to the playhead, counting both."""
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size == 0:
            return 0
        # Snap the start to the nearest beat by INDEX: stored phrase starts are
        # float32 (and may sit off-grid), so matching by time can miss the beat.
        ref_idx = int(np.argmin(np.abs(beats - start)))
        cur_idx = int(np.searchsorted(beats, t + 1e-4, side="right")) - 1
        cur_idx = min(max(cur_idx, ref_idx), beats.size - 1)
        return cur_idx - ref_idx + 1

    def _refresh_phrase_label_buttons(self, label: str | None) -> None:
        """The current phrase's label button is filled in the label's colour (an
        UNASSIGNED phrase lights CLEAR); the others carry their colour as an underline."""
        if label == self._phrase_lit:
            return
        self._phrase_lit = label
        for name, btn in self.phrase_label_buttons.items():
            rgb = phrase_color(name)
            if name == label:
                btn.setStyleSheet(_lit_style(rgb))
            else:
                btn.setStyleSheet("QToolButton {{ border-bottom: 3px solid rgb({}, {}, {}); padding: 0px; }}".format(*rgb))
        self.phrase_unassign.setStyleSheet(_lit_style(phrase_color(UNASSIGNED)) if label == UNASSIGNED else "")

    # ---- edits ----

    def _assign_phrase_label(self, label: str) -> None:
        idx = self._current_phrase_row()
        if idx is None or self._phrase_segments[idx]["label"] == label:
            return
        rows = [dict(row) for row in self._phrase_segments]
        rows[idx]["label"] = label
        self._set_phrases_edited(rows)

    def _nudge_phrase_boundary(self, which: str, direction: int) -> None:
        idx = self._current_phrase_row()
        if idx is None:
            return
        rows = [dict(row) for row in self._phrase_segments]
        by_bar = self.phrase_step_bar.isChecked()
        if which == "start":
            if idx == 0:
                return  # the track start does not move
            new = self._step_on_grid(float(rows[idx]["start"]), direction, by_bar)
            if new is None or not (rows[idx - 1]["start"] < new < rows[idx]["end"]):
                return
            rows[idx - 1]["end"] = new
            rows[idx]["start"] = new
        else:
            if idx == len(rows) - 1:
                return  # the track end does not move
            new = self._step_on_grid(float(rows[idx]["end"]), direction, by_bar)
            if new is None or not (rows[idx]["start"] < new < rows[idx + 1]["end"]):
                return
            rows[idx]["end"] = new
            rows[idx + 1]["start"] = new
        self._set_phrases_edited(rows)

    def _split_phrase(self) -> None:
        idx = self._current_phrase_row()
        if idx is None:
            return
        t = float(self._current_time)
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size:
            t = float(beats[int(np.argmin(np.abs(beats - t)))])
        rows = [dict(row) for row in self._phrase_segments]
        if not (rows[idx]["start"] + 1e-3 < t < rows[idx]["end"] - 1e-3):
            return
        second = dict(rows[idx], start=t)
        rows[idx]["end"] = t
        rows.insert(idx + 1, second)
        self._set_phrases_edited(rows)

    def _merge_phrase(self, direction: int) -> None:
        idx = self._current_phrase_row()
        if idx is None:
            return
        rows = [dict(row) for row in self._phrase_segments]
        other = idx + direction
        if not 0 <= other < len(rows):
            return
        if direction < 0:
            rows[idx]["start"] = rows[other]["start"]
        else:
            rows[idx]["end"] = rows[other]["end"]
        del rows[other]
        self._set_phrases_edited(rows)


def _button_text(label: str) -> str:
    """A label button's text: the alias, else the label (shortened to fit)."""
    name = phrase_display_name(label)
    return _BUTTON_TEXT.get(label, label) if name == label else name


def _lit_style(rgb: tuple[int, int, int]) -> str:
    bg = "rgb({}, {}, {})".format(*rgb)
    fg = "#000000" if _luma(rgb) > 140 else "#ffffff"
    return (
        f"QToolButton {{ background-color: {bg}; color: {fg}; border-color: {bg}; padding: 0px; }}"
        f"QToolButton:hover {{ border-color: {fg}; }}"
    )


def _luma(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b
