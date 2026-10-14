from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from ui.edit_panel.history import logElement
from ui.edit_panel.styles import WARNING_BUTTON_STYLESHEET
from utils.phrases import (
    PHRASE_LABELS,
    assign_phrase_to_selection,
    build_phrase_segments_np,
    clear_phrase_in_selection,
    extract_phrase_segments,
    number_phrase_labels,
)

class PhraseEditMixin:
    """Phrase section: assign / clear phrase labels on the selection."""

    def _build_phrase_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """Phrase widgets on the Phrase/JumpCUE page; returns the edit buttons."""

        self.phrase_assign_combo = QtWidgets.QComboBox()
        self.phrase_assign_combo.setEditable(True)
        self.phrase_assign_combo.addItems(PHRASE_LABELS)
        self.phrase_assign_combo.setToolTip("Phrase label to assign (type for a custom label)")
        self.phrase_assign_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.phrase_sel_start = QtWidgets.QToolButton()
        self.phrase_sel_start.setText("SelST")
        self.phrase_sel_start.setToolTip("Set selection start to nearest beat at playhead")

        self.phrase_sel_end = QtWidgets.QToolButton()
        self.phrase_sel_end.setText("SelED")
        self.phrase_sel_end.setToolTip("Set selection end to nearest beat at playhead")

        self.phrase_deselect = QtWidgets.QToolButton()
        self.phrase_deselect.setText("DeSel")
        self.phrase_deselect.setToolTip("Clear selection")

        self.phrase_sel_all = QtWidgets.QToolButton()
        self.phrase_sel_all.setText("SelAL")
        self.phrase_sel_all.setToolTip("Select entire track")

        self.phrase_assign_button = QtWidgets.QToolButton()
        self.phrase_assign_button.setText("Assign")
        self.phrase_assign_button.setToolTip("Assign the chosen phrase label to the current selection")

        self.phrase_clear_button = QtWidgets.QToolButton()
        self.phrase_clear_button.setText("Clear")
        self.phrase_clear_button.setToolTip("Remove phrase labels within the current selection")

        self.phrase_status = QtWidgets.QLabel("-- | +0 beats")
        self.phrase_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        grid.addWidget(self.phrase_assign_combo, 0, 0, _AL)
        grid.addWidget(self.phrase_assign_button, 0, 1, _AL)
        grid.addWidget(self.phrase_clear_button, 0, 2, _AL)
        grid.addWidget(self.phrase_sel_start, 1, 0, _AL)
        grid.addWidget(self.phrase_sel_end, 1, 1, _AL)
        grid.addWidget(self.phrase_deselect, 2, 0, _AL)
        grid.addWidget(self.phrase_sel_all, 2, 1, _AL)
        grid.addWidget(self.phrase_status, 3, 0, 1, 4, _AL)

        # Phrase page (selection is shared with the Key page handlers)
        self.phrase_sel_start.clicked.connect(self._key_selection_set_start)
        self.phrase_sel_end.clicked.connect(self._key_selection_set_end)
        self.phrase_deselect.clicked.connect(self._key_selection_clear)
        self.phrase_sel_all.clicked.connect(self._key_selection_select_all)
        self.phrase_assign_button.clicked.connect(self._assign_phrase_selection)
        self.phrase_clear_button.clicked.connect(self._clear_phrase_selection)

        return [
            self.phrase_assign_combo,
            self.phrase_sel_start,
            self.phrase_sel_end,
            self.phrase_deselect,
            self.phrase_sel_all,
            self.phrase_assign_button,
            self.phrase_clear_button,
        ]

    def set_phrase_segments(self, phrases, *, initialize: bool = True) -> None:
        """Load phrase segments (list of {start, end, label}). Default does not re-broadcast."""
        if phrases is None:
            self._phrase_segments = []
        elif isinstance(phrases, list):
            self._phrase_segments = [dict(p) for p in phrases]
        else:
            self._phrase_segments = extract_phrase_segments({"phrase_segments_np": phrases})
        self._update_phrase_status()
        if not initialize:
            self._broadcast_phrase()

    # Phrase editing (selection -> assign label), mirrors the Key flow.
    def _assign_phrase_selection(self) -> None:
        bounds = self._selection_bounds()
        if bounds is None:
            return
        start, end = bounds
        label = self.phrase_assign_combo.currentText().strip()
        if not label:
            return
        try:
            self._phrase_segments = assign_phrase_to_selection(self._phrase_segments, start, end, label)
        except ValueError:
            return
        self._broadcast_phrase(record=True)

    def _clear_phrase_selection(self) -> None:
        bounds = self._selection_bounds()
        if bounds is None:
            return
        start, end = bounds
        self._phrase_segments = clear_phrase_in_selection(self._phrase_segments, start, end)
        self._broadcast_phrase(record=True)

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
            self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        if self.model is not None:
            try:
                phrase_segments_np = build_phrase_segments_np(self._phrase_segments)
                self.model.features["phrase_segments_np"] = phrase_segments_np
            except Exception:
                pass
        self._update_phrase_status()
        self.bus.sig_phrase_segments_updated.emit()

    def _update_phrase_status(self) -> None:
        t = float(self._current_time)
        label = self._phrase_label_at(t)
        beats = self._phrase_beats_elapsed(t)
        self.phrase_status.setText(f"{label or '--'} | +{beats} beats")

    def _phrase_beat_reference(self, t: float) -> float:
        """Reference time the phrase beat counter is measured from.

        - inside an assigned phrase: that phrase's start
        - otherwise, if an earlier phrase exists: the previous phrase's start
        - nothing assigned before t: the first beat (track start)
        """
        rows = sorted(self._phrase_segments, key=lambda r: float(r.get("start", 0.0)))
        for row in rows:
            if float(row.get("start", 0.0)) <= t < float(row.get("end", 0.0)):
                return float(row.get("start", 0.0))
        previous = [r for r in rows if float(r.get("start", 0.0)) <= t]
        if previous:
            return float(max(previous, key=lambda r: float(r.get("start", 0.0)))["start"])
        beats = np.asarray(self._beatgrid, dtype=float)
        return float(beats[0]) if beats.size else 0.0

    def _phrase_beats_elapsed(self, t: float) -> int:
        ref_t = self._phrase_beat_reference(t)
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size == 0:
            return 0
        # Snap the reference to the nearest beat by INDEX. Stored phrase starts are
        # float32 (and may be edited off-grid), so matching by time with a tiny
        # tolerance can miss the start beat; nearest-index is robust.
        ref_idx = int(np.argmin(np.abs(beats - ref_t)))
        # Current beat = the most recent beat the playhead has reached.
        cur_idx = int(np.searchsorted(beats, t + 1e-4, side="right")) - 1
        cur_idx = min(max(cur_idx, ref_idx), beats.size - 1)
        return cur_idx - ref_idx + 1

    def _phrase_label_at(self, t: float) -> str:
        rows = sorted(self._phrase_segments, key=lambda r: float(r.get("start", 0.0)))
        numbered = number_phrase_labels(rows)
        for row, disp in zip(rows, numbered):
            if float(row.get("start", 0.0)) <= t < float(row.get("end", 0.0)):
                return disp
        return ""
