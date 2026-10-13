from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from analyzer_core.editor.keystrip import update_key_segments_with_selection
from ui.edit_panel.history import logElement
from ui.edit_panel.styles import WARNING_BUTTON_STYLESHEET
from utils.labels import KEY_DISPLAY_LABELS

class KeyEditMixin:
    """Key section: key reanalysis and key assignment on the selection."""

    def _build_key_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """Key widgets on the Beat/Key page; returns the edit buttons."""

        self.key_reanalyze = QtWidgets.QToolButton()
        self.key_reanalyze.setText("RA")
        self.key_reanalyze.setToolTip("Full-track key reanalyze using current beatgrid (dynamic)")

        self.key_sel_reanalyze = QtWidgets.QToolButton()
        self.key_sel_reanalyze.setText("SelRA")
        self.key_sel_reanalyze.setToolTip("Key reanalyze only the selected range (static snapshot)")

        self.key_sel_start = QtWidgets.QToolButton()
        self.key_sel_start.setText("SelST")
        self.key_sel_start.setToolTip("Set selection start to nearest beat at playhead")

        self.key_sel_end = QtWidgets.QToolButton()
        self.key_sel_end.setText("SelED")
        self.key_sel_end.setToolTip("Set selection end to nearest beat at playhead")

        self.key_deselect = QtWidgets.QToolButton()
        self.key_deselect.setText("DeSel")
        self.key_deselect.setToolTip("Clear key selection")

        self.key_sel_all = QtWidgets.QToolButton()
        self.key_sel_all.setText("SelAL")
        self.key_sel_all.setToolTip("Select entire track for key editing")

        self.key_assign_combo = QtWidgets.QComboBox()
        self.key_assign_combo.addItems(KEY_DISPLAY_LABELS)
        self.key_assign_combo.setToolTip("Select target key for current selection")
        self.key_assign_combo.setFixedHeight(self.BTN_HEIGHT)
        self.key_assign_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        view = self.key_assign_combo.view()
        if view is not None:
            view.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)

        self.key_assign_button = QtWidgets.QToolButton()
        self.key_assign_button.setText("KeyAssign")
        self.key_assign_button.setToolTip("Write selected key to the current selection")

        self.key_rel_minor_button = QtWidgets.QToolButton()
        self.key_rel_minor_button.setText("RelMinor")
        self.key_rel_minor_button.setToolTip("Convert selection key to relative minor/major")

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        grid.addWidget(self.key_reanalyze, 0, 0, _AL)
        grid.addWidget(self.key_sel_reanalyze, 0, 1, _AL)
        grid.addWidget(self.key_assign_combo, 0, 2, _AL)
        grid.addWidget(self.key_sel_start, 1, 0, _AL)
        grid.addWidget(self.key_sel_end, 1, 1, _AL)
        grid.addWidget(self.key_assign_button, 1, 2, _AL)
        grid.addWidget(self.key_deselect, 2, 0, _AL)
        grid.addWidget(self.key_sel_all, 2, 1, _AL)
        grid.addWidget(self.key_rel_minor_button, 2, 2, _AL)

        self.key_reanalyze.clicked.connect(self._reanalyze_key_dynamic)
        self.key_sel_reanalyze.clicked.connect(self._reanalyze_key_selection)
        self.key_sel_start.clicked.connect(self._key_selection_set_start)
        self.key_sel_end.clicked.connect(self._key_selection_set_end)
        self.key_deselect.clicked.connect(self._key_selection_clear)
        self.key_sel_all.clicked.connect(self._key_selection_select_all)
        self.key_assign_button.clicked.connect(self._assign_key_selection)
        self.key_rel_minor_button.clicked.connect(self._convert_selection_relative_minor)

        return [
            self.key_reanalyze,
            self.key_sel_reanalyze,
            self.key_sel_start,
            self.key_sel_end,
            self.key_deselect,
            self.key_sel_all,
            self.key_assign_button,
            self.key_rel_minor_button,
            self.key_assign_combo,
        ]

    def set_key_segments(self, key_segments, *, initialize: bool = False) -> None:
        if key_segments is None:
            self._key_segments = np.empty((0, 4), dtype=float)
        else:
            arr = np.asarray(key_segments, dtype=float)
            if arr.ndim == 1 and arr.size % 4 == 0:
                arr = arr.reshape((-1, 4))
            if arr.ndim != 2 or arr.shape[1] < 4:
                self._key_segments = np.empty((0, 4), dtype=float)
            else:
                self._key_segments = arr[:, :4].copy()
        if initialize:
            return
        self._emit_key_update()

    def _emit_key_update(self) -> None:
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
        self._broadcast_key_segments()
        self._update_undo_redo_buttons()

    def _broadcast_key_segments(self) -> None:
        """Publish the key segments; views render the key strip from them."""
        if self.model is not None:
            try:
                self.model.features["key_segments"] = np.asarray(self._key_segments, dtype=float)
            except Exception:
                pass
        self.bus.sig_key_segments_updated.emit()

    def _reanalyze_key_selection(self) -> None:
        self._request_segment_reanalysis_key()

    def _reanalyze_key_dynamic(self) -> None:
        self._request_key_reanalysis_dynamic()

    def _request_segment_reanalysis_key(self) -> None:
        if not self._has_active_segment():
            return
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        selection = self._current_selection_payload()
        if selection is None:
            return
        start, end = selection
        start_val = start if start is not None else 0.0
        end_val = end if end is not None else start_val
        if end_val - start_val <= 1e-6:
            return
        payload = {
            "path": self._current_track_path(),
            "segment_index": int(self._cur_idx),
            "segments": np.asarray(self._segments, dtype=float).copy(),
            "beats_time_sec": np.asarray(self._beatgrid, dtype=float).copy(),
            "key_segments": np.asarray(self._key_segments, dtype=float).copy(),
            "selection": (float(start_val), float(end_val)),
            "analyze_type": "key_static",
            "duration_sec": self.duration
        }
        self.bus.sig_segment_reanalyze_requested.emit(payload)

    def _request_key_reanalysis_dynamic(self) -> None:
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        payload = {
            "path": self._current_track_path(),
            "segment_index": int(self._cur_idx),
            "segments": np.asarray(self._segments, dtype=float).copy(),
            "beats_time_sec": np.asarray(self._beatgrid, dtype=float).copy(),
            "key_segments": np.asarray(self._key_segments, dtype=float).copy(),
            "analyze_type": "key_dynamic",
        }
        self.bus.sig_segment_reanalyze_requested.emit(payload)

    def _assign_key_selection(self) -> None:
        bounds = self._selection_bounds()
        if bounds is None:
            return
        start, end = bounds
        idx = int(self.key_assign_combo.currentIndex())
        if idx < 0:
            return
        pitch = idx % 12
        mode_flag = 0 if idx < 12 else 1
        self._apply_key_selection_update(pitch, mode_flag, start, end)

    def _convert_selection_relative_minor(self) -> None:
        bounds = self._selection_bounds()
        if bounds is None:
            return
        start, end = bounds
        segs = np.asarray(self._key_segments, dtype=float)
        if segs.ndim != 2 or segs.shape[1] < 4 or segs.size == 0:
            return
        overlap = segs[(segs[:, 2] < end) & (segs[:, 3] > start)]
        if overlap.size == 0:
            return
        current = overlap[0]
        pitch = int(current[0]) % 12
        mode_flag = 0 if not np.isfinite(current[1]) else int(current[1]) & 1
        if mode_flag == 0:
            pitch = (pitch + 9) % 12
            mode_flag = 1
        else:
            pitch = (pitch + 3) % 12
            mode_flag = 0
        self._apply_key_selection_update(pitch, mode_flag, start, end)

    def _apply_key_selection_update(self, pitch: int, mode_flag: int, start: float, end: float) -> None:
        if end <= start:
            return
        segs = np.asarray(self._key_segments, dtype=float)
        if segs.ndim != 2 or segs.shape[1] < 4:
            segs = np.empty((0, 4), dtype=float)
        selection = (float(pitch), float(mode_flag), float(start), float(end))
        updated = update_key_segments_with_selection(
            segs,
            selection,
            beat_times=self._beatgrid,
        )
        self._key_segments = np.asarray(updated, dtype=float)
        self._emit_key_update()
