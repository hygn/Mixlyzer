from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from core.analysis_lib_handler import FeatureNPZStore
from core.config import load_cfg
from core.event_bus import EventBus
from core.model import DataModel
from ui.edit_panel.beat_counter import BeatCounterMixin
from ui.edit_panel.beatgrid import BeatgridEditMixin
from ui.edit_panel.history import logElement, logs
from ui.edit_panel.jumpcue import JumpCUEEditMixin
from ui.edit_panel.key import KeyEditMixin
from ui.edit_panel.phrase import PhraseEditMixin
from ui.edit_panel.save_worker import SaveWorker
from ui.edit_panel.selection import SelectionMixin
from ui.edit_panel.styles import BUTTON_STYLESHEET, WARNING_BUTTON_STYLESHEET
from utils.phrases import build_phrase_segments_np, normalize_phrase_segments


class EditPanel(
    BeatgridEditMixin,
    KeyEditMixin,
    PhraseEditMixin,
    JumpCUEEditMixin,
    BeatCounterMixin,
    SelectionMixin,
    QtWidgets.QWidget,
):
    """Track editor panel: Beat/Key and Phrase/JumpCUE pages plus the shared Edits column
    (undo / redo / save). Sections live in their mixins; this class holds the shared
    state, edit history and saving."""

    BTN_WIDTH = 56
    BTN_HEIGHT = 25
    ROW_VSPACING = 2   # tight gaps between button rows (packed look)
    STATUS_H = 16      # the status line under a page (segment / phrase)
    PAGE_BTN_H = 20    # page selector row on top
    # Panel height: page selector + 3 button rows + the status line and their
    # gaps; it sits in the track info header (TrackInfoPanel.HEADER_H = 120)
    # between the info and the deck.
    ROW_H = PAGE_BTN_H + BTN_HEIGHT * 3 + STATUS_H + ROW_VSPACING * 4
    PAGES = ("Beatgrid", "Key", "CUE", "Phrase")

    def __init__(self, bus: EventBus, model: DataModel | None = None, parent=None):
        super().__init__(parent)
        self.setStyleSheet(BUTTON_STYLESHEET)
        self._segments: np.ndarray = np.empty((0, 5), dtype=float)
        self._beatgrid: np.ndarray = np.asarray([])
        self._cur_idx: int = -1
        self._current_time: float = 0.0
        self._key_segments: np.ndarray = np.empty((0, 4), dtype=float)
        self._phrase_segments: list[dict] = []
        self.uid = None
        self.bus = bus
        self.model = model
        self.prev_tap_time = None
        self.prev_tap_BPM = 0
        self._selection_start: float | None = None
        self._selection_end: float | None = None
        self._save_thread = None
        self._save_worker = None
        self._edit_enabled = True
        self._save_in_progress = False
        self.tempo_multiplier = 1
        self.duration = 1e-3
        self._JumpCUE = []
        self._jumpcue_graph_cues: list[dict] = []
        self._jump_armed = False
        # BeatCounter: while the Count button is held, count beats crossed by
        # the playhead relative to the beat index at press time.
        self._beat_counting = False
        self._beat_count = 0
        self._beat_count_anchor_idx = 0
        # Downbeats at/after the press time; the first one is bar 1 for the
        # Bar.Beat display (counting starts once that bar begins).
        self._bb_downbeats = None
        self._init_ui()
        self.edit_log = logs(
            default=logElement(
                beats_time=self._beatgrid,
                beat_segments=self._segments,
                key_segments=self._key_segments,
                jump_cues=self._JumpCUE,
                phrases=self._phrase_segments,
            )
        )
        self.bus.sig_segment_reanalyze_state.connect(self._set_edit_enabled)
        self.bus.sig_tempo_factor_changed.connect(self._on_playback_tempo_changed)
        self._set_edit_enabled(True)

    def _init_ui(self) -> None:
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(self.ROW_H)

        self.cuepoint_placeholder = QtWidgets.QToolButton()
        self.cuepoint_placeholder.setText("Coming Soon")
        self.cuepoint_placeholder.setToolTip("CUEPoint editor placeholder")
        self.cuepoint_placeholder.setEnabled(False)
        self.cuepoint_placeholder.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)

        self.btn_undo = QtWidgets.QToolButton()
        self.btn_undo.setText("↩")
        self.btn_undo.setToolTip("Undo Changes")

        self.btn_redo = QtWidgets.QToolButton()
        self.btn_redo.setText("↪")
        self.btn_redo.setToolTip("Redo Changes")

        self.btn_save = QtWidgets.QToolButton()
        self.btn_save.setText("SAVE")
        self.btn_save.setToolTip("Save Edits")

        # [page selector]
        # [page stack] [undo / redo / save]
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(self.ROW_VSPACING)

        # page selector: toggle buttons like the beatgrid edit-scope pair
        selector = QtWidgets.QHBoxLayout()
        selector.setContentsMargins(0, 0, 0, 0)
        selector.setSpacing(4)
        self._page_group = QtWidgets.QButtonGroup(self)
        self._page_group.setExclusive(True)
        self.page_buttons: list[QtWidgets.QToolButton] = []
        page_font = QtGui.QFont(self.font())
        page_font.setPointSize(9)
        page_font.setBold(True)
        for i, name in enumerate(self.PAGES):
            btn = QtWidgets.QToolButton()
            btn.setText(name.upper())
            btn.setFont(page_font)
            btn.setCheckable(True)
            btn.setFixedHeight(self.PAGE_BTN_H)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn.setStyleSheet("QToolButton { padding: 0 6px; }")
            btn.setToolTip("Edit the CUEs (JumpCUE)" if name == "CUE" else f"Edit the {name.lower()}")
            self._page_group.addButton(btn, i)
            self.page_buttons.append(btn)
            selector.addWidget(btn)
        self.page_buttons[0].setChecked(True)  # Beatgrid by default
        self._page_group.idClicked.connect(self.set_active_page)
        root.addLayout(selector)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        root.addLayout(body)

        self.editor_stack = QtWidgets.QStackedWidget()
        grids = []
        for _ in self.PAGES:
            page = QtWidgets.QWidget()
            g = QtWidgets.QGridLayout(page)
            g.setContentsMargins(0, 0, 0, 0)
            g.setHorizontalSpacing(4)
            g.setVerticalSpacing(self.ROW_VSPACING)
            for _r in range(3):
                g.setRowMinimumHeight(_r, self.BTN_HEIGHT)
            # a fixed row for the status line, so it is never squeezed
            g.setRowMinimumHeight(3, self.STATUS_H)
            g.setRowStretch(4, 1)
            self.editor_stack.addWidget(page)
            grids.append(g)
        grid_beat, grid_key, grid_cue, grid_phrase = grids
        body.addWidget(self.editor_stack)

        beatgrid_buttons = self._build_beatgrid_ui(grid_beat)
        text_buttons = [
            *self._build_key_ui(grid_key),
            *self._build_jumpcue_ui(grid_cue),
            *self._build_phrase_ui(grid_phrase),
            *self._build_beat_counter_ui(grid_phrase),
            self.btn_undo,
            self.btn_redo,
            self.btn_save,
        ]
        self._edit_buttons = beatgrid_buttons + text_buttons
        for btn in beatgrid_buttons + text_buttons:
            btn.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)
        # the key button spans two button cells
        self.key_pick.setFixedSize(self.BTN_WIDTH * 2 + grid_key.horizontalSpacing(), self.BTN_HEIGHT)
        # the listening aids and the metronome stay enabled while editing is locked,
        # but are sized like the rest
        for btn in (self.key_drone, self.key_chord, self.key_cadence, self.btn_metronome):
            btn.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)
        for btn in self._edit_buttons:
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        grid_cue.addWidget(self.cuepoint_placeholder, 0, 3, _AL)
        status_font = QtGui.QFont(self.font())
        status_font.setPointSize(9)
        for lbl in (self.beatgrid_status, self.key_status, self.phrase_status):
            lbl.setFont(status_font)
            lbl.setFixedHeight(self.STATUS_H)
            # spans the button columns without widening them (an Expanding label
            # spread the spare page width over the columns it spans)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        for g in grids:
            g.setColumnStretch(g.columnCount(), 1)

        edits = QtWidgets.QVBoxLayout()
        edits.setContentsMargins(0, 0, 0, 0)
        edits.setSpacing(self.ROW_VSPACING)
        for btn in (self.btn_undo, self.btn_redo, self.btn_save):
            edits.addWidget(btn)
        edits.addStretch(1)
        body.addLayout(edits)

        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo.clicked.connect(self._redo)
        self.btn_save.clicked.connect(self._save)

    def set_active_page(self, index: int) -> None:
        """Switch the editor page: 0 Beatgrid, 1 Key, 2 CUE, 3 Phrase."""
        self.editor_stack.setCurrentIndex(int(index))
        btn = self._page_group.button(int(index))
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def set_track_uid(self, uid):
        self.uid = uid
        self._zeroBPM()
        self.btn_save.setStyleSheet(BUTTON_STYLESHEET)
    
    def set_track_duration(self, duration):
        value = float(duration or 0.0)
        self.duration = value if np.isfinite(value) and value > 0.0 else 0.0
        if self.duration > 0.0 and self._phrase_segments:
            self._phrase_segments = normalize_phrase_segments(
                self._phrase_segments,
                min_time=0.0,
                max_time=self.duration,
            )
            self._broadcast_phrase()

    def set_segments(self, beatgrid: np.ndarray, segments: Optional[np.ndarray]) -> None:
        """Set tempo segments: (N, 5) [start, end, bpm, inizio, ts_num]."""
        self._clear_selection()
        if segments is None:
            self._segments = np.empty((0, 5), dtype=float)
            self._cur_idx = -1
            return
        if beatgrid is None:
            beatgrid = np.asarray([])
            return
        self._beatgrid = beatgrid
        arr = np.asarray(segments, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 4:
            self._segments = np.empty((0, 5), dtype=float)
            self._cur_idx = -1
            return
        # Keep 5 columns [start, end, bpm, inizio, ts_num]; backfill ts=4 for legacy N×4.
        arr = self._ensure_ts_column(arr)
        # Keep only valid rows: finite start/end and end > start
        starts = arr[:, 0]
        ends = arr[:, 1]
        valid = np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
        if not np.any(valid):
            self._segments = np.empty((0, 5), dtype=float)
            self._cur_idx = -1
            return
        arr = arr[valid]
        # Sort by start ascending
        order = np.argsort(arr[:, 0])
        self._segments = arr[order]
        self._cur_idx = -1
        self.edit_log = logs(
            default=logElement(
                beats_time=self._beatgrid,
                beat_segments=self._segments,
                key_segments=self._key_segments,
                jump_cues=self._JumpCUE,
                phrases=self._phrase_segments,
            )
        )
        self._update_undo_redo_buttons()

    def update_time(self, t: float) -> None:
        self._current_time = float(t)
        self._update_key_current()
        self._update_beat_counter(float(t))
        self._update_barbeat_display(float(t))
        segs = self._segments
        if segs.size == 0:
            self._set_status(None, 0.0, 0.0)
            return
        starts = segs[:, 0]
        ends = segs[:, 1]
        cur_t = float(t)
        self._current_time = cur_t
        # Find the segment containing t; if none, snap to nearest
        within = (starts <= cur_t) & (cur_t < ends)
        if np.any(within):
            idx = int(np.argmax(within))
        else:
            if cur_t < starts[0]:
                idx = 0
            elif cur_t >= ends[-1]:
                idx = len(starts) - 1
            else:
                idx = int(np.searchsorted(starts, cur_t, side="left"))
        # Elapsed uses segment start; Left uses end - t
        elapsed = max(0.0, cur_t - float(starts[idx]))
        left = max(0.0, float(ends[idx]) - cur_t)
        self._cur_idx = idx
        self._set_status(idx, elapsed, left)

    def _emit_edit_update(self) -> None:
        """Notify listeners that the beatgrid/segments changed."""
        self.edit_log.push(
            logElement(
                beats_time=self._beatgrid,
                beat_segments=self._segments,
                key_segments=self._key_segments,
                jump_cues=self._JumpCUE,
                phrases=self._phrase_segments,
            )
        )
        if self.model is not None:
            try:
                self.model.features["beats_time_sec"] = np.asarray(self._beatgrid, dtype=float)
                self.model.features["tempo_segments"] = np.asarray(self._segments, dtype=float)
            except Exception:
                pass
        self.bus.sig_beatgrid_edited.emit()
        self._update_undo_redo_buttons()
        self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET)

    def apply_update_from_external(self, *, beatgrid=None, segments=None, key_segments=None) -> None:
        updated = False
        if beatgrid is not None and segments is not None:
            self._beatgrid = beatgrid
            self._segments = segments
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
            if self.model is not None:
                try:
                    self.model.features["beats_time_sec"] = np.asarray(self._beatgrid, dtype=float)
                    self.model.features["tempo_segments"] = np.asarray(self._segments, dtype=float)
                except Exception:
                    pass
            self.bus.sig_beatgrid_edited.emit()
            updated = True
        if key_segments is not None:
            self.set_key_segments(key_segments)
        if key_segments is not None:
            self.bus.sig_key_selection_changed.emit(None)
        if updated:
            self._update_undo_redo_buttons()

    def _undo(self) -> None:
        loge = self.edit_log.undo()
        self._update_undo_redo_buttons()
        if loge is None:
            return
        self._beatgrid = loge.beats_time
        self._segments = loge.beat_segments
        if self.model is not None:
            try:
                self.model.features["beats_time_sec"] = np.asarray(self._beatgrid, dtype=float)
                self.model.features["tempo_segments"] = np.asarray(self._segments, dtype=float)
            except Exception:
                pass
        self.bus.sig_beatgrid_edited.emit()
        if loge.key_segments is not None:
            self._key_segments = loge.key_segments
            self._broadcast_key_segments()
        if loge.jump_cues is not None:
            restored = copy.deepcopy(loge.jump_cues)
            self.set_JumpCUE(restored)
            self._broadcast_jumpcue()
        if loge.phrases is not None:
            self._phrase_segments = copy.deepcopy(loge.phrases)
            self._broadcast_phrase()
        self.update_time(self._current_time)

    def _redo(self) -> None:
        loge = self.edit_log.redo()
        self._update_undo_redo_buttons()
        if loge is None:
            return
        self._beatgrid = loge.beats_time
        self._segments = loge.beat_segments
        if self.model is not None:
            try:
                self.model.features["beats_time_sec"] = np.asarray(self._beatgrid, dtype=float)
                self.model.features["tempo_segments"] = np.asarray(self._segments, dtype=float)
            except Exception:
                pass
        self.bus.sig_beatgrid_edited.emit()
        if loge.key_segments is not None:
            self._key_segments = loge.key_segments
            self._broadcast_key_segments()
        if loge.jump_cues is not None:
            restored = copy.deepcopy(loge.jump_cues)
            self.set_JumpCUE(restored)
            self._broadcast_jumpcue()
        if loge.phrases is not None:
            self._phrase_segments = copy.deepcopy(loge.phrases)
            self._broadcast_phrase()
        self.update_time(self._current_time)
    
    def _save(self) -> None:
        if self.uid is None or self._save_in_progress:
            return
        cfg = load_cfg()
        store = FeatureNPZStore(base_dir=cfg.libconfig.libpath, compressed=True)
        thread = QtCore.QThread(self)
        save_worker = SaveWorker()
        save_worker.moveToThread(thread)
        beatgrid = np.asarray(self._beatgrid, dtype=float).copy() if self._beatgrid is not None else None
        segments = np.asarray(self._segments, dtype=float).copy() if self._segments is not None else None
        key_segments = np.asarray(self._key_segments, dtype=float).copy() if self._key_segments is not None else None
        try:
            jump_cues_np = self._jumpcues_to_np(self._JumpCUE)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid JumpCUE Label", str(exc))
            return
        jump_cues_extracted = copy.deepcopy(self._JumpCUE)
        auto_block = self.model.features.get("jump_cues_auto_np") if self.model is not None else None
        jump_cues_auto_np = auto_block if isinstance(auto_block, dict) else None
        self._phrase_segments = normalize_phrase_segments(
            self._phrase_segments,
            min_time=0.0,
            max_time=self.duration if self.duration > 0.0 else None,
        )
        phrase_segments_np = build_phrase_segments_np(self._phrase_segments)
        self._save_in_progress = True
        self.btn_save.setEnabled(False)
        thread.started.connect(
            lambda: save_worker.save(
                store,
                cfg.libconfig.libpath,
                self.uid,
                beatgrid=beatgrid,
                segments=segments,
                key_segments=key_segments,
                jump_cues_np=jump_cues_np,
                jump_cues_extracted=jump_cues_extracted,
                jump_cues_auto_np=jump_cues_auto_np,
                phrase_segments_np=phrase_segments_np,
            )
        )
        save_worker.finished.connect(thread.quit)
        save_worker.finished.connect(save_worker.deleteLater)
        save_worker.finished.connect(thread.deleteLater)
        save_worker.finished.connect(self._on_save_finished)
        thread.finished.connect(lambda: setattr(self, "_save_thread", None))
        thread.finished.connect(lambda: setattr(self, "_save_worker", None))
        self._save_thread = thread
        self._save_worker = save_worker
        thread.start()

    def _on_save_finished(self, saved_uid=None) -> None:
        self._save_in_progress = False
        self.btn_save.setEnabled(self._edit_enabled)
        self.btn_save.setStyleSheet(BUTTON_STYLESHEET)
        uid = str(saved_uid or "").strip()
        if uid:
            try:
                self.bus.sig_rekordbox_sync_track_requested.emit(uid)
            except Exception:
                pass

    def _current_track_path(self) -> Optional[str]:
        if self.model is None:
            return None
        props = getattr(self.model, "properties", {}) or {}
        try:
            return props.get("path") or props.get("track_id")
        except Exception:
            return None

    def _set_edit_enabled(self, enabled: bool) -> None:
        state = bool(enabled)
        self._edit_enabled = state
        for btn in self._edit_buttons:
            if btn is self.btn_save and getattr(self, "_save_in_progress", False):
                btn.setEnabled(False)
            else:
                btn.setEnabled(state)
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self) -> None:
        if not hasattr(self, "btn_undo") or not hasattr(self, "btn_redo"):
            return
        can_undo = bool(getattr(self, "edit_log", None)) and self.edit_log.can_undo()
        can_redo = bool(getattr(self, "edit_log", None)) and self.edit_log.can_redo()
        self.btn_undo.setEnabled(self._edit_enabled and can_undo)
        self.btn_redo.setEnabled(self._edit_enabled and can_redo)
