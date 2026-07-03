from __future__ import annotations
import math
from typing import Optional
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from analyzer_core.editor.beatgrid import shift_grid_in_seg, rebuild_grid_from_segments, canonical_inizio
from analyzer_core.editor.keystrip import update_key_segments_with_selection
from core.event_bus import EventBus
import time
from dataclasses import dataclass, field
import copy
import os
from core.analysis_lib_handler import FeatureNPZStore
from core.config import load_cfg
from core.library_handler import LibraryDB
from core.model import DataModel
from core.linear_segments import build_bpm_segments, build_key_segments
from core.beat_geometry import bar_beat_position, downbeat_beat_indices
from utils.keystrip import build_keystrip_buffer
from utils.labels import KEY_DISPLAY_LABELS
from utils.jump_cues import build_jump_cues_np, extract_jump_cue_pairs, build_jump_cue_graph
from utils.jumpcue_colors import get_jumpcue_pair_color
from utils.phrases import (
    PHRASE_LABELS,
    assign_phrase_to_selection,
    build_phrase_segments_np,
    clear_phrase_in_selection,
    extract_phrase_segments,
    number_phrase_labels,
)

MAX_LOG_HISTORY = 50
BUTTON_STYLESHEET = """
QToolButton {
    color: #FFFFFF;
    background-color: #2f2f2f;
}
QToolButton:disabled {
    color: #666666;
    background-color: #1f1f1f;
}
"""
WARNING_BUTTON_STYLESHEET = """
QToolButton {
    background-color: #8A8A00;
}
QToolButton:disabled {
    color: #666666;
    background-color: #4A4A1f;
}
"""

@dataclass
class logElement:
    beat_segments: np.ndarray
    beats_time: np.ndarray
    key_segments: np.ndarray | None = None
    jump_cues: list[dict] | None = None
    phrases: list[dict] | None = None

    def __post_init__(self) -> None:
        self.beat_segments = np.asarray(self.beat_segments, dtype=float).copy()
        self.beats_time = np.asarray(self.beats_time, dtype=float).copy()
        if self.key_segments is not None:
            self.key_segments = np.asarray(self.key_segments, dtype=float).copy()
        if self.jump_cues is not None:
            self.jump_cues = copy.deepcopy(self.jump_cues)
        if self.phrases is not None:
            self.phrases = copy.deepcopy(self.phrases)

    def clone(self) -> logElement:
        return logElement(
            beat_segments=self.beat_segments,
            beats_time=self.beats_time,
            key_segments=self.key_segments,
            jump_cues=self.jump_cues,
            phrases=self.phrases,
        )


@dataclass
class logs:
    default: logElement
    log: list[logElement] = field(default_factory=list)
    relidx: int = -1

    def __post_init__(self) -> None:
        default_entry = self.default.clone()
        self.default = default_entry
        if not self.log:
            self.log = [default_entry]
        else:
            self.log = [entry.clone() for entry in self.log]
        self.relidx = len(self.log) - 1

    def push(self, loge: logElement) -> None:
        entry = loge.clone()
        if self.relidx < len(self.log) - 1:
            self.log = self.log[: self.relidx + 1]
        self.log.append(entry)
        if len(self.log) > MAX_LOG_HISTORY:
            drop = len(self.log) - MAX_LOG_HISTORY
            self.log = self.log[drop:]
        self.relidx = len(self.log) - 1

    def undo(self) -> Optional[logElement]:
        if len(self.log) <= 1:
            self.relidx = len(self.log) - 1
            return None
        if self.relidx <= 0:
            self.relidx = 0
            return None
        self.relidx -= 1
        return self.log[self.relidx].clone()

    def redo(self) -> Optional[logElement]:
        if not self.log:
            self.relidx = -1
            return None
        if self.relidx >= len(self.log) - 1:
            self.relidx = len(self.log) - 1
            return None
        self.relidx += 1
        return self.log[self.relidx].clone()

    def can_undo(self) -> bool:
        return len(self.log) > 1 and self.relidx > 0

    def can_redo(self) -> bool:
        return 0 <= self.relidx < len(self.log) - 1

class SaveWorker(QtCore.QObject):
    finished = QtCore.Signal(object)

    def save(self, store: FeatureNPZStore, libpath: str, uid, *, beatgrid=None, segments=None, key_segments=None, key_np=None, jump_cues_np=None, jump_cues_extracted=None, phrase_segments_np=None):
        try:
            if uid is None:
                return
            prev_feat = store.load(uid)
            if beatgrid is not None:
                prev_feat["beats_time_sec"] = beatgrid
            if segments is not None:
                prev_feat["tempo_segments"] = segments
            if key_segments is not None:
                prev_feat["key_segments"] = key_segments
                if key_np is None:
                    duration = prev_feat.get("duration_sec", 0.0)
                    key_np = build_keystrip_buffer(key_segments, duration)
            if key_np is not None:
                prev_feat["key_np"] = key_np
            if jump_cues_np is not None:
                prev_feat["jump_cues_np"] = jump_cues_np
            if jump_cues_extracted is not None:
                prev_feat["jump_cues_extracted"] = jump_cues_extracted
            if phrase_segments_np is not None:
                prev_feat["phrase_segments_np"] = phrase_segments_np
            store.save(uid, prev_feat)
            db = LibraryDB(os.path.join(libpath, "library.db"))
            db.connect()
            try:
                db.replace_bpm_segments(uid, build_bpm_segments(prev_feat.get("tempo_segments")))
                db.replace_key_segments(uid, build_key_segments(prev_feat.get("key_segments")))
            finally:
                db.close()
        except Exception:
            pass
        finally:
            self.finished.emit(uid)
        return
            


class BeatgridEditPanel(QtWidgets.QWidget):
    """Beatgrid edit helper panel (dummy buttons + status).

    - Segment internal beatgrid shift (dummy)
    - Segment internal beatgrid phase +180° (dummy)
    - Shows current segment index, elapsed from segment start, and time left
    """

    BTN_WIDTH = 88
    BTN_HEIGHT = 28
    ROW_VSPACING = 2   # tight gaps between button rows (packed look)
    STATUS_H = 24      # reserved height for the bottom status label row
                       # (label is ~20px at the app's 11pt global font)
    # Panel height: root v-margins (4+4) + 3 button rows + 3 inter-row gaps +
    # the reserved status-label row. Kept tight so button rows stay packed
    # while the status line (Beatgrid Seg / Phrase) is never clipped.
    ROW_H = 8 + BTN_HEIGHT * 3 + ROW_VSPACING * 3 + STATUS_H

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
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        approx_h = self.ROW_H
        self.setMinimumHeight(approx_h)
        self.setMaximumHeight(approx_h)

        # Beatgrid Edit Panel
        self.beatgrid_label = QtWidgets.QLabel("Beatgrid: ")
        f = self.beatgrid_label.font()
        f.setBold(True)
        self.beatgrid_label.setFont(f)
        
        self.btn_shift_back = QtWidgets.QToolButton()
        self.btn_shift_back.setText("<<")
        self.btn_shift_back.setToolTip("Shift current segment beatgrid backward by 10 ms")

        self.btn_shift_fwd = QtWidgets.QToolButton()
        self.btn_shift_fwd.setText(">>")
        self.btn_shift_fwd.setToolTip("Shift current segment beatgrid forward by 10 ms")

        self.btn_phase_plus_180 = QtWidgets.QToolButton()
        self.btn_phase_plus_180.setText("+½Beat")
        self.btn_phase_plus_180.setToolTip("Shift beatgrid phase by +0.5 beat (half-cycle)")

        self.btn_phase_minus_180 = QtWidgets.QToolButton()
        self.btn_phase_minus_180.setText("-½Beat")
        self.btn_phase_minus_180.setToolTip("Shift beatgrid phase by -0.5 beat (half-cycle)")

        self.btn_cutseg = QtWidgets.QToolButton()
        self.btn_cutseg.setText("Split")
        self.btn_cutseg.setToolTip("Cut current segment at current play position")

        self.btn_merge_prev_keep_prev = QtWidgets.QToolButton()
        self.btn_merge_prev_keep_prev.setText("Merge[<")
        self.btn_merge_prev_keep_prev.setToolTip("Merge previous+current; keep previous tempo/phase")

        self.btn_merge_prev_keep_current = QtWidgets.QToolButton()
        self.btn_merge_prev_keep_current.setText("Merge<]")
        self.btn_merge_prev_keep_current.setToolTip("Merge previous+current; keep current tempo/phase")

        self.btn_merge_next_keep_next = QtWidgets.QToolButton()
        self.btn_merge_next_keep_next.setText("Merge>]")
        self.btn_merge_next_keep_next.setToolTip("Merge current+next; keep next tempo/phase")

        self.btn_merge_next_keep_current = QtWidgets.QToolButton()
        self.btn_merge_next_keep_current.setText("Merge[>")
        self.btn_merge_next_keep_current.setToolTip("Merge current+next; keep current tempo/phase")

        self.btn_reanalyze_segment = QtWidgets.QToolButton()
        self.btn_reanalyze_segment.setText("SegRA")
        self.btn_reanalyze_segment.setToolTip("Recompute beatgrid for the current segment from audio")

        self.btn_reanalyze_segment_ref = QtWidgets.QToolButton()
        self.btn_reanalyze_segment_ref.setText("SegRARef")
        self.btn_reanalyze_segment_ref.setToolTip("Reanalyze current segment constrained to the reference BPM")

        self.btn_tapBPM = QtWidgets.QToolButton()
        self.btn_tapBPM.setText("TAP")
        self.btn_tapBPM.setToolTip("Tap to set reference BPM used by SegRARef/Assign")

        self.inp_refBPM = QtWidgets.QLineEdit()
        ref_bpm_regex = QtCore.QRegularExpression(r"^$|^(?:\d+\.?\d*|\.\d+)$")
        self._ref_bpm_validator = QtGui.QRegularExpressionValidator(
            ref_bpm_regex, self.inp_refBPM
        )
        self.inp_refBPM.setValidator(self._ref_bpm_validator)
        self.inp_refBPM.setPlaceholderText("Ref BPM")
        self.inp_refBPM.setToolTip("Reference BPM for reanalysis and Assign button")

        self.btn_setBPM = QtWidgets.QToolButton()
        self.btn_setBPM.setText("Assign")
        self.btn_setBPM.setToolTip("Assign Reference BPM to current segment (sets downbeat to start)")

        self.btn_zeroBPM = QtWidgets.QToolButton()
        self.btn_zeroBPM.setText("ZERO")
        self.btn_zeroBPM.setToolTip("Clear Reference BPM and tap history")

        # Per-segment time signature (numerator, /4 denominator).
        self.cmb_time_sig = QtWidgets.QComboBox()
        for _n in range(1, 17):
            self.cmb_time_sig.addItem(f"{_n}/4", _n)
        self.cmb_time_sig.setCurrentIndex(3)  # 4/4
        self.cmb_time_sig.setToolTip("Select the time signature for the current segment")

        self.btn_apply_time_sig = QtWidgets.QToolButton()
        self.btn_apply_time_sig.setText("Set TS")
        self.btn_apply_time_sig.setToolTip("Apply the selected time signature to the current segment")

        # Key Edit Panel
        self.key_label = QtWidgets.QLabel("Key: ")
        f = self.key_label.font()
        f.setBold(True)
        self.key_label.setFont(f)

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

        # JumpCUE Edit Panel
        self.JumpCUE_label = QtWidgets.QLabel("JumpCUE: ")
        f = self.JumpCUE_label.font()
        f.setBold(True)
        self.JumpCUE_label.setFont(f)

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

        # BeatCounter: hold the button to count beats passing under the playhead.
        self.beatcounter_label = QtWidgets.QLabel("BeatCounter: ")
        f = self.beatcounter_label.font()
        f.setBold(True)
        self.beatcounter_label.setFont(f)

        self.beat_counter_button = QtWidgets.QToolButton()
        self.beat_counter_button.setText("Count")
        self.beat_counter_button.setToolTip("Hold to count beats passing under the playhead")
        self.beat_counter_button.setAutoRepeat(False)

        self.beat_counter_display = QtWidgets.QLineEdit("0")
        self.beat_counter_display.setReadOnly(True)
        self.beat_counter_display.setFocusPolicy(QtCore.Qt.NoFocus)
        self.beat_counter_display.setAlignment(QtCore.Qt.AlignCenter)
        self.beat_counter_display.setToolTip("Beats counted while the Count button is held")
        self.beat_counter_display.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)

        self.beat_counter_barbeat_display = QtWidgets.QLineEdit("--")
        self.beat_counter_barbeat_display.setReadOnly(True)
        self.beat_counter_barbeat_display.setFocusPolicy(QtCore.Qt.NoFocus)
        self.beat_counter_barbeat_display.setAlignment(QtCore.Qt.AlignCenter)
        self.beat_counter_barbeat_display.setToolTip("Bar.Beat of the current playhead position")
        self.beat_counter_barbeat_display.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)

        # Phrase Edit Panel
        self.phrase_label = QtWidgets.QLabel("Phrase: ")
        f = self.phrase_label.font()
        f.setBold(True)
        self.phrase_label.setFont(f)

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

        self.phrase_status = QtWidgets.QLabel("<b>Phrase</b>: --")
        self.phrase_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.cuepoint_label = QtWidgets.QLabel("CUEPoint: ")
        f = self.cuepoint_label.font()
        f.setBold(True)
        self.cuepoint_label.setFont(f)
        self.cuepoint_placeholder = QtWidgets.QToolButton()
        self.cuepoint_placeholder.setText("Coming Soon")
        self.cuepoint_placeholder.setToolTip("CUEPoint editor placeholder")
        self.cuepoint_placeholder.setEnabled(False)
        self.cuepoint_placeholder.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)

        # Edit Panel
        self.edits_label = QtWidgets.QLabel("Edits: ")
        f = self.edits_label.font()
        f.setBold(True)
        self.edits_label.setFont(f)

        self.btn_undo = QtWidgets.QToolButton()
        self.btn_undo.setText("↩")
        self.btn_undo.setToolTip("Undo Changes")

        self.btn_redo = QtWidgets.QToolButton()
        self.btn_redo.setText("↪")
        self.btn_redo.setToolTip("Redo Changes")

        self.btn_save = QtWidgets.QToolButton()
        self.btn_save.setText("SAVE")
        self.btn_save.setToolTip("Save Edits")

        self._edit_buttons = [
            self.btn_shift_back,
            self.btn_shift_fwd,
            self.btn_phase_plus_180,
            self.btn_phase_minus_180,
            self.btn_cutseg,
            self.btn_merge_prev_keep_prev,
            self.btn_merge_prev_keep_current,
            self.btn_merge_next_keep_next,
            self.btn_merge_next_keep_current,
            self.btn_reanalyze_segment,
            self.btn_reanalyze_segment_ref,
            self.btn_tapBPM,
            self.inp_refBPM,
            self.btn_setBPM,
            self.btn_zeroBPM,
            self.cmb_time_sig,
            self.btn_apply_time_sig,
            self.btn_undo,
            self.btn_redo,
            self.btn_save,
            self.key_reanalyze,
            self.key_sel_reanalyze,
            self.key_sel_start,
            self.key_sel_end,
            self.key_deselect,
            self.key_sel_all,
            self.key_assign_button,
            self.key_rel_minor_button,
            self.key_assign_combo,
            self.jc_reanalyze,
            self.jc_jumptest,
            self.beat_counter_button,
            self.phrase_assign_combo,
            self.phrase_sel_start,
            self.phrase_sel_end,
            self.phrase_deselect,
            self.phrase_sel_all,
            self.phrase_assign_button,
            self.phrase_clear_button,
        ]
        for btn in self._edit_buttons:
            btn.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.beatgrid_status = QtWidgets.QLabel("<b>Beatgrid Seg</b>: -- | Elapsed: 00:00.00 | Left: 00:00.00")
        self.beatgrid_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter

        # Top-level: [stacked pages] [shared Edits column]. The page toggle
        # (Beat/Key | Phrase/JumpCUE) lives outside this panel, next to the
        # "Show Editors" checkbox; see MainPane.
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(12, 4, 12, 4)
        root.setSpacing(8)

        # Stacked pages
        self.editor_stack = QtWidgets.QStackedWidget()
        page_beatkey = QtWidgets.QWidget()
        page_phrasejc = QtWidgets.QWidget()
        self.editor_stack.addWidget(page_beatkey)
        self.editor_stack.addWidget(page_phrasejc)
        root.addWidget(self.editor_stack, 1)

        gridA = QtWidgets.QGridLayout(page_beatkey)
        gridA.setContentsMargins(0, 0, 0, 0)
        gridA.setHorizontalSpacing(8)
        gridA.setVerticalSpacing(self.ROW_VSPACING)
        for _r in range(3):
            gridA.setRowMinimumHeight(_r, self.BTN_HEIGHT)
        # Reserve a fixed row for the status label so it is never squeezed /
        # clipped, while the button rows above stay tightly packed.
        gridA.setRowMinimumHeight(3, self.STATUS_H)

        gridB = QtWidgets.QGridLayout(page_phrasejc)
        gridB.setContentsMargins(0, 0, 0, 0)
        gridB.setHorizontalSpacing(8)
        gridB.setVerticalSpacing(self.ROW_VSPACING)
        for _r in range(3):
            gridB.setRowMinimumHeight(_r, self.BTN_HEIGHT)
        gridB.setRowMinimumHeight(3, self.STATUS_H)

        # ---- Page A: Beatgrid + Key ----
        gridA.addWidget(self.beatgrid_label, 0, 0, _AL)
        gridA.addWidget(self.btn_shift_back, 0, 1, _AL)
        gridA.addWidget(self.btn_shift_fwd, 0, 2, _AL)
        gridA.addWidget(self.btn_phase_plus_180, 0, 3, _AL)
        gridA.addWidget(self.btn_phase_minus_180, 0, 4, _AL)
        gridA.addWidget(self.btn_cutseg, 1, 5, _AL)
        gridA.addWidget(self.btn_merge_prev_keep_prev, 1, 1, _AL)
        gridA.addWidget(self.btn_merge_prev_keep_current, 1, 2, _AL)
        gridA.addWidget(self.btn_merge_next_keep_next, 1, 4, _AL)
        gridA.addWidget(self.btn_merge_next_keep_current, 1, 3, _AL)
        gridA.addWidget(self.btn_reanalyze_segment, 0, 5, _AL)
        gridA.addWidget(self.btn_tapBPM, 2, 1, _AL)
        gridA.addWidget(self.inp_refBPM, 2, 3, _AL)
        gridA.addWidget(self.btn_setBPM, 2, 4, _AL)
        gridA.addWidget(self.btn_zeroBPM, 2, 2, _AL)
        gridA.addWidget(self.btn_reanalyze_segment_ref, 2, 5, _AL)
        gridA.addWidget(self.beatgrid_status, 3, 0, 1, 7, _AL)
        gridA.addWidget(self.cmb_time_sig, 0, 6, _AL)
        gridA.addWidget(self.btn_apply_time_sig, 1, 6, _AL)
        gridA.addWidget(self.key_label, 0, 7, _AL)
        gridA.addWidget(self.key_reanalyze, 0, 8, _AL)
        gridA.addWidget(self.key_sel_reanalyze, 0, 9, _AL)
        gridA.addWidget(self.key_assign_combo, 0, 10, _AL)
        gridA.addWidget(self.key_sel_start, 1, 8, _AL)
        gridA.addWidget(self.key_sel_end, 1, 9, _AL)
        gridA.addWidget(self.key_assign_button, 1, 10, _AL)
        gridA.addWidget(self.key_deselect, 2, 8, _AL)
        gridA.addWidget(self.key_sel_all, 2, 9, _AL)
        gridA.addWidget(self.key_rel_minor_button, 2, 10, _AL)
        gridA.setColumnStretch(11, 1)

        # ---- Page B: Phrase + JumpCUE ----
        gridB.addWidget(self.phrase_label, 0, 0, _AL)
        gridB.addWidget(self.phrase_assign_combo, 0, 1, _AL)
        gridB.addWidget(self.phrase_assign_button, 0, 2, _AL)
        gridB.addWidget(self.phrase_clear_button, 0, 3, _AL)
        gridB.addWidget(self.phrase_sel_start, 1, 1, _AL)
        gridB.addWidget(self.phrase_sel_end, 1, 2, _AL)
        gridB.addWidget(self.phrase_deselect, 2, 1, _AL)
        gridB.addWidget(self.phrase_sel_all, 2, 2, _AL)
        gridB.addWidget(self.phrase_status, 3, 0, 1, 4, _AL)
        gridB.addWidget(self.JumpCUE_label, 0, 5, _AL)
        gridB.addWidget(self.jc_reanalyze, 0, 6, _AL)
        gridB.addWidget(self.jc_selection_wrap, 1, 6, _AL)
        gridB.addWidget(self.jc_jumptest, 2, 6, _AL)
        gridB.addWidget(self.beatcounter_label, 0, 7, _AL)
        gridB.addWidget(self.beat_counter_button, 0, 8, _AL)
        gridB.addWidget(self.beat_counter_display, 1, 8, _AL)
        gridB.addWidget(self.beat_counter_barbeat_display, 2, 8, _AL)
        gridB.addWidget(self.cuepoint_label, 0, 9, _AL)
        gridB.addWidget(self.cuepoint_placeholder, 0, 10, _AL)
        gridB.setColumnStretch(11, 1)

        # ---- Shared Edits column (right) ----
        edits_grid = QtWidgets.QGridLayout()
        edits_grid.setContentsMargins(0, 0, 0, 0)
        edits_grid.setHorizontalSpacing(8)
        edits_grid.setVerticalSpacing(8)
        edits_grid.addWidget(self.edits_label, 0, 0, _AL)
        edits_grid.addWidget(self.btn_undo, 0, 1, _AL)
        edits_grid.addWidget(self.btn_redo, 1, 1, _AL)
        edits_grid.addWidget(self.btn_save, 2, 1, _AL)
        root.addLayout(edits_grid)


        # Wire buttons to handlers (still mostly placeholders)
        self.btn_shift_back.clicked.connect(self._shift_beatgrid_backward)
        self.btn_shift_fwd.clicked.connect(self._shift_beatgrid_forward)
        self.btn_phase_plus_180.clicked.connect(self._shift_beatgrid_plus_halfpi)
        self.btn_phase_minus_180.clicked.connect(self._shift_beatgrid_minus_halfpi)
        self.btn_cutseg.clicked.connect(self._cut_current_segment)
        self.btn_merge_prev_keep_prev.clicked.connect(self._merge_prev_keep_prev)
        self.btn_merge_prev_keep_current.clicked.connect(self._merge_prev_keep_current)
        self.btn_merge_next_keep_next.clicked.connect(self._merge_next_keep_next)
        self.btn_merge_next_keep_current.clicked.connect(self._merge_next_keep_current)
        self.btn_reanalyze_segment.clicked.connect(self._reanalyze_current_segment)
        self.btn_reanalyze_segment_ref.clicked.connect(self._reanalyze_with_reference_bpm)
        self.btn_tapBPM.clicked.connect(self._tapBPM_set)
        self.btn_zeroBPM.clicked.connect(self._zeroBPM)
        self.btn_setBPM.clicked.connect(self._setBPM_set)
        self.btn_apply_time_sig.clicked.connect(self._apply_time_signature)
        self.key_reanalyze.clicked.connect(self._reanalyze_key_dynamic)
        self.key_sel_reanalyze.clicked.connect(self._reanalyze_key_selection)

        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo.clicked.connect(self._redo)
        self.btn_save.clicked.connect(self._save)
        self.key_sel_start.clicked.connect(self._key_selection_set_start)
        self.key_sel_end.clicked.connect(self._key_selection_set_end)
        self.key_deselect.clicked.connect(self._key_selection_clear)
        self.key_sel_all.clicked.connect(self._key_selection_select_all)
        self.key_assign_button.clicked.connect(self._assign_key_selection)
        self.key_rel_minor_button.clicked.connect(self._convert_selection_relative_minor)
        
        self.jc_jumptest.clicked.connect(self._arm_next_jumpCUE_jump)
        self.jc_reanalyze.clicked.connect(self._reanalyze_jumpCUE)
        self.beat_counter_button.pressed.connect(self._on_beat_counter_pressed)
        self.beat_counter_button.released.connect(self._on_beat_counter_released)
        self.jc_source_selection.currentIndexChanged.connect(self._jc_source_selection_changed)
        self.jc_target_selection.currentIndexChanged.connect(self._jc_target_selection_changed)

        # Phrase page (selection is shared with the Key page handlers)
        self.phrase_sel_start.clicked.connect(self._key_selection_set_start)
        self.phrase_sel_end.clicked.connect(self._key_selection_set_end)
        self.phrase_deselect.clicked.connect(self._key_selection_clear)
        self.phrase_sel_all.clicked.connect(self._key_selection_select_all)
        self.phrase_assign_button.clicked.connect(self._assign_phrase_selection)
        self.phrase_clear_button.clicked.connect(self._clear_phrase_selection)
    def set_active_page(self, index: int) -> None:
        """Switch the editor page: 0 = Beat/Key, 1 = Phrase/JumpCUE."""
        self.editor_stack.setCurrentIndex(int(index))

    def set_track_uid(self, uid):
        self.uid = uid
        self._zeroBPM()
        self.btn_save.setStyleSheet(BUTTON_STYLESHEET)
    
    def set_track_duration(self, duration):
        self.duration = float(duration)

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
                phrases=self._phrase_segments,
            )
        )
        self._update_undo_redo_buttons()

    def set_key_segments(self, key_segments, *, initialize: bool = False, key_image=None) -> None:
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
        self._emit_key_update(key_image=key_image)
    
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

    def update_time(self, t: float) -> None:
        self._current_time = float(t)
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

    def _beats_before(self, t: float) -> int:
        """Number of beatgrid beats at or before ``t``."""
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size == 0:
            return 0
        return int(np.searchsorted(beats, float(t), side="right"))

    def _on_beat_counter_pressed(self) -> None:
        self._beat_counting = True
        press_beat = self._beats_before(self._current_time)
        self._beat_count_anchor_idx = press_beat
        self._beat_count = 0
        # Bar.Beat counts from the first bar that starts at/after the press
        # point: keep only the downbeat indices from here on, so the first one
        # becomes bar 1 (counting begins once that bar's downbeat is reached).
        db_idx = downbeat_beat_indices(self._beatgrid, self._segments)
        self._bb_downbeats = db_idx[db_idx >= press_beat]
        self._refresh_beat_counter_display()
        self._update_barbeat_display(self._current_time)

    def _on_beat_counter_released(self) -> None:
        self._beat_counting = False

    def _update_beat_counter(self, t: float) -> None:
        if not self._beat_counting:
            return
        count = max(0, self._beats_before(t) - self._beat_count_anchor_idx)
        if count != self._beat_count:
            self._beat_count = count
            self._refresh_beat_counter_display()

    def _refresh_beat_counter_display(self) -> None:
        self.beat_counter_display.setText(str(self._beat_count))

    def _update_barbeat_display(self, t: float) -> None:
        # Edit-panel Bar.Beat only runs while the Count button is held, and
        # starts counting once the first bar after the press begins.
        if not self._beat_counting:
            return
        pos = bar_beat_position(
            self._beatgrid, t, downbeat_indices=self._bb_downbeats
        )
        if pos is None or pos[0] <= 0:
            self.beat_counter_barbeat_display.setText("--")
        else:
            self.beat_counter_barbeat_display.setText(f"{pos[0]}.{pos[1]}")

    def _set_status(self, idx: Optional[int], elapsed: float, left: float) -> None:
        def fmt(sec: float) -> str:
            m = int(sec // 60)
            s = int(sec - 60 * m)
            cs = int((sec - 60 * m - s) * 100)
            return f"{m:02d}:{s:02d}.{cs:02d}"
        if idx is None:
            self.beatgrid_status.setText("<b>Beatgrid Seg</b>: -- | Elapsed: 00:00.00 | Left: 00:00.00")
        else:
            self.beatgrid_status.setText(f"<b>Beatgrid Seg</b>: {idx+1} | Elapsed: {fmt(elapsed)} | Left: {fmt(left)}")
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
        """Show the active segment's time signature in the selector (no apply)."""
        ts = self._current_segment_time_sig()
        idx = self.cmb_time_sig.findData(int(ts))
        if idx < 0:
            idx = self.cmb_time_sig.findData(4)
        if idx >= 0 and self.cmb_time_sig.currentIndex() != idx:
            self.cmb_time_sig.setCurrentIndex(idx)

    def _apply_time_signature(self) -> None:
        """Apply the selected time signature to the current segment."""
        if not self._has_active_segment():
            return
        try:
            value = int(self.cmb_time_sig.currentData())
        except (TypeError, ValueError):
            return
        if value < 1:
            return
        self._segments = self._ensure_ts_column(self._segments)
        idx = self._cur_idx
        if int(round(float(self._segments[idx, 4]))) == value:
            return
        self._segments[idx, 4] = float(value)
        self._emit_edit_update()

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


    def _has_active_segment(self) -> bool:
        """Return True when a valid editing target exists."""
        return self._segments.size > 0 and 0 <= self._cur_idx < len(self._segments)

    def _shift_segment_by_offset(self, offset: float) -> None:
        if not (self._has_active_segment() and len(self._beatgrid) > 0):
            return
        self._beatgrid, self._segments = shift_grid_in_seg(
            self._beatgrid,
            self._segments,
            self._cur_idx,
            offset,
        )
        self._emit_edit_update()

    def _shift_segment_by_phase(self, half_cycles: float) -> None:
        """Shift by a multiple of half cycles (0.5 == 180°)."""
        if not self._has_active_segment():
            return
        bpm = float(self._segments[self._cur_idx][2])
        if not (np.isfinite(bpm) and bpm > 0):
            return
        period = 60.0 / bpm
        self._shift_segment_by_offset(period * half_cycles)

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
    
    def _emit_key_update(self, key_image=None) -> None:
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
        self._broadcast_key_segments(key_image=key_image)
        self._update_undo_redo_buttons()

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

    def _broadcast_key_segments(self, key_image=None) -> None:
        segments_arr = np.asarray(self._key_segments, dtype=float)
        if self.model is not None:
            try:
                self.model.features["key_segments"] = segments_arr
                if key_image is None and segments_arr.size:
                    duration = float(self._track_time_bounds()[1])
                    key_image = build_keystrip_buffer(segments_arr, duration)
                if key_image is not None:
                    self.model.features["key_np"] = np.asarray(key_image, dtype=np.uint8)
            except Exception:
                pass
        self.bus.sig_key_segments_updated.emit()

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
    
    def apply_update_from_external(self, *, beatgrid=None, segments=None, key_segments=None, key_image=None) -> None:
        updated = False
        if beatgrid is not None and segments is not None:
            self._beatgrid = beatgrid
            self._segments = segments
            self.edit_log.push(
                logElement(
                    beats_time=self._beatgrid,
                    beat_segments=self._segments,
                    key_segments=self._key_segments,
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
            self.set_key_segments(key_segments, key_image=key_image)
        elif key_image is not None and self._key_segments is not None:
            self._broadcast_key_segments(key_image=key_image)
        if key_segments is not None or key_image is not None:
            self.bus.sig_key_selection_changed.emit(None)
        if updated:
            self._update_undo_redo_buttons()

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

    def _shift_beatgrid_forward(self) -> None:
        self._shift_segment_by_offset(0.01)

    def _shift_beatgrid_backward(self) -> None:
        self._shift_segment_by_offset(-0.01)

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

    def _shift_beatgrid_minus_halfpi(self) -> None:
        self._shift_segment_by_phase(-0.5)

    def _shift_beatgrid_plus_halfpi(self) -> None:
        self._shift_segment_by_phase(0.5)

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

        next_inz = self._next_downbeat_after(cut_t, inz, bpm, st)
        # New split half inherits the source segment's time signature.
        next_seg = np.array([cut_t, ed, bpm, next_inz, ts_num], dtype=float)
        new_segments = np.insert(new_segments, idx + 1, next_seg, axis=0)

        self._segments = new_segments
        self._beatgrid = rebuild_grid_from_segments(self._segments)
        self._emit_edit_update()
        self.update_time(self._current_time)

    def _merge_prev_keep_prev(self) -> None:
        self._merge_segments(direction="prev", keep="neighbor")

    def _merge_prev_keep_current(self) -> None:
        self._merge_segments(direction="prev", keep="current")

    def _merge_next_keep_next(self) -> None:
        self._merge_segments(direction="next", keep="neighbor")

    def _merge_next_keep_current(self) -> None:
        self._merge_segments(direction="next", keep="current")

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
        updated[target_idx, 3] = canonical_inizio(inz, seg_start, bpm)

        updated = np.delete(updated, donor_idx, axis=0)
        if donor_idx < target_idx:
            target_idx -= 1

        self._segments = updated
        self._beatgrid = rebuild_grid_from_segments(self._segments)
        self._cur_idx = target_idx
        self._emit_edit_update()
        self.update_time(self._current_time)
    def _reanalyze_jumpCUE(self) -> None:
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        payload = {
            "path": self._current_track_path(),
            "beats_time_sec": np.asarray(self._beatgrid, dtype=float).copy(),
        }
        payload["analyze_type"] = "jumpcue"
        self.bus.sig_segment_reanalyze_requested.emit(payload)

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

    def _reanalyze_key_selection(self) -> None:
        self._request_segment_reanalysis_key()

    def _reanalyze_key_dynamic(self) -> None:
        self._request_key_reanalysis_dynamic()

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
    
    def _on_playback_tempo_changed(self, tempofactor):
        if abs(tempofactor - 1) > 0.0001:
            self.btn_tapBPM.setToolTip("Tap to set reference BPM (WARNING! Tempo factor is not 1. RefBPM Will be scaled accordingly)")
            self.inp_refBPM.setToolTip("Reference BPM for Analysis and BPM Assigning (WARNING! Tempo factor is not 1. RefBPM Will be scaled accordingly)")
            self.btn_tapBPM.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        else:
            self.btn_tapBPM.setToolTip("Tap to set reference BPM")
            self.inp_refBPM.setToolTip("Reference BPM for Analysis and BPM Assigning")
            self.btn_tapBPM.setStyleSheet(BUTTON_STYLESHEET)
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
                )
            )
            self._update_undo_redo_buttons()
            self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        if self.model is not None:
            try:
                self.model.features["phrase_segments_np"] = build_phrase_segments_np(self._phrase_segments)
            except Exception:
                pass
        self._update_phrase_status()
        self.bus.sig_phrase_segments_updated.emit()

    def _update_phrase_status(self) -> None:
        t = float(self._current_time)
        label = self._phrase_label_at(t)
        beats = self._phrase_beats_elapsed(t)
        self.phrase_status.setText(f"<b>Phrase</b>: {label or '--'} | +{beats} beats")

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

    def _key_selection_set_start(self) -> None:
        # Always reset the selection to just the start point.
        anchor = self._selection_anchor_time()
        self._selection_start = anchor
        self._selection_end = None
        self._emit_selection_range()

    def _key_selection_set_end(self) -> None:
        anchor = self._selection_anchor_time()
        start = self._selection_start
        end = self._selection_end
        if start is None:
            # No start point yet: nothing to anchor an end against.
            return
        if end is None:
            # Only a start point exists -> set the end.
            self._selection_end = anchor
            self._emit_selection_range()
            return
        # A full range exists; reassign based on the playhead position.
        range_start = min(float(start), float(end))
        range_end = max(float(start), float(end))
        if anchor > range_end:
            # After the range end: old end becomes the new start, click is the end.
            self._selection_start = range_end
            self._selection_end = anchor
        elif anchor < range_start:
            # Before the range start: old start becomes the end, click is the start.
            self._selection_start = anchor
            self._selection_end = range_start
        else:
            # Inside the range: drop the old end, click becomes the new end.
            self._selection_start = range_start
            self._selection_end = anchor
        self._emit_selection_range()

    def _key_selection_clear(self) -> None:
        self._clear_selection()

    def _key_selection_select_all(self) -> None:
        start, end = self._track_time_bounds()
        if end - start <= 1e-6:
            return
        self._selection_start = start
        self._selection_end = end
        self._emit_selection_range()

    def _clear_selection(self, notify: bool = True) -> None:
        self._selection_start = None
        self._selection_end = None
        if notify:
            self.bus.sig_key_selection_changed.emit(None)

    def _selection_anchor_time(self, t: float | None = None) -> float:
        cur = self._current_time if t is None else float(t)
        start, end = self._track_time_bounds()
        cur = float(np.clip(cur, start, end))
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size == 0:
            return cur
        idx = int(np.searchsorted(beats, cur))
        if idx <= 0:
            return start
        if idx >= len(beats):
            return end
        prev_val = float(beats[idx - 1])
        next_val = float(beats[idx])
        return prev_val if (cur - prev_val) <= (next_val - cur) else next_val

    def _track_time_bounds(self) -> tuple[float, float]:
        if self.duration:
            start = 0
            end = self.duration
        elif self._segments.size > 0:
            start = float(self._segments[0, 0])
            end = float(self._segments[-1, 1])
        elif self._beatgrid is not None and len(self._beatgrid) > 0:
            start = float(self._beatgrid[0])
            end = float(self._beatgrid[-1])
        else:
            start = 0.0
            end = max(self._current_time, 0.0)
        return (start, end)

    def _emit_selection_range(self) -> None:
        payload = self._current_selection_payload()
        self.bus.sig_key_selection_changed.emit(payload)

    def _current_selection_payload(self):
        start = self._selection_start
        end = self._selection_end
        if start is None and end is None:
            return None
        s_val = float(start) if start is not None and np.isfinite(start) else None
        e_val = float(end) if end is not None and np.isfinite(end) else None
        if s_val is not None and e_val is not None:
            if e_val < s_val:
                s_val, e_val = e_val, s_val
            if abs(e_val - s_val) <= 1e-6:
                e_val = None
        return (s_val, e_val)

    def _selection_bounds(self) -> Optional[tuple[float, float]]:
        sel = self._current_selection_payload()
        if sel is None:
            return None
        start, end = sel
        if start is None or end is None:
            return None
        return float(start), float(end)

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

    @staticmethod
    def _next_downbeat_after(t: float, inizio: float, bpm: float, fallback_start: float) -> float:
        """Return the absolute time of the first downbeat >= t."""
        target = float(t)
        ref = float(inizio) if np.isfinite(inizio) else float(fallback_start)
        if target <= ref + 1e-9:
            return ref
        if not (np.isfinite(bpm) and bpm > 0):
            return target
        bar = 4.0 * (60.0 / float(bpm))
        if not np.isfinite(bar) or bar <= 0:
            return target
        steps = math.ceil((target - ref) / bar)
        return float(ref + steps * bar)
