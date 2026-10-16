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
from ui.edit_panel.cue_slots import CuePageWidget
from ui.edit_panel.cuepoint import CuePointEditMixin
from ui.edit_panel.history import logElement, logs
from ui.edit_panel.icons import icon
from ui.edit_panel.jumpcue import JumpCUEEditMixin
from ui.edit_panel.key import KeyEditMixin
from ui.edit_panel.phrase import PhraseEditMixin
from ui.edit_panel.save_worker import SaveWorker
from ui.edit_panel.styles import BUTTON_STYLESHEET, WARNING_BUTTON_STYLESHEET
from ui.theme import current_theme
from utils.cue_points import build_cue_points_np
from utils.phrases import assigned_phrases, build_phrase_segments_np


class EditPanel(
    BeatgridEditMixin,
    KeyEditMixin,
    PhraseEditMixin,
    JumpCUEEditMixin,
    CuePointEditMixin,
    BeatCounterMixin,
    QtWidgets.QWidget,
):
    """Track editor panel: Beat/Key and Phrase/JumpCUE pages plus the shared Edits column
    (undo / redo / save). Sections live in their mixins; this class holds the shared
    state, edit history and saving.

    The pages are shown 1, 2 or 4 side by side (set_column_count, as the header
    has room): one at a time picked by its button, in pairs (Beatgrid + Key,
    CUE + Phrase) picked by the pair's button, or all four."""

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
    # pages side by side: this much space either side of a thin rule between them
    COLUMN_GAP = 18
    # the pages shown together for each column count
    PAGE_GROUPS = {1: ((0,), (1,), (2,), (3,)), 2: ((0, 1), (2, 3)), 4: ((0, 1, 2, 3),)}

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
        self._save_thread = None
        self._save_worker = None
        self._edit_enabled = True
        self._save_in_progress = False
        self.tempo_multiplier = 1
        self.duration = 1e-3
        self._JumpCUE = []
        self._jumpcue_graph_cues: list[dict] = []
        self._cue_points: list[dict] = []
        self._playing = False
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
                cue_points=self._cue_points,
            )
        )
        self.bus.sig_segment_reanalyze_state.connect(self._set_edit_enabled)
        self.bus.sig_tempo_factor_changed.connect(self._on_playback_tempo_changed)
        self.bus.sig_playback_status.connect(self._on_playback_status)
        self._set_edit_enabled(True)

    def _on_playback_status(self, playing: bool) -> None:
        """CALL works only while stopped; an armed jump ends with playback."""
        self._playing = bool(playing)
        if not self._playing:
            self._disarm_jumpcue()
        self._refresh_jumpcue_slots()
        self._refresh_cue_point_slots()

    def _init_ui(self) -> None:
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(self.ROW_H)

        self.btn_undo = QtWidgets.QToolButton()
        self.btn_undo.setIcon(icon("edit_undo"))
        self.btn_undo.setToolTip("Undo Changes")

        self.btn_redo = QtWidgets.QToolButton()
        self.btn_redo.setIcon(icon("edit_redo"))
        self.btn_redo.setToolTip("Redo Changes")

        self.btn_save = QtWidgets.QToolButton()
        self.btn_save.setIcon(icon("edit_save"))
        self.btn_save.setToolTip("Save Edits")
        for btn in (self.btn_undo, self.btn_redo, self.btn_save):
            btn.setIconSize(QtCore.QSize(19, 19))

        # [page selector                          ]
        # [page] [page] ...  [undo / redo / save]
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(self.ROW_VSPACING)
        self._selector = QtWidgets.QHBoxLayout()
        self._selector.setContentsMargins(0, 0, 0, 0)
        self._selector.setSpacing(4)
        root.addLayout(self._selector)
        self._page_group = QtWidgets.QButtonGroup(self)
        self._page_group.setExclusive(True)
        self._page_group.idClicked.connect(self._show_group)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        root.addLayout(body)
        self._columns_row = QtWidgets.QHBoxLayout()
        self._columns_row.setContentsMargins(0, 0, 0, 0)
        self._columns_row.setSpacing(self.COLUMN_GAP)
        body.addLayout(self._columns_row)

        self._pages: list[QtWidgets.QWidget] = []
        grids = []
        for _ in self.PAGES:
            page = QtWidgets.QWidget(self)
            page.hide()
            g = QtWidgets.QGridLayout(page)
            g.setContentsMargins(0, 0, 0, 0)
            g.setHorizontalSpacing(4)
            g.setVerticalSpacing(self.ROW_VSPACING)
            for _r in range(3):
                g.setRowMinimumHeight(_r, self.BTN_HEIGHT)
            # a fixed row for the status line, so it is never squeezed
            g.setRowMinimumHeight(3, self.STATUS_H)
            g.setRowStretch(4, 1)
            self._pages.append(page)
            grids.append(g)
        grid_beat, grid_key, grid_cue, grid_phrase = grids

        beatgrid_buttons = self._build_beatgrid_ui(grid_beat)
        # the CUE page: the JumpCUE and CUE point lists (its own slot sizes) over
        # the button rows and the status line
        cue_page = CuePageWidget(self.BTN_HEIGHT * 3 + self.STATUS_H + self.ROW_VSPACING * 3)
        grid_cue.addWidget(cue_page, 0, 0, 4, 1)
        cue_buttons = [*self._build_jumpcue_ui(cue_page), *self._build_cuepoint_ui(cue_page)]
        text_buttons = [
            *self._build_key_ui(grid_key),
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
        for btn in (self.key_root, self.key_drone, self.key_chord, self.btn_metronome):
            btn.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)
        for btn in self._edit_buttons:
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._edit_buttons += cue_buttons
        status_font = QtGui.QFont(self.font())
        status_font.setPointSize(9)
        for lbl in (self.beatgrid_status, self.key_status, self.phrase_status):
            lbl.setFont(status_font)
            lbl.setFixedHeight(self.STATUS_H)
            # spans the button columns without widening them (an Expanding label
            # spread the spare page width over the columns it spans)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        for g in grids:
            if g is not grid_cue:
                g.setColumnStretch(g.columnCount(), 1)
        # every column is as wide as the widest page
        self._page_w = max(page.sizeHint().width() for page in self._pages)
        self._hosts: list[QtWidgets.QWidget] = []  # one per column, holding its page
        self._shown: tuple[int, ...] = ()
        self._column_count = 0
        self.set_column_count(1)

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
        # never asks the header for more than one column: wider (more columns) only
        # while it has the room, squeezed (fewer columns) when it has not
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(self.sizeHint().width())

    def _selector_button(self, text: str, tooltip: str) -> QtWidgets.QToolButton:
        font = QtGui.QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setFont(font)
        btn.setFixedHeight(self.PAGE_BTN_H)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn.setStyleSheet("QToolButton { padding: 0 6px; }")
        btn.setToolTip(tooltip)
        return btn

    @classmethod
    def _page_tooltip(cls, index: int) -> str:
        name = cls.PAGES[index]
        return "Edit the JumpCUEs and CUE points" if name == "CUE" else f"Edit the {name.lower()}"

    def column_count(self) -> int:
        return self._column_count

    def column_step(self) -> int:
        """Width one more column takes."""
        return self._page_w + 2 * self.COLUMN_GAP + 1

    def set_column_count(self, count: int) -> None:
        """Show the pages 1, 2 or 4 at a time (``count`` is rounded down to one of
        them); the group shown is the one holding the page shown first."""
        count = 4 if count >= 4 else 2 if count >= 2 else 1
        if count == self._column_count:
            return
        first = self._shown[0] if self._shown else 0
        self._release_pages()  # out of the hosts before any is deleted
        self._column_count = count
        # the columns, a thin rule between neighbours
        while self._columns_row.count():
            item = self._columns_row.takeAt(0)
            if item.widget() is not None and item.widget() not in self._hosts:
                item.widget().deleteLater()  # a rule
        while len(self._hosts) > count:
            self._hosts.pop().deleteLater()
        while len(self._hosts) < count:
            host = QtWidgets.QWidget(self)
            host.setFixedWidth(self._page_w)
            QtWidgets.QVBoxLayout(host).setContentsMargins(0, 0, 0, 0)
            self._hosts.append(host)
        for k, host in enumerate(self._hosts):
            if k:
                rule = QtWidgets.QFrame(self)
                rule.setFixedSize(1, self.BTN_HEIGHT * 3 + self.STATUS_H + self.ROW_VSPACING * 3)
                rule.setStyleSheet(f"background: {current_theme().separator};")
                self._columns_row.addWidget(rule, 0, QtCore.Qt.AlignTop)
            self._columns_row.addWidget(host)
        # the selector: a button per group (a pair's names joined); with all four
        # shown, just each page's name over it
        for btn in self._page_group.buttons():
            self._page_group.removeButton(btn)
        while self._selector.count():
            item = self._selector.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        groups = self.PAGE_GROUPS[count]
        if count == 4:
            for k, i in enumerate(groups[0]):
                if k:
                    self._selector.addSpacing(2 * self.COLUMN_GAP + 1 - self._selector.spacing() * 2)
                btn = self._selector_button(self.PAGES[i].upper(), self._page_tooltip(i))
                btn.setCheckable(True)
                btn.setChecked(True)
                btn.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
                btn.setFixedWidth(self._page_w)
                self._selector.addWidget(btn)
            self._selector.addStretch(1)
        else:
            for gi, group in enumerate(groups):
                btn = self._selector_button(
                    " · ".join(self.PAGES[i].upper() for i in group),
                    "\n".join(self._page_tooltip(i) for i in group),
                )
                btn.setCheckable(True)
                self._page_group.addButton(btn, gi)
                self._selector.addWidget(btn)
        self._show_group(next(gi for gi, group in enumerate(groups) if first in group))

    def _release_pages(self) -> None:
        for i in self._shown:
            page = self._pages[i]
            page.hide()
            page.setParent(self)
        self._shown = ()

    def _show_group(self, group_index: int) -> None:
        """Show the pages of group ``group_index`` (of the current column count)."""
        group = self.PAGE_GROUPS[self._column_count][int(group_index)]
        self._release_pages()
        for host, i in zip(self._hosts, group):
            host.layout().addWidget(self._pages[i])
            self._pages[i].show()
        self._shown = tuple(group)
        btn = self._page_group.button(int(group_index))
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def set_track_uid(self, uid):
        self.uid = uid
        self._zeroBPM()
        self._mark_unsaved(False)
    
    def set_track_duration(self, duration):
        value = float(duration or 0.0)
        self.duration = value if np.isfinite(value) and value > 0.0 else 0.0
        if self.duration > 0.0:
            # the editor's own rows only: the model's phrases are not edited by this
            self._phrase_segments = self._phrase_rows(self._phrase_segments)
            self._phrase_cur = -1
            self._update_phrase_status()

    def set_segments(self, beatgrid: np.ndarray, segments: Optional[np.ndarray]) -> None:
        """Set tempo segments: (N, 5) [start, end, bpm, inizio, ts_num]."""
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
                cue_points=self._cue_points,
            )
        )
        self._update_undo_redo_buttons()

    def update_time(self, t: float) -> None:
        self._current_time = float(t)
        self._follow_jumpcue_arm(float(t))
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
                cue_points=self._cue_points,
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
        self._mark_unsaved(True)

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
                    cue_points=self._cue_points,
                )
            )
            self._mark_unsaved(True)
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
        if loge.cue_points is not None:
            self.set_cue_points(copy.deepcopy(loge.cue_points))
            self._broadcast_cue_points()
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
        if loge.cue_points is not None:
            self.set_cue_points(copy.deepcopy(loge.cue_points))
            self._broadcast_cue_points()
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
        auto_block = self.model.features.get("jump_cues_auto_np") if self.model is not None else None
        jump_cues_auto_np = auto_block if isinstance(auto_block, dict) else None
        # UNASSIGNED is not stored: it is where no phrase is
        phrase_segments_np = build_phrase_segments_np(assigned_phrases(self._phrase_segments))
        cue_points_np = build_cue_points_np(copy.deepcopy(self._cue_points))
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
                jump_cues_auto_np=jump_cues_auto_np,
                phrase_segments_np=phrase_segments_np,
                cue_points_np=cue_points_np,
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
        self._mark_unsaved(False)
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
        self._refresh_jumpcue_slots()  # an empty slot's SET follows the lock
        self._update_undo_redo_buttons()

    def _mark_unsaved(self, unsaved: bool) -> None:
        """The save button lit (warning colour, its icon inverted to show on it) while
        there are edits not saved."""
        self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET if unsaved else BUTTON_STYLESHEET)
        self.btn_save.setIcon(icon("edit_save", inverted=bool(unsaved)))

    def _update_undo_redo_buttons(self) -> None:
        if not hasattr(self, "btn_undo") or not hasattr(self, "btn_redo"):
            return
        can_undo = bool(getattr(self, "edit_log", None)) and self.edit_log.can_undo()
        can_redo = bool(getattr(self, "edit_log", None)) and self.edit_log.can_redo()
        self.btn_undo.setEnabled(self._edit_enabled and can_undo)
        self.btn_redo.setEnabled(self._edit_enabled and can_redo)
