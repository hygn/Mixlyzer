from __future__ import annotations

import html
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from audioengine.native import client as deck
from analyzer.music.beat_geometry import downbeat_beat_indices
from analyzer.music.marks import CuePoint, Phrase
from core.workflows.event_bus import EventBus
from core.track.model import DataModel
from ui.edit_panel.beat_counter import BeatCounterMixin
from ui.edit_panel.beatgrid import BeatgridEditMixin
from ui.edit_panel.cue_slots import SLOTS_PER_PAGE, CuePageWidget
from ui.edit_panel.cuepoint import CuePointEditMixin
from ui.edit_panel.flash import flash
from ui.edit_panel.icons import icon
from ui.edit_panel.jumpcue import JumpCUEEditMixin
from ui.edit_panel.key import KeyEditMixin
from ui.edit_panel.phrase import PhraseEditMixin
from core.analysis_library import jobs as analysis_jobs
from ui.edit_panel.styles import BUTTON_STYLESHEET, WARNING_BUTTON_STYLESHEET
from ui.theme import current_theme
from system.i18n import tr, tr_mark


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
    Phrase + CUE) picked by the pair's button, or all four."""

    BTN_WIDTH = 56
    BTN_HEIGHT = 25
    ROW_VSPACING = 2   # tight gaps between button rows (packed look)
    STATUS_H = 16      # the status line under a page (segment / phrase)
    PAGE_BTN_H = 20    # page selector row on top
    LOCK_W = 28        # the edit lock, first in the selector row
    # Panel height: page selector + 3 button rows + the status line and their
    # gaps; it sits in the track info header (TrackInfoPanel.HEADER_H = 120)
    # between the info and the deck.
    ROW_H = PAGE_BTN_H + BTN_HEIGHT * 3 + STATUS_H + ROW_VSPACING * 4
    PAGES = ("Beatgrid", "Key", "Phrase", "CUE")
    # the pages' names shown (translated; Key and CUE as they are) and their tooltips
    PAGE_TITLES = {"Beatgrid": tr_mark("Beatgrid"), "Phrase": tr_mark("Phrase")}
    PAGE_TIPS = {"Beatgrid": tr_mark("Edit the beatgrid"), "Key": tr_mark("Edit the key"),
                 "Phrase": tr_mark("Edit the phrase"), "CUE": tr_mark("Edit the JumpCUEs and CUE points")}
    # pages side by side: this much space either side of a thin rule between them
    COLUMN_GAP = 18
    # the pages shown together for each column count
    PAGE_GROUPS = {1: ((0,), (1,), (2,), (3,)), 2: ((0, 1), (2, 3)), 4: ((0, 1, 2, 3),)}

    def __init__(self, bus: EventBus, model: DataModel, library=None, parent=None):
        """``model``: the loaded track (core.track.model), which holds the marks shown here and is asked to change
        them; ``library``: the library's owner (core.library.service), which saves the edits."""
        super().__init__(parent)
        self._library = library
        self.setStyleSheet(BUTTON_STYLESHEET)
        self._cur_idx: int = -1          # the tempo segment under the playhead
        self._current_time: float = 0.0
        self.bus = bus
        self.model = model
        self.prev_tap_time = None
        self.prev_tap_BPM = 0
        self._edit_enabled = True
        # editing needs the lock open (it is on at every start) and no segment reanalysis running
        self._locked = True
        self._reanalysis_idle = True
        # the pages whose visualizations the views emphasize (sig_edit_focus_changed)
        self._edit_focus: frozenset = frozenset()
        self._save_in_progress = False
        self.tempo_multiplier = 1
        self._playing = False
        # BeatCounter: while the Count button is held, count beats crossed by
        # the playhead relative to the beat index at press time.
        self._beat_counting = False
        self._beat_count = 0
        self._beat_count_anchor_idx = 0
        # Downbeats at/after the press time; the first one is bar 1 for the
        # Bar.Beat display (counting starts once that bar begins).
        self._bb_downbeats = None
        # the current page (index in PAGES): the one the editor's page-wide shortcuts work on; with pages side
        # by side, the selector marks it
        self._context = 0
        self._page_titles: dict[int, QtWidgets.QToolButton] = {}   # 4 pages shown: the title over each
        self._pair_labels: dict[int, QtWidgets.QLabel] = {}         # 2 shown: the names on each pair's button
        self._init_ui()
        # the marks shown follow the model: an edit here, a reanalysis, undo / redo, a track loaded
        model.sig_loaded.connect(self._on_track_loaded)
        model.sig_beatgrid_changed.connect(self._on_beatgrid_changed)
        model.sig_keys_changed.connect(self._on_keys_changed)
        model.sig_phrases_changed.connect(self._on_phrases_changed)
        model.sig_jump_cues_changed.connect(self._refresh_jumpcue_slots)
        model.sig_cue_points_changed.connect(self._refresh_cue_point_slots)
        model.sig_edits_changed.connect(self._on_edits_changed)
        self.bus.sig_phrase_labels_changed.connect(self.refresh_phrase_label_names)
        self.bus.sig_segment_reanalyze_state.connect(self._set_edit_enabled)
        # the marks the player's deck works with, sent when an edit (or a new track) replaced them
        self._marks_sent: Optional[tuple] = None
        self._marks_timer = QtCore.QTimer(self)
        self._marks_timer.setInterval(100)
        self._marks_timer.timeout.connect(self._send_deck_marks)
        self._marks_timer.start()
        self.bus.sig_tempo_factor_changed.connect(self._on_playback_tempo_changed)
        self.bus.sig_playback_status.connect(self._on_playback_status)
        self._set_locked(self.btn_lock.isChecked())

    # ---- the marks shown: the model's (read-only here; an edit asks the model, model.set_*) ----
    @property
    def _beatgrid(self) -> np.ndarray:
        return self.model.beats

    @property
    def _segments(self) -> np.ndarray:
        return self.model.tempo_segments

    @property
    def _key_segments(self) -> np.ndarray:
        return self.model.key_segments

    @property
    def _phrase_segments(self) -> list[Phrase]:
        return self.model.phrases

    @property
    def _JumpCUE(self) -> list[dict]:
        return self.model.jump_cues

    @property
    def _cue_points(self) -> list[CuePoint]:
        return self.model.cue_points

    @property
    def _jumpcue_graph_cues(self) -> list[dict]:
        """The JumpCUEs' cues with their slots (letters kept: lettered by point only by the analysis)."""
        return self.model.jump_cue_graph[0]

    @property
    def uid(self) -> Optional[str]:
        return self.model.properties.get("uid")

    @property
    def duration(self) -> float:
        value = float(self.model.duration_sec or 0.0)
        return value if np.isfinite(value) and value > 0.0 else 0.0

    def _on_track_loaded(self) -> None:
        self._cur_idx = -1
        self._zeroBPM()
        self._on_keys_changed()
        self._on_phrases_changed()
        self._refresh_jumpcue_slots()
        self._refresh_cue_point_slots()
        self.update_time(self._current_time)

    def _on_beatgrid_changed(self) -> None:
        if not 0 <= self._cur_idx < len(self._segments):
            self._cur_idx = -1
        self.update_time(self._current_time)
        self._refresh_time_sig_selector()

    def _on_keys_changed(self) -> None:
        self._key_cur = -1
        self._update_key_current()

    def _on_phrases_changed(self) -> None:
        self._phrase_cur = -1
        self._update_phrase_status()

    def _on_edits_changed(self) -> None:
        self._update_undo_redo_buttons()
        self._mark_unsaved(self.model.unsaved)

    def _on_playback_status(self, playing: bool) -> None:
        """CALL works only while stopped (the player reports it)."""
        self._playing = bool(playing)
        self._refresh_jumpcue_slots()
        self._refresh_cue_point_slots()

    def _send_deck_marks(self) -> None:
        """The beats (with their downbeats), CUE points and JumpCUEs as edited, to the player's deck (its beat /
        bar steps, CUE, CALL, JUMP; a MIDI controller's too), whenever one of them was replaced."""
        held = (self._beatgrid, self._segments, self._jumpcue_graph_cues, self._cue_points)
        if self._marks_sent is not None and all(a is b for a, b in zip(held, self._marks_sent)):
            return
        self._marks_sent = held
        beats = np.asarray(self._beatgrid if self._beatgrid is not None else [], dtype=float).reshape(-1)
        down = np.zeros(beats.size, dtype=bool)
        if beats.size:
            idx = np.asarray(downbeat_beat_indices(beats, self._segments), dtype=np.int64).reshape(-1)
            down[idx[(idx >= 0) & (idx < beats.size)]] = True
        cues = np.asarray([float(p.time_sec) for p in self._cue_points], dtype=float)
        jumps = np.full(26, np.nan)
        for c in self._jumpcue_graph_cues:
            i = ord(str(c.get("label", "?"))[:1] or "?") - ord("A")
            if 0 <= i < 26:
                jumps[i] = float(c.get("point", 0.0))
        self.bus.sig_deck_marks.emit((beats, down, cues, jumps))

    def _init_ui(self) -> None:
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(self.ROW_H)
        # the edit modes' button pairs (top, bottom), each filling a column three rows high
        self._mode_pairs: list[tuple[QtWidgets.QAbstractButton, QtWidgets.QAbstractButton]] = []

        self.btn_undo = QtWidgets.QToolButton()
        self.btn_undo.setIcon(icon("edit_undo"))
        self.btn_undo.setToolTip(tr("Undo Changes"))

        self.btn_redo = QtWidgets.QToolButton()
        self.btn_redo.setIcon(icon("edit_redo"))
        self.btn_redo.setToolTip(tr("Redo Changes"))

        self.btn_save = QtWidgets.QToolButton()
        self.btn_save.setIcon(icon("edit_save"))
        self.btn_save.setToolTip(tr("Save Edits"))
        for btn in (self.btn_undo, self.btn_redo, self.btn_save):
            btn.setIconSize(QtCore.QSize(19, 19))

        # the edit lock: on at every start; locked, the pages still switch but nothing is
        # edited or saved
        self.btn_lock = QtWidgets.QToolButton()
        self.btn_lock.setCheckable(True)
        self.btn_lock.setChecked(True)
        self.btn_lock.setFixedSize(self.LOCK_W, self.PAGE_BTN_H)
        self.btn_lock.setIconSize(QtCore.QSize(15, 15))
        self.btn_lock.setStyleSheet("QToolButton { padding: 0; }")
        self.btn_lock.toggled.connect(self._set_locked)

        # [lock] [page selector                   ]
        # [page] [page] ...  [undo / redo / save]
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(self.ROW_VSPACING)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        top.addWidget(self.btn_lock)
        self._selector = QtWidgets.QHBoxLayout()
        self._selector.setContentsMargins(0, 0, 0, 0)
        self._selector.setSpacing(4)
        top.addLayout(self._selector, 1)
        root.addLayout(top)
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
        grid_beat, grid_key, grid_phrase, grid_cue = grids

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
        # the key picker and the buttons with a wide icon span two button cells
        for btn in (self.key_pick, self.key_relative, self.phrase_unassign, self.phrase_cue_assign):
            btn.setFixedSize(self.BTN_WIDTH * 2 + grid_key.horizontalSpacing(), self.BTN_HEIGHT)
        # an edit mode's two buttons fill their column, three button rows high
        block_h = self.BTN_HEIGHT * 3 + self.ROW_VSPACING * 2
        mode_h = (block_h - self.ROW_VSPACING) // 2
        for top, bottom in self._mode_pairs:
            top.setFixedSize(self.BTN_WIDTH, mode_h)
            bottom.setFixedSize(self.BTN_WIDTH, block_h - self.ROW_VSPACING - mode_h)
        # the listening aids and the metronome stay enabled while editing is locked,
        # but are sized like the rest
        for btn in (self.key_root, self.key_drone, self.key_chord, self.btn_metronome):
            btn.setFixedSize(self.BTN_WIDTH, self.BTN_HEIGHT)
        for btn in self._edit_buttons:
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # what never changes the track stays usable while editing is locked, as the listening
        # aids and the metronome: the reference BPM (tap / type / clear) and the beat counter
        never_locked = (self.zero_tap_cell, self.inp_refBPM, self.beat_counter_button)
        self._edit_buttons = [b for b in self._edit_buttons if b not in never_locked]
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
        # a click anywhere on a page makes it the current page
        self._page_of = {w: i for i, page in enumerate(self._pages)
                         for w in (page, *page.findChildren(QtWidgets.QWidget))}
        for w in self._page_of:
            w.installEventFilter(self)
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
        self.layout().activate()   # with the undo / redo / save column
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
        return tr(cls.PAGE_TIPS[cls.PAGES[index]])

    @classmethod
    def _page_title(cls, index: int) -> str:
        name = cls.PAGES[index]
        return tr(cls.PAGE_TITLES[name]).upper() if name in cls.PAGE_TITLES else name.upper()

    def column_count(self) -> int:
        return self._column_count

    def column_step(self) -> int:
        """Width one more column takes."""
        return self._page_w + 2 * self.COLUMN_GAP + 1

    def set_column_count(self, count: int) -> None:
        """Show the pages 1, 2 or 4 at a time (``count`` is rounded down to one of
        them); the group shown is the one holding the current page."""
        count = 4 if count >= 4 else 2 if count >= 2 else 1
        if count == self._column_count:
            return
        first = self._context
        self._release_pages()  # out of the hosts before any is deleted
        self._column_count = count
        # the columns, a thin rule between neighbours
        while self._columns_row.count():
            item = self._columns_row.takeAt(0)
            if item.widget() is not None and item.widget() not in self._hosts:
                item.widget().hide()  # a rule; gone now, deleted later
                item.widget().deleteLater()
        while len(self._hosts) > count:
            host = self._hosts.pop()
            host.hide()
            host.deleteLater()
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
                item.widget().hide()  # gone now, deleted later
                item.widget().deleteLater()
        groups = self.PAGE_GROUPS[count]
        self._page_titles = {}
        self._pair_labels = {}
        if count == 4:
            for k, i in enumerate(groups[0]):
                if k:
                    self._selector.addSpacing(2 * self.COLUMN_GAP + 1 - self._selector.spacing() * 2)
                btn = self._selector_button(self._page_title(i), self._page_tooltip(i))
                btn.setCheckable(True)
                # lit: the current page; clicked, it becomes the current page
                btn.clicked.connect(lambda _=False, i=i: self._set_context(i))
                # over its column: the first one shares its width with the lock
                btn.setFixedWidth(self._page_w - (0 if k else self.LOCK_W + self._selector.spacing()))
                self._selector.addWidget(btn)
                self._page_titles[i] = btn
            self._selector.addStretch(1)
        else:
            for gi, group in enumerate(groups):
                names = " · ".join(self._page_title(i) for i in group)
                btn = self._selector_button(names, "\n".join(self._page_tooltip(i) for i in group))
                btn.setCheckable(True)
                if count == 2:
                    # the names in a label: the current page's lit, the other's dimmed (_mark_context)
                    label = QtWidgets.QLabel(names, btn)
                    label.setFont(btn.font())
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
                    QtWidgets.QHBoxLayout(btn).setContentsMargins(0, 0, 0, 0)
                    btn.layout().addWidget(label)
                    btn.setText("")
                    btn.setMinimumWidth(label.sizeHint().width() + 12)
                    self._pair_labels[gi] = label
                self._page_group.addButton(btn, gi)
                self._selector.addWidget(btn)
        self._show_group(next(gi for gi, group in enumerate(groups) if first in group))
        # the new width at once in sizeHint(), not after the next layout pass (the selector
        # sits in a nested row beside the lock)
        self.layout().activate()

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
        self._set_context(self._context if self._context in group else group[0])
        self._emit_edit_focus()

    def _set_context(self, index: int) -> None:
        """Make page ``index`` (shown) the current page."""
        self._context = int(index)
        self._mark_context()

    def _mark_context(self) -> None:
        """The current page in the selector while pages are side by side: its title lit over its column (4
        shown), or its name lit on its pair's button and the other's dimmed (2 shown)."""
        for i, btn in self._page_titles.items():
            btn.setChecked(i == self._context)
        theme = current_theme()
        dim = QtGui.QColor(theme.on_accent)
        dim.setAlphaF(0.45)
        for gi, label in self._pair_labels.items():
            group = self.PAGE_GROUPS[2][gi]
            shown = tuple(group) == self._shown
            names = []
            for i in group:
                name = html.escape(self._page_title(i))
                if shown and i != self._context:
                    name = f"<span style='color: {dim.name(QtGui.QColor.HexArgb)}'>{name}</span>"
                names.append(name)
            label.setText(" · ".join(names))
            label.setStyleSheet(f"color: {theme.on_accent if shown else theme.text};")

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind == QtCore.QEvent.MouseButtonPress:
            page = self._page_of.get(obj)
            if page is not None and page in self._shown and page != self._context:
                self._set_context(page)
        return super().eventFilter(obj, event)

    def _emit_edit_focus(self) -> None:
        """Unlocked, the views emphasize the visualizations of the pages shown (beatgrid, key,
        phrase, CUE) and dim the others'; locked, nothing is emphasized."""
        focus = frozenset() if self._locked else frozenset(self.PAGES[i].lower() for i in self._shown)
        if focus != self._edit_focus:
            self._edit_focus = focus
            self.bus.sig_edit_focus_changed.emit(focus)

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

    def _undo(self) -> None:
        self.model.undo()

    def _redo(self) -> None:
        self.model.redo()

    def _save(self) -> None:
        if self.uid is None or self._save_in_progress or not self._edit_enabled:
            return
        if self._library is None:
            return
        try:
            edits = self.model.stored_edits()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, tr("Invalid JumpCUE Label"), str(exc))
            return
        self._save_in_progress = True
        self.btn_save.setEnabled(False)
        # written by the library's owner (analysis_jobs.save_edits); a failure said, the edits kept unsaved
        self._library.write(analysis_jobs.save_edits, self.uid, edits, done=self._on_save_finished,
                            failed=self._on_save_failed)

    def _on_save_failed(self, exc: Exception) -> None:
        self._save_in_progress = False
        self.btn_save.setEnabled(self._edit_enabled)
        QtWidgets.QMessageBox.warning(self, tr("Save"), tr("The edits could not be saved:\n{error}").format(error=exc))

    def _on_save_finished(self, saved_uid=None) -> None:
        self._save_in_progress = False
        self.btn_save.setEnabled(self._edit_enabled)
        uid = str(saved_uid or "").strip()
        if uid and uid == self.uid:      # still the loaded track: its edits are stored
            self.model.mark_saved()
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
        """A segment reanalysis has finished (True) or is running (False)."""
        self._reanalysis_idle = bool(enabled)
        self._apply_edit_enabled()

    def _set_locked(self, locked: bool) -> None:
        """The edit lock (its button): locked, the pages still switch but nothing is edited or saved."""
        self._locked = bool(locked)
        self.btn_lock.setIcon(icon("edit_lock" if self._locked else "edit_unlock"))
        self.btn_lock.setToolTip(
            tr("<p>Editing is locked: pages switch, but nothing is edited or saved. Click to unlock.</p>")
            if self._locked else
            tr("<p>Editing is unlocked: the shown pages' graphics are emphasized. Click to lock: no edits, "
               "no saving (on at every start).</p>"))
        self._apply_edit_enabled()
        self._emit_edit_focus()

    def _apply_edit_enabled(self) -> None:
        state = self._reanalysis_idle and not self._locked
        self._edit_enabled = state
        for btn in self._edit_buttons:
            if btn is self.btn_save and getattr(self, "_save_in_progress", False):
                btn.setEnabled(False)
            else:
                btn.setEnabled(state)
        self._refresh_jumpcue_slots()  # an empty slot's SET follows the lock
        for slot in (*self.jc_slots, *self.cp_slots):
            slot.set_renamable(state)   # a cue's name is an edit too
        self._update_undo_redo_buttons()

    def _mark_unsaved(self, unsaved: bool) -> None:
        """The save button lit (warning colour, its icon inverted to show on it) while
        there are edits not saved."""
        self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET if unsaved else BUTTON_STYLESHEET)
        self.btn_save.setIcon(icon("edit_save", inverted=bool(unsaved)))

    # ---- keyboard shortcuts (ui.shortcuts) ----

    def shortcut_handlers(self) -> dict:
        """Action id -> handler of the editor's shortcuts. A shortcut works a button as the mouse does: clicked
        and flashing. A page-wide one ("editor." moves, split / merge, reanalysis, ...) works the button of the
        current page (the same operation on each page having it); one of a page's own tools shows its page first (and the CUE list page holding its slot),
        which becomes the current page. An edit while editing is locked does nothing; the status bar says why."""
        B, K, P, C = range(len(self.PAGES))
        # the page-wide shortcuts (an operation the same on each page having it): the button each page has for it
        context_tools = {
            "editor.start_earlier": {K: self.key_start_back, P: self.phrase_start_back},
            "editor.start_later": {K: self.key_start_fwd, P: self.phrase_start_fwd},
            "editor.end_earlier": {K: self.key_end_back, P: self.phrase_end_back},
            "editor.end_later": {K: self.key_end_fwd, P: self.phrase_end_fwd},
            "editor.split": {B: self.btn_cutseg, K: self.key_split, P: self.phrase_split},
            "editor.merge_previous": {B: self.btn_merge_prev, K: self.key_merge_prev, P: self.phrase_merge_prev},
            "editor.merge_next": {B: self.btn_merge_next, K: self.key_merge_next, P: self.phrase_merge_next},
            "editor.reanalyze_segment": {B: self.btn_reanalyze_segment, K: self.key_sel_reanalyze},
            "editor.reanalyze_all": {K: self.key_reanalyze, P: self.phrase_reanalyze, C: self.jc_reanalyze},
            "editor.scope_all": {B: self.btn_scope_all, K: self.key_scope_all},
            "editor.scope_segment": {B: self.btn_scope_segment, K: self.key_scope_segment},
            "editor.step_beat": {K: self.key_step_beat, P: self.phrase_step_beat},
            "editor.step_bar": {K: self.key_step_bar, P: self.phrase_step_bar},
        }
        # a page's own tools
        tools = {
            "beatgrid.shift_half_back": (self.btn_shift_half_back, B),
            "beatgrid.shift_coarse_back": (self.btn_shift_coarse_back, B),
            "beatgrid.shift_fine_back": (self.btn_shift_fine_back, B),
            "beatgrid.shift_fine_forward": (self.btn_shift_fine_fwd, B),
            "beatgrid.shift_coarse_forward": (self.btn_shift_coarse_fwd, B),
            "beatgrid.shift_half_forward": (self.btn_shift_half_fwd, B),
            "beatgrid.zero": (self.btn_zeroBPM, B), "beatgrid.tap": (self.btn_tapBPM, B),
            "beatgrid.assign": (self.btn_setBPM, B),
            "beatgrid.ramp_start": (self.btn_ramp_start, B), "beatgrid.ramp_end": (self.btn_ramp_end, B),
            "beatgrid.downbeat": (self.btn_set_downbeat, B),
            "beatgrid.bpm_half": (self.btn_bpm_half, B), "beatgrid.bpm_double": (self.btn_bpm_double, B),
            **{f"beatgrid.time_signature_{n}": (btn, B) for n, btn in self.ts_buttons.items()},
            "key.pick": (self.key_pick, K), "key.flat": (self.key_flat, K), "key.sharp": (self.key_sharp, K),
            "key.fifth_down": (self.key_fifth_down, K), "key.fifth_up": (self.key_fifth_up, K),
            "key.relative": (self.key_relative, K), "key.merge_same": (self.key_merge_same, K),
            "key.root": (self.key_root, K), "key.drone": (self.key_drone, K), "key.chord": (self.key_chord, K),
            "phrase.cue_points": (self.phrase_cue_assign, P), "phrase.unassign": (self.phrase_unassign, P),
            **{f"phrase.label.{label}": (btn, P) for label, btn in self.phrase_label_buttons.items()},
            "editor.undo": (self.btn_undo, None), "editor.redo": (self.btn_redo, None),
            "editor.save": (self.btn_save, None),
        }
        handlers = {action: (lambda buttons=buttons: self._shortcut_context(buttons))
                    for action, buttons in context_tools.items()}
        handlers.update({action: (lambda btn=btn, page=page: self._shortcut_click(btn, page))
                         for action, (btn, page) in tools.items()})
        handlers.update({
            "editor.lock": lambda: self._shortcut_toggle(self.btn_lock),
            # the deck's metronome (also a setting): its button flashes only when shown
            "playback.metronome": lambda: self._shortcut_toggle(self.btn_metronome),
            **{f"editor.page_{name.lower()}": (lambda i=i: self._show_shortcut_page(i))
               for i, name in enumerate(self.PAGES)},
            "cue.call_previous": lambda: self._shortcut_call(-1),
            "cue.call_next": lambda: self._shortcut_call(1),
            "cue.memory": self._shortcut_memory,
            "cue.memory_delete": self._shortcut_memory_delete,
        })
        for letter in "ABCDEFGH":
            for op in ("jump", "set", "delete"):
                handlers[f"cue.{op}.{letter}"] = lambda letter=letter, op=op: self._shortcut_jumpcue(letter, op)
        return handlers

    def show_page(self, index: int) -> None:
        """Show page ``index`` (in its group)."""
        groups = self.PAGE_GROUPS[self._column_count]
        self._show_group(next(gi for gi, group in enumerate(groups) if index in group))

    def _show_shortcut_page(self, index: int) -> None:
        """The page a shortcut works on, shown when it is not, as the current page."""
        if index not in self._shown:
            self.show_page(index)
        self._set_context(index)

    def _page_name(self, index: int) -> str:
        name = self.PAGES[index]
        return tr(self.PAGE_TITLES[name]) if name in self.PAGE_TITLES else name

    def _shortcut_context(self, buttons: dict) -> str | None:
        """A page-wide shortcut: the current page's button for it (``buttons``: page -> button)."""
        page = self._context
        button = buttons.get(page)
        if button is None:
            return tr("This shortcut does nothing on the {page} page.").format(page=self._page_name(page))
        if button is self.jc_reanalyze:
            self.cue_page.show_list(0)  # beside the JumpCUE list
        return self._shortcut_click(button, page)

    def _edit_blocked(self) -> str | None:
        """Why editing is off: the lock or a segment reanalysis (None: editing is on)."""
        if self._locked:
            return tr("Editing is locked.")
        if not self._reanalysis_idle:
            return tr("Editing waits for the segment reanalysis to finish.")
        return None

    def _shortcut_click(self, button: QtWidgets.QAbstractButton, page: int | None) -> str | None:
        """A tool's shortcut: its page shown, its button clicked as by the mouse, flashing."""
        if page is not None:
            self._show_shortcut_page(page)
        if not button.isEnabled():
            return self._edit_blocked()
        button.click()
        flash(button)
        return None

    def _shortcut_toggle(self, button: QtWidgets.QAbstractButton) -> None:
        """A switch that works locked too (the lock, the metronome), flashing while shown."""
        button.toggle()
        if button.isVisible():
            flash(button)

    def _show_cue_list(self, list_idx: int, page: int | None = None) -> None:
        """The CUE page at list ``list_idx`` (0 = JumpCUE, 1 = CUE points), at ``page`` when given."""
        self._show_shortcut_page(self.PAGES.index("CUE"))
        self.cue_page.show_list(list_idx, page)

    def _shortcut_jumpcue(self, letter: str, op: str) -> str | None:
        """JumpCUE ``letter``'s slot by its shortcut, shown in the JumpCUE list: "jump" (JUMP), "set" (SET on its
        empty slot) or "delete" (its colour menu's)."""
        index = ord(letter) - ord("A")
        self._show_cue_list(0, index // SLOTS_PER_PAGE)
        slot = self.jc_slots[index % SLOTS_PER_PAGE]
        cue = self._jc_cue_by_letter(letter)
        if op == "jump":
            if cue is None:
                return tr("JumpCUE {letter} is not set.").format(letter=letter)
            self._jc_jump_cue(cue)
            flash(slot.btn_action)
            return None
        refusal = self._edit_blocked()
        if refusal is not None:
            return refusal
        if op == "set":
            if cue is not None:
                return tr("JumpCUE {letter} is already set.").format(letter=letter)
            self._new_jumpcue_group(letter)
            flash(slot.btn_action)
        elif cue is None:
            return tr("JumpCUE {letter} is not set.").format(letter=letter)
        else:
            flash(slot.squircle)
            self._remove_jumpcue_member(float(cue.get("point", 0.0)))
        return None

    def _shortcut_memory(self) -> str | None:
        """MEMORY: a CUE point at the nearest beat (the list turns to it)."""
        self._show_cue_list(1)
        return self._shortcut_click(self.cp_memory, None)

    def _shortcut_call(self, step: int) -> str | None:
        """CALL the previous (``step`` < 0) / next CUE point (the player's deck: as a MIDI controller's), shown in
        the CUE point list."""
        self._show_cue_list(1)
        if getattr(self, "_playing", False):
            return tr("CALL works while stopped.")
        cue = self._cp_call_target(step)
        if cue is not None:
            slot = self._show_cue_point_slot(cue)
            self.bus.sig_deck_action.emit(deck.ACT_CALL_NEXT if step > 0 else deck.ACT_CALL_PREVIOUS, True)
            flash(slot.btn_action)
        return None

    def _shortcut_memory_delete(self) -> str | None:
        """Delete the CUE point the playhead is on (as after CALL)."""
        self._show_cue_list(1)
        refusal = self._edit_blocked()
        if refusal is not None:
            return refusal
        cue = self._cue_point_at_playhead()
        if cue is None:
            return tr("No CUE point at the playhead.")
        flash(self._show_cue_point_slot(cue).squircle)
        self._delete_cue_point(cue)
        return None

    def _update_undo_redo_buttons(self) -> None:
        if not hasattr(self, "btn_undo") or not hasattr(self, "btn_redo"):
            return
        self.btn_undo.setEnabled(self._edit_enabled and self.model.can_undo())
        self.btn_redo.setEnabled(self._edit_enabled and self.model.can_redo())
