"""Settings > MIDI: a MIDI controller for the deck, read by the native audio engine itself (never through the GUI's
Python; none on the Python engine): its input port, its jog wheel's resolution, and the MIDI messages of the deck's
and its mixer channel's actions (``core.controls.midi.ACTIONS``: a DJ controller's; a mapping's entries of no action
are dropped at the next save). Learn takes the next control moved or pressed: a fader's 14-bit CC
or pitch bend, a knob's turn (a button bound to a knob: the direction turned), a button; Import reads a rekordbox
MIDI mapping (CSV). OK / Apply keep the mapping as the user's (``core.controls.midi.save``)."""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional

from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.controls import midi
from system.i18n import tr

_ACTION_ROLE = QtCore.Qt.UserRole
# Learn: a fader's CC waits this long for its LSB (data 1 + 32: 14 bits); a button's CC sending 127 for its 0 (let
# go: a button; else a knob turned counterclockwise one step)
LEARN_WAIT_MS = 300


class _Wait(enum.Enum):
    """A CC heard while learning, waiting LEARN_WAIT_MS for what tells what it is."""
    FADER_LSB = enum.auto()         # a fader's CC under 32: its LSB (data 1 + 32) makes it 14 bits
    BUTTON_RELEASE = enum.auto()    # a CC at 127: its 0 after it is a button let go, none a knob's step back


@dataclasses.dataclass(frozen=True)
class _Learning:
    """Learn on for ``action_id``: listening, or a CC waiting (``wait``: its status and data 1, ``seen`` what it
    waits for came)."""
    action_id: str
    wait: Optional[_Wait] = None
    status: int = 0
    d1: int = 0
    seen: bool = False


def _labels(texts: list[str]) -> str:
    return ", ".join(m.label() for m in (midi.Message.parse(t) for t in texts) if m is not None)


def build_midi_tab(dialog) -> None:
    dialog.tab_midi = QWidget()
    lay = QVBoxLayout(dialog.tab_midi)

    dialog.lbl_midi_note = QLabel()
    dialog.lbl_midi_note.setWordWrap(True)
    lay.addWidget(dialog.lbl_midi_note)

    form = QFormLayout()
    row = QHBoxLayout()
    dialog.cb_midi_input = QComboBox()
    dialog.cb_midi_input.setToolTip(tr("The controller's MIDI input; its output of the same name lights its PLAY and "
                                       "CUE lamps."))
    dialog.btn_midi_refresh = QPushButton(tr("Refresh"))
    dialog.btn_midi_refresh.setToolTip(tr("List the MIDI inputs connected now."))
    row.addWidget(dialog.cb_midi_input, 1)
    row.addWidget(dialog.btn_midi_refresh)
    holder = QWidget()
    holder.setLayout(row)
    row.setContentsMargins(0, 0, 0, 0)
    form.addRow(tr("MIDI input"), holder)
    dialog.lbl_midi_state = QLabel()
    form.addRow("", dialog.lbl_midi_state)
    dialog.sp_midi_jog_ticks = QSpinBox()
    dialog.sp_midi_jog_ticks.setRange(16, 8192)
    dialog.sp_midi_jog_ticks.setToolTip(tr(
        "The jog wheel's messages in one turn. DDJ-400: 720."))
    form.addRow(tr("Jog wheel: messages per turn"), dialog.sp_midi_jog_ticks)
    lay.addLayout(form)

    dialog.ed_midi_filter = QLineEdit()
    dialog.ed_midi_filter.setPlaceholderText(tr("Find an action or a message ..."))
    dialog.ed_midi_filter.setClearButtonEnabled(True)
    lay.addWidget(dialog.ed_midi_filter)

    dialog.tree_midi = QTreeWidget()
    dialog.tree_midi.setColumnCount(3)
    dialog.tree_midi.setHeaderLabels([tr("Action"), tr("MIDI"), tr("Default")])
    dialog.tree_midi.setRootIsDecorated(False)
    dialog.tree_midi.setUniformRowHeights(True)
    header = dialog.tree_midi.header()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    for column in (1, 2):
        header.setSectionResizeMode(column, QHeaderView.Interactive)
        dialog.tree_midi.setColumnWidth(column, 190)
    bold = QtGui.QFont(dialog.tree_midi.font())
    bold.setBold(True)
    dialog._midi_items = {}   # action id -> its row
    groups = [(group, [(a.id, midi.action_label(a), f"<p>{tr(a.tip)}</p>" if a.tip else "")
                       for a in midi.ACTIONS if a.group == group]) for group in midi.GROUPS]
    for group, rows in groups:
        if not rows:
            continue
        top = QTreeWidgetItem([tr(group)])
        top.setFont(0, bold)
        top.setFlags(QtCore.Qt.ItemIsEnabled)
        dialog.tree_midi.addTopLevelItem(top)
        for action_id, label, tip in rows:
            item = QTreeWidgetItem([label, "", ""])
            item.setData(0, _ACTION_ROLE, action_id)
            item.setToolTip(0, tip)
            top.addChild(item)
            dialog._midi_items[action_id] = item
        top.setExpanded(True)
    lay.addWidget(dialog.tree_midi, 1)

    buttons = QHBoxLayout()
    dialog.btn_midi_learn = QPushButton(tr("Learn"))
    dialog.btn_midi_learn.setCheckable(True)
    dialog.btn_midi_learn.setToolTip(tr("Then move or press a control on the controller: its message is added to "
                                        "the selected action."))
    dialog.btn_midi_clear = QPushButton(tr("Clear"))
    dialog.btn_midi_clear.setToolTip(tr("The selected action's MIDI messages removed."))
    dialog.btn_midi_revert = QPushButton(tr("Revert"))
    dialog.btn_midi_revert.setToolTip(tr("The selected action's shipped MIDI messages again."))
    dialog.btn_midi_import = QPushButton(tr("Import rekordbox CSV..."))
    dialog.btn_midi_import.setToolTip(tr(
        "A MIDI mapping exported from rekordbox (CSV): its deck 1 controls, mixer channel 1 and the browser replace "
        "the mapping."))
    dialog.btn_midi_revert_all = QPushButton(tr("Revert All"))
    dialog.btn_midi_revert_all.setToolTip(tr("Every action's shipped MIDI messages again."))
    dialog.lbl_midi_learn = QLabel()
    buttons.addWidget(dialog.btn_midi_learn)
    buttons.addWidget(dialog.btn_midi_clear)
    buttons.addWidget(dialog.lbl_midi_learn, 1)
    buttons.addWidget(dialog.btn_midi_import)
    buttons.addWidget(dialog.btn_midi_revert)
    buttons.addWidget(dialog.btn_midi_revert_all)
    lay.addLayout(buttons)

    dialog.tabs.addTab(dialog.tab_midi, "MIDI")


class MidiSettingsMixin:
    """The mapping being edited (``_midi_map``: action -> messages) until OK / Apply keeps it."""

    def _initialize_midi_settings(self, bus) -> None:
        self._midi_map: dict[str, list[str]] = {}
        self._midi_defaults: dict[str, list[str]] = {}
        self._midi_available = False
        self._midi_ports_provider = None
        self._midi_learn_setter = None
        self._midi_wanted_input = ""
        self._learn: _Learning | None = None    # Learn on (None: off), set by _set_learn alone
        self._midi_learn_timer = QtCore.QTimer(self)    # a CC's wait
        self._midi_learn_timer.setSingleShot(True)
        self._midi_learn_timer.setInterval(LEARN_WAIT_MS)
        self._midi_learn_timer.timeout.connect(self._finish_pending_learn)
        self.tree_midi.currentItemChanged.connect(lambda *_: self._show_selected_midi())
        self.ed_midi_filter.textChanged.connect(self._filter_midi)
        self.btn_midi_refresh.clicked.connect(lambda: self._load_midi_inputs(self._midi_input()))
        self.btn_midi_learn.toggled.connect(self._toggle_midi_learn)
        self.btn_midi_clear.clicked.connect(lambda: self._assign_midi(self._selected_midi_action(), []))
        self.btn_midi_revert.clicked.connect(self._revert_selected_midi)
        self.btn_midi_revert_all.clicked.connect(self._revert_all_midi)
        self.btn_midi_import.clicked.connect(self._import_rekordbox_midi)
        bus.sig_midi_events.connect(self._on_midi_events)
        bus.sig_deck_state.connect(self._on_midi_state)
        self._midi_bus = bus
        self._load_midi_settings()
        self.set_midi_access(False, None, None)

    def set_midi_access(self, available: bool, ports, learn) -> None:
        """Whether a controller can be read (the native engine plays), ``ports()`` -> its MIDI inputs, and
        ``learn(on)`` -> every message reported while on."""
        self._midi_available = bool(available)
        self._midi_ports_provider = ports
        self._midi_learn_setter = learn
        self.lbl_midi_note.setText(
            tr("The native audio engine reads the controller itself: its deck controls answer at once, whatever the "
               "app is doing. Deck 1's controls and mixer channel 1's fader and EQ are supported.") if available else
            tr("A MIDI controller needs the native audio engine (Settings > Audio > Engine); the Python "
               "engine (the compatibility mode) has no MIDI."))
        for w in (self.cb_midi_input, self.btn_midi_refresh, self.sp_midi_jog_ticks, self.ed_midi_filter,
                  self.tree_midi, self.btn_midi_learn, self.btn_midi_clear, self.btn_midi_revert,
                  self.btn_midi_import, self.btn_midi_revert_all):
            w.setEnabled(self._midi_available)
        self._load_midi_inputs(self._midi_wanted_input)

    def _set_midi_config(self, p) -> None:
        self._midi_wanted_input = str(getattr(p, "midi_input", "") or "")
        self.sp_midi_jog_ticks.setValue(int(getattr(p, "midi_jog_ticks", 720) or 720))
        self._load_midi_inputs(self._midi_wanted_input)

    def _midi_input(self) -> str:
        return str(self.cb_midi_input.currentData() or "")

    def _load_midi_inputs(self, wanted: str) -> None:
        combo = self.cb_midi_input
        combo.clear()
        combo.addItem(tr("None"), "")
        names = self._midi_ports_provider() if (self._midi_available and self._midi_ports_provider) else []
        for name in names:
            combo.addItem(name, name)
        if wanted and wanted not in names:
            combo.addItem(tr("{name} (not connected)").format(name=wanted), wanted)
        combo.setCurrentIndex(max(0, combo.findData(wanted)))

    def _on_midi_state(self, report) -> None:
        state = int(getattr(report, "midi_state", 0))
        self.lbl_midi_state.setText(
            {1: tr("Open."), 2: tr("Could not be opened (in use by another program, or disconnected).")}.get(state, ""))

    def _load_midi_settings(self) -> None:
        """The mapping in use (the files), dropping edits not kept."""
        self._stop_midi_learn()
        self._midi_defaults = midi.defaults()
        in_use = midi.mapping()
        self._midi_map = {action_id: list(in_use.get(action_id, [])) for action_id in self._midi_items}
        self._refresh_midi_items()
        self._show_selected_midi()

    def _save_midi_settings(self) -> None:
        self._stop_midi_learn()
        midi.save(self._midi_map)
        self._midi_bus.sig_midi_changed.emit()

    def _selected_midi_action(self) -> str | None:
        item = self.tree_midi.currentItem()
        return item.data(0, _ACTION_ROLE) if item is not None else None

    def _refresh_midi_items(self) -> None:
        for action_id, item in self._midi_items.items():
            texts = self._midi_map.get(action_id, [])
            default = self._midi_defaults.get(action_id, [])
            item.setText(1, _labels(texts))
            item.setText(2, _labels(default))
            font = item.font(1)
            font.setBold(texts != default)
            item.setFont(1, font)
        self._filter_midi(self.ed_midi_filter.text())

    def _filter_midi(self, text: str) -> None:
        needle = str(text or "").strip().casefold()
        for i in range(self.tree_midi.topLevelItemCount()):
            top = self.tree_midi.topLevelItem(i)
            shown = 0
            for j in range(top.childCount()):
                item = top.child(j)
                hit = not needle or any(needle in item.text(c).casefold() for c in range(3))
                item.setHidden(not hit)
                shown += hit
            top.setHidden(shown == 0)

    def _show_selected_midi(self) -> None:
        action_id = self._selected_midi_action()
        has = action_id is not None and self._midi_available
        for w in (self.btn_midi_learn, self.btn_midi_clear, self.btn_midi_revert):
            w.setEnabled(has)
        action = midi.BY_ID.get(action_id) if action_id else None
        if action is not None and action.kind == midi.METER:
            self.btn_midi_learn.setEnabled(False)   # an output: nothing comes from it to learn
        if self._learn is not None and self._learn.action_id != action_id:
            self._stop_midi_learn()

    def _assign_midi(self, action_id: str | None, texts: list[str]) -> None:
        """Give ``action_id`` ``texts``; a message another action has is taken from it once the user agrees."""
        if action_id is None:
            return
        new = [t for t in texts if t not in self._midi_map.get(action_id, [])]
        taken = [(t, other) for t in new for other, other_texts in self._midi_map.items()
                 if other != action_id and t in other_texts]
        if taken:
            names = "\n".join(f"{midi.Message.parse(t).label()}: {self._action_name(o)}" for t, o in taken)
            answer = QMessageBox.question(
                self, "MIDI",
                tr("Already used:\n{uses}\n\nUse them here instead (taken from those actions)?").format(uses=names))
            if answer != QMessageBox.Yes:
                return
            for t, other in taken:
                self._midi_map[other] = [x for x in self._midi_map[other] if x != t]
        self._midi_map[action_id] = list(dict.fromkeys(texts))
        self._refresh_midi_items()

    def _action_name(self, action_id: str) -> str:
        action = midi.BY_ID.get(action_id)
        return midi.action_label(action) if action is not None else action_id

    def _revert_selected_midi(self) -> None:
        action_id = self._selected_midi_action()
        if action_id is not None:
            self._assign_midi(action_id, list(self._midi_defaults.get(action_id, [])))

    def _revert_all_midi(self) -> None:
        self._midi_map = {**self._midi_map, **{a: list(self._midi_defaults.get(a, [])) for a in self._midi_items}}
        self._refresh_midi_items()

    def _import_rekordbox_midi(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Import rekordbox CSV..."), "", tr("rekordbox MIDI mapping (*.csv)"))
        if not path:
            return
        try:
            imported, skipped = midi.import_rekordbox_csv(path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            QMessageBox.warning(self, "MIDI", tr("The file could not be read: {error}").format(error=exc))
            return
        if not imported:
            QMessageBox.information(self, "MIDI", tr("No deck 1 control in this file that Mixlyzer has."))
            return
        self._midi_map = {**{a: list(m) for a, m in imported.items()},
                          **{a: list(imported.get(a, [])) for a in self._midi_items}}
        self._refresh_midi_items()
        n = sum(len(v) for v in imported.values())
        message = tr("{count} MIDI messages for {actions} actions imported (kept with OK / Apply).").format(
            count=n, actions=len(imported))
        if skipped:
            message += "\n\n" + tr("Not taken (another deck, or a function Mixlyzer has not): {names}").format(
                names=", ".join(skipped))
        QMessageBox.information(self, "MIDI", message)

    # Learn: the next control moved or pressed
    def _toggle_midi_learn(self, on: bool) -> None:
        action_id = self._selected_midi_action()
        if on and action_id is not None and self._midi_available:
            self._set_learn(_Learning(action_id))
        else:
            self._stop_midi_learn()

    def _stop_midi_learn(self) -> None:
        if getattr(self, "_learn", None) is None and not self.btn_midi_learn.isChecked():
            return
        self._set_learn(None)

    def _set_learn(self, learn: _Learning | None) -> None:
        """Learn on (listening, or a CC waiting: its timer started) or off; the button, its line and the engine's
        report of every message follow."""
        was = self._learn
        self._learn = learn
        if learn is None or learn.wait is None:
            self._midi_learn_timer.stop()
        elif was is None or was.wait is None:   # a wait begun (what it waits for coming keeps its time)
            self._midi_learn_timer.start()
        self.lbl_midi_learn.setText(tr("Move or press a control ...") if learn is not None else "")
        self.btn_midi_learn.blockSignals(True)
        self.btn_midi_learn.setChecked(learn is not None)
        self.btn_midi_learn.blockSignals(False)
        if (was is None) != (learn is None) or learn is None:
            if self._midi_learn_setter is not None:
                self._midi_learn_setter(learn is not None)

    def _learn_kind(self) -> str:
        action = midi.BY_ID.get(self._learn.action_id if self._learn is not None else "")
        return action.kind if action is not None else midi.BUTTON

    def _on_midi_events(self, events) -> None:
        if self._learn is None:
            return
        kind = self._learn_kind()
        for status, d1, d2, _flags, _action, _steps in events:
            is_cc = status & 0xF0 == 0xB0
            steps = midi.relative_steps(d2) if is_cc else 0   # as a knob's (not bound yet: told apart by the value)
            is_note = status & 0xF0 == 0x90
            learn = self._learn
            if learn.wait is not None:
                if learn.wait is _Wait.FADER_LSB and is_cc and status == learn.status and d1 == learn.d1 + 32:
                    self._set_learn(dataclasses.replace(learn, seen=True))   # its LSB: 14 bits
                elif (learn.wait is _Wait.BUTTON_RELEASE and is_cc and status == learn.status and d1 == learn.d1
                      and d2 == 0):
                    self._set_learn(dataclasses.replace(learn, seen=True))   # let go: a button
                continue
            if kind == midi.FADER:
                if status & 0xF0 == 0xE0:   # a pitch bend: 14 bits of its own
                    self._learned(midi.Message(status, 0))
                    return
                if is_cc:
                    if d1 < 32:
                        self._wait_learn(_Wait.FADER_LSB, status, d1)
                        continue   # its LSB may come in this report too
                    self._learned(midi.Message(status, d1))
                    return
                continue
            if kind == midi.KNOB:
                if is_cc:
                    self._learned(midi.Message(status, d1))
                    return
                continue
            if not (is_cc or is_note) or d2 == 0:
                continue   # a key let go: the press is what is learned
            if kind == midi.BUTTON and is_cc and d2 == 127:
                self._wait_learn(_Wait.BUTTON_RELEASE, status, d1)   # a button, or a knob's step back (two's complement)
                continue   # its let go may come in this report too
            if kind == midi.BUTTON and is_cc and d2 != 127 and steps:
                # a knob turned: this action on that direction only
                self._learned(midi.Message(status, d1, direction=1 if steps > 0 else -1))
                return
            self._learned(midi.Message(status, d1))
            return

    def _wait_learn(self, wait: _Wait, status: int, d1: int) -> None:
        self._set_learn(_Learning(self._learn.action_id, wait, status, d1))

    def _finish_pending_learn(self) -> None:
        learn = self._learn
        if learn is None or learn.wait is None:
            return
        if learn.wait is _Wait.FADER_LSB:
            self._learned(midi.Message(learn.status, learn.d1, hires=learn.seen))
        elif learn.seen:
            self._learned(midi.Message(learn.status, learn.d1))   # 127 then 0: a button
        else:
            self._learned(midi.Message(learn.status, learn.d1, direction=-1))   # 127 alone: a knob's step back

    def _learned(self, message: midi.Message) -> None:
        action_id = self._learn.action_id if self._learn is not None else None
        self._stop_midi_learn()
        if action_id is None:
            return
        texts = list(self._midi_map.get(action_id, []))
        if message.text() not in texts:
            self._assign_midi(action_id, texts + [message.text()])
