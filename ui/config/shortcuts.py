"""Settings > Shortcuts: every action's keys (``ui.shortcuts.ACTIONS``), a main and an alternative one. OK / Apply
keep them as the user's (``core.controls.shortcuts.save``: only what differs from the shipped keys, never the shipped file);
Revert takes the selected action back to its shipped keys, Revert All every action."""

from __future__ import annotations

from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.controls import shortcuts
from system import os_support
from system.i18n import tr
from ui.shortcuts import ACTIONS, GROUPS
from ui.theme import current_theme

_ACTION_ROLE = QtCore.Qt.UserRole
_BY_ID = {action.id: action for action in ACTIONS}


def _native(keys: list[str]) -> str:
    """Keys as this system writes them (Ctrl / ⌘)."""
    return ", ".join(QtGui.QKeySequence(k).toString(QtGui.QKeySequence.NativeText) for k in keys)


def _portable(seq: QtGui.QKeySequence) -> str:
    """The first key combination of ``seq`` as stored ("" for none)."""
    if seq.isEmpty():
        return ""
    return QtGui.QKeySequence(seq[0]).toString(QtGui.QKeySequence.PortableText)


def build_shortcuts_tab(dialog) -> None:
    dialog.tab_shortcuts = QWidget()
    lay = QVBoxLayout(dialog.tab_shortcuts)

    intro = QLabel(tr("No modifier: changes no data · Ctrl: edits · Ctrl+Shift: deletes"))
    lay.addWidget(intro)

    dialog.ed_shortcut_filter = QLineEdit()
    dialog.ed_shortcut_filter.setPlaceholderText(tr("Find an action or a key ..."))
    dialog.ed_shortcut_filter.setClearButtonEnabled(True)
    lay.addWidget(dialog.ed_shortcut_filter)

    dialog.tree_shortcuts = QTreeWidget()
    dialog.tree_shortcuts.setColumnCount(3)
    dialog.tree_shortcuts.setHeaderLabels([tr("Action"), tr("Shortcut"), tr("Default")])
    dialog.tree_shortcuts.setRootIsDecorated(False)
    dialog.tree_shortcuts.setUniformRowHeights(True)
    header = dialog.tree_shortcuts.header()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    for column in (1, 2):
        header.setSectionResizeMode(column, QHeaderView.Interactive)
        dialog.tree_shortcuts.setColumnWidth(column, 170)
    bold = QtGui.QFont(dialog.tree_shortcuts.font())
    bold.setBold(True)
    dialog._shortcut_items = {}   # action id -> its row
    for group in GROUPS:
        top = QTreeWidgetItem([tr(group) if group not in ("Key", "CUE") else group])
        top.setFont(0, bold)
        top.setFlags(QtCore.Qt.ItemIsEnabled)
        dialog.tree_shortcuts.addTopLevelItem(top)
        for action in (a for a in ACTIONS if a.group == group):
            item = QTreeWidgetItem([action.label(), "", ""])
            item.setData(0, _ACTION_ROLE, action.id)
            item.setToolTip(0, action.tooltip())
            top.addChild(item)
            dialog._shortcut_items[action.id] = item
        top.setExpanded(True)
    lay.addWidget(dialog.tree_shortcuts, 1)

    row = QHBoxLayout()
    dialog.kse_shortcut_main = QKeySequenceEdit()
    dialog.kse_shortcut_alt = QKeySequenceEdit()
    for kse, tip in ((dialog.kse_shortcut_main, tr("Click, then press the keys (one combination).")),
                     (dialog.kse_shortcut_alt, tr("A second combination doing the same."))):
        kse.setMaximumSequenceLength(1)
        kse.setClearButtonEnabled(True)
        kse.setToolTip(tip)
    dialog.btn_shortcut_revert = QPushButton(tr("Revert"))
    dialog.btn_shortcut_revert.setToolTip(tr("The selected action's shipped shortcut again."))
    dialog.btn_shortcut_revert_all = QPushButton(tr("Revert All"))
    dialog.btn_shortcut_revert_all.setToolTip(tr("Every action's shipped shortcut again."))
    row.addWidget(QLabel(tr("Shortcut")))
    row.addWidget(dialog.kse_shortcut_main, 1)
    row.addWidget(QLabel(tr("Alternative")))
    row.addWidget(dialog.kse_shortcut_alt, 1)
    lay.addLayout(row)
    buttons = QHBoxLayout()
    buttons.addStretch(1)
    buttons.addWidget(dialog.btn_shortcut_revert)
    buttons.addWidget(dialog.btn_shortcut_revert_all)
    lay.addLayout(buttons)

    dialog.tabs.addTab(dialog.tab_shortcuts, tr("Shortcuts"))


class ShortcutSettingsMixin:
    """The keys being edited (``_shortcut_keys``: action -> key sequences) until OK / Apply keeps them."""

    def _initialize_shortcut_settings(self) -> None:
        self._shortcut_keys: dict[str, list[str]] = {}
        self._shortcut_defaults: dict[str, list[str]] = {}
        self.tree_shortcuts.currentItemChanged.connect(lambda *_: self._show_selected_shortcut())
        self.ed_shortcut_filter.textChanged.connect(self._filter_shortcuts)
        for slot, kse in enumerate((self.kse_shortcut_main, self.kse_shortcut_alt)):
            # after the key event that set it (a question box may open)
            kse.keySequenceChanged.connect(
                lambda _seq, slot=slot: QtCore.QTimer.singleShot(0, lambda: self._on_shortcut_edited(slot)))
        self.btn_shortcut_revert.clicked.connect(self._revert_selected_shortcut)
        self.btn_shortcut_revert_all.clicked.connect(self._revert_all_shortcuts)
        self._load_shortcut_settings()

    def _load_shortcut_settings(self) -> None:
        """The shortcuts in use (the files), dropping edits not kept."""
        self._shortcut_defaults = shortcuts.defaults()
        in_use = shortcuts.keymap()
        self._shortcut_keys = {action.id: list(in_use.get(action.id, [])) for action in ACTIONS}
        self._refresh_shortcut_items()
        self._show_selected_shortcut()

    def _save_shortcut_settings(self) -> None:
        shortcuts.save(self._shortcut_keys)
        self._bus.sig_shortcuts_changed.emit(shortcuts.keymap())

    def _selected_shortcut_action(self) -> str | None:
        item = self.tree_shortcuts.currentItem()
        return item.data(0, _ACTION_ROLE) if item is not None else None

    def _refresh_shortcut_items(self) -> None:
        """Each row's keys; changed ones in bold, a key used by another action or held by another program in the
        warning colour."""
        users: dict[str, list[str]] = {}
        for action_id, keys in self._shortcut_keys.items():
            for k in keys:
                users.setdefault(k, []).append(action_id)
        held = {k for k in users if os_support.hotkey_taken(k)}
        warning = QtGui.QBrush(QtGui.QColor(current_theme().danger))
        for action_id, item in self._shortcut_items.items():
            keys = self._shortcut_keys.get(action_id, [])
            default = self._shortcut_defaults.get(action_id, [])
            item.setText(1, _native(keys))
            item.setText(2, _native(default))
            font = item.font(1)
            font.setBold(keys != default)
            item.setFont(1, font)
            others = [_BY_ID[a].label() for k in keys for a in users.get(k, []) if a != action_id]
            taken = [k for k in keys if k in held]
            notes = []
            if others:
                notes.append(tr("Also used by: {actions}").format(actions=", ".join(others)))
            if taken:
                notes.append(tr("Another program on this computer takes {keys}: Mixlyzer never receives it.")
                             .format(keys=_native(taken)))
            item.setForeground(1, warning if notes else QtGui.QBrush())
            item.setToolTip(1, "".join(f"<p>{note}</p>" for note in notes))
        self._filter_shortcuts(self.ed_shortcut_filter.text())

    def _filter_shortcuts(self, text: str) -> None:
        needle = str(text or "").strip().casefold()
        for i in range(self.tree_shortcuts.topLevelItemCount()):
            top = self.tree_shortcuts.topLevelItem(i)
            shown = 0
            for j in range(top.childCount()):
                item = top.child(j)
                hit = not needle or any(needle in item.text(c).casefold() for c in range(3))
                item.setHidden(not hit)
                shown += hit
            top.setHidden(shown == 0)

    def _show_selected_shortcut(self) -> None:
        action_id = self._selected_shortcut_action()
        keys = self._shortcut_keys.get(action_id, []) if action_id else []
        for slot, kse in enumerate((self.kse_shortcut_main, self.kse_shortcut_alt)):
            kse.blockSignals(True)
            kse.setKeySequence(QtGui.QKeySequence(keys[slot]) if slot < len(keys) else QtGui.QKeySequence())
            kse.setEnabled(action_id is not None)
            kse.blockSignals(False)
        self.btn_shortcut_revert.setEnabled(action_id is not None)

    def _on_shortcut_edited(self, slot: int) -> None:
        action_id = self._selected_shortcut_action()
        if action_id is None:
            return
        keys = list(self._shortcut_keys.get(action_id, []))
        keys += [""] * (2 - len(keys))
        keys[slot] = _portable((self.kse_shortcut_main, self.kse_shortcut_alt)[slot].keySequence())
        self._assign_shortcut(action_id, list(dict.fromkeys(k for k in keys if k)))

    def _assign_shortcut(self, action_id: str, keys: list[str]) -> None:
        """Give ``action_id`` ``keys``; a key another action has is taken from it once the user agrees."""
        new = [k for k in keys if k not in self._shortcut_keys.get(action_id, [])]
        held = [k for k in new if os_support.hotkey_taken(k)]
        if held:
            answer = QMessageBox.question(
                self, tr("Shortcuts"),
                tr("Another program on this computer takes {keys} (a graphics card overlay, for one): Mixlyzer "
                   "never receives it.\n\nUse it anyway?").format(keys=_native(held)))
            if answer != QMessageBox.Yes:
                self._show_selected_shortcut()
                return
        taken = [(k, other) for k in new
                 for other, other_keys in self._shortcut_keys.items() if other != action_id and k in other_keys]
        if taken:
            names = "\n".join(f"{_native([k])}: {_BY_ID[other].label()}" for k, other in taken)
            answer = QMessageBox.question(
                self, tr("Shortcuts"),
                tr("Already used:\n{uses}\n\nUse them here instead (taken from those actions)?").format(uses=names))
            if answer != QMessageBox.Yes:
                self._show_selected_shortcut()
                return
            for k, other in taken:
                self._shortcut_keys[other] = [x for x in self._shortcut_keys[other] if x != k]
        self._shortcut_keys[action_id] = keys
        self._refresh_shortcut_items()
        self._show_selected_shortcut()

    def _revert_selected_shortcut(self) -> None:
        action_id = self._selected_shortcut_action()
        if action_id is not None:
            self._assign_shortcut(action_id, list(self._shortcut_defaults.get(action_id, [])))

    def _revert_all_shortcuts(self) -> None:
        self._shortcut_keys = {action.id: list(self._shortcut_defaults.get(action.id, [])) for action in ACTIONS}
        self._refresh_shortcut_items()
        self._show_selected_shortcut()
