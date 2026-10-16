"""CUE page widgets, laid out like rekordbox's cue list: a side strip (list tabs,
pages) and 4 x 2 cue slots, each a colour squircle | name | SET / DEL | action
(JumpCUEs: SET / JUMP in one button)."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ui.edit_panel.icons import icon
from ui.theme import current_theme

SLOT_ROWS = 4
SLOT_COLS = 2
SLOTS_PER_PAGE = SLOT_ROWS * SLOT_COLS


def color_swatch_icon(rgb: tuple[int, int, int], size: int = 14) -> QtGui.QIcon:
    """A squircle in ``rgb`` (for colour menus)."""
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(*rgb))
    p.drawRoundedRect(QtCore.QRectF(0.5, 0.5, size - 1, size - 1), size * 0.3, size * 0.3)
    p.end()
    return QtGui.QIcon(pix)


def slot_letter(index: int) -> str:
    """Letter of the ``index``-th cue: A..Z, then AA, AB, ..."""
    out = ""
    n = int(index) + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


class _NameLabel(QtWidgets.QLabel):
    double_clicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        self.double_clicked.emit()


class CueSlot(QtWidgets.QFrame):
    """One cue slot: colour squircle | letter | name | SET / DEL | action (JUMP / CALL).
    With ``set_in_action`` there is no SET / DEL button: the action button reads SET
    on an empty slot (``action_clicked`` either way). Double-clicking the name edits
    it (Enter keeps it, Esc cancels): ``renamed(str)``."""

    squircle_clicked = QtCore.Signal()
    set_del_clicked = QtCore.Signal()
    action_clicked = QtCore.Signal()
    renamed = QtCore.Signal(str)

    def __init__(self, action_text: str, height: int, parent=None, set_in_action: bool = False):
        super().__init__(parent)
        self._action_text = action_text
        self._set_in_action = bool(set_in_action)
        self.setObjectName("cueSlot")
        self.setFixedHeight(height)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 1, 2, 1)
        lay.setSpacing(4)
        side = height - 6
        self.squircle = QtWidgets.QToolButton()
        self.squircle.setFixedSize(side, side)
        self.squircle.setCursor(QtCore.Qt.PointingHandCursor)
        font = self.font()
        font.setPointSize(9)
        self.letter = QtWidgets.QLabel()
        bold = QtGui.QFont(font)
        bold.setBold(True)
        self.letter.setFont(bold)
        self.letter.setFixedWidth(QtGui.QFontMetrics(bold).horizontalAdvance("WW"))
        self.name = _NameLabel()
        self.name.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.name.setFont(font)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setFont(font)
        self.name_edit.setFixedHeight(height - 4)
        self.name_edit.setStyleSheet("QLineEdit { padding: 0 2px; }")
        self.name_stack = QtWidgets.QStackedWidget()
        self.name_stack.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.name_stack.addWidget(self.name)
        self.name_stack.addWidget(self.name_edit)
        self.name.double_clicked.connect(self._start_rename)
        self.name_edit.editingFinished.connect(self._finish_rename)
        self.name_edit.installEventFilter(self)
        self._raw_name = ""
        self.btn_set = QtWidgets.QToolButton()
        self.btn_action = QtWidgets.QToolButton()
        self.btn_action.setText(action_text)
        action_w = 34 + 4 + 40 if self._set_in_action else 40  # also takes the SET / DEL place
        for b, w in ((self.btn_set, 34), (self.btn_action, action_w)):
            b.setFixedSize(w, height - 4)
            f = b.font()
            f.setPointSize(8)
            f.setBold(True)
            b.setFont(f)
            b.setStyleSheet("QToolButton { padding: 0; }")
        lay.addWidget(self.squircle)
        lay.addWidget(self.letter)
        lay.addWidget(self.name_stack, 1)
        lay.addWidget(self.btn_set)
        lay.addWidget(self.btn_action)
        self.btn_set.setVisible(not self._set_in_action)
        self.squircle.clicked.connect(self.squircle_clicked)
        self.btn_set.clicked.connect(self.set_del_clicked)
        self.btn_action.clicked.connect(self.action_clicked)
        theme = current_theme()
        self.setStyleSheet(
            f"QFrame#cueSlot {{ background: {theme.base}; border: 1px solid {theme.border}; border-radius: 2px; }}"
        )
        self.show_cue(None)

    def show_cue(self, rgb: tuple[int, int, int] | None, letter: str = "", name: str = "", detail: str = "") -> None:
        """A cue (its colour, slot letter, name and a dim detail such as the time,
        shown when it has no name), or an empty slot."""
        theme = current_theme()
        self._cancel_rename()
        self._raw_name = name
        filled = rgb is not None
        side = self.squircle.width()
        radius = max(2, int(side * 0.3))
        if filled:
            color = "rgb({}, {}, {})".format(*rgb)
            self.squircle.setStyleSheet(
                f"QToolButton {{ background: {color}; border: 1px solid {color}; border-radius: {radius}px; }}"
                f"QToolButton:hover {{ border-color: {theme.text}; }}"
            )
        else:
            self.squircle.setStyleSheet(
                f"QToolButton {{ background: transparent; border: 1px solid {theme.border}; border-radius: {radius}px; }}"
            )
        self.squircle.setEnabled(filled)
        self.letter.setText(letter if filled else "")
        if name:
            self.name.setText(name)
            self.name.setStyleSheet("")
        else:
            self.name.setText(detail)
            self.name.setStyleSheet(f"color: {theme.text_dim};")
        self.name.setToolTip("Double-click to rename" if filled else "")
        self.btn_set.setText("DEL" if filled else "SET")
        if self._set_in_action:
            self.btn_action.setText(self._action_text if filled else "SET")
        else:
            self.btn_action.setEnabled(filled)
        self._filled = filled

    # ---- renaming ----

    def _start_rename(self) -> None:
        if not getattr(self, "_filled", False):
            return
        self.name_edit.setText(self._raw_name)
        self.name_stack.setCurrentWidget(self.name_edit)
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _finish_rename(self) -> None:
        if self.name_stack.currentWidget() is not self.name_edit:
            return
        self.name_stack.setCurrentWidget(self.name)
        text = self.name_edit.text().strip()
        if text != self._raw_name:
            self.renamed.emit(text)

    def _cancel_rename(self) -> None:
        self.name_stack.setCurrentWidget(self.name)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.name_edit and event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
            self._cancel_rename()
            return True
        return super().eventFilter(obj, event)

    def set_action_lit(self, lit: bool) -> None:
        theme = current_theme()
        self.btn_action.setStyleSheet(
            f"QToolButton {{ padding: 0; background: {theme.accent}; border-color: {theme.accent}; color: {theme.on_accent}; }}"
            if lit else "QToolButton { padding: 0; }"
        )


class CuePageWidget(QtWidgets.QWidget):
    """Side strip (JumpCUE / CUE tabs, page arrows and number, an extra tool button)
    and a 4 x 2 slot grid per list. ``list_changed(int)``: 0 = JumpCUE, 1 = CUE."""

    list_changed = QtCore.Signal(int)
    page_changed = QtCore.Signal()

    def __init__(self, height: int, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        gap = 3
        row_h = (height - gap * (SLOT_ROWS - 1)) // SLOT_ROWS

        side = QtWidgets.QGridLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setHorizontalSpacing(2)
        side.setVerticalSpacing(gap)
        half = 21

        def _side_btn(name: str, tip: str, w: int, checkable: bool = False) -> QtWidgets.QToolButton:
            b = QtWidgets.QToolButton()
            b.setIcon(icon(name))
            b.setIconSize(QtCore.QSize(14, 14))
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setFixedSize(w, row_h)
            return b

        self.tab_jump = _side_btn("cue_tab_jump", "JumpCUEs", half * 2 + 2, checkable=True)
        self.tab_cue = _side_btn("cue_tab_memory", "CUE points", half * 2 + 2, checkable=True)
        self.tab_jump.setChecked(True)
        tabs = QtWidgets.QButtonGroup(self)
        tabs.setExclusive(True)
        tabs.addButton(self.tab_jump, 0)
        tabs.addButton(self.tab_cue, 1)
        tabs.idClicked.connect(self._on_tab)
        self.btn_prev = _side_btn("cue_page_prev", "Previous page", half)
        self.btn_next = _side_btn("cue_page_next", "Next page", half)
        self.page_label = QtWidgets.QLabel("1")
        self.page_label.setAlignment(QtCore.Qt.AlignCenter)
        self.page_label.setFixedSize(half, row_h)
        f = self.page_label.font()
        f.setPointSize(9)
        self.page_label.setFont(f)
        self.page_label.setToolTip("Page")
        self.tool_slot = QtWidgets.QStackedWidget()  # per-list tool (JumpCUE: reanalyze, CUE: MEMORY)
        self.tool_slot.setFixedSize(half, row_h)
        side.addWidget(self.tab_jump, 0, 0, 1, 2)
        side.addWidget(self.tab_cue, 1, 0, 1, 2)
        side.addWidget(self.btn_prev, 2, 0)
        side.addWidget(self.btn_next, 2, 1)
        side.addWidget(self.page_label, 3, 0)
        side.addWidget(self.tool_slot, 3, 1)
        root.addLayout(side)

        self.lists = QtWidgets.QStackedWidget()
        self.slots: list[list[CueSlot]] = []
        for action in ("JUMP", "CALL"):
            page = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(page)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(gap)
            slots = []
            for i in range(SLOTS_PER_PAGE):
                # JumpCUEs: SET in the JUMP button, deleted from the squircle's menu
                slot = CueSlot(action, row_h, set_in_action=(action == "JUMP"))
                grid.addWidget(slot, i % SLOT_ROWS, i // SLOT_ROWS)  # A-D left, E-H right
                slots.append(slot)
            self.lists.addWidget(page)
            self.slots.append(slots)
        root.addWidget(self.lists, 1)

        self.pages = [0, 0]
        self.btn_prev.clicked.connect(lambda: self._step_page(-1))
        self.btn_next.clicked.connect(lambda: self._step_page(1))

    @property
    def current_list(self) -> int:
        return self.lists.currentIndex()

    def _on_tab(self, idx: int) -> None:
        self.lists.setCurrentIndex(idx)
        if idx < self.tool_slot.count():
            self.tool_slot.setCurrentIndex(idx)
        self.list_changed.emit(idx)
        self.page_changed.emit()

    def _step_page(self, step: int) -> None:
        i = self.current_list
        self.pages[i] = max(0, self.pages[i] + step)
        self.page_changed.emit()

    def update_pager(self, list_idx: int, n_items: int, max_items: int | None = None) -> None:
        """Keep list ``list_idx``'s page in range for ``n_items`` slots in use (one
        spare page for a new cue once the last is full, up to ``max_items`` slots);
        the shown list's page is displayed."""
        i = int(list_idx)
        n_pages = n_items // SLOTS_PER_PAGE + 1
        if max_items is not None:
            n_pages = min(n_pages, -(-int(max_items) // SLOTS_PER_PAGE))
        self.pages[i] = min(self.pages[i], n_pages - 1)
        if i != self.current_list:
            return
        self.page_label.setText(str(self.pages[i] + 1))
        self.page_label.setToolTip(f"Page {self.pages[i] + 1} of {n_pages}")
        self.btn_prev.setEnabled(self.pages[i] > 0)
        self.btn_next.setEnabled(self.pages[i] < n_pages - 1)

    def page_start(self, list_idx: int) -> int:
        return self.pages[list_idx] * SLOTS_PER_PAGE
