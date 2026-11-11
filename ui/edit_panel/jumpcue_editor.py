"""JumpCUE editor: one JumpCUE's comment, its colour squircles (the colour is its group) as the key picker's cells, its
group moved by a beat or a bar, its DEL; shown under the slot list and open until its close button (a move clicked again
and again needs no reopening). It holds which cue it edits (its slot letter) and where it is shown; the edits are its
owner's (ui.edit_panel.jumpcue), which fills it from the model."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from core.marks.colors import JUMPCUE_PAIR_COLORS
from system.i18n import tr
from ui.edit_panel.icons import badged_icon, icon
from ui.theme import current_theme
from ui.widgets.jumpcue_picker import text_on


class JumpCueEditor(QtWidgets.QFrame):
    """A child of the window over its widgets (not a popup: a click elsewhere leaves it open, and it takes no focus
    from the window's shortcuts). Shown while it edits a cue and its anchor, the squircle it opened from, is shown."""

    SQUIRCLE, TOOL_W, TOOL_H = 30, 40, 25

    comment_edited = QtCore.Signal(str)   # the comment field kept (Enter, or the focus gone): its text

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setObjectName("jumpCueEditor")
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self._letter: str | None = None            # the cue edited; None: closed
        self._anchor: QtWidgets.QWidget | None = None
        self._under: QtWidgets.QWidget | None = None  # shown below it (the slot list: none of its slots covered)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 4, 6)
        lay.setSpacing(4)

        head = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.btn_close = QtWidgets.QToolButton()
        self.btn_close.setIcon(icon("cue_editor_close"))
        self.btn_close.setIconSize(QtCore.QSize(14, 14))
        self.btn_close.setFixedSize(self.TOOL_H, self.TOOL_H)
        self.btn_close.setToolTip(tr("Close"))
        self.btn_close.clicked.connect(self.close_editor)
        self.comment = QtWidgets.QLineEdit()
        self.comment.setPlaceholderText(tr("Comment"))
        self.comment.setFixedHeight(self.TOOL_H)
        self.comment.setMinimumWidth(160)
        self.comment.editingFinished.connect(lambda: self.comment_edited.emit(self.comment.text()))
        self.comment.returnPressed.connect(self.comment.clearFocus)     # the keys back to the window's shortcuts
        self.comment.installEventFilter(self)
        self._comment_of: str | None = None          # the cue whose comment the field shows
        self._comment_shown = ""                      # its comment as kept (Esc goes back to it)
        head.addWidget(self.title)
        head.addWidget(self.comment, 1)
        head.addWidget(self.btn_close)
        lay.addLayout(head)

        colors = QtWidgets.QHBoxLayout()
        colors.setSpacing(3)
        self.color_cells: list[QtWidgets.QToolButton] = []
        cell_font = QtGui.QFont(font)
        cell_font.setPointSize(8)
        for _rgb in JUMPCUE_PAIR_COLORS:
            cell = QtWidgets.QToolButton()
            cell.setFixedSize(self.SQUIRCLE, self.SQUIRCLE)
            cell.setFont(cell_font)
            colors.addWidget(cell)
            self.color_cells.append(cell)
        colors.addStretch(1)
        lay.addLayout(colors)

        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(2)

        def tool(name: str, badge: str | None = None) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setIcon(badged_icon(name, badge) if badge else icon(name))
            btn.setIconSize(QtCore.QSize(19, 19))
            btn.setFixedSize(self.TOOL_W, self.TOOL_H)
            return btn

        self.btn_bar_earlier = tool("beatgrid_shift_fine_back", "step_bar")
        self.btn_beat_earlier = tool("beatgrid_shift_fine_back", "step_beat")
        self.btn_beat_later = tool("beatgrid_shift_fine_fwd", "step_beat")
        self.btn_bar_later = tool("beatgrid_shift_fine_fwd", "step_bar")
        self.btn_bar_earlier.setToolTip(tr("This color's JumpCUEs one bar earlier"))
        self.btn_beat_earlier.setToolTip(tr("This color's JumpCUEs one beat earlier"))
        self.btn_beat_later.setToolTip(tr("This color's JumpCUEs one beat later"))
        self.btn_bar_later.setToolTip(tr("This color's JumpCUEs one bar later"))
        self.btn_delete = QtWidgets.QToolButton()
        self.btn_delete.setText("DEL")
        self.btn_delete.setFixedSize(self.TOOL_W, self.TOOL_H)
        bold = self.btn_delete.font()
        bold.setPointSize(8)
        bold.setBold(True)
        self.btn_delete.setFont(bold)
        for btn in (self.btn_bar_earlier, self.btn_beat_earlier, self.btn_beat_later, self.btn_bar_later):
            tools.addWidget(btn)
        tools.addSpacing(8)
        tools.addWidget(self.btn_delete)
        tools.addStretch(1)
        lay.addLayout(tools)
        self.edit_buttons = [self.comment, *self.color_cells, self.btn_bar_earlier, self.btn_beat_earlier,
                             self.btn_beat_later, self.btn_bar_later, self.btn_delete]
        self.hide()

    # ---- which cue, where ----

    @property
    def letter(self) -> str | None:
        """The slot letter of the cue edited (None: closed)."""
        return self._letter

    def open_for(self, anchor: QtWidgets.QWidget, letter: str, under: QtWidgets.QWidget | None = None) -> None:
        """Edit the cue in slot ``letter``, shown at ``anchor`` (its colour squircle), below ``under`` (the slot list,
        so that every slot stays in sight and clickable) or else below the anchor, until closed."""
        self._letter = letter
        self._under = under
        self.set_anchor(anchor)

    def follow(self, letter: str) -> None:
        """The cue edited moved to slot ``letter`` (its slot changed, the cue the same)."""
        if self._letter is not None:
            self._letter = letter

    def close_editor(self) -> None:
        self._letter = None
        self.hide()

    def set_anchor(self, anchor: QtWidgets.QWidget) -> None:
        """Shown under ``anchor`` (in its window, over the window's widgets)."""
        if anchor is not self._anchor:
            if self._anchor is not None:
                self._anchor.removeEventFilter(self)
            self._anchor = anchor
            anchor.installEventFilter(self)
        window = anchor.window()
        if self.parentWidget() is not window:
            if self.parentWidget() is not None:
                self.parentWidget().removeEventFilter(self)
            self.setParent(window)
            window.installEventFilter(self)
        self._sync()

    def _sync(self) -> None:
        """Shown while open and its anchor is (the CUE page's JumpCUE list in sight), placed under it."""
        if self._letter is None or self._anchor is None or not self._anchor.isVisible():
            self.hide()
            return
        self.ensurePolished()
        self.layout().activate()                      # the texts just set counted in its size
        self.resize(self.sizeHint())
        window = self.parentWidget()
        under = self._under if self._under is not None and self._under.isVisible() else self._anchor
        x = self._anchor.mapTo(window, QtCore.QPoint(0, 0)).x()
        x = min(max(0, x), max(0, window.width() - self.width()))
        y = under.mapTo(window, QtCore.QPoint(0, under.height() + 2)).y()
        if y + self.height() > window.height():      # no room below: above
            y = max(0, under.mapTo(window, QtCore.QPoint(0, 0)).y() - self.height() - 2)
        self.move(x, y)
        self.raise_()
        self.show()

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if obj is self.comment and kind in (QtCore.QEvent.ShortcutOverride, QtCore.QEvent.KeyPress) \
                and event.key() == QtCore.Qt.Key_Escape:
            event.accept()                       # Esc cancels the comment typed, not the window's shortcut
            if kind == QtCore.QEvent.KeyPress:
                self.comment.setText(self._comment_shown)
                self.comment.clearFocus()
            return kind == QtCore.QEvent.KeyPress
        if (obj is self._anchor and kind in (QtCore.QEvent.Show, QtCore.QEvent.Hide, QtCore.QEvent.Move,
                                             QtCore.QEvent.Resize)) \
                or (obj is self.parentWidget() and kind == QtCore.QEvent.Resize):
            # after the layout has placed the anchor
            QtCore.QTimer.singleShot(0, self._sync)
        return False

    # ---- what it shows ----

    def show_cue(self, title: str, comment: str, colors: list[tuple[tuple[int, int, int], str]], current: int,
                 editable: bool) -> None:
        """The cue edited (``title``: its letter and time; ``comment``: its own, "" when it is just the letter),
        each colour with the letters of its cues (``current``: the cue's colour, ringed), its edits on while
        ``editable``. A comment being typed is left as it is, unless the editor turned to another cue."""
        self.title.setText(title)
        self._comment_shown = comment
        if not self.comment.hasFocus() or self._comment_of != self._letter:
            self.comment.setText(comment)
        self._comment_of = self._letter
        theme = current_theme()
        radius = max(2, int(self.SQUIRCLE * 0.3))
        for ci, (cell, (rgb, letters)) in enumerate(zip(self.color_cells, colors)):
            bg, fg = "rgb({}, {}, {})".format(*rgb), text_on(rgb)
            ring = fg if ci == current else bg
            cell.setText(letters)
            cell.setToolTip(tr("Cues of one color jump between each other: {letters}").format(letters=" ".join(letters))
                            if letters else tr("(unused)"))
            cell.setStyleSheet(
                f"QToolButton {{ background-color: {bg}; color: {fg}; border: 2px solid {ring};"
                f" border-radius: {radius}px; padding: 0; }}"
                f"QToolButton:hover {{ border-color: {fg}; }}"
                f"QToolButton:disabled {{ background-color: {theme.window}; color: {theme.text_disabled};"
                f" border-color: {bg}; }}")
        for btn in self.edit_buttons:
            btn.setEnabled(editable)
        self._sync()
