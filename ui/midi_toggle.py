"""The toolbar's MIDI button: the MIDI controller's input on or off (Settings > MIDI's port kept either way). Its
look is the input as the native engine reports it (DeckReport.midi_state): lit while open, outlined while asked
for but closed (no port picked, External Sync following another deck), outlined in the danger colour when the port
could not be opened (in use by another program, or disconnected), dim when switched off. A click is a request
(``toggled_by_hand``); the window keeps the setting and opens or closes the port. Disabled on the Python engine (no
MIDI)."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from foundation.i18n import tr
from ui.theme import current_theme


class MidiToggle(QtWidgets.QToolButton):
    toggled_by_hand = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setText("MIDI")
        f = self.font()
        f.setPointSize(8)
        f.setBold(True)
        self.setFont(f)
        self.setFixedHeight(20)
        self.setMinimumWidth(44)
        self.setProperty("look", "off")
        theme = current_theme()
        self.setStyleSheet(
            f"QToolButton {{ border: 1px solid {theme.border}; background: {theme.raised}; color: {theme.text_dim};"
            f" border-radius: 3px; padding: 0 6px; }}\n"
            f"QToolButton:hover {{ background: {theme.raised_hover}; color: {theme.text}; }}\n"
            f"QToolButton[look=open] {{ background: {theme.accent}; border-color: {theme.accent};"
            f" color: {theme.on_accent}; }}\n"
            f"QToolButton[look=closed] {{ border-color: {theme.accent}; color: {theme.text}; }}\n"
            f"QToolButton[look=failed] {{ border-color: {theme.danger}; color: {theme.danger}; }}\n"
            f"QToolButton:disabled {{ color: {theme.text_disabled}; border-color: {theme.border}; }}")
        self.clicked.connect(self.toggled_by_hand.emit)
        self._shown: tuple | None = None
        self.show_state(False, "", 0)

    def show_state(self, enabled: bool, port: str, state: int) -> None:
        """``enabled``: the input switched on; ``port``: Settings > MIDI's ("" none); ``state``: the engine's
        (0 closed, 1 open, 2 not opened)."""
        if (enabled, port, state) == self._shown:
            return
        self._shown = (enabled, port, state)
        if not enabled:
            look = "off"
            tip = tr("MIDI input off: the controller is not read. Click: on.")
        elif not port:
            look = "closed"
            tip = tr("MIDI input on, no port picked (Settings > MIDI).")
        elif state == 1:
            look = "open"
            tip = tr("MIDI input on: {port}. Click: off.").format(port=port)
        elif state == 2:
            look = "failed"
            tip = tr("MIDI input on, {port} could not be opened (in use by another program, or disconnected). "
                     "Click: off.").format(port=port)
        else:
            look = "closed"
            tip = tr("MIDI input on, {port} closed (External Sync follows another deck). Click: off.").format(port=port)
        if self.property("look") != look:
            self.setProperty("look", look)
            self.style().unpolish(self)
            self.style().polish(self)
        self.setToolTip(tip)
