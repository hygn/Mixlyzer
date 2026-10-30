from __future__ import annotations

import os
from datetime import datetime
from typing import Callable

from PySide6 import QtGui, QtWidgets

from system.atomic_io import atomic_write_text
from system.i18n import tr


class DiagnosticsDialog(QtWidgets.QDialog):
    """The diagnostics (app.diagnostics.report, from ``make``) as the user would send them: read here first, then
    copied or saved as a text file; the app sends nothing. Made again each time the window is shown, and by
    Refresh."""

    def __init__(self, make: Callable[[], str], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._make = make
        self.setWindowTitle(tr("Diagnostics"))
        self.resize(820, 620)
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(tr("What a report of a playback, device or decoding problem needs."))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(self.text, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        refresh = buttons.addButton(tr("Refresh"), QtWidgets.QDialogButtonBox.ActionRole)
        copy = buttons.addButton(tr("Copy"), QtWidgets.QDialogButtonBox.ActionRole)
        save = buttons.addButton(tr("Save..."), QtWidgets.QDialogButtonBox.ActionRole)
        refresh.clicked.connect(self.refresh)
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.text.toPlainText()))
        save.clicked.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        self.refresh()
        super().showEvent(event)

    def refresh(self) -> None:
        self.text.setPlainText(self._make())

    def _save(self) -> None:
        name = f"mixlyzer-diagnostics-{datetime.now():%Y%m%d-%H%M%S}.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Save Diagnostics"), os.path.join(os.path.expanduser("~"), name), tr("Text files (*.txt)"))
        if not path:
            return
        try:
            atomic_write_text(path, self.text.toPlainText())
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, tr("Save Diagnostics"),
                                          tr("The file could not be written: {error}").format(error=exc))
