from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)
from system.i18n import tr, tr_n


class LibrarySyncProgressDialog(QDialog):
    """Settings > Library's Sync Now as it runs (core.workflows.library_sync), as the Rekordbox XML sync's."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Sync Library Database"))
        self.setModal(False)
        self._running = True

        self.lbl_status = QLabel(tr("Reading the stored analyses"))
        self.lbl_status.setWordWrap(True)
        self.bar_progress = QProgressBar()
        self.bar_progress.setRange(0, 100)
        self.btn_close = QPushButton(tr("Close"))
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.bar_progress)
        layout.addWidget(self.btn_close, alignment=Qt.AlignRight)
        self.resize(520, 125)

    def closeEvent(self, event) -> None:
        if self._running:
            event.ignore()
            return
        super().closeEvent(event)

    def set_progress(self, done: int, total: int) -> None:
        self.bar_progress.setValue(int(100 * done / total) if total else 0)
        self.lbl_status.setText(tr("{done} / {total} tracks").format(done=f"{done:,}", total=f"{total:,}"))

    def set_finished(self, result) -> None:
        self._running = False
        self.bar_progress.setValue(100)
        text = tr_n("Done: {n} track changed.", "Done: {n} tracks changed.", result.changed).format(
            n=f"{result.changed:,}")
        if result.unread:
            text += "\n" + tr_n("{n} track has no stored analysis that reads: left as it is.",
                                "{n} tracks have no stored analysis that reads: left as they are.",
                                result.unread).format(n=f"{result.unread:,}")
        self.lbl_status.setText(text)
        self.btn_close.setEnabled(True)

    def set_failed(self, message: str) -> None:
        self._running = False
        self.lbl_status.setText(tr("The library database could not be synced:\n{error}").format(error=message))
        self.btn_close.setEnabled(True)
