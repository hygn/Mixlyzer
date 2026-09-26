from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class RekordboxSyncProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sync Rekordbox XML")
        self.setModal(False)
        self._running = True

        self.lbl_status = QLabel("Waiting for Rekordbox XML worker")
        self.lbl_status.setWordWrap(True)
        self.bar_progress = QProgressBar()
        self.bar_progress.setRange(0, 100)
        self.btn_close = QPushButton("Close")
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

    def set_progress(self, value: int, text: str) -> None:
        self.bar_progress.setValue(max(0, min(100, int(value))))
        self.lbl_status.setText(str(text or "Syncing Rekordbox XML"))

    def set_finished(self, result) -> None:
        self._running = False
        self.bar_progress.setValue(100)
        summary = f"Done — {result.entry_count} XML entries"
        if result.reused_count:
            summary += f", {result.reused_count} unchanged entries reused"
        self.lbl_status.setText(f"{summary}\n{result.output_path}")
        self.btn_close.setEnabled(True)

    def set_failed(self, message: str) -> None:
        self._running = False
        self.lbl_status.setText(f"Rekordbox XML sync failed:\n{message}")
        self.btn_close.setEnabled(True)
