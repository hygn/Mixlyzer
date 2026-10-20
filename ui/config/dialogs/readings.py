from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class ReadingRefreshProgressDialog(QDialog):
    """Progress of the library's re-read (Settings > Search); closing it leaves the run going."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Re-read Library")
        self.setModal(False)

        self.lbl_status = QLabel("Starting the reading worker")
        self.lbl_status.setWordWrap(True)
        self.bar_progress = QProgressBar()
        self.bar_progress.setRange(0, 100)
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close this window; the re-read goes on in the background.")
        self.btn_close.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.bar_progress)
        layout.addWidget(self.btn_close, alignment=Qt.AlignRight)
        self.resize(520, 125)

    def set_started(self) -> None:
        self.bar_progress.setValue(0)
        self.lbl_status.setText("Starting the reading worker")

    def set_progress(self, value: int, text: str) -> None:
        self.bar_progress.setValue(max(0, min(100, int(value))))
        self.lbl_status.setText(str(text or "Reading the library"))

    def set_finished(self, result) -> None:
        self.bar_progress.setValue(100)
        self.lbl_status.setText(
            f"Done: {result.tracks} tracks read, the readings of {result.changed_fields} "
            "titles / artists / albums changed")

    def set_failed(self, message: str) -> None:
        self.lbl_status.setText(f"Re-reading the library failed:\n{message}")
