from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)
from system.i18n import tr, tr_n


class MusicIndexProgressDialog(QDialog):
    """Progress of Index Now (Settings > Search > Search by Music); closing it leaves the run going."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Index for Search by Music, Variants and Mix Candidates"))
        self.setModal(False)

        self.lbl_status = QLabel(tr("Starting the indexing workers"))
        self.lbl_status.setWordWrap(True)
        self.bar_progress = QProgressBar()
        self.bar_progress.setRange(0, 100)
        self.btn_close = QPushButton(tr("Close"))
        self.btn_close.setToolTip(tr("Close this window; the indexing goes on in the background."))
        self.btn_close.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.bar_progress)
        layout.addWidget(self.btn_close, alignment=Qt.AlignRight)
        self.resize(520, 125)

    def set_started(self, run) -> None:
        self.bar_progress.setValue(0)
        if run.total:
            self.lbl_status.setText(tr_n("Indexing {n} track", "Indexing {n} tracks", run.total).format(n=run.total))
        else:
            self.lbl_status.setText(tr("Every library track is indexed already"))

    def set_progress(self, value: int, text: str) -> None:
        self.bar_progress.setValue(max(0, min(100, int(value))))
        self.lbl_status.setText(str(text))

    def set_finished(self, summary) -> None:
        self.bar_progress.setValue(100)
        text = tr_n("Done: {n} track indexed", "Done: {n} tracks indexed", summary.indexed).format(n=summary.indexed)
        if summary.failed:
            text += "\n" + tr_n("{n} file could not be read:", "{n} files could not be read:",
                                len(summary.failed)).format(n=len(summary.failed))
            text += "\n" + "\n".join(summary.failed[:5]) + ("\n…" if len(summary.failed) > 5 else "")
        self.lbl_status.setText(text)

    def set_failed(self, message: str) -> None:
        self.lbl_status.setText(tr("Indexing for Search by Music, Variants and Mix Candidates failed:\n{error}").format(
            error=message))
