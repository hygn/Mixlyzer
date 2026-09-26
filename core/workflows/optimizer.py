from __future__ import annotations

from PySide6 import QtCore

from core.workers.optimizer import OptimizerWorker


class OptimizerWorkflow(QtCore.QObject):
    """Own one optimizer worker and expose its lifecycle to the UI."""

    featureProgress = QtCore.Signal(int, str)
    optimizeProgress = QtCore.Signal(int, str)
    stageStarted = QtCore.Signal(int, int, str, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        process_group: str,
        jobs: list[dict[str, object]],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._process_group = str(process_group)
        self._jobs = [dict(job) for job in jobs]
        self._worker: OptimizerWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None

    def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError(f"{self._process_group} optimizer is already running")
        worker = OptimizerWorker(
            self._process_group,
            self._jobs,
            parent=self,
        )
        worker.featureProgress.connect(self.featureProgress)
        worker.optimizeProgress.connect(self.optimizeProgress)
        worker.stageStarted.connect(self.stageStarted)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    @QtCore.Slot(object)
    def _on_finished(self, result) -> None:
        self.finished.emit(result)
        self._release_worker()

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self.failed.emit(str(message))
        self._release_worker()

    def _release_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
