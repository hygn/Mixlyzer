from __future__ import annotations

from PySide6 import QtCore

from core.library import jobs as library_jobs
from core.workers.optimizer import OptimizerWorker


class OptimizerWorkflow(QtCore.QObject):
    """Own one optimizer worker and expose its lifecycle to the UI. The library's owner gives the run its tracks
    (core.library.jobs.training_tracks) before the process starts: the optimizers never open library.db."""

    featureProgress = QtCore.Signal(int, str)
    optimizeProgress = QtCore.Signal(int, str)
    stageStarted = QtCore.Signal(int, int, str, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        process_group: str,
        jobs: list[dict[str, object]],
        library,
        parent: QtCore.QObject | None = None,
    ) -> None:
        """``library``: the library's owner (core.library.service)."""
        super().__init__(parent)
        self._process_group = str(process_group)
        self._jobs = [dict(job) for job in jobs]
        self._library = library
        self._starting = False
        self._worker: OptimizerWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._starting or self._worker is not None

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError(f"{self._process_group} optimizer is already running")
        self._starting = True
        self._library.read(library_jobs.training_tracks, done=self._spawn, failed=self._not_started)

    def _spawn(self, tracks: list[dict]) -> None:
        self._starting = False
        worker = OptimizerWorker(
            self._process_group,
            self._jobs,
            tracks,
            parent=self,
        )
        worker.featureProgress.connect(self.featureProgress)
        worker.optimizeProgress.connect(self.optimizeProgress)
        worker.stageStarted.connect(self.stageStarted)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _not_started(self, exc: BaseException) -> None:
        self._starting = False
        self.failed.emit(str(exc))

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
