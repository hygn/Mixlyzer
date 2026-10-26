from __future__ import annotations

from PySide6 import QtCore

from core.analysis_library import jobs as analysis_jobs
from core.i18n import tr
from core.workers.optimizer import OptimizerWorker


class OptimizerWorkflow(QtCore.QObject):
    """Own one optimizer worker and expose its lifecycle to the UI. The library's owner gives the run its tracks and
    what the optimizers read of their stored analyses (core.analysis_library.jobs.training_tracks, training_features)
    before the process starts: the optimizers never open the library's files."""

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
        self._library.read(analysis_jobs.training_tracks, done=self._read_features, failed=self._not_started)

    def _read_features(self, tracks: list[dict], start: int = 0) -> None:
        """The tracks' stored analyses, TRAINING_BATCH tracks a job, each asked once the last came: the owner's
        other jobs (the GUI's) run between two."""
        if start >= len(tracks):
            self._spawn(tracks)
            return
        self.featureProgress.emit(0, tr("Reading the stored analyses: {done} / {total} tracks").format(
            done=start, total=len(tracks)))
        batch = tracks[start:start + analysis_jobs.TRAINING_BATCH]

        def came(read: list) -> None:
            for track, (features, error) in zip(batch, read):
                track["features"] = features
                if error:
                    track["error"] = error
            self._read_features(tracks, start + len(batch))

        self._library.read(analysis_jobs.training_features, [t["uid"] for t in batch], done=came,
                           failed=self._not_started)

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
