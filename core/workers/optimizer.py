"""One-shot optimizer subprocess worker and optimizer registry."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any, Callable

from PySide6 import QtCore

from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog


ProgressCallback = Callable[[int, str], None]
OptimizerCallable = Callable[..., object]


@dataclass(frozen=True)
class _OptimizerDefinition:
    request_type: type
    optimize: OptimizerCallable


def _load_onset_optimizer() -> _OptimizerDefinition:
    from optimizer.onset_parameter_optimizer import OnsetOptimizationRequest, optimize_onset_parameters
    return _OptimizerDefinition(OnsetOptimizationRequest, optimize_onset_parameters)


def _load_beat_phase_optimizer() -> _OptimizerDefinition:
    from optimizer.beat_phase_parameter_optimizer import BeatPhaseOptimizationRequest, optimize_beat_phase_parameters
    return _OptimizerDefinition(BeatPhaseOptimizationRequest, optimize_beat_phase_parameters)


def _load_downbeat_optimizer() -> _OptimizerDefinition:
    from optimizer.downbeat_parameter_optimizer import DownbeatOptimizationRequest, optimize_downbeat_parameters
    return _OptimizerDefinition(DownbeatOptimizationRequest, optimize_downbeat_parameters)


def _load_phrase_optimizer() -> _OptimizerDefinition:
    from optimizer.phrase_parameter_optimizer import PhraseOptimizationRequest, optimize_phrase_parameters
    return _OptimizerDefinition(PhraseOptimizationRequest, optimize_phrase_parameters)


# New optimizers only need a loader here and a UI job using the corresponding key.
_OPTIMIZER_LOADERS: dict[str, Callable[[], _OptimizerDefinition]] = {
    "onset": _load_onset_optimizer,
    "beat_phase": _load_beat_phase_optimizer,
    "downbeat": _load_downbeat_optimizer,
    "phrase": _load_phrase_optimizer,
}


def _optimizer_process_entry(
    process_group: str,
    jobs: tuple[dict[str, object], ...],
    queue: Any,
) -> None:
    """Execute all jobs for one optimizer run inside one child process."""
    install_parent_watchdog()
    loaded: dict[str, _OptimizerDefinition] = {}
    try:
        total = len(jobs)
        for index, job in enumerate(jobs, start=1):
            optimizer_name = str(job["optimizer_name"])
            display_name = str(job.get("display_name") or optimizer_name)
            queue.put(
                {
                    "type": "stage_started",
                    "optimizer_name": optimizer_name,
                    "display_name": display_name,
                    "index": index,
                    "total": total,
                }
            )

            def feature_progress(value: int, text: str) -> None:
                queue.put({"type": "feature_progress", "value": int(value), "text": str(text)})

            def optimize_progress(value: int, text: str) -> None:
                queue.put({"type": "optimize_progress", "value": int(value), "text": str(text)})

            loader = _OPTIMIZER_LOADERS.get(optimizer_name)
            if loader is None:
                raise ValueError(f"Unsupported parameter optimizer: {optimizer_name}")
            definition = loaded.get(optimizer_name)
            if definition is None:
                definition = loader()
                loaded[optimizer_name] = definition
            request = definition.request_type(**dict(job["request_values"]))
            result = definition.optimize(
                request,
                feature_progress=feature_progress,
                optimize_progress=optimize_progress,
            )
            queue.put({"type": "job_result", "optimizer_name": optimizer_name, "payload": result})
    except Exception:
        queue.put({"type": "error", "process_group": process_group, "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "done", "process_group": process_group})


class OptimizerWorker(SpawnProcessWorker):
    """Qt-facing worker for a single Beat or Phrase optimizer process."""

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
        self._process_group = str(process_group)
        self._jobs = tuple(dict(job) for job in jobs)
        self._results: list[object] = []
        self._error_emitted = False
        super().__init__(
            target=_optimizer_process_entry,
            args=(self._process_group, self._jobs),
            # Optimizers create ProcessPoolExecutor children for feature caches.
            daemon=False,
            process_name=f"Mixlyzer-{self._process_group}-optimizer",
            parent=parent,
        )

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "feature_progress":
            self.featureProgress.emit(int(message["value"]), str(message["text"]))
        elif message_type == "optimize_progress":
            self.optimizeProgress.emit(int(message["value"]), str(message["text"]))
        elif message_type == "stage_started":
            self.stageStarted.emit(
                int(message["index"]),
                int(message["total"]),
                str(message["optimizer_name"]),
                str(message["display_name"]),
            )
        elif message_type == "job_result":
            self._results.append(message.get("payload"))
        elif message_type == "error" and not self._error_emitted:
            self._error_emitted = True
            self.failed.emit(str(message.get("traceback", "Unknown optimizer error")))

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        if not graceful and not self._error_emitted:
            self._error_emitted = True
            self.failed.emit(f"{self._process_group} optimizer process exited unexpectedly ({exitcode}).")
            return
        if self._error_emitted:
            return
        if len(self._results) != len(self._jobs):
            self._error_emitted = True
            self.failed.emit(
                f"{self._process_group} optimizer returned "
                f"{len(self._results)} of {len(self._jobs)} result(s)."
            )
            return
        payload: object = self._results[0] if len(self._results) == 1 else tuple(self._results)
        self.finished.emit(payload)
