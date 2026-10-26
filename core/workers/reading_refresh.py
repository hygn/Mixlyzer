"""The library's readings read again (core.reading), in a spawned process: the readers are Python (a thread would
share the interpreter lock with the GUI). The process only reads texts: the library's owner gives it the library's
texts (core.library.jobs.reading_texts) and writes what it read (jobs.apply_readings, jobs.apply_work_readings), the
readings the user set kept: core.workflows.reading_refresh runs it."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any

from PySide6 import QtCore

from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog
from core.i18n import tr

# tracks per write job: the owner's other jobs (the GUI's reads, an analysis stored) wait at most about this many
# tracks' writing
BATCH_TRACKS = 50
# texts read between two progress reports
PROGRESS_TEXTS = 200


@dataclass(frozen=True)
class ReadingRefreshRequest:
    texts: tuple[str, ...]       # every text to read, each once (jobs.reading_texts)
    languages: tuple[str, ...]   # the search settings' db_languages
    han_run: int                 # its han_run()
    tracks: int                  # the tracks they are of (the progress is counted in tracks)


@dataclass(frozen=True)
class ReadingRefreshResult:
    tracks: int
    changed_fields: int   # title / artist / album fields whose readings changed


def read_texts(texts, languages, han_run: int, progress=None) -> dict[str, list[tuple[str, str]]]:
    """Each text's readings (core.reading.readings, in ``languages``): the process's work. ``progress(done,
    total)`` in texts."""
    from core import reading
    reading.warm_up()
    read = {}
    for i, text in enumerate(texts):
        if progress is not None and i % PROGRESS_TEXTS == 0:
            progress(i, len(texts))
        read[text] = reading.readings(text, list(languages), han_run=han_run)
    if progress is not None:
        progress(len(texts), len(texts))
    return read


def _reading_refresh_process_entry(request: ReadingRefreshRequest, queue: Any) -> None:
    install_parent_watchdog()
    try:
        texts = request.texts
        read = read_texts(texts, request.languages, request.han_run, lambda done, total: queue.put(
            {"type": "progress", "done": request.tracks * done // max(total, 1), "total": request.tracks}))
        queue.put({"type": "result", "payload": read})
    except Exception:
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "done"})


class ReadingRefreshWorker(SpawnProcessWorker):
    """Qt-facing worker for one library re-read: the texts read, handed back (finished)."""

    progress = QtCore.Signal(int, int)      # done, total tracks
    finished = QtCore.Signal(object)        # {text: readings}
    failed = QtCore.Signal(str)

    def __init__(self, request: ReadingRefreshRequest, parent: QtCore.QObject | None = None) -> None:
        self._result: dict | None = None
        self._error_emitted = False
        super().__init__(
            target=_reading_refresh_process_entry,
            args=(request,),
            daemon=True,
            process_name="Mixlyzer-reading-refresh",
            parent=parent,
        )

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "progress":
            self.progress.emit(int(message["done"]), int(message["total"]))
        elif kind == "result":
            self._result = message.get("payload")
        elif kind == "error" and not self._error_emitted:
            self._error_emitted = True
            self.failed.emit(str(message.get("traceback", "Unknown reading refresh error")))

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        if self._error_emitted:
            return
        if not graceful or self._result is None:
            self._error_emitted = True
            self.failed.emit(tr("The reading refresh process exited unexpectedly ({code}).").format(code=exitcode))
            return
        self.finished.emit(self._result)
