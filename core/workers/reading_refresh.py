"""The library's readings read again (core.reading), in a spawned process: the readers are Python
(a thread would share the interpreter lock with the GUI). The readings the user set are kept."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any

from PySide6 import QtCore

from core.concurrency.process_worker import SpawnProcessWorker, install_parent_watchdog

# tracks per transaction: the app's own writes (an analysis, Edit Track) wait at most about this many
# tracks' reading (a few ms each)
BATCH_TRACKS = 50


@dataclass(frozen=True)
class ReadingRefreshRequest:
    db_path: str
    languages: tuple[str, ...]   # the search settings' db_languages


@dataclass(frozen=True)
class ReadingRefreshResult:
    tracks: int
    changed_fields: int   # title / artist / album fields whose readings changed


def refresh_library_readings(db_path: str, languages: tuple[str, ...], progress=None) -> ReadingRefreshResult:
    """Read every track's title / artist / album again in ``languages`` (the user's readings kept);
    ``progress(done, total)`` after each batch."""
    from core import reading
    from core.library_handler import READING_FIELDS, LibraryDB

    reading.warm_up()
    db = LibraryDB(db_path, reading_languages=list(languages))
    db.connect()
    try:
        uids = [uid for (uid,) in db.conn.execute("SELECT uid FROM tracks ORDER BY added_ts, uid;")]
        changed = 0
        if progress is not None:
            progress(0, len(uids))
        for start in range(0, len(uids), BATCH_TRACKS):
            with db.tx():
                for uid in uids[start:start + BATCH_TRACKS]:
                    before = db.readings(uid)
                    db.refresh_readings(uid)
                    after = db.readings(uid)
                    changed += sum(before.get(f) != after.get(f) for f in READING_FIELDS)
            if progress is not None:
                progress(min(start + BATCH_TRACKS, len(uids)), len(uids))
        return ReadingRefreshResult(tracks=len(uids), changed_fields=changed)
    finally:
        db.close()


def _reading_refresh_process_entry(db_path: str, languages: tuple[str, ...], queue: Any) -> None:
    install_parent_watchdog()
    try:
        result = refresh_library_readings(
            db_path, languages, lambda done, total: queue.put({"type": "progress", "done": done, "total": total}))
        queue.put({"type": "result", "payload": result})
    except Exception:
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "done"})


class ReadingRefreshWorker(SpawnProcessWorker):
    """Qt-facing worker for one library re-read."""

    progress = QtCore.Signal(int, int)      # done, total tracks
    finished = QtCore.Signal(object)        # ReadingRefreshResult
    failed = QtCore.Signal(str)

    def __init__(self, request: ReadingRefreshRequest, parent: QtCore.QObject | None = None) -> None:
        self._result: ReadingRefreshResult | None = None
        self._error_emitted = False
        super().__init__(
            target=_reading_refresh_process_entry,
            args=(str(request.db_path), tuple(request.languages)),
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
            self.failed.emit(f"The reading refresh process exited unexpectedly ({exitcode}).")
            return
        self.finished.emit(self._result)
