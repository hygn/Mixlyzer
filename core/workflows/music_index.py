from __future__ import annotations

from dataclasses import dataclass
import os

from PySide6 import QtCore

from foundation.i18n import tr
from core.library import jobs
from core.workers.music_index import MusicIndexRequest, MusicIndexWorker

MAX_PROCESSES = 4


@dataclass(frozen=True)
class MusicIndexRun:
    total: int          # tracks to index
    processes: int


@dataclass(frozen=True)
class MusicIndexSummary:
    indexed: int
    failed: tuple[str, ...]
    dropped: int


class LibraryMusicIndex(QtCore.QObject):
    """Index Now (Settings > Search > Search by Music): the library tracks not in the music search's index, or
    without a Variants fingerprint of the current method (analyzed before they existed, or before the method
    changed), indexed; the tracks gone from the library dropped; once done, the fingerprints of earlier methods the
    current one has replaced removed. One run at a time. The audio is read in up to MAX_PROCESSES processes (a
    quarter of the cores), each its share of the tracks; everything else, and every write, is the library's owner's
    (core.library.service), a track a job. The analysis indexes what it analyzes."""

    started = QtCore.Signal(object)               # MusicIndexRun
    progress = QtCore.Signal(object, int, str)    # run, percent, text
    finished = QtCore.Signal(object, object)      # run, MusicIndexSummary
    failed = QtCore.Signal(object, str)

    def __init__(self, *, library, parent=None) -> None:
        """``library``: the library's owner (core.library.service)."""
        super().__init__(parent)
        self._library = library
        self._planning = False
        self._run: MusicIndexRun | None = None
        self._workers: list[MusicIndexWorker] = []
        self._ended = 0                         # the workers whose processes ended
        self._pending: dict[str, set] = {}      # a track -> its parts not done yet ("music", "variants")
        self._tracks: dict = {}                 # uid -> core.library.records.IndexTrack
        self._handed: set[str] = set()
        self._indexed = 0
        self._failed: list[str] = []
        self._errors: list[str] = []
        self._dropped = 0

    def running(self) -> bool:
        return self._run is not None or self._planning

    @QtCore.Slot(object)
    def index_requested(self, _payload: object = None) -> None:
        if self.running():
            return
        self._planning = True       # what to do read by the library's owner, then the run (_start)
        self._library.read(jobs.index_plan, done=self._start, failed=self._plan_failed)

    def _plan_failed(self, exc: Exception) -> None:
        self._planning = False
        run = MusicIndexRun(0, 0)
        self.started.emit(run)
        self.failed.emit(run, str(exc))

    def _start(self, plan: jobs.IndexPlan) -> None:
        self._planning = False
        todo = list(plan.todo)
        music = [t for t in todo if t.music]
        n = max(1, min(MAX_PROCESSES, (os.cpu_count() or 4) // 4, len(music))) if music else 0
        run = MusicIndexRun(len(todo), n)
        self._run, self._ended, self._indexed, self._failed, self._errors = run, 0, 0, [], []
        self._tracks = {t.uid: t for t in todo}
        self._handed = set()        # the tracks whose entry a process handed back
        self._pending = {t.uid: {p for p, on in (("music", t.music), ("variants", t.variants)) if on} for t in todo}
        self._dropped = len(set(plan.drop) | set(plan.drop_variants))
        self.started.emit(run)
        if plan.drop or plan.drop_variants:
            self._library.write(jobs.drop_gone, plan)
        # the variants' fingerprints: from the stored features, by the library's owner, a track a job
        for t in todo:
            if t.variants:
                self._library.write(jobs.fingerprint, t.uid,
                                    done=lambda _r, uid=t.uid: self._part_done(uid, "variants", True),
                                    failed=lambda _exc, uid=t.uid: self._part_done(uid, "variants", False))
        # Search by Music's: the audio read in the processes, each entry stored by the owner as it comes
        self._workers = []
        for k in range(n):
            worker = MusicIndexWorker(MusicIndexRequest(tuple(music[k::n])), parent=self)
            worker.entry.connect(self._on_entry)
            worker.finished.connect(self._on_worker_ended)
            worker.failed.connect(self._on_worker_failed)
            self._workers.append(worker)
            worker.start()
        self._maybe_finish()

    @QtCore.Slot(str, object)
    def _on_entry(self, uid: str, entry) -> None:
        track = self._tracks.get(uid)
        if track is None or self._run is None:
            return
        self._handed.add(uid)
        if entry is None:       # its audio could not be read
            self._part_done(uid, "music", False)
            return
        self._library.write(jobs.store_music_entry, track, entry, done=lambda _r: self._part_done(uid, "music", True),
                            failed=lambda _exc: self._part_done(uid, "music", False))

    def _part_done(self, uid: str, part: str, ok: bool, check: bool = True) -> None:
        parts = self._pending.get(uid)
        if parts is None or part not in parts or self._run is None:
            return
        parts.discard(part)
        track = self._tracks[uid]
        if not ok and track.path not in self._failed:
            self._failed.append(track.path)
        if not parts:
            del self._pending[uid]
            self._indexed += track.path not in self._failed
            done, total = self._run.total - len(self._pending), self._run.total
            self.progress.emit(self._run, 100 * done // total if total else 100,
                               tr("Indexing for Search by Music and Variants: {done} / {total} tracks").format(
                                   done=done, total=total))
        if check:
            self._maybe_finish()

    @QtCore.Slot()
    def _on_worker_ended(self) -> None:
        self._ended += 1
        self._maybe_finish()

    @QtCore.Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        self._errors.append(str(message))
        self._ended += 1
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        run = self._run
        if run is None or self._ended < len(self._workers):
            return
        if self._errors:
            # a process failed: its tracks not handed back are not indexed
            for uid in [u for u, parts in self._pending.items() if "music" in parts and u not in self._handed]:
                self._part_done(uid, "music", False, check=False)
        if self._pending:
            return      # the owner's jobs still running: done as they come
        for worker in self._workers:
            worker.deleteLater()
        self._workers = []
        self._run = None
        if self._errors:
            self.failed.emit(run, self._errors[0])
            return
        summary = MusicIndexSummary(indexed=self._indexed, failed=tuple(self._failed), dropped=self._dropped)

        def failed(exc: Exception) -> None:
            print(f"[Variants] earlier methods not removed: {exc}")
            self.finished.emit(run, summary)

        # every track done: the fingerprints of earlier methods removed by the library's owner; finished told after
        # it (the library's Variants read again by then)
        self._library.write(jobs.prune_fingerprints, done=lambda _r: self.finished.emit(run, summary), failed=failed)
