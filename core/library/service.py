"""The library's one owner: a thread of its own that does every read and write of the library (library.db, the
features' NPZ files, the thumbnails, the Variants' and Search by Music's indexes beside them). The rest of the app
asks it by name (core.library.jobs; the stored analyses' core.analysis_library.jobs) and is told; nothing else opens
the library.

    the GUI           library.read(jobs.track_tieups, uid, done=show)     done runs on the GUI thread
                      library.write(jobs.rename_playlist, uid, name)      then the snapshot is read again
    another thread    library.call(jobs.song_of, uid)                     waits for the result (never the GUI)
    a process         computes only; what it hands back is stored by a job (the analysis: jobs.store_analysis)

- `snapshot` (`LibrarySnapshot`): what the GUI shows of the library (the tracks, their readings and tie-ups for the
  search, the playlists and their entries, the Variants' relations), read on the owner's thread after writes; the GUI
  reads it at once, in memory. A write job names the parts it changes (`changes`, on the jobs) and only those are
  read again (the tracks and their readings are most of the cost: ~35 ms a thousand tracks); one that names none
  changes none. `updated` and `EventBus.sig_lib_updated` are sent with it when a part changed, before the writes'
  ``done`` (they see it).
- The jobs run one at a time in the order asked: a read asked after a write sees it.
- The connection lives on the owner's thread alone (SQLite refuses it on another), and a job cannot `call` another
  (it would wait for itself): both raise at once.
"""
from __future__ import annotations

import os
import queue
import threading
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from PySide6 import QtCore

from core.library import tieups, variants
from core.library.db import LibraryDB
from core.analysis_library.store import FeatureNPZStore
from core.library.records import PlaylistRow, TrackRow

READ, WRITE = "read", "write"

# the snapshot's parts, read again after a write that changes them
TRACKS, READINGS, TIEUPS, PLAYLISTS, VARIANTS = "tracks", "readings", "tieups", "playlists", "variants"
PARTS = frozenset({TRACKS, READINGS, TIEUPS, PLAYLISTS, VARIANTS})


def changes(*parts: str):
    """A write job's mark: the snapshot's parts it changes (none: the snapshot is not read again). A job without the
    mark may change anything: the snapshot is read whole after it."""
    unknown = set(parts) - PARTS
    if unknown:
        raise ValueError(f"not a part of the snapshot: {sorted(unknown)}")

    def mark(job):
        job.changes = frozenset(parts)
        return job
    return mark


def _changed_by(job) -> frozenset:
    return getattr(job, "changes", PARTS) if job is not None else PARTS


@dataclass
class Library:
    """A job's library, on the owner's thread: its connection, its folder, its feature store."""
    db: LibraryDB
    path: str
    features: FeatureNPZStore


@dataclass(frozen=True)
class LibrarySnapshot:
    """The library as the GUI shows it, read at one moment."""
    tracks: tuple[TrackRow, ...] = ()
    readings: dict = field(default_factory=dict)           # uid -> field -> readings (LibraryDB.all_readings)
    tieups: dict = field(default_factory=dict)             # uid -> core.tieup.TrackTieups
    playlists: tuple[PlaylistRow, ...] = ()
    entries: dict = field(default_factory=dict)            # playlist uid -> ((position, track uid), ...)
    variant_pairs: tuple = ()                              # core.variants.relations.Pair
    by_uid: dict = field(default_factory=dict)
    by_path: dict = field(default_factory=dict)

    @staticmethod
    def read(lib: Library, parts: frozenset = PARTS, old: Optional["LibrarySnapshot"] = None) -> "LibrarySnapshot":
        """The ``parts`` read from the library, the others kept from ``old`` (without it, every part read)."""
        if old is None:
            old, parts = LibrarySnapshot(), PARTS
        db = lib.db
        kept: dict = {}
        if TRACKS in parts:
            tracks = tuple(db.list_all())
            kept.update(tracks=tracks, by_uid={t.uid: t for t in tracks}, by_path={t.path: t for t in tracks})
        if READINGS in parts:
            kept.update(readings=db.all_readings())
        if TIEUPS in parts:
            kept.update(tieups=tieups.track_texts(db.conn))
        if PLAYLISTS in parts:
            kept.update(playlists=tuple(db.list_playlists()), entries=db.playlist_entries())
        if VARIANTS in parts:
            kept.update(variant_pairs=tuple(variants.pairs(variants.path_for(lib.path))))
        return replace(old, **kept)

    def track(self, uid: Optional[str]) -> Optional[TrackRow]:
        return self.by_uid.get(uid or "")

    def get(self, path: str) -> Optional[TrackRow]:
        """The track stored under exactly ``path`` (LibraryDB.get)."""
        return self.by_path.get(path)

    def find_file(self, path: str) -> Optional[TrackRow]:
        """The track of the file ``path`` points to, however it is spelled (LibraryDB.find_file: stored under it,
        else the stored path opening the same file)."""
        row = self.by_path.get(path)
        if row is not None:
            return row
        try:
            st = os.stat(path)
        except (OSError, ValueError):
            return None
        for t in self.tracks:
            if t.file_size not in (None, st.st_size):
                continue
            try:
                if os.path.samestat(st, os.stat(t.path)):
                    return t
            except (OSError, ValueError):
                continue
        return None

    def playlist_tracks(self, playlist_uid: str) -> list[tuple[int, TrackRow]]:
        return [(pos, self.by_uid[uid]) for pos, uid in self.entries.get(playlist_uid, ()) if uid in self.by_uid]


def _alive(fn: Callable) -> bool:
    """A callback still has its widget (a bound method of a QObject deleted meanwhile is not called)."""
    owner = getattr(fn, "__self__", None)
    if isinstance(owner, QtCore.QObject):
        import shiboken6
        return shiboken6.isValid(owner)
    return True


Job = Callable[..., Any]       # job(lib, *args): core.library.jobs'


class LibraryService(QtCore.QObject):
    """See the module. ``settings``: the search settings (their reading languages read a written track's readings
    again, LibraryDB.reading_languages / han_run), taken when a job is asked; ``bus``: sent sig_lib_updated."""

    updated = QtCore.Signal(object)       # LibrarySnapshot, after a write
    _deliver = QtCore.Signal(object)      # a callable run on the GUI thread, in the order the jobs ran

    def __init__(self, libpath: str, *, settings: Optional[Callable[[], Any]] = None, bus=None, parent=None):
        super().__init__(parent)
        self._libpath = str(libpath)
        self._settings = settings
        self._bus = bus
        self.snapshot = LibrarySnapshot()
        self._jobs: queue.Queue = queue.Queue()
        self._deliver.connect(self._on_gui)
        self._thread = threading.Thread(target=self._loop, name="library", daemon=True)
        self._thread.start()

    # ---- asking ----
    def attach(self, *, bus=None, settings: Optional[Callable[[], Any]] = None) -> None:
        """The window's bus and search settings, for an owner started before it (the startup's)."""
        self._bus = bus
        self._settings = settings

    @property
    def libpath(self) -> str:
        return self._libpath

    def read(self, job: Job, *args, done: Optional[Callable[[Any], None]] = None,
             failed: Optional[Callable[[Exception], None]] = None) -> None:
        """``job(lib, *args)`` on the owner's thread; its result to ``done`` on the GUI thread (an exception to
        ``failed``, else printed)."""
        self._put(READ, job, args, done, failed)

    def write(self, job: Job, *args, done: Optional[Callable[[Any], None]] = None,
              failed: Optional[Callable[[Exception], None]] = None) -> None:
        """As `read`, then the snapshot read again (``done`` sees it)."""
        self._put(WRITE, job, args, done, failed)

    def call(self, job: Job, *args, write: bool = False) -> Any:
        """``job(lib, *args)`` on the owner's thread, waited for: its result returned, its exception raised. For
        another thread's work (a web lookup, a tag writer, the Rekordbox XML sync): never the GUI thread (it would
        freeze), never a job (it would wait for itself). ``write``: as `write`."""
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("LibraryService.call waits: not on the GUI thread (read / write instead)")
        if threading.current_thread() is self._thread:
            raise RuntimeError("LibraryService.call from a job would wait for itself: call the other job directly")
        box: dict = {}
        ran = threading.Event()
        self._put(WRITE if write else READ, job, args, None, None, (box, ran))
        ran.wait()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def refresh(self, done: Optional[Callable[[Any], None]] = None) -> None:
        """The snapshot read again."""
        self._put(WRITE, None, (), done, None)

    def set_libpath(self, libpath: str) -> None:
        """Another library folder: the jobs from now on are its, the snapshot read from it."""
        if str(libpath) != self._libpath:
            self._libpath = str(libpath)
            self.refresh()

    def wait(self, timeout: float = 30.0) -> bool:
        """Blocks until every job asked so far has run and its result was handed over (the start, tests)."""
        ran = threading.Event()
        self._jobs.put((READ, lambda _lib: ran.set(), (), None, None, None, self._libpath, None))
        ok = ran.wait(timeout)
        QtCore.QCoreApplication.sendPostedEvents(self, 0)
        return ok

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5.0)

    def _put(self, kind: str, job, args: tuple, done, failed, reply=None) -> None:
        s = self._settings() if self._settings is not None else None
        reading = (list(s.db_languages), int(s.han_run())) if s is not None else None
        self._jobs.put((kind, job, args, done, failed, reading, self._libpath, reply))

    def _on_gui(self, fn: Callable[[], None]) -> None:
        fn()

    def _apply(self, snap: LibrarySnapshot) -> None:
        self.snapshot = snap
        self.updated.emit(snap)
        if self._bus is not None:
            self._bus.sig_lib_updated.emit(list(snap.tracks))

    def _call(self, fn: Optional[Callable], arg) -> Callable[[], None]:
        def run() -> None:
            if fn is not None and _alive(fn):
                fn(arg)
        return run

    # ---- the owner's thread ----
    def _loop(self) -> None:
        lib: Optional[Library] = None
        last: Optional[LibrarySnapshot] = None   # the snapshot this thread read last (None: none of this library)
        dirty = False                            # writes ran since the snapshot was read
        changed: set[str] = set()                # the parts they changed
        after: list[Callable[[], None]] = []     # the writes' callbacks, handed over after the snapshot

        def flush() -> None:
            nonlocal dirty, changed, after, last
            if changed and lib is not None:
                try:
                    last = LibrarySnapshot.read(lib, frozenset(changed), last)
                    self._deliver.emit(lambda s=last: self._apply(s))
                except Exception:
                    last = None          # read whole next time
                    print("[Library] the library could not be read:\n" + traceback.format_exc())
            for fn in after:
                self._deliver.emit(fn)
            dirty, changed, after = False, set(), []

        while True:
            item = self._jobs.get()
            if item is None:
                break
            kind, job, args, done, failed, reading, libpath, reply = item
            if kind == READ and dirty:
                flush()          # a read after writes: their snapshot and callbacks first, in order
            result, error = None, None
            try:
                if lib is None or lib.path != libpath:
                    if lib is not None:
                        lib.db.close()
                        lib = None
                    db = LibraryDB(os.path.join(libpath, "library.db"))
                    db.connect()
                    lib = Library(db, libpath, FeatureNPZStore(base_dir=libpath, compressed=True))
                    last, dirty = None, True
                    changed.update(PARTS)
                if reading is not None:
                    lib.db.reading_languages, lib.db.han_run = reading
                result = job(lib, *args) if job is not None else None
                if lib.db.conn.in_transaction:
                    lib.db.conn.commit()
                outcome = self._call(done, result)
            except Exception as exc:
                error = exc
                if lib is not None and lib.db.conn.in_transaction:
                    lib.db.conn.rollback()
                if failed is None and reply is None:
                    print("[Library] a job failed:\n" + traceback.format_exc())
                outcome = self._call(failed, exc)
            if reply is not None:        # another thread waits for it (call)
                box, ran = reply
                box.update({"error": error} if error is not None else {"result": result})
                ran.set()
            if kind == WRITE:
                dirty = True
                changed.update(_changed_by(job))
                after.append(outcome)
            else:
                self._deliver.emit(outcome)
            if self._jobs.empty():
                flush()
        if lib is not None:
            lib.db.close()
