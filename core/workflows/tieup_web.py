"""The tie-ups' web lookups (core.tieups.web) off the GUI thread: a new track left without tie-ups after its analysis
looked up when Settings > Library says so (its sure tie-up written; none found: the work its album's other tracks
share, core.library.tieups.tag_by_album, asked again once the lookups waiting are done, as its album's other tracks
may be tagged after it), and the Edit tab's "Find on the Web". One thread works through them in turn (Wikimedia is
asked one request at a time anyway)."""
from __future__ import annotations

import queue
import threading
from typing import Callable

from PySide6 import QtCore

from core.tieups import web as tieup_web
from core.library import jobs

IDLE_SEC = 5.0      # the lookup thread, given no job this long, asks the albums' works and ends


class TieupWebLookup(QtCore.QObject):
    """``library``: the library's owner (core.library.service), whom the lookup thread asks for the track's title and
    artist and gives the tie-ups to write; ``enabled`` (Settings > Library's option for new tracks) is read at each
    lookup. The thread itself only asks Wikimedia."""

    tagged = QtCore.Signal(list)           # the tracks given a tie-up
    found = QtCore.Signal(str, object)     # a track, its candidates (a list of tieup_web.Candidate), "Find on the Web"
    failed = QtCore.Signal(str, str)       # a track, why
    # a new track's background lookup, for the Workers window: queued (the track, its title), started, done (the
    # track, what came of it); started and done are emitted on the lookup's thread (queued to the receivers')
    queued = QtCore.Signal(str, str)
    started = QtCore.Signal(str)
    done = QtCore.Signal(str, str)

    def __init__(self, *, library, enabled: Callable[[], bool] = lambda: False, parent=None) -> None:
        super().__init__(parent)
        self._library = library
        self._enabled = enabled
        self._jobs: queue.Queue = queue.Queue()
        # the lookup thread: None once it has ended (or is ending with nothing left to do); the lock makes that
        # ending and a job's put one or the other, so a job never waits for a thread that is gone
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def new_track(self, uid: str, title: str = "") -> None:
        """A new track's lookup, when Settings > Library's option is on."""
        if uid and self._enabled():
            self.queued.emit(uid, title)
            self._put(("tag", uid))

    def find(self, uid: str) -> None:
        """Every candidate for the track, sent by `found`."""
        if uid:
            self._put(("find", uid))

    def _put(self, job: tuple) -> None:
        with self._lock:
            self._jobs.put(job)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._work, name="tieup-web", daemon=True)
                self._thread.start()

    def _by_album(self, tracks: list[str]) -> list[str]:
        """The tracks given their album's work (jobs.tag_by_album), said tagged."""
        tagged = self._library.call(jobs.tag_by_album, tracks, write=True)
        if tagged:
            self.tagged.emit(tagged)
            for uid in tagged:
                self.done.emit(uid, "tagged")
        return tagged

    def _work(self) -> None:
        try:
            self._lookups()
        finally:
            with self._lock:
                if self._thread is threading.current_thread():   # ended by an error: the next put starts another
                    self._thread = None

    def _lookups(self) -> None:
        untagged: list[str] = []        # the new tracks nothing was found for: their album's work asked again
        while True:
            try:
                kind, uid = self._jobs.get(timeout=IDLE_SEC)
            except queue.Empty:
                if untagged:
                    try:
                        self._by_album(untagged)
                    except Exception as exc:
                        print(f"[Tie-ups] album's tie-up not written: {exc}")
                    untagged = []
                with self._lock:          # a job put while the album's were asked for: taken, not left waiting
                    if self._jobs.empty():
                        self._thread = None
                        return
                continue
            if kind == "tag":
                self.started.emit(uid)
            try:
                song = self._library.call(jobs.song_of, uid, kind == "tag")
                if kind == "tag":
                    sure = tieup_web.sure_of(tieup_web.shared(), song) if song is not None else None
                    if sure is not None:
                        # written by the library's owner (the readings in its languages)
                        self._library.call(jobs.add_found_tieups, [uid], [sure.as_tieup()], write=True)
                        self.tagged.emit([uid])
                        self.done.emit(uid, "tagged")
                    elif song is not None and not self._by_album([uid]):
                        untagged.append(uid)
                        self.done.emit(uid, "none")
                    elif song is None:
                        self.done.emit(uid, "none")
                else:
                    self.found.emit(uid, tieup_web.lookup(tieup_web.shared(), *song[:2], album=song[2])
                                    if song is not None else [])
            except Exception as exc:     # the network, a page: nothing written
                print(f"[Tie-ups] web lookup failed: {exc}")
                if kind == "find":
                    self.failed.emit(uid, str(exc))
                else:
                    self.done.emit(uid, "failed")
