"""The tie-ups' web lookups (core.tieup_web) off the GUI thread: a new track left without tie-ups after its analysis
looked up when Settings > Library says so (its sure tie-up written), and the Edit tab's "Find on the Web". One
thread works through them in turn (Wikimedia is asked one request at a time anyway)."""
from __future__ import annotations

import os
import queue
import threading
from typing import Callable

from PySide6 import QtCore

from core import tieup_web


class TieupWebLookup(QtCore.QObject):
    """``libpath`` / ``settings`` (the search settings: the readings' languages) / ``enabled`` (Settings > Library's
    option for new tracks) are read at each lookup."""

    tagged = QtCore.Signal(list)           # the tracks given a tie-up
    found = QtCore.Signal(str, object)     # a track, its candidates (a list of tieup_web.Candidate), "Find on the Web"
    failed = QtCore.Signal(str, str)       # a track, why

    def __init__(self, *, libpath: Callable[[], str], settings: Callable[[], object],
                 enabled: Callable[[], bool] = lambda: False, parent=None) -> None:
        super().__init__(parent)
        self._libpath = libpath
        self._settings = settings
        self._enabled = enabled
        self._jobs: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def _library_db(self) -> str:
        return os.path.join(str(self._libpath()), "library.db")

    def new_track(self, uid: str) -> None:
        """A new track's lookup, when Settings > Library's option is on."""
        if uid and self._enabled():
            self._put(("tag", uid))

    def find(self, uid: str) -> None:
        """Every candidate for the track, sent by `found`."""
        if uid:
            self._put(("find", uid))

    def _put(self, job: tuple) -> None:
        self._jobs.put(job)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="tieup-web", daemon=True)
            self._thread.start()

    def _work(self) -> None:
        while True:
            try:
                kind, uid = self._jobs.get(timeout=5.0)
            except queue.Empty:
                return
            s = self._settings()
            try:
                if kind == "tag":
                    if tieup_web.tag_track(self._library_db(), uid, tieup_web.shared(), languages=s.db_languages,
                                           han_run=s.han_run()):
                        self.tagged.emit([uid])
                else:
                    self.found.emit(uid, tieup_web.candidates_for(self._library_db(), uid, tieup_web.shared()))
            except Exception as exc:     # the network, a page: nothing written
                print(f"[Tie-ups] web lookup failed: {exc}")
                if kind == "find":
                    self.failed.emit(uid, str(exc))
