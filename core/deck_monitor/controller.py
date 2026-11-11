"""Deck Monitor on the app's side: its window runs in a process of its own (it receives the decks, decodes their
tracks and draws them with Mixlyzer's own main view and overview there, so none of it takes the app's GIL or GUI
thread). The app starts and ends that process, gives it the settings (again when they change), the library's tracks
(again when the library changes) and the stored analyses it asks for (read by the library's owner), and keeps where
the window listens (deck.json: the window is where it is set). Nothing of the app's deck is touched.

Messages, each a dict with its "type":
    to the process:   settings (cfg: the app's settings, read only there; language), tracks (gen, the library's
                      tracks), analysis (uid, gen, the track's stored analysis or None), close
    from the process: listening (host, port: where it listens now, set in its window), phrase_unit (unit: the decks'
                      phrases by section or by label, clicked there), analysis? (uid: one asked for), done (its
                      window closed)"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Optional

from PySide6 import QtCore

from core.analysis_library import jobs as analysis_jobs
from core.deck_monitor.readout import UNITS
from core.settings import store as settings_store
from system.process_worker import SpawnProcessWorker

SNAPSHOT_POLL_MS = 500       # how often the library's snapshot is checked for a change while the window is open


class DeckMonitorProcess(SpawnProcessWorker):
    """The window's process: ``target(settings, commands, messages)`` (a module-level callable)."""

    message = QtCore.Signal(object)
    ended = QtCore.Signal()

    def __init__(self, target: Callable[..., None], settings: dict, parent=None) -> None:
        self.commands = mp.get_context("spawn").Queue()
        super().__init__(target=target, args=(settings, self.commands), daemon=True, process_name="DeckMonitor",
                         parent=parent)

    def send(self, message: dict) -> None:
        try:
            self.commands.put_nowait(message)
        except (ValueError, OSError):
            pass        # the process gone (its queue closed)

    def _handle_process_message(self, message: dict[str, Any]) -> None:
        self.message.emit(message)

    def _process_exited(self, *, graceful: bool, exitcode: int | None) -> None:
        self.commands.close()
        self.ended.emit()


class DeckMonitorController(QtCore.QObject):
    """Opens and closes the window's process; ``open_changed(open)``."""

    open_changed = QtCore.Signal(bool)

    def __init__(self, *, target: Callable[..., None], settings: Callable[[], Any],
                 snapshot_provider: Callable[[], Any], library=None) -> None:
        """``target``: the window process's entry; ``settings()``: the app's settings now (the store's);
        ``snapshot_provider()``: the library's snapshot; ``library``: its owner (the stored analyses read)."""
        super().__init__()
        self._target = target
        self._settings = settings
        self._snapshot_provider = snapshot_provider
        self._library = library
        self._process: Optional[DeckMonitorProcess] = None
        self._sent_snapshot = None   # the snapshot's tracks as last sent
        self._gen = 0                # the library's tracks as last sent (an analysis read for an older one dropped)
        self._snapshot_timer = QtCore.QTimer(self)
        self._snapshot_timer.setInterval(SNAPSHOT_POLL_MS)
        self._snapshot_timer.timeout.connect(self._send_tracks)
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.close)

    @property
    def is_open(self) -> bool:
        return self._process is not None

    def toggle(self) -> None:
        self.close() if self.is_open else self.open()

    def open(self) -> None:
        if self._process is not None:
            return
        # no parent: let go by its own deleteLater (a parent destroyed before that ran, the app quitting or a test's
        # controller let go, took it along with the delete still posted: the next event loop crashed, exit 127)
        process = DeckMonitorProcess(self._target, self._process_settings())
        process.message.connect(self._on_message)
        process.ended.connect(lambda p=process: self._on_ended(p))
        self._process, self._sent_snapshot = process, None
        process.start()
        self._send_tracks()
        self._snapshot_timer.start()
        self.open_changed.emit(True)

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        process.send({"type": "close"})
        process.stop()               # ended on a thread of its own (system.process_worker): nothing in it to keep
        process.commands.close()
        self._on_ended(process)

    def settings_changed(self, *_args) -> None:
        """The app's settings saved: the window's sent again (it draws with them)."""
        if self._process is not None:
            self._process.send({"type": "settings", **self._process_settings()})

    def _process_settings(self) -> dict:
        from system import i18n
        return {"cfg": self._settings(), "language": i18n.language()}

    def _send_tracks(self) -> None:
        """The library's tracks, when they changed (a snapshot made again for its playlists keeps its tracks)."""
        tracks = getattr(self._snapshot_provider(), "tracks", ())
        if self._process is None or tracks is self._sent_snapshot:
            return
        self._sent_snapshot = tracks
        self._gen += 1
        self._process.send({"type": "tracks", "gen": self._gen, "tracks": tuple(tracks)})

    def _on_message(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "analysis?":
            self._read_analysis(str(message.get("uid") or ""))
        elif kind == "listening":
            self._keep_address(str(message.get("host") or ""), message.get("port"))
        elif kind == "phrase_unit":
            self._keep_phrase_unit(str(message.get("unit") or ""))

    def _keep_phrase_unit(self, unit: str) -> None:
        """The decks' phrases by section or by label, as clicked in the window: kept in deck.json."""
        if unit not in UNITS:
            return
        deck = self._settings().deckconfig
        if deck.monitor_phrase_unit != unit:
            deck.monitor_phrase_unit = unit
            settings_store.deck_changed()

    def _keep_address(self, host: str, port) -> None:
        """Where the window listens now (set there): kept in deck.json for the next time it opens."""
        try:
            port = int(port)
        except (TypeError, ValueError):
            return
        if not host or not 0 < port < 65536:
            return
        deck = self._settings().deckconfig
        if (deck.monitor_host, deck.monitor_port) != (host, port):
            deck.monitor_host, deck.monitor_port = host, port
            settings_store.deck_changed()

    def _read_analysis(self, uid: str) -> None:
        """A track's stored analysis read by the library's owner and sent to the window (None: none could be)."""
        process, gen = self._process, self._gen
        if not uid or process is None:
            return
        if self._library is None:
            process.send({"type": "analysis", "uid": uid, "gen": gen, "analysis": None})
            return

        def answer(analysis) -> None:
            if self._process is process:           # the window still open
                process.send({"type": "analysis", "uid": uid, "gen": gen, "analysis": analysis})
        self._library.read(analysis_jobs.analysis, uid, done=answer, failed=lambda _exc: answer(None))

    def _on_ended(self, process: DeckMonitorProcess) -> None:
        if self._process is not process:
            return
        self._process, self._sent_snapshot = None, None
        self._snapshot_timer.stop()
        process.deleteLater()
        self.open_changed.emit(False)
