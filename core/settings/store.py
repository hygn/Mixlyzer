"""The settings' one owner: config.json and deck.json read once at the start (core.settings.config.load_cfg), kept
in memory and written by it alone, each change told (``changed``).

    store = SettingsStore(load_cfg())       # the start (app.startup), installed for the app (install)
    current().analysisconfig                # what the settings are now: the store's, no file read
    store.save(new)                         # the settings dialog's save: both files written, ``changed`` sent
    cfg.deckconfig.trim = ...; store.deck_changed()   # what playing changes: deck.json a second after it last did

Nothing else reads or writes the files (tests/test_settings_boundary.py); a spawned worker process is given the
settings it runs with, never reads them. No thread of its own (as the library's owner has): the files are a few kB,
written only when the user changes something.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore

from core.settings.config import config, default_cfg, save_cfg, save_deck

DECK_SAVE_DELAY_MS = 1000     # a knob turned reports every frame: deck.json written once it rests


class SettingsStore(QtCore.QObject):
    changed = QtCore.Signal(object, object)     # (old, new): the settings saved (a new config object)

    def __init__(self, cfg: config, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._deck_save = QtCore.QTimer(self)
        self._deck_save.setSingleShot(True)
        self._deck_save.setInterval(DECK_SAVE_DELAY_MS)
        self._deck_save.timeout.connect(self._write_deck)
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.flush)

    @property
    def current(self) -> config:
        """The settings now (the object itself: its deckconfig is changed in place, then deck_changed)."""
        return self._cfg

    def save(self, new: config) -> None:
        """New settings (the settings dialog's, External Sync turned off): config.json and deck.json written at once,
        then ``changed`` sent."""
        self._deck_save.stop()
        save_cfg(new)
        save_deck(new)
        old, self._cfg = self._cfg, new
        self.changed.emit(old, new)

    def deck_changed(self) -> None:
        """The current settings' deckconfig changed in place (what playing changes): deck.json written a second
        after the last change."""
        self._deck_save.start()

    def flush(self) -> None:
        """A deck.json write waiting, done now (the app quitting)."""
        if self._deck_save.isActive():
            self._deck_save.stop()
            self._write_deck()

    def _write_deck(self) -> None:
        save_deck(self._cfg)


_store: Optional[SettingsStore] = None


def install(store: SettingsStore) -> None:
    """The app's store (made by the start): what current() reads."""
    global _store
    _store = store


def current() -> config:
    """The settings now: the installed store's (no file read); the defaults where none is (tests, a part made
    without the app)."""
    return _store.current if _store is not None else default_cfg()


def deck_changed() -> None:
    """current()'s deckconfig changed in place: the installed store writes deck.json (none: nothing written)."""
    if _store is not None:
        _store.deck_changed()
