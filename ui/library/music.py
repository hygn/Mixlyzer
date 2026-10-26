"""Search by Music's part: the microphone, the listening, its results as the table."""
from __future__ import annotations

import copy
from typing import List

from PySide6 import QtWidgets
from PySide6.QtMultimedia import QMediaDevices

from system.i18n import tr
from ui.music_search import DeckSeekAfterLoad, MusicListener, MusicSearchBar, close_icon, mic_icon
from ui.theme import current_theme
from ui.library.common import Track, _mmss


class MusicMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _build_music(self):
        """Search by Music's microphone, its bar and its listener."""
        # Search by Music (ui.music_search): the microphone left of the search field; on, the field becomes the
        # microphone's waveform and the library shows the songs heard, best first, until it is turned off; the close
        # button beside it is the plain library again
        self._mic_icon, self._mic_icon_on = mic_icon(), mic_icon(current_theme().on_accent)   # on: on the accent
        self.btn_mic = QtWidgets.QToolButton()
        self.btn_mic.setIcon(self._mic_icon)
        self.btn_mic.setCheckable(True)
        self.btn_mic.setAutoRaise(True)
        self.btn_mic.setToolTip(tr(
            "<p>Search by Music: hum, sing or whistle a melody, or play the song from a phone, and the library "
            "shows the songs the microphone hears, best first. Click again to stop listening (the results stay).</p>"))
        self.btn_mic.clicked.connect(self._toggle_mic)
        self.btn_music_close = QtWidgets.QToolButton()
        self.btn_music_close.setIcon(close_icon())
        self.btn_music_close.setAutoRaise(True)
        self.btn_music_close.setToolTip(tr("Back to the library"))
        self.btn_music_close.clicked.connect(self._exit_music_mode)
        self.btn_music_close.hide()
        self.music_bar = MusicSearchBar()
        self.music_bar.hide()
        self._music_mode = False
        self._music_matches: list = []          # analyzer.search.music.search.MusicMatch, best first
        self._music_saved_text = ""
        self._music = MusicListener(self.bus, lambda: self.cfg, self.library, parent=self)
        self._music.audio.connect(self.music_bar.wave.add)
        self._music.rate_changed.connect(self.music_bar.wave.reset)
        self._music.status.connect(self.music_bar.status.setText)
        self._music.results.connect(self._on_music_results)
        self._music.listening_changed.connect(self._on_listening_changed)
        self._music.device_changed.connect(self._show_music_device)
        self.music_bar.device_picked.connect(self._music.pick)
        QMediaDevices(self).audioInputsChanged.connect(self._show_music_device)    # plugged in / out
        self._deck_seek = DeckSeekAfterLoad(self.bus, self.model, parent=self)

    # Search by Music
    def _toggle_mic(self) -> None:
        """The microphone on (into music mode, listening) or off (the results stay, no longer followed)."""
        if self._music.listening():
            self._music.stop()
            return
        if not self._music_mode:
            self._enter_music_mode()
        self.music_bar.wave.reset(48000)
        if self._music.start():
            self.music_bar.wave.reset(self._music.rate)
        else:
            self.btn_mic.setChecked(False)

    def _enter_music_mode(self) -> None:
        self._music_mode = True
        self._music_matches = []
        self._music_saved_text = self.e_search.text()
        for widget in (self.e_search, self.cb_column, self.cb_fuzzy):
            widget.hide()
        self.music_bar.show()
        self.btn_music_close.show()
        self._show_music_device()
        self.e_search.blockSignals(True)
        self.e_search.clear()
        self.e_search.blockSignals(False)
        self._refresh_current_source(reset_sort=True)

    def _exit_music_mode(self) -> None:
        self._music.stop()
        if not self._music_mode:
            return
        self._music_mode = False
        self._music_matches = []
        self.music_bar.hide()
        self.btn_music_close.hide()
        for widget in (self.e_search, self.cb_column, self.cb_fuzzy):
            widget.show()
        self.e_search.blockSignals(True)
        self.e_search.setText(self._music_saved_text)
        self.e_search.blockSignals(False)
        self._refresh_current_source(reset_sort=True)

    def _show_music_device(self) -> None:
        """The music bar's input: the one heard (or to be heard) and the one picked."""
        dev = self._music.device()
        self.music_bar.show_device("" if dev.isNull() else dev.description(), self._music.picked(),
                                   self._music.listening(), self._music.inputs)

    def _on_listening_changed(self, on: bool) -> None:
        self.btn_mic.setChecked(on)
        self.btn_mic.setIcon(self._mic_icon_on if on else self._mic_icon)
        if not on:
            self.music_bar.wave.silence()

    def _on_music_results(self, result) -> None:
        """A search while listening: the library shows its songs, best first (kept while the selection is in it)."""
        if not self._music_mode:
            return
        library = {t.uid for t in self._library_tracks}
        selected = self._row_uid(self.table.currentRow()) if self.table.currentRow() >= 0 else ""
        self._music_matches = [m for m in result.matches if m.uid in library]
        self._refresh_current_source(reset_sort=True)
        for r in range(self.table.rowCount()):
            if selected and self._row_uid(r) == selected:
                self._select_row(r)
                break

    def _music_tracks(self) -> List[Track]:
        by_uid = {t.uid: t for t in self._library_tracks}
        tracks = []
        for rank, match in enumerate(self._music_matches, start=1):
            track = copy.copy(by_uid[match.uid])
            track.playlist_index = rank
            track._disp_playlist_index = str(rank)
            tracks.append(track)
        return tracks

    def _music_tooltips(self) -> None:
        """Each result row says where in the song it matched and how."""
        matches = {m.uid: m for m in self._music_matches}
        for r in range(self.table.rowCount()):
            m = matches.get(self._row_uid(r))
            if m is None:
                continue
            if m.kind == "recording":
                tip = tr("Matched at {start} - {end}, by the sound of the recording ({n} matches)").format(
                    start=_mmss(m.start_sec), end=_mmss(m.end_sec), n=int(m.score))
            else:
                tip = tr("Matched at {start} - {end}, by the melody").format(
                    start=_mmss(m.start_sec), end=_mmss(m.end_sec))
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item is not None:
                    item.setToolTip(tip)
