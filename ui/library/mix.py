"""The Mix Candidates' part: the tracks to play after or before the loaded one (core.search.mixability), asked of the
library's owner, and their criteria at the end of the search bar's row (kept in deck.json), shown while they are the
source (the search field narrower meanwhile)."""
from __future__ import annotations

import copy
import html
from typing import List, Optional

from PySide6 import QtWidgets

from analyzer.music.key_names import ALPHANUMERIC_LABELS
from core.library import jobs
from core.search import mixability
from core.settings import store as settings_store
from system.i18n import tr, tr_ctx
from ui.library.common import Track, _mmss
from ui.library.search import SEARCH_MIN_WIDTH, _MatchToggle, compact_spin, dim_label, separator

MIX_SOURCE = "mix_candidates"
MIX_SEARCH_MIN_WIDTH = 120      # the search field's least while the criteria share its row


class _CriteriaToggle(_MatchToggle):
    """AND | OR of the mix criteria."""

    def _update_tooltip(self) -> None:
        self.setToolTip(tr("AND: every criterion turned on (click to switch)") if self.match_all()
                        else tr("OR: any criterion turned on (click to switch)"))


class _DirectionToggle(_MatchToggle):
    """Next | Previous: the tracks to play after the loaded one, or before it (checked)."""

    def __init__(self, parent=None):
        self.LABELS = (tr_ctx("mix candidates", "Next"), tr_ctx("mix candidates", "Previous"))
        super().__init__(parent)

    def previous(self) -> bool:
        return self.isChecked()

    def _update_tooltip(self) -> None:
        self.setToolTip(tr("<p>Next: tracks to blend in from the loaded one's interlude or outro, at their "
                           "start (click to switch).</p>") if not self.previous()
                        else tr("<p>Previous: tracks whose interlude or outro blends into the loaded one's start "
                                "(click to switch).</p>"))


class MixMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _build_mix(self) -> QtWidgets.QWidget:
        """The criteria's card (``mix_bar``, at the end of the search bar's row), hidden but while the Mix Candidates
        are shown."""
        deck = self.cfg.deckconfig
        self.mix_direction = _DirectionToggle()
        self.mix_direction.setChecked(bool(deck.mix_previous))
        self.cb_mix_bpm = self._chip(
            "BPM", tr("<p>The two tracks' BPM within this range where they meet, or one at half / double the "
                      "other.</p>"), small=True)
        self.sp_mix_tempo = compact_spin(QtWidgets.QDoubleSpinBox(), 0.5, 50.0, float(deck.mix_tempo_range), "%")
        self.sp_mix_tempo.setToolTip(tr("BPM range"))
        self.cb_mix_key = self._chip(
            "Key", tr("<p>The two tracks' keys where they meet the same, relative or ±1 on the Alphanumeric "
                      "wheel.</p>"),
            small=True)
        self.cb_mix_audio = self._chip(
            tr("Sound"), tr("<p>Phrases on both tracks line up and the bars sound alike where they would be "
                            "blended: energy, rhythm and harmony.</p>"), small=True)
        self.mix_toggle = _CriteriaToggle()
        self.cb_mix_bpm.setChecked(bool(deck.mix_bpm))
        self.cb_mix_key.setChecked(bool(deck.mix_key))
        self.cb_mix_audio.setChecked(bool(deck.mix_audio))
        self.mix_toggle.setChecked(not bool(deck.mix_match_all))
        for switch in (self.mix_direction, self.mix_toggle, self.sp_mix_tempo):   # one line, as the chips
            switch.setFixedHeight(self.cb_mix_bpm.sizeHint().height())
        self.mix_bar = self._card(tr("Mix"), (self.mix_direction, separator(), self.cb_mix_bpm, dim_label("±"),
                                              self.sp_mix_tempo, self.cb_mix_key, self.cb_mix_audio, self.mix_toggle))
        self.mix_bar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.mix_bar.hide()
        for chip in (self.mix_direction, self.cb_mix_bpm, self.cb_mix_key, self.cb_mix_audio, self.mix_toggle):
            chip.toggled.connect(self._mix_options_changed)
        self.sp_mix_tempo.valueChanged.connect(self._mix_options_changed)
        # the loaded track's candidates asked (loaded uid, options), and the last answer: (that request, candidates)
        self._mix_asked: Optional[tuple] = None
        self._mix_found: Optional[tuple] = None
        self._mix_rows: dict[str, mixability.Candidate] = {}
        return self.mix_bar

    def _mix_options(self) -> mixability.MixOptions:
        return mixability.MixOptions(self.cb_mix_bpm.isChecked(), self.cb_mix_key.isChecked(),
                                     self.cb_mix_audio.isChecked(), self.mix_toggle.match_all(),
                                     float(self.sp_mix_tempo.value()), self.mix_direction.previous())

    def _mix_options_changed(self, *_args) -> None:
        """A criterion changed: kept in deck.json, the candidates asked again."""
        options = self._mix_options()
        deck = self.cfg.deckconfig
        deck.mix_previous = options.previous
        deck.mix_bpm, deck.mix_key, deck.mix_audio = options.bpm, options.key, options.audio
        deck.mix_match_all, deck.mix_tempo_range = options.match_all, options.tempo_range
        self.sp_mix_tempo.setEnabled(options.bpm)
        settings_store.deck_changed()
        if self._source_kind == MIX_SOURCE:
            self._refresh_current_source()

    def _sync_mix_bar(self) -> None:
        shown = self._source_kind == MIX_SOURCE and not self._music_mode
        self.mix_bar.setVisible(shown)
        self.e_search.setMinimumWidth(MIX_SEARCH_MIN_WIDTH if shown else SEARCH_MIN_WIDTH)   # room for the card
        self.sp_mix_tempo.setEnabled(self.cb_mix_bpm.isChecked())

    def _mix_tracks(self) -> List[Track]:
        """The candidates of the loaded track as rows, # their rank; asked of the library's owner when not answered
        yet for this track and these criteria (shown when the answer comes; none meanwhile)."""
        if not self._loaded_uid:
            self._mix_rows = {}
            return []
        request = (self._loaded_uid, self._mix_options())
        if self._mix_found is None or self._mix_found[0] != request:
            if self._mix_asked != request:
                self._mix_asked = request

                def done(found, request=request) -> None:
                    self._mix_found = (request, found)
                    if self._source_kind == MIX_SOURCE and self._mix_asked == request:
                        self._refresh_current_source()

                self.library.read(jobs.mix_candidates, *request, done=done)
            self._mix_rows = {}
            return []
        by_uid = {t.uid: t for t in self._library_tracks if t.uid}
        tracks = []
        self._mix_rows = {}
        for rank, candidate in enumerate(self._mix_found[1], start=1):
            track = by_uid.get(candidate.uid)
            if track is None:
                continue
            track = copy.copy(track)
            track.playlist_index = rank
            track._disp_playlist_index = str(rank)
            tracks.append(track)
            self._mix_rows[candidate.uid] = candidate
        return tracks

    def _mix_forget(self) -> None:
        """The library changed: the candidates asked again when shown."""
        self._mix_found = None
        self._mix_asked = None

    def _mix_marks(self) -> None:
        """Each row's tooltip: where the blend goes, the BPM and the key on both sides, the criteria met."""
        previous = self._mix_found[0][1].previous if self._mix_found else False
        for r in range(self.table.rowCount()):
            candidate = self._mix_rows.get(self._row_uid(r))
            if candidate is None:
                continue
            tip = "<p>" + "<br>".join(html.escape(line) for line in _mix_lines(candidate, previous)) + "</p>"
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item is not None:
                    item.setToolTip(tip)


def _key_name(key: Optional[int]) -> str:
    return ALPHANUMERIC_LABELS[int(key) % 24] if key is not None else "?"


def _mix_lines(c: mixability.Candidate, previous: bool) -> list[str]:
    """The tooltip's lines of a candidate (``previous``: it is played before the loaded track, else after)."""
    lines = []
    if c.bars and previous:
        lines.append(tr("Blend: this track {a_from} - {a_to}, the loaded track from {b_from}, {bars} bars").format(
            a_from=_mmss(c.a_from), a_to=_mmss(c.a_to), b_from=_mmss(c.b_from), bars=c.bars))
    elif c.bars:
        lines.append(tr("Blend: the loaded track {a_from} - {a_to}, this track from {b_from}, {bars} bars").format(
            a_from=_mmss(c.a_from), a_to=_mmss(c.a_to), b_from=_mmss(c.b_from), bars=c.bars))
    else:
        lines.append(tr("No blend lines up on both tracks' phrases"))
    if c.bpm_a > 0 and c.bpm_b > 0:
        lines.append(tr("BPM {a:.1f} → {b:.1f} ({offset:+.1f} %)").format(a=c.bpm_a, b=c.bpm_b,
                                                                             offset=c.tempo_pct))
    relation = {"same": tr("the same key"), "relative": tr("the relative key"),
                "adjacent": tr("±1 on the Alphanumeric wheel"),
                "other": tr("another key")}.get(mixability.key_relation(c.key_a, c.key_b), "")
    key = tr("Key {a} → {b}").format(a=_key_name(c.key_a), b=_key_name(c.key_b))
    lines.append(tr("{key}: {relation}").format(key=key, relation=relation) if relation else key)
    met = [name for name, ok in zip(("BPM", "Key", tr("Sound")), c.passed) if ok]
    lines.append(tr("Meets: {criteria}").format(criteria=", ".join(met)) if met else tr("Meets no criterion"))
    return lines
