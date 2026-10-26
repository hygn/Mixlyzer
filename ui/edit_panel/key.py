from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from analyzer.music.beat_geometry import downbeat_beat_indices
from ui.edit_panel.blocks import mode_column, rule_cell
from ui.edit_panel.icons import badged_icon, badged_icon_beside, icon
from ui.edit_panel.key_picker import ALPHANUMERIC_INDEX, KeyPickerPopup
from views.render.color import key_cmap_color, key_label_text_color
from views.render.keystrip import key_segment_label
from analyzer.music.key_names import ALPHANUMERIC_LABELS
from core.marks.phrases import assigned_phrases
from core.track.reanalysis import KEY_DYNAMIC, KEY_STATIC, ReanalysisRequest
from system.i18n import tr, tr_mark
from ui.live_label import LiveLabel


# the tools that follow the key scope: icon -> (tooltip with the whole track, with the current segment),
# translated where shown
KEY_SCOPED_TOOLS = {
    "key_flat": (tr_mark("A semitone down: every segment's key"),
                 tr_mark("A semitone down: the current segment's key")),
    "key_sharp": (tr_mark("A semitone up: every segment's key"),
                  tr_mark("A semitone up: the current segment's key")),
    "key_fifth_down": (tr_mark("One step down the circle of fifths (Alphanumeric -1): every segment's key"),
                       tr_mark("One step down the circle of fifths (Alphanumeric -1): the current segment's key")),
    "key_fifth_up": (tr_mark("One step up the circle of fifths (Alphanumeric +1): every segment's key"),
                     tr_mark("One step up the circle of fifths (Alphanumeric +1): the current segment's key")),
    "key_relative": (tr_mark("Relative major / minor (C <-> Am): every segment's key"),
                     tr_mark("Relative major / minor (C <-> Am): the current segment's key")),
}
# these glyphs fill their icon's height: the badge goes beside them (a wide icon), so they keep their size
KEY_BADGE_BESIDE = ("key_flat", "key_sharp", "key_fifth_down", "key_fifth_up")
# boundary nudges follow the step: icon -> (tooltip by one beat, by one bar), translated where shown
KEY_STEP_TOOLS = {
    "key_start_back": (tr_mark("Move the current key segment's start one beat earlier"),
                       tr_mark("Move the current key segment's start one bar (to the neighboring downbeat) earlier")),
    "key_start_fwd": (tr_mark("Move the current key segment's start one beat later"),
                      tr_mark("Move the current key segment's start one bar (to the neighboring downbeat) later")),
    "key_end_back": (tr_mark("Move the current key segment's end one beat earlier"),
                     tr_mark("Move the current key segment's end one bar (to the neighboring downbeat) earlier")),
    "key_end_fwd": (tr_mark("Move the current key segment's end one beat later"),
                    tr_mark("Move the current key segment's end one bar (to the neighboring downbeat) later")),
}


class KeyEditMixin:
    """Key section, edited by segment like the beatgrid: the key segment under the
    playhead is the one edited (its key, its boundaries, split / merge, reanalysis)."""

    def _build_key_ui(self, grid: QtWidgets.QGridLayout) -> list[QtWidgets.QWidget]:
        """Key widgets on the Key page; returns the edit buttons."""

        def _btn(icon_name: str, tip: str, *, checkable: bool = False, wide: bool = False) -> QtWidgets.QToolButton:
            btn = QtWidgets.QToolButton()
            btn.setIcon(icon(icon_name))
            btn.setIconSize(QtCore.QSize(36, 18) if wide else QtCore.QSize(19, 19))
            btn.setToolTip(tip)
            btn.setCheckable(checkable)
            return btn

        # boundaries, by a beat or a bar (icons and tooltips follow the step: _refresh_key_mode_tools)
        self.key_start_back = _btn("key_start_back", "")
        self.key_start_fwd = _btn("key_start_fwd", "")
        self.key_end_back = _btn("key_end_back", "")
        self.key_end_fwd = _btn("key_end_fwd", "")
        self.key_step_beat = _btn("key_step_beat", tr("Boundaries move by one beat"), checkable=True)
        self.key_step_bar = _btn("key_step_bar", tr("Boundaries move by one bar (to the neighboring downbeat)"), checkable=True)
        self.key_step_beat.setChecked(True)
        self._key_step_group = QtWidgets.QButtonGroup(self)
        self._key_step_group.setExclusive(True)
        self._key_step_group.addButton(self.key_step_beat)
        self._key_step_group.addButton(self.key_step_bar)

        # segments and reanalysis
        self.key_split = _btn("beatgrid_split", tr("Split the current key segment at the beat nearest the playhead"))
        self.key_merge_prev = _btn("beatgrid_merge_prev", tr("Merge the previous key segment into the current one (keeps the current key)"))
        self.key_merge_next = _btn("beatgrid_merge_next", tr("Merge the next key segment into the current one (keeps the current key)"))
        self.key_merge_same = _btn("key_merge_same", tr("Merge every run of neighboring segments with the same key"))
        self.key_sel_reanalyze = _btn("beatgrid_reanalyze", tr("Reanalyze the current key segment's key from the audio"))
        self.key_reanalyze = _btn("key_reanalyze_all", tr("Reanalyze the keys of the whole track (dynamic)"))

        # what key changes apply to: the current segment or the whole track
        self.key_scope_all = _btn("beatgrid_scope_all", tr("Key changes (pick, semitone, fifth, relative) apply to the whole track"), checkable=True)
        self.key_scope_segment = _btn("beatgrid_scope_segment", tr("Key changes (pick, semitone, fifth, relative) apply to the current segment"), checkable=True)
        self.key_scope_segment.setChecked(True)
        self._key_scope_group = QtWidgets.QButtonGroup(self)
        self._key_scope_group.setExclusive(True)
        self._key_scope_group.addButton(self.key_scope_all)
        self._key_scope_group.addButton(self.key_scope_segment)

        # the key
        self.key_pick = QtWidgets.QToolButton()
        self.key_pick.setToolTip(tr("Pick the key (Alphanumeric wheel)"))
        self.key_pick.setText("--")
        self._key_picker = KeyPickerPopup(self)
        self._key_picker.picked.connect(self._write_key_scoped)
        # key changes (icons and tooltips follow the scope: _refresh_key_mode_tools)
        self.key_flat = _btn("key_flat", "")
        self.key_sharp = _btn("key_sharp", "")
        self.key_fifth_down = _btn("key_fifth_down", "")
        self.key_fifth_up = _btn("key_fifth_up", "")
        self.key_relative = _btn("key_relative", "", wide=True)

        # listening aids: reference tones in the current key, mixed into the player
        # (also while paused). Held tones follow the key under the playhead and every edit.
        self.key_root = _btn(
            "key_root",
            tr("Root: hold the current key's root (and its octave) under the music.\n"
               "The plainest check of the tonic: it should feel like the note the music rests on."),
            checkable=True,
        )
        self.key_drone = _btn(
            "key_drone",
            tr("Root and fifth: hold the current key's root and fifth under the music.\n"
               "In the right key it sounds at home all the way through; a lasting clash\n"
               "means a wrong tonic (try semitone / fifth steps) or a key change."),
            checkable=True,
        )
        self.key_chord = _btn(
            "key_chord",
            tr("Tonic chord: hold the current key's major / minor triad under the music.\n"
               "Tests the third: if the drone fits but the chord clashes, try the other mode\n"
               "on the same root; if the melody keeps resolving elsewhere, try the relative key."),
            checkable=True,
        )

        self.key_status = LiveLabel(tr("-- | Seg -- | Elapsed: 00:00.00 | Left: 00:00.00"))
        self.key_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        # [scope][step] | [tools 6x3] | [listening aids]: each block three button rows high; the
        # modifiers of the tools' actions all on the left, the listening aids on the right. The
        # tools as on every page (blocks.py): the boundaries and the reanalysis (whole track, then
        # current segment) on top, the key in the middle (picker, semitone / fifth steps), split /
        # merge first in the last row, the relative key at its end as BPM x1/2 / x2 on the
        # beatgrid page. The picker and the relative key (a wide icon) take two cells; the panel
        # sizes them.
        for btn in (self.key_scope_all, self.key_scope_segment):
            btn.setIconSize(QtCore.QSize(26, 26))
        for btn in (self.key_flat, self.key_sharp, self.key_fifth_down, self.key_fifth_up):
            btn.setIconSize(QtCore.QSize(38, 19))
        for btn in (self.key_step_beat, self.key_step_bar):
            btn.setIconSize(QtCore.QSize(46, 23))
        self.key_step_beat.setIcon(badged_icon("key_step_beat", "step_beat"))
        self.key_step_bar.setIcon(badged_icon("key_step_bar", "step_bar"))
        self._mode_pairs += [(self.key_scope_all, self.key_scope_segment), (self.key_step_beat, self.key_step_bar)]
        grid.addWidget(mode_column(self.key_scope_all, self.key_scope_segment), 0, 0, 3, 1)
        grid.addWidget(mode_column(self.key_step_beat, self.key_step_bar), 0, 1, 3, 1)
        grid.addWidget(rule_cell(), 0, 2, 3, 1)
        cells = {
            (0, 3): self.key_start_back, (0, 4): self.key_start_fwd, (0, 5): self.key_end_back,
            (0, 6): self.key_end_fwd, (0, 7): self.key_reanalyze, (0, 8): self.key_sel_reanalyze,
            (1, 5): self.key_flat, (1, 6): self.key_sharp, (1, 7): self.key_fifth_down, (1, 8): self.key_fifth_up,
            (2, 3): self.key_split, (2, 4): self.key_merge_prev, (2, 5): self.key_merge_next,
            (2, 6): self.key_merge_same,
        }
        for (r, c), w in cells.items():
            grid.addWidget(w, r, c, _AL)
        grid.addWidget(self.key_pick, 1, 3, 1, 2, _AL)
        grid.addWidget(self.key_relative, 2, 7, 1, 2, _AL)
        grid.addWidget(rule_cell(), 0, 9, 3, 1)
        grid.addWidget(self.key_root, 0, 10, _AL)
        grid.addWidget(self.key_drone, 1, 10, _AL)
        grid.addWidget(self.key_chord, 2, 10, _AL)
        grid.addWidget(self.key_status, 3, 0, 1, 11, _AL)
        self._key_scoped_tools = {
            "key_flat": self.key_flat, "key_sharp": self.key_sharp, "key_fifth_down": self.key_fifth_down,
            "key_fifth_up": self.key_fifth_up, "key_relative": self.key_relative,
        }
        self._key_scope_group.buttonToggled.connect(self._refresh_key_mode_tools)
        self._key_step_group.buttonToggled.connect(self._refresh_key_mode_tools)
        self._refresh_key_mode_tools()

        self.key_start_back.clicked.connect(lambda: self._nudge_key_boundary("start", -1))
        self.key_start_fwd.clicked.connect(lambda: self._nudge_key_boundary("start", 1))
        self.key_end_back.clicked.connect(lambda: self._nudge_key_boundary("end", -1))
        self.key_end_fwd.clicked.connect(lambda: self._nudge_key_boundary("end", 1))
        self.key_split.clicked.connect(self._split_key_segment)
        self.key_merge_prev.clicked.connect(lambda: self._merge_key_segment(-1))
        self.key_merge_next.clicked.connect(lambda: self._merge_key_segment(1))
        self.key_merge_same.clicked.connect(self._merge_same_key_segments)
        self.key_sel_reanalyze.clicked.connect(self._request_segment_reanalysis_key)
        self.key_reanalyze.clicked.connect(self._request_key_reanalysis_dynamic)
        self.key_pick.clicked.connect(self._open_key_picker)
        self.key_flat.clicked.connect(lambda: self._change_keys(lambda k: (k % 12 - 1) % 12 + 12 * (k >= 12)))
        self.key_sharp.clicked.connect(lambda: self._change_keys(lambda k: (k % 12 + 1) % 12 + 12 * (k >= 12)))
        self.key_fifth_down.clicked.connect(lambda: self._change_keys(lambda k: _alphanumeric_step(k, -1)))
        self.key_fifth_up.clicked.connect(lambda: self._change_keys(lambda k: _alphanumeric_step(k, 1)))
        self.key_relative.clicked.connect(lambda: self._change_keys(_relative_key))
        for aid in (self.key_root, self.key_drone, self.key_chord):
            aid.toggled.connect(lambda on, aid=aid: self._toggle_key_hold(aid, on))

        self._key_cur = -1
        return list(cells.values()) + [self.key_pick, self.key_relative, self.key_scope_all,
                                       self.key_scope_segment, self.key_step_beat, self.key_step_bar]

    def _refresh_key_mode_tools(self, *_args) -> None:
        """The key changes show what they apply to (the scope), the boundary nudges how far
        they move (the step): in the icon's badge and the tooltip."""
        whole = self.key_scope_all.isChecked()
        for name, btn in self._key_scoped_tools.items():
            make = badged_icon_beside if name in KEY_BADGE_BESIDE else badged_icon
            btn.setIcon(make(name, "scope_all" if whole else "scope_segment"))
            btn.setToolTip(tr(KEY_SCOPED_TOOLS[name][0 if whole else 1]))
        self.key_pick.setToolTip(
            tr("Pick one key for the whole track: it becomes one segment in that key (Alphanumeric wheel)") if whole
            else tr("Pick the current segment's key (Alphanumeric wheel)"))
        by_bar = self.key_step_bar.isChecked()
        for btn, name in ((self.key_start_back, "key_start_back"), (self.key_start_fwd, "key_start_fwd"),
                          (self.key_end_back, "key_end_back"), (self.key_end_fwd, "key_end_fwd")):
            btn.setIcon(badged_icon(name, "step_bar" if by_bar else "step_beat"))
            btn.setToolTip(tr(KEY_STEP_TOOLS[name][1 if by_bar else 0]))

    # ---- the current key segment ----

    def _key_index_at(self, t: float) -> int:
        segs = self._key_segments
        if segs.size == 0:
            return -1
        starts, ends = segs[:, 2], segs[:, 3]
        inside = np.nonzero((starts <= t) & (t < ends))[0]
        if inside.size:
            return int(inside[0])
        return 0 if t < starts[0] else len(segs) - 1

    def _update_key_current(self) -> None:
        """Follow the playhead: status line and the picker show the segment under it."""
        if not hasattr(self, "key_status"):
            return
        t = float(self._current_time)
        idx = self._key_index_at(t)
        if idx < 0:
            self._key_cur = -1
            self.key_status.setText(tr("-- | Seg -- | Elapsed: 00:00.00 | Left: 00:00.00"))
            self._show_picker_key(None)
            self._send_key_hold(None)
            return
        start, end = float(self._key_segments[idx, 2]), float(self._key_segments[idx, 3])

        def fmt(sec: float) -> str:
            sec = max(0.0, sec)
            m = int(sec // 60)
            s = sec - 60 * m
            return f"{m:02d}:{s:05.2f}"

        key = self._key_of(idx)
        self.key_status.setText(tr("{key} | Seg {index} | Elapsed: {elapsed} | Left: {left}").format(
            key=key_segment_label(key), index=idx + 1, elapsed=fmt(t - start), left=fmt(end - t)))
        if idx != self._key_cur:
            self._key_cur = idx
            self._show_picker_key(key)
            self._send_key_hold(key)

    def _key_of(self, idx: int) -> int:
        pitch, mode = self._key_segments[idx, 0], self._key_segments[idx, 1]
        return int(pitch) % 12 + (12 if np.isfinite(mode) and int(mode) == 1 else 0)

    def _show_picker_key(self, key: int | None) -> None:
        """The key button shows the key under the playhead, in that key's colour."""
        btn = self.key_pick
        if key is None:
            btn.setText("--")
            btn.setStyleSheet("")
            return
        btn.setText(key_segment_label(key))
        bg = "rgb({}, {}, {})".format(*key_cmap_color(key))
        fg = "rgb({}, {}, {})".format(*key_label_text_color(key))
        btn.setStyleSheet(
            f"QToolButton {{ background-color: {bg}; color: {fg}; border-color: {bg}; font-weight: bold; }}"
            f"QToolButton:hover {{ border-color: {fg}; }}"
        )

    def _open_key_picker(self) -> None:
        idx = self._current_key_row()
        self._key_picker.show_for(self.key_pick, self._key_of(idx) if idx is not None else None)

    def _current_key_row(self) -> int | None:
        idx = self._key_index_at(float(self._current_time))
        return idx if idx >= 0 else None

    # ---- listening aids ----

    def _toggle_key_hold(self, btn: QtWidgets.QToolButton, on: bool) -> None:
        """Root, root and fifth, and chord are one at a time (or none)."""
        if on:
            for other in (self.key_root, self.key_drone, self.key_chord):
                if other is not btn:
                    other.blockSignals(True)
                    other.setChecked(False)
                    other.blockSignals(False)
        idx = self._current_key_row()
        self._send_key_hold(self._key_of(idx) if idx is not None else None)

    def _send_key_hold(self, key: int | None) -> None:
        if not hasattr(self, "key_drone"):
            return
        notes: tuple[int, ...] = ()
        if key is not None and self.key_root.isChecked():
            notes = _root_notes(key)
        elif key is not None and self.key_drone.isChecked():
            notes = _drone_notes(key)
        elif key is not None and self.key_chord.isChecked():
            notes = _tonic_chord_notes(key)
        if notes != getattr(self, "_key_hold_sent", ()):
            self._key_hold_sent = notes
            self.bus.sig_tone_hold.emit(notes)

    # ---- edits ----

    def _set_key_segments_edited(self, segs: np.ndarray) -> None:
        self.model.set_key_segments(segs)

    def _key_scope_whole(self) -> bool:
        return self.key_scope_all.isChecked()

    def _change_keys(self, fn) -> None:
        """Map the key of the current segment, or of every segment, through ``fn``."""
        if self._key_segments.size == 0:
            return
        rows = range(len(self._key_segments)) if self._key_scope_whole() else [self._current_key_row()]
        segs = self._key_segments.copy()
        changed = False
        for idx in rows:
            if idx is None:
                continue
            old = self._key_of(idx)
            new = int(fn(old))
            if new != old:
                segs[idx, 0] = float(new % 12)
                segs[idx, 1] = 1.0 if new >= 12 else 0.0
                changed = True
        if changed:
            self._set_key_segments_edited(segs)

    def _write_key_scoped(self, key: int) -> None:
        """Picked key: the current segment's, or the whole track as one segment in it."""
        if self._key_segments.size == 0:
            return
        if not self._key_scope_whole():
            self._change_keys(lambda _k: key)
            return
        segs = self._key_segments
        whole = np.array([[key % 12, 1.0 if key >= 12 else 0.0, segs[0, 2], segs[-1, 3]]], dtype=float)
        if len(segs) == 1 and self._key_of(0) == key:
            return
        self._set_key_segments_edited(whole)

    def _merge_same_key_segments(self) -> None:
        """Every run of neighbouring segments with one key becomes one segment."""
        segs = self._key_segments
        if len(segs) < 2:
            return
        out = [segs[0].copy()]
        for i in range(1, len(segs)):
            if self._key_of(i) == self._key_of(i - 1):
                out[-1][3] = segs[i, 3]
            else:
                out.append(segs[i].copy())
        if len(out) == len(segs):
            return
        self._set_key_segments_edited(np.vstack(out))

    def _boundary_grid(self, by_bar: bool) -> np.ndarray:
        """Where boundaries may go: every beat, or every downbeat in bar steps."""
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size == 0 or not by_bar:
            return beats
        db = downbeat_beat_indices(self._beatgrid, self._segments)
        return beats[db] if len(db) else beats

    def _step_on_grid(self, t: float, direction: int, by_bar: bool) -> float | None:
        """The grid point one step from boundary ``t`` (key and phrase boundaries).
        A boundary within a quarter step of a grid point counts as on it (stored
        boundaries sit a little off the grid), otherwise the step goes to the next
        grid point that way."""
        grid = self._boundary_grid(by_bar)
        if grid.size == 0:
            return None
        k = int(np.argmin(np.abs(grid - t)))
        spacing = float(np.median(np.diff(grid))) if grid.size > 1 else 0.0
        if abs(grid[k] - t) <= 0.25 * spacing:
            k += direction
        elif (grid[k] > t) != (direction > 0):
            k += direction
        return float(grid[k]) if 0 <= k < grid.size else None

    def _nudge_key_boundary(self, which: str, direction: int) -> None:
        idx = self._current_key_row()
        if idx is None:
            return
        segs = self._key_segments.copy()
        if which == "start":
            if idx == 0:
                return  # the track start does not move
            new = self._step_on_grid(float(segs[idx, 2]), direction, self.key_step_bar.isChecked())
            if new is None or not (segs[idx - 1, 2] < new < segs[idx, 3]):
                return
            segs[idx - 1, 3] = new
            segs[idx, 2] = new
        else:
            if idx == len(segs) - 1:
                return  # the track end does not move
            new = self._step_on_grid(float(segs[idx, 3]), direction, self.key_step_bar.isChecked())
            if new is None or not (segs[idx, 2] < new < segs[idx + 1, 3]):
                return
            segs[idx, 3] = new
            segs[idx + 1, 2] = new
        self._set_key_segments_edited(segs)

    def _split_key_segment(self) -> None:
        idx = self._current_key_row()
        if idx is None:
            return
        t = float(self._current_time)
        beats = np.asarray(self._beatgrid, dtype=float)
        if beats.size:
            t = float(beats[int(np.argmin(np.abs(beats - t)))])
        segs = self._key_segments
        start, end = float(segs[idx, 2]), float(segs[idx, 3])
        if not (start + 1e-3 < t < end - 1e-3):
            return
        first, second = segs[idx].copy(), segs[idx].copy()
        first[3] = t
        second[2] = t
        self._set_key_segments_edited(np.vstack([segs[:idx], first, second, segs[idx + 1:]]))

    def _merge_key_segment(self, direction: int) -> None:
        idx = self._current_key_row()
        if idx is None:
            return
        segs = self._key_segments.copy()
        other = idx + direction
        if not 0 <= other < len(segs):
            return
        if direction < 0:
            segs[idx, 2] = segs[other, 2]
        else:
            segs[idx, 3] = segs[other, 3]
        self._set_key_segments_edited(np.delete(segs, other, axis=0))

    # ---- reanalysis ----

    def _request_segment_reanalysis_key(self) -> None:
        """Reanalyze the current key segment's span (static key)."""
        idx = self._current_key_row()
        if idx is None or not self._has_active_segment():
            return
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        start, end = float(self._key_segments[idx, 2]), float(self._key_segments[idx, 3])
        if end - start <= 1e-6:
            return
        self.bus.sig_segment_reanalyze_requested.emit(ReanalysisRequest(
            kind=KEY_STATIC, path=self._current_track_path(), segment_index=int(self._cur_idx),
            tempo_segments=np.asarray(self._segments, dtype=float).copy(),
            beats=np.asarray(self._beatgrid, dtype=float).copy(),
            key_segments=np.asarray(self._key_segments, dtype=float).copy(), selection=(start, end),
            phrases=tuple(assigned_phrases(self._phrase_segments)), duration_sec=float(self.duration or 0.0)))

    def _request_key_reanalysis_dynamic(self) -> None:
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        self.bus.sig_segment_reanalyze_requested.emit(ReanalysisRequest(
            kind=KEY_DYNAMIC, path=self._current_track_path(), segment_index=int(self._cur_idx),
            tempo_segments=np.asarray(self._segments, dtype=float).copy(),
            beats=np.asarray(self._beatgrid, dtype=float).copy(),
            key_segments=np.asarray(self._key_segments, dtype=float).copy(),
            # the key analysis reads the progressions per phrase
            phrases=tuple(assigned_phrases(self._phrase_segments))))


def _alphanumeric_step(key: int, step: int) -> int:
    """Neighbour on the Alphanumeric wheel (circle of fifths), same letter (major / minor)."""
    label = ALPHANUMERIC_LABELS[key]
    n = (int(label[:-1]) - 1 + step) % 12 + 1
    return ALPHANUMERIC_INDEX[(n, label[-1])]


# Reference tones (MIDI notes) around the C3..B4 range: under most melodies, above the bass.
def _root_midi(key: int) -> int:
    return 48 + key % 12


def _root_notes(key: int) -> tuple[int, ...]:
    """The root and the root an octave up."""
    r = _root_midi(key)
    return (r, r + 12)


def _drone_notes(key: int) -> tuple[int, ...]:
    """Root, fifth and the root an octave up: no third, so it does not take a mode."""
    r = _root_midi(key)
    return (r, r + 7, r + 12)


def _triad(root: int, minor: bool) -> tuple[int, ...]:
    return (root, root + (3 if minor else 4), root + 7)


def _tonic_chord_notes(key: int) -> tuple[int, ...]:
    """Root in the bass and the triad an octave up."""
    r = _root_midi(key)
    return (r,) + _triad(r + 12, key >= 12)


def _relative_key(key: int) -> int:
    """Major <-> its relative minor (C <-> Am): same Alphanumeric number, other letter."""
    return (key + 9) % 12 + 12 if key < 12 else (key % 12 + 3) % 12
