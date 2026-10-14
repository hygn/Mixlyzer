from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from core.beat_geometry import downbeat_beat_indices
from ui.edit_panel.history import logElement
from ui.edit_panel.icons import icon
from ui.edit_panel.key_picker import CAMELOT_INDEX, KeyPickerPopup
from ui.edit_panel.styles import WARNING_BUTTON_STYLESHEET
from ui.theme import current_theme
from utils.color import key_cmap_color, key_label_text_color
from utils.keystrip import key_segment_label
from utils.labels import CAMELOT_LABELS


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

        # boundaries, by a beat or a bar
        self.key_start_back = _btn("key_start_back", "Move the current key segment's start earlier")
        self.key_start_fwd = _btn("key_start_fwd", "Move the current key segment's start later")
        self.key_end_back = _btn("key_end_back", "Move the current key segment's end earlier")
        self.key_end_fwd = _btn("key_end_fwd", "Move the current key segment's end later")
        self.key_step_beat = _btn("key_step_beat", "Boundaries move by one beat", checkable=True, wide=True)
        self.key_step_bar = _btn("key_step_bar", "Boundaries move by one bar (to the neighbouring downbeat)", checkable=True, wide=True)
        self.key_step_beat.setChecked(True)
        self._key_step_group = QtWidgets.QButtonGroup(self)
        self._key_step_group.setExclusive(True)
        self._key_step_group.addButton(self.key_step_beat)
        self._key_step_group.addButton(self.key_step_bar)

        # segments and reanalysis
        self.key_split = _btn("beatgrid_split", "Split the current key segment at the beat nearest the playhead")
        self.key_merge_prev = _btn("beatgrid_merge_prev", "Merge the previous key segment into the current one (keeps the current key)")
        self.key_merge_next = _btn("beatgrid_merge_next", "Merge the next key segment into the current one (keeps the current key)")
        self.key_merge_same = _btn("key_merge_same", "Merge every run of neighbouring segments with the same key")
        self.key_sel_reanalyze = _btn("beatgrid_reanalyze", "Reanalyze the current key segment's key from the audio")
        self.key_reanalyze = _btn("key_reanalyze_all", "Reanalyze the keys of the whole track (dynamic)")

        # what key changes apply to: the current segment or the whole track
        self.key_scope_all = _btn("beatgrid_scope_all", "Key changes (pick, semitone, fifth, relative) apply to the whole track", checkable=True)
        self.key_scope_segment = _btn("beatgrid_scope_segment", "Key changes (pick, semitone, fifth, relative) apply to the current segment", checkable=True)
        self.key_scope_segment.setChecked(True)
        self._key_scope_group = QtWidgets.QButtonGroup(self)
        self._key_scope_group.setExclusive(True)
        self._key_scope_group.addButton(self.key_scope_all)
        self._key_scope_group.addButton(self.key_scope_segment)

        # the key
        self.key_pick = QtWidgets.QToolButton()
        self.key_pick.setToolTip("Pick the key (Camelot wheel)")
        self.key_pick.setText("--")
        self._key_picker = KeyPickerPopup(self)
        self._key_picker.picked.connect(self._write_key_scoped)
        self.key_flat = _btn("key_flat", "A semitone down")
        self.key_sharp = _btn("key_sharp", "A semitone up")
        self.key_fifth_down = _btn("key_fifth_down", "One step down the circle of fifths (Camelot -1)")
        self.key_fifth_up = _btn("key_fifth_up", "One step up the circle of fifths (Camelot +1)")
        self.key_relative = _btn("key_relative", "Relative major / minor (C <-> Am)", wide=True)

        # listening aids: reference tones in the current key, mixed into the player
        # (also while paused). Held tones follow the key under the playhead and every edit.
        self.key_drone = _btn(
            "key_drone",
            "Tonic drone: hold the current key's root and fifth under the music.\n"
            "In the right key it sounds at home all the way through; a lasting clash\n"
            "means a wrong tonic (try semitone / fifth steps) or a key change.",
            checkable=True,
        )
        self.key_chord = _btn(
            "key_chord",
            "Tonic chord: hold the current key's major / minor triad under the music.\n"
            "Tests the third: if the drone fits but the chord clashes, try the other mode\n"
            "on the same root; if the melody keeps resolving elsewhere, try the relative key.",
            checkable=True,
        )
        self.key_cadence = _btn(
            "key_cadence",
            "Cadence: play I - IV - V7 - I (minor: i - iv - V7 - i) in the current key at the\n"
            "track's tempo. The last chord should feel like arriving home in the music.",
            checkable=True,
            wide=True,
        )
        # lit (checked) while the cadence sounds; a click while lit plays it again
        self._key_cadence_timer = QtCore.QTimer(self)
        self._key_cadence_timer.setSingleShot(True)
        self._key_cadence_timer.timeout.connect(lambda: self.key_cadence.setChecked(False))

        self.key_status = QtWidgets.QLabel("-- | Seg -- | Elapsed: 00:00.00 | Left: 00:00.00")
        self.key_status.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        _AL = QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        # Rows by how often they are needed: the current key first (picker, then
        # semitone / fifth / relative nudges), its boundaries (start pair, end
        # pair, step), then the segment structure and reanalysis. The scope pair
        # (whole track / current segment) stands on its own in the last column.
        cells = {
            (0, 2): self.key_flat, (0, 3): self.key_sharp, (0, 4): self.key_fifth_down,
            (0, 5): self.key_fifth_up, (0, 6): self.key_relative,
            (1, 0): self.key_start_back, (1, 1): self.key_start_fwd, (1, 2): self.key_end_back,
            (1, 3): self.key_end_fwd, (1, 4): self.key_step_beat, (1, 5): self.key_step_bar,
            (1, 6): self.key_scope_all,
            (2, 0): self.key_split, (2, 1): self.key_merge_prev, (2, 2): self.key_merge_next,
            (2, 3): self.key_merge_same, (2, 4): self.key_sel_reanalyze, (2, 5): self.key_reanalyze,
            (2, 6): self.key_scope_segment,
        }
        for (r, c), w in cells.items():
            grid.addWidget(w, r, c, _AL)
        grid.addWidget(self.key_pick, 0, 0, 1, 2, _AL)  # two cells; the panel sizes it
        # one rule between the edit block and the listening aids, as on the beatgrid page
        aid_rule = QtWidgets.QFrame()
        aid_rule.setFixedWidth(1)
        aid_rule.setStyleSheet(f"background: {current_theme().border};")
        aid_rule_cell = QtWidgets.QWidget()
        rule_lay = QtWidgets.QHBoxLayout(aid_rule_cell)
        rule_lay.setContentsMargins(4, 2, 4, 2)
        rule_lay.addWidget(aid_rule)
        grid.addWidget(aid_rule_cell, 0, 7, 3, 1)
        grid.addWidget(self.key_drone, 0, 8, _AL)
        grid.addWidget(self.key_chord, 1, 8, _AL)
        grid.addWidget(self.key_cadence, 2, 8, _AL)
        grid.addWidget(self.key_status, 3, 0, 1, 9, _AL)

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
        self.key_fifth_down.clicked.connect(lambda: self._change_keys(lambda k: _camelot_step(k, -1)))
        self.key_fifth_up.clicked.connect(lambda: self._change_keys(lambda k: _camelot_step(k, 1)))
        self.key_relative.clicked.connect(lambda: self._change_keys(_relative_key))
        self.key_drone.toggled.connect(lambda on: self._toggle_key_hold(self.key_drone, on))
        self.key_chord.toggled.connect(lambda on: self._toggle_key_hold(self.key_chord, on))
        self.key_cadence.clicked.connect(self._play_key_cadence)

        self._key_cur = -1
        return list(cells.values()) + [self.key_pick]

    # ---- data in / out ----

    def set_key_segments(self, key_segments, *, initialize: bool = False) -> None:
        if key_segments is None:
            self._key_segments = np.empty((0, 4), dtype=float)
        else:
            arr = np.asarray(key_segments, dtype=float)
            if arr.ndim == 1 and arr.size % 4 == 0:
                arr = arr.reshape((-1, 4))
            if arr.ndim != 2 or arr.shape[1] < 4:
                self._key_segments = np.empty((0, 4), dtype=float)
            else:
                arr = arr[:, :4].copy()
                self._key_segments = arr[np.argsort(arr[:, 2], kind="stable")]
        self._key_cur = -1
        self._update_key_current()
        if initialize:
            return
        self._emit_key_update()

    def _emit_key_update(self) -> None:
        self.edit_log.push(
            logElement(
                beats_time=self._beatgrid,
                beat_segments=self._segments,
                key_segments=self._key_segments,
                jump_cues=self._JumpCUE,
                phrases=self._phrase_segments,
                cue_points=self._cue_points,
            )
        )
        self.btn_save.setStyleSheet(WARNING_BUTTON_STYLESHEET)
        self._broadcast_key_segments()
        self._update_undo_redo_buttons()

    def _broadcast_key_segments(self) -> None:
        """Publish the key segments; views render the key strip from them."""
        if self.model is not None:
            try:
                self.model.features["key_segments"] = np.asarray(self._key_segments, dtype=float)
            except Exception:
                pass
        self._key_cur = -1
        self._update_key_current()
        self.bus.sig_key_segments_updated.emit()

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
            self.key_status.setText("-- | Seg -- | Elapsed: 00:00.00 | Left: 00:00.00")
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
        self.key_status.setText(
            f"{key_segment_label(key)} | Seg {idx + 1} | Elapsed: {fmt(t - start)} | Left: {fmt(end - t)}"
        )
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
        """Drone and chord are one at a time (or neither)."""
        if on:
            other = self.key_chord if btn is self.key_drone else self.key_drone
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        idx = self._current_key_row()
        self._send_key_hold(self._key_of(idx) if idx is not None else None)

    def _send_key_hold(self, key: int | None) -> None:
        if not hasattr(self, "key_drone"):
            return
        notes: tuple[int, ...] = ()
        if key is not None and self.key_drone.isChecked():
            notes = _drone_notes(key)
        elif key is not None and self.key_chord.isChecked():
            notes = _tonic_chord_notes(key)
        if notes != getattr(self, "_key_hold_sent", ()):
            self._key_hold_sent = notes
            self.bus.sig_tone_hold.emit(notes)

    def _play_key_cadence(self) -> None:
        idx = self._current_key_row()
        if idx is None:
            self.key_cadence.setChecked(False)
            return
        beat = self._beat_length_at(float(self._current_time))
        steps = _cadence_steps(self._key_of(idx), beat)
        self.bus.sig_tone_sequence.emit(steps)
        self.key_cadence.setChecked(True)
        self._key_cadence_timer.start(int(1000 * sum(sec for _notes, sec in steps)))

    def _beat_length_at(self, t: float) -> float:
        """Local beat length (s) from the beatgrid, kept between 0.35 and 0.75 s
        (half / double time) so the cadence is neither rushed nor dragging."""
        beats = np.asarray(self._beatgrid, dtype=float)
        beat = 0.5
        if beats.size >= 2:
            i = int(np.clip(np.searchsorted(beats, t), 1, beats.size - 1))
            d = np.diff(beats[max(0, i - 4):i + 4])
            d = d[d > 0]
            if d.size:
                beat = float(np.median(d))
        while beat < 0.35:
            beat *= 2.0
        while beat > 0.75:
            beat *= 0.5
        return beat

    # ---- edits ----

    def _set_key_segments_edited(self, segs: np.ndarray) -> None:
        self._key_segments = np.asarray(segs, dtype=float)
        self._emit_key_update()

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
        payload = {
            "path": self._current_track_path(),
            "segment_index": int(self._cur_idx),
            "segments": np.asarray(self._segments, dtype=float).copy(),
            "beats_time_sec": np.asarray(self._beatgrid, dtype=float).copy(),
            "key_segments": np.asarray(self._key_segments, dtype=float).copy(),
            "selection": (start, end),
            "analyze_type": "key_static",
            "duration_sec": self.duration,
        }
        self.bus.sig_segment_reanalyze_requested.emit(payload)

    def _request_key_reanalysis_dynamic(self) -> None:
        if self._beatgrid is None or len(self._beatgrid) == 0:
            return
        payload = {
            "path": self._current_track_path(),
            "segment_index": int(self._cur_idx),
            "segments": np.asarray(self._segments, dtype=float).copy(),
            "beats_time_sec": np.asarray(self._beatgrid, dtype=float).copy(),
            "key_segments": np.asarray(self._key_segments, dtype=float).copy(),
            "analyze_type": "key_dynamic",
        }
        self.bus.sig_segment_reanalyze_requested.emit(payload)


def _camelot_step(key: int, step: int) -> int:
    """Neighbour on the Camelot wheel (circle of fifths), same letter (major / minor)."""
    label = CAMELOT_LABELS[key]
    n = (int(label[:-1]) - 1 + step) % 12 + 1
    return CAMELOT_INDEX[(n, label[-1])]


# Reference tones (MIDI notes) around the C3..B4 range: under most melodies, above the bass.
def _root_midi(key: int) -> int:
    return 48 + key % 12


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


def _cadence_steps(key: int, beat: float) -> list[tuple[tuple[int, ...], float]]:
    """I - IV - V7 - I (minor: i - iv - V7 - i, the harmonic minor's dominant), two beats
    each and the tonic held for four; upper voices kept close to the tonic triad."""
    minor = key >= 12
    r = _root_midi(key)
    tonic = _triad(r + 12, minor)
    sub = (r + 12, r + 17, r + (20 if minor else 21))                # IV / iv: 1, 4, b6 / 6
    dom = (r + 11, r + 14, r + 17, r + 19)                           # V7: 7, 2, 4, 5
    return [
        ((r,) + tonic, 2 * beat),
        ((r + 5,) + sub, 2 * beat),
        ((r + 7 - 12,) + dom, 2 * beat),
        ((r,) + tonic, 4 * beat),
    ]


def _relative_key(key: int) -> int:
    """Major <-> its relative minor (C <-> Am): same Camelot number, other letter."""
    return (key + 9) % 12 + 12 if key < 12 else (key % 12 + 3) % 12
