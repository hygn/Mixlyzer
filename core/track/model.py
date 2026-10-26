"""The loaded track: its one owner. `DataModel` holds what the deck's track is (its stored analysis, properties,
album art, decoded audio) and its marks as edited (the beatgrid and tempo rows, the keys, the phrases, the
JumpCUEs, the CUE points) with their undo history; everything that changes them goes through its methods, and each
change is told by its own signal at once:

    load()                    a track or its analysis           sig_loaded, sig_properties_changed, sig_album_art_changed
    set_beatgrid()            beats and tempo rows edited       sig_beatgrid_changed
    set_key_segments()        keys edited                       sig_keys_changed
    set_phrases()             phrases edited                    sig_phrases_changed
    set_jump_cues()           JumpCUEs edited                   sig_jump_cues_changed
    set_cue_points()          CUE points edited                 sig_cue_points_changed
    set_automatic()           a reanalysis's automatic result   (saved with the edits)
    undo() / redo()           a snapshot back / on              every mark's signal
    mark_saved()              the edits stored                  sig_edits_changed (as every edit: undo / redo, unsaved)

`analysis` is the track's StoredAnalysis as loaded (core.analysis_library.stored: frozen; the automatic results a
reanalysis gives replace their fields), the marks as edited are the model's own (`beats`, `tempo_segments`, ...);
`properties` is a read-only dict (a write raises). The marks' arrays are read-only too, and the objects the model
holds are replaced, never changed in place (an editor copies, changes, and sets)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtGui import QImage

from core.track import marks
from core.track.history import History, Marks
from analyzer.music.marks import CuePoint, JumpCueGroup, Phrase
from core.analysis_library.jump_cues import has_slot_letters
from core.analysis_library.phrases import assigned_phrases
from core.analysis_library.stored import StoredAnalysis, TrackEdits
from system.i18n import tr


class _ReadOnly(dict):
    """A dict the model alone writes (dict's own methods, on the class): anyone else's write raises."""

    def _refuse(self, *_args, **_kwargs):
        raise TypeError("the loaded track is DataModel's: change it through its methods (DataModel.set_*, load)")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _refuse


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


class DataModel(QtCore.QObject):
    sig_loaded = QtCore.Signal()               # a track (or its analysis): everything read again
    sig_properties_changed = QtCore.Signal()
    sig_album_art_changed = QtCore.Signal()
    sig_audio_changed = QtCore.Signal()        # its length or its decoded audio
    sig_beatgrid_changed = QtCore.Signal()
    sig_keys_changed = QtCore.Signal()
    sig_phrases_changed = QtCore.Signal()
    sig_jump_cues_changed = QtCore.Signal()
    sig_cue_points_changed = QtCore.Signal()
    sig_edits_changed = QtCore.Signal()        # undo / redo possible, unsaved edits

    def __init__(self):
        super().__init__()
        self._analysis = StoredAnalysis()
        # a reanalysis's automatic results, saved with the edits (None: none given since the track was loaded)
        self._keys_auto: Optional[np.ndarray] = None
        self._jump_cues_auto: Optional[tuple[JumpCueGroup, ...]] = None
        self._properties: _ReadOnly = _ReadOnly()
        self._duration_sec: float = 0.0
        self.album_art: Optional[QImage] = None
        self.predecoded_pcm: Any = None
        self.predecoded_rate: int = 0
        self._set_marks(marks.from_stored(self._analysis, 0.0))
        self._history = History.starting_at(self._snapshot())
        self._unsaved = False

    # ---- what the track is ----
    @property
    def analysis(self) -> StoredAnalysis:
        """The track's stored analysis as loaded (the marks as edited are the model's own properties)."""
        return self._analysis

    @property
    def properties(self) -> dict:
        return self._properties

    @property
    def duration_sec(self) -> float:
        return self._duration_sec

    def load(self, analysis: StoredAnalysis, properties: dict, duration_sec: float,
             album_art: Optional[QImage] = None, keep_audio: bool = False) -> None:
        """A track's data, its marks read from its stored analysis and its edit history started; ``keep_audio``: the
        decoded audio kept (the same track's analysis come in while it plays), else dropped (another track: its own
        comes once decoded)."""
        self._analysis = analysis
        self._keys_auto = self._jump_cues_auto = None
        self._properties = _ReadOnly(properties or {})
        self._duration_sec = float(duration_sec or 0.0)
        self.album_art = album_art
        if not keep_audio:
            self.predecoded_pcm = None
            self.predecoded_rate = 0
        self._set_marks(marks.from_stored(self._analysis, self._duration_sec))
        self._history = History.starting_at(self._snapshot())
        self._unsaved = False
        print(f"[DataModel] Summary {self.debug_summary()}")
        self.sig_loaded.emit()
        self.sig_audio_changed.emit()
        self.sig_properties_changed.emit()
        self.sig_album_art_changed.emit()
        self.sig_edits_changed.emit()

    def set_decoded_duration(self, duration_sec: float) -> None:
        """The decoded track's length, for the views until the analysis gives its own."""
        self._duration_sec = float(duration_sec)
        self.sig_audio_changed.emit()
        self.sig_properties_changed.emit()

    def set_predecoded_audio(self, pcm: Any, sample_rate: int) -> None:
        try:
            self.predecoded_pcm = np.asarray(pcm, dtype=np.float32, order="C")
        except Exception:
            self.predecoded_pcm = pcm
        self.predecoded_rate = int(sample_rate)
        self.sig_audio_changed.emit()

    # ---- its marks (held objects: replaced by the set_* methods, never changed in place) ----
    @property
    def beats(self) -> np.ndarray:
        return self._beats

    @property
    def tempo_segments(self) -> np.ndarray:
        """(N, 6) ``[start, end, bpm, inizio, ts_num, bpm_end]``."""
        return self._tempo_segments

    @property
    def key_segments(self) -> np.ndarray:
        """(N, 4) ``[pitch, minor, start, end]``."""
        return self._key_segments

    @property
    def phrases(self) -> list[Phrase]:
        """The phrases covering the track (analyzer.music.marks.Phrase; its stretches without one UNASSIGNED)."""
        return self._phrases

    @property
    def jump_cues(self) -> list[JumpCueGroup]:
        """The JumpCUE groups (analyzer.music.marks.JumpCueGroup: cues that jump between each other, a colour
        preset each), each one's cues by point."""
        return self._jump_cues

    @property
    def cue_points(self) -> list[CuePoint]:
        """The CUE points by time (analyzer.music.marks.CuePoint)."""
        return self._cue_points

    # ---- edits ----
    def set_beatgrid(self, beats, tempo_segments) -> None:
        self._beats = _frozen(np.asarray(beats, dtype=float).copy())
        self._tempo_segments = _frozen(np.asarray(tempo_segments, dtype=float).copy())
        self._edited(self.sig_beatgrid_changed)

    def set_key_segments(self, rows) -> None:
        self._key_segments = _frozen(marks.key_rows(rows))
        self._edited(self.sig_keys_changed)

    def set_phrases(self, rows, *, normalize: bool = False) -> None:
        """``normalize``: rows found (a reanalysis's), ordered and cleaned first; an edit's are taken as they are."""
        self._phrases = marks.phrase_rows(rows, self._duration_sec, normalize=normalize)
        self._edited(self.sig_phrases_changed)

    def set_jump_cues(self, groups) -> None:
        """ValueError (nothing changed) when a cue's letter is not a slot's (A-Z, no two alike)."""
        groups = marks.jump_cue_group_rows(groups)
        if not has_slot_letters(groups):
            raise ValueError(tr("JumpCUE labels must be unique single letters A-Z."))
        self._jump_cues = groups
        self._edited(self.sig_jump_cues_changed)

    def set_cue_points(self, points) -> None:
        self._cue_points = marks.cue_point_rows(points)
        self._edited(self.sig_cue_points_changed)

    def set_automatic(self, *, key_segments=None, jump_cues=None) -> None:
        """A reanalysis's automatic result (the keys / the JumpCUE groups: what the optimizers compare the user's
        edits with), saved with the edits; not an edit itself (no undo)."""
        if key_segments is None and jump_cues is None:
            return
        if key_segments is not None:
            self._keys_auto = _frozen(np.asarray(key_segments, dtype=float).copy())
            self._analysis = replace(self._analysis, key_segments_auto=self._keys_auto)
        if jump_cues is not None:
            self._jump_cues_auto = tuple(jump_cues)
            self._analysis = replace(self._analysis, jump_cues_auto=self._jump_cues_auto)
        self._unsaved = True
        self.sig_edits_changed.emit()

    def can_undo(self) -> bool:
        return self._history.can_undo()

    def can_redo(self) -> bool:
        return self._history.can_redo()

    @property
    def unsaved(self) -> bool:
        return self._unsaved

    def undo(self) -> None:
        snapshot = self._history.undo()
        if snapshot is not None:
            self._restore(snapshot)

    def redo(self) -> None:
        snapshot = self._history.redo()
        if snapshot is not None:
            self._restore(snapshot)

    def mark_saved(self) -> None:
        self._unsaved = False
        self.sig_edits_changed.emit()

    def stored_edits(self) -> TrackEdits:
        """The edits as the analysis library saves them (core.analysis_library.jobs.save_edits), with the automatic
        results a reanalysis gave; ValueError when a JumpCUE's letter is not a slot's."""
        return TrackEdits(beats=self._beats, tempo_segments=self._tempo_segments, key_segments=self._key_segments,
                          # UNASSIGNED is not stored: it is where no phrase is
                          phrases=tuple(assigned_phrases(self._phrases)), jump_cues=tuple(self._jump_cues),
                          cue_points=tuple(self._cue_points), key_segments_auto=self._keys_auto,
                          jump_cues_auto=self._jump_cues_auto)

    # ---- inside ----
    def _set_marks(self, m: dict) -> None:
        self._beats = _frozen(np.array(m["beats"], dtype=float))
        self._tempo_segments = _frozen(np.array(m["tempo_segments"], dtype=float))
        self._key_segments = _frozen(np.array(m["key_segments"], dtype=float))
        self._phrases = m["phrases"]
        self._jump_cues = m["jump_cues"]
        self._cue_points = m["cue_points"]

    def _snapshot(self) -> Marks:
        return Marks(self._beats, self._tempo_segments, self._key_segments, self._phrases, self._jump_cues,
                     self._cue_points)

    def _edited(self, changed: QtCore.SignalInstance) -> None:
        self._history.push(self._snapshot())
        self._unsaved = True
        changed.emit()
        self.sig_edits_changed.emit()

    def _restore(self, s: Marks) -> None:
        self._set_marks({"beats": s.beats, "tempo_segments": s.tempo_segments, "key_segments": s.key_segments,
                         "phrases": s.phrases, "jump_cues": s.jump_cues, "cue_points": s.cue_points})
        self._unsaved = True
        for sig in (self.sig_beatgrid_changed, self.sig_keys_changed, self.sig_jump_cues_changed,
                    self.sig_phrases_changed, self.sig_cue_points_changed, self.sig_edits_changed):
            sig.emit()

    def debug_summary(self) -> str:
        """A human-readable snapshot of the model, for the console."""
        a = self._analysis
        return (
            f"DataModel(beats={len(self._beats)}, tempo rows={len(self._tempo_segments)}, "
            f"keys={len(self._key_segments)}, phrases={len(a.phrases)}, JumpCUE groups={len(a.jump_cues)}, "
            f"CUE points={len(a.cue_points)}, overview={'set' if a.wave is not None else 'none'}, "
            f"duration={self._duration_sec:.3f}s, "
            f"title={self._properties.get('title')}, "
            f"artist={self._properties.get('artist')}, "
            f"album_art={'set' if self.album_art else 'none'})"
        )
