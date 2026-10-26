"""The loaded track's edit history (core.track.model): a snapshot of its marks after each edit, undone and redone by
moving along them. The model alone keeps it."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from analyzer.music.marks import CuePoint, JumpCueGroup, Phrase

MAX_HISTORY = 50


@dataclass
class Marks:
    """The edited marks at one moment, as the model holds them (copies: a snapshot never changes)."""
    beats: np.ndarray
    tempo_segments: np.ndarray
    key_segments: np.ndarray
    phrases: list[Phrase]
    jump_cues: list[JumpCueGroup]
    cue_points: list[CuePoint]

    def __post_init__(self) -> None:
        self.beats = np.asarray(self.beats, dtype=float).copy()
        self.tempo_segments = np.asarray(self.tempo_segments, dtype=float).copy()
        self.key_segments = np.asarray(self.key_segments, dtype=float).copy()
        self.phrases = copy.deepcopy(self.phrases)
        self.jump_cues = copy.deepcopy(self.jump_cues)
        self.cue_points = copy.deepcopy(self.cue_points)

    def clone(self) -> Marks:
        return Marks(self.beats, self.tempo_segments, self.key_segments, self.phrases, self.jump_cues,
                     self.cue_points)


@dataclass
class History:
    """The snapshots since the track was loaded (the first: as loaded), the one shown at ``index``."""
    entries: list[Marks] = field(default_factory=list)
    index: int = -1

    @classmethod
    def starting_at(cls, marks: Marks) -> History:
        return cls([marks.clone()], 0)

    def push(self, marks: Marks) -> None:
        """An edit: the snapshots after the one shown dropped (no redo past an edit), this one added."""
        self.entries = self.entries[: self.index + 1] + [marks.clone()]
        if len(self.entries) > MAX_HISTORY:
            self.entries = self.entries[len(self.entries) - MAX_HISTORY:]
        self.index = len(self.entries) - 1

    def undo(self) -> Optional[Marks]:
        if not self.can_undo():
            return None
        self.index -= 1
        return self.entries[self.index].clone()

    def redo(self) -> Optional[Marks]:
        if not self.can_redo():
            return None
        self.index += 1
        return self.entries[self.index].clone()

    def can_undo(self) -> bool:
        return self.index > 0

    def can_redo(self) -> bool:
        return 0 <= self.index < len(self.entries) - 1
