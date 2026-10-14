from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MAX_LOG_HISTORY = 50


@dataclass
class logElement:
    beat_segments: np.ndarray
    beats_time: np.ndarray
    key_segments: np.ndarray | None = None
    jump_cues: list[dict] | None = None
    phrases: list[dict] | None = None
    cue_points: list[dict] | None = None

    def __post_init__(self) -> None:
        self.beat_segments = np.asarray(self.beat_segments, dtype=float).copy()
        self.beats_time = np.asarray(self.beats_time, dtype=float).copy()
        if self.key_segments is not None:
            self.key_segments = np.asarray(self.key_segments, dtype=float).copy()
        if self.jump_cues is not None:
            self.jump_cues = copy.deepcopy(self.jump_cues)
        if self.phrases is not None:
            self.phrases = copy.deepcopy(self.phrases)
        if self.cue_points is not None:
            self.cue_points = copy.deepcopy(self.cue_points)

    def clone(self) -> logElement:
        return logElement(
            beat_segments=self.beat_segments,
            beats_time=self.beats_time,
            key_segments=self.key_segments,
            jump_cues=self.jump_cues,
            phrases=self.phrases,
            cue_points=self.cue_points,
        )


@dataclass
class logs:
    default: logElement
    log: list[logElement] = field(default_factory=list)
    relidx: int = -1

    def __post_init__(self) -> None:
        default_entry = self.default.clone()
        self.default = default_entry
        if not self.log:
            self.log = [default_entry]
        else:
            self.log = [entry.clone() for entry in self.log]
        self.relidx = len(self.log) - 1

    def push(self, loge: logElement) -> None:
        entry = loge.clone()
        if self.relidx < len(self.log) - 1:
            self.log = self.log[: self.relidx + 1]
        self.log.append(entry)
        if len(self.log) > MAX_LOG_HISTORY:
            drop = len(self.log) - MAX_LOG_HISTORY
            self.log = self.log[drop:]
        self.relidx = len(self.log) - 1

    def undo(self) -> Optional[logElement]:
        if len(self.log) <= 1:
            self.relidx = len(self.log) - 1
            return None
        if self.relidx <= 0:
            self.relidx = 0
            return None
        self.relidx -= 1
        return self.log[self.relidx].clone()

    def redo(self) -> Optional[logElement]:
        if not self.log:
            self.relidx = -1
            return None
        if self.relidx >= len(self.log) - 1:
            self.relidx = len(self.log) - 1
            return None
        self.relidx += 1
        return self.log[self.relidx].clone()

    def can_undo(self) -> bool:
        return len(self.log) > 1 and self.relidx > 0

    def can_redo(self) -> bool:
        return 0 <= self.relidx < len(self.log) - 1
