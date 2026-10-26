"""An editor's partial reanalysis of the loaded track, as the editor asks it and gets it back: the contract between
the editor (ui.edit_panel), its workflow (core.workflows.segment_reanalysis) and the worker process that runs it
(core.workers.segment_reanalysis). Plain values only: importing this imports none of the analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from analyzer.music.marks import JumpCueGroup, Phrase

# what is read again
BEAT = "beat"                  # a tempo segment (its row)
KEY_STATIC = "key_static"      # the key of a span
KEY_DYNAMIC = "key_dynamic"    # the keys of the whole track
JUMPCUE = "jumpcue"            # the JumpCUEs of the whole track
PHRASE = "phrase"              # the phrases of the whole track


@dataclass(frozen=True)
class ReanalysisRequest:
    """An editor's partial reanalysis: what is read again (``kind``) and the track's current marks it reads."""
    kind: str
    beats: np.ndarray
    tempo_segments: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    path: Optional[str] = None                  # None: the deck's track
    segment_index: int = 0                      # BEAT: the row read again
    prev_bpm: Optional[float] = None            # BEAT: the reference BPM
    key_segments: Optional[np.ndarray] = None   # KEY_*: the current keys
    selection: Optional[tuple[float, float]] = None   # KEY_STATIC: the span (s)
    phrases: Optional[tuple[Phrase, ...]] = None      # KEY_*, JUMPCUE: the current phrases (UNASSIGNED left out)
    duration_sec: float = 0.0


@dataclass(frozen=True)
class ReanalysisResult:
    """A partial reanalysis's result: the marks it gives (None: not this kind's)."""
    beats: Optional[np.ndarray] = None
    tempo_segments: Optional[np.ndarray] = None
    key_segments: Optional[np.ndarray] = None
    key_segments_auto: Optional[np.ndarray] = None    # KEY_DYNAMIC: the new automatic result
    jump_cues: Optional[tuple[JumpCueGroup, ...]] = None   # also the new automatic result
    phrases: Optional[list[Phrase]] = None
