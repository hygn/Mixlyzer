"""The library's tracks for this Reoptimize run, given by the library's owner when the run starts
(core.analysis_library.jobs.training_tracks, handed to the optimizer process by core.workers.optimizer): the
optimizers list their tracks from these, each with the user's annotation of it (``annotation``, an `Annotation`), and
read its audio by the path given; they never open the library's files (library.db, the stored analyses)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Annotation:
    """What the optimizers read of a track's stored analysis: the user's annotation and the automatic results kept
    beside it. None where the track has none stored."""
    beats: np.ndarray | None = None
    sample_rate: int | None = None
    duration_sec: float | None = None
    tempo_segments: np.ndarray | None = None       # the rows as stored
    key_segments: np.ndarray | None = None
    key_segments_auto: np.ndarray | None = None    # the automatic key result (the user's keys compared with it)
    phrase_starts: np.ndarray | None = None        # the phrases as stored (UNASSIGNED stretches are gaps)
    phrase_ends: np.ndarray | None = None
    phrase_labels: np.ndarray | None = None
    jump_cue_groups: list = field(default_factory=list)        # each JumpCUE group's cue points, the user's
    jump_cue_auto_groups: list = field(default_factory=list)   # and the automatic result's


_tracks: list[dict] | None = None


def use(tracks) -> None:
    """The tracks of this run (the process's entry sets them)."""
    global _tracks
    _tracks = [dict(t) for t in tracks]


def tracks() -> list[dict]:
    """(uid, title, artist, audio_path, annotation[, error]) of every library track, in uid order (``annotation``
    None: none could be read, ``error`` why)."""
    if _tracks is None:
        raise RuntimeError("the library's tracks were not given to this run "
                           "(core.analysis_library.jobs.training_tracks)")
    return [dict(t) for t in _tracks]
