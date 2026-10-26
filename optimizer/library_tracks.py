"""The library's tracks for this Reoptimize run, given by the library's owner when the run starts
(core.library.jobs.training_tracks, handed to the optimizer process by core.workers.optimizer): the optimizers list
their tracks from these and read each one's analysis file and audio by the paths given, never library.db."""
from __future__ import annotations

_tracks: list[dict] | None = None


def use(tracks) -> None:
    """The tracks of this run (the process's entry sets them)."""
    global _tracks
    _tracks = [dict(t) for t in tracks]


def tracks() -> list[dict]:
    """(uid, title, artist, audio_path, analysis_path) of every library track, in uid order."""
    if _tracks is None:
        raise RuntimeError("the library's tracks were not given to this run (core.library.jobs.training_tracks)")
    return [dict(t) for t in _tracks]
