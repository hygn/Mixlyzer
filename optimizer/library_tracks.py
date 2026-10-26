"""The library's tracks for this Reoptimize run, given by the library's owner when the run starts
(core.analysis_library.jobs.training_tracks, handed to the optimizer process by core.workers.optimizer): the
optimizers list their tracks from these, each with what they read of its stored features (``features``), and read
its audio by the path given; they never open the library's files (library.db, the {uid}.npz files)."""
from __future__ import annotations

_tracks: list[dict] | None = None


def use(tracks) -> None:
    """The tracks of this run (the process's entry sets them)."""
    global _tracks
    _tracks = [dict(t) for t in tracks]


def tracks() -> list[dict]:
    """(uid, title, artist, audio_path, features[, error]) of every library track, in uid order."""
    if _tracks is None:
        raise RuntimeError("the library's tracks were not given to this run "
                           "(core.analysis_library.jobs.training_tracks)")
    return [dict(t) for t in _tracks]
