"""The library's tracks for this run, read through the library's owner when the run starts
(core.analysis_library.jobs.training_tracks / training_annotations, by the CLI: training.__main__): the optimizers
list their tracks from these, each with the user's annotation of it (``annotation``, an `Annotation`), and read its
audio by the path given; they never open the library's files (library.db, the stored analyses)."""
from __future__ import annotations

from core.analysis_library.stored import Annotation

__all__ = ["Annotation", "tracks", "use"]

_tracks: list[dict] | None = None


def use(tracks) -> None:
    """The tracks of this run (the CLI sets them)."""
    global _tracks
    _tracks = [dict(t) for t in tracks]


def tracks() -> list[dict]:
    """(uid, title, artist, audio_path, annotation[, error]) of every library track, in uid order (``annotation``
    None: none could be read, ``error`` why)."""
    if _tracks is None:
        raise RuntimeError("the library's tracks were not given to this run "
                           "(core.analysis_library.jobs.training_tracks)")
    return [dict(t) for t in _tracks]
