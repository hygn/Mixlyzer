"""The training: Reoptimize's optimizers, each retraining a learned model (or refitting settings) of the analysis on
the library's annotations. The app hands a run its tracks (library_tracks) and the files to read and write, and
words what it reports (text.py); the training opens no library file and imports no app code."""

from dataclasses import dataclass
from pathlib import Path

from analyzer.runtime import prime_physical_core_count
from analyzer.training.text import Text, msg

# sklearn (KMeans, HistGradientBoosting) would otherwise launch powershell.exe to count cores.
prime_physical_core_count()


_DISPLAY_FIELD_MAX_LENGTH = 24
_MIDDLE_ELLIPSIS = "..."


def _compact_display_field(value: object) -> str:
    text = str(value or "").strip()
    if len(text) <= _DISPLAY_FIELD_MAX_LENGTH:
        return text
    content_length = _DISPLAY_FIELD_MAX_LENGTH - len(_MIDDLE_ELLIPSIS)
    left_length = (content_length + 1) // 2
    right_length = content_length - left_length
    return f"{text[:left_length]}{_MIDDLE_ELLIPSIS}{text[-right_length:]}"


@dataclass(frozen=True)
class SkippedOptimizationTrack:
    uid: str
    title: str
    artist: str
    path: str
    reason: Text | str

    @property
    def display_name(self) -> Text:
        return _track_words(self.title, self.artist, self.path)


def _track_words(title: object, artist: object, path: object) -> Text:
    title = _compact_display_field(str(title or "").strip() or getattr(Path(str(path or "")), "stem", ""))
    return msg("Title: {title}\nArtist: {artist}", title=title or msg("Untitled"),
               artist=_compact_display_field(artist) or "-")


def optimizer_track_name(track: dict[str, object]) -> tuple:
    """The track as progress names it: its title and artist, its uid (two lines)."""
    uid = str(track.get("uid", "") or "").strip()
    return (_track_words(track.get("title", ""), track.get("artist", ""), track.get("audio_path") or ""),
            f"UUID: {uid or '-'}")


def skipped_track(track: dict[str, object], reason: Text | str) -> SkippedOptimizationTrack:
    return SkippedOptimizationTrack(
        uid=str(track.get("uid", "") or ""),
        title=str(track.get("title", "") or ""),
        artist=str(track.get("artist", "") or ""),
        path=str(track.get("audio_path", "") or ""),
        reason=reason or msg("Unknown validation error"),
    )
