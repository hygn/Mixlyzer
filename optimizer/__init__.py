"""Parameter optimization helpers."""

from dataclasses import dataclass
from pathlib import Path

from analyzer_core.utils import prime_physical_core_count

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
    reason: str

    @property
    def display_name(self) -> str:
        title = self.title.strip()
        if not title:
            title = Path(self.path).stem or "Untitled"
        title = _compact_display_field(title)
        artist = _compact_display_field(self.artist) or "-"
        return f"Title: {title}\nArtist: {artist}"


def optimizer_track_name(track: dict[str, object]) -> str:
    title = str(track.get("title", "") or "").strip()
    if not title:
        audio_path = track.get("audio_path")
        title = getattr(audio_path, "stem", "") or "Untitled"
    title = _compact_display_field(title)
    artist = _compact_display_field(track.get("artist", ""))
    uid = str(track.get("uid", "") or "").strip()
    return f"Title: {title}\nArtist: {artist or '-'}\nUUID: {uid or '-'}"


def skipped_track(track: dict[str, object], reason: str) -> SkippedOptimizationTrack:
    return SkippedOptimizationTrack(
        uid=str(track.get("uid", "") or ""),
        title=str(track.get("title", "") or ""),
        artist=str(track.get("artist", "") or ""),
        path=str(track.get("audio_path", "") or ""),
        reason=str(reason or "Unknown validation error"),
    )
