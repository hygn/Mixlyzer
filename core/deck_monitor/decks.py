"""What Deck Monitor knows of another program's decks, whatever protocol brought it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.deck_monitor.source_marks import SourceMarks

MAX_DECKS = 4        # the decks shown (a protocol's deck numbers 1 ..= MAX_DECKS)


@dataclass(frozen=True)
class TrackInfo:
    """A deck's track as the sender names it: any part may be missing ("", 0)."""
    path: str = ""
    title: str = ""
    artist: str = ""
    duration_sec: float = 0.0

    def named(self) -> bool:
        return bool(self.path or self.title)


@dataclass(frozen=True)
class DeckState:
    """A deck's newest state. Its position comes as seconds, or as samples counted at ``rate`` (0: the file's own
    rate, known only once its track is found in the library)."""
    deck: int
    loaded: bool
    playing: bool                          # its play / pause as sent (not whether it is the master deck)
    seconds: Optional[float] = None
    samples: Optional[int] = None
    rate: int = 0
    track: TrackInfo = field(default_factory=TrackInfo)
    received_at: float = 0.0               # time.monotonic() of its last message
    # its positions heard lately, (time.monotonic(), seconds or samples as sent), the oldest first: its motion
    heard: tuple = ()
    bpm: float = 0.0                       # its BPM as it plays (0: not sent)
    track_bpm: float = 0.0                 # its track's own BPM as the sender knows it (0: not sent)
    master: Optional[bool] = None          # the master deck (None: not sent)
    marks: Optional["SourceMarks"] = None  # the sender's own grid, CUE points and key of its track (None: not sent)

    def sent_position(self) -> float:
        """The position as sent: seconds, or samples."""
        return float(self.seconds) if self.seconds is not None else float(self.samples or 0)

    @property
    def pitch(self) -> Optional[float]:
        """Its tempo as the sender gives it: its BPM over its track's (1: as recorded); None when not sent."""
        return self.bpm / self.track_bpm if self.bpm > 0.0 and self.track_bpm > 0.0 else None
