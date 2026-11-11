"""The decks as the monitor's process sees them: the OSC Basic receiver it runs, each deck's track found among the
library's tracks the app gave it, its position in seconds now and its speed (core.deck_monitor.motion)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from core.deck_monitor.decks import MAX_DECKS, DeckState
from core.deck_monitor.match import TrackMatcher
from core.deck_monitor.motion import Motion, motion
from core.deck_monitor.osc_basic import STALE_SEC, OscBasicReceiver
from core.library.records import TrackRow
from core.library.service import LibrarySnapshot


@dataclass(frozen=True)
class MonitorDeck:
    """A deck as the monitor shows it."""
    deck: int
    state: Optional[DeckState]           # None: never heard from
    stale: bool                          # not heard from for STALE_SEC
    row: Optional[TrackRow]              # its track in the library, when found
    position_sec: Optional[float]        # now; None: not known (samples at the file's own rate, the track not found)
    age_sec: float                       # since its last message
    speed: float = 0.0                   # seconds of the track a second, read from its motion (0: standing still)

    @property
    def shown(self) -> bool:
        """A track loaded and heard from now."""
        return self.state is not None and not self.stale and self.state.loaded

    @property
    def playing(self) -> bool:
        """Its play / pause as the sender reads it (its position and speed are its motion's)."""
        return self.shown and self.state.playing


def seconds_per_unit(state: DeckState, row: Optional[TrackRow]) -> Optional[float]:
    """What one of the deck's sent positions is in seconds: 1 (seconds), a sample at its rate, or (rate 0) the found
    track's share of a sample; None when not known."""
    if state.seconds is not None:
        return 1.0
    if state.rate > 0:
        return 1.0 / state.rate
    if row is not None and row.total_samples and row.duration:
        return float(row.duration) / float(row.total_samples)
    return None


def deck_motion(state: DeckState, row: Optional[TrackRow], now: float) -> Optional[Motion]:
    """The deck's position and speed at ``now`` (core.deck_monitor.motion, on its positions heard), not past its
    track's end when known (before its start: below 0, as sent); None when its position is not known in seconds."""
    scale = seconds_per_unit(state, row)
    if scale is None:
        return None
    points = state.heard or ((state.received_at, state.sent_position()),)
    m = motion([(at, value * scale) for at, value in points], now)
    duration = float(row.duration) if row is not None and row.duration else 0.0
    if m is None or duration <= 0.0:
        return m
    return Motion(min(m.position, duration), m.speed if m.position < duration else 0.0)



class DeckSource:
    """The receiver listening on ``host:port`` (raises OSError when it cannot) and the decks matched against the
    library's tracks (``set_tracks``: the app's, again when the library changed)."""

    def __init__(self, host: str, port: int, tracks=()) -> None:
        self._receiver = OscBasicReceiver(str(host or "").strip() or "127.0.0.1", int(port))
        self._matcher = TrackMatcher(LibrarySnapshot.of_tracks(tracks))

    def set_tracks(self, tracks) -> None:
        self._matcher = TrackMatcher(LibrarySnapshot.of_tracks(tracks))

    @property
    def address(self) -> tuple[str, int]:
        """(host, port) it listens on."""
        host, port = self._receiver.address
        return str(host), int(port)

    def listening_on(self) -> str:
        host, port = self.address
        return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    def decks(self) -> list[MonitorDeck]:
        """Decks 1 ..= MAX_DECKS, each as last heard (none heard: an empty deck)."""
        states = self._receiver.latest()[0]
        now = time.monotonic()
        out = []
        for deck in range(1, MAX_DECKS + 1):
            state = states.get(deck)
            if state is None:
                out.append(MonitorDeck(deck, None, True, None, None, 0.0))
                continue
            row = self._matcher.find(state.track) if state.loaded and state.track.named() else None
            age = now - state.received_at
            m = deck_motion(state, row, now)
            out.append(MonitorDeck(deck, state, age >= STALE_SEC, row, m.position if m else None, age,
                                   m.speed if m else 0.0))
        return out

    def status(self) -> dict[str, Any]:
        """What the settings page shows: where it listens, each deck's last message's age, the last sender."""
        decks, sender, refused = self._receiver.latest()
        now = time.monotonic()
        return {"listening": self.listening_on(), "sender": sender, "refused_packets": int(refused),
                "decks": {n: now - s.received_at for n, s in sorted(decks.items())}}

    def close(self) -> None:
        self._receiver.close()
