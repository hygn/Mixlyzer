"""OSC Basic: each deck's track and position, as OSC messages over UDP, one deck a message, as often as the sender
reads them:

    /mixlyzer/deck/time     i deck  s path  T|F loaded  T|F playing  d|f seconds
    /mixlyzer/deck/samples  i deck  s path  T|F loaded  T|F playing  h|i samples  i rate
    /mixlyzer/deck/track    i deck  s title  s artist  d|f duration
    /mixlyzer/deck/tempo    i deck  d|f bpm  d|f track_bpm
    /mixlyzer/deck/master   i deck  T|F master

``path`` may be "" (the track is then found by the track message's title, artist and length: send it when the track
changes and now and then, as UDP may lose it); ``rate`` is the rate the samples are counted at, 0: the file's own
(known once the track is found in the library); ``duration`` 0: not known; ``playing`` the deck's play / pause. The
tempo and master messages are optional: ``bpm`` the deck's BPM as it plays, ``track_bpm`` the track's own (their
ratio is the deck's tempo: the waveform's zoom follows it; 0: not known), ``master`` whether it is the master deck. A
flag may also come as an i (0 / 1). Decks 1 - 4 are kept; a deck silent for ``STALE_SEC`` is shown as gone. The
receiver runs on its own thread and keeps each deck's newest state; the monitor takes it at its own rate (no queued
signal per message)."""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Optional

from core.deck_monitor import osc
from core.deck_monitor.decks import MAX_DECKS, DeckState, TrackInfo
from core.deck_monitor.motion import KEPT_SEC

TIME_ADDRESS = "/mixlyzer/deck/time"
SAMPLES_ADDRESS = "/mixlyzer/deck/samples"
TRACK_ADDRESS = "/mixlyzer/deck/track"
TEMPO_ADDRESS = "/mixlyzer/deck/tempo"
MASTER_ADDRESS = "/mixlyzer/deck/master"
ADDRESSES = (TIME_ADDRESS, SAMPLES_ADDRESS, TRACK_ADDRESS, TEMPO_ADDRESS, MASTER_ADDRESS)
STALE_SEC = 1.0
MAX_TEXT_CHARS = 4096


def _flag(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if 0.0 <= value < 1e7 else None      # also refuses NaN


def _text(value) -> Optional[str]:
    return value if isinstance(value, str) and len(value) <= MAX_TEXT_CHARS else None


def _deck(value) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= MAX_DECKS else None


def state_message(address: str, args: list, now: float) -> Optional[DeckState]:
    """The deck state a time or samples message carries (its track named by the path alone), None for another
    message or a malformed one."""
    if address not in (TIME_ADDRESS, SAMPLES_ADDRESS) or len(args) < 5:
        return None
    deck, path, loaded, playing = _deck(args[0]), _text(args[1]), _flag(args[2]), _flag(args[3])
    if deck is None or path is None or loaded is None or playing is None:
        return None
    track = TrackInfo(path=path.strip())
    if address == TIME_ADDRESS:
        seconds = _number(args[4]) if len(args) == 5 else None
        if seconds is None:
            return None
        return DeckState(deck, loaded, playing, seconds=seconds, track=track, received_at=now)
    if len(args) != 6:
        return None
    samples, rate = args[4], args[5]
    if isinstance(samples, bool) or not isinstance(samples, int) or isinstance(rate, bool) or not isinstance(rate, int):
        return None
    if samples < 0 or not 0 <= rate <= 1_000_000:
        return None
    return DeckState(deck, loaded, playing, samples=samples, rate=rate, track=track, received_at=now)


def track_message(address: str, args: list) -> Optional[tuple[int, TrackInfo]]:
    """(deck, its title, artist and length) a track message carries, None otherwise."""
    if address != TRACK_ADDRESS or len(args) != 4:
        return None
    deck, title, artist, duration = _deck(args[0]), _text(args[1]), _text(args[2]), _number(args[3])
    if deck is None or title is None or artist is None or duration is None:
        return None
    return deck, TrackInfo(title=title.strip(), artist=artist.strip(), duration_sec=duration)


def tempo_message(address: str, args: list) -> Optional[tuple[int, float, float]]:
    """(deck, its BPM as it plays, the track's own BPM) a tempo message carries, None otherwise."""
    if address != TEMPO_ADDRESS or len(args) != 3:
        return None
    deck, bpm, track_bpm = _deck(args[0]), _number(args[1]), _number(args[2])
    if deck is None or bpm is None or track_bpm is None:
        return None
    return deck, bpm, track_bpm


def master_message(address: str, args: list) -> Optional[tuple[int, bool]]:
    """(deck, whether it is the master deck) a master message carries, None otherwise."""
    if address != MASTER_ADDRESS or len(args) != 2:
        return None
    deck, master = _deck(args[0]), _flag(args[1])
    return None if deck is None or master is None else (deck, master)


def is_loopback(host: str) -> bool:
    try:
        return all(ipaddress.ip_address(info[4][0]).is_loopback
                   for info in socket.getaddrinfo(host, None, type=socket.SOCK_DGRAM))
    except (OSError, ValueError):
        return False


class OscBasicReceiver:
    """A UDP socket on its own thread keeping each deck's newest state and its track's title, artist and length."""

    def __init__(self, host: str, port: int) -> None:
        family = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0][0]
        self._sock = socket.socket(family, socket.SOCK_DGRAM)
        try:
            self._sock.bind((host, port))
        except OSError:
            self._sock.close()
            raise
        self._sock.settimeout(0.2)
        self.address = self._sock.getsockname()[:2]
        self._lock = threading.Lock()
        self._states: dict[int, DeckState] = {}
        self._names: dict[int, TrackInfo] = {}     # each deck's title, artist and length (its track message)
        self._heard: dict[int, deque] = {}         # each deck's positions heard in the last KEPT_SEC (its motion)
        self._tempo: dict[int, tuple[float, float]] = {}   # each deck's (BPM as it plays, the track's own)
        self._master: dict[int, bool] = {}
        self._sender = ""
        self._refused = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="DeckMonitorOSC", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet, sender = self._sock.recvfrom(osc.MAX_PACKET + 1)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            self.take(packet, f"{sender[0]}:{sender[1]}", time.monotonic())

    def take(self, packet: bytes, sender: str, now: float) -> None:
        """One packet's messages kept (a packet holding none of them, or malformed, counted as refused)."""
        try:
            messages = osc.parse(packet)
        except osc.OscError:
            messages = []
        states = [s for s in (state_message(a, args, now) for a, args in messages) if s]
        names = [n for n in (track_message(a, args) for a, args in messages) if n]
        tempos = [t for t in (tempo_message(a, args) for a, args in messages) if t]
        masters = [m for m in (master_message(a, args) for a, args in messages) if m]
        with self._lock:
            if not (states or names or tempos or masters):
                self._refused += 1
                return
            for state in states:
                old = self._states.get(state.deck)
                if old is not None and (old.track.path != state.track.path or not state.loaded):
                    self._names.pop(state.deck, None)    # another track (or none): the last one's names and
                    self._heard.pop(state.deck, None)    # tempo go
                    self._tempo.pop(state.deck, None)
                heard = self._heard.setdefault(state.deck, deque())
                heard.append((now, state.sent_position()))
                while heard and now - heard[0][0] > KEPT_SEC:
                    heard.popleft()
                self._states[state.deck] = state
            for deck, name in names:
                self._names[deck] = name
            for deck, bpm, track_bpm in tempos:
                self._tempo[deck] = (bpm, track_bpm)
            for deck, master in masters:
                self._master[deck] = master
            self._sender = sender

    def latest(self) -> tuple[dict[int, DeckState], str, int]:
        """(each deck's newest state, its track named by its path and its track message, its tempo and master flag,
        the last sender, packets refused)."""
        with self._lock:
            states, names, sender, refused = dict(self._states), dict(self._names), self._sender, self._refused
            heard = {deck: tuple(points) for deck, points in self._heard.items()}
            tempos, masters = dict(self._tempo), dict(self._master)
        decks = {}
        for deck, state in states.items():
            name = names.get(deck, TrackInfo())
            bpm, track_bpm = tempos.get(deck, (0.0, 0.0))
            decks[deck] = replace(state, track=TrackInfo(state.track.path, name.title, name.artist, name.duration_sec),
                                  heard=heard.get(deck, ()), bpm=bpm, track_bpm=track_bpm, master=masters.get(deck))
        return decks, sender, refused

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)
