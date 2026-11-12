"""Pro DJ Link, experimental: the players' decks read natively, as Deep Symmetry's analysis of the protocol lays it out
(djl-analysis.deepsymmetry.org; its offsets checked against beat-link's source). Not AlphaTheta's: a player may answer
otherwise than read here.

Deck Monitor is a virtual CDJ of its own number (``DEVICE``, above the players' 1 - 6): its keep-alive every 1.5 s on
port 50000 (beat-link's bytes, its name "Mixlyzer"); the players then send their status to it (port 50002: the track
loaded, its rekordbox ID and where it is, play / pause, master, pitch, BPM, the beat number), their beats (50001) and,
a CDJ-3000, its precise position every 30 ms (50001). A track's title, artist, length and beat grid are asked of the
player holding it, over its dbserver (TCP: the port asked on 12523, then a setup posing as a player number not in
use, the metadata request and its menu, the beat grid request), once per track on a thread of its own; no file is
fetched from a player. The position is the CDJ-3000's own; an older player sends none (its status, every 200 ms,
tells the beat number; a packet comes as each beat plays), so it is estimated in the beat grid as beat-link's
TimeFinder does: each beat packet sets it to that beat's time, carried on at the pitch (backwards in reverse); a
status keeps the estimate while it lies within a beat of the beat number told, else sets that beat's time (a jump: a
hot cue, a loop, a search); paused, the estimate stays where it stopped until the beat number changes (moved by the
jog, the CUE button: that beat's start). Without the grid it is not known. The CUE points are not read (their format
is not read here yet), so a stop on a CUE point between beats is placed on its beat. Every packet and message comes from the network and is read strictly (lengths checked before
each field; a message's arguments, a string's and a blob's lengths bounded)."""

from __future__ import annotations

import bisect
import select
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.deck_monitor.decks import MAX_DECKS, DeckState, TrackInfo
from core.deck_monitor.motion import KEPT_SEC
from core.deck_monitor.receivers import interface_for
from core.deck_monitor.source_marks import MAX_BEATS, SourceMarks, grid_from_beats

MAGIC = bytes.fromhex("5173707431576d4a4f4c")
ANNOUNCE_PORT, BEAT_PORT, STATUS_PORT = 50000, 50001, 50002
KEEP_ALIVE, CDJ_STATUS, BEAT, PRECISE_POSITION = 0x06, 0x0A, 0x28, 0x0B
DEVICE = 7                                        # its own number: above the players' (1 - 4, 1 - 6 CDJ-3000s)
NAME = b"Mixlyzer"
KEEP_ALIVE_SEC = 1.5
DEVICE_GONE_SEC = 5.0
PITCH_UNITY = 0x100000
DB_QUERY_PORT = 12523
DB_TIMEOUT_SEC = 3.0
DB_RETRY_SEC = 10.0
DB_MAGIC = 0x872349AE
SETUP_TX = 0xFFFFFFFE
SETUP, METADATA_REQ, RENDER_MENU, BEAT_GRID_REQ, TEARDOWN = 0x0000, 0x2002, 0x3000, 0x2204, 0x0100
MENU_ITEM, MENU_FOOTER, SUCCESS = 0x4101, 0x4201, 0x4000
ITEM_TITLE, ITEM_ARTIST, ITEM_DURATION = 0x04, 0x07, 0x0B
MAX_STRING_CHARS = 1024
MAX_BLOB = 4 * 1024 * 1024
MAX_MENU_ITEMS = 64


def keep_alive(device: int, mac: bytes, address: str) -> bytes:
    """The keep-alive (type 06, 54 bytes) presenting a virtual CDJ: beat-link's bytes, its own name and number."""
    p = bytearray(MAGIC + bytes([KEEP_ALIVE, 0x00]) + NAME.ljust(20, b"\x00")
                  + bytes([0x01, 0x02, 0x00, 0x36, device, 0x01]) + bytes(10) + bytes([0x02, 0, 0, 0, 0x01, 0x64]))
    p[0x26:0x2C] = (mac or bytes(6))[:6].ljust(6, b"\x00")
    p[0x2C:0x30] = socket.inet_aton(address)
    return bytes(p)


def _be(packet: bytes, at: int, size: int) -> int:
    return int.from_bytes(packet[at:at + size], "big")


def _beat_time(beats: tuple, number: int) -> float:
    """Beat ``number``'s time in the grid (1: the first; 0: the track's start, where a paused player tells 0)."""
    return beats[min(number, len(beats)) - 1][0] if number > 0 else 0.0


def _beat_at_time(beats: tuple, seconds: float) -> int:
    """The number of the beat ``seconds`` is in (0: before the first)."""
    return bisect.bisect_right(beats, seconds, key=lambda b: b[0])


def _carried(estimate: tuple[float, float, bool], player: "_Player", now: float) -> float:
    """An estimate carried on to ``now`` at the player's pitch, backwards when it plays in reverse (never before 0)."""
    position, at, moving = estimate
    if not moving:
        return position
    moved = (now - at) * player.pitch
    return max(0.0, position - moved) if player.reverse else position + moved


def _estimated(player: "_Player", beats: tuple, beat: int, playing: bool, reverse: bool,
               now: float) -> tuple[float, float, bool]:
    """The position a status tells a player without its own, as beat-link's TimeFinder reads it: the estimate carried
    on to now kept while it lies within a beat of the beat number told (paused: while that number is the same), else
    that beat's time (playing backwards: the next beat's, where it came into the beat)."""
    told = _beat_time(beats, beat)
    if player.estimate is None:
        return told, now, playing
    if not playing and not player.estimate[2]:
        return player.estimate if beat == player.beat else (told, now, False)
    at = _carried(player.estimate, player, now)
    if abs(_beat_at_time(beats, at) - beat) < 2:
        return at, now, playing
    if reverse and beat > 0:
        return _beat_time(beats, beat + 1), now, playing
    return told, now, playing


@dataclass(frozen=True)
class TrackRef:
    """A track as a player names it: the player holding it, its slot and type, its rekordbox ID."""
    player: int
    slot: int
    kind: int
    rekordbox_id: int


@dataclass
class _Player:
    number: int
    address: str = ""
    seen: float = 0.0
    status_at: float = 0.0
    track: Optional[TrackRef] = None
    playing: bool = False
    master: Optional[bool] = None
    pitch: float = 1.0
    bpm: float = 0.0                 # the track's own (status: 100 x BPM)
    reverse: bool = False            # playing backwards
    beat: Optional[int] = None       # the beat number now (1: the first; 0: paused at the track's start)
    estimate: Optional[tuple[float, float, bool]] = None  # (position s, when, moving): a player without its own
    precise: Optional[tuple[float, float]] = None     # (position s, when): a CDJ-3000's own
    length: float = 0.0
    heard: deque = field(default_factory=deque)


@dataclass(frozen=True)
class TrackData:
    title: str = ""
    artist: str = ""
    duration: float = 0.0
    beats: tuple = ()               # (time s, place in its bar) of each beat
    grid: tuple = ()                # its tempo changes


class ProDjLinkReceiver:
    """A virtual CDJ on the network of ``interface`` (a local IPv4 address, "": the first): its sockets on a thread of
    its own (raises OSError when a port cannot be had, ValueError when there is no such interface)."""

    def __init__(self, interface: str = "", *, start: bool = True) -> None:
        self._lock = threading.Lock()
        self._players: dict[int, _Player] = {}
        self._tracks: dict[TrackRef, TrackData] = {}
        self._asking: dict[TrackRef, float] = {}       # a dbserver read running or failed (when: asked again later)
        self._sender, self._refused = "", 0
        self._last_keep_alive = 0.0
        self._sockets: list[socket.socket] = []
        self._stop = threading.Event()
        self.address = ("", 0)
        self.broadcast, self.mac = "255.255.255.255", b""
        if not start:
            return
        found = interface_for(interface)
        if found is None:
            raise ValueError(f"no network interface {interface or ''}".strip())
        self.broadcast, self.mac, self.address = found.broadcast, found.mac, (found.address, ANNOUNCE_PORT)
        try:
            for port in (ANNOUNCE_PORT, BEAT_PORT, STATUS_PORT):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sockets.append(s)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.bind(("", port))
        except OSError:
            self.close()
            raise
        self._thread = threading.Thread(target=self._run, name="DeckMonitorProDJLink", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select(self._sockets, [], [], 0.05)
            except (OSError, ValueError):
                if self._stop.is_set():
                    return
                continue
            for s in ready:
                try:
                    packet, sender = s.recvfrom(2048)
                except OSError:
                    continue
                self.take(packet, sender[0], time.monotonic())
            self.tick(time.monotonic())

    def tick(self, now: float) -> None:
        """The keep-alive every 1.5 s; a dbserver read started for each track loaded and not read yet."""
        if now - self._last_keep_alive >= KEEP_ALIVE_SEC and self._sockets:
            self._last_keep_alive = now
            try:
                self._sockets[0].sendto(keep_alive(DEVICE, self.mac, self.address[0]),
                                        (self.broadcast, ANNOUNCE_PORT))
            except OSError:
                pass
        with self._lock:
            for number in [n for n, p in self._players.items() if now - p.seen > DEVICE_GONE_SEC]:
                del self._players[number]
            wanted = []
            for player in self._players.values():
                ref = player.track
                if ref is None or ref in self._tracks or now - self._asking.get(ref, -1e9) < DB_RETRY_SEC:
                    continue
                holder = self._players.get(ref.player)
                if holder is None or not holder.address:
                    continue
                self._asking[ref] = now
                in_use = set(self._players)
                wanted.append((ref, holder.address, in_use))
        for ref, address, in_use in wanted:
            threading.Thread(target=self._read_track, args=(ref, address, in_use), daemon=True,
                             name="DeckMonitorDbServer").start()

    def _read_track(self, ref: TrackRef, address: str, in_use: set) -> None:
        try:
            data = read_track(address, ref, posing_as(ref.player, in_use))
        except (OSError, ValueError, struct.error):
            return                                                    # asked again after DB_RETRY_SEC
        with self._lock:
            self._tracks[ref] = data
            self._asking.pop(ref, None)

    # ---- what comes in ----

    def take(self, packet: bytes, address: str, now: float) -> None:
        """One packet read (another protocol's, or a malformed one, counted as refused)."""
        if len(packet) < 0x24 or packet[:10] != MAGIC:
            with self._lock:
                self._refused += 1
            return
        kind = packet[0x0A]
        with self._lock:
            if kind == KEEP_ALIVE and len(packet) >= 0x30:
                number = packet[0x24]
                if number != DEVICE and 1 <= number <= 6:
                    player = self._players.setdefault(number, _Player(number))
                    player.address, player.seen = socket.inet_ntoa(packet[0x2C:0x30]), now
                return
            number = packet[0x21]
            player = self._players.get(number)
            if not 1 <= number <= 6:
                return                                               # the mixer, another device
            if player is None:
                player = self._players[number] = _Player(number, address, now)
            player.seen = now
            if kind == CDJ_STATUS:
                if len(packet) < 0xCC:
                    self._refused += 1
                    return
                self._status(player, packet, now)
            elif kind == PRECISE_POSITION and len(packet) >= 0x3C:
                pitch = _be(packet, 0x2C, 4)
                pitch = pitch - 0x100000000 if pitch >= 0x80000000 else pitch
                player.pitch = 1.0 + pitch / 10000.0
                player.length = float(_be(packet, 0x24, 4))
                player.precise = (_be(packet, 0x28, 4) / 1000.0, now)
            elif kind == BEAT and len(packet) >= 0x60:
                self._beat(player, packet, now)
            else:
                return
            self._heard(player, now)
            name = packet[0x0B:0x1F].split(b"\x00", 1)[0].decode("ascii", errors="replace")
            self._sender = f"{name} {address}"

    def _status(self, player: _Player, packet: bytes, now: float) -> None:
        source, slot, kind, rekordbox_id = packet[0x28], packet[0x29], packet[0x2A], _be(packet, 0x2C, 4)
        track = TrackRef(source, slot, kind, rekordbox_id) if source and rekordbox_id else None
        if track != player.track:
            player.track, player.precise, player.beat, player.estimate = track, None, None, None
            player.heard.clear()
        flags = packet[0x89]
        playing = bool(flags & 0x40)
        reverse = playing and packet[0x9D] == 0x01            # P3: "paused or playing in Reverse mode"
        beat = _be(packet, 0xA0, 4)
        beat = None if beat == 0xFFFFFFFF else beat           # no rekordbox-analyzed track
        beats = self._beats(player)
        if player.precise is None:
            player.estimate = (None if beat is None or not beats
                               else _estimated(player, beats, beat, playing, reverse, now))
        player.playing, player.reverse, player.master = playing, reverse, bool(flags & 0x20)
        player.pitch = _be(packet, 0x8C, 4) / PITCH_UNITY
        bpm = _be(packet, 0x92, 2)
        player.bpm = bpm / 100.0 if bpm != 0xFFFF else 0.0
        player.beat, player.status_at = beat, now

    def _beat(self, player: _Player, packet: bytes, now: float) -> None:
        """A beat packet, sent as the beat plays: the next beat's time in the grid (beat-link's TimeFinder), unless
        the position estimated is less than a fifth of a beat into one (that beat's own packet, the status having
        told it first)."""
        player.pitch = _be(packet, 0x54, 4) / PITCH_UNITY
        beats = self._beats(player)
        if player.precise is not None or player.estimate is None or not beats:
            return
        at = _carried(player.estimate, player, now)
        number = _beat_at_time(beats, at)
        bpm = _be(packet, 0x5A, 2) / 100.0 or player.bpm
        fifth = 60.0 / bpm / 5.0 if bpm > 0 else 0.0
        if number < 1 or at - _beat_time(beats, number) >= fifth:
            number = min(number + 1, len(beats))
        player.estimate, player.beat = (_beat_time(beats, number), now, True), number

    def _beats(self, player: _Player) -> tuple:
        data = self._tracks.get(player.track) if player.track is not None else None
        return data.beats if data is not None else ()

    def _position(self, player: _Player, now: float) -> Optional[float]:
        """Its position now: the CDJ-3000's own, else the estimate from its status and beats in the grid."""
        if player.precise is not None:
            position, at = player.precise
            return position + ((now - at) * player.pitch if player.playing else 0.0)
        beats = self._beats(player)
        if not beats or player.beat is None:
            return None
        if player.estimate is None:                           # the grid read after the last status
            return _carried((_beat_time(beats, player.beat), player.status_at, player.playing), player, now)
        return _carried(player.estimate, player, now)

    def _heard(self, player: _Player, now: float) -> None:
        position = self._position(player, now)
        if position is None:
            return
        player.heard.append((now, position))
        while player.heard and now - player.heard[0][0] > KEPT_SEC:
            player.heard.popleft()

    # ---- what the window reads ----

    def latest(self, now: Optional[float] = None) -> tuple[dict[int, DeckState], str, int]:
        """(each player 1 - 4's deck at ``now``, time.monotonic()'s, by default; the last sender; packets
        refused)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            decks = {}
            for number, player in self._players.items():
                if number > MAX_DECKS or not player.status_at:
                    continue
                data = self._tracks.get(player.track) if player.track is not None else None
                position = self._position(player, now)
                duration = data.duration if data is not None and data.duration else player.length
                decks[number] = DeckState(
                    number, player.track is not None, player.playing, seconds=position,
                    track=TrackInfo("", data.title if data else "", data.artist if data else "", duration),
                    received_at=player.status_at, heard=tuple(player.heard) if position is not None else (),
                    bpm=player.bpm * player.pitch, track_bpm=player.bpm, master=player.master,
                    marks=SourceMarks(grid=data.grid) if data is not None and data.grid else None)
            return decks, self._sender, self._refused

    def close(self) -> None:
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
        for s in self._sockets:
            try:
                s.close()
            except OSError:
                pass
        self._sockets = []


def posing_as(target: int, in_use: set) -> int:
    """The player number the dbserver setup names (1 - 4, not the player asked): one no device uses, else the lowest
    other (a player linked to it may then be refused)."""
    free = [n for n in range(1, 5) if n != target and n not in in_use]
    return free[0] if free else next(n for n in range(1, 5) if n != target)


# ---- the dbserver (TCP) ----

def _number(value: int, size: int) -> bytes:
    return bytes([{1: 0x0F, 2: 0x10, 4: 0x11}[size]]) + value.to_bytes(size, "big")


def db_message(tx: int, kind: int, args: list[int]) -> bytes:
    """A dbserver message of 4-byte number arguments."""
    tags = bytes([0x06] * len(args)).ljust(12, b"\x00")
    return (_number(DB_MAGIC, 4) + _number(tx, 4) + _number(kind, 2) + _number(len(args), 1)
            + b"\x14" + (12).to_bytes(4, "big") + tags + b"".join(_number(a, 4) for a in args))


class _Stream:
    """A socket read field by field (each length bounded)."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock, self.buf = sock, b""

    def take(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ValueError("dbserver closed the connection")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def field(self):
        tag = self.take(1)[0]
        if tag in (0x0F, 0x10, 0x11):
            return int.from_bytes(self.take({0x0F: 1, 0x10: 2, 0x11: 4}[tag]), "big")
        if tag == 0x14:
            size = int.from_bytes(self.take(4), "big")
            if size > MAX_BLOB:
                raise ValueError("dbserver blob too long")
            return self.take(size)
        if tag == 0x26:
            chars = int.from_bytes(self.take(4), "big")
            if chars > MAX_STRING_CHARS:
                raise ValueError("dbserver string too long")
            return self.take(2 * chars).decode("utf-16-be", errors="replace").split("\x00", 1)[0]
        raise ValueError(f"dbserver field type {tag:#x}")

    def message(self) -> tuple[int, list]:
        """(its type, its arguments)."""
        if self.field() != DB_MAGIC:
            raise ValueError("not a dbserver message")
        self.field()                                                   # its transaction
        kind, count, tags = self.field(), self.field(), self.field()
        if not isinstance(kind, int) or not isinstance(count, int) or not isinstance(tags, bytes) or count > 12:
            raise ValueError("dbserver message malformed")
        return kind, [self.field() for _ in range(count)]


def read_track(address: str, ref: TrackRef, posing: int, timeout: float = DB_TIMEOUT_SEC,
               query_port: int = DB_QUERY_PORT) -> TrackData:
    """The track's title, artist, length and beat grid, asked of the player at ``address`` (raises OSError,
    ValueError when it cannot be read)."""
    with socket.create_connection((address, query_port), timeout=timeout) as s:
        s.sendall(b"\x00\x00\x00\x0fRemoteDBServer\x00")
        reply = b""
        while len(reply) < 2:
            chunk = s.recv(2 - len(reply))
            if not chunk:
                raise ValueError("no dbserver port")
            reply += chunk
    port = int.from_bytes(reply, "big")
    with socket.create_connection((address, port), timeout=timeout) as s:
        stream = _Stream(s)
        s.sendall(_number(1, 4))
        if stream.field() != 1:
            raise ValueError("dbserver greeting")
        s.sendall(db_message(SETUP_TX, SETUP, [posing]))
        if stream.message()[0] != SUCCESS:
            raise ValueError("dbserver setup refused")
        dmst = (posing << 24) | (1 << 16) | (ref.slot << 8) | ref.kind
        s.sendall(db_message(1, METADATA_REQ, [dmst, ref.rekordbox_id]))
        kind, args = stream.message()
        if kind != SUCCESS or len(args) < 2 or not isinstance(args[1], int) or args[1] == 0xFFFFFFFF:
            raise ValueError("no such track")
        count = min(int(args[1]), MAX_MENU_ITEMS)
        s.sendall(db_message(2, RENDER_MENU, [dmst, 0, count, 0, count, 0]))
        title = artist = ""
        duration = 0.0
        for _ in range(count + 2):
            kind, args = stream.message()
            if kind == MENU_FOOTER:
                break
            if kind != MENU_ITEM or len(args) < 7 or not isinstance(args[6], int):
                continue
            item = args[6] & 0xFFFF
            label = args[3] if isinstance(args[3], str) else ""
            if item == ITEM_TITLE:
                title = label
            elif item == ITEM_ARTIST:
                artist = label
            elif item == ITEM_DURATION and isinstance(args[1], int):
                duration = float(args[1])
        s.sendall(db_message(3, BEAT_GRID_REQ, [dmst, ref.rekordbox_id]))
        kind, args = stream.message()
        beats = parse_beat_grid(args[3]) if len(args) > 3 and isinstance(args[3], bytes) else ()
        try:
            s.sendall(db_message(SETUP_TX, TEARDOWN, []))
        except OSError:
            pass
    times = [t for t, _ in beats]
    return TrackData(title, artist, duration, beats, grid_from_beats(times, [p for _, p in beats]))


def parse_beat_grid(blob: bytes) -> tuple:
    """(time s, place in its bar) of each beat of a dbserver beat grid (20 bytes, then 16 a beat: its place, 100 x
    BPM, its time in ms, little-endian), in time order (a beat not after the one before left out)."""
    beats = []
    for at in range(20, min(len(blob), 20 + 16 * MAX_BEATS) - 15, 16):
        place, _, ms = struct.unpack_from("<HHI", blob, at)
        t = ms / 1000.0
        if beats and t <= beats[-1][0]:
            continue
        beats.append((t, place if 1 <= place <= 4 else 1))
    return tuple(beats)
