"""Deck Monitor's Pro DJ Link (core.deck_monitor.prodjlink, experimental): the keep-alive it sends, a player's packets
as Deep Symmetry's analysis lays them out read into the decks, a track read from a fake dbserver over TCP, the
position from a CDJ-3000's packets or from the beat grid. Fakes only: nothing here needs a player."""
import socket
import struct
import threading
import unittest

from core.deck_monitor import prodjlink as pdl
from core.deck_monitor.decks import DeckState
from core.deck_monitor.source_marks import SourceCue, SourceMarks
from core.deck_monitor.source import seconds_per_unit
from core.library.records import TrackRow


def _packet(kind: int, size: int, device: int) -> bytearray:
    p = bytearray(size)
    p[:10], p[0x0A] = pdl.MAGIC, kind
    p[0x0B:0x0B + 7] = b"CDJ-3000"[:7]
    p[0x21] = device
    return p


def _keep_alive(device: int, address: str) -> bytes:
    p = _packet(pdl.KEEP_ALIVE, 0x36, 0)
    p[0x24], p[0x2C:0x30] = device, socket.inet_aton(address)
    return bytes(p)


def _status(device: int, source: int, rekordbox_id: int, playing: bool, master: bool, pitch: float, bpm: float,
            beat: int, reverse: bool = False, at_cue: bool = False) -> bytes:
    p = _packet(pdl.CDJ_STATUS, 0xD4, device)
    p[0x28], p[0x29], p[0x2A] = source, 3, 1
    p[0x2C:0x30] = rekordbox_id.to_bytes(4, "big")
    p[0x7B] = 0x03 if playing else (pdl.AT_CUE if at_cue else 0x05)        # P1: playing, at the cue point, paused
    p[0x89] = (0x40 if playing else 0) | (0x20 if master else 0)
    p[0x8C:0x90] = int(pitch * pdl.PITCH_UNITY).to_bytes(4, "big")
    p[0x92:0x94] = int(round(bpm * 100)).to_bytes(2, "big")
    p[0x9D] = 0x0D if playing and not reverse else 0x01        # P3: forward (CDJ mode); paused or reverse
    p[0xA0:0xA4] = beat.to_bytes(4, "big")
    return bytes(p)


def _beat_packet(device: int, bpm: float, pitch: float = 1.0) -> bytes:
    p = _packet(pdl.BEAT, 0x60, device)
    p[0x54:0x58] = int(pitch * pdl.PITCH_UNITY).to_bytes(4, "big")
    p[0x5A:0x5C] = int(round(bpm * 100)).to_bytes(2, "big")
    return bytes(p)


def _precise(device: int, length_s: int, position_ms: int, pitch_hundredths: int) -> bytes:
    p = _packet(pdl.PRECISE_POSITION, 0x3C, device)
    p[0x24:0x28] = length_s.to_bytes(4, "big")
    p[0x28:0x2C] = position_ms.to_bytes(4, "big")
    p[0x2C:0x30] = (pitch_hundredths & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(p)


def _grid_blob(beats: list) -> bytes:
    return bytes(20) + b"".join(struct.pack("<HHI", place, 12800, ms) + bytes(8) for place, ms in beats)


def _ext_cue(hot: int, kind: int, ms: int, color_id: int = 0, comment: str = "", rgb=None) -> bytes:
    """An extended cue list entry as beat-link reads one: its length first; the comment and a hot cue's colour after
    it only when given (an entry may end before them)."""
    raw = (comment + "\x00").encode("utf-16-le") if comment else b""
    size = 0x52 + len(raw) if rgb else (0x4A + len(raw) if comment else 0x38)
    e = bytearray(size)
    e[0:4], e[4], e[6] = size.to_bytes(4, "little"), hot, kind
    e[12:16], e[0x22] = ms.to_bytes(4, "little"), color_id
    if comment or rgb:
        e[0x48:0x4A] = len(raw).to_bytes(2, "little")
        e[0x4A:0x4A + len(raw)] = raw
    if rgb:
        e[0x4E + len(raw)] = 0x2A                        # its colour code
        e[0x4F + len(raw):0x52 + len(raw)] = bytes(rgb)
    return bytes(e)


def _legacy_cue(loop: int, cue: int, hot: int, half_frames: int) -> bytes:
    e = bytearray(pdl.LEGACY_CUE_SIZE)
    e[0], e[1], e[2] = loop, cue, hot
    e[12:16] = half_frames.to_bytes(4, "little")
    return bytes(e)


EXT_CUES = [_ext_cue(1, 1, 30500, comment="Drop", rgb=(255, 0, 0)), _ext_cue(0, 1, 1500, color_id=2),
            _ext_cue(0, 2, 60000, comment="Loop"), _ext_cue(2, 1, 45000), _ext_cue(0, 0, 0)]


# ---- a fake dbserver ----

def _field(value) -> bytes:
    if isinstance(value, bytes):
        return b"\x14" + len(value).to_bytes(4, "big") + value
    if isinstance(value, str):
        raw = (value + "\x00").encode("utf-16-be")
        return b"\x26" + (len(raw) // 2).to_bytes(4, "big") + raw
    return b"\x11" + value.to_bytes(4, "big")


def _reply(kind: int, args: list) -> bytes:
    tags = bytes([0x03 if isinstance(a, bytes) else 0x02 if isinstance(a, str) else 0x06 for a in args])
    return (b"\x11" + pdl.DB_MAGIC.to_bytes(4, "big") + b"\x11" + (7).to_bytes(4, "big") + b"\x10"
            + kind.to_bytes(2, "big") + b"\x0f" + bytes([len(args)]) + b"\x14" + (12).to_bytes(4, "big")
            + tags.ljust(12, b"\x00") + b"".join(_field(a) for a in args))


class FakeDbServer:
    """The query port answering with the dbserver's, and the dbserver answering one session as a player would."""

    def __init__(self, title: str, artist: str, seconds: int, grid: bytes, cues=None, cue_count: int = 0,
                 legacy: bytes = b"") -> None:
        """``cues``: the extended cue list answered (None: refused); ``legacy``: a nexus player's (none: refused)."""
        self.query = socket.create_server(("127.0.0.1", 0))
        self.db = socket.create_server(("127.0.0.1", 0))
        self.asked = []
        refused = _reply(0x4003, [])
        self.answers = {pdl.SETUP: [_reply(pdl.SUCCESS, [0, 2])],
                        pdl.METADATA_REQ: [_reply(pdl.SUCCESS, [0x2002, 3])],
                        pdl.RENDER_MENU: [_reply(0x4001, []),
                                          _reply(pdl.MENU_ITEM, [9, 42, 0, title, 0, "", pdl.ITEM_TITLE, 0]),
                                          _reply(pdl.MENU_ITEM, [0, 9, 0, artist, 0, "", pdl.ITEM_ARTIST, 0]),
                                          _reply(pdl.MENU_ITEM, [0, seconds, 0, "", 0, "", 0x1000B, 0]),
                                          _reply(pdl.MENU_FOOTER, [])],
                        pdl.BEAT_GRID_REQ: [_reply(0x4602, [0, 0, len(grid), grid])],
                        pdl.CUE_LIST_EXT_REQ: [_reply(pdl.CUE_LIST_EXT, [0, 0, len(cues), cues, cue_count])
                                               if cues is not None else refused],
                        pdl.CUE_LIST_REQ: [_reply(pdl.CUE_LIST, [0, 0, len(legacy), legacy, 0x24, 0, 0, 0])
                                           if legacy else refused]}
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        conn, _ = self.query.accept()
        with conn:
            conn.recv(19)
            conn.sendall(self.db.getsockname()[1].to_bytes(2, "big"))
        conn, _ = self.db.accept()
        with conn:
            stream = pdl._Stream(conn)
            stream.field()
            conn.sendall(b"\x11\x00\x00\x00\x01")
            while True:
                try:
                    kind, args = stream.message()
                except (ValueError, OSError):
                    return
                self.asked.append((kind, args))
                for reply in self.answers.get(kind, []):
                    conn.sendall(reply)

    def close(self) -> None:
        self.query.close()
        self.db.close()


class ProDjLinkTests(unittest.TestCase):
    def setUp(self):
        self.r = pdl.ProDjLinkReceiver(start=False)

    def test_its_keep_alive(self):
        p = pdl.keep_alive(pdl.DEVICE, bytes.fromhex("0a1b2c3d4e5f"), "192.168.1.20")
        self.assertEqual((len(p), p[:10], p[0x0A], p[0x23], p[0x24]), (0x36, pdl.MAGIC, 0x06, 0x36, pdl.DEVICE))
        self.assertEqual(p[0x0C:0x14], b"Mixlyzer")
        self.assertEqual((p[0x26:0x2C].hex(), socket.inet_ntoa(p[0x2C:0x30])), ("0a1b2c3d4e5f", "192.168.1.20"))

    def test_a_cdj_3000_read_into_its_deck(self):
        self.r.take(_keep_alive(2, "10.0.0.2"), "10.0.0.2", 1.0)
        self.r.take(_status(2, 2, 42, True, True, 1.0 + 326 / 10000, 128.0, 33), "10.0.0.2", 1.0)
        self.r.take(_precise(2, 240, 61500, 326), "10.0.0.2", 1.0)
        d = self.r.latest(1.5)[0][2]
        self.assertEqual((d.loaded, d.playing, d.master, d.track.duration_sec), (True, True, True, 240.0))
        self.assertAlmostEqual(d.seconds, 61.5 + 0.5 * 1.0326)       # carried on from 1.0 at the pitch
        self.assertAlmostEqual(d.track_bpm, 128.0)
        self.assertAlmostEqual(d.pitch, 1.0326, places=3)
        self.assertEqual(d.track.title, "")                         # not read from the dbserver yet
        self.assertEqual(self.r._players[2].address, "10.0.0.2")
        self.r.take(b"not pro dj link at all.............", "10.0.0.9", 1.1)
        self.assertEqual(self.r.latest()[2], 1)
        self.r.take(_packet(pdl.CDJ_STATUS, 0x40, 2), "10.0.0.2", 1.2)   # too short
        self.assertEqual(self.r.latest()[2], 2)

    def test_a_track_read_from_the_dbserver_and_the_position_from_its_grid(self):
        grid = _grid_blob([((k % 4) + 1, 300 + int(k * 468.75)) for k in range(16)])
        server = FakeDbServer("Song", "Band", 240, grid)
        try:
            data = pdl.read_track("127.0.0.1", pdl.TrackRef(1, 3, 1, 42), posing=3, timeout=2.0,
                                  query_port=server.query.getsockname()[1])
        finally:
            server.close()
        self.assertEqual((data.title, data.artist, data.duration), ("Song", "Band", 240.0))
        self.assertEqual(len(data.beats), 16)
        self.assertAlmostEqual(data.marks.grid[0].bpm, 128.0, places=1)
        self.assertEqual(data.marks.grid[0].beat, 1)
        self.assertIsNone(data.marks.cues)             # neither cue list answered: not known (not none), the rest kept
        kinds = [k for k, _ in server.asked]
        self.assertEqual(kinds[:6], [pdl.SETUP, pdl.METADATA_REQ, pdl.RENDER_MENU, pdl.BEAT_GRID_REQ,
                                     pdl.CUE_LIST_EXT_REQ, pdl.CUE_LIST_REQ])
        self.assertEqual(server.asked[0][1], [3])                                 # posing as player 3
        self.assertEqual(server.asked[1][1], [(3 << 24) | (1 << 16) | (3 << 8) | 1, 42])   # the main menu
        self.assertEqual(server.asked[3][1], [(3 << 24) | (8 << 16) | (3 << 8) | 1, 42])   # the data menu

        ref = pdl.TrackRef(1, 3, 1, 42)                              # a nexus player: no precise position
        self.r.take(_status(1, 1, 42, False, False, 1.0, 128.0, 9), "10.0.0.1", 1.0)
        self.assertIsNone(self.r.latest(1.0)[0][1].seconds)          # no grid yet: not known
        self.r._tracks[ref] = data
        d = self.r.latest(1.0)[0][1]
        self.assertAlmostEqual(d.seconds, data.beats[8][0])          # beat 9's time, standing still
        self.assertEqual(d.track.title, "Song")
        self.assertIs(d.marks, data.marks)                           # made once: the window sees no change

    def test_its_cues_from_the_extended_list_or_a_nexus_players(self):
        grid = _grid_blob([((k % 4) + 1, 300 + int(k * 468.75)) for k in range(16)])
        ref = pdl.TrackRef(1, 3, 1, 42)
        server = FakeDbServer("Song", "Band", 240, grid, cues=b"".join(EXT_CUES), cue_count=len(EXT_CUES))
        try:
            data = pdl.read_track("127.0.0.1", ref, posing=3, timeout=2.0, query_port=server.query.getsockname()[1])
        finally:
            server.close()
        self.assertEqual(data.marks.cues, (SourceCue(1.5, "", (255, 0, 0)), SourceCue(30.5, "Drop", (255, 0, 0), 0),
                                           SourceCue(45.0, "", None, 1), SourceCue(60.0, "Loop")))
        self.assertEqual(server.asked[4][1], [(3 << 24) | (8 << 16) | (3 << 8) | 1, 42, 0])
        self.assertNotIn(pdl.CUE_LIST_REQ, [k for k, _ in server.asked])    # the extended one answered
        legacy = _legacy_cue(0, 1, 0, 150) + _legacy_cue(0, 1, 1, 4575) + _legacy_cue(0, 0, 0, 0)
        server = FakeDbServer("Song", "Band", 240, grid, legacy=legacy)
        try:
            data = pdl.read_track("127.0.0.1", ref, posing=3, timeout=2.0, query_port=server.query.getsockname()[1])
        finally:
            server.close()
        self.assertEqual(data.marks.cues, (SourceCue(1.0), SourceCue(30.5, slot=0)))     # 1/150 s, no names
        server = FakeDbServer("Song", "Band", 240, grid, cues=b"", cue_count=0)
        try:
            data = pdl.read_track("127.0.0.1", ref, posing=3, timeout=2.0, query_port=server.query.getsockname()[1])
        finally:
            server.close()
        self.assertEqual(data.marks.cues, ())                        # a list with none: known, none (not None)

    def test_a_cue_list_read_strictly(self):
        blob = b"".join(EXT_CUES)
        self.assertEqual(len(pdl.parse_cue_list_ext(blob, 99)), 4)                # a count past its entries: ends
        cut = b"".join(EXT_CUES[:4])[:-10]                                          # hot cue B cut off: left out
        self.assertEqual([c.seconds for c in pdl.parse_cue_list_ext(cut, 4)], [1.5, 30.5, 60.0])
        bad = bytearray(_ext_cue(1, 1, 1000, comment="X"))
        bad[0x48:0x4A] = (500).to_bytes(2, "little")                              # a comment past its entry
        self.assertEqual(pdl.parse_cue_list_ext(bytes(bad), 1), (SourceCue(1.0, "", None, 0),))
        self.assertEqual(pdl.parse_cue_list_ext(b"\x02\x00\x00\x00" + bytes(40), 1), ())   # a length below 16
        self.assertEqual(pdl.parse_cue_list(bytes(35)), ())

    def test_posing_as_a_free_player(self):
        self.assertEqual(pdl.posing_as(1, {1, 2}), 3)
        self.assertEqual(pdl.posing_as(1, {1, 2, 3, 4}), 2)          # none free: another (may be refused)

    def test_on_the_network(self):
        import time
        try:
            r = pdl.ProDjLinkReceiver("127.0.0.1")
        except (OSError, ValueError) as exc:
            self.skipTest(f"the Pro DJ Link ports are not free here (rekordbox?): {exc}")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(_status(3, 3, 7, True, False, 1.0, 120.0, 1), ("127.0.0.1", pdl.STATUS_PORT))
                until = time.monotonic() + 2.0
                while time.monotonic() < until:          # every 30 ms as a CDJ-3000 sends it (two sockets: the
                    s.sendto(_precise(3, 200, 12000, 0), ("127.0.0.1", pdl.BEAT_PORT))    # order not kept)
                    time.sleep(0.03)
                    deck = r.latest()[0].get(3)
                    if deck is not None and deck.seconds is not None:
                        break
            self.assertGreaterEqual(r.latest()[0][3].seconds, 12.0)
        finally:
            r.close()

    def test_a_position_not_known_stays_unknown(self):
        row = TrackRow(uid="a", path="/a", duration=240.0, total_samples=240 * 44100)
        self.assertIsNone(seconds_per_unit(DeckState(1, True, True), row))


SPB = 0.46875                                                    # a beat at 128 BPM


def _t(number: int) -> float:
    """Beat ``number``'s time in the test grid (the first at 0.3 s)."""
    return 0.3 + (number - 1) * SPB


class OlderPlayerPositionTests(unittest.TestCase):
    """A player before the CDJ-3000 sends no position: estimated from its status and beat packets in the grid, as
    beat-link's TimeFinder does."""

    def setUp(self):
        self.r = pdl.ProDjLinkReceiver(start=False)
        self.r._tracks[pdl.TrackRef(1, 3, 1, 42)] = pdl.TrackData(
            beats=tuple((_t(k), (k - 1) % 4 + 1) for k in range(1, 65)),
            marks=SourceMarks(cues=(SourceCue(_t(9) + 0.1), SourceCue(_t(12) + 0.2, slot=0))))   # a memory, a hot

    def status(self, at: float, beat: int, playing: bool = True, reverse: bool = False, at_cue: bool = False) -> None:
        self.r.take(_status(1, 1, 42, playing, False, 1.0, 128.0, beat, reverse, at_cue), "10.0.0.1", at)

    def beat(self, at: float) -> None:
        self.r.take(_beat_packet(1, 128.0), "10.0.0.1", at)

    def seconds(self, at: float) -> float:
        return self.r.latest(at)[0][1].seconds

    def test_each_beat_packet_sets_its_beat_time(self):
        self.status(10.0, 5)
        self.assertAlmostEqual(self.seconds(10.2), _t(5) + 0.2)
        self.beat(10.0 + SPB + 0.03)                         # heard late: still beat 6, not 7
        self.assertAlmostEqual(self.seconds(10.6), _t(6) + 0.6 - SPB - 0.03)
        self.beat(10.0 + 2 * SPB - 0.2)                      # the pitch read low: the next beat all the same
        self.assertAlmostEqual(self.seconds(11.0), _t(7) + 1.0 - 2 * SPB + 0.2)

    def test_a_status_telling_the_beat_first_counts_it_once(self):
        self.status(10.0, 5)
        self.status(10.47, 6)                                # just after beat 6: the estimate kept
        self.assertAlmostEqual(self.seconds(10.47), _t(5) + 0.47)
        self.beat(10.475)                                    # beat 6's own packet
        self.assertAlmostEqual(self.seconds(10.475), _t(6))

    def test_paused_between_beats_it_stays_where_it_stopped(self):
        self.status(10.0, 5)
        self.status(10.2, 5, playing=False)
        self.assertAlmostEqual(self.seconds(11.0), _t(5) + 0.2)
        self.status(10.4, 5, playing=False)
        self.assertAlmostEqual(self.seconds(11.0), _t(5) + 0.2)
        self.status(10.6, 9, playing=False)                  # moved while paused (the jog, CUE): that beat
        self.assertAlmostEqual(self.seconds(11.0), _t(9))

    def test_paused_at_its_cue_point_it_is_the_memory_cue_in_that_beat(self):
        self.status(10.0, 5)
        self.status(10.2, 9, playing=False, at_cue=True)     # CUE pressed: back to the memory cue in beat 9
        self.assertAlmostEqual(self.seconds(11.0), _t(9) + 0.1)
        self.status(10.4, 12, playing=False, at_cue=True)    # only a hot cue in beat 12: that beat's start
        self.assertAlmostEqual(self.seconds(11.0), _t(12))
        self.status(10.6, 9, playing=False)                  # paused elsewhere in beat 9: its start
        self.assertAlmostEqual(self.seconds(11.0), _t(9))

    def test_a_jump_goes_to_the_beat_told(self):
        self.status(10.0, 5)
        self.status(10.2, 40)                                # a hot cue, a loop's start
        self.assertAlmostEqual(self.seconds(10.3), _t(40) + 0.1)

    def test_in_reverse_it_runs_backwards(self):
        self.status(10.0, 9, reverse=True)
        self.status(10.2, 9, reverse=True)                   # 0.2 s back, within a beat of 9: kept
        self.assertAlmostEqual(self.seconds(10.3), _t(9) - 0.3)
        self.status(10.4, 3, reverse=True)                   # far back: where it came into beat 3
        self.assertAlmostEqual(self.seconds(10.4), _t(4))

    def test_paused_at_the_tracks_start(self):
        self.status(10.0, 0, playing=False)
        self.assertEqual(self.seconds(10.5), 0.0)

    def test_no_grid_no_position(self):
        self.r._tracks.clear()
        self.status(10.0, 5)
        self.beat(10.3)
        self.assertIsNone(self.seconds(10.4))


if __name__ == "__main__":
    unittest.main()
