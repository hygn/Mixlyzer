import os
import socket
import struct
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore  # noqa: E402

from core.playback import external_sync as xs  # noqa: E402
from core.playback import osc  # noqa: E402
from core.settings.config import externalsyncconfig  # noqa: E402
from core.workflows.event_bus import EventBus  # noqa: E402
from core.track.model import DataModel  # noqa: E402


class OscTests(unittest.TestCase):
    def test_round_trip_and_bundles(self):
        packet = osc.message(xs.TIME_ADDRESS, 1, "C:/a b.mp3", True, False, 12.5)
        self.assertEqual(osc.parse(packet), [(xs.TIME_ADDRESS, [1, "C:/a b.mp3", True, False, 12.5])])
        two = osc.message("/a", 1) + b""
        bundle = b"#bundle\x00" + bytes(8)
        for element in (two, osc.message("/b", "x")):
            bundle += struct.pack(">i", len(element)) + element
        self.assertEqual(osc.parse(bundle), [("/a", [1]), ("/b", ["x"])])
        self.assertEqual(osc.parse(osc.message("/x", 1 << 40)), [("/x", [1 << 40])])

    def test_malformed_refused(self):
        good = osc.message("/a", "abc", 1)
        nested = osc.message("/a")
        for _ in range(osc.MAX_DEPTH + 1):
            nested = b"#bundle\x00" + bytes(8) + struct.pack(">i", len(nested)) + nested
        cases = {
            "unterminated": b"/abc",
            "padding": b"/a\x00\x01,i\x00\x00" + bytes(4),
            "unknown type": osc.message("/a")[:-4] + b",b\x00\x00" + bytes(4),
            "trailing": good + bytes(4),
            "short argument": good[:-4],
            "not an address": osc.message("a"),
            "oversized": osc.message("/a", "x" * osc.MAX_PACKET),
            "nested": nested,
            "element past": b"#bundle\x00" + bytes(8) + struct.pack(">i", 400) + good,
            "negative element": b"#bundle\x00" + bytes(8) + struct.pack(">i", -4) + good,
            "not utf-8": b"/\xff\x00\x00",
        }
        for name, packet in cases.items():
            with self.subTest(name), self.assertRaises(osc.OscError):
                osc.parse(packet)

    def test_deck_reports_checked(self):
        now = 1.0
        ok = xs.deck_report(xs.SAMPLES_ADDRESS, [2, "/m/a.flac", 1, 0, 44100, 44100], now)
        self.assertEqual((ok.deck, ok.loaded, ok.active, ok.samples, ok.rate), (2, True, False, 44100, 44100))
        bad = [
            (xs.TIME_ADDRESS, [True, "/a", True, True, 1.0]),            # a bool is no deck
            (xs.TIME_ADDRESS, [0, "/a", True, True, 1.0]),               # decks from 1
            (xs.TIME_ADDRESS, [1, "/a", 2, True, 1.0]),                  # a flag 0 / 1
            (xs.TIME_ADDRESS, [1, "/a", True, True, float("nan")]),
            (xs.TIME_ADDRESS, [1, "/a", True, True, -1.0]),
            (xs.TIME_ADDRESS, [1, "/a", True, True, 1.0, 3]),            # one too many
            (xs.SAMPLES_ADDRESS, [1, "/a", True, True, -5, 44100]),
            (xs.SAMPLES_ADDRESS, [1, "/a", True, True, 5, 2.0]),
            (xs.TIME_ADDRESS, [1, "x" * (xs.MAX_PATH_CHARS + 1), True, True, 1.0]),
            ("/other", [1, "/a", True, True, 1.0]),
        ]
        for address, args in bad:
            with self.subTest(args=args[:3]):
                self.assertIsNone(xs.deck_report(address, args, now))

    def test_network_paths(self):
        for path in ("\\\\host\\share\\a.mp3", "//host/share/a.mp3", "/\\host\\share\\a.mp3",
                     "\\/host/share/a.mp3", "\\\\?\\UNC\\host\\share\\a.mp3", "\\\\.\\pipe\\x"):
            with self.subTest(path):
                self.assertTrue(xs._network_path(path))
        for path in ("C:\\Music\\a.mp3", "C:/Music/a.mp3", "/home/u/a.mp3"):
            with self.subTest(path):
                self.assertFalse(xs._network_path(path))
        self.assertEqual(xs.ExternalSyncController._local_file("/\\host\\share\\a.mp3"), "")


class ReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.track = str(Path(self.dir.name, "a.flac").resolve())
        Path(self.track).write_bytes(b"x")
        self.library: set[str] = set()
        self.loads: list[tuple[str, float]] = []
        self.seeks: list[float] = []
        self.ctl = xs.ExternalSyncController(
            EventBus(), DataModel(), externalsyncconfig(enabled=True, osc_host="127.0.0.1", osc_port=0),
            load_track_callback=lambda p, t: self.loads.append((p, t)), seek_callback=self.seeks.append,
            track_exists_callback=lambda p: p in self.library, track_duration_callback=lambda p: 200.0)
        self.port = int(self.ctl.listening_on().rsplit(":", 1)[1])
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tearDown(self):
        self.out.close()
        self.ctl.close()
        self.dir.cleanup()

    def send(self, *packets: bytes) -> None:
        """The packets sent and received (each a new deck state or a refusal), then one poll."""
        sent_at = time.monotonic()
        refused_before = self.ctl._receiver.latest()[2]
        for packet in packets:
            self.out.sendto(packet, ("127.0.0.1", self.port))
        deadline = sent_at + 2.0
        while time.monotonic() < deadline:
            decks, _sender, refused = self.ctl._receiver.latest()
            new = sum(1 for r in decks.values() if r.received_at >= sent_at)
            if new + refused - refused_before >= len(packets):
                break
            time.sleep(0.01)
        self.ctl._poll_external()

    def test_follows_the_lowest_loaded_and_active_deck(self):
        self.library.add(self.track)
        self.send(osc.message(xs.TIME_ADDRESS, 1, self.track, True, False, 5.0),
                  osc.message(xs.SAMPLES_ADDRESS, 2, self.track, True, True, 441000, 44100))
        self.assertEqual(self.ctl.snapshot()["osc"]["following"], 2)
        self.assertEqual(self.loads, [(self.track, 10.0)])

    def test_garbage_counted_and_ignored(self):
        self.send(b"not osc at all!!", osc.message("/other", 1))
        _decks, _sender, refused = self.ctl._receiver.latest()
        self.assertEqual(refused, 2)
        self.assertEqual(self.loads, [])

    def test_stale_deck_not_followed(self):
        self.library.add(self.track)
        stale, xs.STALE_SEC = xs.STALE_SEC, 0.05
        try:
            self.send(osc.message(xs.TIME_ADDRESS, 1, self.track, True, True, 1.0))
            self.assertEqual(len(self.loads), 1)
            time.sleep(0.1)
            self.ctl._poll_external()
            self.assertEqual(self.ctl.snapshot()["osc"]["following"], 0)
        finally:
            xs.STALE_SEC = stale

    def test_network_sender_only_drives_library_tracks(self):
        self.ctl._library_only = True   # as when listening on a network address
        self.send(osc.message(xs.TIME_ADDRESS, 1, self.track, True, True, 1.0))
        self.assertEqual(self.loads, [])
        self.library.add(self.track)
        self.ctl._validated_external_paths.clear()
        self.ctl._clear_pending_external()
        self.send(osc.message(xs.TIME_ADDRESS, 1, self.track, True, True, 2.0))
        self.assertEqual(self.loads, [(self.track, 2.0)])

    def test_unsafe_path_never_loaded(self):
        self.send(osc.message(xs.TIME_ADDRESS, 1, "/\\host\\share\\a.flac", True, True, 1.0))
        self.assertEqual(self.loads, [])


if __name__ == "__main__":
    unittest.main()
