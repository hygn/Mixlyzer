"""Deck Monitor (core.deck_monitor): OSC Basic's messages read strictly, each deck's track found in the library by its
path or by its title, artist and length, and the decks as the window shows them. Nothing here touches an app deck."""
import os
import socket
import struct
import time
import unittest
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.deck_monitor import osc  # noqa: E402
from core.deck_monitor import osc_basic as ob  # noqa: E402
from core.deck_monitor.controller import DeckMonitorController  # noqa: E402
from core.deck_monitor.motion import AHEAD_SEC, motion  # noqa: E402
from core.deck_monitor.receivers import Where  # noqa: E402
from core.deck_monitor.source import DeckSource, deck_motion, seconds_per_unit  # noqa: E402
from core.deck_monitor.decks import DeckState, TrackInfo  # noqa: E402
from core.deck_monitor.match import TrackMatcher, network_path  # noqa: E402
from core.deck_monitor.readout import LABEL, SECTION, TrackPhrases, bars_beats, bpm_at, key_at  # noqa: E402
from core.deck_monitor.source_marks import (SourceCue, SourceMarks, TempoChange, grid_from_beats,  # noqa: E402
                                            parse_key, source_analysis)
from core.library.records import TrackRow  # noqa: E402
from core.settings.config import default_cfg  # noqa: E402


class OscTests(unittest.TestCase):
    def test_round_trip_and_bundles(self):
        packet = osc.message(ob.TIME_ADDRESS, 1, "C:/a b.mp3", True, False, 12.5)
        self.assertEqual(osc.parse(packet), [(ob.TIME_ADDRESS, [1, "C:/a b.mp3", True, False, 12.5])])
        bundle = b"#bundle\x00" + bytes(8)
        for element in (osc.message("/a", 1), osc.message("/b", "x")):
            bundle += struct.pack(">i", len(element)) + element
        self.assertEqual(osc.parse(bundle), [("/a", [1]), ("/b", ["x"])])

    def test_malformed_refused(self):
        good = osc.message("/a", "abc", 1)
        for name, packet in {"unterminated": b"/abc", "trailing": good + bytes(4), "short argument": good[:-4],
                             "oversized": osc.message("/a", "x" * osc.MAX_PACKET)}.items():
            with self.subTest(name), self.assertRaises(osc.OscError):
                osc.parse(packet)


class MessageTests(unittest.TestCase):
    def test_states_checked(self):
        ok = ob.state_message(ob.SAMPLES_ADDRESS, [2, "/m/a.flac", 1, 0, 44100, 44100], 1.0)
        self.assertEqual((ok.deck, ok.loaded, ok.playing, ok.samples, ok.rate, ok.track.path),
                         (2, True, False, 44100, 44100, "/m/a.flac"))
        self.assertEqual(ob.state_message(ob.TIME_ADDRESS, [1, "", True, True, 3.5], 1.0).track, TrackInfo())
        # before the track's start: below 0
        self.assertEqual(ob.state_message(ob.TIME_ADDRESS, [1, "", True, False, -1.5], 1.0).seconds, -1.5)
        self.assertEqual(ob.state_message(ob.SAMPLES_ADDRESS, [1, "", True, False, -4410, 44100], 1.0).samples, -4410)
        for args in ([True, "/a", True, True, 1.0],          # a bool is no deck
                     [0, "/a", True, True, 1.0],             # decks from 1
                     [5, "/a", True, True, 1.0],             # to 4
                     [1, "/a", 2, True, 1.0],                # a flag 0 / 1
                     [1, "/a", True, True, -1e8],
                     [1, "/a", True, True, float("nan")],
                     [1, "/a", True, True]):
            with self.subTest(args):
                self.assertIsNone(ob.state_message(ob.TIME_ADDRESS, args, 1.0))

    def test_track_messages(self):
        self.assertEqual(ob.track_message(ob.TRACK_ADDRESS, [3, " Song ", "Band", 241.5]),
                         (3, TrackInfo(title="Song", artist="Band", duration_sec=241.5)))
        self.assertIsNone(ob.track_message(ob.TRACK_ADDRESS, [3, "Song", "Band"]))
        self.assertIsNone(ob.track_message(ob.TRACK_ADDRESS, [3, "Song", 1, 241.5]))

    def test_names_kept_per_deck_and_dropped_with_another_track(self):
        r = _Receiver()
        r.take(osc.message(ob.TRACK_ADDRESS, 1, "Song", "Band", 200.0)
               + b"", "x", 1.0)
        r.take(osc.message(ob.TIME_ADDRESS, 1, "", True, True, 5.0), "x", 1.1)
        self.assertEqual(r.latest()[0][1].track, TrackInfo("", "Song", "Band", 200.0))
        r.take(osc.message(ob.TIME_ADDRESS, 1, "/m/other.flac", True, True, 0.0), "x", 1.2)   # another track
        self.assertEqual(r.latest()[0][1].track, TrackInfo(path="/m/other.flac"))
        r.take(b"garbage", "x", 1.3)
        self.assertEqual(r.latest()[2], 1)

    def test_tempo_and_master_kept_the_tempo_dropped_with_another_track(self):
        self.assertEqual(ob.tempo_message(ob.TEMPO_ADDRESS, [2, 135.68, 128.0]), (2, 135.68, 128.0))
        self.assertIsNone(ob.tempo_message(ob.TEMPO_ADDRESS, [2, 135.68]))
        self.assertEqual(ob.master_message(ob.MASTER_ADDRESS, [1, True]), (1, True))
        self.assertIsNone(ob.master_message(ob.MASTER_ADDRESS, [1, 2]))
        r = _Receiver()
        r.take(osc.message(ob.TIME_ADDRESS, 1, "/m/a.flac", True, True, 5.0), "x", 1.0)
        r.take(osc.message(ob.TEMPO_ADDRESS, 1, 132.0, 120.0) + b"", "x", 1.1)
        r.take(osc.message(ob.MASTER_ADDRESS, 1, True), "x", 1.1)
        state = r.latest()[0][1]
        self.assertAlmostEqual(state.pitch, 1.1)
        self.assertTrue(state.master)
        r.take(osc.message(ob.TIME_ADDRESS, 1, "/m/b.flac", True, True, 0.0), "x", 1.2)    # another track
        state = r.latest()[0][1]
        self.assertIsNone(state.pitch)
        self.assertTrue(state.master)                                  # the deck's, not the track's

    def test_osc_full_marks_read_strictly_kept_and_dropped_with_another_track(self):
        self.assertEqual(ob.grid_message(ob.GRID_ADDRESS, [1, 0.25, 128.0, 1, 60.25, 130.0, 3]),
                         (1, (TempoChange(0.25, 128.0, 1), TempoChange(60.25, 130.0, 3))))
        for args in ([1, 0.25, 128.0], [1, 0.25, 128.0, 5], [1, 0.25, 0.0, 1], [1, 5.0, 128.0, 1, 4.0, 128.0, 1]):
            with self.subTest(args):
                self.assertIsNone(ob.grid_message(ob.GRID_ADDRESS, args))     # whole triples, beat 1 - 4, a
                                                                                # tempo, starts in order
        self.assertEqual(ob.cues_message(ob.CUES_ADDRESS, [2, 30.0, " Drop ", 0xFF0000, 1.5, "", -1]),
                         (2, (SourceCue(1.5, "", None), SourceCue(30.0, "Drop", (255, 0, 0)))))
        self.assertIsNone(ob.cues_message(ob.CUES_ADDRESS, [2, 30.0, "Drop", 0x1000000]))
        self.assertEqual(ob.key_message(ob.KEY_ADDRESS, [1, "8A"]), (1, 21))
        self.assertEqual(ob.key_message(ob.KEY_ADDRESS, [1, ""]), (1, None))
        self.assertIsNone(ob.key_message(ob.KEY_ADDRESS, [1, "H moll"]))          # not read: refused
        r = _Receiver()
        r.take(osc.message(ob.TRACK_ADDRESS, 1, "Song", "Band", 200.0), "x", 1.0)
        r.take(osc.message(ob.TIME_ADDRESS, 1, "", True, True, 5.0), "x", 1.0)
        r.take(osc.message(ob.GRID_ADDRESS, 1, 0.25, 128.0, 1), "x", 1.1)
        r.take(osc.message(ob.KEY_ADDRESS, 1, "Am"), "x", 1.1)
        marks = r.latest()[0][1].marks
        self.assertEqual((marks.grid, marks.key), ((TempoChange(0.25, 128.0, 1),), 21))
        r.take(osc.message(ob.TRACK_ADDRESS, 1, "Song", "Band", 200.0), "x", 1.2)    # again: kept
        self.assertIs(r.latest()[0][1].marks, marks)
        r.take(osc.message(ob.TRACK_ADDRESS, 1, "Other", "Band", 180.0), "x", 1.3)   # another track: dropped
        self.assertIsNone(r.latest()[0][1].marks)


class SourceMarksTests(unittest.TestCase):
    """core.deck_monitor.source_marks: a sender's own grid, CUE points and key as the app's analysis."""

    def test_keys_by_any_name(self):
        for text, key in (("8A", 21), ("8a", 21), ("Am", 21), ("A minor", 21), ("Amin", 21), ("C", 0), ("Db", 1),
                          ("C#m", 13), ("F# major", 6), ("Bbm", 22), ("", None), ("H", None), ("m", None)):
            with self.subTest(text):
                self.assertEqual(parse_key(text), key)

    def test_every_beat_read_into_tempo_changes(self):
        # 120 BPM, rounded to the ms as a source sends it, then 125 BPM from beat 9; the first beat a 3
        t = [round(k * 0.5, 3) for k in range(9)] + [round(4.0 + k * 0.48, 3) for k in range(1, 9)]
        places = [(2 + k) % 4 + 1 for k in range(len(t))]
        grid = grid_from_beats(t, places)
        self.assertEqual(len(grid), 2)
        self.assertAlmostEqual(grid[0].bpm, 120.0, places=6)
        self.assertAlmostEqual(grid[0].start, 0.0, places=9)
        self.assertEqual(grid[0].beat, 3)
        self.assertAlmostEqual(grid[1].start, 4.0, places=6)
        self.assertAlmostEqual(grid[1].bpm, 125.0, places=3)
        self.assertEqual(grid[1].beat, places[8])
        self.assertEqual(grid_from_beats([1.0], [1]), ())
        self.assertEqual(grid_from_beats([1.0, 0.5], [1, 2]), ())                  # out of order: none

    def test_the_marks_as_the_apps_analysis(self):
        marks = SourceMarks(grid=(TempoChange(0.5, 120.0, 3),), cues=(SourceCue(10.0, "Drop", (255, 40, 40)),
                                                                      SourceCue(99.0, "past the end")), key=21)
        a = source_analysis(marks, 20.0)
        self.assertEqual(a.tempo_segments.tolist(), [[0.5, 20.0, 120.0, 1.5, 4.0, 120.0]])   # its downbeat a 1
        self.assertAlmostEqual(float(a.beats[0]), 0.5)
        self.assertEqual(a.key_segments.tolist(), [[9.0, 1.0, 0.0, 20.0]])
        self.assertEqual([(c.time_sec, c.label, c.color_idx) for c in a.cue_points], [(10.0, "Drop", 0)])
        self.assertIsNone(source_analysis(marks, 0.0))                                # its length not known


class _Receiver(ob.OscBasicReceiver):
    """The receiver's bookkeeping without its socket."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._states, self._names, self._heard, self._sender, self._refused = {}, {}, {}, "", 0
        self._tempo, self._master, self._marks = {}, {}, {}


class ReadoutTests(unittest.TestCase):
    """core.deck_monitor.readout: 120 BPM in 4/4 (a bar 2 s), INTRO 8 bars, two VERSEs of 8, CHORUS 8."""

    def setUp(self):
        from analyzer.music.marks import Phrase
        self.beats = np.arange(0.0, 64.0, 0.5)
        self.rows = np.array([[0.0, 64.0, 120.0, 0.0, 4, 120.0]])
        self.phrases = [Phrase(0.0, 16.0, "INTRO"), Phrase(16.0, 32.0, "VERSE"), Phrase(32.0, 48.0, "VERSE"),
                        Phrase(48.0, 64.0, "CHORUS")]

    def test_the_phrase_now_and_next_by_section_their_bars_and_the_countdown(self):
        now = TrackPhrases(self.phrases, self.beats, self.rows).at(30.0, SECTION)
        self.assertEqual((now.now.name, now.now.bars), ("VERSE1", 8))
        self.assertEqual((now.next.name, now.next.bars), ("VERSE2", 8))
        self.assertEqual((now.left_beats, now.meter), (4, 4))                # 30.5, 31, 31.5 and 32 to come
        self.assertEqual(bars_beats(now.left_beats, now.meter), "1.0")
        self.assertEqual(bars_beats(13, 4), "3.1")
        last = TrackPhrases(self.phrases, self.beats, self.rows).at(50.0)
        self.assertEqual((last.now.label, last.next, last.left_beats), ("CHORUS", None, None))
        self.assertEqual(TrackPhrases((), None, None).at(1.0).now, None)
        before = TrackPhrases(self.phrases, self.beats, self.rows).at(-2.0)   # before the track's start
        self.assertEqual((before.now, before.next.name, before.left_beats), (None, "INTRO", None))   # no beats there
        from analyzer.music.marks import Phrase
        gap = TrackPhrases([Phrase(0.0, 16.0, "UNASSIGNED"), *self.phrases[1:]], self.beats, self.rows).at(2.0)
        self.assertEqual((gap.now, gap.next.name), (None, "VERSE1"))          # where no phrase is: none

    def test_by_label_a_run_of_one_label_is_one(self):
        now = TrackPhrases(self.phrases, self.beats, self.rows).at(30.0, LABEL)
        self.assertEqual((now.now.name, now.now.bars, now.now.start, now.now.end), ("VERSE", 16, 16.0, 48.0))
        self.assertEqual((now.next.name, now.next.bars), ("CHORUS", 8))
        self.assertEqual(bars_beats(now.left_beats, now.meter), "9.0")      # 18 s, 36 beats to the chorus

    def test_tempo_and_key_where_the_deck_is(self):
        ramp = np.array([[0.0, 10.0, 100.0, 0.0, 4, 120.0], [10.0, 20.0, 128.0, 10.0, 4, 128.0]])
        self.assertAlmostEqual(bpm_at(ramp, 5.0), 110.0)
        self.assertAlmostEqual(bpm_at(ramp, 15.0), 128.0)
        self.assertAlmostEqual(bpm_at(ramp, 99.0), 128.0)
        self.assertIsNone(bpm_at(np.empty((0, 6)), 1.0))
        self.assertEqual(key_at([(8, 0.0, 10.0), (13, 10.0, 20.0)], 12.0), 13)
        self.assertIsNone(key_at([(8, 0.0, 10.0)], 12.0))


@dataclass
class _Snapshot:
    tracks: tuple

    def get(self, path):
        return next((t for t in self.tracks if t.path == path), None)

    def find_file(self, path):
        return None


def _row(uid, path, title, artist, duration, total=None):
    return TrackRow(uid=uid, path=path, title=title, artist=artist, duration=duration, total_samples=total)


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.full = _row("a", "/m/song.flac", "Song", "Band", 240.0, 240 * 44100)
        self.tv = _row("b", "/m/song_tv.flac", "Song", "Band", 89.0)
        self.other = _row("c", "/m/other.flac", "Ｓｏｎｇ", "Other Band", 200.0)
        self.matcher = TrackMatcher(_Snapshot((self.full, self.tv, self.other)))

    def test_by_path_first(self):
        self.assertIs(self.matcher.find(TrackInfo(path="/m/song_tv.flac", title="Song", artist="Band")), self.tv)

    def test_by_name_and_length(self):
        find = self.matcher.find
        self.assertIs(find(TrackInfo(title="song", artist="BAND", duration_sec=241.0)), self.full)
        self.assertIs(find(TrackInfo(title="Song", artist="Band", duration_sec=90.0)), self.tv)
        self.assertIsNone(find(TrackInfo(title="Song", artist="Band", duration_sec=150.0)))   # no such version
        self.assertIs(find(TrackInfo(title="Song", artist="Other  band")), self.other)         # NFKC, spaces
        self.assertIsNone(find(TrackInfo(title="Unknown", artist="Band")))

    def test_network_paths_never_looked_up(self):
        for path in ("\\\\host\\share\\a.flac", "//host/share/a.flac", "\\/host/a.flac"):
            with self.subTest(path):
                self.assertTrue(network_path(path))
                self.assertIsNone(TrackMatcher(_Snapshot((_row("n", path, "", "", 1.0),))).find(TrackInfo(path)))

    def test_position_from_samples(self):
        state = DeckState(1, True, True, samples=44100 * 60, rate=0, received_at=5.0)
        self.assertAlmostEqual(deck_motion(state, self.full, 5.0).position, 60.0)   # the file's own rate: its share
        self.assertIsNone(deck_motion(state, None, 5.0))                            # not found: not known
        self.assertAlmostEqual(seconds_per_unit(DeckState(1, True, True, samples=1, rate=48000), None), 1 / 48000)
        self.assertEqual(deck_motion(DeckState(1, True, True, seconds=999.0), self.full, 0.0).position, 240.0)
        self.assertEqual(deck_motion(DeckState(1, True, False, seconds=-3.0), self.full, 0.0).position, -3.0)


def _heard(start: float, until: float, position, rate: float = 120.0):
    """Positions heard at ``rate`` a second from ``start`` to ``until``: ``position(t)``, rekordbox's 259-sample
    steps (it moves its position every 6 ms)."""
    n = int(round((until - start) * rate))
    return [(start + i / rate, int(position(start + i / rate) * 44100 / 259) * 259 / 44100) for i in range(n + 1)]


class MotionTests(unittest.TestCase):
    def test_a_playing_deck_moves_steadily_between_its_messages(self):
        for rate in (60.0, 120.0):
            points = _heard(0.0, 2.0, lambda t: 30.0 + 1.06 * t, rate)       # +6 %
            shown, moves = [], []
            for frame in range(60):                                         # the window's frames, 60 a second
                now = 1.0 + frame / 60
                heard = [p for p in points if p[0] <= now]
                shown.append(motion(heard, now).position)
            moves = [b - a for a, b in zip(shown, shown[1:])]
            with self.subTest(rate=rate):
                self.assertTrue(all(abs(m - 1.06 / 60) < 0.002 for m in moves), moves)     # within 2 ms a frame
                self.assertAlmostEqual(motion(points, 2.0).speed, 1.06, delta=0.02)

    def test_standing_on_a_cue_shown_still_whatever_the_sender_flags(self):
        points = _heard(0.0, 1.0, lambda t: 30.0 + t) + _heard(1.0 + 1 / 120, 1.5, lambda t: 12.0)   # CUE: back, still
        for now in (1.05, 1.2, 1.4, 1.5, 1.6):
            heard = [p for p in points if p[0] <= now]
            m = motion(heard, now)
            self.assertEqual((m.position, m.speed), (heard[-1][1], 0.0))

    def test_moving_again_from_where_it_stood(self):
        points = _heard(0.0, 1.0, lambda t: 12.0) + _heard(1.0 + 1 / 120, 1.1, lambda t: 12.0 + (t - 1.0))
        self.assertAlmostEqual(motion(points, 1.1).speed, 1.0, delta=0.05)  # not slowed by the still part

    def test_a_sender_gone_quiet_stops_where_it_was_going(self):
        points = _heard(0.0, 1.0, lambda t: 30.0 + t)
        self.assertAlmostEqual(motion(points, 9.0).position, 31.0 + AHEAD_SEC, delta=0.01)
        self.assertIsNone(motion([], 1.0))


class SourceTests(unittest.TestCase):
    def test_decks_received_and_matched(self):
        row = _row("a", "/m/song.flac", "Song", "Band", 240.0)
        source = DeckSource(Where(host="127.0.0.1", port=_free_port()), (row,))
        try:
            host, port = source.listening_on().split(":")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(osc.message(ob.TRACK_ADDRESS, 2, "Song", "Band", 240.0), (host, int(port)))
                s.sendto(osc.message(ob.TIME_ADDRESS, 2, "", True, True, 12.5), (host, int(port)))
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and source.decks()[1].state is None:
                time.sleep(0.01)
            decks = source.decks()
            self.assertEqual([d.deck for d in decks], [1, 2, 3, 4])
            self.assertIsNone(decks[0].state)
            self.assertIs(decks[1].row, row)
            self.assertEqual(decks[1].position_sec, 12.5)
            self.assertTrue(decks[1].shown)
            self.assertEqual(list(source.status()["decks"]), [2])
            source.set_tracks(())                              # the library changed: matched again
            self.assertIsNone(source.decks()[1].row)
        finally:
            source.close()

    def test_a_port_taken_refused(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            with self.assertRaises(OSError):
                DeckSource(Where(host="127.0.0.1", port=s.getsockname()[1]))


def echo_window(settings, commands, messages):
    """A stand-in for the window's process (given the app's settings, pickled): asks for the analysis of each track
    it is given; once one comes, says it listens on the next port after its settings'; told the settings changed,
    on the one after. Ends on close."""
    port = int(settings["cfg"].deckconfig.monitor_port)
    while True:
        message = commands.get(timeout=20)
        kind = message["type"]
        if kind == "close":
            break
        if kind == "tracks":
            for track in message["tracks"]:
                messages.put({"type": "analysis?", "uid": track.uid})
        elif kind == "analysis" and message.get("analysis") == "the analysis":
            messages.put({"type": "listening", "host": "127.0.0.1", "port": port + 1})
        elif kind == "settings":
            messages.put({"type": "listening", "host": "127.0.0.1", "port": port + 2})
    messages.put({"type": "done"})


class ControllerTests(unittest.TestCase):
    def test_a_window_process_served(self):
        from PySide6 import QtCore, QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        row = _row("a", "/m/song.flac", "Song", "Band", 240.0)
        read = []

        class Library:
            def read(self, job, uid, done=None, failed=None):
                read.append(uid)
                done("the analysis")

        cfg = default_cfg()
        cfg.deckconfig.monitor_port = 40000
        snapshot = _Snapshot((row,))           # the same until the library changes, as the owner's
        c = DeckMonitorController(target=echo_window, settings=lambda: cfg,
                                  snapshot_provider=lambda: snapshot, library=Library())
        opened = []
        c.open_changed.connect(opened.append)
        c.open()

        def wait_for(port):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and cfg.deckconfig.monitor_port != port:
                app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            return cfg.deckconfig.monitor_port

        self.assertEqual(wait_for(40001), 40001)       # its track's analysis read and answered; where it listens kept
        self.assertEqual(read, ["a"])                  # read by the library's owner
        c.settings_changed()
        self.assertEqual(wait_for(40002), 40002)       # the settings sent again
        c.close()
        self.assertEqual(opened, [True, False])
        self.assertFalse(c.is_open)

    def test_the_phrase_unit_clicked_in_the_window_kept(self):
        from PySide6 import QtWidgets
        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cfg = default_cfg()
        c = DeckMonitorController(target=echo_window, settings=lambda: cfg, snapshot_provider=lambda: None)
        c._on_message({"type": "phrase_unit", "unit": "label"})
        self.assertEqual(cfg.deckconfig.monitor_phrase_unit, "label")
        c._on_message({"type": "phrase_unit", "unit": "bars"})                # not a unit: refused
        self.assertEqual(cfg.deckconfig.monitor_phrase_unit, "label")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
