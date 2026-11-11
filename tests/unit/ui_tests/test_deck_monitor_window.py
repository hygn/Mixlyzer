"""Deck Monitor's window (ui.windows.deck_monitor, run in a process of its own): each deck drawn by Mixlyzer's own
main view and overview on a model of its own (a library track's analysis asked once and again for a new library, its
audio decoded once and loaded as the app loads a track); the main views' zoom following the deck's tempo; the deck
panel's BPM, tempo, key, MASTER and phrases; where it listens set in the window (a port taken said, the one before
kept)."""
import socket
import unittest

import numpy as np

from unit.ui_tests.qt_env import APP, drain  # noqa: F401  (offscreen Qt, one QApplication)

from analyzer.music.key_names import idx_to_labels
from analyzer.music.marks import Phrase
from core.analysis_library.stored import StoredAnalysis
from core.deck_monitor.decks import DeckState, TrackInfo
from core.deck_monitor.source import MonitorDeck
from core.library.records import TrackRow
from core.settings.config import default_cfg
from ui.windows import deck_monitor as dm
from views.timeline import DEFAULT_WINDOW_SEC, step_window

ROW = TrackRow(uid="a", path="/m/a.flac", title="Song", artist="Band", duration=64.0)
ANALYSIS = StoredAnalysis(duration_sec=64.0, beats=np.arange(0.0, 64.0, 0.5),
                          tempo_segments=np.array([[0.0, 64.0, 120.0, 0.0, 4, 120.0]]),
                          key_segments=np.array([[8, 1, 0.0, 64.0]]),
                          phrases=(Phrase(0.0, 16.0, "INTRO"), Phrase(16.0, 32.0, "VERSE"),
                                   Phrase(32.0, 48.0, "VERSE"), Phrase(48.0, 64.0, "CHORUS")))


class _Source:
    def __init__(self):
        self.position, self.bpm, self.track_bpm, self.master = 30.0, 0.0, 0.0, None

    def decks(self):
        state = DeckState(1, True, True, seconds=self.position, track=TrackInfo(title="Song"), bpm=self.bpm,
                          track_bpm=self.track_bpm, master=self.master)
        return [MonitorDeck(1, state, False, ROW, self.position, 0.0, 1.0),
                *(MonitorDeck(n, None, True, None, None, 0.0) for n in (2, 3, 4))]

    def set_tracks(self, tracks):
        pass

    def status(self):
        return {"listening": "127.0.0.1:1", "decks": {}}

    def close(self):
        pass


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DeckMonitorWindowTests(unittest.TestCase):
    def setUp(self):
        self.messages = _Queue()
        self.w = dm.DeckMonitorWindow(default_cfg(), self.messages)
        self.w.source = self.source = _Source()
        self.decodes = []
        self.w.decoders.start = lambda job: self.decodes.append(job.uid)     # decoded by the test

    def tearDown(self):
        self.w.shut_down()
        self.w.close()
        self.w.deleteLater()
        drain()

    def asked(self):
        return [m["uid"] for m in self.messages.items if m["type"] == "analysis?"]

    def loaded(self):
        self.w.refresh()
        self.w.command({"type": "analysis", "uid": "a", "gen": 0, "analysis": ANALYSIS})
        self.w.refresh()

    def test_the_analysis_asked_once_and_again_for_a_new_library(self):
        self.w.refresh()
        self.w.refresh()
        self.assertEqual(self.asked(), ["a"])
        self.assertEqual(self.decodes, ["a"])
        self.w.command({"type": "tracks", "gen": 1, "tracks": ()})        # the library changed meanwhile
        self.w.command({"type": "analysis", "uid": "a", "gen": 0, "analysis": StoredAnalysis()})   # dropped
        self.assertIsNone(self.w.tracks["a"].analysis)
        self.w.refresh()
        self.assertEqual(self.asked(), ["a", "a"])
        self.w.command({"type": "analysis", "uid": "a", "gen": 1, "analysis": None})   # none could be read
        self.w.refresh()
        self.assertEqual(self.asked(), ["a", "a"])                       # not asked every frame

    def test_another_tracks_change_reloads_nothing_shown(self):
        self.loaded()
        shown = self.w.views[0].model.analysis
        self.w.command({"type": "tracks", "gen": 1, "tracks": (ROW,)})   # the library read again
        self.w.refresh()
        self.assertEqual(self.asked(), ["a", "a"])                       # asked again, shown meanwhile
        self.assertIs(self.w.views[0].model.analysis, shown)
        again = StoredAnalysis(duration_sec=64.0, beats=ANALYSIS.beats.copy(), tempo_segments=ANALYSIS.tempo_segments,
                               key_segments=ANALYSIS.key_segments, phrases=ANALYSIS.phrases)
        self.w.command({"type": "analysis", "uid": "a", "gen": 1, "analysis": again})
        self.w.refresh()
        self.assertIs(self.w.views[0].model.analysis, shown)             # the same marks: not loaded again
        edited = StoredAnalysis(duration_sec=64.0, beats=ANALYSIS.beats, tempo_segments=ANALYSIS.tempo_segments,
                                key_segments=ANALYSIS.key_segments, phrases=ANALYSIS.phrases[:2])
        self.w.command({"type": "tracks", "gen": 2, "tracks": (ROW,)})
        self.w.refresh()
        self.w.command({"type": "analysis", "uid": "a", "gen": 2, "analysis": edited})
        self.w.refresh()
        self.assertIs(self.w.views[0].model.analysis, edited)            # edited: shown

    def test_the_deck_loaded_into_the_main_views_model(self):
        views = self.w.views[0]
        self.assertIn("WaveformView", {type(p).__name__ for p in views.plugins})     # the app's own views
        self.loaded()
        self.assertEqual(views.model.duration_sec, 64.0)
        self.assertEqual(views.model.beats.size, 128)
        pcm = np.zeros(64 * dm.RATE, dtype=np.float32)
        self.w._decoded("a", pcm, None)
        self.w.refresh()
        self.assertIs(views.model.predecoded_pcm, pcm)
        self.assertEqual(views.model.predecoded_rate, dm.RATE)
        self.assertEqual(views.time, 30.0)
        self.assertEqual(views.tl.current_time, 30.0)

    def test_the_zoom_follows_the_decks_tempo(self):
        self.source.bpm, self.source.track_bpm = 132.0, 120.0
        self.w.refresh()
        self.assertAlmostEqual(self.w.views[0].window, DEFAULT_WINDOW_SEC * 1.1)   # one span of time: more track
        self.assertEqual(self.w.views[1].window, DEFAULT_WINDOW_SEC)
        self.w.zoom(-1)
        self.w.refresh()
        self.assertAlmostEqual(self.w.views[0].window, step_window(DEFAULT_WINDOW_SEC, -1) * 1.1)
        self.assertAlmostEqual(self.w.views[0].tl.window_sec, self.w.views[0].window)

    def test_the_panel_reads_the_deck(self):
        self.source.bpm, self.source.track_bpm, self.source.master = 132.0, 120.0, True
        self.loaded()
        panel = self.w.panels[0]
        self.assertEqual(panel.bpm.text(), "132.00")              # the grid's tempo there, played 10 % faster
        self.assertEqual(panel.pitch.text(), "+10.00%")           # as a CDJ shows the tempo
        self.assertEqual(panel.key.text(), idx_to_labels(20)[0])
        self.assertEqual((panel.remaining.text(), panel.elapsed.text()), ("−00:34.0", "00:30.0"))
        self.assertEqual((dm.deck_clock(-3.04), dm.deck_clock(-0.04)), ("−00:03.0", "00:00.0"))  # before the start
        self.assertTrue(panel.master.property("lit"))
        self.assertFalse(self.w.panels[1].master.isVisibleTo(self.w.panels[1]))   # no deck there: no MASTER

        self.assertEqual([p.body.indexOf(p.inner) for p in self.w.panels], [1, 0, 1, 0])   # facing the middle
        self.assertEqual([p.inner.layout().indexOf(p.record) == 0 for p in self.w.panels],  # the record above the
                         [True, True, False, False])                                        # BPM on 1, 2; under on 3, 4
        self.assertEqual(len({p.inner.width() for p in self.w.panels}), 1)                  # one width: they line up

        def phrases(panel):
            line = panel.phrase_line
            return (line.now_name.text(), line.now_bars.text(), line.next_name.text(), line.next_bars.text(),
                    line.left.text(), line.left_time.text())

        self.assertEqual(phrases(panel), ("VERSE1", "8", "VERSE2", "8", "1.0", "00:02.0"))   # by section; NEXT IN
        panel.phrase_line.clicked.emit()                                   # clicked: by label, every deck
        self.assertEqual(self.messages.items[-1], {"type": "phrase_unit", "unit": "label"})
        self.assertEqual({p.phrase_line.unit for p in self.w.panels}, {"label"})
        self.assertEqual(phrases(panel), ("VERSE", "16", "CHORUS", "8", "9.0", "00:18.0"))

    def test_the_phrase_line_never_moves_when_its_values_change(self):
        from core.deck_monitor.readout import PhraseNow, PhraseSpan
        line = self.w.panels[0].phrase_line

        def widths():
            parts = (line.now_name, line.now_bars, line.left, line.left_time, line.next_name, line.next_bars,
                     line.now_caption, line.next_caption, *line.units)
            assert all(w.minimumWidth() == w.maximumWidth() > 0 for w in parts)    # fixed
            return [w.maximumWidth() for w in parts] + [line.sizeHint().width()]

        before = widths()
        short = PhraseSpan("VERSE", "V", (1, 2, 3), 0.0, 1.0, 1)
        long = PhraseSpan("INTERLUDE", "INTERLUDE10", (1, 2, 3), 1.0, 2.0, 128)
        line.show_phrases(PhraseNow(short, long, 513, 4), 0.5)
        self.assertEqual(widths(), before)
        line.show_phrases(PhraseNow(long, short, 1, 4), 1.5)
        line.set_unit("label")
        line.show_phrases(None)
        self.assertEqual(widths(), before)                       # every cell as wide as its widest value

    def test_a_track_not_in_the_library_shown_by_the_senders_own_marks(self):
        from dataclasses import replace
        from core.deck_monitor.source_marks import SourceCue, SourceMarks, TempoChange
        marks = SourceMarks(grid=(TempoChange(0.5, 120.0, 1),), cues=(SourceCue(10.0, "Drop", (255, 40, 40)),),
                            key=21)
        sent = {"marks": marks, "duration": 64.0}

        def decks():
            state = DeckState(1, True, True, seconds=30.0, track=TrackInfo(title="Elsewhere", duration_sec=sent["duration"]),
                              bpm=132.0, track_bpm=120.0, marks=sent["marks"])
            return [MonitorDeck(1, state, False, None, 30.0, 0.0, 1.0),
                    *(MonitorDeck(n, None, True, None, None, 0.0) for n in (2, 3, 4))]

        self.source.decks = decks
        self.w.refresh()
        views, panel = self.w.views[0], self.w.panels[0]
        self.assertEqual(views.model.beats.size, 128)                          # its grid (0.5 - 64 s), no audio
                                                                                # asked for
        self.assertEqual([c.label for c in views.model.cue_points], ["Drop"])
        self.assertEqual((panel.bpm.text(), panel.key.text()), ("132.00", idx_to_labels(21)[0]))
        self.assertEqual((self.asked(), self.decodes), ([], []))
        shown = views.uid
        sent["marks"] = replace(marks, key=0)                                  # the same track's marks again
        self.w.refresh()
        self.assertEqual((views.uid, panel.key.text()), (shown, idx_to_labels(0)[0]))
        sent["duration"] = 0.0                                                 # its length not known: none
        self.w.refresh()
        self.assertEqual(views.model.beats.size, 0)

    def test_the_phrase_unit_as_kept(self):
        cfg = default_cfg()
        cfg.deckconfig.monitor_phrase_unit = "label"
        w = dm.DeckMonitorWindow(cfg)
        try:
            self.assertEqual({p.phrase_line.unit for p in w.panels}, {"label"})
        finally:
            w.shut_down()
            w.close()
            w.deleteLater()

    def test_where_it_listens_set_in_the_window(self):
        self.w.source = None
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as taken:
            taken.bind(("127.0.0.1", 0))
            busy = taken.getsockname()[1]
            self.assertFalse(self.w.listen("127.0.0.1", busy))
            self.assertIsNone(self.w.source)
            self.assertIn(str(busy), self.w.connection.status.text())         # said in the window
            port = _free_port()
            self.assertTrue(self.w.listen("127.0.0.1", port))
            self.assertEqual(self.messages.items[-1], {"type": "listening", "host": "127.0.0.1", "port": port})
            self.assertFalse(self.w.listen("127.0.0.1", busy))
            self.assertEqual(self.w.source.address, ("127.0.0.1", port))      # the one before kept
        self.w.connection.set_address("192.168.0.5", port)
        self.assertFalse(self.w.connection.network.isHidden())               # a network address: warned
        self.w.connection.set_address("", port)
        self.assertTrue(self.w.connection.network.isHidden())


if __name__ == "__main__":
    unittest.main()
