"""Search by Music's listener (ui.library.music_search.MusicListener): a search started before the microphone was turned
off and on again is the session before's: its result (the audio heard then) is not joined into the new one. The
microphone, the index and the search stood in for; the search run when the test says."""
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from unit.ui_tests.qt_env import APP  # noqa: F401

from analyzer.search.music.results import MusicMatch, SearchResult
from core.workflows.event_bus import EventBus
from ui.library import music_search

RATE = 8000


def _cfg():
    return SimpleNamespace(searchconfig=SimpleNamespace(music_listen_window=4.0, music_listen_interval=1.0),
                           deckconfig=SimpleNamespace(music_input_device=""))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.listener = music_search.MusicListener(EventBus(), _cfg, library=mock.Mock())
        self.listener._index, self.listener._stale = object(), False     # an index read: none loaded here
        opened = mock.patch.object(music_search.MusicListener, "_open_source", side_effect=self._open)
        closed = mock.patch.object(music_search.MusicListener, "_close_source", side_effect=self._close)
        opened.start()
        closed.start()
        self.addCleanup(opened.stop)
        self.addCleanup(closed.stop)
        self.shown = []
        self.listener.results.connect(self.shown.append)

    def _open(self):
        self.listener._source, self.listener.rate = object(), RATE
        return True

    def _close(self):
        self.listener._source = None
        self.listener._live, self.listener._live_n = [], 0

    def _search_started(self):
        """A tick with 4 s heard: the search's thread taken, not run."""
        tone = 0.1 * np.sin(np.arange(4 * RATE) * 0.05)
        self.listener._live, self.listener._live_n = [tone], tone.size
        threads = []
        with mock.patch.object(music_search.threading, "Thread",
                               side_effect=lambda **kw: threads.append(kw) or mock.Mock()):
            self.listener._on_tick()
        self.assertEqual(len(threads), 1)
        return threads[0]

    @staticmethod
    def _run(thread, found: str):
        result = SearchResult("recording", (MusicMatch(found, "recording", 30.0, 10.0, 14.0, 0.0),), 0.0)
        search = SimpleNamespace(search_audio=lambda *_a, **_k: result)
        with mock.patch.dict(sys.modules, {"analyzer.search.music.search": search}):
            thread["target"](*thread["args"])

    def test_a_search_from_before_the_microphone_turned_off_and_on_is_left_out(self):
        self.listener.start()
        old = self._search_started()
        self.listener.stop()
        self.listener.start()                     # a new session: the other song asked for
        self._run(old, "song-heard-before")       # the first session's search ends now
        self.assertEqual(self.shown, [])
        self.assertFalse(self.listener._searching)
        self._run(self._search_started(), "song-now")
        self.assertEqual([m.uid for m in self.shown[-1].matches], ["song-now"])

    def test_a_search_of_this_session_shown(self):
        self.listener.start()
        self._run(self._search_started(), "song-now")
        self.assertEqual([m.uid for m in self.shown[-1].matches], ["song-now"])


if __name__ == "__main__":
    unittest.main()
