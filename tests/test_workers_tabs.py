"""The Workers window's tabs per task kind, and a new track's background tie-up lookup reported as a task."""
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.library import jobs
from core.tieups import web as tieup_web
from core.workflows import tieup_web as tieup_web_lookup
from core.workflows.event_bus import EventBus
from core.workflows.taskmanager import ANALYSIS, TIEUP_LOOKUP, taskmanager
from core.workflows.tieup_web import TieupWebLookup
from ui.windows.workers import WorkersDialog

APP = QApplication.instance() or QApplication([])


class WorkersTabTests(unittest.TestCase):
    def test_a_tab_per_kind(self):
        bus = EventBus()
        tasks = taskmanager(bus)
        dialog = WorkersDialog()
        dialog.set_bus(bus)
        a = tasks.addtask("song", None, "Decoding", 0.1)
        b = tasks.addtask("song 2", None, "Waiting", 0.0, kind=TIEUP_LOOKUP)
        tasks.addtask("song 3", None, "Waiting", 0.0, kind=TIEUP_LOOKUP)
        self.assertEqual((dialog.lists[ANALYSIS].count(), dialog.lists[TIEUP_LOOKUP].count()), (1, 2))
        self.assertTrue(dialog.tabs.tabText(1).endswith("(2)"))
        tasks.rmtask(b.taskid)
        tasks.rmtask(a.taskid)
        self.assertEqual((dialog.lists[ANALYSIS].count(), dialog.lists[TIEUP_LOOKUP].count()), (0, 1))
        self.assertFalse(dialog.tabs.tabText(0).endswith(")"))

    def test_a_background_lookup_reported(self):
        class Owner:        # the library's owner: the song asked for, the tie-up given to write
            writes = 0

            def call(self, _job, *_args, write=False):
                Owner.writes += write
                return None if write else ("曲", "歌手", "")

        lookup = TieupWebLookup(library=Owner(), enabled=lambda: True)
        events = []
        lookup.queued.connect(lambda uid, title: events.append(("queued", uid, title)))
        lookup.started.connect(lambda uid: events.append(("started", uid)))
        lookup.done.connect(lambda uid, outcome: events.append(("done", uid, outcome)))
        with mock.patch.object(tieup_web, "sure_of", return_value=SimpleNamespace(as_tieup=lambda: None)):
            lookup.new_track("u1", "曲")
            deadline = time.monotonic() + 5
            while len(events) < 3 and time.monotonic() < deadline:
                APP.processEvents()
                time.sleep(0.01)
        self.assertEqual(events, [("queued", "u1", "曲"), ("started", "u1"), ("done", "u1", "tagged")])
        self.assertEqual(Owner.writes, 1)        # written by the owner, not the lookup's thread

    def test_a_lookup_asked_while_the_thread_ends_is_done(self):
        """A new track put while the idle thread asks its albums' works (its last step): looked up all the same."""
        album_asked, release = threading.Event(), threading.Event()
        album_calls = []

        class Owner:
            def call(self, job, *args, write=False):
                if job is jobs.tag_by_album:
                    album_calls.append(list(args[0]))
                    if len(album_calls) == 2:       # the idle thread's: the track put now, then it goes on
                        album_asked.set()
                        release.wait(5.0)
                    return []
                return None if write else ("曲", "歌手", "")

        lookup = TieupWebLookup(library=Owner(), enabled=lambda: True)
        done = []
        lookup.done.connect(lambda uid, outcome: done.append(uid))
        with mock.patch.object(tieup_web_lookup, "IDLE_SEC", 0.05), \
                mock.patch.object(tieup_web, "sure_of", return_value=None):
            lookup.new_track("u1", "曲")                # nothing found: its album asked at once, then when idle
            self.assertTrue(album_asked.wait(5.0))
            lookup.new_track("u2", "曲 2")
            release.set()
            deadline = time.monotonic() + 5
            while "u2" not in done and time.monotonic() < deadline:
                APP.processEvents()
                time.sleep(0.01)
        self.assertIn("u2", done)


if __name__ == "__main__":
    unittest.main()
