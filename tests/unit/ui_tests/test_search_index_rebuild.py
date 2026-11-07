"""The Search Index tab's Rebuild Audio Fingerprints (ui.library.tabs.track_search_index): the track dropped from the
indexes, Index Now asked, its run's end checked. A run that ends while the track is still being dropped is another's
(not taken as the rebuild's), and a rebuild after a failed one starts afresh. A stand-in library: its writes answered
by the test."""
import os
import tempfile
import unittest

from unit.ui_tests.qt_env import APP  # noqa: F401

from core.library import jobs
from core.library.records import TrackRow
from core.workflows.event_bus import EventBus
from ui.library.tabs.track_search_index import SearchIndexPanel


class _Library:
    def __init__(self):
        self.writes = []            # (job, done, failed) not answered yet
        self.indexed = True         # what index_state reads

    def read(self, job, *args, done=None, failed=None):
        if job is jobs.index_state:
            done((self.indexed, self.indexed))
        elif job is jobs.readings:
            done(({}, {}))

    def write(self, job, *args, done=None, failed=None):
        self.writes.append((job, done, failed))


class RebuildTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.bus = EventBus()
        self.library = _Library()
        self.asked = []
        self.bus.sig_music_index_requested.connect(self.asked.append)
        self.panel = SearchIndexPanel(self.bus, self.library)
        self.panel.set_track(TrackRow(uid="u1", path=self.path))

    def tearDown(self):
        self.panel.deleteLater()
        os.remove(self.path)

    def dropped(self):
        _job, done, _failed = self.library.writes.pop()
        done(None)

    def test_a_run_ending_while_the_track_is_dropped_is_not_the_rebuilds(self):
        self.panel.rebuild()
        self.bus.sig_music_index_finished.emit(None, None)    # a run started before: still indexed, not ours
        self.assertNotIn("Rebuilt", self.panel.lbl_rebuild.text())
        self.library.indexed = False
        self.dropped()
        self.assertEqual(len(self.asked), 1)                  # Index Now asked once dropped
        self.library.indexed = True
        self.bus.sig_music_index_finished.emit(None, None)    # its run
        self.assertEqual(self.panel.lbl_rebuild.text(), "Rebuilt.")
        self.assertEqual(len(self.asked), 1)

    def test_a_rebuild_after_a_failed_one_starts_afresh(self):
        self.panel.rebuild()
        self.dropped()
        self.bus.sig_music_index_failed.emit(None, "no disk")
        self.assertIn("no disk", self.panel.lbl_rebuild.text())
        self.panel.rebuild()
        self.library.indexed = False
        self.panel._show_index_state()                        # the track shown again meanwhile: a plain read
        self.assertIn("Taking the track out", self.panel.lbl_rebuild.text())
        self.dropped()
        self.library.indexed = True
        self.bus.sig_music_index_finished.emit(None, None)
        self.assertEqual(self.panel.lbl_rebuild.text(), "Rebuilt.")


if __name__ == "__main__":
    unittest.main()
