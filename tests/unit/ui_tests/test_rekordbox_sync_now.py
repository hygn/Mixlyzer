"""Settings > Library's Rekordbox Sync Now (ui.config.library) and the sync queue (core.workflows.rekordbox_sync): the
page's settings saved first; the whole sync that save starts (the window alone decides it:
libconfig.rekordbox_xml_resync) is the one shown, else the page asks its own; two whole syncs of the same settings
waiting are one. A stand-in window: the save answered as AppWindow._on_settings_save answers it."""
import unittest
from pathlib import Path

from unit.ui_tests.qt_env import APP, isolated  # noqa: F401

from core.settings.config import default_cfg
from core.workers.rekordbox_sync import RekordboxSyncRequest
from core.workflows.event_bus import EventBus
from core.workflows.rekordbox_sync import RekordboxXmlSync
from ui.config.window import SettingsDialog

_ISOLATION = None


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


class _Queue(RekordboxXmlSync):
    """The queue without its worker thread: a request started is reported, then holds the queue (busy)."""

    def __init__(self, cfg_getter):
        super().__init__(cfg_getter=cfg_getter, library=None)
        self.ran = []

    def _start_next(self):
        if self._thread is not None or not self._queue:
            return
        request = self._queue.pop(0)
        self._thread = object()
        self.ran.append(request)
        self.started.emit(request)


class SyncNowTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.saved = default_cfg()
        self.queue = _Queue(lambda: self.saved)
        self.bus.sig_rekordbox_sync_requested.connect(self.queue.sync_requested)
        self.queue.started.connect(self.bus.sig_rekordbox_sync_started)
        self.bus.sig_setting_saveJsonRequested.connect(self._save)
        self.d = SettingsDialog(self.bus)
        self.d.set_config(default_cfg())

    def tearDown(self):
        self.d.close()
        self.d.deleteLater()

    def _save(self, cfg):
        """AppWindow._on_settings_save's part: the XML written again when its settings changed."""
        rewrite = cfg.libconfig.rekordbox_xml_resync(self.saved.libconfig)
        self.saved = cfg
        if rewrite:
            self.bus.sig_rekordbox_sync_requested.emit(True)

    def test_settings_unchanged_the_page_asks_its_own(self):
        self.d._current_cfg = None            # the page knowing no settings before: it predicts nothing
        self.d._sync_rekordbox_now()
        self.assertEqual(len(self.queue.ran), 1)
        self.assertTrue(self.queue.ran[0].show_progress)
        self.assertIs(self.d._rekordbox_progress_request, self.queue.ran[0])

    def test_the_sync_the_save_starts_is_the_one_shown(self):
        self.d.ed_rekordbox_xml_path.setText("./elsewhere.xml")
        self.d._sync_rekordbox_now()
        self.assertEqual(len(self.queue.ran), 1)              # one XML written, not two
        self.assertIs(self.d._rekordbox_progress_request, self.queue.ran[0])

    def test_busy_the_saves_sync_and_sync_now_are_one(self):
        self.queue._thread = object()                         # another sync running
        self.d.ed_rekordbox_xml_path.setText("./elsewhere.xml")
        self.d._sync_rekordbox_now()
        self.assertEqual(len(self.queue._queue), 1)
        self.assertTrue(self.queue._queue[0].show_progress and self.queue._queue[0].full_rebuild)
        self.queue._thread = None
        self.queue._start_next()
        self.assertIs(self.d._rekordbox_progress_request, self.queue.ran[0])


class QueueMergeTests(unittest.TestCase):
    def test_whole_syncs_of_other_settings_stay_apart(self):
        queue = _Queue(lambda: None)
        queue._thread = object()
        a = RekordboxSyncRequest(library_dir=Path("lib"), xml_path=Path("a.xml"), mode="library")
        queue._enqueue(a)
        queue._enqueue(RekordboxSyncRequest(library_dir=Path("lib"), xml_path=Path("b.xml"), mode="library"))
        queue._enqueue(RekordboxSyncRequest(library_dir=Path("lib"), xml_path=Path("a.xml"), mode="library",
                                            show_progress=True))
        self.assertEqual([(r.xml_path.name, r.show_progress) for r in queue._queue], [("a.xml", True),
                                                                                      ("b.xml", False)])


if __name__ == "__main__":
    unittest.main()
