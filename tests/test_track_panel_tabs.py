"""The track panel's Search Index and Tie-ups tabs and the works window, on a library of their own (offscreen Qt)."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from core.tieups import domain as tieup
from core.settings.config import default_cfg
from core.workflows.event_bus import EventBus
from core.library import tieups
from core.library.db import LibraryDB
from core.library.records import TrackRow
from core.library.service import LibraryService
from ui.tieup_editor import TieupPanel
from ui.tieup_works import WorksWindow
from ui.track_search_index import SearchIndexPanel

APP = QApplication.instance() or QApplication([])


class TrackPanelTabTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = self._tmp.name
        self.db = LibraryDB(str(Path(self.lib) / "library.db"))
        self.db.connect()
        self.uids = [str(uuid4()) for _ in range(2)]
        for i, uid in enumerate(self.uids):
            self.db.upsert(TrackRow(uid=uid, path=f"{i}.wav", title=f"曲{i}", artist="歌手", added_ts=1))
        self.db.conn.commit()
        self.cfg = default_cfg()
        self.cfg.libconfig.libpath = self.lib
        self.bus = EventBus()
        self.updates = []
        self.bus.sig_lib_updated.connect(self.updates.append)
        # the library's owner: the panels ask it, its thread reads and writes
        self.library = LibraryService(self.lib, settings=lambda: self.cfg.searchconfig, bus=self.bus)
        self.settle()

    def settle(self) -> None:
        """The jobs asked so far run and their answers handed over (an answer may ask again: a few rounds)."""
        for _ in range(4):
            self.library.wait()

    def tearDown(self):
        self.library.close()
        self.db.close()
        self._tmp.cleanup()

    def test_search_index_tab_saves_readings_alone(self):
        panel = SearchIndexPanel(bus=self.bus, library=self.library, search_settings=lambda: self.cfg.searchconfig)
        panel.set_track(self.db.get_by_uid(self.uids[0]))
        self.settle()
        self.assertFalse(panel.has_changes())
        item = panel.tbl_readings.item(5, 0)
        item.setText("my reading")
        self.assertTrue(panel.has_changes())
        self.updates.clear()
        self.assertTrue(panel.apply())
        self.assertFalse(panel.has_changes())
        self.settle()
        self.assertIn(("my reading", "user"), self.db.readings(self.uids[0])["title"])
        self.assertEqual(self.db.get_by_uid(self.uids[0]).title, "曲0")      # the metadata untouched
        self.assertTrue(self.updates)

    def test_tieups_tab_and_the_works_window(self):
        w = tieup.Work("", "anime", [("ja-JP", "作品")])
        other = tieup.Work("", "game", [("ja-JP", "作品の別項目")])
        with self.db.conn:
            tieups.save_tieup(self.db.conn, self.uids[0], tieup.TieUp(w, "opening"))
            tieups.save_tieup(self.db.conn, self.uids[1], tieup.TieUp(other, "ending"))
        panel = TieupPanel(bus=self.bus, library=self.library, settings=lambda: self.cfg.searchconfig)
        panel.set_track(self.db.get_by_uid(self.uids[0]))
        self.settle()
        self.assertEqual(panel.box.table.rowCount(), 1)
        self.assertEqual(panel.track_uid(), self.uids[0])

        window = WorksWindow(self.bus, lambda: self.cfg, self.library)

        def find(text: str) -> None:        # typed, the pause waited out
            window.ed_find.setText(text)
            window._find_timer.stop()
            window._find_timer.timeout.emit()

        window.show_works()
        self.settle()
        self.assertEqual(window.table.rowCount(), 2)
        find("べつこうもく")                           # found by its reading
        self.assertEqual(window.table.rowCount(), 1)
        find("")
        # edited as a whole
        window._fill(w.id)
        self.assertEqual(window._current.id, w.id)
        window.tbl_titles.set_values([("ja-JP", "作品"), ("ko-KR", "작품")])
        self.assertTrue(window._changed())
        find("べつこうもく")       # the list narrowed past it: its edits kept in the editor
        find("")
        self.assertEqual(window._current.id, w.id)
        self.assertTrue(window._changed())
        self.assertTrue(window._save())
        self.assertFalse(window._changed())          # meanwhile stored as edited
        self.settle()
        self.assertEqual(tieups.work(self.db.conn, w.id).title("ko-KR"), "작품")
        self.assertEqual(window.list_tracks.count(), 1)     # its track, read by the owner
        # the other entry deleted (asked: yes)
        window._fill(other.id)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            window._delete()
        self.settle()
        self.assertIsNone(tieups.work(self.db.conn, other.id))
        self.assertEqual(tieups.track_tieups(self.db.conn, self.uids[1]), [])
        self.assertEqual(window.table.rowCount(), 1)
        window.close()


if __name__ == "__main__":
    unittest.main()
