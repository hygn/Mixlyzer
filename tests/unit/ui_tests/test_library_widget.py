"""The library widget on a library of its own (offscreen Qt): the track table made from the owner's snapshot, the text
search and its modes, the phrase / transition filters, a library update changing only the rows it touches, the
selection acted on, playlists (an entry by its position), Variants, a folder's files, Search by Music's results, the
track panel following the selection and the remove action asking first."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import numpy as np
import shiboken6

from unit.ui_tests.qt_env import APP, isolated

from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox, QTreeWidgetItemIterator

from analyzer.audio.wave_overview import WaveOverview
from analyzer.music.key_names import idx_to_labels
from analyzer.search import variants as fpm
from analyzer.search.music.search import MusicMatch
from core.analysis_library.stored import StoredAnalysis
from core.library import jobs
from core.library import variants as variant_store
from core.library.db import LibraryDB
from core.library.records import BpmSegmentRow, TrackRow
from core.library.service import LibraryService
from core.settings.config import default_cfg
from core.track.model import DataModel
from core.workflows.event_bus import EventBus
from ui.library import widget as widget_module
from ui.library.common import _PLAYLIST_TRACKS_MIME, _SOURCE_KIND_ROLE, _SOURCE_VALUE_ROLE
from ui.library.widget import LibraryWidget
from ui.reanalyze_dialog import ReanalyzeDialog
from ui.theme import current_theme

_ISOLATION = None
TITLE, ARTIST, ALBUM, BPM, KEY, DURATION, RATING, PATH, INDEX = (
    LibraryWidget.HEADERS.index(h) for h in ("Title", "Artist", "Album", "BPM", "Key", "Duration", "Rating", "Path",
                                             "#"))


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


def features(seed: int, n: int = 600, bpm: float = 150.0) -> tuple:
    """A synthetic song's beats, overview bands and length (as the Variants' fingerprints read them)."""
    rng = np.random.default_rng(seed)
    bands = rng.integers(40, 250, size=(2048, 12)).astype(np.uint8)
    wave = WaveOverview(np.zeros((0, 2), np.int8), np.ones(1, np.float32), bands, np.ones(12, np.float32),
                        np.zeros(3, np.float32))
    return np.arange(n) * 60.0 / bpm, wave, n * 60.0 / bpm


def tree_items(tree):
    it = QTreeWidgetItemIterator(tree)
    while it.value():
        yield it.value()
        it += 1


class LibraryWidgetTests(unittest.TestCase):
    SONGS = (  # title, artist, album, bpm, key, duration, rating, added
        ("あのね", "歌手", "Album One", 128.0, 0, 200.0, 3, 1000),
        ("Hello World", "Band", "", 140.0, 5, 180.0, 0, 2000),
        ("Third Song", "Other", "Album One", 125.0, 12, 240.0, 5, 3000),
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = self._tmp.name
        self.music = os.path.join(self.lib, "music")
        os.makedirs(self.music)
        db = LibraryDB(os.path.join(self.lib, "library.db"))
        db.connect()
        self.uids, self.paths = [], []
        for i, (title, artist, album, bpm, key, duration, rating, added) in enumerate(self.SONGS):
            uid, path = str(uuid4()), os.path.join(self.music, f"{i}.wav")
            open(path, "wb").close()
            db.upsert(TrackRow(uid=uid, path=path, title=title, artist=artist, album=album, bpm=bpm, key=key,
                               duration=duration, rating=rating, added_ts=added))
            self.uids.append(uid)
            self.paths.append(path)
        # Hello World's tempo changes 125 -> 129 (the BPM transition filter's default values)
        db.replace_bpm_segments(self.uids[1], [BpmSegmentRow(self.uids[1], 0, 0.0, 90.0, 90.0, 125.0),
                                               BpmSegmentRow(self.uids[1], 1, 90.0, 180.0, 90.0, 129.0)])
        db.conn.commit()
        db.close()
        self.cfg = default_cfg()
        self.cfg.libconfig.libpath = self.lib
        self.cfg.libconfig.local_music_directories = [self.music]
        self.bus = EventBus()
        self.library = LibraryService(self.lib, settings=lambda: self.cfg.searchconfig, bus=self.bus)
        self.library.refresh()
        self.settle()
        self.model = DataModel()
        self.w = LibraryWidget(self.bus, self.cfg, self.library, model=self.model)

    def tearDown(self):
        self.w.close()
        self.w.deleteLater()
        self.settle()
        self.library.close()
        self._tmp.cleanup()

    # ---- helpers ----
    def settle(self) -> None:
        for _ in range(4):
            self.library.wait()

    def apply_update(self) -> None:
        """The owner's writes done and the library update applied (its delay waited out)."""
        self.settle()
        self.w._library_timer.stop()
        self.w._apply_library_update()

    def search(self, text: str, mode: str = "All") -> None:
        self.w.cb_column.setCurrentIndex(self.w.cb_column.findData(mode))
        self.w.e_search.setText(text)
        self.w._search_timer.stop()
        self.w._search_timer.timeout.emit()

    def column(self, col: int, shown_only: bool = True) -> list[str]:
        t = self.w.table
        return [t.item(r, col).text() for r in range(t.rowCount()) if not (shown_only and t.isRowHidden(r))]

    def row_of(self, uid: str) -> int:
        return next(r for r in range(self.w.table.rowCount()) if self.w._row_uid(r) == uid)

    def source(self, kind: str, value=None):
        return next(i for i in tree_items(self.w.source_tree) if i.data(0, _SOURCE_KIND_ROLE) == kind and (
            value is None or i.data(0, _SOURCE_VALUE_ROLE) == value))

    # ---- the table ----
    def test_rows_made_from_the_snapshot_newest_first(self):
        self.assertEqual(self.column(TITLE), ["Third Song", "Hello World", "あのね"])
        row = self.row_of(self.uids[0])
        t = self.w.table
        self.assertEqual(t.item(row, BPM).text(), "128.00")
        self.assertEqual(t.item(row, KEY).text(), idx_to_labels(0)[0])
        self.assertEqual(t.item(row, DURATION).text(), "03:20")
        self.assertEqual(t.item(row, RATING).text(), "★★★☆☆")
        self.assertEqual(t.item(row, PATH).text(), self.paths[0])

    def test_bpm_column_sorts_by_number(self):
        self.w.table.sortItems(BPM, QtCore.Qt.AscendingOrder)
        self.assertEqual(self.column(BPM), ["125.00", "128.00", "140.00"])

    # ---- the text search ----
    def test_typing_applies_once_it_pauses(self):
        self.w.e_search.setText("hello")
        self.assertTrue(self.w._search_timer.isActive())
        self.assertEqual(len(self.column(TITLE)), 3)           # not yet
        self.w._search_timer.stop()
        self.w._search_timer.timeout.emit()
        self.assertEqual(self.column(TITLE), ["Hello World"])

    def test_search_modes_look_in_their_column(self):
        self.search("hello")
        self.assertEqual(self.column(TITLE), ["Hello World"])
        self.search("band", "Artist")
        self.assertEqual(self.column(TITLE), ["Hello World"])
        self.search("band", "Title")
        self.assertEqual(self.column(TITLE), [])
        self.search("album one", "Album")
        self.assertEqual(self.column(TITLE), ["Third Song", "あのね"])
        self.search(idx_to_labels(5)[0].lower(), "Key")
        self.assertEqual(self.column(TITLE), ["Hello World"])
        self.search("")
        self.assertEqual(len(self.column(TITLE)), 3)

    def test_titles_found_by_their_reading(self):
        self.search("anone", "Title")
        self.assertEqual(self.column(TITLE), ["あのね"])
        self.search("아노네")
        self.assertEqual(self.column(TITLE), ["あのね"])

    def test_clear_removes_every_condition(self):
        self.search("hello")
        self.w.cb_transition_bpm.setChecked(True)
        self.w.btn_clear_filters.click()
        self.settle()
        self.assertEqual(self.w.e_search.text(), "")
        self.assertFalse(self.w.cb_transition_bpm.isChecked())
        self.assertEqual(len(self.column(TITLE)), 3)

    def test_bpm_transition_filter_and_the_text_joined_by_and_or(self):
        self.w.cb_transition_bpm.setChecked(True)          # 125 -> 129, the editors' defaults
        self.w._search_timer.stop()
        self.w._apply_filter()                             # asked of the owner; applied when it answers
        self.settle()
        self.assertEqual(self.column(TITLE), ["Hello World"])
        self.w.match_toggle.setChecked(True)               # OR
        self.search("third")
        self.settle()
        self.assertEqual(self.column(TITLE), ["Third Song", "Hello World"])
        self.w.match_toggle.setChecked(False)              # AND
        self.w._search_timer.stop()
        self.w._apply_filter()
        self.assertEqual(self.column(TITLE), [])

    # ---- the selection ----
    def test_hidden_rows_never_acted_on(self):
        self.w.table.selectAll()
        self.search("hello")
        self.assertEqual(self.w._selected_uids(), [self.uids[1]])
        self.assertEqual(len(self.w.table.selectionModel().selectedRows()), 1)    # the hidden ones deselected
        self.w.table.selectAll()                           # Ctrl+A selects hidden rows too
        self.assertEqual(len(self.w.table.selectionModel().selectedRows()), 3)
        self.assertEqual(self.w._selected_rows(), [self.row_of(self.uids[1])])
        self.assertEqual(self.w._selected_paths(), [self.paths[1]])

    def test_step_selection_skips_hidden_rows(self):
        self.search("o")             # all three
        self.search("song", "Title")
        self.w._step_selection(1)
        self.assertEqual(self.w._selected_uids(), [self.uids[2]])
        self.w._step_selection(1)    # nothing shown after it
        self.assertEqual(self.w._selected_uids(), [self.uids[2]])

    # ---- library updates ----
    def test_a_few_changes_change_their_rows_in_place(self):
        kept = self.w.table.item(self.row_of(self.uids[0]), TITLE)
        new_uid = str(uuid4())
        self.library.write(jobs.save_track, TrackRow(uid=new_uid, path=os.path.join(self.music, "n.wav"),
                                                     title="Newest", added_ts=4000), False)
        changed = self.library.snapshot.track(self.uids[1])
        self.library.write(jobs.save_track, replace(changed, title="Hello Again"), False)
        self.library.write(jobs.remove_tracks, [self.uids[2]])
        self.apply_update()
        self.assertTrue(shiboken6.isValid(kept))
        self.assertIs(self.w.table.item(self.row_of(self.uids[0]), TITLE), kept)
        self.assertEqual(self.column(TITLE), ["Newest", "Hello Again", "あのね"])     # each in its sorted place

    def test_more_changes_than_the_limit_rebuild_the_table(self):
        kept = self.w.table.item(self.row_of(self.uids[0]), TITLE)
        for uid, title in ((self.uids[1], "B"), (self.uids[2], "C")):
            row = self.library.snapshot.track(uid)
            self.library.write(jobs.save_track, replace(row, title=title), False)
        with mock.patch.object(widget_module, "INCREMENTAL_ROWS_MAX", 1):
            self.apply_update()
        self.assertFalse(shiboken6.isValid(kept))
        self.assertEqual(self.column(TITLE), ["C", "B", "あのね"])

    def test_the_search_kept_over_an_update(self):
        self.search("hello")
        for title, added in (("Hello Again", 4000), ("Unrelated", 5000)):
            self.library.write(jobs.save_track, TrackRow(uid=str(uuid4()), path=os.path.join(self.music, title),
                                                         title=title, added_ts=added), False)
        self.apply_update()
        self.assertEqual(self.column(TITLE), ["Hello Again", "Hello World"])
        self.assertEqual(len(self.column(TITLE, shown_only=False)), 5)

    def test_updates_in_a_row_applied_once(self):
        calls = []
        with mock.patch.object(self.w, "_refresh_current_source", side_effect=lambda **k: calls.append(k)):
            self.bus.sig_lib_updated.emit(list(self.library.snapshot.tracks))
            self.bus.sig_lib_updated.emit(list(self.library.snapshot.tracks))
            self.assertTrue(self.w._library_timer.isActive())
            self.w._library_timer.stop()
            self.w._apply_library_update()
            self.w._apply_library_update()      # nothing pending
        self.assertEqual(len(calls), 1)

    # ---- removing ----
    def test_remove_asks_first_no_by_default(self):
        self.w._select_row(self.row_of(self.uids[1]))
        synced = []
        self.bus.sig_rekordbox_sync_track_requested.connect(synced.append)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No) as asked:
            self.w._on_remove_selected()
        self.assertIn("Hello World", asked.call_args.args[2])
        self.assertEqual(asked.call_args.args[4], QMessageBox.No)       # the default answer
        self.apply_update()
        self.assertEqual(len(self.library.snapshot.tracks), 3)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.w._on_remove_selected()
        self.apply_update()
        self.assertIsNone(self.library.snapshot.track(self.uids[1]))
        self.assertEqual(self.column(TITLE), ["Third Song", "あのね"])
        self.assertEqual(synced, [self.uids[1]])

    def test_removing_the_decks_track_unloads_it(self):
        unloaded = []
        self.bus.sig_unload_track_requested.connect(lambda: unloaded.append(True))
        self.w._loaded_uid = self.uids[0]                 # the deck holds the first track
        self.w._select_row(self.row_of(self.uids[1]))
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.w._on_remove_selected()
        self.apply_update()
        self.assertEqual(unloaded, [])                    # another track removed: the deck keeps its own
        self.w._select_row(self.row_of(self.uids[0]))
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.w._on_remove_selected()
        self.apply_update()
        self.assertEqual(unloaded, [True])

    def test_removing_several_says_how_many(self):
        self.w.table.selectAll()
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No) as asked:
            self.w._on_remove_selected()
        self.assertIn("3", asked.call_args.args[2])

    # ---- playlists ----
    def _new_playlist(self, name: str) -> str:
        with mock.patch.object(QInputDialog, "getText", return_value=(name, True)):
            self.w._create_playlist()
        self.settle()
        return next(p.uid for p in self.library.snapshot.playlists if p.name == name)

    def test_playlist_entries_by_position(self):
        pl = self._new_playlist("Warmup")
        self.assertEqual((self.w._source_kind, self.w._source_value), ("playlist", pl))   # shown once made
        self.assertEqual(self.source("playlist", pl).text(0), "Warmup")
        self.assertEqual(self.column(TITLE), [])
        self.w._add_paths_to_playlist(pl, [self.paths[0], self.paths[1], self.paths[0]])     # the same track twice
        self.settle()
        self.assertEqual(self.column(TITLE), ["あのね", "Hello World", "あのね"])
        self.assertEqual(self.column(INDEX), ["1", "2", "3"])
        # the first entry moved to the end
        self.w._drop_playlist_tracks([0], None)
        self.settle()
        self.assertEqual(self.column(TITLE), ["Hello World", "あのね", "あのね"])
        # the second entry removed: the track's other entry stays
        self.w.table.clearSelection()
        self.w._select_row(1)
        self.w._remove_selected_from_playlist()
        self.settle()
        self.assertEqual(self.column(TITLE), ["Hello World", "あのね"])
        self.assertEqual(self.column(INDEX), ["1", "2"])
        self.assertEqual([uid for _p, uid in self.library.snapshot.entries[pl]], [self.uids[1], self.uids[0]])

    def test_playlist_renamed_moved_and_deleted_when_agreed(self):
        a = self._new_playlist("A")
        b = self._new_playlist("B")
        with mock.patch.object(QInputDialog, "getText", return_value=("Peak", True)):
            self.w._rename_playlist(self.source("playlist", a))
        self.settle()
        self.w._move_playlist(self.source("playlist", b), -1)
        self.settle()
        self.assertEqual([p.name for p in self.library.snapshot.playlists], ["B", "Peak"])
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.w._delete_playlist(self.source("playlist", b))
        self.settle()
        self.assertEqual(len(self.library.snapshot.playlists), 2)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.w._delete_playlist(self.source("playlist", b))
        self.settle()
        self.assertEqual([p.name for p in self.library.snapshot.playlists], ["Peak"])
        self.assertEqual(self.w._source_kind, "library")
        self.assertEqual(len(self.column(TITLE)), 3)

    def test_an_empty_name_makes_no_playlist(self):
        with mock.patch.object(QInputDialog, "getText", return_value=("  ", True)):
            self.w._create_playlist()
        self.settle()
        self.assertEqual(self.library.snapshot.playlists, ())

    # ---- other sources ----
    def test_a_folder_lists_its_audio_files(self):
        open(os.path.join(self.music, "new.wav"), "wb").close()
        open(os.path.join(self.music, "notes.txt"), "wb").close()
        self.w.source_tree.setCurrentItem(self.source("directory", self.music))
        self.assertEqual(self.column(TITLE), ["Hello World", "new", "Third Song", "あのね"])   # by title, any case
        dim = current_theme().text_dim
        row = next(r for r in range(self.w.table.rowCount()) if self.w.table.item(r, TITLE).text() == "new")
        self.assertEqual(self.w._row_uid(row), "")                                      # not analyzed
        self.assertEqual(self.w.table.item(row, TITLE).foreground().color(), QtGui.QColor(dim))

    def test_variants_show_each_song_and_the_loaded_tracks(self):
        con = variant_store.open_writer(variant_store.path_for(self.lib))
        song = features(1)
        for uid, f in ((self.uids[0], song), (self.uids[1], song), (self.uids[2], features(2))):
            variant_store.put(con, uid, fpm.track_fingerprints(*f))
        con.close()
        self.library.refresh()
        self.settle()
        self.w.source_tree.setCurrentItem(self.source("variants"))
        self.assertEqual(sorted(self.column(TITLE)), ["Hello World", "あのね"])
        self.assertEqual(self.column(INDEX), ["1", "1"])
        tip = self.w.table.item(self.row_of(self.uids[0]), TITLE).toolTip()
        self.assertIn("Hello World", tip)
        self.assertIn("the same mix", tip)
        # Loaded Track: the deck's track's song only, green
        self.w.source_tree.setCurrentItem(self.source("variants_loaded"))
        self.assertEqual(self.column(TITLE), [])
        self.model.load(StoredAnalysis.from_arrays({"duration_sec": 200.0}), {"uid": self.uids[1]}, 200.0)
        self.assertEqual(sorted(self.column(TITLE)), ["Hello World", "あのね"])
        green = QtGui.QColor(current_theme().success)
        self.assertEqual(self.w.table.item(self.row_of(self.uids[1]), TITLE).foreground().color(), green)
        self.model.load(StoredAnalysis.from_arrays({"duration_sec": 240.0}), {"uid": self.uids[2]}, 240.0)
        self.assertEqual(self.column(TITLE), [])

    def test_search_by_music_results_ranked_and_loaded_at_their_place(self):
        self.search("hello")
        self.w._enter_music_mode()
        self.assertTrue(self.w.e_search.isHidden())
        self.w._on_music_results(SimpleNamespace(matches=[
            MusicMatch(self.uids[2], "melody", 0.2, 30.0, 40.0, 0.0),
            MusicMatch(self.uids[0], "recording", 12, 10.0, 20.0, 0.0),
            MusicMatch(str(uuid4()), "melody", 0.1, 0.0, 5.0, 0.0)]))       # not in the library: left out
        self.assertEqual(self.column(TITLE), ["Third Song", "あのね"])
        self.assertEqual(self.column(INDEX), ["1", "2"])
        self.assertIn("0:10 - 0:20", self.w.table.item(1, TITLE).toolTip())
        loads = []
        self.bus.sig_request_load_track.connect(loads.append)
        self.w._on_activate(self.w.table.model().index(0, 0))
        self.assertEqual(loads, [self.paths[2]])
        self.assertEqual(self.w._deck_seek._pending[:2], (self.paths[2], 29.0))   # seeked once its audio is ready
        self.w._exit_music_mode()
        self.assertEqual(self.w.e_search.text(), "hello")                         # the search as it was
        self.assertEqual(self.column(TITLE), ["Hello World"])

    def test_track_loading_refused_while_external_sync_runs(self):
        loads = []
        self.bus.sig_request_load_track.connect(loads.append)
        self.bus.sig_external_sync_enabled.emit(True)
        with mock.patch.object(QMessageBox, "information") as told:
            self.w._on_activate(self.w.table.model().index(0, 0))
        self.assertTrue(told.called)
        self.assertEqual(loads, [])
        self.bus.sig_external_sync_enabled.emit(False)
        self.w._on_activate(self.w.table.model().index(0, 0))
        self.assertEqual(loads, [self.paths[2]])

    # ---- dragging, dropping, reanalyzing ----
    def test_dragged_rows_carry_their_files_hidden_ones_left_out(self):
        self.search("song", "Title")
        self.w.table.selectAll()
        data = self.w.table.mimeData(self.w.table.selectedItems())
        self.assertEqual([u.toLocalFile() for u in data.urls()], [self.paths[2].replace(os.sep, "/")])
        self.assertFalse(data.hasFormat(_PLAYLIST_TRACKS_MIME))
        pl = self._new_playlist("Set")
        self.w._add_paths_to_playlist(pl, [self.paths[1], self.paths[1]])
        self.settle()
        self.assertEqual(self.column(TITLE), [])               # the search goes on in the playlist
        self.search("")
        self.w.table.selectAll()
        data = self.w.table.mimeData(self.w.table.selectedItems())
        self.assertEqual(json.loads(bytes(data.data(_PLAYLIST_TRACKS_MIME)).decode()), [0, 1])   # by position

    def test_files_dropped_on_the_table_analyzed_not_loaded(self):
        analyzed, loaded = [], []
        self.bus.sig_request_analyze.connect(analyzed.append)
        self.bus.sig_request_load_track.connect(loaded.append)
        new = os.path.join(self.music, "dropped.wav")
        open(new, "wb").close()
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(new), QtCore.QUrl.fromLocalFile(os.path.join(self.music, "gone.wav"))])
        event = QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton,
                                 QtCore.Qt.NoModifier)
        self.w.table.dropEvent(event)
        self.assertEqual([os.path.normcase(p) for p in analyzed], [os.path.normcase(new)])     # only a file there
        self.assertEqual(loaded, [])

    def test_reanalyze_asks_for_this_runs_settings(self):
        asked, analyzed = [], []
        self.bus.sig_reanalyze_requested.connect(lambda path, analysis: asked.append((path, analysis)))
        self.bus.sig_request_analyze.connect(analyzed.append)
        self.w._select_row(self.row_of(self.uids[1]))
        with mock.patch.object(ReanalyzeDialog, "exec", return_value=QDialog.Rejected):
            self.w._reanalyze_selected()
        self.assertEqual(asked, [])

        def accepted(dialog) -> int:
            dialog.sp_bpm_min.setValue(90)
            return QDialog.Accepted

        with mock.patch.object(ReanalyzeDialog, "exec", accepted):
            self.w._reanalyze_selected()
        self.assertEqual(asked, [(self.paths[1], {"bpm_min": 90})])       # only what differs from the saved
        with mock.patch.object(ReanalyzeDialog, "exec", return_value=QDialog.Accepted):
            self.w._reanalyze_selected()
        self.assertEqual(asked[-1], (self.paths[1], None))
        # a file not analyzed yet: analyzed, nothing asked
        open(os.path.join(self.music, "new.wav"), "wb").close()
        self.w.source_tree.setCurrentItem(self.source("directory", self.music))
        self.w.table.clearSelection()
        self.w._select_row(next(r for r in range(self.w.table.rowCount())
                                if self.w.table.item(r, TITLE).text() == "new"))
        self.w._reanalyze_selected()
        self.assertEqual([os.path.basename(p) for p in analyzed], ["new.wav"])
        self.assertEqual(len(asked), 2)

    # ---- the track panel ----
    def test_track_panel_follows_the_selection_and_asks_about_changes(self):
        self.w.resize(1400, 700)
        self.w.show()
        self.w._select_row(self.row_of(self.uids[0]))
        self.w._edit_selected()
        panel = self.w.edit_panel
        self.assertEqual(panel.ed_title.text(), "あのね")
        panel.ed_title.setText("あのね (edit)")
        self.assertTrue(panel.has_changes())

        def select(uid: str) -> None:
            self.w.table.clearSelection()
            self.w._select_row(self.row_of(uid))
            self.w._panel_follow.stop()
            self.w._show_selection_in_panel()

        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            select(self.uids[1])                       # dropped
        self.assertEqual(panel.ed_title.text(), "Hello World")
        self.settle()
        self.assertEqual(self.library.snapshot.track(self.uids[0]).title, "あのね")
        panel.ed_title.setText("Hello Edited")
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            select(self.uids[2])                       # applied
        self.assertEqual(panel.ed_title.text(), "Third Song")
        self.apply_update()
        self.assertEqual(self.library.snapshot.track(self.uids[1]).title, "Hello Edited")
        self.assertIn("Hello Edited", self.column(TITLE))
        # clicking the open tab folds the panel
        self.w._on_panel_tab_clicked(self.w.EDIT_TAB)
        self.assertFalse(self.w.panel_stack.isVisible())


if __name__ == "__main__":
    unittest.main()
