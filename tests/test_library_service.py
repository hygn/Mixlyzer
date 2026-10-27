"""The library's owner (core.library.service): its jobs in order on its thread, the snapshot read after writes, other
threads' calls, failures rolled back, the snapshot's file lookup, another library, callbacks of deleted widgets."""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from core.workflows.event_bus import EventBus
from core.library import jobs
from core.library.records import TrackRow
from core.library.service import LibraryService

APP = QApplication.instance() or QApplication([])


def _row(path: str, title: str = "曲") -> TrackRow:
    return TrackRow(uid=str(uuid4()), path=path, title=title, artist="歌手", added_ts=1)


def _image():
    from PySide6 import QtGui
    image = QtGui.QImage(8, 8, QtGui.QImage.Format_RGB32)
    image.fill(0x336699)
    return image


class LibraryServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.lib = self._tmp.name
        self.bus = EventBus()
        self.updates = []
        self.bus.sig_lib_updated.connect(self.updates.append)
        self.library = LibraryService(self.lib, bus=self.bus)

    def tearDown(self):
        self.library.close()
        self._tmp.cleanup()

    def settle(self):
        for _ in range(3):
            self.library.wait()

    def test_jobs_in_order_and_the_snapshot_after_writes(self):
        row = _row("a.wav")
        seen = []
        self.library.write(jobs.save_track, row, False,
                           done=lambda _r: seen.append(("written", len(self.library.snapshot.tracks))))
        self.library.read(jobs.track, row.uid, done=lambda r: seen.append(("read", r.title)))
        self.settle()
        # the write's callback saw its snapshot; the read after it saw the write
        self.assertEqual(seen, [("written", 1), ("read", "曲")])
        self.assertEqual(self.library.snapshot.track(row.uid).title, "曲")
        self.assertTrue(self.library.snapshot.readings.get(row.uid))      # read with the snapshot
        self.assertEqual(len(self.updates[-1]), 1)                         # sig_lib_updated with the rows
        # a playlist and its entry in the snapshot
        self.library.write(jobs.create_playlist, "P", done=lambda p: self.library.write(
            jobs.add_to_playlists, {p.uid: [row.uid]}))
        self.settle()
        playlist = self.library.snapshot.playlists[0]
        self.assertEqual([r.uid for _p, r in self.library.snapshot.playlist_tracks(playlist.uid)], [row.uid])

    def test_only_the_parts_a_write_changes_read_again(self):
        row = _row("e.wav")
        self.library.write(jobs.save_track, row, False)
        self.settle()
        before, sent = self.library.snapshot, len(self.updates)
        done = []
        # a write that changes none of the snapshot: not read again, nothing sent, its done called
        self.library.write(jobs.set_tag_sources, row.uid, {"title": ["id3:TIT2"]}, done=done.append)
        self.settle()
        self.assertIs(self.library.snapshot, before)
        self.assertEqual((len(self.updates), done), (sent, [None]))
        # a playlist's: the playlists read again, the tracks and readings kept as they were read
        self.library.write(jobs.create_playlist, "P")
        self.settle()
        after = self.library.snapshot
        self.assertEqual([p.name for p in after.playlists], ["P"])
        self.assertIs(after.tracks, before.tracks)
        self.assertIs(after.readings, before.readings)
        self.assertEqual(len(self.updates), sent + 1)
        # a job without the mark: everything read again
        self.library.write(lambda lib: lib.db.upsert(_row("f.wav")))
        self.settle()
        self.assertEqual(len(self.library.snapshot.tracks), 2)

    def test_the_editors_save_through_the_owner(self):
        import numpy as np
        from core.analysis_library import jobs as analysis_jobs
        row = _row("g.wav")
        self.library.write(jobs.save_track, row, False)
        from analyzer.music.marks import JumpCue, JumpCueGroup
        from core.analysis_library.stored import TrackEdits
        self.library.write(lambda lib: lib.features.save(row.uid, {
            "beats_time_sec": np.arange(8) * 0.5, "tempo_segments": np.asarray([[0, 4, 120, 0, 4, 120]], float),
            "jump_cues_np": {"cue_id": np.asarray([1, 2]), "cue_point": np.asarray([1.0, 2.0]),
                             "stale": np.asarray([7])}}))
        self.settle()
        saved, errors = [], []
        tempo = np.asarray([[0, 4, 128, 0, 4, 128]], float)
        group = JumpCueGroup((JumpCue("A", "A", 1.5, 1.5, 1.5), JumpCue("B", "B", 3.0, 3.0, 3.0)), 2)
        keys = np.asarray([[0, 0, 0, 1], [9, 1, 1, 4]], float)   # C for 2 beats, then A minor for 6
        edits = TrackEdits(beats=np.arange(8) * 0.5, tempo_segments=tempo, key_segments=keys,
                           phrases=(), jump_cues=(group,), cue_points=())
        self.library.write(jobs.save_edits, row.uid, edits, done=saved.append, failed=errors.append)
        with self.assertRaises(ValueError):                       # a letter not a slot's: refused before the save
            TrackEdits(beats=edits.beats, tempo_segments=tempo, key_segments=edits.key_segments, phrases=(),
                       jump_cues=(JumpCueGroup((JumpCue("AA", "AA", 1.0, 1.0, 1.0),), 0),), cue_points=())
        self.settle()
        self.assertEqual((saved, errors), ([row.uid], []))
        got = {}
        self.library.read(analysis_jobs.analysis, row.uid, done=lambda a: got.update(analysis=a))
        self.library.read(lambda lib: lib.features.load(row.uid), done=lambda a: got.update(arrays=a))
        self.library.read(lambda lib: lib.db.conn.execute(
            "SELECT bpm FROM track_bpm_segments WHERE track_uid=?", (row.uid,)).fetchall(),
            done=lambda rows: got.update(bpm=[r[0] for r in rows]))
        self.settle()
        self.assertEqual(got["analysis"].tempo_segments.tolist(), tempo.tolist())
        self.assertEqual(list(got["analysis"].jump_cues), [group])
        self.assertNotIn("jump_cues_np.stale", got["arrays"])     # the block replaced whole
        self.assertEqual(got["bpm"], [128.0])                     # its segment table written from it
        # the track's BPM and key (what the Export tab writes to the file's tags) made from the marks as edited
        track = self.library.snapshot.track(row.uid)
        self.assertEqual((track.bpm, track.key), (128.0, 21))
        self.assertEqual(float(got["arrays"]["tempo_global"]), 128.0)

    def test_library_db_synced_with_the_stored_analyses(self):
        """Sync Now (core.workflows.library_sync, jobs.sync_index): each track's BPM, key and segment tables written
        from its stored marks, its metadata untouched; a track without a stored analysis counted; the editor's save
        without the setting writes the stored analysis alone."""
        import numpy as np
        from unittest import mock
        from core.analysis_library.stored import TrackEdits
        from core.workflows.library_sync import LibraryIndexSync
        stale, bare = _row("s.wav", title="手で直した"), _row("t.wav")
        stale.bpm, stale.key, stale.rating = 99.0, 3, 5
        for row in (stale, bare):
            self.library.write(jobs.save_track, row, False)
        phrases = {"start": np.asarray([0.0, 3.0]), "end": np.asarray([3.0, 4.0]),
                   "label": np.asarray(["VERSE", "OUTRO"])}
        self.library.write(lambda lib: lib.features.save(stale.uid, {
            "duration_sec": 4.0, "tempo_segments": np.asarray([[0, 4, 128, 0, 4, 128]], float),
            "key_segments": np.asarray([[0, 0, 0, 3], [9, 1, 3, 4]], float), "phrase_segments_np": phrases}))
        self.settle()
        runs, steps = [], []
        sync = LibraryIndexSync(self.library)
        sync.finished.connect(runs.append)
        sync.progress.connect(lambda done, total: steps.append((done, total)))
        with mock.patch.object(jobs, "SYNC_BATCH", 1):
            sync.sync_requested()
            self.settle()
        self.assertEqual(runs, [jobs.IndexSync(changed=1, unread=1)])
        self.assertEqual(steps, [(0, 2), (1, 2), (2, 2)])
        track = self.library.snapshot.track(stale.uid)
        self.assertEqual((track.bpm, track.key, track.title, track.rating), (128.0, 21, "手で直した", 5))  # the OUTRO's
        got = []
        self.library.read(lambda lib: lib.db.conn.execute(
            "SELECT COUNT(*) FROM track_phrase_sections WHERE track_uid=?", (stale.uid,)).fetchone()[0],
            done=got.append)
        self.settle()
        self.assertEqual(got, [2])
        sync.sync_requested()                   # again: nothing to change
        self.settle()
        self.assertEqual(runs[-1], jobs.IndexSync(changed=0, unread=1))
        # the editor's save with the setting off: the stored analysis alone
        edits = TrackEdits(beats=np.arange(8) * 0.5, tempo_segments=np.asarray([[0, 4, 140, 0, 4, 140]], float),
                           key_segments=np.asarray([[2, 0, 0, 4]], float), phrases=(), jump_cues=(), cue_points=())
        self.library.write(jobs.save_edits, stale.uid, edits, False)
        self.settle()
        self.assertEqual(self.library.snapshot.track(stale.uid).bpm, 128.0)
        sync.sync_requested()
        self.settle()
        self.assertEqual(runs[-1], jobs.IndexSync(changed=1, unread=1))
        self.assertEqual((self.library.snapshot.track(stale.uid).bpm, self.library.snapshot.track(stale.uid).key),
                         (140.0, 2))

    def test_an_editors_save_that_fails_leaves_library_db_as_it_was(self):
        """One transaction: a failure of its last library.db write or of the NPZ saved after it rolls back the BPM, key
        and segment tables written before it, and the NPZ is left as it was."""
        import numpy as np
        from unittest import mock
        from core.analysis_library.store import FeatureNPZStore
        from core.analysis_library.stored import TrackEdits
        from core.library.db import LibraryDB
        row = _row("i.wav")
        row.bpm, row.key = 120.0, 0
        self.library.write(jobs.save_track, row, False)
        self.library.write(lambda lib: lib.features.save(row.uid, {
            "beats_time_sec": np.arange(8) * 0.5, "tempo_segments": np.asarray([[0, 4, 120, 0, 4, 120]], float)}))
        self.settle()
        edits = TrackEdits(beats=np.arange(8) * 0.5, tempo_segments=np.asarray([[0, 4, 128, 0, 4, 128]], float),
                           key_segments=np.asarray([[9, 1, 0, 4]], float), phrases=(), jump_cues=(), cue_points=())
        for target, name in ((LibraryDB, "replace_phrase_index"), (FeatureNPZStore, "save")):
            with self.subTest(failing=name):
                errors, got = [], {}
                with mock.patch.object(target, name, side_effect=OSError("disk full")):
                    self.library.write(jobs.save_edits, row.uid, edits, failed=errors.append)
                    self.settle()
                self.library.read(lambda lib: lib.db.conn.execute(
                    "SELECT COUNT(*) FROM track_bpm_segments WHERE track_uid=?", (row.uid,)).fetchone()[0],
                    done=lambda n: got.update(segments=n))
                self.library.read(lambda lib: lib.features.load(row.uid), done=lambda a: got.update(arrays=a))
                self.settle()
                self.assertEqual(len(errors), 1)
                track = self.library.snapshot.track(row.uid)
                self.assertEqual((track.bpm, track.key), (120.0, 0))
                self.assertEqual(got["segments"], 0)
                self.assertEqual(got["arrays"]["tempo_segments"][0][2], 120)

    def test_an_analysis_of_a_track_in_the_library_keeps_the_users_metadata(self):
        """A library track analyzed again (its stored analysis did not read, or a reanalysis): the metadata the user
        edited stays over the file's tags; its uid and added time too."""
        import numpy as np
        row = _row("h.wav", title="直した曲名")
        row.rating, row.comment, row.album = 4, "メモ", "盤"
        self.library.write(jobs.save_track, row, False)
        self.settle()
        tags = {"path": "h.wav", "title": "tag title", "artist": "tag artist", "album": "tag album", "rating": 0,
                "comment": "tag comment", "added_ts": 99, "bpm": 128.0, "key": 21, "duration_sec": 4.0}
        payload = {"properties": tags, "arrays": {"beats_time_sec": np.arange(8) * 0.5},
                   "store": {"bpm_segments": [], "key_segments": [], "phrase_sections": [], "music_entry": None,
                             "variants": None, "cover": None}}
        stored = []
        self.library.write(jobs.store_analysis, payload, done=stored.append)
        self.settle()
        track = self.library.snapshot.track(row.uid)
        self.assertEqual((track.title, track.artist, track.album, track.comment, track.rating),
                         ("直した曲名", "歌手", "盤", "メモ", 4))
        self.assertEqual((track.bpm, track.added_ts), (128.0, 1))     # the analysis's values, the row's added time
        self.assertEqual((stored[0]["properties"]["uid"], stored[0]["new_track"]), (row.uid, False))
        # a new file takes its tags
        self.library.write(jobs.store_analysis, dict(payload, properties=dict(tags, path="i.wav")), done=stored.append)
        self.settle()
        self.assertEqual(self.library.snapshot.track(stored[1]["properties"]["uid"]).title, "tag title")

    def test_the_start_drops_what_tracks_gone_left_behind(self):
        """check_stores: every store beside library.db made to agree with its tracks; a stored analysis set aside, not
        deleted; a library.db without tracks touches nothing."""
        import numpy as np
        from core.library import music_index, thumbnails, variants
        kept, gone = _row("k.wav"), str(uuid4())
        self.library.write(jobs.save_track, kept, False)
        self.library.write(jobs.create_playlist, "P", done=lambda p: self.library.write(
            jobs.add_to_playlists, {p.uid: [kept.uid]}))
        self.settle()
        playlist = self.library.snapshot.playlists[0].uid

        def leftovers(lib):
            for uid in (kept.uid, gone):
                lib.features.save(uid, {"beats_time_sec": np.arange(4) * 0.5})
                thumbnails.save_thumbnail(lib.path, uid, _image())
            lib.db.conn.execute("PRAGMA foreign_keys=OFF;")    # outside a transaction: inside one it does nothing
            with lib.db.tx():       # rows of a track removed before every table followed it
                lib.db.conn.execute("INSERT INTO track_tag_sources(track_uid, field, slot, source) VALUES(?, 'title', 0, '');", (gone,))
                lib.db.conn.execute("INSERT INTO playlist_tracks(playlist_uid, position, track_uid) VALUES(?, 1, ?);", (playlist, gone))
            lib.db.conn.execute("PRAGMA foreign_keys=ON;")
            con = music_index.open_writer(music_index.path_for(lib.path))
            with con:
                con.execute("INSERT INTO landmark_song(uid) VALUES(?)", (gone,))
            con.close()
            con = variants.open_writer(variants.path_for(lib.path))
            with con:
                con.execute("INSERT INTO variant_song(uid, method, frames) VALUES(?, 1, 0)", (gone,))
                con.execute("INSERT INTO variant_pair(a, b, method, kind) VALUES(?, ?, 1, 'same')", (kept.uid, gone))
            con.close()
        self.library.write(leftovers)
        found = []
        self.library.write(jobs.check_stores, done=found.append)
        self.settle()
        self.assertEqual(found, [jobs.StoreCheck(rows=2, music=1, variants=1, thumbnails=1, analyses=1)])
        lib = self.lib
        self.assertEqual(thumbnails.thumbnail_uids(lib), {kept.uid})
        self.assertEqual(music_index.stored_uids(music_index.path_for(lib)), set())
        self.assertEqual(variants.pairs(variants.path_for(lib)), [])
        self.assertTrue(os.path.isfile(os.path.join(lib, f"{kept.uid}.npz")))
        self.assertTrue(os.path.isfile(os.path.join(lib, jobs.ORPHANED_ANALYSES, f"{gone}.npz")))   # kept aside
        self.assertEqual([r.uid for _p, r in self.library.snapshot.playlist_tracks(playlist)], [kept.uid])
        # again: nothing left over
        self.library.write(jobs.check_stores, done=found.append)
        self.settle()
        self.assertEqual(found[-1].total(), 0)
        # a library.db without tracks (lost, not emptied) beside stores that have some: nothing touched
        with tempfile.TemporaryDirectory() as other:
            self.library.set_libpath(other)
            self.library.write(lambda lib: lib.features.save(gone, {"beats_time_sec": np.arange(4) * 0.5}))
            self.library.write(jobs.check_stores, done=found.append)
            self.settle()
            self.assertTrue(found[-1].skipped)
            self.assertTrue(os.path.isfile(os.path.join(other, f"{gone}.npz")))
            self.library.set_libpath(self.lib)
            self.settle()

    def test_another_thread_calls_and_waits(self):
        row = _row("b.wav")
        got = {}

        def worker():
            self.library.call(jobs.save_track, row, False, write=True)
            got["title"] = self.library.call(jobs.track, row.uid).title
            try:
                self.library.call(lambda lib: lib.db.conn.execute("SELECT * FROM no_such_table"))
            except Exception as exc:        # the job's exception, raised in the caller
                got["error"] = type(exc).__name__
            try:                            # a job waiting for a job would wait for itself
                self.library.call(lambda lib: self.library.call(jobs.track, row.uid))
            except RuntimeError:
                got["nested"] = "refused"

        t = threading.Thread(target=worker)
        t.start()
        t.join(10)
        self.settle()
        self.assertEqual(got, {"title": "曲", "error": "OperationalError", "nested": "refused"})
        self.assertIsNotNone(self.library.snapshot.track(row.uid))       # the snapshot read after its write
        with self.assertRaises(RuntimeError):      # the GUI thread never waits
            self.library.call(lambda lib: None)

    def test_a_failed_write_rolled_back(self):
        row = _row("c.wav")
        errors = []

        def write(lib):
            lib.db.conn.execute("INSERT INTO playlists(uid, name, sort_index) VALUES ('x', 'kept?', 0)")
            raise ValueError("refused")

        self.library.write(jobs.save_track, row, False)
        self.library.write(write, failed=errors.append)
        self.settle()
        self.assertEqual([str(e) for e in errors], ["refused"])
        self.assertEqual(self.library.snapshot.playlists, ())                # its own changes undone
        self.assertIsNotNone(self.library.snapshot.track(row.uid))          # the earlier write kept

    def test_the_snapshot_finds_a_file_however_spelled(self):
        path = Path(self.lib) / "Song.wav"
        path.write_bytes(b"RIFF")
        row = _row(str(path))
        row.file_size = 4          # as the analysis stores it: the lookup compares sizes first
        self.library.write(lambda lib: lib.db.upsert(row))
        self.settle()
        snap = self.library.snapshot
        self.assertEqual(snap.get(str(path)).uid, row.uid)
        other = str(path.parent) + os.sep + "." + os.sep + path.name     # another spelling of the same file
        self.assertIsNone(snap.get(other))
        self.assertEqual(snap.find_file(other).uid, row.uid)
        self.assertIsNone(snap.find_file(str(Path(self.lib) / "none.wav")))

    def test_another_library(self):
        self.library.write(jobs.save_track, _row("d.wav"), False)
        self.settle()
        with tempfile.TemporaryDirectory() as other:
            self.library.set_libpath(other)
            self.settle()
            self.assertEqual(self.library.snapshot.tracks, ())
            self.library.set_libpath(self.lib)
            self.settle()
        self.assertEqual(len(self.library.snapshot.tracks), 1)

    def test_a_deleted_widgets_callback_not_called(self):
        called = []

        class Widget(QtCore.QObject):
            def came(self, result):
                called.append(result)

        w = Widget()
        self.library.read(lambda lib: 1, done=w.came)
        import shiboken6
        shiboken6.delete(w)
        self.settle()
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
