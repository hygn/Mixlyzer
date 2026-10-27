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
