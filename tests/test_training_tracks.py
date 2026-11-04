"""The training's tracks: the library's owner gives them with what the optimizers read of their stored analyses
(core.analysis_library.jobs.training_tracks), read by the CLI in batches (training.__main__.read_tracks), the run
holds them (training.library_tracks) and the optimizers list theirs from them, never opening the library's files."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import numpy as np

from core.analysis_library import jobs as analysis_jobs
from core.analysis_library.store import FeatureNPZStore
from core.library.db import LibraryDB
from core.library.records import TrackRow
from training import library_tracks
from training.onset_training import MIN_BEATS, _list_beat_tracks


class OptimizerTrackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        library_tracks._tracks = None
        self._tmp.cleanup()

    def test_not_given_is_an_error(self):
        library_tracks._tracks = None
        with self.assertRaises(RuntimeError):
            library_tracks.tracks()

    def test_the_owners_tracks_listed_without_the_database(self):
        db = LibraryDB(str(self.dir / "library.db"))
        db.connect()
        store = FeatureNPZStore(str(self.dir))
        uids = sorted(str(uuid4()) for _ in range(3))
        for i, uid in enumerate(uids):
            # the first: beats and audio; the second: no audio; the third: no analysis
            audio = self.dir / f"{i}.wav"
            if i != 1:
                audio.write_bytes(b"RIFF")
            db.upsert(TrackRow(uid=uid, path=str(audio), title=f"t{i}", added_ts=1))
            if i < 2:
                store.save(uid, {"beats_time_sec": np.arange(MIN_BEATS + 4) * 0.5, "sr": np.int64(22050),
                                 "wave_minmax_np": np.zeros((8, 2), np.int8)})
        db.conn.commit()
        lib = SimpleNamespace(db=db, path=str(self.dir), features=store)
        given = analysis_jobs.training_tracks(lib)
        # as the CLI does: the stored analyses read after, in batches
        for track, (annotation, error) in zip(given, analysis_jobs.training_annotations(lib, [t["uid"] for t in given])):
            track["annotation"] = annotation
            if error:
                track["error"] = error
        db.close()
        self.assertEqual([t["uid"] for t in given], uids)
        a = given[0]["annotation"]                         # what training reads, no more (the JumpCUEs as groups)
        self.assertEqual((a.sample_rate, a.beats.size, a.tempo_segments, a.jump_cue_groups), (22050, MIN_BEATS + 4, None, []))
        self.assertIsNone(given[2]["annotation"])
        for p in self.dir.glob("*.npz"):                   # the process never opens the library's files
            p.unlink()
        (self.dir / "library.db").unlink()
        library_tracks.use(given)
        skipped = []
        listed = _list_beat_tracks(self.dir, skipped)
        self.assertEqual([t["uid"] for t in listed], uids[:1])
        self.assertEqual(sorted(s.uid for s in skipped), sorted(uids[1:]))

    def test_the_cli_reads_them_through_the_owner_in_batches(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from core.library.service import LibraryService
        from training import __main__ as cli
        QApplication.instance() or QApplication([])
        library = LibraryService(str(self.dir))
        try:
            uids = sorted(str(uuid4()) for _ in range(5))
            for i, uid in enumerate(uids):
                library.write(lambda lib, uid=uid, i=i: lib.db.upsert(
                    TrackRow(uid=uid, path=str(self.dir / f"{i}.wav"), title=f"t{i}", added_ts=1)))
                library.write(lambda lib, uid=uid: lib.features.save(uid, {"beats_time_sec": np.arange(4.0)}))
            library.wait()
        finally:
            library.close()
        asked = []
        real = analysis_jobs.training_annotations

        def counting(lib, batch):
            asked.append(len(batch))
            return real(lib, batch)
        with mock.patch.object(analysis_jobs, "TRAINING_BATCH", 2), \
                mock.patch.object(analysis_jobs, "training_annotations", counting), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            tracks = cli.read_tracks(str(self.dir))
        self.assertEqual(asked, [2, 2, 1])                                       # 5 tracks, 2 a job
        self.assertEqual([t["uid"] for t in tracks], uids)
        self.assertTrue(all(t["annotation"].beats.tolist() == [0.0, 1.0, 2.0, 3.0] for t in tracks))

    def test_no_library_no_run(self):
        from training import __main__ as cli
        with self.assertRaises(ValueError):
            cli.read_tracks(str(self.dir / "missing"))
        self.assertFalse((self.dir / "missing").exists())


if __name__ == "__main__":
    unittest.main()
