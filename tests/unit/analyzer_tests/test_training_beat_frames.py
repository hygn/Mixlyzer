"""analyzer.training.beat_frames: the beat models' frame cache read back as written (float16 features, exact times and
contours), stale or broken caches taken as misses, the tracks with tempo rows, and ensure_frames: no process pool when
every cache is current, the stale ones built, a failed build leaving its track out with the reason."""
import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.beat.frame_features import FrameFeatures
from analyzer.training import beat_frames as BF
from analyzer.training.library_tracks import Annotation
from analyzer.training.text import render

from unit.analyzer_tests import training_fixtures as F


class CacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.track = F.given_track(self.dir, 0, None)
        beats = F.steady_beats(4.0, 120.0)
        self.frames = F.synthetic_frames(300, beats, 1)
        self.path = F.write_beat_frames(self.dir, self.track, self.frames)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_back_as_written(self):
        got = BF.read_frames(self.path, self.track)
        for field in dataclasses.fields(FrameFeatures):
            a, b = getattr(got, field.name), getattr(self.frames, field.name)
            with self.subTest(field=field.name):
                if field.name in BF._SCALARS:
                    self.assertEqual(a, b)
                elif field.name in BF._EXACT:
                    np.testing.assert_array_equal(a, b)
                else:
                    self.assertEqual(a.dtype, np.float32)
                    np.testing.assert_allclose(a, np.asarray(b, np.float16).astype(np.float32), equal_nan=True)

    def test_stale_or_broken_caches_are_misses(self):
        os.utime(self.track["audio_path"], ns=(5, 5))                     # the audio file changed
        self.assertIsNone(BF.read_frames(self.path, self.track))
        self.path = F.write_beat_frames(self.dir, self.track, self.frames)
        self.assertIsNotNone(BF.read_frames(self.path, self.track))
        with np.load(self.path) as z:
            values = {k: z[k] for k in z.files}
        np.savez(self.path, **{**values, "cache_format": np.asarray("mixlyzer_beat_frame_cache_v0")})
        self.assertIsNone(BF.read_frames(self.path, self.track))
        np.savez(self.path, **{k: v for k, v in values.items() if k != "tonnetz"})
        self.assertIsNone(BF.read_frames(self.path, self.track))
        self.path.write_bytes(b"\x00" * 64)
        self.assertIsNone(BF.read_frames(self.path, self.track))
        self.assertIsNone(BF.read_frames(self.dir / "missing.npz", self.track))


class GridTrackTests(unittest.TestCase):
    def test_tracks_need_tempo_rows(self):
        good = F.beat_annotation()
        cases = {"good": good, "none": Annotation(**{**good.__dict__, "tempo_segments": None}),
                 "empty": Annotation(**{**good.__dict__, "tempo_segments": np.zeros((0, 6))}),
                 "narrow": Annotation(**{**good.__dict__, "tempo_segments": np.zeros((1, 3))}),
                 "text": Annotation(**{**good.__dict__, "tempo_segments": [["a", "b", "c", "d"]]})}
        with tempfile.TemporaryDirectory() as tmp:
            F.use_tracks([F.given_track(Path(tmp), i, a, title=name) for i, (name, a) in enumerate(cases.items())])
            try:
                ignored = []
                listed = BF.list_grid_tracks(Path(tmp), ignored)
            finally:
                F.library_tracks._tracks = None
        self.assertEqual([t["title"] for t in listed], ["good"])
        np.testing.assert_array_equal(listed[0]["segments"], good.tempo_segments)
        reasons = {s.title: render(s.reason) for s in ignored}
        self.assertEqual({k: v for k, v in reasons.items() if k != "text"},
                         {"none": "no tempo rows", "empty": "no tempo rows", "narrow": "no tempo rows"})
        self.assertTrue(reasons["text"].startswith("invalid tempo rows: "))


class EnsureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.tracks = [F.given_track(self.dir, i, None) for i in range(3)]
        self.frames = F.synthetic_frames(100, F.steady_beats(1.0, 120.0), 0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_pool_when_every_cache_is_current(self):
        for t in self.tracks:
            F.write_beat_frames(self.cache, t, self.frames)
        with mock.patch.object(BF, "spawn_pool", F.no_pool):
            self.assertEqual(BF.ensure_frames(self.tracks, self.cache, []), self.tracks)

    def test_stale_caches_built_and_failures_left_out(self):
        F.write_beat_frames(self.cache, self.tracks[0], self.frames)
        built = []

        def build(track, cache_dir_text):
            if track is self.tracks[2]:
                raise RuntimeError("decoder failed")
            built.append(track["uid"])
            F.write_beat_frames(Path(cache_dir_text), track, self.frames)
            return track["uid"]
        ignored, progress = [], []
        with mock.patch.object(BF, "spawn_pool", F.thread_pool), mock.patch.object(BF, "build_frames", build):
            kept = BF.ensure_frames(self.tracks, self.cache, ignored, progress=lambda v, t: progress.append(v),
                                    progress_span=(10, 50))
        self.assertEqual(built, [self.tracks[1]["uid"]])                 # only the stale one built
        self.assertEqual(kept, self.tracks[:2])
        self.assertEqual([render(s.reason) for s in ignored], ["frame feature cache build failed: decoder failed"])
        self.assertEqual(progress, [30, 50])

    def test_rebuild_builds_every_cache(self):
        for t in self.tracks:
            F.write_beat_frames(self.cache, t, self.frames)
        built = []
        with mock.patch.object(BF, "spawn_pool", F.thread_pool), \
                mock.patch.object(BF, "build_frames", lambda t, d: built.append(t["uid"]) or t["uid"]):
            BF.ensure_frames(self.tracks, self.cache, [], rebuild=True)
        self.assertEqual(sorted(built), sorted(t["uid"] for t in self.tracks))


if __name__ == "__main__":
    unittest.main()
