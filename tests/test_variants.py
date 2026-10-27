import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.audio.wave_overview import WaveOverview
from core.library import variants as store
from analyzer.search import variants as fpm


def features(seed: int, n: int = 600, bpm: float = 150.0) -> tuple:
    """A synthetic song's analysis (``seed``): its beats, its overview's 12 bands, its length."""
    rng = np.random.default_rng(seed)
    duration = n * 60.0 / bpm
    columns = 2048
    bands = rng.integers(40, 250, size=(columns, 12)).astype(np.uint8)
    wave = WaveOverview(np.zeros((0, 2), np.int8), np.ones(1, np.float32), bands, np.ones(12, np.float32),
                        np.zeros(3, np.float32))
    return np.arange(n) * 60.0 / bpm, wave, float(duration)


class VariantMethodTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "variants.db")

    def tearDown(self):
        self._tmp.cleanup()

    def _put(self, uid: str, f: tuple) -> None:
        con = store.open_writer(self.path)
        try:
            store.put(con, uid, fpm.track_fingerprints(*f))
        finally:
            con.close()

    def test_the_same_audio_found(self):
        song = features(1)
        self._put("a", song)
        self._put("b", song)
        self._put("c", features(2))
        self.assertEqual([(p.a, p.b, p.kind) for p in store.pairs(self.path)], [("a", "b", fpm.SAME_MIX_KIND)])

    def test_methods_kept_apart_until_pruned(self):
        song = features(1)
        self._put("a", song)
        self._put("b", song)
        with mock.patch.object(store, "METHOD", "fp-next"):
            self.assertEqual(store.indexed_uids(self.path), set())          # none by the next method yet
            self.assertEqual(store.indexed_uids(self.path, any_method=True), {"a", "b"})
            self._put("a", song)                                            # a re-indexed by it
            # the next method's a meets no fingerprint of its own method: the earlier relation still shows
            self.assertEqual(len(store.pairs(self.path)), 1)
            con = store.open_writer(self.path)
            self.assertEqual(store.prune_old_methods(con), 0)                # a's earlier one went with put
            con.close()
            self._put("b", song)
            self.assertEqual(store.indexed_uids(self.path), {"a", "b"})
            con = store.open_writer(self.path)
            store.prune_old_methods(con)
            methods = [m for (m,) in con.execute("SELECT name FROM variant_method")]
            con.close()
            self.assertEqual(methods, ["fp-next"])
            self.assertEqual([(p.a, p.b) for p in store.pairs(self.path)], [("a", "b")])

    def test_an_index_without_methods_made_again(self):
        import sqlite3
        con = sqlite3.connect(self.path)
        con.execute("CREATE TABLE variant_song (id INTEGER PRIMARY KEY, uid TEXT UNIQUE, frames INTEGER)")
        con.execute("INSERT INTO variant_song (uid) VALUES ('old')")
        con.commit()
        con.close()
        self.assertIsNone(store.open_reader(self.path))
        store.open_writer(self.path).close()
        self.assertEqual(store.indexed_uids(self.path, any_method=True), set())

    def test_a_removed_track_leaves_the_indexes(self):
        """Removing tracks from the library drops their fingerprints and relations, and their Search by Music index."""
        from types import SimpleNamespace
        from core.library import jobs, music_index as music_store
        from analyzer.search.music import index as indexing, melody
        f = features(1)
        self._put("a", f)
        self._put("b", f)
        self.assertTrue(store.pairs(self.path))
        mpath = music_store.path_for(self._tmp.name)
        con = music_store.open_writer(mpath)
        entry = indexing.IndexEntry({s: np.zeros((2, 5), np.uint8) for s in melody.FRAMES},
                                    np.arange(3, dtype=np.int64), np.arange(3, dtype=np.int64))
        with con:
            music_store.put(con, "a", 1, 1.0, entry)
            music_store.put(con, "b", 1, 1.0, entry)
        con.close()
        jobs.drop_from_indexes(SimpleNamespace(path=self._tmp.name), ["a"])     # the library's folder alone
        self.assertEqual(store.indexed_uids(self.path, any_method=True), {"b"})
        self.assertEqual(store.pairs(self.path), [])
        self.assertEqual(music_store.indexed_uids(mpath), {"b"})

    def test_the_method_names_the_features_read(self):
        self.assertIn("beats=", fpm.METHOD)
        self.assertIn("bands=", fpm.METHOD)


if __name__ == "__main__":
    unittest.main()
