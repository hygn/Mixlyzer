"""The mix candidates (analyzer.search.mixability, core.search.mixability) and their index (core.library.mixability),
on synthetic tracks: the bars, where a blend can go by the phrases, the criteria AND / OR, the index's rows."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyzer.audio.wave_overview import WaveOverview
from analyzer.search import mixability as mx
from core.library import mixability as store
from core.search import mixability as ms

BPM = 120.0
BAR = 4 * 60.0 / BPM


def track(seed: int, bars: int = 64, bpm: float = BPM):
    """A synthetic track's beats, tempo rows, overview, length and beats' features (``seed``)."""
    rng = np.random.default_rng(seed)
    beats = np.arange(bars * 4 + 1) * 60.0 / bpm
    duration = float(beats[-1])
    rows = np.array([[0.0, duration, bpm, 0.0, 4.0, bpm]])
    wave = WaveOverview(np.zeros((0, 2), np.int8), np.ones(1, np.float32),
                        rng.integers(40, 250, size=(4096, 12)).astype(np.uint8), np.ones(12, np.float32),
                        np.zeros(3, np.float32))
    n = beats.size - 1
    features = mx.BeatFeatures(rng.random((n, 12)).astype(np.float32),
                               rng.random((n, 3, mx.POSITIONS)).astype(np.float32))
    return beats, rows, wave, duration, features


def units(labels_bars: list[tuple[str, int]], bar: float = BAR) -> list[mx.Unit]:
    """Phrases of so many bars each, from 0."""
    out, t = [], 0.0
    for label, n in labels_bars:
        out.append(mx.Unit(t, t + n * bar, label))
        t += n * bar
    return out


SONG = [("INTRO", 8), ("VERSE", 16), ("CHORUS", 16), ("INTERLUDE", 8), ("VERSE", 8), ("OUTRO", 8)]


class BarsTests(unittest.TestCase):
    def test_a_bar_a_downbeat_to_the_next(self):
        beats, rows, wave, duration, features = track(1)
        bars = mx.bars(beats, rows, wave, duration, features)
        self.assertEqual(bars.size, 64)
        self.assertEqual(bars.rows.shape, (64, mx.COLUMNS))
        self.assertAlmostEqual(bars.bpm(3), BPM, places=3)
        self.assertTrue(np.all(bars.beats == 4))

    def test_beat_features_pool_the_frames_of_each_beat(self):
        frame_times = np.arange(0, 10.0, 0.01)
        chroma = np.zeros((12, frame_times.size))
        chroma[3] = 1.0
        bands = np.zeros((3, frame_times.size))
        bands[0, 50] = 5.0        # one kick at 0.5 s: beat 1 (0.5 - 1.0 s), its first twelfth
        f = mx.beat_features(np.arange(0, 10.0, 0.5), frame_times, chroma, bands)
        self.assertEqual(f.chroma.shape, (19, 12))
        self.assertTrue(np.allclose(f.chroma[:, 3], 1.0))
        self.assertEqual(f.onsets[1, 0, 0], 5.0)
        self.assertEqual(float(f.onsets[1, 0, 1:].sum() + f.onsets[0].sum()), 0.0)


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.A = mx.bars(*track(1))
        self.B = mx.bars(*track(2))

    def test_a_leaves_at_a_run_end_and_b_comes_in_from_its_top(self):
        found = mx.best_overlap(self.A, self.B, units(SONG), units(SONG))
        self.assertIsNotNone(found)
        self.assertEqual(found.b_bar, 0)                     # B's first phrase
        # A's overlap ends where its INTERLUDE (bars 40-48) or OUTRO (56-64) ends, and starts on a phrase head
        self.assertIn(found.a_bar + found.bars, (48, 64))
        self.assertIn(found.a_bar, (40, 56))
        self.assertEqual(found.bars, 8)                      # B's intro, as long as the exit run

    def test_no_chorus_no_blend(self):
        flat = [("VERSE", 64)]
        self.assertIsNone(mx.best_overlap(self.A, self.B, units(flat), units(SONG)))

    def test_an_uneven_grid_is_no_place_to_blend(self):
        beats, rows, wave, duration, features = track(2)
        beats = beats.copy()
        beats[1:33] += 0.1          # B's intro bars stretched apart from the rest
        B = mx.bars(beats, rows, wave, duration, features)
        found = mx.best_overlap(self.A, B, units(SONG), units(SONG))
        self.assertTrue(found is None or found.b_bar != 0)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tracks, self.bars = [], {}
        for k, (bpm, key) in enumerate(((120.0, 0), (122.0, 7), (150.0, 0), (60.5, 9), (121.0, 6))):
            uid = f"t{k}"
            beats, rows, wave, duration, features = track(k + 1, bpm=bpm)
            self.bars[uid] = mx.bars(beats, rows, wave, duration, features)
            bar = 4 * 60.0 / bpm
            self.tracks.append(ms.TrackMarks(uid, f"Song {k}", tuple(units(SONG, bar)),
                                             ((0.0, duration, bpm, bpm),), ((0.0, duration, key),)))
        self.a = self.tracks[0]

    def uids(self, **options) -> list[str]:
        return [c.uid for c in ms.candidates(self.a, self.tracks, self.bars, set(), ms.MixOptions(**options))]

    def test_bpm_alone_takes_half_and_double_tempo(self):
        got = self.uids(bpm=True, key=False, audio=False)
        self.assertEqual(set(got), {"t1", "t3", "t4"})       # 122, 60.5 (x2 = 121), 121; not 150

    def test_and_or(self):
        both = set(self.uids(bpm=True, key=True, audio=False, match_all=True))
        either = set(self.uids(bpm=True, key=True, audio=False, match_all=False))
        self.assertEqual(both, {"t1"})                        # 122 at G (1B, +-1 of C's 8B)
        self.assertEqual(either, {"t1", "t2", "t3", "t4"})    # 150 at C: its key; F# (2B) at 121: its BPM

    def test_nothing_on_lists_every_other_track(self):
        self.assertEqual(set(self.uids(bpm=False, key=False, audio=False)), {"t1", "t2", "t3", "t4"})

    def test_the_loaded_track_its_versions_and_instrumentals_left_out(self):
        tracks = self.tracks + [ms.TrackMarks("inst", "Song 0 (Off Vocal)", self.tracks[1].phrases,
                                              self.tracks[1].tempo, self.tracks[1].keys)]
        got = [c.uid for c in ms.candidates(self.a, tracks, self.bars, {"t1"},
                                            ms.MixOptions(bpm=False, key=False, audio=False))]
        self.assertNotIn("t0", got)
        self.assertNotIn("t1", got)
        self.assertNotIn("inst", got)

    def test_the_blends_found_come_first(self):
        found = ms.candidates(self.a, self.tracks, self.bars, set(), ms.MixOptions(bpm=False, key=False,
                                                                                    audio=False))
        scored = [c.score is not None for c in found]
        self.assertEqual(scored, sorted(scored, reverse=True))
        # among them, the blend's score with the tempo's distance
        ranks = [c.score + abs(c.tempo_pct) / ms.TEMPO_RANK_UNIT for c in found if c.score is not None]
        self.assertEqual(ranks, sorted(ranks))

    def test_previous_blends_the_candidate_out_into_the_loaded_track(self):
        # the loaded track ends on its chorus (nowhere to leave it), but its intro takes another track's outro
        ends_on_chorus = [("INTRO", 8), ("VERSE", 24), ("CHORUS", 32)]
        loaded = ms.TrackMarks("t0", "Song 0", tuple(units(ends_on_chorus)), self.a.tempo, self.a.keys)
        none = ms.MixOptions(bpm=False, key=False, audio=False)
        after = ms.candidates(loaded, self.tracks, self.bars, set(), none)
        before = ms.candidates(loaded, self.tracks, self.bars, set(),
                               ms.MixOptions(bpm=False, key=False, audio=False, previous=True))
        self.assertTrue(all(c.bars == 0 for c in after))
        blended = [c for c in before if c.bars]
        self.assertTrue(blended)
        self.assertTrue(all(c.b_from == 0.0 for c in blended))      # into the loaded track's top

    def test_key_relations(self):
        self.assertEqual(ms.key_relation(0, 0), "same")
        self.assertEqual(ms.key_relation(0, 21), "relative")      # C, Am
        self.assertEqual(ms.key_relation(0, 7), "adjacent")       # C, G
        self.assertEqual(ms.key_relation(0, 5), "adjacent")       # C, F
        self.assertEqual(ms.key_relation(0, 2), "other")          # C, D
        self.assertEqual(ms.key_relation(None, 2), "")


class IndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / store.FILE_NAME)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_row_kept_and_its_bars_made_again_from_its_features(self):
        beats, rows, wave, duration, features = track(1)
        bars = mx.bars(beats, rows, wave, duration, features)
        con = store.open_writer(self.path)
        try:
            with con:
                store.put(con, "u1", beats, features, bars)
            times, kept = store.features_of(con, "u1")
        finally:
            con.close()
        self.assertTrue(store.same_beats(times, beats))
        self.assertTrue(np.allclose(kept.chroma, features.chroma, atol=1e-3))
        loaded = store.all_bars(self.path)["u1"]
        self.assertTrue(np.array_equal(loaded.start, bars.start))
        self.assertTrue(np.allclose(loaded.rows, bars.rows, atol=1e-2, rtol=1e-3))
        self.assertEqual(set(store.feature_beats(self.path)), {"u1"})
        # the downbeats moved a beat: the bars made again without the audio
        moved = rows.copy()
        moved[0, 3] = 60.0 / BPM
        con = store.open_writer(self.path)
        try:
            with con:
                store.put_bars(con, "u1", mx.bars(beats, moved, wave, duration, kept))
                store.drop(con, "u2")
        finally:
            con.close()
        self.assertAlmostEqual(float(store.all_bars(self.path)["u1"].start[0]), 60.0 / BPM, places=5)

    def test_other_beats_need_the_features_read_again(self):
        beats = np.arange(10) * 0.5
        self.assertTrue(store.same_beats(beats, beats + 1e-9))
        self.assertFalse(store.same_beats(beats, beats + 0.01))
        self.assertFalse(store.same_beats(beats, beats[:-1]))


if __name__ == "__main__":
    unittest.main()
