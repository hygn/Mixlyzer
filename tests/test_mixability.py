"""The mix candidates (core.search.mixability) on synthetic tracks: where two tracks meet by their phrases, the BPM
and key criteria AND / OR, what is left out, the nearest tempo first."""
import unittest

from core.search import mixability as ms

BPM = 120.0
BAR = 4 * 60.0 / BPM


def units(labels_bars: list[tuple[str, int]], bar: float = BAR) -> list[ms.Unit]:
    """Phrases of so many bars each, from 0."""
    out, t = [], 0.0
    for label, n in labels_bars:
        out.append(ms.Unit(t, t + n * bar, label))
        t += n * bar
    return out


SONG = [("LEAD_IN", 1), ("INTRO", 8), ("VERSE", 16), ("CHORUS", 16), ("INTERLUDE", 8), ("VERSE", 8), ("OUTRO", 8)]


class MeetingTests(unittest.TestCase):
    def test_a_leaves_at_its_first_interlude_after_the_chorus_and_b_comes_in_after_its_lead_in(self):
        song = units(SONG)
        self.assertAlmostEqual(ms.exit_time(song), 41 * BAR)       # the INTERLUDE after the chorus
        self.assertAlmostEqual(ms.entry_time(song), 1 * BAR)       # the INTRO after the lead-in

    def test_no_chorus_leaves_at_the_last_phrase(self):
        song = units([("INTRO", 8), ("VERSE", 32)])
        self.assertAlmostEqual(ms.exit_time(song), 8 * BAR)
        self.assertEqual(ms.exit_time([]), 0.0)
        self.assertEqual(ms.entry_time([]), 0.0)

    def test_the_tempo_where_they_meet(self):
        rows = ((0.0, 10.0, 120.0, 120.0), (10.0, 20.0, 120.0, 140.0))      # steady, then a ramp
        self.assertEqual(ms.tempo_at(rows, 5.0), 120.0)
        self.assertAlmostEqual(ms.tempo_at(rows, 15.0), 130.0)
        self.assertEqual(ms.tempo_at((), 1.0), 0.0)
        self.assertEqual(ms.tempo_offset(120.0, 0.0), float("inf"))


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tracks = []
        for k, (bpm, key) in enumerate(((120.0, 0), (122.0, 7), (150.0, 0), (60.5, 9), (121.0, 6))):
            duration = 65 * 4 * 60.0 / bpm
            self.tracks.append(ms.TrackMarks(f"t{k}", f"Song {k}", tuple(units(SONG, 4 * 60.0 / bpm)),
                                             ((0.0, duration, bpm, bpm),), ((0.0, duration, key),)))
        self.a = self.tracks[0]

    def found(self, **options) -> list[ms.Candidate]:
        return ms.candidates(self.a, self.tracks, set(), ms.MixOptions(**options))

    def uids(self, **options) -> list[str]:
        return [c.uid for c in self.found(**options)]

    def test_bpm_alone_takes_half_and_double_tempo(self):
        self.assertEqual(set(self.uids(bpm=True, key=False)), {"t1", "t3", "t4"})   # 122, 60.5 (x2 = 121), 121

    def test_and_or(self):
        self.assertEqual(set(self.uids(bpm=True, key=True, match_all=True)), {"t1"})    # 122 at G (1B, +-1 of 8B)
        self.assertEqual(set(self.uids(bpm=True, key=True, match_all=False)),
                         {"t1", "t2", "t3", "t4"})              # 150 at C: its key; F# (2B) at 121: its BPM

    def test_nothing_on_lists_every_other_track(self):
        self.assertEqual(set(self.uids(bpm=False, key=False)), {"t1", "t2", "t3", "t4"})

    def test_the_nearest_tempo_first(self):
        found = self.found(bpm=False, key=False)
        self.assertEqual([c.uid for c in found], ["t3", "t4", "t1", "t2"])      # +0.8 %, +0.8 %, +1.7 %, +25 %
        offsets = [abs(c.tempo_pct) for c in found]
        self.assertEqual(offsets, sorted(offsets))

    def test_an_unknown_tempo_comes_last(self):
        unknown = ms.TrackMarks("u", "No grid", self.tracks[1].phrases, (), self.tracks[1].keys)
        found = ms.candidates(self.a, self.tracks + [unknown], set(), ms.MixOptions(bpm=False, key=False))
        self.assertEqual(found[-1].uid, "u")

    def test_the_loaded_track_its_versions_and_instrumentals_left_out(self):
        tracks = self.tracks + [ms.TrackMarks("inst", "Song 0 (Off Vocal)", self.tracks[1].phrases,
                                              self.tracks[1].tempo, self.tracks[1].keys)]
        got = [c.uid for c in ms.candidates(self.a, tracks, {"t1"}, ms.MixOptions(bpm=False, key=False))]
        self.assertNotIn("t0", got)
        self.assertNotIn("t1", got)
        self.assertNotIn("inst", got)

    def test_an_instrumental_by_its_title(self):
        for title in ("紅蓮華 (Instrumental)", "Butter-Fly -instrumental-", "ハレ晴レユカイ (Off Vocal)",
                      "다시 만난 세계 (MR)", "다시 만난 세계 [반주]", "孤勇者 (伴奏)", "孤勇者 (纯音乐)"):
            self.assertTrue(ms.is_instrumental(title), title)
        for title in ("MR.TAXI", "Mr.Children", "紅蓮華 (TV size)"):    # "MR." a name, not an MR
            self.assertFalse(ms.is_instrumental(title), title)

    def test_previous_meets_the_candidate_s_exit_and_the_loaded_track_s_top(self):
        after = {c.uid: c for c in self.found(bpm=False, key=False)}
        before = {c.uid: c for c in self.found(bpm=False, key=False, previous=True)}
        bar_a, bar_b = 4 * 60.0 / 120.0, 4 * 60.0 / 122.0
        self.assertAlmostEqual(after["t1"].a_from, 41 * bar_a)       # the loaded track leaves, t1 comes in
        self.assertAlmostEqual(after["t1"].b_from, 1 * bar_b)
        self.assertAlmostEqual(before["t1"].a_from, 41 * bar_b)      # t1 leaves, the loaded track comes in
        self.assertAlmostEqual(before["t1"].b_from, 1 * bar_a)
        self.assertAlmostEqual(before["t1"].tempo_pct, 100.0 * (120.0 / 122.0 - 1.0))

    def test_key_relations(self):
        self.assertEqual(ms.key_relation(0, 0), "same")
        self.assertEqual(ms.key_relation(0, 21), "relative")      # C, Am
        self.assertEqual(ms.key_relation(0, 7), "adjacent")       # C, G
        self.assertEqual(ms.key_relation(0, 5), "adjacent")       # C, F
        self.assertEqual(ms.key_relation(0, 2), "other")          # C, D
        self.assertEqual(ms.key_relation(None, 2), "")


if __name__ == "__main__":
    unittest.main()
