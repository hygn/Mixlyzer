"""core.marks: the phrases ordered and cleaned and the track covered (UNASSIGNED where none is, never stored), the
labels numbered and named, the JumpCUE slot rule (one letter A-Z each) and the colour presets."""
import math
import random
import unittest

from analyzer.music.marks import JumpCue, JumpCueGroup, Phrase
from core.marks import colors, jump_cues, phrases as ph
from core.marks.phrases import PHRASE_SLIVER_SEC, UNASSIGNED

SEED = 20261027


def covers(rows, duration):
    """Ordered, touching, from 0 to the duration."""
    return (rows and math.isclose(rows[0].start, 0.0, abs_tol=1e-9) and math.isclose(rows[-1].end, duration)
            and all(math.isclose(a.end, b.start, abs_tol=1e-9) for a, b in zip(rows, rows[1:])))


def random_phrases(rng: random.Random, duration: float, unassigned: bool = True) -> list[Phrase]:
    out = []
    for _ in range(rng.randint(0, 12)):
        start = rng.uniform(-10.0, duration + 10.0)
        length = rng.choice((rng.uniform(0.0, 0.2), rng.uniform(0.0, 60.0), -rng.uniform(0, 5)))
        label = rng.choice(ph.PHRASE_LABELS + ["", " chorus "] + ([UNASSIGNED] if unassigned else []))
        out.append(Phrase(start, start + length, label))
    if rng.random() < 0.2:
        out.append(Phrase(float("nan"), 3.0, "INTRO"))
    if rng.random() < 0.2:
        out.append(Phrase(1.0, float("inf"), "OUTRO"))
    return out


class NormalizeTests(unittest.TestCase):
    def test_cases(self):
        n = ph.normalize_phrase_segments
        self.assertEqual(n([]), [])
        self.assertEqual(n(None), [])
        self.assertEqual(n([Phrase(10, 20, "VERSE"), Phrase(0, 10, "INTRO")]),
                         [Phrase(0, 10, "INTRO"), Phrase(10, 20, "VERSE")])                 # ordered
        self.assertEqual(n([Phrase(0, 12, "INTRO"), Phrase(10, 20, "VERSE")]),
                         [Phrase(0, 12, "INTRO"), Phrase(12, 20, "VERSE")])                 # overlap: the later cut
        self.assertEqual(n([Phrase(0, 10, "INTRO"), Phrase(10.01, 20, "VERSE")]),
                         [Phrase(0, 10.01, "INTRO"), Phrase(10.01, 20, "VERSE")])           # hairline gap closed
        self.assertEqual(n([Phrase(0, 10, "INTRO"), Phrase(10, 10.02, "X"), Phrase(10.02, 20, "VERSE")]),
                         [Phrase(0, 10.02, "INTRO"), Phrase(10.02, 20, "VERSE")])           # a sliver: the one before
        self.assertEqual(n([Phrase(5, 3, "VERSE")]), [])                                    # reversed
        self.assertEqual(n([Phrase(float("nan"), 3, "A"), Phrase(0, float("inf"), "B")]), [])
        self.assertEqual(n([Phrase(0, 10, "  CHORUS ")]), [Phrase(0, 10, "CHORUS")])        # label trimmed
        self.assertEqual(n([Phrase(0, 10, None)]), [Phrase(0, 10, "")])

    def test_bounds(self):
        rows = [Phrase(-5, 5, "INTRO"), Phrase(5, 30, "VERSE"), Phrase(30, 50, "OUTRO")]
        self.assertEqual(ph.normalize_phrase_segments(rows, min_time=0, max_time=40),
                         [Phrase(0, 5, "INTRO"), Phrase(5, 30, "VERSE"), Phrase(30, 40, "OUTRO")])
        self.assertEqual(ph.normalize_phrase_segments(rows, min_time=40, max_time=0),        # swapped: the same
                         ph.normalize_phrase_segments(rows, min_time=0, max_time=40))
        self.assertEqual(ph.normalize_phrase_segments(rows, min_time=float("nan")), ph.normalize_phrase_segments(rows))

    def test_not_a_phrase_is_an_error(self):
        with self.assertRaises(AttributeError):
            ph.normalize_phrase_segments([(0, 10, "INTRO")])

    def test_random_inputs(self):
        """Whatever comes in: ordered, apart, finite, none shorter than a sliver, and a second pass changes
        nothing."""
        rng = random.Random(SEED)
        for _ in range(2000):
            rows = ph.normalize_phrase_segments(random_phrases(rng, 240.0))
            for r in rows:
                self.assertTrue(math.isfinite(r.start) and math.isfinite(r.end))
                self.assertGreaterEqual(r.end - r.start, PHRASE_SLIVER_SEC)
            for a, b in zip(rows, rows[1:]):
                self.assertLessEqual(a.end, b.start)
            self.assertEqual(ph.normalize_phrase_segments(rows), rows)


class CoverageTests(unittest.TestCase):
    def test_fill(self):
        rows = ph.fill_unassigned([Phrase(10, 20, "VERSE")], 30.0)
        self.assertEqual(rows, [Phrase(0, 10, UNASSIGNED), Phrase(10, 20, "VERSE"), Phrase(20, 30, UNASSIGNED)])
        self.assertEqual(ph.fill_unassigned([], 30.0), [Phrase(0, 30, UNASSIGNED)])
        self.assertEqual(ph.fill_unassigned([Phrase(0, 40, "INTRO")], 30.0), [Phrase(0, 30, "INTRO")])  # cut at the end

    def test_slivers_at_the_ends_covered(self):
        """The whole track is covered, its first and last few ms too (a playhead there is in a phrase)."""
        for rows in ([Phrase(0.02, 30, "INTRO")], [Phrase(0, 29.99, "INTRO")]):
            with self.subTest(rows=rows):
                self.assertTrue(covers(ph.fill_unassigned(rows, 30.0), 30.0))

    def test_duration_forms(self):
        import numpy as np
        for duration in (30.0, np.float32(30.0), np.array([30.0]), [30.0], "30"):
            with self.subTest(duration=duration):
                self.assertTrue(covers(ph.fill_unassigned([Phrase(5, 10, "VERSE")], duration), 30.0))
        for duration in (0.0, -1.0, float("nan"), None, "x", []):
            with self.subTest(duration=duration):
                self.assertEqual(ph.fill_unassigned([Phrase(5, 10, "VERSE")], duration), [Phrase(5, 10, "VERSE")])

    def test_random_tracks_covered(self):
        rng = random.Random(SEED + 1)
        for _ in range(2000):
            duration = rng.uniform(1.0, 600.0)
            filled = ph.fill_unassigned(random_phrases(rng, duration), duration)
            # covered but for a sliver at either end (test_slivers_at_the_ends_covered)
            self.assertLess(filled[0].start, PHRASE_SLIVER_SEC)
            self.assertLess(duration - filled[-1].end, PHRASE_SLIVER_SEC)
            self.assertTrue(covers([Phrase(0.0, filled[0].end, ""), *filled[1:-1],
                                    Phrase(filled[-1].start, duration, "")] if len(filled) > 1
                                   else [Phrase(0.0, duration, "")], duration), filled)
            stored = ph.assigned_phrases(filled)
            self.assertNotIn(UNASSIGNED, [r.label for r in stored])

    def test_stored_phrases_read_back(self):
        """A track's phrases as the editor saves them (UNASSIGNED left out) read back as they were shown."""
        rng = random.Random(SEED + 2)
        for _ in range(2000):
            duration = rng.uniform(1.0, 600.0)
            shown = ph.fill_unassigned(random_phrases(rng, duration, unassigned=False), duration)
            self.assertEqual(ph.fill_unassigned(ph.assigned_phrases(shown), duration), shown)


class LabelTests(unittest.TestCase):
    def tearDown(self):
        ph.set_phrase_label_names({})

    def test_numbers(self):
        rows = [Phrase(0, 10, "INTRO"), Phrase(10, 20, "CHORUS"), Phrase(20, 30, UNASSIGNED),
                Phrase(30, 40, "CHORUS"), Phrase(40, 50, UNASSIGNED), Phrase(50, 60, "OUTRO")]
        self.assertEqual(ph.number_phrase_labels(rows), ["INTRO", "CHORUS1", UNASSIGNED, "CHORUS2", UNASSIGNED, "OUTRO"])
        self.assertEqual(ph.abbreviate_phrase_labels(rows), ["I", "C1", "U", "C2", "U", "O"])

    def test_names(self):
        class Name:
            def __init__(self, alias, abbrev):
                self.alias, self.abbrev = alias, abbrev

        ph.set_phrase_label_names({"CHORUS": Name(" Hook ", ""), "VERSE": Name("", "Vs"), "INTRO": Name(None, None)})
        rows = [Phrase(0, 10, "CHORUS"), Phrase(10, 20, "VERSE"), Phrase(20, 30, "INTRO")]
        self.assertEqual(ph.number_phrase_labels(rows), ["Hook", "VERSE", "INTRO"])
        self.assertEqual(ph.abbreviate_phrase_labels(rows), ["H", "Vs", "I"])
        ph.set_phrase_label_names({})
        self.assertEqual(ph.number_phrase_labels(rows), ["CHORUS", "VERSE", "INTRO"])

    def test_colours(self):
        self.assertEqual(ph.phrase_color("no such label"), ph.phrase_color(UNASSIGNED))
        self.assertEqual(len({ph.phrase_color(label) for label in ph.PHRASE_LABELS}), len(ph.PHRASE_LABELS))
        self.assertEqual([s[:2] for s in ph.phrase_strip_segments([Phrase(5, 9, "VERSE")])], [(5, 9)])


def cue(label, point, comment=None):
    return JumpCue(label, label if comment is None else comment, point - 1.0, point + 1.0, point)


class JumpCueSlotTests(unittest.TestCase):
    def test_rule(self):
        ok = [JumpCueGroup((cue("A", 10), cue("B", 20)), 0), JumpCueGroup((cue("Z", 5),), 1), JumpCueGroup((), 2)]
        self.assertTrue(jump_cues.has_slot_letters(ok))
        self.assertTrue(jump_cues.has_slot_letters([]))
        for bad in ("a", "AA", "", " A", "1", "Ä", "Ａ"):
            with self.subTest(label=bad):
                self.assertFalse(jump_cues.has_slot_letters([JumpCueGroup((cue(bad, 1),), 0)]))
                with self.assertRaises(ValueError):
                    jump_cues.check_slot_letters([JumpCueGroup((cue(bad, 1),), 0)])
        twice = [JumpCueGroup((cue("A", 1),), 0), JumpCueGroup((cue("A", 2),), 1)]
        self.assertFalse(jump_cues.has_slot_letters(twice))

    def test_lettered_by_point(self):
        groups = [JumpCueGroup((cue("x", 30, "Drop"), cue("y", 10)), 4), JumpCueGroup((cue("q", 20),), 1)]
        out = jump_cues.lettered_by_point(groups)
        self.assertTrue(jump_cues.has_slot_letters(out))
        letters = {c.point: (c.label, c.comment) for g in out for c in g.cues}
        self.assertEqual(letters, {10: ("A", "A"), 20: ("B", "B"), 30: ("C", "Drop")})   # a name kept, a letter follows
        self.assertEqual([g.color_idx for g in out], [4, 1])                              # the groups as they were

    def test_past_slot_z(self):
        many = [JumpCueGroup(tuple(cue("?", float(k)) for k in range(20)), 0),
                JumpCueGroup(tuple(cue("?", 100.0 + k) for k in range(10)), 1),
                JumpCueGroup((cue("?", 500.0),), 2)]
        out = jump_cues.lettered_by_point(many)
        points = sorted(c.point for g in out for c in g.cues)
        self.assertEqual(len(points), 26)
        self.assertEqual(points[-1], 105.0)                                  # the latest ones left out
        self.assertEqual(len(out), 2)                                        # a group left without cues goes
        self.assertTrue(jump_cues.has_slot_letters(out))

    def test_ties_by_point_keep_their_order(self):
        out = jump_cues.lettered_by_point([JumpCueGroup((cue("?", 5.0), cue("?", 5.0)), 0)])
        self.assertEqual([c.label for c in out[0].cues], ["A", "B"])


class ColourTests(unittest.TestCase):
    def test_presets(self):
        n = len(colors.JUMPCUE_PAIR_COLORS)
        self.assertEqual(colors.get_jumpcue_pair_color(n), colors.get_jumpcue_pair_color(0))
        self.assertEqual(colors.get_jumpcue_pair_color(-1), colors.get_jumpcue_pair_color(n - 1))
        self.assertEqual(colors.get_cue_point_color(-1), colors.CUE_POINT_COLORS[0])
        self.assertEqual(colors.get_cue_point_color(len(colors.CUE_POINT_COLORS)), colors.CUE_POINT_COLORS[0])
        self.assertNotIn(colors.CUE_POINT_COLORS[0], colors.JUMPCUE_PAIR_COLORS)     # the CUE point red apart
        self.assertEqual(len(set(colors.JUMPCUE_PAIR_COLORS)), n)

    def test_cues_with_their_colours_by_point(self):
        groups = [JumpCueGroup((cue("B", 20),), 1), JumpCueGroup((cue("A", 10),), 0)]
        got = colors.jump_cue_colors(groups)
        self.assertEqual([(c.label, rgb) for c, rgb in got],
                         [("A", colors.JUMPCUE_PAIR_COLORS[0]), ("B", colors.JUMPCUE_PAIR_COLORS[1])])


if __name__ == "__main__":
    unittest.main()
