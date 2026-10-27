"""analyzer.training.text, model_start, library_tracks and the package's helpers: the training's words (marked texts with
their values, one a line, on one line), refusals as TrainingError with the needed and found song counts for each kind
of training, the run's tracks given and handed out as copies, and the tracks as progress and skip lists name them."""
import unittest
from pathlib import Path

from analyzer.training import _compact_display_field, library_tracks, optimizer_track_name, skipped_track
from analyzer.training import model_start as MS
from analyzer.training.text import Concat, Text, TrainingError, msg, render


class TextTests(unittest.TestCase):
    def test_marked_text_rendered_with_its_values(self):
        t = msg("Key features {done}/{total}", done=3, total=10)
        self.assertEqual(t, Text("Key features {done}/{total}", (("done", 3), ("total", 10))))
        self.assertEqual(render(t), "Key features 3/10")
        self.assertEqual(str(t), "Key features 3/10")

    def test_translated_through_the_apps_catalog(self):
        catalog = {"{stage}: step {done}": "{stage} - {done}", "Training": "Lernen"}
        t = msg("{stage}: step {done}", stage=msg("Training"), done=2)
        self.assertEqual(render(t, lambda s: catalog.get(s, s)), "Lernen - 2")

    def test_lines_and_one_line(self):
        self.assertEqual(render((msg("a"), "b", msg("{x}", x=1))), "a\nb\n1")
        self.assertEqual(render(Concat((msg("({n})", n=1), " ", msg("x")))), "(1) x")
        self.assertEqual(render(msg("{v}", v=(msg("a"), msg("b")))), "a\nb")
        self.assertEqual(render(5), "5")
        self.assertEqual(render(None), "None")

    def test_refusal_carries_its_text(self):
        e = TrainingError(msg("Need {n} songs", n=5))
        self.assertIsInstance(e, RuntimeError)
        self.assertEqual(str(e), "Need 5 songs")
        self.assertEqual(e.text.msgid, "Need {n} songs")


class MinimumTests(unittest.TestCase):
    def test_each_kinds_minimum(self):
        for kind, (blocked, warned) in MS.MIN_TRACKS.items():
            with self.subTest(kind=kind):
                self.assertLessEqual(blocked, warned)
                MS.require_tracks(blocked, kind)                         # at the minimum: allowed
                with self.assertRaises(TrainingError) as caught:
                    MS.require_tracks(blocked - 1, kind)
                text = str(caught.exception)
                self.assertIn(str(blocked), text)
                self.assertIn(f"(the library has {blocked - 1})", text)
                self.assertIn("songs with beat grids", text)
        with self.assertRaises(TrainingError) as caught:
            MS.require_tracks(0, "fine_tune", MS.KEY_SONGS)
        self.assertIn("songs with checked key annotations", str(caught.exception))
        self.assertIn("from scratch", self.refusal("scratch"))
        self.assertTrue(self.refusal("search").startswith("Refitting"))

    def refusal(self, kind):
        try:
            MS.require_tracks(0, kind)
        except TrainingError as e:
            return str(e)
        self.fail("not refused")

    def test_scratch_needs_more_than_fine_tuning(self):
        self.assertGreater(MS.MIN_TRACKS["scratch"][0], MS.MIN_TRACKS["fine_tune"][0])
        self.assertEqual(MS.MIN_TRACKS["scratch"][1], MS.SHIPPED_TRAINING_TRACKS)
        self.assertEqual(MS.kind_of(MS.SCRATCH), "scratch")
        self.assertEqual(MS.kind_of(MS.SHIPPED), "fine_tune")
        self.assertEqual(MS.kind_of(MS.CURRENT), "fine_tune")
        with self.assertRaises(KeyError):
            MS.require_tracks(1, "unknown")


class TrackTests(unittest.TestCase):
    def tearDown(self):
        library_tracks._tracks = None

    def test_given_tracks_handed_out_as_copies(self):
        given = [{"uid": "b"}, {"uid": "a"}]
        library_tracks.use(given)
        given[0]["uid"] = "changed"
        out = library_tracks.tracks()
        self.assertEqual([t["uid"] for t in out], ["b", "a"])
        out[0]["beats"] = [1.0]
        self.assertNotIn("beats", library_tracks.tracks()[0])
        library_tracks._tracks = None
        with self.assertRaises(RuntimeError):
            library_tracks.tracks()

    def test_annotation_defaults(self):
        a = library_tracks.Annotation()
        self.assertIsNone(a.beats)
        self.assertEqual((a.jump_cue_groups, a.jump_cue_auto_groups), ([], []))
        self.assertIsNot(a.jump_cue_groups, library_tracks.Annotation().jump_cue_groups)

    def test_names_cut_in_the_middle(self):
        self.assertEqual(_compact_display_field("  short  "), "short")
        long = "a" * 20 + "MIDDLE" + "z" * 20
        cut = _compact_display_field(long)
        self.assertEqual(len(cut), 24)
        self.assertTrue(cut.startswith("a" * 11) and cut.endswith("z" * 10) and "..." in cut)
        self.assertEqual(_compact_display_field(None), "")

    def test_track_names(self):
        words, uid = optimizer_track_name({"uid": "u-1", "title": "", "artist": "", "audio_path": Path("x/My Song.flac")})
        self.assertEqual(render(words), "Title: My Song\nArtist: -")
        self.assertEqual(uid, "UUID: u-1")
        self.assertEqual(render(optimizer_track_name({})[0]), "Title: Untitled\nArtist: -")
        self.assertEqual(optimizer_track_name({})[1], "UUID: -")
        s = skipped_track({"uid": "u", "title": "T", "artist": "A", "audio_path": Path("p.wav")}, "")
        self.assertEqual(render(s.reason), "Unknown validation error")
        self.assertEqual(render(s.display_name), "Title: T\nArtist: A")
        self.assertEqual(s.path, "p.wav")


if __name__ == "__main__":
    unittest.main()
