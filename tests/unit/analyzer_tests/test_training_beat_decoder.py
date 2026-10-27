"""analyzer.training.beat_decoder_optimizer: the decoder costs chosen from every song's wrong time per cost combination
(each song's tracking replaced by a synthetic table): the least wrong time wins, ties keep the current costs, Quick
mode keeps the section cost, the held-out check keeps worse costs from being written, the costs file is read by the
tracker's own loader, the per-song table is written without pickle, and the refusals."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.beat.general import tracker as TR
from analyzer.training import beat_decoder_optimizer as O
from analyzer.training import model_start as MS
from analyzer.training.text import TrainingError, render

from unit.analyzer_tests import training_fixtures as F

SHIPPED_DECODER = F.shipped("beat_decoder.json")
CURRENT = TR.load_knobs(SHIPPED_DECODER)
SMALL = 4


def grid(quick=False):
    def around(values, now):
        return tuple(sorted(set(values) | {float(now)}))
    return (around(() if quick else O.SECTION_COSTS, CURRENT.section), around(O.HALF_RESET_COSTS, CURRENT.half_reset),
            around(O.ODD_BAR_COSTS, CURRENT.odd_bar))


def table(g, base=5.0, better=None, worse=None):
    """[S, H, O, 2] wrong seconds: ``base`` everywhere, ``better`` / ``worse`` {(section, half, odd): seconds}."""
    t = np.full((len(g[0]), len(g[1]), len(g[2]), 2), base / 2)
    for change, sign in ((better or {}, -1), (worse or {}, 1)):
        for (s, h, o), seconds in change.items():
            t[g[0].index(s), g[1].index(h), g[2].index(o), 0] += sign * seconds
    return t


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.out = self.dir / "weights" / "beat_decoder.json"
        self.enterContext(mock.patch.object(O, "ensure_frames", lambda tracks, *a, **k: tracks))
        self.enterContext(mock.patch.object(O, "spawn_pool", F.thread_pool))
        self.tables = {}
        self.enterContext(mock.patch.object(O, "_score_song", lambda job: (str(job[0]["uid"]),
                                                                           self.tables.get(str(job[0]["uid"])))))

    def tearDown(self):
        F.library_tracks._tracks = None
        self._tmp.cleanup()

    def tracks(self, tables):
        tracks = [F.given_track(self.dir, i, F.beat_annotation()) for i in range(len(tables))]
        self.tables = {t["uid"]: tab for t, tab in zip(tracks, tables) if tab is not None}
        F.use_tracks(tracks)
        return tracks

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.dir / "frames", output_path=self.out,
                      current_path=SHIPPED_DECODER, start_path=SHIPPED_DECODER, max_cache_workers=1)
        values.update(kw)
        return O.BeatDecoderOptimizationRequest(**values)

    def run_small(self, request):
        with mock.patch.dict(MS.MIN_TRACKS, {"search": (SMALL, SMALL)}):
            return O.optimize_beat_decoder(request)

    def test_least_wrong_time_wins_and_is_read_by_the_trackers_loader(self):
        g = grid()
        best = (17.0, 136.0, 57.0)
        self.tracks([table(g, better={best: 1.0, (8.5, 34.0, 14.0): 0.5}) for _ in range(SMALL)])
        result = self.run_small(self.request(table_path=self.dir / "table.npz"))
        self.assertTrue(result.saved)
        self.assertEqual(TR.load_knobs(str(self.out)), TR.Knobs(*best))
        self.assertEqual(result.knobs_after, {"section": 17.0, "half_reset": 136.0, "odd_bar": 57.0})
        self.assertEqual(result.knobs_before, {"section": CURRENT.section, "half_reset": CURRENT.half_reset,
                                               "odd_bar": CURRENT.odd_bar})
        self.assertAlmostEqual(result.wrong_before_sec - result.wrong_after_sec, SMALL * 1.0)
        training = json.loads(self.out.read_text(encoding="utf-8"))["training"]
        self.assertEqual((training["tracks"], training["quick"]), (SMALL, False))
        with np.load(self.dir / "table.npz", allow_pickle=False) as z:
            self.assertEqual(z["tables"].shape, (SMALL, *map(len, g), 2))
            np.testing.assert_array_equal(z["section"], g[0])

    def test_ties_keep_the_current_costs(self):
        g = grid()
        self.tracks([table(g, better={(17.0, 136.0, 57.0): 0.0}) for _ in range(SMALL)])
        result = self.run_small(self.request())
        self.assertTrue(result.saved)
        self.assertEqual(TR.load_knobs(str(self.out)), CURRENT)
        self.assertEqual(result.wrong_before_sec, result.wrong_after_sec)

    def test_quick_mode_keeps_the_section_cost(self):
        g = grid(quick=True)
        self.assertEqual(g[0], (CURRENT.section,))
        self.tracks([table(g, better={(CURRENT.section, 48.0, 28.0): 2.0}) for _ in range(SMALL)])
        result = self.run_small(self.request(quick=True))
        self.assertEqual(TR.load_knobs(str(self.out)), TR.Knobs(CURRENT.section, 48.0, 28.0))
        self.assertEqual(result.knobs_after["section"], CURRENT.section)

    def test_cross_validated_choice_scored_on_held_out_songs(self):
        g = grid()
        best = (8.5, 192.0, 80.0)
        self.tracks([table(g, better={best: 1.0}) for _ in range(SMALL)])
        result = self.run_small(self.request(cv_folds=2))
        self.assertTrue(result.saved and result.cross_validated)
        self.assertEqual(result.grid.songs, SMALL)                            # every song held out once
        self.assertEqual(TR.load_knobs(str(self.out)), TR.Knobs(*best))

    def test_worse_held_out_costs_leave_the_file_unwritten(self):
        g = grid()
        x = (17.0, 34.0, 14.0)
        # songs 0 and 1 gain a little with x, songs 2 and 3 lose a lot: chosen on 0 and 1, x is checked on 2 and 3
        self.tracks([table(g, better={x: 1.0}), table(g, better={x: 1.0}),
                     table(g, worse={x: 4.0}), table(g, worse={x: 4.0})])

        class Split:
            def __init__(self, n_splits):
                pass

            def split(self, X, groups=None):
                yield np.array([0, 1]), np.array([2, 3])
        with mock.patch.object(O, "GroupKFold", Split):
            result = self.run_small(self.request(cv_folds=2))
        self.assertFalse(result.saved or self.out.exists())
        self.assertAlmostEqual(result.wrong_after_sec - result.wrong_before_sec, 8.0)
        self.assertEqual([c.title for c in result.grid.worse], ["Song 2", "Song 3"])

    def test_songs_without_a_table_skipped_and_counted(self):
        g = grid()
        self.tracks([table(g)] * SMALL + [None])
        result = self.run_small(self.request())
        self.assertEqual(result.track_count, SMALL)
        self.assertEqual([render(s.reason) for s in result.skipped_tracks], ["frame feature cache is not current"])
        self.tracks([table(g)] * (SMALL - 1) + [None])
        with self.assertRaises(TrainingError):
            self.run_small(self.request())

    def test_too_few_songs_refused(self):
        self.tracks([None] * (MS.MIN_TRACKS["search"][0] - 1))
        with mock.patch.object(O, "spawn_pool", F.no_pool), self.assertRaises(TrainingError):
            O.optimize_beat_decoder(self.request())
        self.assertFalse(self.out.exists())

    def test_output_must_be_json(self):
        with self.assertRaises(ValueError):
            O.optimize_beat_decoder(self.request(output_path=self.dir / "decoder.npz"))


if __name__ == "__main__":
    unittest.main()
