"""analyzer.training.jumpcue_parameter_optimizer: the candidate cache as NPZ without pickle (round trip, refusals), the
cache key, the saved JumpCUEs snapped to beats and matched to the analysis's groups (accepted or corrected), the
candidate rows, the set-likelihood objective's hand-written gradient against central differences, a fit that ranks
the saved choice first, and small runs of the optimizer on synthetic candidate caches: the model written where asked
and read by the analyzer's own loader, and the refusals."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from analyzer.jumpcue.jump_point_model import JumpPointModel
from analyzer.training import jumpcue_parameter_optimizer as O
from analyzer.training.library_tracks import Annotation
from analyzer.training.text import TrainingError, render

from unit.analyzer_tests import training_fixtures as F

RNG_SEED = 20261027
SHIPPED_JUMP = F.shipped("jumpcue_jump_point.json")
MODEL = json.loads(Path(SHIPPED_JUMP).read_text(encoding="utf-8"))
NAMES = list(MODEL["features"])
THRESHOLD = float(MODEL["short_threshold"])
PAIRS = [(0, 1), (0, 2), (1, 2)]
DURATION = 120.0
BEATS = F.steady_beats(DURATION, 120.0)


def record(rng, members=(16, 48, 80), good=1, shifts=range(-2, 3)):
    """One engine group's candidates: the shift ``good`` sounds best."""
    cands = []
    for d in shifts:
        score = 1.0 if d == good else 0.3
        cands.append({"d": int(d),
                      "member": {k: rng.random(3) + (score if k == "boundary" else 0.0) for k in O._MEMBER_KEYS},
                      "long": {p: score + 0.05 * rng.random() for p in PAIRS},
                      "short": {p: THRESHOLD + 0.05 + 0.1 * rng.random() for p in PAIRS},
                      "pre": {p: score for p in PAIRS}})
    return {"members": np.asarray(members), "pairs": list(PAIRS), "cands": cands}


def seconds(members, d):
    return [float(BEATS[m + d]) for m in members]


class CacheTests(unittest.TestCase):
    def test_round_trip_without_pickle(self):
        obj = {"key": "abc", "records": [{"members": np.arange(3), "pairs": [(0, 1), (1, 2)],
                                          "cands": [{"d": -1, "long": {(0, 1): 0.5, (1, 2): np.float64(0.25)},
                                                     "member": {"onset": np.array([0.1, 0.2, 0.3])}}]}],
               "harmony": np.ones((4, 18), np.float32), "flags": (True, None, 3, "x"), 7: [1.5, float("nan")]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.npz"
            O.save_cache(path, obj)
            with np.load(path, allow_pickle=False) as z:          # nothing in it needs pickle
                self.assertTrue(all(z[k].dtype != object for k in z.files))
            back = O.load_cache(path)
        self.assertEqual(back["key"], "abc")
        self.assertEqual(back["flags"], (True, None, 3, "x"))
        rec = back["records"][0]
        self.assertEqual(rec["pairs"], [(0, 1), (1, 2)])
        self.assertEqual(rec["cands"][0]["long"], {(0, 1): 0.5, (1, 2): 0.25})
        np.testing.assert_array_equal(rec["members"], np.arange(3))
        self.assertEqual(back["harmony"].dtype, np.float32)
        self.assertTrue(np.isnan(back[7][1]))

    def test_other_objects_are_refused_and_nothing_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.npz"
            for bad in ({1, 2}, {"x": object()}, [b"bytes"]):
                with self.subTest(bad=type(bad)):
                    with self.assertRaises(TypeError):
                        O.save_cache(path, bad)
                    self.assertFalse(path.exists())

    def test_cache_key(self):
        settings = {"sr": 22050, "phrase_mtime": 1.0}
        rows = np.zeros((1, 6))
        key = O._cache_key(BEATS, rows, settings)
        self.assertEqual(key, O._cache_key(BEATS.copy(), rows, dict(settings)))
        self.assertNotEqual(key, O._cache_key(BEATS + 1e-6, rows, settings))
        self.assertNotEqual(key, O._cache_key(BEATS, rows, {**settings, "phrase_mtime": 2.0}))
        with mock.patch.object(O, "_CACHE_VERSION", O._CACHE_VERSION + 1):
            self.assertNotEqual(key, O._cache_key(BEATS, rows, settings))


class MatchTests(unittest.TestCase):
    def test_saved_points_snapped_to_beats(self):
        sets = O._beat_sets([[BEATS[4] + 0.05, BEATS[9] - 0.09], [BEATS[4] + 0.2, BEATS[9]],
                             [BEATS[4], BEATS[4] + 0.01], [BEATS[1], BEATS[2], BEATS[3]]], BEATS)
        self.assertEqual(sets, [{4, 9}, {1, 2, 3}])          # too far from a beat, or one beat only: left out

    def test_choice_for_a_saved_group(self):
        rec = record(np.random.default_rng(RNG_SEED))
        self.assertEqual(O._choice_for(rec, {17, 49, 81}), (1, (0, 1, 2)))
        self.assertEqual(O._choice_for(rec, {14, 78}), (-2, (0, 2)))
        self.assertIsNone(O._choice_for(rec, {17, 50}))
        self.assertIsNone(O._choice_for(rec, {30, 60}))

    def test_library_groups_accepted_or_corrected(self):
        rng = np.random.default_rng(RNG_SEED)
        recs = [record(rng), record(rng, members=(120, 152, 184))]
        track = {"uid": "u", "beats": BEATS,
                 "saved": [seconds((16, 48, 80), 1), seconds((120, 152), 0), seconds((200, 210), 0)],
                 "auto": [seconds((16, 48, 80), 0), seconds((120, 152), 0)]}
        groups = O._library_groups(track, recs)
        self.assertEqual([(g["kind"], g["label"], g["rejected"]) for g in groups],
                         [("corrected", (1, (0, 1, 2)), (0, (0, 1, 2))), ("accepted", (0, (0, 1)), None)])
        self.assertTrue(all(c["masking"] == {} for g in groups for c in g["cands"]))
        self.assertEqual(O._library_groups({**track, "saved": []}, recs), [])

    def test_candidate_rows(self):
        rng = np.random.default_rng(RNG_SEED)
        track = {"uid": "u", "beats": BEATS, "saved": [seconds((16, 48, 80), 1)], "auto": [seconds((16, 48, 80), 0)]}
        group = O._library_groups(track, [record(rng)])[0]
        rows = O._candidate_rows(group, NAMES, THRESHOLD)
        self.assertEqual(rows.raw_X.shape, (5 * 4, len(NAMES)))             # 5 shifts x (the 3 and its 3 pairs)
        self.assertEqual(rows.choices[rows.target_idx], (1, (0, 1, 2)))
        self.assertEqual(rows.choices[rows.rejected_idx], (0, (0, 1, 2)))
        self.assertEqual(int(rows.is_full.sum()), 5)
        share = NAMES.index("share")
        self.assertEqual(sorted(set(rows.raw_X[:, share])), [2 / 3, 1.0])
        idx, X = O._eligible_matrix(rows, THRESHOLD)
        np.testing.assert_allclose(X.mean(0), 0.0, atol=1e-9)
        group["label"] = (5, (0, 1, 2))
        with self.assertRaises(ValueError):
            O._candidate_rows(group, NAMES, THRESHOLD)

    def test_subsets_below_the_short_threshold_left_out(self):
        rng = np.random.default_rng(RNG_SEED)
        rec = record(rng)
        for c in rec["cands"]:
            c["short"][(0, 2)] = THRESHOLD - 0.2
        track = {"uid": "u", "beats": BEATS, "saved": [seconds((16, 48, 80), 1)], "auto": []}
        rows = O._candidate_rows(O._library_groups(track, [rec])[0], NAMES, THRESHOLD)
        kept = {c[1] for c in rows.choices}
        self.assertEqual(kept, {(0, 1, 2), (0, 1), (1, 2)})          # the full set always; (0, 2) never


def groups_of(n_tracks, rng, corrected=True):
    out = []
    for i in range(n_tracks):
        good = int(rng.integers(-2, 3))
        track = {"uid": f"u{i}", "beats": BEATS, "saved": [seconds((16, 48, 80), good)],
                 "auto": [seconds((16, 48, 80), (good + 1) % 3)] if corrected else []}
        out += [O._candidate_rows(g, NAMES, THRESHOLD) for g in O._library_groups(track, [record(rng, good=good)])]
    return out


class FitTests(unittest.TestCase):
    def test_objective_gradient_matches_central_differences(self):
        rng = np.random.default_rng(RNG_SEED)
        groups = groups_of(6, rng)
        O._assign_positive_sets(groups, O._metrics(groups, np.asarray(MODEL["weights"]), THRESHOLD), NAMES, THRESHOLD)
        captured = {}

        def fake_minimize(fun, x0, **kw):
            captured["f"] = fun
            return SimpleNamespace(x=x0, success=True, message="")
        with mock.patch.object(O, "minimize", fake_minimize):
            O._fit(groups, np.zeros(len(NAMES)), {"u1"}, l2=0.5, hard_weight=2.0, corrected_weight=3.0,
                   short_threshold=THRESHOLD, preference_weight=1.5)
        f = captured["f"]
        w = rng.normal(0, 0.3, len(NAMES))
        loss, grad = f(w)
        self.assertTrue(np.isfinite(loss))
        numeric = F.numeric_grad(lambda P: f(P["w"])[0], {"w": w}, "w", range(w.size), eps=1e-6)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-6)

    def test_fit_ranks_the_saved_choice_first(self):
        rng = np.random.default_rng(RNG_SEED + 1)
        groups = groups_of(8, rng)
        weights = O._fit(groups, np.zeros(len(NAMES)), set(), l2=0.1, hard_weight=1.0, short_threshold=THRESHOLD,
                         preference_weight=3.0)
        metrics = O._metrics(groups, weights, THRESHOLD)
        self.assertEqual((metrics["exact_accuracy"], metrics["groups"], metrics["tracks"]), (1.0, 8, 8))
        values = O._rank_values(groups, weights, THRESHOLD, NAMES)
        self.assertEqual(values["user_above_automatic"], [1.0] * 8)
        self.assertGreater(min(values["auc"]), 0.9)
        self.assertEqual(O._mean_values({"a": [], "b": [1.0, 0.0]})["b"], 0.5)
        self.assertTrue(np.isnan(O._mean_values({"a": []})["a"]))

    def test_positive_sets_take_equivalent_shifts_of_the_saved_members(self):
        rng = np.random.default_rng(RNG_SEED + 2)
        g = groups_of(1, rng)[0]
        current = O._metrics([g], np.zeros(len(NAMES)), THRESHOLD)
        O._assign_positive_sets([g], current, NAMES, THRESHOLD, max_positive=4)
        self.assertIn(g.target_idx, g.positive_idx)
        self.assertLessEqual(len(g.positive_idx), 5)
        self.assertTrue(all(g.choices[i][1] == g.target[1] for i in g.positive_idx))
        if current["rows"][0]["pred_idx"] != g.target_idx:              # a corrected group's own wrong pick never
            self.assertNotIn(current["rows"][0]["pred_idx"], g.positive_idx)


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "jumpcue"
        self.out = self.dir / "weights" / "jumpcue_jump_point.json"
        self.phrase = self.dir / "phrase_analyzer.npz"
        self.enterContext(mock.patch("analyzer.runtime.spawn_pool", F.no_pool))

    def tearDown(self):
        F.library_tracks._tracks = None
        self._tmp.cleanup()

    def tracks(self, n, unmatched=0, caches=True):
        """``n`` tracks with saved JumpCUEs, the last ``unmatched`` of them matching no found group, and one without."""
        rng = np.random.default_rng(RNG_SEED)
        settings = O._analysis_settings(self.phrase)
        rows = F.tempo_rows(BEATS, DURATION, 120.0)
        tracks = []
        for i in range(n):
            good = int(rng.integers(-2, 3))
            saved = [seconds((16, 48, 80), good if i < n - unmatched else 30)]
            a = Annotation(beats=BEATS, sample_rate=F.SR, duration_sec=DURATION, tempo_segments=rows,
                           jump_cue_groups=saved, jump_cue_auto_groups=[seconds((16, 48, 80), (good + 1) % 3)])
            t = F.given_track(self.dir, i, a)
            tracks.append(t)
            if caches:
                O.save_cache(self.cache / f"{t['uid']}.npz", {"key": O._cache_key(BEATS, rows, settings),
                                                               "records": [record(rng, good=good)], "harmony": None})
        tracks.append(F.given_track(self.dir, n, Annotation(beats=BEATS, tempo_segments=rows)))   # no JumpCUEs
        F.use_tracks(tracks)
        return tracks

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out,
                      phrase_parameter_path=self.phrase, current_path=SHIPPED_JUMP, cv_folds=0)
        values.update(kw)
        return O.JumpCueOptimizationRequest(**values)

    def test_model_written_and_read_by_the_analyzers_loader(self):
        self.tracks(5)
        progress = []
        result = O.optimize_jumpcue_parameters(self.request(), optimize_progress=lambda v, t: progress.append(v))
        self.assertEqual((result.track_count, result.group_count, result.preference_pairs), (5, 5, 5))
        self.assertFalse(result.cross_validated)
        model = JumpPointModel.load(self.out)
        self.assertIsNotNone(model)
        self.assertEqual(model.names, NAMES)
        self.assertEqual(model.weights.shape, (len(NAMES),))
        self.assertFalse(np.allclose(model.weights, MODEL["weights"]))
        self.assertEqual(json.loads(self.out.read_text(encoding="utf-8"))["trained_on"]["corrected"], 5)
        self.assertGreater(result.user_above_automatic, 0.5)
        self.assertEqual(progress[-1], 100)

    def test_cross_validated_run(self):
        self.tracks(5)
        result = O.optimize_jumpcue_parameters(self.request(cv_folds=2))
        self.assertTrue(result.cross_validated)
        self.assertTrue(0.0 <= result.label_auc <= 1.0)
        self.assertTrue(self.out.is_file())

    def test_fewer_than_five_tracks_with_saved_jumpcues_refused(self):
        self.tracks(4, caches=False)
        with self.assertRaises(TrainingError) as caught:
            O.optimize_jumpcue_parameters(self.request())
        self.assertIn("(4 found)", str(caught.exception))
        self.assertFalse(self.out.exists())

    def test_tracks_left_out_with_their_reasons(self):
        tracks = self.tracks(7, unmatched=1)
        tracks[0]["audio_path"].unlink()
        result = O.optimize_jumpcue_parameters(self.request())
        self.assertEqual(result.track_count, 5)
        self.assertEqual(sorted(render(s.reason) for s in result.skipped_tracks),
                         ["Audio file not found", "No saved JumpCUE group matches an automatically found group"])

    def test_stale_caches_are_rebuilt(self):
        self.tracks(5)
        built = []

        def build(track, cache_dir, settings, files):
            built.append(track["uid"])
            raise RuntimeError("no audio decoder here")
        with mock.patch("analyzer.runtime.spawn_pool", F.thread_pool), mock.patch.object(O, "_build_track_cache", build):
            with self.assertRaises(Exception):
                O.optimize_jumpcue_parameters(self.request(rebuild_cache=True))
        self.assertEqual(len(built), 5)
        self.assertFalse(self.out.exists())

    # Bug: too few tracks whose saved JumpCUEs match a found group raises a RuntimeError (shown as a traceback,
    # its text untranslated) instead of the module's TrainingError refusal.
    @unittest.expectedFailure
    def test_too_few_tracks_with_matching_groups_refused_as_a_training_error(self):
        self.tracks(5, unmatched=1)
        with self.assertRaises(TrainingError):
            O.optimize_jumpcue_parameters(self.request())
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
