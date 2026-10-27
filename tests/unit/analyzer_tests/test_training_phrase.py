"""analyzer.training.phrase_backend, phrase_refinement, phrase_lead_in and phrase_parameter_optimizer: the stored
phrases read as training targets (and every malformed annotation refused), the phrases snapped to beats, the tracks
with phrase annotations, the decoded pipeline's metrics, the LEAD_IN targets, accuracy and models on synthetic
features, the optimizer's caches' validity checks and its refusal of too few annotated tracks."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.training import phrase_backend as B
from analyzer.training import phrase_lead_in as L
from analyzer.training import phrase_parameter_optimizer as P
from analyzer.training import phrase_refinement as R
from analyzer.training.library_tracks import Annotation
from analyzer.training.text import Concat, TrainingError, render

from unit.analyzer_tests import training_fixtures as F

RNG_SEED = 20261027


def phrases(starts, ends, labels):
    return Annotation(phrase_starts=np.asarray(starts, float), phrase_ends=np.asarray(ends, float),
                      phrase_labels=np.asarray(labels))


class AnnotationTests(unittest.TestCase):
    def test_contiguous_phrases_read(self):
        s, e, lab = B._load_phrase_annotation(phrases([0, 10, 20], [10, 20, 30], ["INTRO", "VERSE", "CHORUS"]))
        np.testing.assert_array_equal(s, [0, 10, 20])
        self.assertEqual(lab.tolist(), ["INTRO", "VERSE", "CHORUS"])

    def test_no_phrases_is_no_training_data(self):
        for a in (Annotation(), phrases([5.0], [5.0], ["INTRO"])):         # none, or only a collapsed row
            s, e, lab = B._load_phrase_annotation(a)
            self.assertEqual((s.size, e.size, lab.size), (0, 0, 0))

    def test_collapsed_rows_dropped(self):
        s, _, lab = B._load_phrase_annotation(phrases([0, 10, 10], [10, 10 + 1e-5, 20], ["A", "B", "C"]))
        self.assertEqual(lab.tolist(), ["A", "C"])

    def test_malformed_annotations_refused(self):
        bad = {"missing": Annotation(phrase_starts=np.zeros(1), phrase_ends=np.ones(1)),
               "mismatched": phrases([0, 10], [10], ["A", "B"]),
               "nan": phrases([0, np.nan], [10, 20], ["A", "B"]),
               "empty_label": phrases([0, 10], [10, 20], ["A", " "]),
               "unsorted": phrases([10, 0], [20, 10], ["A", "B"]),
               "overlap": phrases([0, 9], [10, 20], ["A", "B"]),
               "gap": phrases([0, 11], [10, 20], ["A", "B"])}
        for name, a in bad.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                B._load_phrase_annotation(a)

    def test_duration(self):
        self.assertEqual(B._load_duration_sec(Annotation(duration_sec=12.5)), 12.5)
        for d in (None, 0.0, -1.0, float("nan")):
            self.assertEqual(B._load_duration_sec(Annotation(duration_sec=d)), float("inf"))

    def test_predictor_grid_needs_beats_and_tempo_rows(self):
        a = F.beat_annotation(duration=20.0)
        grid = B.predictor_grid({"annotation": a})
        np.testing.assert_allclose(grid.beat_times_sec, a.beats)
        for missing in ("beats", "tempo_segments"):
            with self.assertRaises(KeyError):
                B.predictor_grid({"annotation": Annotation(**{**a.__dict__, missing: None})})


class TrackListTests(unittest.TestCase):
    def test_annotated_tracks_listed_and_the_others_left_out(self):
        good = phrases([0, 10], [10, 20], ["INTRO", "VERSE"])
        cases = [("good", good, True, {}), ("none", None, True, {"error": "unreadable"}), ("no_audio", good, False, {}),
                 ("one", phrases([0], [20], ["INTRO"]), True, {}), ("no_phrases", Annotation(), True, {}),
                 ("bad", phrases([0, 9], [10, 20], ["A", "B"]), True, {})]
        with tempfile.TemporaryDirectory() as tmp:
            F.use_tracks([F.given_track(Path(tmp), i, a, audio=audio, title=name, **extra)
                          for i, (name, a, audio, extra) in enumerate(cases)])
            try:
                ignored = []
                listed = B._list_annotated_tracks(Path(tmp), ignored)
            finally:
                F.library_tracks._tracks = None
        self.assertEqual([t["title"] for t in listed], ["good"])
        self.assertEqual(listed[0]["duration_sec"], float("inf"))
        reasons = {s.title: render(s.reason) for s in ignored}
        self.assertEqual(reasons["none"], "invalid Phrase annotation: unreadable")
        self.assertEqual(reasons["no_audio"], "audio file is missing")
        self.assertEqual(reasons["one"], "fewer than two usable Phrase segments")
        self.assertTrue(reasons["bad"].startswith("invalid Phrase annotation: Overlapping"))
        self.assertNotIn("no_phrases", reasons)                    # a track without phrases is just not training data


class BeatTargetTests(unittest.TestCase):
    BEATS = np.arange(32) * 0.5

    def track(self, starts, ends, labels, **extra):
        return {"uid": "u", "beat_times_sec": self.BEATS, "phrase_starts_sec": np.asarray(starts, float),
                "phrase_ends_sec": np.asarray(ends, float), "phrase_labels": labels, **extra}

    def test_phrases_on_beat_indices(self):
        bounds, labels = B._gt_segments_beats(self.track([0, 4, 8], [4, 8, 16], ["INTRO", "VERSE", "CHORUS"]))
        np.testing.assert_array_equal(bounds, [0, 8, 16, 32])
        self.assertEqual(labels, ["INTRO", "VERSE", "CHORUS"])

    def test_starts_snap_to_the_nearest_beat(self):
        bounds, _ = B._gt_segments_beats(self.track([0, 4.2, 7.8], [4.2, 7.8, 16], ["A", "B", "C"]))
        np.testing.assert_array_equal(bounds, [0, 8, 16, 32])

    def test_phrases_on_one_beat_keep_the_later_label(self):
        bounds, labels = B._gt_segments_beats(self.track([0, 4.0, 4.1, 8], [4.0, 4.1, 8, 16], ["A", "B", "C", "D"]))
        np.testing.assert_array_equal(bounds, [0, 8, 16, 32])
        self.assertEqual(labels, ["A", "C", "D"])

    def test_coverage_required(self):
        with self.assertRaises(ValueError):
            B._gt_segments_beats(self.track([1.0, 4], [4, 16], ["A", "B"]))         # not from the first beat
        with self.assertRaises(ValueError):
            B._gt_segments_beats(self.track([0, 4], [4, 12], ["A", "B"]))           # not through the last beat
        with self.assertRaises(ValueError):
            B._gt_segments_beats(self.track([0, 4], [4, 16], ["A"]))
        # beats past the audio's end are absorbed by the last phrase
        bounds, _ = B._gt_segments_beats(self.track([0, 4], [4, 12], ["A", "B"], duration_sec=12.0))
        np.testing.assert_array_equal(bounds, [0, 8, 32])

    def test_nearest_beat(self):
        self.assertEqual(B._nearest_beat_index(self.BEATS, 1.24), 2)
        self.assertEqual(B._nearest_beat_index(self.BEATS, 1.26), 3)
        self.assertEqual(B._nearest_beat_index(self.BEATS, -5.0), 0)
        self.assertEqual(B._nearest_beat_index(self.BEATS, 99.0), 31)


class Recorder:
    """In place of HistGradientBoostingClassifier: keeps what it is fitted on."""
    fitted = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, sample_weight=None):
        self.X, self.y, self.w = X, np.asarray(y), sample_weight
        self.classes_ = np.unique(self.y)
        Recorder.fitted.append(self)
        return self


class BackendFitTests(unittest.TestCase):
    def setUp(self):
        Recorder.fitted = []
        self.enterContext(mock.patch("sklearn.ensemble.HistGradientBoostingClassifier", Recorder))

    def test_boundary_samples_each_boundary_positive_its_neighbours_left_out(self):
        n = 40
        track = {"boundary_feature_z": np.arange(n, dtype=float)[:, None], "valid_mask": np.r_[np.zeros(2, bool),
                                                                                              np.ones(n - 2, bool)]}
        targets = [(np.array([0, 16, 32, n]), ["A", "B", "C"])]
        clf = B._fit_boundary_gbm([track], [0], targets, positive_radius=0, negative_guard=2, seed=3, max_iter=7)
        used = clf.X[:, 0].astype(int)
        self.assertEqual(sorted(used[clf.y == 1].tolist()), [16, 32])
        for left_out in (0, 1, 14, 15, 17, 18, 30, 31, 33, 34):             # invalid, or beside a boundary
            self.assertNotIn(left_out, used)
        self.assertEqual(used.size, n - 2 - 8)
        self.assertAlmostEqual(float(clf.w.mean()), 1.0)                    # weights normalised to their count
        pos, neg = clf.w[clf.y == 1][0], clf.w[clf.y == 0][0]
        self.assertAlmostEqual(pos / neg, (used.size - 2) / 2)               # the classes balanced
        self.assertEqual((clf.params["max_iter"], clf.params["random_state"]), (7, 3))

    def test_boundary_radius_widens_the_positives(self):
        n = 40
        track = {"boundary_feature_z": np.arange(n, dtype=float)[:, None], "valid_mask": np.ones(n, bool)}
        clf = B._fit_boundary_gbm([track], [0], [(np.array([0, 20, n]), ["A", "B"])], positive_radius=1,
                                  negative_guard=1, seed=0)
        self.assertEqual(sorted(clf.X[clf.y == 1, 0].astype(int).tolist()), [19, 20, 21])
        self.assertEqual(clf.X.shape[0], n)

    def test_label_model_transitions_and_lengths(self):
        rng = np.random.default_rng(RNG_SEED)
        tracks = [{"uid": f"u{i}", "feature_z": rng.normal(0, 1, (64, 3)), "label_context_beats": (16, 32)}
                  for i in range(2)]
        targets = [(np.array([0, 16, 48, 64]), ["INTRO", "CHORUS", "OUTRO"]), (np.array([0, 32, 64]), ["INTRO", "CHORUS"])]
        model = B._fit_label_model(tracks, [0, 1], targets, seed=1, boundary_jitter_views=2)
        clf = Recorder.fitted[-1]
        self.assertEqual(clf.X.shape[0], 3 * (3 + 2))                           # each track and its two jittered views
        self.assertEqual(model.labels, ["CHORUS", "INTRO", "OUTRO"])
        np.testing.assert_allclose(np.exp(model.transition).sum(1), 1.0)
        i = {lab: k for k, lab in enumerate(model.labels)}
        start = len(model.labels)
        self.assertEqual(int(np.argmax(model.transition[start])), i["INTRO"])     # songs start with their intro
        self.assertEqual(int(np.argmax(model.transition[i["INTRO"]])), i["CHORUS"])
        self.assertAlmostEqual(model.length_mu[i["CHORUS"]], np.mean(np.log([32.0, 32.0])))
        self.assertEqual(model.length_sigma[i["CHORUS"]], 0.25)                   # the floor
        by_label = {lab: float(clf.w[clf.y == lab].sum()) for lab in model.labels}
        self.assertAlmostEqual(by_label["OUTRO"], by_label["INTRO"])              # every label weighed alike
        with self.assertRaises(ValueError):
            B._fit_label_model(tracks, [0], [(np.array([0, 16, 64]), ["INTRO"])], seed=1)


class MetricTests(unittest.TestCase):
    def test_interior_boundaries_and_bar_labels(self):
        self.assertEqual(R._interior(np.array([0, 8, 16, 32]), 32), [(8, 1), (16, 2)])
        labels = R.bar_labels(np.array([0, 8, 16, 32]), ["A", "B", "C"], np.array([0, 4, 8, 12, 16, 28]))
        self.assertEqual(labels.tolist(), ["A", "A", "B", "B", "C", "C"])
        self.assertEqual(R._matches(np.array([8, 16]), np.array([16, 24])), 1)

    def test_perfect_and_imperfect_pipelines(self):
        prepared = [{"feature_z": np.zeros((32, 2))}, {"feature_z": np.zeros((16, 2))}]
        targets = [(np.array([0, 4, 16, 32]), ["LEAD_IN", "INTRO", "CHORUS"]), (np.array([0, 8, 16]), ["VERSE", "CHORUS"])]
        perfect = R.pipeline_metrics(prepared, targets, targets)
        self.assertEqual(perfect, {"boundary_f1": 1.0, "label_change_f1": 1.0, "key_change_f1": 1.0,
                                   "label_accuracy": 1.0, "lead_in_accuracy": 1.0})
        predictions = [(np.array([0, 16, 32]), ["INTRO", "CHORUS"]), (np.array([0, 8, 16]), ["VERSE", "VERSE"])]
        m = R.pipeline_metrics(prepared, targets, predictions)
        self.assertAlmostEqual(m["boundary_f1"], 2 * 2 / (2 + 3))
        self.assertAlmostEqual(m["label_accuracy"], (28 + 8) / 48)
        self.assertEqual(m["lead_in_accuracy"], 0.5)
        self.assertLess(m["label_change_f1"], 1.0)


def lead_in_case(rng, has, n_cand=5):
    end = int(rng.integers(n_cand)) if has else -1
    cands = np.arange(n_cand) * 16 + 16
    location = rng.normal(0, 0.3, (n_cand, 3))
    if has:
        location[end, 0] += 3.0
    presence = rng.normal(3.0 if has else -3.0, 0.5, 4)
    target = (np.array([0, cands[end], 160, 200]), ["LEAD_IN", "INTRO", "CHORUS"]) if has \
        else (np.array([0, 160, 200]), ["INTRO", "CHORUS"])
    return {"candidates": cands, "location": location, "presence": presence}, target


class LeadInTests(unittest.TestCase):
    def test_targets(self):
        f = {"candidates": np.array([16, 32, 48])}
        self.assertEqual(L.track_lead_in(f, (np.array([0, 32, 64]), ["LEAD_IN", "INTRO"])), (True, 1))
        self.assertEqual(L.track_lead_in(f, (np.array([0, 40, 64]), ["LEAD_IN", "INTRO"])), (True, -1))
        self.assertEqual(L.track_lead_in(f, (np.array([0, 32, 64]), ["INTRO", "VERSE"])), (False, -1))
        self.assertEqual(L.track_lead_in(f, (np.array([0, 64]), ["LEAD_IN"])), (False, -1))

    def test_accuracy(self):
        t = [(np.array([0, 8, 32]), ["LEAD_IN", "A"]), (np.array([0, 32]), ["A"])]
        self.assertEqual(L.lead_in_accuracy(t, t), 1.0)
        self.assertEqual(L.lead_in_accuracy(t, [(np.array([0, 16, 32]), ["LEAD_IN", "A"]), t[1]]), 0.5)
        self.assertEqual(L.lead_in_accuracy([], []), 0.0)
        with self.assertRaises(ValueError):
            L.lead_in_accuracy(t, t[:1])

    def test_models_learn_separable_lead_ins(self):
        rng = np.random.default_rng(RNG_SEED)
        cases = [lead_in_case(rng, has=i % 2 == 0) for i in range(40)]
        features = {i: f for i, (f, _) in enumerate(cases)}
        targets = [t for _, t in cases]
        models, threshold = L.fit_lead_in(features, range(40), targets, seed=1)
        self.assertIsNotNone(models)
        self.assertTrue(0.0 <= threshold <= 1.0)
        right = 0
        for f, (bounds, labels) in cases:
            s = models["presence"].predict_proba(f["presence"][None])[0, 1]
            pick = int(np.argmax(models["location"].predict_proba(f["location"])[:, 1]))
            has = labels[0] == "LEAD_IN"
            right += (s >= threshold) == has and (not has or f["candidates"][pick] == bounds[1])
        self.assertGreaterEqual(right, 38)

    def test_too_few_of_a_kind_fits_nothing(self):
        rng = np.random.default_rng(RNG_SEED)
        cases = [lead_in_case(rng, has=i < L.MIN_TRACKS_PER_CLASS - 1) for i in range(30)]
        self.assertEqual(L.fit_lead_in({i: f for i, (f, _) in enumerate(cases)}, range(30), [t for _, t in cases], 1),
                         (None, 0.5))


class OptimizerTests(unittest.TestCase):
    def test_progress_scaled_into_its_span(self):
        seen = []
        scaled = P._scaled(lambda v, t: seen.append((v, render(t))), 20, 60, "Stage: ")
        scaled(50, "half")
        scaled(100, "done")
        self.assertEqual(seen, [(40, "Stage: half"), (60, "Stage: done")])
        self.assertIsNone(P._scaled(None, 0, 1))
        self.assertIsInstance(Concat(("a", "b")), Concat)

    def test_cache_workers(self):
        self.assertEqual(P._auto_cache_workers(None, 0), 0)
        self.assertEqual(P._auto_cache_workers(9, 2), 2)
        self.assertLessEqual(P._auto_cache_workers(None, 50), 4)

    def test_structure_and_repeat_caches_checked_against_the_beat_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            beat = Path(tmp) / "u.npz"
            beat.write_bytes(b"x")
            path = P._structure_cache_path(Path(tmp), "u")
            np.save(path, np.zeros((10, P.MULTI_VIEW_STRUCTURE_FEATURE_DIM), np.float32))
            self.assertTrue(P._structure_cache_is_current(path, beat, 10))
            self.assertFalse(P._structure_cache_is_current(path, beat, 11))
            np.save(path, np.zeros((10, P.MULTI_VIEW_STRUCTURE_FEATURE_DIM), np.float64))
            self.assertFalse(P._structure_cache_is_current(path, beat, 10))
            with self.assertRaises(ValueError):
                P._load_current_structure_cache(path, beat, 10)
            repeat = P._repeat_cache_path(Path(tmp), "u")
            np.savez(repeat, **{k: np.zeros(10) for k in P._REPEAT_KEYS})
            self.assertTrue(P._repeat_cache_is_current(repeat, beat, 10))
            self.assertFalse(P._repeat_cache_is_current(repeat, beat, 12))
            os.utime(beat, ns=(repeat.stat().st_mtime_ns + 10**9,) * 2)       # the beat cache refreshed since
            self.assertFalse(P._repeat_cache_is_current(repeat, beat, 10))
            self.assertFalse(P._structure_cache_is_current(path, beat, 10))
            self.assertFalse(P._structure_cache_is_current(Path(tmp) / "none.npy", beat, 10))

    def test_too_few_annotated_tracks_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            given = []
            for i in range(5):
                a = F.beat_annotation(duration=20.0)
                fields = {**a.__dict__, "phrase_starts": np.array([0.0, 10.0]), "phrase_ends": np.array([10.0, 20.0]),
                          "phrase_labels": np.array(["INTRO", "VERSE"])}
                if i == 0:
                    fields["tempo_segments"] = None                  # fails its validation
                given.append(F.given_track(Path(tmp), i, Annotation(**fields)))
            F.use_tracks(given)
            try:
                request = P.PhraseOptimizationRequest(library_dir=Path(tmp), cache_dir=Path(tmp) / "phrase",
                                                      output_path=Path(tmp) / "weights" / "phrase_analyzer.npz")
                with self.assertRaises(TrainingError) as caught:
                    P.optimize_phrase_parameters(request)
            finally:
                F.library_tracks._tracks = None
            self.assertIn("(4 valid, 1 ignored)", str(caught.exception))
            self.assertFalse((Path(tmp) / "weights" / "phrase_analyzer.npz").exists())


if __name__ == "__main__":
    unittest.main()
