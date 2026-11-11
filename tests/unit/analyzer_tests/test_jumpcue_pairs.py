"""analyzer.jumpcue's pair model on synthetic tracks (no audio): the candidates, the shipped models reading the app's
features, the one pair chosen under the user's rules (its sections, the value it needs, its gap, none early in the
first round, none wholly on CUE points), the chorus sections, a model file of another format refused, and the entry
function's pair."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyzer.jumpcue import pairs as P
from analyzer.jumpcue.JumpCUE import JumpContext, jump_cue_pairs
from analyzer.jumpcue.vocal_context import VocalContext
from analyzer.settings import JumpCueRules

SHIPPED = str(Path(__file__).resolve().parents[3] / "assets" / "weights" / "default_jumpcue_pairs.npz")
LABELS = ["INTRO", "VERSE", "VERSE", "CHORUS", "CHORUS", "INTERLUDE", "VERSE", "CHORUS", "CHORUS", "BRIDGE", "CHORUS",
          "OUTRO"]
PHRASE = 16
SUNG = {"VERSE", "CHORUS", "BRIDGE"}
# the repeats, (a0, a1, b0, b1, z) in beats (analyzer.jumpcue.repeats): the choruses and the verses
LINKS = [(48, 79, 112, 143, 0.8), (48, 63, 160, 175, 0.7), (112, 127, 160, 175, 0.75), (16, 31, 96, 111, 0.6)]


def synthetic(seed: int = 0):
    """(beats, tempo rows, phrases, the per-beat arrays as pairs.beat_features makes them): a song of LABELS, 16
    beats a phrase at 120 BPM, each label's passages alike (the same prototypes in every song), with noise."""
    proto = np.random.default_rng(1234)
    rng = np.random.default_rng(seed)
    n = len(LABELS) * PHRASE
    beats = 0.5 + 0.5 * np.arange(n)
    rows = np.asarray([[0.0, beats[-1] + 0.5, 120.0, beats[0], 4.0, 120.0]])
    phrases = [(0.0 if i == 0 else float(beats[i * PHRASE]),
                float(beats[(i + 1) * PHRASE]) if (i + 1) * PHRASE < n else float(beats[-1] + 0.5), lab)
               for i, lab in enumerate(LABELS)]
    shapes = {"mel": (96,), "chroma": (12,), "ceps": (8, 12), "level": (4,)}
    protos = {lab: {k: proto.normal(0, 1, (PHRASE, *s)) for k, s in shapes.items()} for lab in sorted(set(LABELS))}

    def per_beat(kind, scale=0.15):
        out = np.stack([protos[LABELS[b // PHRASE]][kind][b % PHRASE] for b in range(n)])
        return out + scale * rng.normal(0, 1, out.shape)
    level = per_beat("level") - 20.0
    sung = np.asarray([LABELS[b // PHRASE] in SUNG for b in range(n)])
    onset = 0.2 * rng.random((n, 4))
    onset[:, 0] += 1.0
    feats = {
        "links": np.asarray(LINKS, np.float64),
        "v_beat_mel": per_beat("mel").astype(np.float16),
        "v_beat_db": level.mean(1).astype(np.float32),
        "v_beat_ceps": per_beat("ceps").astype(np.float16),
        "q_vocal": np.repeat(sung[:, None], 4, axis=1).astype(np.float32),
        "q_level": level.astype(np.float32),
        "q_low": (level - 3.0).astype(np.float32),
        "q_high": (level - 6.0).astype(np.float32),
        "q_flat": (0.1 * rng.random((n, 4))).astype(np.float32),
        "q_onset": onset.astype(np.float32),
        "chroma": np.abs(per_beat("chroma")).astype(np.float32),
    }
    return beats, rows, phrases, feats


def track(seed: int = 0) -> P.PairTrack:
    beats, rows, phrases, feats = synthetic(seed)
    return P.pair_track(beats, rows, phrases, feats)


class CandidateTests(unittest.TestCase):
    def test_links_structure_and_the_ending(self):
        t = track()
        cand = P.candidates(t)
        pairs = set(map(tuple, cand.tolist()))
        self.assertTrue(np.all(cand[:, 1] - cand[:, 0] >= P.MIN_LAG))
        self.assertTrue(np.all(t.bar_phase[cand[:, 0]] == t.bar_phase[cand[:, 1]]))
        self.assertIn((48, 112), pairs)                    # a link's lag at its start
        self.assertIn((40, 104), pairs)                    # LINK_MARGIN beats before it
        self.assertIn((32, 96), pairs)                     # verse to verse, on no link (the phrases' family)
        self.assertIn((160, 176), pairs)                   # a chorus head into the outro's head
        self.assertNotIn((160, 176), set(map(tuple, P.candidates(t, cross_ending=False).tolist())))


ANY = JumpCueRules(min_score=0.0)      # the synthetic scores are low: no value asked for


def brute(t, cand, scores, rules, cue_beats=frozenset()):
    """The pair choose_pair should give, every allowed pair looked at."""
    spans, m = P.chorus_spans(t), rules.section_margin_beats
    start = P.start_beat(t, rules)
    score = {(int(a), int(b)): float(s) for (a, b), s in zip(cand.tolist(), scores)}

    def held(x):
        return {k for k, (s, e) in enumerate(spans) if s - m <= x < e + m}
    ok = {"any": lambda x: {-1}, "chorus": held, "first_chorus": lambda x: held(x) & {0},
          "last_chorus": lambda x: held(x) & {len(spans) - 1}}
    best, best_v = None, 0.0
    for p, q in score:
        if q - p < rules.min_gap_beats or p < start:
            continue
        a, b = P.off_cue(t, p, q, cue_beats, start) if rules.avoid_cue_points else (p, q)
        ka, kb = ok[rules.earlier](a), ok[rules.later](b)
        if not ka or not kb:
            continue
        if rules.earlier != "any" and rules.later != "any" and not any(j > i for i in ka for j in kb):
            continue
        v = score.get((a, b), 0.0)
        if v > best_v or (v == best_v and best is not None and b < best[1]):
            best, best_v = (a, b), v
    return best, best_v


def scored():
    t = track()
    cand = P.candidates(t)
    models = P.load_models(SHIPPED)
    X, names = P.features(t, cand)
    return t, cand, P.pair_scores(X, models), models, names


class ShippedModelTests(unittest.TestCase):
    def test_the_shipped_models_read_the_apps_features(self):
        t, cand, scores, models, names = scored()
        self.assertEqual(names, models.pair_names)
        self.assertTrue(np.all((scores > 0) & (scores < 1)))


class ChooseTests(unittest.TestCase):
    def test_chorus_spans(self):
        self.assertEqual(P.chorus_spans(track()), [(48, 80), (112, 144), (160, 176)])      # phrases 3-4, 7-8, 10

    def test_the_section_margin_takes_a_point_just_before_the_chorus(self):
        t, cand, scores, _models, _names = scored()
        best = {}
        for m in (0, 4):
            got = P.choose_pair(t, cand, scores, dataclasses.replace(ANY, section_margin_beats=m))
            best[m] = got
            p, q, _v = got
            self.assertTrue(48 - m <= p < 80 + m and 160 - m <= q < 176 + m)
        self.assertGreaterEqual(best[4][2], best[0][2])         # a wider margin offers more, never less

    def test_the_start_follows_the_first_singing(self):
        t = track()
        self.assertEqual(P.start_beat(t, JumpCueRules()), PHRASE + 20)     # the first verse's head + 20 beats
        self.assertEqual(P.start_beat(t, JumpCueRules(first_sung_beats=0)), PHRASE)

    def test_the_chosen_pair_is_the_best_the_rules_allow(self):
        t, cand, scores, _models, _names = scored()
        cues = frozenset({48, 160})
        for rules in (ANY, dataclasses.replace(ANY, earlier="chorus", later="chorus"),
                      dataclasses.replace(ANY, earlier="any", later="any"), dataclasses.replace(ANY, min_gap_beats=80),
                      dataclasses.replace(ANY, first_sung_beats=60), dataclasses.replace(ANY, avoid_cue_points=False),
                      dataclasses.replace(ANY, section_margin_beats=0), dataclasses.replace(ANY, section_margin_beats=12)):
            with self.subTest(rules=rules):
                got = P.choose_pair(t, cand, scores, rules, cues)
                want, want_v = brute(t, cand, scores, rules, cues)
                if want is None:                   # the start past the first chorus's candidates: none there
                    self.assertIsNone(got)
                    continue
                self.assertEqual(got[:2], want)
                self.assertAlmostEqual(got[2], want_v)
                p, q, _v = got
                self.assertGreaterEqual(q - p, rules.min_gap_beats)
                self.assertGreaterEqual(p, P.start_beat(t, rules))
        p, q, _v = P.choose_pair(t, cand, scores, dataclasses.replace(ANY, section_margin_beats=0))
        self.assertTrue(48 <= p < 80 and 160 <= q < 176)                     # the first chorus into the last

    def test_none_under_the_value_asked_for(self):
        t, cand, scores, _models, _names = scored()
        _p, _q, v = P.choose_pair(t, cand, scores, ANY)
        self.assertIsNotNone(P.choose_pair(t, cand, scores, dataclasses.replace(ANY, min_score=v)))
        self.assertIsNone(P.choose_pair(t, cand, scores, dataclasses.replace(ANY, min_score=v + 1e-6)))

    def test_a_pair_wholly_on_cue_points_moves_earlier(self):
        t = track()
        self.assertEqual(P.off_cue(t, 64, 128, frozenset({64, 128}), 0), (56, 120))
        self.assertEqual(P.off_cue(t, 64, 128, frozenset({64}), 0), (64, 128))           # partly: kept
        self.assertEqual(P.off_cue(t, 64, 128, frozenset({64, 128, 56}), 0), (64, 128))   # a CUE point there: kept
        self.assertEqual(P.off_cue(t, 64, 128, frozenset({64, 128}), 60), (64, 128))      # before the start: kept

    def test_a_pair_on_cue_points_is_not_chosen_there(self):
        t, cand, scores, _models, _names = scored()
        p, q, _v = P.choose_pair(t, cand, scores, ANY)
        moved = P.choose_pair(t, cand, scores, ANY, frozenset({p, q}))
        self.assertTrue(moved is None or moved[:2] != (p, q))
        self.assertEqual(P.choose_pair(t, cand, scores, dataclasses.replace(ANY, avoid_cue_points=False),
                                       frozenset({p, q}))[:2], (p, q))


class ModelFileTests(unittest.TestCase):
    def test_another_format_or_other_features_refused(self):
        with np.load(SHIPPED, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        with tempfile.TemporaryDirectory() as tmp:
            for name, change in (("format", {"format": np.asarray("mixlyzer_jumpcue_pairs_v1")}),
                                 ("other", {"format": np.asarray("mixlyzer_other")})):
                with self.subTest(name):
                    path = Path(tmp) / f"{name}.npz"
                    np.savez(path, **{**arrays, **change})
                    with self.assertRaises(ValueError):
                        P.load_models(str(path))


class EntryTests(unittest.TestCase):
    def test_nothing_repeats_no_pairs(self):
        self.assertEqual(jump_cue_pairs(JumpContext([], None), model_path=SHIPPED, beats_time=np.arange(64) * 0.5,
                                        tempo_segments=np.zeros((0, 6)), phrases=[], frames=None,
                                        key_frames=np.zeros((36, 0))), [])

    def test_the_pair_chosen(self):
        beats, rows, phrases, feats = synthetic()
        n_frames = int((beats[-1] + 1.0) / (256 / 22050))
        ft = np.arange(n_frames) * (256 / 22050)
        rng = np.random.default_rng(0)
        frames = SimpleNamespace(rms_db=(-20 + rng.random(n_frames)).astype(np.float32), frame_times=ft,
                                 band_ratios=rng.random((3, n_frames)).astype(np.float32),
                                 flatness=rng.random(n_frames).astype(np.float32),
                                 librosa_onset=rng.random(n_frames).astype(np.float32))
        key_frames = rng.random((36, int((beats[-1] + 1.0) / P.KEY_FRAME_SEC))).astype(np.float32)
        vft = np.arange(n_frames) * (512 / 22050)
        vocal = VocalContext(frame_times=vft, gap_activity=rng.random(n_frames) > 0.5,
                             beat_mel=feats["v_beat_mel"].astype(np.float32), beat_db=feats["v_beat_db"],
                             beat_ceps=feats["v_beat_ceps"].astype(np.float32))
        links = np.asarray(LINKS, np.float64)
        out = jump_cue_pairs(JumpContext(links, vocal), model_path=SHIPPED, beats_time=beats, tempo_segments=rows,
                             phrases=phrases, frames=frames, key_frames=key_frames, rules=ANY,
                             cue_times=[float(beats[48])])
        self.assertEqual(len(out), 1)                          # one pair
        pair = out[0]
        self.assertEqual((pair["backward"]["label"], pair["forward"]["label"]), ("A", "B"))
        for side in ("forward", "backward"):
            p = pair[side]
            self.assertEqual(p["start"], p["point"])
            self.assertGreater(p["end"], p["start"])
        self.assertEqual(pair["lag_sec"], pair["forward"]["point"] - pair["backward"]["point"])
        self.assertGreater(pair["lag_beats"], 0)
        self.assertTrue(0 < pair["score"] < 1 and pair["confidence"] == pair["score"])
        none = jump_cue_pairs(JumpContext(links, vocal), model_path=SHIPPED, beats_time=beats, tempo_segments=rows,
                              phrases=phrases, frames=frames, key_frames=key_frames, rules=JumpCueRules(min_score=1.0))
        self.assertEqual(none, [])                             # the value asked for not reached


if __name__ == "__main__":
    unittest.main()
