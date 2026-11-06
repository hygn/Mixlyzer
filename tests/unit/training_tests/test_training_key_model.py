"""training.key_model_optimizer: a member's torch network against the analyzer's NumPy inference of its exported
arrays (the chord posteriors and the key evidence), its root equivariance, the loss against the bars' key
cross-entropy and the template-chord term computed with the analyzer's model, the optimizer's step (the chord-in-key
tables trained with the weights) and the learning-rate schedule, which keys train it (the user's annotations: edited,
legacy, still automatic, raw-looking; or a key GT file), the templates of every chord quality, the training songs and
crops, the new members' weights, and small runs of the whole optimizer on synthetic tracks on the CPU: the model
written where asked with its members and read by the analyzer's own loader, the decoder settings kept, the refusals,
and the held-out check that keeps worse keys from being written. The torch parts need the training group (skipped
without it)."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.key import chord_key as CK
from training import key_model_optimizer as O
from training import model_start as MS
from training.library_tracks import Annotation
from training.text import TrainingError, render

from unit.training_tests import training_fixtures as F

RNG_SEED = 20261027
SHIPPED_KEY = F.shipped("key_chord_cnn.npz")
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "torch is the training group's (uv sync --group training)")
Q = len(CK.QUALITIES)


def beat_inputs(rng, n):
    """[n, 36] beat inputs shaped as chord_key.pool_beats makes them (sqrt, each block over its max)."""
    x = np.sqrt(rng.random((n, 36)))
    return np.concatenate([b / b.max(axis=1, keepdims=True) for b in np.split(x, 3, axis=1)], axis=1).astype(np.float32)


def transposed(a, s, blocks):
    """Every pitch class (each 12-block) up ``s`` semitones."""
    return np.concatenate([np.roll(a[..., 12 * b:12 * (b + 1)], s, axis=-1) for b in range(blocks)], axis=-1)


def a_member(seed=3, qualities=Q):
    """A new member with the shipped model's layer sizes, its weights sharper than a new model's and its tables not
    flat (a trained member's shape)."""
    like = O.member_of(CK.read_model_arrays(SHIPPED_KEY), 0)
    m = O._init_weights(like, qualities, seed)
    rng = np.random.default_rng(seed)
    for k in ("c1_w", "c2_w", "out_w"):
        m[k] = 4.0 * m[k]
    m["tab"] = rng.normal(0, 1, m["tab"].shape).astype(np.float32)
    return m


def tiny_batch(rng, B=2, n=12, qualities=Q):
    x = np.stack([beat_inputs(rng, n) for _ in range(B)])
    y = rng.integers(0, 24, (B, n))
    w = (rng.random((B, n)) < 0.8).astype(np.float32)
    bar = np.tile(np.arange(n) // 4, (B, 1))
    pos = rng.integers(0, 3, (B, n))
    return x, y, w, bar, pos, O.template_chords(x.reshape(-1, 36), qualities).reshape(B, n, 12 * qualities)


def numpy_loss(arrays, batch, distill):
    """The training loss with the analyzer's model: each bar's key evidence (the sum of its keyed beats') against the
    share of each key among them, averaged over the bars with a keyed beat, plus ``distill`` times KL(templates ||
    chords) averaged over every beat of the crops."""
    model = CK.ChordKeyModel(arrays=arrays)
    x, y, w, bar, pos, teach = batch
    ce, kl = [], []
    for b in range(x.shape[0]):
        logpc = model.chords(x[b])
        L = model.key_loglik(logpc, pos[b])
        for j in np.unique(bar[b]):
            sel = bar[b] == j
            if w[b][sel].sum() > 0:
                target = np.bincount(y[b][sel], weights=w[b][sel], minlength=24) / w[b][sel].sum()
                ce.append(-(target * CK._log_softmax((L[sel] * w[b][sel, None]).sum(0))).sum())
        kl.extend((teach[b] * (np.log(teach[b] + 1e-8) - logpc[0])).sum(-1))
    return float(np.mean(ce) + distill * np.mean(kl))


@needs_torch
class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.arrays = CK.read_model_arrays(SHIPPED_KEY)
        self.rng = np.random.default_rng(RNG_SEED)

    def test_state_dict_and_export_are_a_members_arrays(self):
        member = O.member_of(self.arrays, 0)
        net = O.network(member)
        self.assertEqual(list(net.state_dict()), list(CK.WEIGHTS))
        P = O.export(net)
        self.assertEqual({k: (v.shape, v.dtype) for k, v in P.items()},
                         {k: (v.shape, np.dtype(np.float32)) for k, v in member.items()})
        for k in P:
            np.testing.assert_array_equal(P[k], member[k])

    def test_exported_member_is_the_analyzers_model(self):
        import torch
        for qualities in (2, Q):
            with self.subTest(qualities=qualities):
                member = a_member(qualities=qualities)
                net = O.network(member)
                P = O.export(net)
                x, pos = beat_inputs(self.rng, 600), self.rng.integers(0, 3, 600)
                self.assertLess(O.check_export(net, P, self.arrays, x, pos), 1e-5)
                model = CK.ChordKeyModel(arrays=O.stacked([P], self.arrays))
                with torch.no_grad():
                    logpc, L = net(torch.as_tensor(x)[None], torch.as_tensor(pos)[None])
                np.testing.assert_allclose(logpc[0].numpy(), model.chords(x)[0], atol=1e-4)
                np.testing.assert_allclose(L[0].numpy(), model.key_loglik(model.chords(x), pos), atol=1e-4)

    def test_the_models_evidence_is_its_members_mean(self):
        members = [a_member(seed) for seed in (1, 2, 3)]
        model = CK.ChordKeyModel(arrays=O.stacked(members, self.arrays))
        x, pos = beat_inputs(self.rng, 50), self.rng.integers(0, 3, 50)
        alone = [CK.ChordKeyModel(arrays=O.stacked([m], self.arrays)) for m in members]
        np.testing.assert_allclose(model.key_loglik(model.chords(x), pos),
                                   np.mean([a.key_loglik(a.chords(x), pos) for a in alone], axis=0), atol=1e-5)
        p, top = model.beat_chords(x)
        np.testing.assert_allclose(p, np.mean([np.exp(a.chords(x)[0]) for a in alone], axis=0), atol=1e-6)
        np.testing.assert_array_equal(top, np.argmax(p, axis=1))

    def test_transposed_input_transposes_the_chords_and_keys(self):
        import torch
        net = O.network(a_member())
        x, pos = beat_inputs(self.rng, 80), torch.as_tensor(self.rng.integers(0, 3, 80))[None]
        with torch.no_grad():
            logpc, L = net(torch.as_tensor(x)[None], pos)
            for s in (1, 5, 11):
                moved, moved_keys = net(torch.as_tensor(transposed(x, s, 3))[None], pos)
                np.testing.assert_allclose(moved.numpy(), transposed(logpc.numpy(), s, Q), atol=1e-5)
                np.testing.assert_allclose(moved_keys.numpy(), transposed(L.numpy(), s, 2), atol=1e-5)

    def test_loss_is_the_bars_keys_and_the_template_term(self):
        member = a_member()
        arrays = O.stacked([member], self.arrays)
        batch = tiny_batch(self.rng, B=3, n=20)
        loss = O.key_loss(O.network(member), batch, 2.0)
        self.assertAlmostEqual(float(loss.detach()), numpy_loss(arrays, batch, 2.0), places=5)
        no_template = O.key_loss(O.network(member), batch, 0.0)
        self.assertAlmostEqual(float(no_template.detach()), numpy_loss(arrays, batch, 0.0), places=5)

    def test_no_keyed_beat_no_loss(self):
        x, y, w, bar, pos, teach = tiny_batch(self.rng)
        self.assertIsNone(O.key_loss(O.network(a_member()), (x, y, 0 * w, bar, pos, teach), 2.0))

    def test_one_step_is_adam_with_l2_on_every_weight_and_the_tables(self):
        net = O.network(a_member())
        opt = O.optimizer(net, 1e-3)
        O.key_loss(net, tiny_batch(self.rng), 2.0).backward()
        before = {k: getattr(net, k).detach().clone().numpy() for k in O.TRAINED}
        grads = {k: getattr(net, k).grad.clone().numpy() for k in O.TRAINED}
        self.assertIn("tab", O.TRAINED)
        opt.step()
        for k in O.TRAINED:
            g = grads[k] + O.WEIGHT_DECAY * before[k]          # the first step: m / (1 - b1) = g, v / (1 - b2) = g^2
            np.testing.assert_allclose(getattr(net, k).detach().numpy(), before[k] - 1e-3 * g / (np.abs(g) + 1e-8),
                                       atol=1e-7, err_msg=k)


class ModelFileTests(unittest.TestCase):
    def test_the_shipped_model_reads(self):
        arrays = CK.read_model_arrays(SHIPPED_KEY)
        model = CK.ChordKeyModel(arrays=arrays)
        self.assertGreaterEqual(len(model.members), 1)
        self.assertIn(model.Q, range(2, Q + 1))
        x = beat_inputs(np.random.default_rng(0), 30)
        self.assertEqual(model.chords(x).shape, (len(model.members), 30, 12 * model.Q))
        self.assertEqual(model.key_loglik(model.chords(x), np.zeros(30, int)).shape, (30, 24))

    def test_inconsistent_arrays_are_refused(self):
        good = CK.read_model_arrays(SHIPPED_KEY)
        bad = {"no member axis": {**good, "c1_w": good["c1_w"][0]},
               "tables of another vocabulary": {**good, "tab": good["tab"][..., :12]},
               "members disagree": {**good, "c2_b": good["c2_b"][:0]},
               "too many qualities": {**good, "out_w": np.concatenate([good["out_w"]] * 4, axis=1)},
               "a non-finite weight": {**good, "c1_b": np.full_like(good["c1_b"], np.nan)}}
        for name, arrays in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                CK.check_model_arrays(arrays)

    def test_chord_names(self):
        self.assertEqual([CK.chord_name(i) for i in (0, 12 + 9, 24 + 7, 36 + 5, 48 + 2)],
                         ["C", "Am", "G7", "FM7", "Dm7"])

    def test_a_fifths_change_costs_j_fifth_times_more(self):
        """32 bars of 4 beats: C major's evidence, then another key's by the same margin a beat (64 in all). A change
        costing 40 is taken; a dear J_FIFTH refuses the change to G major (a fifth) and still takes A major's (a
        minor third's signature). (Two changes through a key between, 80 here, would be cheaper than a dear fifth's
        when the costs are low: J_FIFTH makes the direct change dearer, not every way there.)"""
        arrays = CK.read_model_arrays(SHIPPED_KEY)
        nb = 128
        grid = CK.BeatGrid(beat_t=np.arange(nb) * 0.5, bar=np.arange(nb) // 4, phrase_first=np.zeros(32, bool),
                           pos=np.full(nb, 2))
        for other, fifth in ((7, True), (9, False)):
            L = np.zeros((nb, 24))
            L[:64, 0] = 1.0
            L[64:, other] = 1.0
            for j_fifth in (1.0, 1000.0):
                model = CK.ChordKeyModel(arrays={**arrays, "J": np.asarray(40.0), "JX": np.asarray(0.0),
                                                 "LAM": np.asarray(0.1), "J_FIFTH": np.asarray(j_fifth)})
                _, keys = model.decode(L, grid, 64.0)
                changed = keys[-1] == other
                with self.subTest(other=other, j_fifth=j_fifth):
                    self.assertEqual(changed, not (fifth and j_fifth > 1.0))


class CostTests(unittest.TestCase):
    """The change costs as asked (the model in use's times cost_scale), J_FIFTH as asked or kept, and the evidence's
    strength measured: a model whose evidence is twice as strong reads 2."""

    def songs(self, n=3, nb=40):
        rng = np.random.default_rng(RNG_SEED)
        return [O.Song(x=beat_inputs(rng, nb), y=np.zeros(nb, np.int64), w=np.ones(nb, np.float32),
                       bar=np.arange(nb) // 4, pos=np.full(nb, 2), teach=np.zeros((nb, 12 * Q), np.float32), grid=None,
                       duration=float(nb)) for _ in range(n)]

    def test_the_evidences_strength(self):
        arrays = CK.read_model_arrays(SHIPPED_KEY)
        songs = self.songs()
        once = O.evidence_scale(arrays, songs, range(3))
        real = CK.ChordKeyModel.key_loglik
        with mock.patch.object(CK.ChordKeyModel, "key_loglik", lambda m, lp, pos: 2.0 * real(m, lp, pos)):
            twice = O.evidence_scale(arrays, songs, range(3))
        self.assertGreater(once, 0.0)
        self.assertAlmostEqual(twice / once, 2.0, places=5)

    def test_costs_and_j_fifth_as_asked(self):
        reference = CK.read_model_arrays(SHIPPED_KEY)
        out = O.with_costs(reference, reference, 1.0)
        for k in CK.SETTINGS:
            np.testing.assert_array_equal(out[k], reference[k])
        out = O.with_costs(reference, reference, 0.5, fifth_cost=3.5)
        self.assertEqual(float(out["J_FIFTH"]), 3.5)
        for k in O.COSTS:
            self.assertAlmostEqual(float(out[k]), 0.5 * float(reference[k]), places=5)
        for k in set(CK.SETTINGS) - set(O.COSTS) - {"J_FIFTH"}:
            np.testing.assert_array_equal(out[k], reference[k])


class ScheduleTests(unittest.TestCase):
    def test_tuning_decays_from_the_first_step(self):
        steps = 30
        f = np.array([O._rate(s, steps, 0.0) for s in range(1, steps + 1)])
        self.assertEqual(f[0], 1.0)
        np.testing.assert_allclose(f, 0.5 * (1 + np.cos(np.pi * np.arange(steps) / steps)))

    def test_from_scratch_warmed_up_then_decayed(self):
        steps = 100
        f = np.array([O._rate(s, steps, O.SCRATCH_WARMUP) for s in range(1, steps + 1)])
        self.assertTrue(np.all(np.diff(f[:30]) > 0))
        self.assertAlmostEqual(f[29], 1.0)
        self.assertTrue(np.all(np.diff(f[30:]) < 0))
        self.assertLess(f[-1], 0.01)

    def test_steps_are_epochs_of_batches(self):
        self.assertEqual(O.steps_of(33, 15), 15 * 3)
        self.assertEqual(O.steps_of(3, 2), 2)


class TemplateTests(unittest.TestCase):
    def beat(self, bass, notes):
        x = np.zeros((1, 36), np.float32)
        x[0, bass] = 1.0
        x[0, [12 + n for n in notes]] = 1.0
        return x

    def test_chords_named_by_their_templates(self):
        cases = {(0, (0, 4, 7)): 0, (9, (9, 0, 4)): 12 + 9, (7, (7, 11, 2)): 7,             # C, Am, G
                 (7, (7, 11, 2, 5)): 24 + 7, (5, (5, 9, 0, 4)): 36 + 5, (2, (2, 5, 9, 0)): 48 + 2}  # G7, FM7, Dm7
        for (bass, notes), index in cases.items():
            with self.subTest(chord=CK.chord_name(index)):
                self.assertEqual(int(np.argmax(O.template_chords(self.beat(bass, notes))[0])), index)
        triads = O.template_chords(self.beat(0, (0, 4, 7)), 2)
        self.assertEqual(triads.shape, (1, 24))
        p = O.template_chords(np.random.default_rng(RNG_SEED).random((50, 36)))
        np.testing.assert_allclose(p.sum(1), 1.0, rtol=1e-5)

    def test_silent_beat_is_finite(self):
        self.assertTrue(np.all(np.isfinite(O.template_chords(np.zeros((3, 36))))))


def keys(*rows):
    return np.asarray(rows, float)


class AnnotationTests(unittest.TestCase):
    def test_rows_sorted_by_start_and_cut_to_four_columns(self):
        r = O._rows([[2, 0, 50, 90, 7], [0, 1, 0, 50, 7]])
        np.testing.assert_array_equal(r, [[0, 1, 0, 50], [2, 0, 50, 90]])
        self.assertEqual(O._rows([]).shape, (0, 4))

    def test_same_keys_tolerates_float32_times_only(self):
        a = keys([0, 1, 0.0, 50.0], [2, 0, 50.0, 90.0])
        self.assertTrue(O._same_keys(a, a + [0, 0, 5e-4, -5e-4]))
        self.assertFalse(O._same_keys(a, a + [0, 0, 0.01, 0]))
        self.assertFalse(O._same_keys(a, a + [1, 0, 0, 0]))
        self.assertFalse(O._same_keys(a, a[:1]))

    def test_raw_looking(self):
        clean = keys([0, 0, 0, 60], [7, 0, 60, 120], [0, 0, 120, 180])
        self.assertFalse(O.raw_looking(clean))
        many = keys(*[[i % 2, 0, 10 * i, 10 * i + 10] for i in range(10)])
        self.assertTrue(O.raw_looking(many))
        fragments = keys([0, 0, 0, 60], [0, 0, 60, 62], [0, 0, 62, 63], [5, 0, 63, 120])
        self.assertTrue(O.raw_looking(fragments))                # short fragments beside the same key

    def listed(self, cases, key_gt=None):
        with tempfile.TemporaryDirectory() as tmp:
            given = []
            for i, (name, values) in enumerate(cases.items()):
                a = F.beat_annotation(duration=20.0)
                fields = {**a.__dict__, "phrase_starts": np.array([0.25, 8.25]), **values}
                given.append(F.given_track(Path(tmp), i, Annotation(**fields), title=name))
            F.use_tracks(given)
            try:
                ignored = []
                gt = None if key_gt is None else {given[i]["uid"]: rows for i, rows in key_gt.items()}
                return O.list_key_tracks(Path(tmp), ignored, gt), ignored
            finally:
                F.library_tracks._tracks = None

    def test_the_users_keys_are_the_training_songs(self):
        auto = keys([0, 0, 0, 20])
        cases = {
            "edited": dict(key_segments=keys([9, 1, 0, 20]), key_segments_auto=auto),
            "still_auto": dict(key_segments=auto.copy(), key_segments_auto=auto),
            "legacy": dict(key_segments=keys([2, 0, 0, 10], [9, 1, 10, 20])),
            "raw": dict(key_segments=keys(*[[i % 3, 0, 2 * i, 2 * i + 2] for i in range(10)])),
            "no_keys": dict(key_segments=None),
            "no_tempo": dict(key_segments=keys([0, 0, 0, 20]), tempo_segments=None),
        }
        listed, ignored = self.listed(cases)
        self.assertEqual([(t["title"], t["kind"]) for t in listed], [("edited", "edited"), ("legacy", "legacy")])
        reasons = {s.title: render(s.reason) for s in ignored}
        self.assertEqual(set(reasons), {"no_keys", "no_tempo"})
        self.assertEqual(reasons["no_keys"], "no key annotation")
        self.assertIn("no tempo rows", reasons["no_tempo"])
        np.testing.assert_array_equal(listed[0]["phrase_starts"], [0.25, 8.25])

    def test_a_key_gt_names_the_songs_and_their_keys(self):
        auto = keys([0, 0, 0, 20])
        cases = {"still_auto": dict(key_segments=auto.copy(), key_segments_auto=auto),       # in the GT: used
                 "edited": dict(key_segments=keys([9, 1, 0, 20]), key_segments_auto=auto),  # not in it: left out
                 "no_keys": dict(key_segments=None)}                                         # in it: used
        gt = {0: keys([7, 0, 2.0, 12.0]), 2: keys([4, 1, 0.0, 20.0])}
        listed, ignored = self.listed(cases, gt)
        self.assertEqual([(t["title"], t["kind"]) for t in listed], [("still_auto", "gt"), ("no_keys", "gt")])
        np.testing.assert_array_equal(listed[0]["keys"], gt[0])
        self.assertEqual(ignored, [])

    def test_a_key_gt_file_read_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keys.json"
            path.write_text(json.dumps({"tracks": {"u1": [[9, 1, 10.0, 20.0], [0, 0, 0.0, 10.0]], "u2": []}}),
                            encoding="utf-8")
            gt = O.read_key_gt(path)
            np.testing.assert_array_equal(gt["u1"], [[0, 0, 0, 10], [9, 1, 10, 20]])
            self.assertEqual(gt["u2"].shape, (0, 4))
            for rows in ([[12, 0, 0, 1]], [[0, 2, 0, 1]], [[0, 0, 5, 5]], [[0.5, 0, 0, 1]]):
                path.write_text(json.dumps({"tracks": {"u": rows}}), encoding="utf-8")
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    O.read_key_gt(path)
            path.write_text(json.dumps([1, 2]), encoding="utf-8")
            with self.assertRaises(ValueError):
                O.read_key_gt(path)


def track_for(beats, rows, duration, ks, phrase_starts=(0.0,)):
    return {"beats": beats, "segments": rows, "phrase_starts": np.asarray(phrase_starts, float),
            "duration": duration, "keys": ks}


def key_frames(duration, rng):
    return rng.random((36, int(duration * CK.SR / CK.HOP) + 1)).astype(np.float32)


class SongTests(unittest.TestCase):
    def test_beats_labelled_by_the_key_they_fall_in(self):
        rng = np.random.default_rng(RNG_SEED)
        duration = 20.0
        beats = F.steady_beats(duration, 120.0)
        ks = keys([2, 0, 0.0, 10.0], [9, 1, 10.0, 15.0])            # D major, then A minor, then no key
        s = O.song_of(track_for(beats, F.tempo_rows(beats, duration, 120.0), duration, ks), key_frames(duration, rng))
        self.assertEqual(s.x.shape, (beats.size, 36))
        first, second, rest = beats < 10.0, (beats >= 10.0) & (beats < 15.0), beats >= 15.0
        self.assertTrue(np.all(s.y[first] == 2) and np.all(s.y[second] == 12 + 9))
        np.testing.assert_array_equal(s.w, (beats < 15.0).astype(np.float32))
        self.assertTrue(np.all(s.w[rest] == 0))
        np.testing.assert_array_equal(s.bar, np.arange(beats.size) // 4)
        self.assertEqual(s.teach.shape, (beats.size, 12 * Q))

    def test_too_few_beats_is_no_song(self):
        beats = np.arange(10) * 0.5
        self.assertIsNone(O.song_of(track_for(beats, F.tempo_rows(beats, 6.0, 120.0), 6.0, keys([0, 0, 0, 6])),
                                    key_frames(6.0, np.random.default_rng(0))))

    def test_crops_padded_with_unkeyed_beats(self):
        rng = np.random.default_rng(RNG_SEED)
        duration = 30.0
        beats = F.steady_beats(duration, 120.0)
        s = O.song_of(track_for(beats, F.tempo_rows(beats, duration, 120.0), duration, keys([0, 0, 0, 30])),
                      key_frames(duration, rng))
        x, y, w, bar, pos, teach = O._batch([s, s], [0, 1], rng)
        n = beats.size
        self.assertEqual(x.shape, (2, O.CROP, 36))
        self.assertTrue(np.all(w[:, :n] == 1) and np.all(w[:, n:] == 0))
        self.assertTrue(np.all(pos[:, n:] == 2))
        self.assertTrue(np.all(bar[:, n:] == bar[:, n - 1:n] + 1))     # the padding is a bar of its own
        self.assertTrue(np.all(teach[:, n:] == 0))

    def test_new_members_within_their_fan_in_bounds_with_flat_tables(self):
        like = O.member_of(CK.read_model_arrays(SHIPPED_KEY), 0)
        for qualities in (2, Q):
            P = O._init_weights(like, qualities, 0)
            self.assertEqual(set(P), set(CK.WEIGHTS))
            self.assertEqual(P["out_w"].shape, (qualities,) + like["out_w"].shape[1:])
            self.assertEqual(P["tab"].shape, (2, 3, 12 * qualities))
            self.assertFalse(np.any(P["tab"]))
            for k in ("c1_w", "c1_b", "c2_w", "c2_b", "out_w", "out_b"):
                fan_in = int(np.prod(P[k.replace("_b", "_w")].shape[1:]))
                self.assertLessEqual(float(np.abs(P[k]).max()), 1 / np.sqrt(fan_in))
        self.assertFalse(np.array_equal(O._init_weights(like, Q, 0)["c1_w"], O._init_weights(like, Q, 1)["c1_w"]))


SMALL = 4


class _Library(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.cache.mkdir()
        self.out = self.dir / "weights" / "key_chord_cnn.npz"
        self.enterContext(mock.patch.object(O, "spawn_pool", F.no_pool))

    def tearDown(self):
        F.library_tracks._tracks = None
        self._tmp.cleanup()

    def tracks(self, n, frames=True):
        rng = np.random.default_rng(RNG_SEED)
        tracks = []
        for i in range(n):
            duration = 24.0
            a = F.beat_annotation(duration=duration, bpm=120.0)
            a = Annotation(**{**a.__dict__, "phrase_starts": np.array([0.25]),
                              "key_segments": keys([i % 12, 0, 0, 12], [(i + 7) % 12, 1, 12, duration]),
                              "key_segments_auto": keys([0, 0, 0, duration])})
            t = F.given_track(self.dir, i, a)
            tracks.append(t)
            if frames:
                np.savez_compressed(self.cache / f"{t['uid']}.npz", cache_format=np.asarray(O.KEY_FRAME_CACHE_FORMAT),
                                    audio_mtime_ns=np.int64(t["audio_path"].stat().st_mtime_ns),
                                    frames=key_frames(duration, rng).astype(np.float16))
        F.use_tracks(tracks)
        return tracks

    def request(self, **kw):
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, start=MS.SHIPPED,
                      current_path=SHIPPED_KEY, start_path=SHIPPED_KEY, epochs=1, device="cpu")
        values.update(kw)
        return O.KeyModelOptimizationRequest(**values)

    def small(self):
        self.enterContext(mock.patch.dict(MS.MIN_TRACKS, {"fine_tune": (SMALL, SMALL), "key_scratch": (SMALL, SMALL)}))
        return self.tracks(SMALL)


class RefusalTests(_Library):
    """Refused before any training (no torch needed)."""

    def test_too_few_checked_songs_refused(self):
        self.tracks(MS.MIN_TRACKS["fine_tune"][0] - 1, frames=False)
        with mock.patch.object(O, "ensure_key_frames", side_effect=AssertionError("no features for a refused run")):
            with self.assertRaises(TrainingError) as caught:
                O.optimize_key_model(self.request())
        self.assertIn("songs with checked key annotations", str(caught.exception))
        self.assertFalse(self.out.exists())

    def test_from_scratch_needs_the_key_models_own_minimum(self):
        self.assertLess(MS.MIN_TRACKS["key_scratch"][0], MS.MIN_TRACKS["scratch"][0])
        self.tracks(MS.MIN_TRACKS["key_scratch"][0] - 1, frames=False)
        with mock.patch.object(O, "ensure_key_frames", side_effect=AssertionError("no features for a refused run")):
            with self.assertRaises(TrainingError) as caught:
                O.optimize_key_model(self.request(start=MS.SCRATCH, start_path=None))
        self.assertIn(f"at least {MS.MIN_TRACKS['key_scratch'][0]}", str(caught.exception))

    def test_songs_without_key_frames_count_against_the_minimum(self):
        tracks = self.small()
        (self.cache / f"{tracks[0]['uid']}.npz").unlink()
        with mock.patch.object(O, "ensure_key_frames", lambda tracks, *a, **k: tracks):
            with self.assertRaises(TrainingError):
                O.optimize_key_model(self.request())
        self.assertFalse(self.out.exists())

    def test_stale_key_frames_are_rebuilt(self):
        tracks = self.small()
        path = self.cache / f"{tracks[0]['uid']}.npz"
        self.assertIsNotNone(O.read_key_frames(path, tracks[0]))
        tracks[0]["audio_path"].write_bytes(b"RIFF and more")       # the audio changed since
        os.utime(tracks[0]["audio_path"], ns=(1, 1))
        self.assertIsNone(O.read_key_frames(path, tracks[0]))
        path.write_bytes(b"corrupt")
        self.assertIsNone(O.read_key_frames(path, tracks[0]))

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            O.optimize_key_model(self.request(output_path=self.dir / "key.json"))

    def test_a_vocabulary_out_of_range_is_refused(self):
        for qualities in (1, Q + 1):
            with self.subTest(qualities=qualities), self.assertRaises(ValueError):
                O.optimize_key_model(self.request(start=MS.SCRATCH, start_path=None, qualities=qualities))

    def test_costs_out_of_range_are_refused(self):
        for kw in (dict(fifth_cost=-1.0), dict(cost_scale=0.0)):
            with self.subTest(**kw), self.assertRaises(ValueError):
                O.optimize_key_model(self.request(**kw))


@needs_torch
class RunTests(_Library):
    def test_retrained_model_written_and_read_by_the_analyzers_loader(self):
        self.small()
        progress = []
        result = O.optimize_key_model(self.request(), optimize_progress=lambda v, t: progress.append(v))
        self.assertTrue(result.saved)
        shipped = CK.read_model_arrays(SHIPPED_KEY)
        members = shipped["c1_w"].shape[0]
        self.assertEqual((result.track_count, result.edited_count, result.legacy_count, result.gt_count),
                         (SMALL, SMALL, 0, 0))
        self.assertEqual((result.steps, result.device, result.members), (members, "cpu", members))
        self.assertTrue(np.isfinite(result.seconds_per_step))
        arrays = CK.read_model_arrays(self.out)
        with np.load(self.out, allow_pickle=False) as z:
            self.assertEqual(set(z.files), set(CK.WEIGHTS + CK.SETTINGS))
        for k in CK.WEIGHTS + CK.SETTINGS:                   # improving a model keeps its members and vocabulary
            self.assertEqual(arrays[k].shape, shipped[k].shape)
        for k in CK.SETTINGS:                                # the decoder's settings kept (cost_scale 1)
            np.testing.assert_array_equal(arrays[k], shipped[k])
        self.assertTrue(np.isfinite(result.evidence_ratio) and result.evidence_ratio > 0)
        self.assertFalse(np.array_equal(arrays["c1_w"], shipped["c1_w"]))
        self.assertFalse(np.array_equal(arrays["tab"], shipped["tab"]))      # the tables trained too
        model = CK.ChordKeyModel(self.out)
        self.assertTrue(np.all(np.isfinite(model.chords(np.random.default_rng(0).random((20, 36), np.float32)))))
        self.assertTrue(0.0 <= result.accuracy_before <= 1.0 and 0.0 <= result.accuracy_after <= 1.0)
        self.assertEqual(progress[-1], 100)

    def test_progress_texts_are_marked_for_translation(self):
        self.small()
        texts = []
        O.optimize_key_model(self.request(), feature_progress=lambda v, t: texts.append(t),
                             optimize_progress=lambda v, t: texts.append(t))
        self.assertEqual([w for t in texts for w in F.plain_words(t)], [])

    def test_from_scratch_members_and_vocabulary_as_asked(self):
        self.small()
        result = O.optimize_key_model(self.request(start=MS.SCRATCH, start_path=None, members=2, qualities=3))
        self.assertTrue(result.saved and self.out.is_file())
        self.assertEqual((result.epochs, result.members, result.qualities), (1, 2, 3))
        arrays = CK.read_model_arrays(self.out)
        self.assertEqual(arrays["out_w"].shape[:2], (2, 3))
        self.assertEqual(arrays["tab"].shape, (2, 2, 3, 36))
        like = O.member_of(CK.read_model_arrays(SHIPPED_KEY), 0)
        for m in range(2):                                   # one step from each member's own new weights
            new = O._init_weights(like, 3, m)
            self.assertLess(float(np.abs(arrays["c1_w"][m] - new["c1_w"]).max()), 0.05)

    def test_a_key_gt_trains_on_its_keys(self):
        tracks = self.small()
        gt = self.dir / "keys.json"
        gt.write_text(json.dumps({"tracks": {t["uid"]: [[5, 0, 0.0, 24.0]] for t in tracks}}), encoding="utf-8")
        result = O.optimize_key_model(self.request(key_gt=str(gt)))
        self.assertTrue(result.saved)
        self.assertEqual((result.track_count, result.gt_count, result.edited_count), (SMALL, SMALL, 0))

    def test_worse_held_out_keys_leave_the_model_unwritten(self):
        self.small()
        with mock.patch.object(O, "evaluate", side_effect=[(0.9, 0.0), (0.5, 0.1)]):
            result = O.optimize_key_model(self.request(cv_folds=2))
        self.assertFalse(result.saved or self.out.exists())
        self.assertTrue(result.cross_validated)
        self.assertEqual((result.accuracy_before, result.accuracy_after, result.eval_songs), (0.9, 0.5, SMALL // 2))

    def test_an_export_unlike_the_trained_net_is_not_written(self):
        self.small()
        with mock.patch.object(O, "check_export", return_value=1.0):
            with self.assertRaises(RuntimeError):
                O.optimize_key_model(self.request())
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
