"""The phrase model's stack (analyzer.phrase.stack, dstcn, phrase_analyzer's stacked artifact): the DS-TCN's NumPy
operations against plain loops (PyTorch's zero-padded cross-correlation, GroupNorm over the whole map, FiLM), its
outputs' form, the beat positions, the stacked columns and the vocal cues' contrast, the label mix, and a stacked
artifact written and read back (its members averaged, its parts checked). Synthetic weights only; the shipped
phrase model is read, never written."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyzer.phrase import dstcn, stack
from analyzer.phrase import phrase_analyzer as PA
from system.resource_paths import resource_path

D_IN = 6          # the net's inputs in these tests (155 in the shipped one)


def weights(rng, d_in=D_IN, c=4, e=3, r=5, dilations=(1, 2), refine=(1, 2), film=True):
    w = {"inp.weight": rng.normal(0, 0.3, (c, d_in + dstcn.POSITIONS, 1)), "inp.bias": rng.normal(0, 0.1, c),
         "song.0.weight": rng.normal(0, 0.3, (e, 2 * (d_in + dstcn.POSITIONS))), "song.0.bias": rng.normal(0, 0.1, e),
         "film.weight": rng.normal(0, 0.3, (2 * c * len(dilations), e)) if film else np.zeros((2 * c * len(dilations), e)),
         "film.bias": rng.normal(0, 0.1, 2 * c * len(dilations)) if film else np.zeros(2 * c * len(dilations)),
         "head.weight": rng.normal(0, 0.3, (10, c, 1)), "head.bias": rng.normal(0, 0.1, 10),
         "rin.weight": rng.normal(0, 0.3, (r, 10, 1)), "rin.bias": rng.normal(0, 0.1, r),
         "rhead.weight": rng.normal(0, 0.3, (10, r, 1)), "rhead.bias": rng.normal(0, 0.1, 10)}
    for name, width, depth in (("blocks", c, len(dilations)), ("refine", r, len(refine))):
        for k in range(depth):
            for part in ("a", "b"):
                w[f"{name}.{k}.{part}.weight"] = rng.normal(0, 0.5, (width, 1, 3))
                w[f"{name}.{k}.{part}.bias"] = rng.normal(0, 0.1, width)
            w[f"{name}.{k}.pw.weight"] = rng.normal(0, 0.3, (width, 2 * width, 1))
            w[f"{name}.{k}.pw.bias"] = rng.normal(0, 0.1, width)
            w[f"{name}.{k}.norm.weight"] = rng.normal(1, 0.1, width)
            w[f"{name}.{k}.norm.bias"] = rng.normal(0, 0.1, width)
    arrays = {f"w.{k}": v.astype(np.float32) for k, v in w.items()}
    arrays.update(mu=rng.normal(0, 1, d_in).astype(np.float32), sd=rng.uniform(0.5, 2, d_in).astype(np.float32),
                  dilations=np.asarray(dilations), refine_dilations=np.asarray(refine), labels=np.asarray(dstcn.LABELS))
    return arrays


def _gelu(v):
    from math import erf, sqrt
    return 0.5 * v * (1 + erf(v / sqrt(2)))


def reference(arrays, x, down):
    """The net written out with loops, in PyTorch's terms."""
    w = {k[2:]: np.asarray(v, np.float64) for k, v in arrays.items() if k.startswith("w.")}
    X = np.hstack([(x - arrays["mu"].astype(np.float64)) / arrays["sd"].astype(np.float64), dstcn.positions(down)])
    T = X.shape[0]

    def conv1(name, h):
        return np.array([[sum(w[f"{name}.weight"][o, i, 0] * h[i, t] for i in range(h.shape[0])) + w[f"{name}.bias"][o]
                          for t in range(h.shape[1])] for o in range(w[f"{name}.weight"].shape[0])])

    def depthwise(name, h, d):
        k = w[f"{name}.weight"]
        out = np.zeros_like(h)
        for ch in range(h.shape[0]):
            for t in range(T):
                s = w[f"{name}.bias"][ch]
                for j in range(3):
                    u = t + (j - 1) * d
                    if 0 <= u < T:
                        s += k[ch, 0, j] * h[ch, u]
                out[ch, t] = s
        return out

    def block(name, h, d):
        u = conv1(f"{name}.pw", np.vstack([depthwise(f"{name}.a", h, d), depthwise(f"{name}.b", h, 2 * d)]))
        m, v = u.mean(), ((u - u.mean()) ** 2).mean()
        u = (u - m) / np.sqrt(v + 1e-5) * w[f"{name}.norm.weight"][:, None] + w[f"{name}.norm.bias"][:, None]
        return h + np.vectorize(_gelu)(u)

    desc = np.concatenate([X.mean(0), X.std(0, ddof=1)])
    song = np.vectorize(_gelu)(w["song.0.weight"] @ desc + w["song.0.bias"])
    dil = [int(d) for d in arrays["dilations"]]
    c = w["inp.weight"].shape[0]
    film = (w["film.weight"] @ song + w["film.bias"]).reshape(len(dil), 2, c)
    h = conv1("inp", X.T)
    for k, d in enumerate(dil):
        h = block(f"blocks.{k}", h, d) * (1 + film[k, 0][:, None]) + film[k, 1][:, None]
    o1 = conv1("head", h)
    e = np.exp(o1[:9] - o1[:9].max(0))
    r = conv1("rin", np.vstack([e / e.sum(0), 1 / (1 + np.exp(-o1[9:]))]))
    for k, d in enumerate(int(d) for d in arrays["refine_dilations"]):
        r = block(f"refine.{k}", r, d)
    o2 = conv1("rhead", r)
    e = np.exp(o2[:9] - o2[:9].max(0))
    return (e / e.sum(0)).T, 1 / (1 + np.exp(-o2[9]))


class DsTcnTests(unittest.TestCase):
    def test_the_net_as_written_out(self):
        for seed, film in ((0, True), (1, True), (2, False)):
            with self.subTest(seed=seed, film=film):
                rng = np.random.default_rng(seed)
                arrays = weights(rng, film=film)
                x = rng.normal(0, 1, (23, D_IN)).astype(np.float32)
                down = np.arange(23) % 4 == 1
                p, b = dstcn.DsTcn(arrays)(x, down)
                rp, rb = reference(arrays, x, down)
                self.assertEqual((p.shape, b.shape, p.dtype, b.dtype), ((23, 9), (23,), np.float32, np.float32))
                np.testing.assert_allclose(p, rp, atol=1e-3)        # the outputs as float16 rounds them
                np.testing.assert_allclose(b, rb, atol=1e-3)
                np.testing.assert_allclose(p.sum(1), 1, atol=5e-3)

    def test_a_dilation_longer_than_the_song(self):
        rng = np.random.default_rng(3)
        arrays = weights(rng, dilations=(1, 16), refine=(8,))
        x = rng.normal(0, 1, (9, D_IN)).astype(np.float32)
        p, b = dstcn.DsTcn(arrays)(x, np.arange(9) % 4 == 0)
        rp, rb = reference(arrays, x, np.arange(9) % 4 == 0)
        np.testing.assert_allclose(p, rp, atol=1e-3)
        np.testing.assert_allclose(b, rb, atol=1e-3)

    def test_wrong_weights_or_inputs_refused(self):
        rng = np.random.default_rng(4)
        arrays = weights(rng)
        for name in ("w.blocks.1.pw.weight", "w.film.weight", "w.rhead.weight"):
            broken = dict(arrays)
            broken[name] = broken[name][:-1]
            with self.subTest(name=name), self.assertRaises(ValueError):
                dstcn.DsTcn(broken)
        broken = dict(arrays)
        del broken["w.refine.0.a.weight"]
        with self.assertRaises(ValueError):
            dstcn.DsTcn(broken)
        with self.assertRaises(ValueError):
            dstcn.DsTcn({**arrays, "sd": np.zeros(D_IN, np.float32)})
        net = dstcn.DsTcn(arrays)
        with self.assertRaises(ValueError):
            net(np.zeros((10, D_IN + 1), np.float32), np.zeros(10, bool))
        with self.assertRaises(ValueError):
            net(np.zeros((10, D_IN), np.float32), np.zeros(11, bool))

    def test_positions(self):
        down = np.zeros(14, bool)
        down[[2, 6, 10]] = True
        pos = dstcn.positions(down)
        self.assertTrue(np.all(pos.sum(1) == 2))
        self.assertEqual(pos[:, :4].argmax(1).tolist(), [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(pos[:, 4:].argmax(1).tolist(), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    def test_inputs_bounded(self):
        x = dstcn.inputs(np.array([[np.nan, np.inf]]), np.array([[-np.inf]]), np.array([[1e6, -1e6]]))
        self.assertEqual(x.tolist(), [[0.0, 50.0, -50.0, 300.0, -50.0]])
        self.assertEqual(x.dtype, np.float32)


class _Half:
    """A specialist calling every segment mixable with probability ``p``."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


class StackTests(unittest.TestCase):
    def test_contrast_as_a_loop(self):
        v = np.random.default_rng(5).normal(0, 1, (30, 3)).astype(np.float32)
        got = stack.contrast(v)
        for i in (0, 1, 7, 8, 15, 29):
            before = v[max(i - 8, 0):i].mean(0) if i else np.zeros(3)
            np.testing.assert_allclose(got[i], v[i:i + 8].mean(0) - before, atol=1e-5)

    def test_columns(self):
        rng = np.random.default_rng(6)
        fz, bfz = rng.normal(0, 1, (20, 5)), rng.normal(0, 1, (20, 7)).astype(np.float32)
        vocal, tcn, tb = rng.normal(0, 1, (20, 8)), rng.uniform(0, 1, (20, 9)), rng.uniform(0, 1, 20)
        beat, bound = stack.columns(fz, bfz, vocal, tcn, tb)
        self.assertEqual((beat.shape, bound.shape, bound.dtype), ((20, 5 + 8 + 10), (20, 7 + 8 + 8 + 10), np.float32))
        np.testing.assert_array_equal(beat[:, :5], fz)
        np.testing.assert_allclose(beat[:, -10:-1], tcn, atol=1e-6)
        np.testing.assert_allclose(bound[:, -1], tb, atol=1e-6)

    def test_riff_cues(self):
        x = np.random.default_rng(7).normal(0, 1, (80, 100)).astype(np.float32)
        down = np.arange(80) % 4 == 0
        cues = stack.riff_cues(x, down)
        self.assertEqual(cues.shape, (80, 4))
        np.testing.assert_allclose(cues[:32, 0], 1, atol=1e-5)          # the first 8 bars match themselves
        np.testing.assert_allclose(cues[-32:, 1], 1, atol=1e-5)
        self.assertEqual(stack.riff_cues(x[:40], down[:40]).tolist(), np.zeros((40, 4)).tolist())   # too few bars

    def test_label_mix(self):
        labels = list(dstcn.LABELS)
        bounds = np.array([0, 4, 10])
        logp = np.log(np.full((2, 9), 1 / 9))
        x = np.zeros((10, 3), np.float32)
        tcn = np.full((10, 9), 1 / 9)
        self.assertEqual(stack.mix_label_logp(logp, bounds, labels, _Half(0.5), x, tcn, (0.0, 0.0)).tolist(),
                         logp.tolist())
        mixed = stack.mix_label_logp(logp, bounds, labels, _Half(0.9), x, tcn, (0.5, 0.0))
        for name in stack.MIXABLE:            # + 0.5 log 0.9 on the mixable labels, + 0.5 log 0.1 on the body's
            self.assertAlmostEqual(mixed[0, labels.index(name)] - logp[0, 0], 0.5 * np.log(0.9))
        for name in stack.BODY:
            self.assertAlmostEqual(mixed[0, labels.index(name)] - logp[0, 0], 0.5 * np.log(0.1))
        self.assertEqual(mixed[0, labels.index("SILENCE")], logp[0, 0])
        tcn = np.full((10, 9), 1e-6)
        tcn[:4, labels.index("CHORUS")] = 1.0
        mixed = stack.mix_label_logp(logp, bounds, labels, _Half(0.5), x, tcn, (0.0, 0.25))
        self.assertEqual(int(mixed[0].argmax()), labels.index("CHORUS"))
        np.testing.assert_allclose(mixed[1] - logp[1], 0.25 * np.log(1e-4))    # clipped as in training


class StackedArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        # a plain phrase artifact: the shipped stack's first member alone
        with np.load(resource_path("assets/weights/default_phrase_analyzer.npz"), allow_pickle=False) as z:
            plain = {k: z[k] for k in z.files if not k.startswith(("m1.", "m2.", "dstcn.", "spec_"))}
        settings = json.loads(str(plain["settings_json"].item()))
        for key in ("ensemble_members", "stack_version", "stack_weights"):
            settings.pop(key, None)
        plain["settings_json"] = np.asarray(json.dumps(settings))
        self.shipped = self.dir / "plain.npz"
        np.savez(self.shipped, **plain)
        from sklearn.ensemble import HistGradientBoostingClassifier
        rng = np.random.default_rng(8)
        X, y = rng.normal(0, 1, (60, 4)), (np.arange(60) % 2)
        clf = HistGradientBoostingClassifier(max_iter=5, random_state=0).fit(X, y)
        payload = {}
        PA._hgb_to_npz_payload("spec", clf, payload)
        np.savez(self.dir / "spec.npz", **payload)
        np.savez(self.dir / "dstcn.npz", **weights(rng))

    def tearDown(self):
        PA.load_two_stage_model.cache_clear()
        self._tmp.cleanup()

    def test_written_and_read_back(self):
        out = self.dir / "stack.npz"
        PA.export_stack_artifact([self.shipped, self.shipped], self.dir / "spec.npz", self.dir / "dstcn.npz", out)
        art = PA.load_two_stage_model(str(out))
        single = PA.load_two_stage_model(str(self.shipped))
        self.assertIsInstance(art["boundary_clf"], PA.MeanClassifier)
        self.assertEqual(len(art["label_clf"].members), 2)
        self.assertEqual(art["stack"]["weights"], (0.5, 0.25))
        self.assertIsInstance(art["stack"]["dstcn"], dstcn.DsTcn)
        X = np.random.default_rng(9).normal(0, 1, (5, single["boundary_clf"].node_feature_idx.max() + 1))
        np.testing.assert_allclose(art["boundary_clf"].predict_proba(X), single["boundary_clf"].predict_proba(X))
        self.assertEqual(art["settings"]["ensemble_members"], 2)
        # what the members share is the first's (the refinement reads the label count of the baseline)
        self.assertEqual(art["label_clf"].baseline.size, single["label_clf"].baseline.size)
        self.assertEqual(set(art["refinement"]), set(single["refinement"]))
        self.assertIsNone(single["stack"])                         # an artifact without them runs as before
        with self.assertRaises(ValueError):                        # a stack is no member
            PA.export_stack_artifact([out], self.dir / "spec.npz", self.dir / "dstcn.npz", self.dir / "nested.npz")

    def test_the_shipped_model_is_a_stack(self):
        art = PA.load_two_stage_model(str(resource_path("assets/weights/default_phrase_analyzer.npz")))
        self.assertIsNotNone(art["stack"])
        self.assertEqual(art["settings"]["ensemble_members"], 3)

    def test_a_broken_stack_refused(self):
        out = self.dir / "stack.npz"
        PA.export_stack_artifact([self.shipped], self.dir / "spec.npz", self.dir / "dstcn.npz", out)
        with np.load(out, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        settings = json.loads(str(arrays["settings_json"].item()))
        for change in ({"stack_version": 99}, {"stack_weights": "0.5"}, {"ensemble_members": 2}):
            broken = self.dir / "broken.npz"
            np.savez(broken, **{**arrays, "settings_json": np.asarray(json.dumps({**settings, **change}))})
            PA.load_two_stage_model.cache_clear()
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                PA.load_two_stage_model(str(broken))


if __name__ == "__main__":
    unittest.main()
