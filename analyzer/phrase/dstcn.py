"""The phrase DS-TCN, run in NumPy from its trained weights: per beat the 9 phrase labels' probabilities and the
boundary probability, which the phrase model's GBMs read as features and its label decoding mixes in (stack.py).

A dilated depthwise-separable temporal convolution net over the beats (8 residual blocks, dilations 1 .. 128, each
two depthwise convolutions of dilation d and 2d, a pointwise one, GroupNorm, GELU), each block's output scaled and
shifted by the song's own character (FiLM: the mean and spread of the inputs over the song through a small MLP), a
label + boundary head, and an MS-TCN refinement stage (6 blocks on the head's probabilities). Trained in PyTorch by
the developers' training (research: dl_dstcn_film.py); its weights are stored by their PyTorch names (``w.<name>``)
beside the input statistics. Computed in float64; the outputs rounded to float16 as the training's stacked models
read them.

    inputs(feature_z, vocal, style)    the net's per-beat inputs (the phrase features, the song cues)
    DsTcn(arrays)(x, downbeat)         (probabilities (beats, 9) in LABELS order, boundary probability (beats,))
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.special import erf

LABELS = ("BREAK_CHORUS", "BRIDGE", "CHORUS", "INTERLUDE", "INTRO", "LEAD_IN", "OUTRO", "SILENCE", "VERSE")
POSITIONS = 8            # the beat in its bar (4) and the bar in its 4 bars (4), one-hot
GROUP_NORM_EPS = 1e-5


def inputs(feature_z: np.ndarray, vocal: np.ndarray, style: np.ndarray) -> np.ndarray:
    """(beats, 133 + 8 + 14) float32: the phrase features and the song cues, NaN as 0, bounded as in training."""
    x = np.hstack([np.asarray(feature_z, np.float32), np.asarray(vocal, np.float32), np.asarray(style, np.float32)])
    return np.clip(np.nan_to_num(x, posinf=50, neginf=-50), -50, 300).astype(np.float32)


def positions(downbeat: np.ndarray) -> np.ndarray:
    """(beats, 8): the beat's place in its bar (counted from the last downbeat, the first bar's from the first) and its
    bar's place in 4 bars."""
    down = np.asarray(downbeat, bool)
    n = down.size
    bar = np.cumsum(down) - 1
    first = int(np.argmax(down)) if down.any() else 0
    pos = np.zeros(n, int)
    last = -first
    for i in range(n):
        if down[i]:
            last = i
        pos[i] = (i - last) % 4
    out = np.zeros((n, POSITIONS), np.float32)
    out[np.arange(n), pos] = 1
    out[np.arange(n), 4 + np.maximum(bar, 0) % 4] = 1
    return out


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class DsTcn:
    """The net of a phrase model's arrays (``dstcn.``-prefixed names stripped: mu, sd, dilations, refine_dilations,
    w.<PyTorch name>), its shapes checked on loading."""

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        a = {k: np.asarray(v) for k, v in arrays.items()}
        self.mu = a["mu"].astype(np.float64)
        self.sd = a["sd"].astype(np.float64)
        self.dilations = [int(d) for d in a["dilations"]]
        self.refine_dilations = [int(d) for d in a["refine_dilations"]]
        self.w = {k[2:]: v.astype(np.float64) for k, v in a.items() if k.startswith("w.")}
        d_in = self.mu.size + POSITIONS
        c = self.w["inp.weight"].shape[0]
        expected = {"inp.weight": (c, d_in, 1), "song.0.weight": (self.w["song.0.weight"].shape[0], 2 * d_in),
                    "film.weight": (2 * c * len(self.dilations), self.w["song.0.weight"].shape[0]),
                    "head.weight": (len(LABELS) + 1, c, 1), "rhead.weight": (len(LABELS) + 1,
                                                                             self.w["rin.weight"].shape[0], 1)}
        for name, width, depth in (("blocks", c, len(self.dilations)),
                                   ("refine", self.w["rin.weight"].shape[0], len(self.refine_dilations))):
            for k in range(depth):
                for part in ("a", "b"):
                    expected[f"{name}.{k}.{part}.weight"] = (width, 1, 3)
                expected[f"{name}.{k}.pw.weight"] = (width, 2 * width, 1)
                expected[f"{name}.{k}.norm.weight"] = (width,)
        if self.sd.shape != self.mu.shape or np.any(self.sd <= 0):
            raise ValueError("phrase DS-TCN: bad input statistics")
        for name, shape in expected.items():
            if name not in self.w or self.w[name].shape != shape:
                raise ValueError(f"phrase DS-TCN: {name} missing or not {shape}")
        self.channels = c

    def _block(self, prefix: str, h: np.ndarray, d: int) -> np.ndarray:
        w = self.w
        u = np.vstack([self._depthwise(f"{prefix}.a", h, d), self._depthwise(f"{prefix}.b", h, 2 * d)])
        u = self._pointwise(f"{prefix}.pw", u)
        u = (u - u.mean()) / np.sqrt(u.var() + GROUP_NORM_EPS)             # GroupNorm(1, C): over the whole map
        u = u * w[f"{prefix}.norm.weight"][:, None] + w[f"{prefix}.norm.bias"][:, None]
        return h + _gelu(u)

    def _depthwise(self, prefix: str, h: np.ndarray, d: int) -> np.ndarray:
        """Each channel's kernel-3 convolution of dilation d, zero padded (PyTorch's cross-correlation)."""
        k = self.w[f"{prefix}.weight"][:, 0, :]
        n = h.shape[1]
        pad = np.zeros((h.shape[0], n + 2 * d))
        pad[:, d:d + n] = h
        return (k[:, 0:1] * pad[:, :n] + k[:, 1:2] * pad[:, d:d + n] + k[:, 2:3] * pad[:, 2 * d:2 * d + n]
                + self.w[f"{prefix}.bias"][:, None])

    def _pointwise(self, prefix: str, h: np.ndarray) -> np.ndarray:
        return self.w[f"{prefix}.weight"][:, :, 0] @ h + self.w[f"{prefix}.bias"][:, None]

    def __call__(self, x: np.ndarray, downbeat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``x``: inputs(...) of the song's beats; ``downbeat``: the phrase grid's downbeat mask."""
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or x.shape[1] != self.mu.size or x.shape[0] != np.asarray(downbeat).size:
            raise ValueError(f"phrase DS-TCN: inputs {x.shape} for {self.mu.size} features")
        X = np.hstack([(x - self.mu) / self.sd, positions(downbeat)])            # (beats, d_in)
        w = self.w
        desc = np.concatenate([X.mean(0), X.std(0, ddof=1)])                   # the song's character
        song = _gelu(w["song.0.weight"] @ desc + w["song.0.bias"])
        film = (w["film.weight"] @ song + w["film.bias"]).reshape(len(self.dilations), 2, self.channels)
        h = self._pointwise("inp", X.T)
        for k, d in enumerate(self.dilations):
            h = self._block(f"blocks.{k}", h, d)
            h = h * (1.0 + film[k, 0][:, None]) + film[k, 1][:, None]
        o1 = self._pointwise("head", h)
        n = len(LABELS)
        z = np.vstack([_softmax(o1[:n], 0), _sigmoid(o1[n:])])
        r = self._pointwise("rin", z)
        for k, d in enumerate(self.refine_dilations):
            r = self._block(f"refine.{k}", r, d)
        o2 = self._pointwise("rhead", r)
        probabilities = _softmax(o2[:n], 0).T.astype(np.float16).astype(np.float32)
        boundary = _sigmoid(o2[n]).astype(np.float16).astype(np.float32)
        return probabilities, boundary
