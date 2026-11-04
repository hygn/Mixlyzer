"""The phrase model's stack: what a stacked phrase model (phrase_analyzer.load_two_stage_model's ``stack``) reads beside
the phrase features, and how its label decoding mixes in the DS-TCN and the mix-label specialist.

* columns(...): the GBMs' extra columns. The beat features get the vocal cues (song_cues.vocal_cues, 8) and the
  DS-TCN's outputs (9 label probabilities, the boundary probability); the boundary features get the same with the
  vocal cues' 8-beat contrast (the mean of the 8 beats after minus the 8 before) between them.
* The mix-label specialist: a GBM telling a segment's mixable labels (LEAD_IN / INTRO / OUTRO / INTERLUDE) from the
  sung body (VERSE / BRIDGE / CHORUS / BREAK_CHORUS) by its own content: the mean, spread and maximum over the
  segment of the DS-TCN's inputs and the riff cues (each bar's best chroma / MFCC match with the song's first and
  last 8 bars), and its length.
* mix_label_logp(...): the label GBM's segment log-probabilities before the label Viterbi, plus the specialist's
  log-odds (weight ``weights[0]``: log p on the mixable labels, log (1 - p) on the body's) and the DS-TCN's mean
  log-probability over the segment (``weights[1]``).

As the research trained them (research/benchmark/phrase_boundary: cv_all.py append / dsn columns, exp_specialist.py
riff_cues / seg_rows / mixed): the same operations in the same order."""
from __future__ import annotations

import numpy as np

from analyzer.phrase.dstcn import LABELS as DSTCN_LABELS

MIXABLE = ("LEAD_IN", "INTRO", "OUTRO", "INTERLUDE")
BODY = ("VERSE", "BRIDGE", "CHORUS", "BREAK_CHORUS")
CONTRAST_BEATS = 8
# the DS-TCN inputs' columns the riff cues compare bars by: the chroma (12) and 20 MFCCs of the phrase features
_RIFF_VIEWS = (slice(73, 85), slice(48, 68))


def contrast(v: np.ndarray, beats: int = CONTRAST_BEATS) -> np.ndarray:
    """Per beat the mean of the ``beats`` beats from it on minus the mean of the ``beats`` before it."""
    c = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, 0)])
    n = v.shape[0]
    i = np.arange(n)
    lo, hi = np.maximum(i - beats, 0), np.minimum(i + beats, n)
    left = (c[i] - c[lo]) / np.maximum(i - lo, 1)[:, None]
    right = (c[hi] - c[i]) / np.maximum(hi - i, 1)[:, None]
    return right - left


def columns(feature_z: np.ndarray, boundary_feature_z: np.ndarray, vocal: np.ndarray, tcn: np.ndarray,
            tcn_boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(beat features, boundary features) with the stack's columns."""
    fz = np.asarray(feature_z)
    v = np.asarray(vocal, np.float32)
    cols = np.hstack([np.asarray(tcn, np.float32), np.asarray(tcn_boundary, np.float32)[:, None]])
    beat = np.hstack([np.hstack([fz, v]).astype(fz.dtype), cols])
    bound = np.hstack([np.hstack([boundary_feature_z, v, contrast(v)]).astype(np.float32), cols])
    return beat, bound


def riff_cues(x: np.ndarray, downbeat: np.ndarray) -> np.ndarray:
    """(beats, 4): its bar's best match (chroma, MFCC) with the song's first and with its last 8 bars."""
    bars = np.flatnonzero(np.asarray(downbeat, bool))
    out = np.zeros((x.shape[0], 4), np.float32)
    if bars.size < 12:
        return out
    edges = np.r_[bars, x.shape[0]]
    for c, sl in enumerate(_RIFF_VIEWS):
        v = np.stack([x[a:b, sl].mean(0) for a, b in zip(edges[:-1], edges[1:])])
        v = v - v.mean(1, keepdims=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        m = v.shape[0]
        first, last = v[:min(8, m)], v[max(0, m - 8):]
        sf, sl_ = (v @ first.T).max(1), (v @ last.T).max(1)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            out[a:b, 2 * c], out[a:b, 2 * c + 1] = sf[i], sl_[i]
    return out


def segment_rows(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Per segment: the mean, spread and maximum of ``x`` over it, and its length's log."""
    return np.vstack([np.concatenate([x[a:max(b, a + 1)].mean(0), x[a:max(b, a + 1)].std(0),
                                      x[a:max(b, a + 1)].max(0), [np.log(max(b - a, 1))]])
                      for a, b in zip(bounds[:-1], bounds[1:])])


def mix_label_logp(logp: np.ndarray, bounds: np.ndarray, labels: list[str], specialist, specialist_x: np.ndarray,
                   tcn: np.ndarray, weights: tuple[float, float]) -> np.ndarray:
    """The segments' label log-probabilities (``labels`` order) with the specialist's and the DS-TCN's mixed in.
    ``specialist_x``: the DS-TCN's inputs and the riff cues per beat; ``tcn``: the DS-TCN's probabilities."""
    b = np.asarray(bounds, int)
    lp = np.asarray(logp, float).copy()
    p = np.clip(specialist.predict_proba(segment_rows(specialist_x, b))[:, 1], 1e-3, 1 - 1e-3)
    mix = [labels.index(x) for x in MIXABLE]
    body = [labels.index(x) for x in BODY]
    lp[:, mix] += weights[0] * np.log(p)[:, None]
    lp[:, body] += weights[0] * np.log(1 - p)[:, None]
    if weights[1]:
        lt = np.log(np.clip(np.asarray(tcn, np.float64), 1e-4, 1))[:, [DSTCN_LABELS.index(x) for x in labels]]
        lp = lp + weights[1] * np.vstack([lt[b[k]:max(b[k + 1], b[k] + 1)].mean(0) for k in range(len(b) - 1)])
    return lp
