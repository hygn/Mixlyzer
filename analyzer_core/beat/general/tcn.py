"""The frame TCN in NumPy: beat and downbeat activations from the production frame features.

A small dilated temporal convolution network (58k parameters, about 12 s of context each side):
a 1x1 input projection of the 77 frame-feature channels, nine residual blocks (dilated convolution of
width 5 with dilations 1 .. 256 frames, ELU, 1x1 mix), ELU, a 1x1 output to two logits per frame.
Musical: a beat is heard from many cues at once (bass, harmony changes, drums, melody onsets) over
several bars of context. Engineering: the same frames the rest of the analyzer uses, no torch.

Weights: a npz per training seed with the torch state dict's arrays plus the input statistics
(``mu``, ``sd``), see ``assets/weights/beat_tcn_s*.npz``.
"""
from __future__ import annotations

import numpy as np

from core.resource_paths import resource_path

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
K = 5
CHANNELS = ('chroma', 'bass_chroma', 'mfcc', 'tonnetz', 'onset_bands', 'rms_db', 'band_ratios', 'flatness',
            'librosa_onset', 'harmonic_onset', 'percussive_bands', 'percussive_share', 'melody_chroma', 'melody_onset')
WEIGHT_PATHS = ('assets/weights/beat_tcn_s7.npz', 'assets/weights/beat_tcn_s8.npz', 'assets/weights/beat_tcn_s9.npz')
# The frame grid every shipped model of the tracker was trained on (11.6 ms frames): the dilations count
# frames and the spectral features depend on the rate, so the tracker reads frames on this grid only.
SAMPLE_RATE = 22050
FRAME_HOP_LENGTH = 256


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def load_weights(path: str) -> dict:
    with np.load(resource_path(path)) as z:
        return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}


def frame_inputs(frames) -> np.ndarray:
    """[T, 77] frame features in the training order."""
    T = int(np.asarray(frames.frame_times).size)
    return np.concatenate([np.atleast_2d(np.asarray(getattr(frames, c), dtype=np.float32)).reshape(-1, T)
                           for c in CHANNELS]).T


def _conv1x1(W, b, x):
    return W[:, :, 0] @ x + b[:, None]


def _conv_dilated(W, b, x, d):
    C, T = x.shape
    xp = np.pad(x, ((0, 0), (2 * d, 2 * d)))
    cols = np.concatenate([xp[:, j * d:j * d + T] for j in range(K)], axis=0)
    Wm = W.transpose(0, 2, 1).reshape(W.shape[0], -1)
    return Wm @ cols + b[:, None]


def activations(P: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x [T, 77] raw frame features -> (beat, downbeat) probabilities per frame."""
    h = ((x - P['mu']) / P['sd']).T.astype(np.float32)
    h = _conv1x1(P['inp.weight'], P['inp.bias'], h)
    for i, d in enumerate(DILATIONS):
        a = _conv_dilated(P[f'blocks.{i}.conv.weight'], P[f'blocks.{i}.conv.bias'], h, d)
        h = h + _conv1x1(P[f'blocks.{i}.mix.weight'], P[f'blocks.{i}.mix.bias'], elu(a))
    out = _conv1x1(P['out.weight'], P['out.bias'], elu(h))
    p = 1.0 / (1.0 + np.exp(-out.astype(np.float64)))
    return p[0], p[1]
