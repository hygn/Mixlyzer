from __future__ import annotations
import math
import numpy as np
from typing import Tuple

class SpeedResampler:
    """
    5-tap Lagrange(4차) 다항식 sinc 근사. set_factor()/process() API.
    """
    def __init__(self, channels: int):
        self.ch = int(channels)
        self.pos = 0.0
        self.factor = 1.0

    def set_factor(self, f: float):
        self.factor = float(min(4.0, max(0.25, f)))

    @staticmethod
    def _lagrange5_coeffs(mu: np.ndarray) -> np.ndarray:
        c_m2 = ( 1/24.0)*mu*(mu-1)*(mu-2)*(mu-3)
        c_m1 = (-1/6.0 )*(mu+1)*(mu-1)*(mu-2)*(mu-3)
        c_0  = ( 1/4.0 )*(mu+1)* mu   *(mu-2)*(mu-3)
        c_p1 = (-1/6.0 )*(mu+1)* mu   *(mu-1)*(mu-3)
        c_p2 = ( 1/24.0)*(mu+1)* mu   *(mu-1)*(mu-2)
        return np.stack([c_m2, c_m1, c_0, c_p1, c_p2], axis=-1)

    def process(self, fifo: np.ndarray, max_out_frames: int) -> Tuple[np.ndarray, int]:
        if fifo.size == 0 or max_out_frames <= 0:
            return np.zeros((0, self.ch), np.float32), 0

        fifo = np.asarray(fifo)
        avail = fifo.shape[0]
        if avail < 2:
            return np.zeros((0, self.ch), np.float32), 0

        max_m = int((avail - 2 - self.pos) / max(self.factor, 1e-12)) + 1
        M = max(0, min(max_out_frames, max_m))
        if M <= 0:
            return np.zeros((0, self.ch), np.float32), 0

        idx = self.pos + self.factor * np.arange(M, dtype=np.float64)
        i0 = np.floor(idx).astype(np.int64)
        mu = (idx - i0).astype(np.float64)

        i_m2 = np.clip(i0 - 2, 0, avail - 1)
        i_m1 = np.clip(i0 - 1, 0, avail - 1)
        i_0  = np.clip(i0 + 0, 0, avail - 1)
        i_p1 = np.clip(i0 + 1, 0, avail - 1)
        i_p2 = np.clip(i0 + 2, 0, avail - 1)

        X = np.stack([
            fifo[i_m2].astype(np.float64, copy=False),
            fifo[i_m1].astype(np.float64, copy=False),
            fifo[i_0 ].astype(np.float64, copy=False),
            fifo[i_p1].astype(np.float64, copy=False),
            fifo[i_p2].astype(np.float64, copy=False),
        ], axis=1)  # [M,5,ch]

        C = self._lagrange5_coeffs(mu)  # [M,5]
        out = np.einsum("mk,mkc->mc", C, X, optimize=True)
        end_idx = idx[-1]
        consumed = int(max(0, math.floor(end_idx)))
        self.pos = float(end_idx - consumed)
        return out.astype(np.float32, copy=False), consumed