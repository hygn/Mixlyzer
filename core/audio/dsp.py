from __future__ import annotations
import math
import numpy as np


def fade_ramp(n: int, fade_frames: int, rising: bool) -> np.ndarray:
    """[n, 1] linear gain ramp over fade_frames (0->1 when rising), held after that."""
    r = (np.arange(n, dtype=np.float32) + 0.5) / float(max(1, fade_frames))
    r = np.clip(r, 0.0, 1.0)
    return (r if rising else 1.0 - r)[:, None]


class SpeedResampler:
    """
    5-tap Lagrange(4차) polynomial sinc approx varispeed reader.

    Reads a fully decoded [N, ch] buffer at absolute (fractional) input positions, so
    consecutive render() calls are seamless: taps are taken from the real neighbouring
    samples across block boundaries and no output frame is ever emitted twice.
    """
    def __init__(self, channels: int):
        self.ch = int(channels)

    @staticmethod
    def _lagrange5_coeffs(mu: np.ndarray) -> np.ndarray:
        c_m2 = ( 1/24.0)*mu*(mu-1)*(mu-2)*(mu-3)
        c_m1 = (-1/6.0 )*(mu+1)*(mu-1)*(mu-2)*(mu-3)
        c_0  = ( 1/4.0 )*(mu+1)* mu   *(mu-2)*(mu-3)
        c_p1 = (-1/6.0 )*(mu+1)* mu   *(mu-1)*(mu-3)
        c_p2 = ( 1/24.0)*(mu+1)* mu   *(mu-1)*(mu-2)
        return np.stack([c_m2, c_m1, c_0, c_p1, c_p2], axis=-1)

    def render(self, pcm: np.ndarray, pos: float, factor: float, max_out_frames: int) -> np.ndarray:
        """
        Render up to max_out_frames output frames starting at absolute input frame `pos`,
        advancing `factor` input frames per output frame. Output frame k is the signal at
        input position pos + k*factor, so the caller's next block starts at
        pos + factor * len(out).
        """
        n = int(pcm.shape[0])
        if max_out_frames <= 0 or pos >= n:
            return np.zeros((0, self.ch), np.float32)
        m = min(int(max_out_frames), int(math.ceil((n - pos) / factor)))
        if m <= 0:
            return np.zeros((0, self.ch), np.float32)

        if factor == 1.0 and float(pos).is_integer():
            i = int(pos)
            return np.array(pcm[i:i + m], dtype=np.float32)

        idx = pos + factor * np.arange(m, dtype=np.float64)
        i0 = np.floor(idx).astype(np.int64)
        mu = idx - i0

        # The coefficients' nodes are i0-1 .. i0+3 (at mu=0 only c_m1, the i0 tap, is non-zero).
        # One window covers every tap of the block; edges are held (same as clipping).
        lo = int(i0[0]) - 1
        hi = int(i0[-1]) + 4
        win = np.asarray(pcm[max(lo, 0):min(hi, n)], dtype=np.float64)
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - n)
        if pad_lo or pad_hi:
            win = np.pad(win, ((pad_lo, pad_hi), (0, 0)), mode="edge")

        rel = i0 - i0[0]
        X = np.stack([win[rel + k] for k in range(5)], axis=1)  # [M,5,ch], taps i0-1+k
        C = self._lagrange5_coeffs(mu)  # [M,5]
        out = np.einsum("mk,mkc->mc", C, X, optimize=True)
        return out.astype(np.float32, copy=False)
