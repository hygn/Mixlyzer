from __future__ import annotations
import math
import numpy as np
from scipy.signal import lfilter

# Imported here, not lazily: this module is imported on the main thread, and the time
# stretcher's first scipy.fft import must not happen on the audio QThread (importing it
# there first crashed the process later: heap corruption / access violation in
# scipy.fft, 5 of 5 runs; importing scipy.fft on the main thread first: 0 of 5).
from core.audio.timestretch import StreamingTimeStretcher


def make_speed_renderer(channels: int, sample_rate: int, timestretch: bool = False):
    """Renderer for the "speed" mode, with the SpeedResampler.render(pcm, pos, factor,
    max_out_frames) contract: varispeed (pitch follows tempo), or with ``timestretch``
    the pitch-preserving time stretch (core/audio/timestretch.py). The stretcher uses
    its numba kernels once timestretch.warm_timestretch_numba() has loaded them, NumPy
    until then."""
    if timestretch:
        return StreamingTimeStretcher(channels, sample_rate)
    return SpeedResampler(channels)


def fade_ramp(n: int, fade_frames: int, rising: bool) -> np.ndarray:
    """[n, 1] linear gain ramp over fade_frames (0->1 when rising), held after that."""
    r = (np.arange(n, dtype=np.float32) + 0.5) / float(max(1, fade_frames))
    r = np.clip(r, 0.0, 1.0)
    return (r if rising else 1.0 - r)[:, None]


def soft_clip(x: np.ndarray, knee: float) -> np.ndarray:
    """Linear up to |x| = knee, then a tanh curve to the ceiling 1.0 (continuous slope at
    the knee). Returns x itself when nothing exceeds the knee."""
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    y = np.array(x, dtype=np.float32)
    span = np.float32(1.0 - knee)
    y[over] = np.sign(x[over]) * (np.float32(knee) + span * np.tanh((a[over] - np.float32(knee)) / span))
    return y


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


class ScratchPlatter:
    """The record under the hand while scratching: per output frame, the input
    position it is at.

    The pointer's positions arrive unevenly (bunched or late on a busy GUI
    thread), each stamped with the time it was taken. The platter follows that
    path DELAY_SEC late, interpolated linearly between the samples (a jitter
    buffer: the next sample is normally there already, so the speed carries on
    between events instead of stopping and starting); past the last sample (an
    event later than the delay) it carries on at the last speed for up to
    EXTRAPOLATE_SEC, then holds. Smoothed by two one-pole lowpasses (critically
    damped, SMOOTH_SEC).

    Output frames are timed on a clock running at the sample rate, kept on the
    time they are heard (now + the queued audio) by a gentle correction.
    """

    DELAY_SEC = 0.06
    EXTRAPOLATE_SEC = 0.02
    SMOOTH_SEC = 0.008
    HISTORY_SEC = 1.0
    CLOCK_GAIN = 0.02     # per block
    CLOCK_RESYNC_SEC = 0.05

    def __init__(self, sample_rate: int):
        self.rate = int(sample_rate)
        self._pole = math.exp(-1.0 / (self.SMOOTH_SEC * self.rate))
        self.reset(0.0, 0.0)

    def reset(self, pos: float, now: float) -> None:
        """Held still at ``pos`` from time ``now`` (s)."""
        pos = float(pos)
        self._times = [float(now)]
        self._path = [pos]
        self._clock = float(now)
        self._zi = [np.array([self._pole * pos]), np.array([self._pole * pos])]
        self.position = pos

    def set_target(self, pos: float, at: float) -> None:
        """The hand was at ``pos`` (input frames) at time ``at`` (s)."""
        at = max(float(at), self._times[-1])
        self._times.append(at)
        self._path.append(float(pos))
        cut = 0
        while cut < len(self._times) - 2 and self._times[cut] < at - self.HISTORY_SEC:
            cut += 1
        if cut:
            del self._times[:cut], self._path[:cut]

    def positions(self, n: int, heard_at: float) -> np.ndarray:
        """Input positions of the next ``n`` output frames, the first heard at
        time ``heard_at`` (s)."""
        error = float(heard_at) - self._clock
        if abs(error) > self.CLOCK_RESYNC_SEC:
            self._clock = float(heard_at)
        else:
            self._clock += self.CLOCK_GAIN * error
        times = self._clock + np.arange(int(n), dtype=np.float64) / self.rate
        self._clock += int(n) / self.rate
        lookup = times - self.DELAY_SEC
        hand = np.interp(lookup, self._times, self._path)
        late = lookup > self._times[-1]
        if len(self._times) >= 2 and late.any():
            span = self._times[-1] - self._times[-2]
            if span > 1e-4:
                speed = (self._path[-1] - self._path[-2]) / span
                hand[late] += speed * np.minimum(lookup[late] - self._times[-1], self.EXTRAPOLATE_SEC)
        b, a = [1.0 - self._pole], [1.0, -self._pole]
        x, self._zi[0] = lfilter(b, a, hand, zi=self._zi[0])
        x, self._zi[1] = lfilter(b, a, x, zi=self._zi[1])
        if x.size:
            self.position = float(x[-1])
        return x


def read_at(pcm: np.ndarray, positions: np.ndarray, previous: float | None = None) -> np.ndarray:
    """[M, ch] float32: ``pcm`` read at fractional input ``positions`` (any order,
    either direction) by 4-point cubic Hermite (Catmull-Rom) interpolation; edges held.

    Simple anti-aliasing: where the read moves faster than 1 input frame per
    output frame, each output frame is (blended, fully from speed 2 on, into) the
    mean of the input it passed over since the previous one (a box filter as wide
    as the speed); ``previous`` is the position before ``positions[0]``."""
    n = int(pcm.shape[0])
    if n == 0 or positions.size == 0:
        return np.zeros((positions.size, pcm.shape[1] if pcm.ndim == 2 else 1), np.float32)
    p = np.clip(positions, 0.0, n - 1)
    i0 = np.floor(p).astype(np.int64)
    mu = (p - i0)[:, np.newaxis]
    before = np.concatenate(([p[0] if previous is None else min(max(float(previous), 0.0), n - 1)], p[:-1]))
    lo = int(min(i0.min(), np.floor(before.min()))) - 1
    hi = int(max(i0.max(), np.floor(before.max()))) + 3
    win = np.asarray(pcm[max(lo, 0):min(hi, n)], dtype=np.float64)
    if lo < 0 or hi > n:
        win = np.pad(win, ((max(0, -lo), max(0, hi - n)), (0, 0)), mode="edge")
    rel = i0 - lo
    xm1, x0, x1, x2 = win[rel - 1], win[rel], win[rel + 1], win[rel + 2]
    out = x0 + 0.5 * mu * (x1 - xm1 + mu * (2.0 * xm1 - 5.0 * x0 + 4.0 * x1 - x2 + mu * (3.0 * (x0 - x1) + x2 - xm1)))
    speed = np.abs(p - before)
    fast = speed > 1.0
    if np.any(fast):
        # running integral of the (sample-and-hold) input over the window
        integral = np.concatenate((np.zeros((1, win.shape[1])), np.cumsum(win, axis=0)))

        def integral_at(x):
            r = x - lo
            k = np.floor(r).astype(np.int64)
            return integral[k] + (r - k)[:, np.newaxis] * win[k]

        a, b = before[fast], p[fast]
        box = (integral_at(b) - integral_at(a)) / (b - a)[:, np.newaxis]
        w = np.clip(speed[fast] - 1.0, 0.0, 1.0)[:, np.newaxis]
        out[fast] = (1.0 - w) * out[fast] + w * box
    return out.astype(np.float32)
