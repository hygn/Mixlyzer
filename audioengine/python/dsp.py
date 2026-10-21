from __future__ import annotations
import math
import numpy as np
from scipy.signal import lfilter

# Imported here, not lazily: this module is imported on the main thread, and the time
# stretcher's first scipy.fft import must not happen on the audio QThread (importing it
# there first crashed the process later: heap corruption / access violation in
# scipy.fft, 5 of 5 runs; importing scipy.fft on the main thread first: 0 of 5).
from audioengine.python.timestretch import StreamingTimeStretcher


def make_speed_renderer(channels: int, sample_rate: int, timestretch: bool = False):
    """Renderer for the "speed" mode, with the SpeedResampler.render(pcm, pos, factor,
    max_out_frames) contract: varispeed (pitch follows tempo), or with ``timestretch``
    the pitch-preserving time stretch (audioengine/python/timestretch.py). The stretcher uses
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

    The hand's positions arrive unevenly (bunched or late on a busy GUI thread),
    each stamped with the time it was taken. The platter follows that path
    DELAY_SEC late, interpolated linearly between the samples (a jitter buffer:
    the next sample is normally there already, so the speed carries on between
    events instead of stopping and starting). Past the last sample (an event later
    than the delay) it carries on at the speed of the last RATE_SEC of samples for
    up to EXTRAPOLATE_SEC, then holds; it holds at once after a rest sample (the
    hand stopped there). With ``pixel`` (input frames per pointer pixel, samples
    being pixel-boundary crossings: audioengine/python/cursor_poller.py) it never carries
    on past the next boundary, which the hand has not crossed yet. Smoothed by two
    one-pole lowpasses (critically damped, SMOOTH_SEC).

    Output frames are timed on a clock running at the sample rate, kept on the
    time they are heard (now + the queued audio) by a gentle correction.
    """

    DELAY_SEC = 0.06
    EXTRAPOLATE_SEC = 0.02
    RATE_SEC = 0.012
    SMOOTH_SEC = 0.008
    HISTORY_SEC = 1.0
    CLOCK_GAIN = 0.02     # per block
    CLOCK_RESYNC_SEC = 0.05

    def __init__(self, sample_rate: int):
        self.rate = int(sample_rate)
        self._pole = math.exp(-1.0 / (self.SMOOTH_SEC * self.rate))
        self.reset(0.0, 0.0)

    def reset(self, pos: float, now: float, pixel: float = 0.0) -> None:
        """Held still at ``pos`` from time ``now`` (s); ``pixel``: see the class."""
        pos = float(pos)
        self._times = [float(now)]
        self._path = [pos]
        self._rest = [True]
        self._pixel = abs(float(pixel))
        self._clock = float(now)
        self._zi = [np.array([self._pole * pos]), np.array([self._pole * pos])]
        self.position = pos

    def set_target(self, pos: float, at: float, rest: bool = False) -> None:
        """The hand was at ``pos`` (input frames) at time ``at`` (s); ``rest``: and
        stopped there."""
        at = max(float(at), self._times[-1])
        self._times.append(at)
        self._path.append(float(pos))
        self._rest.append(bool(rest))
        cut = 0
        while cut < len(self._times) - 2 and self._times[cut] < at - self.HISTORY_SEC:
            cut += 1
        if cut:
            del self._times[:cut], self._path[:cut], self._rest[:cut]

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
        if len(self._times) >= 2 and not self._rest[-1] and late.any():
            # from the newest sample at least RATE_SEC before the last one (or the one before it)
            t_last = self._times[-1]
            j = len(self._times) - 2
            while j > 0 and self._times[j] > t_last - self.RATE_SEC:
                j -= 1
            span = t_last - self._times[j]
            if span > 1e-4:
                speed = (self._path[-1] - self._path[j]) / span
                ahead = speed * np.minimum(lookup[late] - t_last, self.EXTRAPOLATE_SEC)
                if self._pixel:
                    ahead = np.clip(ahead, -self._pixel, self._pixel)
                hand[late] += ahead
        b, a = [1.0 - self._pole], [1.0, -self._pole]
        x, self._zi[0] = lfilter(b, a, hand, zi=self._zi[0])
        x, self._zi[1] = lfilter(b, a, x, zi=self._zi[1])
        if x.size:
            self.position = float(x[-1])
        return x


# Scratch reader (read_at): Kaiser-windowed sinc, cutoff READ_CUTOFF of the Nyquist
# divided by the speed r (input frames per output frame, at least 1). READ_HALF zero
# crossings each side, but at most READ_REACH input frames each side: from r = 2 on
# fewer crossings (8 at r = READ_STRIDE_STRETCH). Faster, the 8-crossing kernel's
# taps are r / READ_STRIDE_STRETCH frames apart, each the input triangle-filtered over
# its neighbours' span. At most 2 * READ_REACH taps per output frame at any speed.
READ_HALF = 16
READ_REACH = 32
READ_STRIDE_STRETCH = 4
READ_BETA = 8.0
READ_CUTOFF = 0.9
_READ_SINC_RES = 512  # table points per unit (a zero crossing at the full cutoff)
_READ_WIN_RES = 4096  # table points over the window's half
_READ_SINC = READ_CUTOFF * np.sinc(READ_CUTOFF * np.arange(READ_HALF * _READ_SINC_RES + 2) / _READ_SINC_RES)
_read_x = np.arange(_READ_WIN_RES + 2) / _READ_WIN_RES
_READ_WIN = np.i0(READ_BETA * np.sqrt(np.clip(1.0 - _read_x ** 2, 0.0, None))) / np.i0(READ_BETA)
_READ_WIN[_read_x >= 1.0] = 0.0
# Up to r = 2 the kernel is one function of u = t / r (t: input frames from the
# position): sinc(u) * window(u / READ_HALF), tabulated like _READ_SINC.
_read_u = np.arange(READ_HALF * _READ_SINC_RES + 2) / _READ_SINC_RES
_READ_UNIT = (_READ_SINC * np.i0(READ_BETA * np.sqrt(np.clip(1.0 - (_read_u / READ_HALF) ** 2, 0.0, None)))
              / np.i0(READ_BETA))
_READ_UNIT[_read_u >= READ_HALF] = 0.0
_read_m = np.arange(1 - READ_REACH, READ_REACH)
_READ_STRIDE_W = (READ_CUTOFF * np.sinc(READ_CUTOFF * _read_m / READ_STRIDE_STRETCH)
                  * np.i0(READ_BETA * np.sqrt(1.0 - (_read_m / READ_REACH) ** 2)) / np.i0(READ_BETA))
_READ_STRIDE_W /= _READ_STRIDE_W.sum()
del _read_x, _read_u, _read_m


# The compiled read_at kernel (audioengine/python/_scratch_numba.py) once warm_scratch_numba()
# has loaded it, False if that failed; read_at uses NumPy until then (several times
# slower: fast scratches may not keep up in real time).
_SCRATCH_NUMBA = None


def warm_scratch_numba() -> bool:
    """Load (disk cache) or compile read_at's numba kernel. Call it outside the audio
    thread, which never waits for it. Failure leaves the NumPy path."""
    global _SCRATCH_NUMBA
    if _SCRATCH_NUMBA is None:
        try:
            from audioengine.python import _scratch_numba as kernels
            kernels.warm()
            _SCRATCH_NUMBA = kernels
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[Scratch] numba kernel unavailable, using NumPy: {exc}")
            _SCRATCH_NUMBA = False
    return bool(_SCRATCH_NUMBA)


def _lerp(table: np.ndarray, a: np.ndarray) -> np.ndarray:
    """table read at fractional indices a >= 0 (past its end: its last value, 0)."""
    i = np.minimum(a.astype(np.int64), table.size - 2)
    return table[i] + np.minimum(a - i, 1.0) * (table[i + 1] - table[i])


def read_at(pcm: np.ndarray, positions: np.ndarray, previous: float | None = None) -> np.ndarray:
    """[M, ch] float32: ``pcm`` read at fractional input ``positions`` (any order,
    either direction) by windowed-sinc interpolation, anti-aliased by the speed
    (see READ_HALF); edges held. The speed of each output frame is its distance from
    the previous position (``previous`` is the one before ``positions[0]``). Each
    output frame is normalised by its kernel's sum (no ripple at DC)."""
    n = int(pcm.shape[0])
    ch = pcm.shape[1] if pcm.ndim == 2 else 1
    if n == 0 or positions.size == 0:
        return np.zeros((positions.size, ch), np.float32)
    kernels = _SCRATCH_NUMBA or None
    if kernels is not None and pcm.ndim == 2 and pcm.dtype == np.float32 and pcm.flags.c_contiguous:
        out = np.empty((positions.size, ch), np.float32)
        kernels.read(pcm, np.asarray(positions, dtype=np.float64), np.nan if previous is None else float(previous),
                     _READ_UNIT, _READ_SINC, _READ_WIN, _READ_STRIDE_W, out)
        return out
    p = np.clip(np.asarray(positions, dtype=np.float64), 0.0, n - 1)
    prev = p[0] if previous is None else min(max(float(previous), 0.0), n - 1)
    r = np.abs(np.diff(p, prepend=prev))
    np.maximum(r, 1.0, out=r)

    out = np.empty((p.size, ch))
    slow = r <= READ_STRIDE_STRETCH
    if slow.any():
        ps, rs = p[slow], r[slow]
        reach = np.minimum(READ_HALF * rs, READ_REACH)[:, None]
        j = np.floor(ps).astype(np.int64)[:, None] + np.arange(1 - READ_REACH, READ_REACH + 1)
        t = np.abs(j - ps[:, None])
        u = t / rs[:, None] * _READ_SINC_RES
        w = np.where(rs[:, None] <= 2.0, _lerp(_READ_UNIT, u),
                     _lerp(_READ_SINC, u) * _lerp(_READ_WIN, t / reach * _READ_WIN_RES))
        x = pcm[np.clip(j, 0, n - 1)]
        out[slow] = np.einsum("mk,mkc->mc", w, x) / w.sum(axis=1)[:, None]
    if not slow.all():
        fast = ~slow
        pf, d = p[fast], r[fast] / READ_STRIDE_STRETCH  # tap spacing, > 1
        span = READ_REACH * float(d.max()) + 2.0
        lo = int(np.floor(pf.min() - span))
        win = np.asarray(pcm[np.clip(np.arange(lo, int(np.ceil(pf.max() + span)) + 2), 0, n - 1)], dtype=np.float64)
        # first and second running integrals of the input, frame i held over [i - 0.5, i + 0.5)
        c1 = np.concatenate((np.zeros((1, ch)), np.cumsum(win, axis=0)))
        c2 = np.concatenate((np.zeros((1, ch)), np.cumsum(c1[:-1] + 0.5 * win, axis=0)))
        q = pf[:, None] + np.arange(-READ_REACH, READ_REACH + 1) * d[:, None] - lo + 0.5
        k = np.floor(q).astype(np.int64)
        f = (q - k)[..., None]
        g = c2[k] + f * (c1[k] + 0.5 * f * win[k])
        # tap = triangle (span 2 d) mean: second difference of the second integral
        taps = (g[:, 2:] - 2.0 * g[:, 1:-1] + g[:, :-2]) / (d * d)[:, None, None]
        out[fast] = np.einsum("k,mkc->mc", _READ_STRIDE_W, taps)
    return out.astype(np.float32)
