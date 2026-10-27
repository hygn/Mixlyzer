import heapq
import unittest

import numpy as np

from audioengine.python import timestretch

timestretch.warm_timestretch_numba()
from audioengine.python import _timestretch_numba as kernels  # noqa: E402


def wrap(value):
    return value - 2.0 * np.pi * np.rint(value / (2.0 * np.pi))


def reference_integration(analysis, joint, previous_analysis, previous_synthesis, previous_joint, ha, hs, n_fft,
                          tolerance):
    """The phases integrated in order of magnitude with one heap of (-magnitude, item) pairs, item = 2 * bin +
    flag (0 in time, 1 in frequency): the order the kernel must keep."""
    channels, bins = analysis.shape
    out = analysis.copy()
    stretch = hs / ha
    floor = tolerance * max(1e-12, float(previous_joint.max()), float(joint.max()))
    remaining = joint > floor
    heap = [(-float(previous_joint[m]), 2 * m) for m in range(bins) if remaining[m]]
    heapq.heapify(heap)
    while remaining.any():
        _key, item = heapq.heappop(heap)
        m = item >> 1
        if item & 1 == 0:
            if remaining[m]:
                omega = 2.0 * np.pi * m / n_fft
                for c in range(channels):
                    out[c, m] = previous_synthesis[c, m] + omega * hs + stretch * wrap(
                        analysis[c, m] - previous_analysis[c, m] - omega * ha)
                remaining[m] = False
                heapq.heappush(heap, (-float(joint[m]), 2 * m + 1))
            continue
        for other, sign in ((m + 1, 1.0), (m - 1, -1.0)):
            if 0 <= other < bins and remaining[other]:
                for c in range(channels):
                    step = analysis[c, other] - analysis[c, m] if sign > 0 else analysis[c, m] - analysis[c, other]
                    out[c, other] = out[c, m] + sign * stretch * wrap(step)
                remaining[other] = False
                heapq.heappush(heap, (-float(joint[other]), 2 * other + 1))
    return wrap(out)


class PhaseKernelTests(unittest.TestCase):
    def test_integration_order_kept(self):
        rng = np.random.default_rng(0)
        bins, n_fft = 513, 1024
        for trial in range(6):
            with self.subTest(trial=trial):
                magnitude = rng.gamma(0.6, 1.0, (2, bins)).astype(np.float32)
                if trial % 2:
                    magnitude = np.round(magnitude * 4.0) / 4.0    # many equal magnitudes: ties by item
                joint = np.sqrt(np.sum(magnitude * magnitude, axis=0)).astype(np.float32)
                previous_joint = np.roll(joint, trial).astype(np.float32)
                analysis = rng.uniform(-np.pi, np.pi, (2, bins))
                previous_analysis = rng.uniform(-np.pi, np.pi, (2, bins))
                previous_synthesis = rng.uniform(-np.pi, np.pi, (2, bins))
                ha, hs = 480 + 20 * trial, 512
                got = kernels.rtpghi_frame(analysis, magnitude, joint, previous_analysis, previous_synthesis,
                                           previous_joint, ha, hs, n_fft, 1e-8, 0.0, False)
                want = reference_integration(analysis, joint, previous_analysis, previous_synthesis,
                                             previous_joint, ha, hs, n_fft, 1e-8)
                np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
