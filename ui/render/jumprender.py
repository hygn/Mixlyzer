from __future__ import annotations

import numpy as np


def jump_renderer(samp: np.ndarray, sr: int, st_s: float, ed_s: float, crossfade_sample: int) -> tuple[np.ndarray, int]:
    "Prerender JumpCUE audio samples (mono). Handles both forward and backward jumps."
    st_samp = int(max(0.0, st_s) * sr)
    ed_samp = int(max(0.0, ed_s) * sr)
    n_samp = int(samp.shape[0])

    st_samp = min(st_samp, n_samp)
    ed_samp = min(ed_samp, n_samp)

    cf_samp = int(max(0, crossfade_sample))
    # Crossfade cannot exceed the available samples around the jump points.
    cf_samp = min(cf_samp, st_samp, max(0, n_samp - ed_samp))

    if cf_samp <= 0:
        # No crossfade available; simple splice/append.
        samp_rendered = np.concatenate((samp[:st_samp], samp[ed_samp:]))
        return samp_rendered, sr

    pre_end = max(0, st_samp - cf_samp)
    samp_st = samp[:pre_end]
    samp_cf_st = samp[pre_end:st_samp]
    samp_cf_ed = samp[ed_samp : ed_samp + cf_samp]

    theta = np.linspace(0, np.pi / 2, cf_samp, endpoint=False)
    cf_st_gain = np.cos(theta)
    cf_ed_gain = np.sin(theta)
    samp_cf = (samp_cf_st * cf_st_gain) + (samp_cf_ed * cf_ed_gain)

    samp_ed = samp[ed_samp:]
    samp_rendered = np.concatenate((samp_st, samp_cf, samp_ed))
    return samp_rendered, sr
