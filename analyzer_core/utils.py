import numpy as np

def gen_key_mask(major=True, minor=True, min_offset=0.5):
    # base templates
    maj_pattern = np.array([1,0,1,0,1,1,0,1,0,1,0,1])   # Ionian
    min_pattern = np.array([1,0,1,1,0,1,0,1,1,0,1,0])   # Aeolian
    keyprob = []
    # 12 major (Camelot B ring)
    if major:
        for root in range(12):
            keyprob.append(np.roll(maj_pattern, root).tolist())
    # minor
    if minor:
        for root in range(12):
            keyprob.append(np.roll(min_pattern, root).tolist())
    
    keyprob = np.array(keyprob)
    keyprob = np.add(keyprob, min_offset)
    keyprob = np.clip(keyprob, 0, 1)
    return keyprob

def fuse_key_mode_to_24_path(key_path12, mode_path01):
    k = np.asarray(key_path12, dtype=int).ravel()
    m = np.asarray(mode_path01, dtype=int).ravel()
    rel_minor_pc = (k + 9) % 12
    out = np.where(m == 0, k, 12 + rel_minor_pc)
    return out

import numpy as np

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    x_pad = np.pad(x, pad, mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")

    return y
