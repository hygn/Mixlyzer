import math
import numpy as np
from utils.color import key_cmap_color


def build_keystrip_buffer(
    key_segments,
    duration_sec: float,
    *,
    width: int = 4096,
) -> np.ndarray | None:
    """Return (key_image, time_grid, key_series) for the key strip."""
    duration = float(duration_sec or 0.0)
    if duration <= 0:
        return None

    seg_arr = None
    if key_segments is not None:
        try:
            seg_arr = np.asarray(key_segments, dtype=float)
        except Exception:
            seg_arr = None
        if seg_arr is not None and (seg_arr.ndim != 2 or seg_arr.shape[1] < 4):
            seg_arr = None

    strip_h = 1
    key_img = np.zeros((strip_h, width, 3), dtype=np.uint8)
    if seg_arr is None:
        return None

    for seg in seg_arr:
        pitch = seg[0]
        mode_flag = seg[1] if seg.size >= 2 else 0
        start = float(seg[2])
        end = float(seg[3])
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end <= start:
            continue
        if duration <= 1e-6:
            start_px = 0
            end_px = width
        else:
            start_px = int(np.clip(np.floor(start / duration * width), 0, width - 1))
            end_px = int(np.clip(math.ceil(end / duration * width), 0, width))
        if end_px <= start_px:
            end_px = min(width, start_px + 1)
        try:
            key_idx = int(pitch)
            if math.isfinite(mode_flag) and int(mode_flag) == 1:
                key_idx += 12
            color = key_cmap_color(key_idx)
        except Exception:
            continue
        key_img[:, start_px:end_px, :] = color

    return key_img.swapaxes(0, 1)
