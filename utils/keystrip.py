import math
import numpy as np
from utils.color import key_cmap_color


def build_keystrip_buffer(
    key_segments,
    duration_sec: float,
    *,
    width: int = 4096,
) -> np.ndarray | None:
    """Key strip image (width x 1 x RGB) of ``key_segments`` rows
    [pitch, mode, start, end] over ``duration_sec``; None without segments."""
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


def render_keystrip(features: dict, duration_sec: float | None = None) -> np.ndarray | None:
    """The key strip, rendered just in time from the stored key segments.

    The stored ``key_np`` image is deprecated and never read: it could go stale
    against the segments (edits, reanalysis over a different length)."""
    if not isinstance(features, dict):
        return None
    duration = duration_sec if duration_sec else features.get("duration_sec")
    return build_keystrip_buffer(features.get("key_segments"), float(duration or 0.0))


def key_segment_rows(key_segments) -> list[tuple[int, float, float]]:
    """(key index 0..23, start, end) of the valid rows of [pitch, mode, start, end], by start."""
    try:
        arr = np.asarray(key_segments, dtype=float)
    except Exception:
        return []
    if arr.ndim != 2 or arr.shape[1] < 4:
        return []
    rows = []
    for pitch, mode, start, end in arr[:, :4]:
        if not (math.isfinite(pitch) and math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        idx = int(pitch) % 12 + (12 if math.isfinite(mode) and int(mode) == 1 else 0)
        rows.append((idx, float(start), float(end)))
    rows.sort(key=lambda r: r[1])
    return rows


def key_segment_label(idx: int) -> str:
    """Key strip label: Camelot / classical, as in the track info ("3B / C#")."""
    from utils.labels import idx_to_labels
    cam, cls = idx_to_labels(idx)
    return f"{cam}\u2009/\u2009{cls}"
