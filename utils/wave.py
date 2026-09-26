import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d


# Overview waveform size: ~WAVE_PREVIEW_WIDTH columns.
WAVE_PREVIEW_WIDTH = 4096
WAVE_PREVIEW_HEIGHT = 128


def frame_minmax(y: np.ndarray, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Min and max of each ``hop``-sample frame; the last frame may be shorter."""
    n = len(y)
    n_frames = int(np.ceil(n / hop))
    n_full = n // hop
    mins, maxs = np.empty(n_frames, np.float32), np.empty(n_frames, np.float32)
    frames = np.asarray(y[: n_full * hop]).reshape(n_full, hop)
    mins[:n_full] = frames.min(axis=1)
    maxs[:n_full] = frames.max(axis=1)
    if n_frames > n_full:
        mins[n_full] = np.min(y[n_full * hop:])
        maxs[n_full] = np.max(y[n_full * hop:])
    return mins, maxs


def wave_colors(lo, mid, hi, white_threshold=0.05) -> np.ndarray:
    """Per-column RGB (uint8, ``[T, 3]``) from the normalized low / mid / high band levels."""
    rgb = np.stack([lo, mid, hi], axis=1)
    magnitudes = np.linalg.norm(rgb, axis=1)
    threshold = max(0.0, float(white_threshold))
    low_mask = magnitudes < threshold
    scale = np.clip(np.max(rgb, axis=1), 1e-8, None)
    rgb_norm = np.ones_like(rgb)
    active_mask = ~low_mask
    rgb_norm[active_mask] = rgb[active_mask] / scale[active_mask, None]
    rgb_norm[low_mask] = [1.0, 1.0, 1.0]
    if np.any(active_mask):
        active_rgb = rgb_norm[active_mask]
        sat = 1.0 - np.min(active_rgb, axis=1)
        sat_floor = 0.7
        boost_mask = sat < sat_floor
        if np.any(boost_mask):
            boost_idx = np.where(boost_mask)[0]
            sat_safe = np.clip(sat[boost_idx], 1e-8, None)
            boost = sat_floor / sat_safe
            boosted = 1.0 - (1.0 - active_rgb[boost_idx]) * boost[:, None]
            active_rgb[boost_idx] = np.clip(boosted, 0.0, 1.0)
        rgb_norm[active_mask] = active_rgb
    return (np.clip(rgb_norm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def build_wave_preview(
    y: np.ndarray,
    sample_rate: int,
    bands: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    frame_hop: int,
    magnitude: np.ndarray | None = None,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    width: int = WAVE_PREVIEW_WIDTH,
    height_px: int = WAVE_PREVIEW_HEIGHT,
    white_threshold: float = 0.05,
) -> np.ndarray:
    """Overview waveform image ``[columns, height_px, 3]`` (uint8) of mono ``y``.

    Short frames of ``frame_hop`` samples are drawn as vertical bars from the
    frame's min to its max, colored by the RMS of the low / mid / high
    ``bands`` (Hz, each scaled to its maximum over the track). Each of the
    ~``width`` columns is the average of its frames' bars: a pixel's color is
    the sum of the colors of the bars covering it over the number of frames
    (area anti-aliasing), computed with cumulative sums instead of drawing the
    full-resolution image. The band RMS comes from the magnitude STFT
    ``magnitude`` (``n_fft`` / ``hop_length``, centered frames), computed here
    when omitted.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    frame_hop = max(1, int(frame_hop))
    min_env, max_env = frame_minmax(y, frame_hop)
    n_frames = min_env.size
    if n_frames == 0:
        return np.zeros((0, height_px, 3), dtype=np.uint8)

    if magnitude is None:
        magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True))
    power = np.square(magnitude, dtype=np.float32)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    stft_centers = np.arange(power.shape[1]) * float(hop_length)
    frame_centers = (np.arange(n_frames) + 0.5) * float(frame_hop)
    levels = []
    for low, high in bands:
        rms = np.sqrt(power[(frequencies >= low) & (frequencies < high)].sum(axis=0))
        rms = np.interp(frame_centers, stft_centers, rms)
        peak = float(rms.max()) if rms.size else 0.0
        levels.append(rms / peak if peak > 1e-12 else rms)
    colors = wave_colors(*levels, white_threshold).astype(np.float64)

    per_column = int(np.ceil(n_frames / max(1, int(width))))
    image = wave_coverage_columns(min_env, max_env, colors, np.arange(n_frames) // per_column, height_px)
    image = gaussian_filter1d(image, sigma=0.8, axis=0)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def wave_coverage_columns(
    min_env: np.ndarray,
    max_env: np.ndarray,
    colors: np.ndarray,
    column: np.ndarray,
    height_px: int,
) -> np.ndarray:
    """Average of thin per-frame bars in each column: float image ``[columns, height_px, 3]``.

    Frame ``i`` is a bar from ``min_env[i]`` to ``max_env[i]`` (-1..1; row 0 is the top, a flat frame is one pixel on the centre row) colored ``colors[i]`` (RGB 0..255), and falls in ``column[i]``
    (non-decreasing, from 0). A pixel is the sum of the colors of the bars
    covering it over the column's frame count (area anti-aliasing).
    """
    H = int(height_px)
    half = H // 2
    top = np.clip(half - (np.nan_to_num(max_env) * half).astype(int), 0, H - 1)
    bottom = np.clip(half - (np.nan_to_num(min_env) * half).astype(int), 0, H - 1)
    first, last = np.minimum(top, bottom), np.maximum(top, bottom)
    flat = top == bottom
    first[flat] = last[flat] = min(max(0, half), H - 1)

    # +color where a bar starts, -color after it ends, cumulative sum down the rows.
    column = np.asarray(column, dtype=np.int64)
    n_columns = int(column[-1]) + 1 if column.size else 0
    start_index = column * (H + 1) + first
    end_index = column * (H + 1) + last + 1
    image = np.empty((n_columns, H, 3))
    for channel in range(3):
        edges = np.bincount(start_index, colors[:, channel], minlength=n_columns * (H + 1))
        edges -= np.bincount(end_index, colors[:, channel], minlength=n_columns * (H + 1))
        image[:, :, channel] = np.cumsum(edges.reshape(n_columns, H + 1), axis=1)[:, :H]
    frames_per_column = np.maximum(np.bincount(column, minlength=n_columns), 1).astype(np.float64)
    image /= frames_per_column[:, np.newaxis, np.newaxis]
    return image
