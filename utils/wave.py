import numpy as np
from scipy.signal import get_window

from analyzer_core.dsp.conversion import fft_frequencies
from analyzer_core.dsp.spectral import stft


# Overview waveform: stored at WAVE_OVERVIEW_COLUMNS columns over the track, as
# the peak, the body and the low / mid / high band levels of each column (just
# the levels, so the overview is drawn in any style when shown), and drawn at
# WAVE_OVERVIEW_RENDER_COLUMNS columns.
WAVE_OVERVIEW_COLUMNS = 16384
WAVE_OVERVIEW_RENDER_COLUMNS = 4096
# Channels of the stored overview. The body is the mean peak of the column's
# WAVE_BODY_FRAME_SEC frames: an RGB bar is solid up to it and fades out to the peak.
WAVE_PEAK, WAVE_BODY, WAVE_LOW, WAVE_MID, WAVE_HIGH = range(5)
WAVE_CHANNELS = 5
WAVE_BODY_FRAME_SEC = 0.001

WAVEFORM_STYLES = ("rgb", "3band")
# RGB overview bars reach this fraction of the lane at the track's peak.
RGB_OVERVIEW_PEAK = 0.99
# RGB bars fade from their body to their peak as (1 - t) ** RGB_FADE_GAMMA:
# below 1 keeps the tops visible.
RGB_FADE_GAMMA = 0.5
# Three-band colors and band weights of rekordbox / CDJ-3000 (after Deep
# Symmetry's beat-link): the preview stacks the bands from the bottom (low,
# low + mid, low + mid + high); the scrolling waveform draws them mirrored, each
# to its own height, with the low / mid overlap in brown and the highs on top.
THREE_BAND_LOW = (32, 83, 217)
THREE_BAND_LOW_AND_MID = (169, 107, 39)
THREE_BAND_MID = (242, 170, 60)
THREE_BAND_HIGH = (255, 255, 255)
THREE_BAND_PREVIEW_WEIGHTS = (0.49, 0.32, 0.25)
THREE_BAND_DETAIL_WEIGHTS = (0.4, 0.3, 0.06)
# The three bands share one scale (the loudest band's track maximum), so a
# band's height shows how loud it is against the others.
# RMS window of the scrolling three-band waveform: a CDJ frame (1/75 s), long
# enough for the low band's periods.
THREE_BAND_DETAIL_RMS_SEC = 1.0 / 75.0


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


def build_wave_overview(
    y: np.ndarray,
    sample_rate: int,
    bands: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    magnitude: np.ndarray | None = None,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    columns: int = WAVE_OVERVIEW_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Stored overview of mono ``y``: ``(levels uint8 [C, 5], ref float32 [5])``.

    Per column: the peak |y|, the mean |y| peak of its WAVE_BODY_FRAME_SEC frames
    and the RMS of the low / mid / high ``bands`` (Hz),
    from the magnitude STFT ``magnitude`` (``n_fft`` / ``hop_length``, centered
    frames; computed here when omitted). ``ref`` is each channel's maximum over
    the track (sample units); a level is sqrt(value / ref) * 255.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n_columns = max(1, min(int(columns), y.size))
    if y.size == 0:
        return np.zeros((0, WAVE_CHANNELS), np.uint8), np.zeros(WAVE_CHANNELS, np.float32)
    starts = (np.arange(n_columns, dtype=np.int64) * y.size) // n_columns
    values = np.empty((n_columns, WAVE_CHANNELS), np.float64)
    values[:, WAVE_PEAK] = np.maximum.reduceat(np.abs(y), starts)
    frame = max(1, int(round(WAVE_BODY_FRAME_SEC * sample_rate)))
    frame_min, frame_max = frame_minmax(y, frame)
    frame_column = np.searchsorted(starts, np.arange(frame_min.size) * frame, side="right") - 1
    values[:, WAVE_BODY] = (np.bincount(frame_column, np.maximum(-frame_min, frame_max), minlength=n_columns)
                            / np.maximum(np.bincount(frame_column, minlength=n_columns), 1))

    if magnitude is None:
        magnitude = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length, center=True))
    power = np.square(magnitude, dtype=np.float32)
    frequencies = fft_frequencies(sr=sample_rate, n_fft=n_fft)
    # One-sided Parseval with the Hann window: mean square of the band in the frame.
    to_mean_square = 2.0 / (n_fft * float(np.sum(np.square(get_window("hann", n_fft, fftbins=True)))))
    frame_centers = np.arange(power.shape[1]) * float(hop_length)
    column_centers = starts + 0.5 * np.diff(np.append(starts, y.size))
    for channel, (low, high) in zip((WAVE_LOW, WAVE_MID, WAVE_HIGH), bands):
        mean_square = power[(frequencies >= low) & (frequencies < high)].sum(axis=0) * to_mean_square
        values[:, channel] = np.sqrt(np.interp(column_centers, frame_centers, mean_square))

    ref = values.max(axis=0)
    normalized = values / np.where(ref > 0.0, ref, 1.0)
    levels = np.rint(np.sqrt(np.clip(normalized, 0.0, 1.0)) * 255.0).astype(np.uint8)
    return levels, ref.astype(np.float32)


def overview_values(levels: np.ndarray, columns: int = WAVE_OVERVIEW_RENDER_COLUMNS) -> np.ndarray:
    """Stored levels -> ``[columns, 4]`` values 0..1 (each channel over its track
    maximum): the peak is the column's maximum, the bands its mean."""
    values = np.square(np.asarray(levels, dtype=np.float64) / 255.0)
    n = values.shape[0]
    n_columns = max(1, min(int(columns), n))
    starts = (np.arange(n_columns, dtype=np.int64) * n) // n_columns
    counts = np.diff(np.append(starts, n)).astype(np.float64)
    out = np.add.reduceat(values, starts, axis=0) / counts[:, np.newaxis]
    out[:, WAVE_PEAK] = np.maximum.reduceat(values[:, WAVE_PEAK], starts)
    return out


def paint_layers(layers, height: int, *, mirrored: bool = False) -> np.ndarray:
    """Image ``[columns, height, 3]`` (uint8) of bars painted in order over black.

    ``layers``: ``(heights [C] 0..1, color [3] or [C, 3])``. A bar rises from the
    first row (``mirrored``: spreads from the centre both ways); its top pixel is
    covered by its fractional height.
    """
    H = int(height)
    first = np.asarray(layers[0][0])
    image = np.zeros((first.size, H, 3), dtype=np.float64)
    if mirrored:
        half = 0.5 * H
        distance = np.abs(np.arange(H) + 0.5 - half) - 0.5
        extent = half
    else:
        distance = np.arange(H, dtype=np.float64)
        extent = float(H)
    for heights, color in layers:
        cover = np.clip(np.asarray(heights, dtype=np.float64)[:, np.newaxis] * extent - distance[np.newaxis, :], 0.0, 1.0)
        color = np.asarray(color, dtype=np.float64)
        color = color[np.newaxis, np.newaxis, :] if color.ndim == 1 else color[:, np.newaxis, :]
        image += cover[:, :, np.newaxis] * (color - image)
    return np.clip(image + 0.5, 0.0, 255.0).astype(np.uint8)


def three_band_detail_layers(low, mid, high) -> list:
    """Mirrored three-band layers of the scrolling waveform from band levels 0..1."""
    w_low, w_mid, w_high = (w / THREE_BAND_DETAIL_WEIGHTS[0] for w in THREE_BAND_DETAIL_WEIGHTS)
    low, mid, high = np.asarray(low) * w_low, np.asarray(mid) * w_mid, np.asarray(high) * w_high
    return [
        (low, THREE_BAND_LOW),
        (mid, THREE_BAND_MID),
        (np.minimum(low, mid), THREE_BAND_LOW_AND_MID),
        (high, THREE_BAND_HIGH),
    ]


def _faded_bars(peak, body, colors, height: int, columns: int, *, mirrored: bool) -> np.ndarray:
    """A bar of each stored column (heights 0..1 of the lane), solid up to
    ``body`` and fading out to ``peak``, colored ``colors``; area-averaged into
    ``columns`` image columns (drawn at the display's pixels, so not blurred)."""
    n = peak.size
    H = int(height)
    rows = np.arange(H) + 0.5
    position = np.abs(rows - 0.5 * H) / (0.5 * H) if mirrored else rows / H
    cover = np.clip((peak[:, np.newaxis] - position) / np.maximum(peak - body, 1e-6)[:, np.newaxis], 0.0, 1.0)
    cover **= RGB_FADE_GAMMA
    n_columns = max(1, min(int(columns), n))
    starts = (np.arange(n_columns, dtype=np.int64) * n) // n_columns
    counts = np.diff(np.append(starts, n)).astype(np.float32)[:, np.newaxis, np.newaxis]
    image = np.add.reduceat(cover.astype(np.float32)[:, :, np.newaxis] * colors.astype(np.float32)[:, np.newaxis, :],
                            starts, axis=0)
    return np.clip(image / counts + 0.5, 0.0, 255.0).astype(np.uint8)


def render_wave_overview(
    levels: np.ndarray,
    ref: np.ndarray,
    style: str,
    height: int,
    *,
    columns: int = WAVE_OVERVIEW_RENDER_COLUMNS,
) -> np.ndarray:
    """Overview image ``[columns, height, 3]`` of the stored levels / ``ref``,
    bars rising from the first row, in ``style`` ('rgb': bars of the peak
    colored by the band levels, each over its track maximum; '3band': the bands
    stacked on one scale)."""
    if style != "3band":
        values = np.square(np.asarray(levels, dtype=np.float64) / 255.0)
        ref = np.asarray(ref, dtype=np.float64)
        peak = values[:, WAVE_PEAK] * RGB_OVERVIEW_PEAK
        body = np.minimum(values[:, WAVE_BODY] * ref[WAVE_BODY] / (ref[WAVE_PEAK] or 1.0) * RGB_OVERVIEW_PEAK, peak)
        colors = wave_colors(values[:, WAVE_LOW], values[:, WAVE_MID], values[:, WAVE_HIGH])
        return _faded_bars(peak, body, colors, height, columns, mirrored=False)
    values = overview_values(levels, columns)
    low, mid, high = values[:, WAVE_LOW], values[:, WAVE_MID], values[:, WAVE_HIGH]
    band_ref = np.asarray(ref, dtype=np.float64)[WAVE_LOW:WAVE_HIGH + 1]
    low, mid, high = (v * r / (band_ref.max() or 1.0) for v, r in zip((low, mid, high), band_ref))
    w_low, w_mid, w_high = THREE_BAND_PREVIEW_WEIGHTS
    stack_low = w_low * low
    stack_mid = stack_low + w_mid * mid
    stack_high = stack_mid + w_high * high
    scale = float(stack_high.max()) or 1.0
    layers = [
        (stack_high / scale, THREE_BAND_HIGH),
        (stack_mid / scale, THREE_BAND_MID),
        (stack_low / scale, THREE_BAND_LOW),
    ]
    return paint_layers(layers, height)


def render_wave_detail_preview(
    levels: np.ndarray,
    ref: np.ndarray,
    style: str,
    height: int,
    *,
    columns: int = WAVE_OVERVIEW_RENDER_COLUMNS,
) -> np.ndarray:
    """Stand-in for the main view's detail waveform until its chunks are
    rendered, drawn like them from the stored levels (mirrored, in sample units):
    RGB bars colored by the band levels (each over its track maximum: closer to
    the chunks' colors, averaged from short frames, than the column RMS), 3band
    the layers of the scrolling waveform (bands on one scale)."""
    if style != "3band":
        values = np.square(np.asarray(levels, dtype=np.float64) / 255.0)
        amplitude = values * np.asarray(ref, dtype=np.float64)
        peak = np.clip(amplitude[:, WAVE_PEAK], 0.0, 1.0)
        body = np.minimum(amplitude[:, WAVE_BODY], peak)
        colors = wave_colors(values[:, WAVE_LOW], values[:, WAVE_MID], values[:, WAVE_HIGH], white_threshold=0.0)
        return _faded_bars(peak, body, colors, height, columns, mirrored=True)
    values = overview_values(levels, columns)
    band_ref = np.asarray(ref, dtype=np.float64)[WAVE_LOW:WAVE_HIGH + 1]
    scale = float(band_ref.max()) or 1.0
    low, mid, high = (values[:, channel] * r / scale for channel, r in zip((WAVE_LOW, WAVE_MID, WAVE_HIGH), band_ref))
    return paint_layers(three_band_detail_layers(low, mid, high), height, mirrored=True)


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
