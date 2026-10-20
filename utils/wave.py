import numpy as np
from scipy.signal import get_window

from analyzer_core.dsp.conversion import fft_frequencies
from analyzer_core.dsp.spectral import stft


# Overview waveform, stored by the analysis from its own decode at WAVE_OVERVIEW_SAMPLE_RATE:
# - wave_minmax_np: the min and max sample of WAVE_MINMAX_COLUMNS columns over the track (int8, signed square
#   root of the sample over wave_minmax_ref, the track's peak |sample|);
# - wave_bands_np: the mean-square power of WAVE_BAND_COUNT log-spaced bands (WAVE_BAND_EDGES, Hz) in
#   WAVE_BAND_COLUMNS columns (uint8, the 4th root of the power over wave_bands_ref, each band's track maximum).
# The low / mid / high levels are summed from the bands when drawn (the configured ranges, each band weighted by
# its log-frequency overlap with a range), so a change of the ranges redraws at once, without re-analysis.
# Drawn at WAVE_OVERVIEW_RENDER_COLUMNS columns.
WAVE_MINMAX_COLUMNS = 32768
WAVE_BAND_COLUMNS = 8192
WAVE_BAND_COUNT = 12
WAVE_BAND_EDGES = np.geomspace(20.0, 20000.0, WAVE_BAND_COUNT + 1)
WAVE_OVERVIEW_RENDER_COLUMNS = 4096
# The overview is measured from its own decode at this rate, whatever the analysis rate: bands reach 20 kHz.
WAVE_OVERVIEW_SAMPLE_RATE = 48000
WAVE_OVERVIEW_KEYS = ("wave_minmax_np", "wave_minmax_ref", "wave_bands_np", "wave_bands_ref")
# The detail (scrolling) waveform is drawn from the audio; it reads one stored number: the three-band layers'
# common scale (wave_detail_band_ref: the low / mid / high RMS track maxima of the ranges set at analysis, each the
# maximum of the range's RMS at 16384 column centres).

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
# RGB detail waveform colors: the band levels smoothed over this time (so the
# color changes gradually), and a lower saturation floor than the overview's.
RGB_DETAIL_COLOR_SMOOTH_SEC = 0.05
RGB_DETAIL_SAT_FLOOR = 0.6
# The detail waveform's centre line: where the waveform leaves the centre row
# dark (quiet audio), it is drawn in this colour, fading out as the waveform's
# own pixel there brightens up to WAVE_CENTER_LINE_FADE (its brightest channel).
WAVE_CENTER_LINE_RGB = (255, 255, 255)
WAVE_CENTER_LINE_FADE = 64.0


def add_center_line(image: np.ndarray) -> np.ndarray:
    """``image`` ``[columns, height, 3]`` (a mirrored waveform) with the centre
    line on its centre row (the row a flat frame is drawn on); a float copy."""
    out = np.array(image, dtype=np.float64)
    if out.ndim != 3 or out.shape[1] == 0:
        return out
    row = out[:, out.shape[1] // 2, :]
    alpha = np.clip(1.0 - row.max(axis=1) / WAVE_CENTER_LINE_FADE, 0.0, 1.0)[:, np.newaxis]
    row += alpha * (np.asarray(WAVE_CENTER_LINE_RGB, dtype=np.float64) - row)
    return out


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


def wave_colors(lo, mid, hi, white_threshold=0.05, sat_floor=0.7) -> np.ndarray:
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
) -> dict[str, np.ndarray]:
    """Stored overview of mono ``y`` (the WAVE_OVERVIEW_KEYS arrays): the min / max of WAVE_MINMAX_COLUMNS
    columns and the WAVE_BAND_COUNT bands' mean-square power in WAVE_BAND_COLUMNS columns, from the magnitude
    STFT ``magnitude`` (``n_fft`` / ``hop_length``, centered frames; computed here when omitted); and the detail
    waveform's three-band scale for the low / mid / high ``bands`` (Hz)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return {"wave_minmax_np": np.zeros((0, 2), np.int8), "wave_minmax_ref": np.zeros(1, np.float32),
                "wave_bands_np": np.zeros((0, WAVE_BAND_COUNT), np.uint8),
                "wave_bands_ref": np.zeros(WAVE_BAND_COUNT, np.float32),
                "wave_detail_band_ref": np.zeros(3, np.float32)}
    n = max(1, min(WAVE_MINMAX_COLUMNS, y.size))
    starts = (np.arange(n, dtype=np.int64) * y.size) // n
    minmax = np.stack([np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts)], axis=1).astype(np.float64)
    peak = float(np.abs(minmax).max()) or 1.0
    companded = np.sign(minmax) * np.sqrt(np.abs(minmax) / peak) * 127.0
    out = {"wave_minmax_np": np.rint(companded).astype(np.int8), "wave_minmax_ref": np.asarray([peak], np.float32)}

    if magnitude is None:
        magnitude = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length, center=True))
    power = np.square(magnitude, dtype=np.float32)
    frequencies = fft_frequencies(sr=sample_rate, n_fft=n_fft)
    # One-sided Parseval with the Hann window: mean square of the band in the frame.
    to_mean_square = 2.0 / (n_fft * float(np.sum(np.square(get_window("hann", n_fft, fftbins=True)))))
    band_of_bin = np.searchsorted(WAVE_BAND_EDGES, frequencies, side="right") - 1
    frame_power = np.zeros((WAVE_BAND_COUNT, power.shape[1]), np.float64)
    for band in range(WAVE_BAND_COUNT):
        frame_power[band] = power[band_of_bin == band].sum(axis=0) * to_mean_square
    # each column: the mean of the frames centred in it (interpolated where a column holds no frame)
    m = max(1, min(WAVE_BAND_COLUMNS, y.size))
    frame_centers = np.arange(power.shape[1]) * float(hop_length)
    frame_column = np.clip((frame_centers * m / y.size).astype(np.int64), 0, m - 1)
    counts = np.bincount(frame_column, minlength=m)
    columns = np.zeros((m, WAVE_BAND_COUNT), np.float64)
    for band in range(WAVE_BAND_COUNT):
        columns[:, band] = np.bincount(frame_column, frame_power[band], minlength=m)
    columns /= np.maximum(counts, 1)[:, np.newaxis]
    empty = counts == 0
    if empty.any() and power.shape[1]:
        centers = (np.arange(m) + 0.5) * y.size / m
        for band in range(WAVE_BAND_COUNT):
            columns[empty, band] = np.interp(centers[empty], frame_centers, frame_power[band])
    ref = columns.max(axis=0)
    levels = np.power(np.clip(columns / np.where(ref > 0.0, ref, 1.0), 0.0, 1.0), 0.25) * 255.0
    out["wave_bands_np"] = np.rint(levels).astype(np.uint8)
    out["wave_bands_ref"] = ref.astype(np.float32)
    # the detail waveform's scale, measured as before: each range's RMS at 16384 column centres, its maximum
    n_old = max(1, min(16384, y.size))
    old_starts = (np.arange(n_old, dtype=np.int64) * y.size) // n_old
    old_centers = old_starts + 0.5 * np.diff(np.append(old_starts, y.size))
    detail_ref = []
    for low, high in bands:
        mean_square = power[(frequencies >= low) & (frequencies < high)].sum(axis=0) * to_mean_square
        detail_ref.append(float(np.sqrt(np.interp(old_centers, frame_centers, mean_square)).max()))
    out["wave_detail_band_ref"] = np.asarray(detail_ref, np.float32)
    return out


def band_ranges(viewconfig) -> tuple[tuple[float, float], ...]:
    """The configured low / mid / high ranges (Hz)."""
    return tuple((float(low), float(high)) for low, high in (viewconfig.env_lo, viewconfig.env_mid, viewconfig.env_hi))


def band_weights(ranges) -> np.ndarray:
    """``[3, WAVE_BAND_COUNT]``: the share of each stored band inside each range, by log-frequency overlap."""
    log_edges = np.log(WAVE_BAND_EDGES)
    weights = np.zeros((len(ranges), WAVE_BAND_COUNT))
    for r, (low, high) in enumerate(ranges):
        if high <= low or high <= 0.0:
            continue
        lo, hi = np.log(max(float(low), 1e-3)), np.log(float(high))
        overlap = np.clip(np.minimum(hi, log_edges[1:]) - np.maximum(lo, log_edges[:-1]), 0.0, None)
        weights[r] = overlap / np.diff(log_edges)
    return weights


def has_wave_overview(features) -> bool:
    return isinstance(features, dict) and all(k in features for k in WAVE_OVERVIEW_KEYS)


def wave_overview_source(features):
    """The stored array the overview is drawn from (its identity tells a cached drawing is current)."""
    return features.get("wave_minmax_np") if isinstance(features, dict) else None


def _stored_minmax(features) -> np.ndarray:
    """``[N, 2]`` min / max per column in sample units."""
    q = np.asarray(features["wave_minmax_np"], dtype=np.float64) / 127.0
    return np.sign(q) * np.square(q) * float(np.asarray(features["wave_minmax_ref"]).reshape(-1)[0])


def _stored_band_rms(features, ranges) -> np.ndarray:
    """``[M, 3]`` low / mid / high RMS per band column in sample units, summed from the stored bands."""
    power = np.power(np.asarray(features["wave_bands_np"], dtype=np.float64) / 255.0, 4.0)
    power *= np.asarray(features["wave_bands_ref"], dtype=np.float64)[np.newaxis, :]
    return np.sqrt(power @ band_weights(ranges).T)


def _peak_body(minmax: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per group of min / max columns: the peak |sample| and the body (the mean half-range)."""
    n = minmax.shape[0]
    g = max(1, min(int(groups), n))
    starts = (np.arange(g, dtype=np.int64) * n) // g
    counts = np.diff(np.append(starts, n)).astype(np.float64)
    peak = np.maximum.reduceat(np.abs(minmax).max(axis=1), starts)
    body = np.add.reduceat(0.5 * (minmax[:, 1] - minmax[:, 0]), starts) / counts
    return peak, np.minimum(body, peak)


def _column_means(values: np.ndarray, columns: int) -> np.ndarray:
    n = values.shape[0]
    c = max(1, min(int(columns), n))
    starts = (np.arange(c, dtype=np.int64) * n) // c
    return np.add.reduceat(values, starts, axis=0) / np.diff(np.append(starts, n)).astype(np.float64)[:, np.newaxis]


def wave_detail_band_ref(features) -> float | None:
    """The detail waveform's three-band scale: the loudest band's track maximum RMS (sample units, the ranges set
    at analysis)."""
    if not isinstance(features, dict) or "wave_detail_band_ref" not in features:
        return None
    return float(np.max(np.asarray(features["wave_detail_band_ref"])))


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
    features: dict,
    ranges,
    style: str,
    height: int,
    *,
    columns: int = WAVE_OVERVIEW_RENDER_COLUMNS,
) -> np.ndarray:
    """Overview image ``[columns, height, 3]`` of a track's stored overview, bars rising from the first row, in
    ``style`` ('rgb': bars of the peak, solid to the body, colored by the low / mid / high ``ranges`` each over its
    track maximum; '3band': the three bands stacked on one scale)."""
    rms = _stored_band_rms(features, ranges)
    if style != "3band":
        peak, body = _peak_body(_stored_minmax(features), rms.shape[0])
        top = float(peak.max()) or 1.0
        peak, body = peak / top * RGB_OVERVIEW_PEAK, body / top * RGB_OVERVIEW_PEAK
        band_max = rms.max(axis=0)
        normalized = rms / np.where(band_max > 0.0, band_max, 1.0)
        colors = wave_colors(normalized[:, 0], normalized[:, 1], normalized[:, 2])
        return _faded_bars(peak, body, colors, height, columns, mirrored=False)
    values = _column_means(rms, columns) / (float(rms.max()) or 1.0)
    w_low, w_mid, w_high = THREE_BAND_PREVIEW_WEIGHTS
    stack_low = w_low * values[:, 0]
    stack_mid = stack_low + w_mid * values[:, 1]
    stack_high = stack_mid + w_high * values[:, 2]
    scale = float(stack_high.max()) or 1.0
    return paint_layers([(stack_high / scale, THREE_BAND_HIGH), (stack_mid / scale, THREE_BAND_MID),
                         (stack_low / scale, THREE_BAND_LOW)], height)


def render_wave_detail_preview(
    features: dict,
    ranges,
    style: str,
    height: int,
    *,
    columns: int = WAVE_OVERVIEW_RENDER_COLUMNS,
) -> np.ndarray:
    """Stand-in for the main view's detail waveform until its chunks are rendered, drawn like them from the
    stored overview (mirrored, sample units): RGB the min / max bars colored as the chunks are (the square roots
    of the band RMS), 3band the layers of the scrolling waveform (bands on one scale)."""
    rms = _stored_band_rms(features, ranges)
    if style != "3band":
        minmax = np.clip(_stored_minmax(features), -1.0, 1.0)
        n = minmax.shape[0]
        band_col = np.minimum((np.arange(n, dtype=np.int64) * rms.shape[0]) // max(n, 1), rms.shape[0] - 1)
        amp = np.sqrt(rms[band_col])
        colors = wave_colors(amp[:, 0], amp[:, 1], amp[:, 2], white_threshold=0.0,
                             sat_floor=RGB_DETAIL_SAT_FLOOR).astype(np.float64)
        column = (np.arange(n, dtype=np.int64) * max(1, min(int(columns), n))) // n
        image = wave_coverage_columns(minmax[:, 0], minmax[:, 1], colors, column, height,
                                      flat_color=WAVE_CENTER_LINE_RGB)
        return np.clip(image + 0.5, 0.0, 255.0).astype(np.uint8)
    values = _column_means(rms, columns) / (float(rms.max()) or 1.0)
    return paint_layers(three_band_detail_layers(values[:, 0], values[:, 1], values[:, 2]), height, mirrored=True)


def wave_coverage_columns(
    min_env: np.ndarray,
    max_env: np.ndarray,
    colors: np.ndarray,
    column: np.ndarray,
    height_px: int,
    flat_color: tuple[float, float, float] | None = None,
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
    if flat_color is not None:
        # a frame too quiet to span a pixel is drawn in ``flat_color``, not its band colour
        colors = np.array(colors, dtype=np.float64)
        colors[flat] = flat_color

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
