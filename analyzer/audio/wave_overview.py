"""The overview waveform's measurement: the stored overview of a track (its min / max columns and its bands' power,
the detail waveform's three-band scale), from the analysis's own decode at WAVE_OVERVIEW_SAMPLE_RATE."""
import numpy as np
from scipy.signal import get_window

from analyzer.audio.conversion import fft_frequencies
from analyzer.audio.spectral import stft


# Overview waveform, stored by the analysis from its own decode at WAVE_OVERVIEW_SAMPLE_RATE:
# - wave_minmax_np: the min and max sample of WAVE_MINMAX_COLUMNS columns over the track (int8, signed square
#   root of the sample over wave_minmax_ref, the track's peak |sample|);
# - wave_bands_np: the mean-square power of WAVE_BAND_COUNT log-spaced bands (WAVE_BAND_EDGES, Hz) in
#   WAVE_BAND_COLUMNS columns (uint8, the 4th root of the power over wave_bands_ref, each band's track maximum).
# The low / mid / high levels are summed from the bands when drawn (the configured ranges, each band weighted by
# its log-frequency overlap with a range), so a change of the ranges redraws at once, without re-analysis.
WAVE_MINMAX_COLUMNS = 32768
# the version of the stored bands (wave_bands_np / _ref): raised whenever their making changes (what is made from
# them carries it: the Variants' fingerprint, core.variants.fingerprint.METHOD)
WAVE_BANDS_VERSION = 1
WAVE_BAND_COLUMNS = 8192
WAVE_BAND_COUNT = 12
WAVE_BAND_EDGES = np.geomspace(20.0, 20000.0, WAVE_BAND_COUNT + 1)
# The overview is measured from its own decode at this rate, whatever the analysis rate: bands reach 20 kHz.
WAVE_OVERVIEW_SAMPLE_RATE = 48000


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
