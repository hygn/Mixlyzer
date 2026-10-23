from __future__ import annotations

from PySide6 import QtMultimedia

VOLUME_CURVE_GAMMA = 0.4
# the trim (Settings > Audio > Mixer, the player's TRIM knob): the native engine's channel TRIM knob, its bottom .. top
# (the middle 0 dB; src/mixer.rs trim_knob_gain)
TRIM_DB_RANGE = (-24.0, 9.0)


def trim_db_to_knob(db: float) -> float:
    """The TRIM knob's position (0 .. 1) at ``db`` (held to TRIM_DB_RANGE)."""
    lo, hi = TRIM_DB_RANGE
    db = min(hi, max(lo, float(db)))
    return 0.5 * (1.0 - db / lo) if db < 0.0 else 0.5 + 0.5 * db / hi


def trim_knob_to_db(position: float) -> float:
    """The trim (dB) at the TRIM knob's ``position``."""
    lo, hi = TRIM_DB_RANGE
    k = min(1.0, max(0.0, float(position)))
    return lo * (1.0 - 2.0 * k) if k < 0.5 else hi * (2.0 * k - 1.0)


def trim_dbfs_to_linear(trim_dbfs: float) -> float:
    linear = 10.0 ** (float(trim_dbfs) / 20.0)
    return max(0.0, min(1.0, float(linear)))


def slider_percent_to_linear(percent: int | float, trim_dbfs: float) -> float:
    slider_norm = max(0.0, min(1.0, float(percent) / 100.0))
    slider_norm = slider_norm ** VOLUME_CURVE_GAMMA
    base_linear = float(
        QtMultimedia.QAudio.convertVolume(
            slider_norm,
            QtMultimedia.QAudio.LogarithmicVolumeScale,
            QtMultimedia.QAudio.LinearVolumeScale,
        )
    )
    return max(0.0, min(1.0, base_linear * trim_dbfs_to_linear(trim_dbfs)))
