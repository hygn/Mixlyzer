from __future__ import annotations

from PySide6 import QtMultimedia

VOLUME_CURVE_GAMMA = 0.4


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
