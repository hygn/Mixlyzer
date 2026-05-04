from __future__ import annotations

from PySide6 import QtMultimedia


def trim_dbfs_to_linear(trim_dbfs: float) -> float:
    linear = 10.0 ** (float(trim_dbfs) / 20.0)
    return max(0.0, min(1.0, float(linear)))


def slider_percent_to_linear(percent: int | float, trim_dbfs: float) -> float:
    slider_norm = max(0.0, min(1.0, float(percent) / 100.0))
    base_linear = float(
        QtMultimedia.QAudio.convertVolume(
            slider_norm,
            QtMultimedia.QAudio.LogarithmicVolumeScale,
            QtMultimedia.QAudio.LinearVolumeScale,
        )
    )
    return max(0.0, min(1.0, base_linear * trim_dbfs_to_linear(trim_dbfs)))
