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


def offset_beats_and_segments(
    beats_time,
    tempo_segments,
    offset_sec: float,
    track_duration: float,
):
    offset_sec = float(offset_sec or 0.0)
    beats = np.asarray(beats_time, dtype=float)
    segments = np.asarray(tempo_segments, dtype=float)
    if abs(offset_sec) < 1e-12:
        return beats, segments

    if beats.size:
        beats = beats + offset_sec
        beats = beats[np.isfinite(beats)]
        beats = beats[(beats >= 0.0) & (beats <= track_duration)]
        beats = beats.astype(np.float32, copy=False)

    if segments.ndim == 2 and segments.shape[1] >= 4 and segments.size:
        shifted = segments.astype(float, copy=True)
        shifted[:, 0] = np.clip(shifted[:, 0] + offset_sec, 0.0, track_duration)
        shifted[:, 1] = np.clip(shifted[:, 1] + offset_sec, 0.0, track_duration)
        shifted[:, 3] = np.clip(shifted[:, 3] + offset_sec, 0.0, track_duration)
        segments = shifted

    return beats, segments


def prime_physical_core_count() -> None:
    """Pre-fill joblib/loky's physical-core cache on Windows.

    sklearn (KMeans) sizes its OpenMP pool from ``loky.cpu_count(only_physical_cores=True)``,
    which on Windows launches ``powershell.exe`` once per process (~1.5-2 s, paid by
    every analysis process). The same number is read here from
    GetLogicalProcessorInformationEx; on any failure loky keeps its own lookup.
    """
    from foundation import os_support

    if not os_support.IS_WINDOWS:
        return
    try:
        import ctypes
        from ctypes import wintypes

        from joblib.externals.loky.backend import context

        if context.physical_cores_cache is not None:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        relation_processor_core = 0
        size = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not kernel32.GetLogicalProcessorInformationEx(relation_processor_core, buffer, ctypes.byref(size)):
            return
        # Variable-size records, each starting with (Relationship, Size).
        cores, offset = 0, 0
        while offset < size.value:
            relationship, record_size = ctypes.cast(
                ctypes.byref(buffer, offset), ctypes.POINTER(ctypes.c_uint32 * 2)
            ).contents
            if record_size == 0:
                return
            cores += relationship == relation_processor_core
            offset += record_size
        if cores >= 1:
            context.physical_cores_cache = cores
    except Exception:
        pass
