import numpy as np

def gen_key_mask(major=True, minor=True, min_offset=0.5):
    # base templates
    maj_pattern = np.array([1,0,1,0,1,1,0,1,0,1,0,1])   # Ionian
    min_pattern = np.array([1,0,1,1,0,1,0,1,1,0,1,0])   # Aeolian
    keyprob = []
    # 12 major (Camelot B ring)
    if major:
        for root in range(12):
            keyprob.append(np.roll(maj_pattern, root).tolist())
    # minor
    if minor:
        for root in range(12):
            keyprob.append(np.roll(min_pattern, root).tolist())
    
    keyprob = np.array(keyprob)
    keyprob = np.add(keyprob, min_offset)
    keyprob = np.clip(keyprob, 0, 1)
    return keyprob

def fuse_key_mode_to_24_path(key_path12, mode_path01):
    k = np.asarray(key_path12, dtype=int).ravel()
    m = np.asarray(mode_path01, dtype=int).ravel()
    rel_minor_pc = (k + 9) % 12
    out = np.where(m == 0, k, 12 + rel_minor_pc)
    return out


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
    import sys

    if sys.platform != "win32":
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
