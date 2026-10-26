"""The analyzer's process: the operating system it runs on (detected here: the analyzer imports no app code),
console programs started without a window, the physical core count sklearn sizes its pool by; and the small
signal helpers the beat and JumpCUE code share."""
import subprocess
import sys

import numpy as np

IS_WINDOWS = sys.platform == "win32"


def hidden_subprocess_kwargs() -> dict:
    """Popen keyword arguments that keep a console program's window from opening (Windows); none elsewhere."""
    if not IS_WINDOWS:
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0), "startupinfo": startupinfo}


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    x_pad = np.pad(x, pad, mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")

    return y


def prime_physical_core_count() -> None:
    """Pre-fill joblib/loky's physical-core cache on Windows.

    sklearn (KMeans) sizes its OpenMP pool from ``loky.cpu_count(only_physical_cores=True)``,
    which on Windows launches ``powershell.exe`` once per process (~1.5-2 s, paid by
    every analysis process). The same number is read here from
    GetLogicalProcessorInformationEx; on any failure loky keeps its own lookup.
    """
    if not IS_WINDOWS:
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


def spawn_pool(max_workers: int):
    """A bounded pool of spawned processes for the training's CPU-bound jobs (inside its own, non-daemon process)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=max(1, int(max_workers)), mp_context=multiprocessing.get_context("spawn"))
