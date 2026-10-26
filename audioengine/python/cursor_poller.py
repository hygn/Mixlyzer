"""Scratch input from the mouse cursor, polled at ~1 kHz off the GUI thread.

The GUI thread's mouse events come at its own pace (~60 Hz while it repaints, stamped
when handled) and in whole pixels: a slow drag reaches the platter as a stair of
speed steps (pitch warble). The poller reads the cursor itself, every ~1 ms, in a
numba loop that never holds the GIL (audioengine/python/_cursor_numba.py, its own thread at
high priority), and keeps what is exact about a whole-pixel cursor: the moment it
crosses a pixel boundary, and the pixel it stops on. The events go to a ring buffer
that the audio thread drains (no Qt events on the way).

Windows only (Win32 cursor and timer); elsewhere, or if the loop could not be loaded,
available() is False and scratching follows the GUI's mouse events.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from system import os_support

_kernels = None    # audioengine.python._cursor_numba once loaded, False if that failed
_timer_api = None  # (CreateWaitableTimerExW, CloseHandle, QPC frequency)

STOP_MIN_SEC = 0.015
STOP_MAX_SEC = 0.05
RING = 4096
_HIGH_RESOLUTION = 0x2       # CREATE_WAITABLE_TIMER_HIGH_RESOLUTION (Windows 10 1803+)
_TIMER_ALL_ACCESS = 0x1F0003
_THREAD_PRIORITY_HIGHEST = 2


def _create_timer() -> int:
    create, _close, _freq = _timer_api
    return int(create(None, None, _HIGH_RESOLUTION, _TIMER_ALL_ACCESS) or 0)


def warm_cursor_poller() -> bool:
    """Load (disk cache) or compile the poll loop. Call it outside the audio thread.
    False (and available() False) off Windows, without a high-resolution timer, or
    if the loop could not be loaded."""
    global _kernels, _timer_api
    if _kernels is None:
        _kernels = False
        if os_support.IS_WINDOWS:
            try:
                import ctypes
                k32 = ctypes.WinDLL("kernel32")
                create = k32.CreateWaitableTimerExW
                create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
                create.restype = ctypes.c_void_p
                close = k32.CloseHandle
                close.argtypes = [ctypes.c_void_p]
                freq = ctypes.c_int64()
                k32.QueryPerformanceFrequency(ctypes.byref(freq))
                _timer_api = (create, close, float(freq.value))
                timer = _create_timer()
                if not timer:
                    raise OSError("no high-resolution waitable timer")
                try:
                    from audioengine.python import _cursor_numba as kernels
                    kernels.warm(timer, _timer_api[2])
                finally:
                    close(timer)
                _kernels = kernels
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[Scratch] cursor poller unavailable, using mouse events: {exc}")
    return bool(_kernels)


def available() -> bool:
    return bool(_kernels)


class CursorPoller:
    """One poll thread per start() .. stop(). drain() returns the new events as
    [k, 3] (time.perf_counter() s, cursor x in physical px, rest 1/0); see
    _cursor_numba.poll for what they are."""

    def __init__(self) -> None:
        self._ctrl = np.zeros(2, np.int64)   # run flag, events written
        self._ring = np.zeros((RING, 3))
        self._read = 0
        self._thread: Optional[threading.Thread] = None
        self._timer = 0

    def start(self) -> bool:
        if not _kernels:
            return False
        self.stop()
        self._timer = _create_timer()
        if not self._timer:
            return False
        self._ctrl[:] = (1, 0)
        self._read = 0
        self._thread = threading.Thread(target=self._run, name="ScratchCursorPoller", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        import ctypes
        k32 = ctypes.WinDLL("kernel32")
        k32.SetThreadPriority(k32.GetCurrentThread(), _THREAD_PRIORITY_HIGHEST)
        _kernels.poll(self._timer, _timer_api[2], self._ctrl, self._ring, STOP_MIN_SEC, STOP_MAX_SEC)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._ctrl[0] = 0
        self._thread.join()
        self._thread = None
        _timer_api[1](self._timer)
        self._timer = 0

    def drain(self) -> np.ndarray:
        count = int(self._ctrl[1])
        if count - self._read > RING:  # overrun: only the newest RING are still there
            self._read = count - RING
        if count == self._read:
            return np.zeros((0, 3))
        rows = np.arange(self._read, count) % RING
        self._read = count
        return self._ring[rows]
