"""Numba loop of core.audio.cursor_poller (Windows only).

Runs without the GIL: Win32 is called directly from nopython code, so the loop
keeps its ~1 kHz pace whatever Python threads do. Loaded by
cursor_poller.warm_cursor_poller() on the main thread.
"""
from __future__ import annotations

import ctypes

import numpy as np
from llvmlite import binding as llvm
from numba import njit, types

# Win32 bound by symbol name (cacheable, unlike ctypes pointers baked into the code);
# pointers and handles are passed as integers.
_user32 = ctypes.WinDLL("user32")
_kernel32 = ctypes.WinDLL("kernel32")
for _dll, _name in ((_user32, "GetPhysicalCursorPos"), (_kernel32, "QueryPerformanceCounter"),
                    (_kernel32, "SetWaitableTimer"), (_kernel32, "WaitForSingleObject"),
                    (_kernel32, "RtlMoveMemory")):
    llvm.add_symbol(_name, ctypes.cast(getattr(_dll, _name), ctypes.c_void_p).value)

_get_cursor = types.ExternalFunction("GetPhysicalCursorPos", types.int32(types.intp))
_qpc = types.ExternalFunction("QueryPerformanceCounter", types.int32(types.intp))
_set_timer = types.ExternalFunction(
    "SetWaitableTimer", types.int32(types.intp, types.intp, types.int32, types.intp, types.intp, types.int32))
_wait = types.ExternalFunction("WaitForSingleObject", types.uint32(types.intp, types.uint32))
# Shared memory (ctrl, ring) is only touched through this call: the compiler keeps
# such calls in program order, so a reader that sees the event count sees the events.
_move = types.ExternalFunction("RtlMoveMemory", types.void(types.intp, types.intp, types.intp))

POLL_100NS = 5000     # relative timer due time; with the wake-up overhead ~1 ms per poll
RATE_SEC = 0.012      # the rate is taken over the crossings of the last RATE_SEC
SLOW_RATE = 50.0      # px/s assumed for the first crossing of a move
INFINITE = 0xFFFFFFFF


@njit(cache=True, nogil=True)
def _emit(ring, ctrl, slot, cnt, at, pos, rest):
    count = cnt[0]
    slot[0] = at
    slot[1] = pos
    slot[2] = rest
    _move(ring.ctypes.data + (count % ring.shape[0]) * 24, slot.ctypes.data, 24)
    cnt[0] = count + 1
    _move(ctrl.ctypes.data + 8, cnt.ctypes.data, 8)


@njit(cache=True, nogil=True)
def poll(timer, freq, ctrl, ring, stop_min, stop_max):
    """Poll the cursor's x (physical px) until ctrl[0] is 0; write events
    (time.perf_counter() s, x px, rest) to ring and their count to ctrl[1].

    The first event is where the cursor was (rest). Then at each change of x, the
    pixel boundary it just crossed, at the middle of the poll interval it was crossed
    in (exact positions instead of whole pixels); when x has not changed for longer
    than 1.5 crossings at the rate it moved (stop_min .. stop_max s), the pixel itself
    (rest), timed as if it had decelerated evenly over the half pixel.
    """
    pt = np.zeros(2, np.int32)
    q = np.zeros(1, np.int64)
    due = np.full(1, -POLL_100NS, np.int64)
    flag = np.zeros(1, np.int64)
    cnt = np.zeros(1, np.int64)
    slot = np.zeros(3)
    hist_t = np.zeros(64)
    hist_b = np.zeros(64)
    hn = 0
    first = True
    last = 0
    last_dir = 0
    resting = True
    prev_t = 0.0
    cross_t = 0.0
    rate = SLOW_RATE
    while True:
        _move(flag.ctypes.data, ctrl.ctypes.data, 8)
        if flag[0] == 0:
            return
        _get_cursor(pt.ctypes.data)
        _qpc(q.ctypes.data)
        t = q[0] / freq
        x = int(pt[0])
        if first:
            first = False
            last = x
            _emit(ring, ctrl, slot, cnt, t, float(x), 1.0)
        elif x != last:
            d = 1 if x > last else -1
            b = x - 0.5 * d
            at = 0.5 * (prev_t + t)
            if resting or d != last_dir:
                hn = 0
            if hn == hist_t.shape[0]:
                hist_t[:-1] = hist_t[1:]
                hist_b[:-1] = hist_b[1:]
                hn -= 1
            hist_t[hn] = at
            hist_b[hn] = b
            hn += 1
            k = 0
            while k < hn - 2 and hist_t[k + 1] <= at - RATE_SEC:
                k += 1
            if hn >= 2 and at > hist_t[k]:
                rate = abs(b - hist_b[k]) / (at - hist_t[k])
            else:
                rate = SLOW_RATE
            _emit(ring, ctrl, slot, cnt, at, b, 0.0)
            last = x
            last_dir = d
            resting = False
            cross_t = at
        elif not resting:
            stop = min(max(stop_min, 1.5 / rate), stop_max)
            if t - cross_t > stop:
                _emit(ring, ctrl, slot, cnt, cross_t + min(max(1.0 / rate, 0.002), stop), float(x), 1.0)
                resting = True
        prev_t = t
        _set_timer(timer, due.ctypes.data, 0, 0, 0, 0)
        _wait(timer, INFINITE)


def warm(timer: int, freq: float) -> None:
    """Compile (first run) or load (disk cache) the loop: run it with the flag down."""
    poll(timer, freq, np.zeros(2, np.int64), np.zeros((4, 3)), 0.015, 0.05)
