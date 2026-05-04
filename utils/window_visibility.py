from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


if os.name == "nt":
    user32 = ctypes.windll.user32
    try:
        dwmapi = ctypes.windll.dwmapi
    except Exception:
        dwmapi = None

    HRESULT = getattr(wintypes, "HRESULT", ctypes.c_long)
    DWMWA_CLOAKED = 14
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    GA_ROOT = 2

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class POINT(ctypes.Structure):
        _fields_ = [
            ("x", wintypes.LONG),
            ("y", wintypes.LONG),
        ]

    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.WindowFromPoint.argtypes = [POINT]
    user32.WindowFromPoint.restype = wintypes.HWND

    if dwmapi is not None:
        dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        dwmapi.DwmGetWindowAttribute.restype = HRESULT


def _safe_hwnd(value) -> int:
    try:
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, (bytes, bytearray)):
            value = int.from_bytes(value, byteorder="little", signed=False)
        return int(value)
    except Exception:
        return 0


def _rect_from_dwm(hwnd: int):
    if os.name != "nt" or dwmapi is None or hwnd == 0:
        return None
    rect = RECT()
    hr = dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if hr != 0:
        return None
    return rect


def _window_rect(hwnd: int):
    if os.name != "nt" or hwnd == 0:
        return None
    rect = _rect_from_dwm(hwnd)
    if rect is not None:
        return rect
    rect = RECT()
    ok = user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    if not ok:
        return None
    return rect


def _is_cloaked(hwnd: int) -> bool:
    if os.name != "nt" or dwmapi is None or hwnd == 0:
        return False
    cloaked = wintypes.DWORD(0)
    hr = dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        wintypes.DWORD(DWMWA_CLOAKED),
        ctypes.byref(cloaked),
        ctypes.sizeof(cloaked),
    )
    return bool(hr == 0 and cloaked.value != 0)


def _sample_points(rect, *, cols: int = 6, rows: int = 6) -> list[POINT]:
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 2 or height <= 2:
        return []
    inset_x = max(2, min(16, int(round(width * 0.06))))
    inset_y = max(2, min(16, int(round(height * 0.06))))
    left = rect.left + inset_x
    right = rect.right - inset_x
    top = rect.top + inset_y
    bottom = rect.bottom - inset_y
    if right - left <= 2 or bottom - top <= 2:
        left = rect.left + 1
        right = rect.right - 1
        top = rect.top + 1
        bottom = rect.bottom - 1

    xs = []
    ys = []
    for ix in range(cols):
        x = left + int(round((ix + 0.5) * max(1, (right - left)) / cols))
        x = max(left, min(right - 1, x))
        xs.append(x)
    for iy in range(rows):
        y = top + int(round((iy + 0.5) * max(1, (bottom - top)) / rows))
        y = max(top, min(bottom - 1, y))
        ys.append(y)
    points = [POINT(x, y) for y in ys for x in xs]
    center_x = max(left, min(right - 1, (left + right) // 2))
    center_y = max(top, min(bottom - 1, (top + bottom) // 2))
    points.extend(
        [
            POINT(center_x, center_y),
            POINT(left, top),
            POINT(right - 1, top),
            POINT(left, bottom - 1),
            POINT(right - 1, bottom - 1),
        ]
    )
    unique = []
    seen = set()
    for pt in points:
        key = (int(pt.x), int(pt.y))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pt)
    return unique


def is_window_fully_hidden(hwnd_or_widget) -> bool:
    """
    Fast Windows-first heuristic for "the whole top-level window is not visible
    to the user at all".

    Returns True when the window is hidden/minimized/cloaked or when sampled
    points across the window rectangle are entirely covered by other windows.
    Returns False on unsupported platforms.
    """
    if os.name != "nt":
        return False

    hwnd = 0
    if hasattr(hwnd_or_widget, "winId"):
        try:
            hwnd = _safe_hwnd(hwnd_or_widget.winId())
        except Exception:
            hwnd = 0
        try:
            if hasattr(hwnd_or_widget, "isVisible") and not bool(hwnd_or_widget.isVisible()):
                return True
            if hasattr(hwnd_or_widget, "isMinimized") and bool(hwnd_or_widget.isMinimized()):
                return True
        except Exception:
            pass
    else:
        hwnd = _safe_hwnd(hwnd_or_widget)

    if hwnd == 0:
        return False
    if not bool(user32.IsWindowVisible(wintypes.HWND(hwnd))):
        return True
    if bool(user32.IsIconic(wintypes.HWND(hwnd))):
        return True
    if _is_cloaked(hwnd):
        return True

    rect = _window_rect(hwnd)
    if rect is None:
        return False
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return True

    sample_points = _sample_points(rect)
    if not sample_points:
        return True

    root_hwnd = _safe_hwnd(hwnd)
    for pt in sample_points:
        hit = user32.WindowFromPoint(pt)
        if not hit:
            continue
        hit_root = user32.GetAncestor(hit, GA_ROOT)
        if _safe_hwnd(hit_root) == root_hwnd:
            return False
    return True
