import colorsys

_CAMELOT_NUMS = [
    8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1,   # majors: C..B
    5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10,   # minors: Cm..Bm
]


def _cam_hsv(camelot_number: int, *, minor: bool = False) -> tuple[float, float, float]:
    """Map Camelot number (1..12) to HSV; same hue for A/B, dimmer for minors."""
    hue = (camelot_number - 1) / 12.0
    sat = 0.78
    val = 0.95 if not minor else 0.70
    return hue, sat, val


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


_CMAP_BY_IDX = [
    _hsv_to_hex(*_cam_hsv(cam, minor=(i >= 12)))
    for i, cam in enumerate(_CAMELOT_NUMS)
]

def hex_to_rgb(hex_str: str):
    """'#RRGGBB' → (r, g, b) tuple with 0..1 floats."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def key_cmap_color(idx: int):
    """Map key index 0..23 (C..B majors then minors) to a Camelot-aware color."""
    idx = int(idx) % 24
    r, g, b = hex_to_rgb(_CMAP_BY_IDX[idx])
    return int(r * 255), int(g * 255), int(b * 255)


# Label text colour on each key colour (index 0..23, same order as the cmap):
# black or white, whichever has the higher WCAG contrast ratio on it. Computed
# once from _CMAP_BY_IDX; recompute if the key colours change.
_K, _W = (0, 0, 0), (255, 255, 255)
KEY_LABEL_TEXT_COLORS = [
    _K, _K, _W, _K, _K, _K, _K, _W, _K, _K, _K, _K,   # majors: C..B
    _K, _W, _K, _K, _W, _K, _W, _K, _W, _W, _K, _W,   # minors: Cm..Bm
]


def key_label_text_color(idx: int) -> tuple[int, int, int]:
    """Black or white text for a label on key ``idx``'s colour."""
    return KEY_LABEL_TEXT_COLORS[int(idx) % 24]
