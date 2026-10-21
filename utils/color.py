import colorsys

import numpy as np

_ALPHANUMERIC_NUMS = [
    8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1,   # majors: C..B
    5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10,   # minors: Cm..Bm
]


def _key_hsv(alphanumeric_number: int, *, minor: bool = False) -> tuple[float, float, float]:
    """Map Alphanumeric number (1..12) to HSV; same hue for A/B, dimmer for minors."""
    hue = (alphanumeric_number - 1) / 12.0
    sat = 0.78
    val = 0.95 if not minor else 0.70
    return hue, sat, val


def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


_CMAP_BY_IDX = [
    _hsv_to_hex(*_key_hsv(number, minor=(i >= 12)))
    for i, number in enumerate(_ALPHANUMERIC_NUMS)
]

def hex_to_rgb(hex_str: str):
    """'#RRGGBB' → (r, g, b) tuple with 0..1 floats."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def key_cmap_color(idx: int):
    """Map key index 0..23 (C..B majors then minors) to a Alphanumeric-aware color."""
    idx = int(idx) % 24
    r, g, b = hex_to_rgb(_CMAP_BY_IDX[idx])
    return int(r * 255), int(g * 255), int(b * 255)


# Label text colour on each key colour (index 0..23, same order as the cmap):
# black or white, whichever has the higher WCAG contrast ratio on it, computed
# once from _CMAP_BY_IDX (recompute if the key colours change); then set by eye
# to white on the saturated mid-tones where white reads better: 8B, 11B, 12B, 1B.
_K, _W = (0, 0, 0), (255, 255, 255)
KEY_LABEL_TEXT_COLORS = [
    _W, _K, _W, _K, _W, _K, _K, _W, _K, _W, _K, _W,   # majors: C..B
    _K, _W, _K, _K, _W, _K, _W, _K, _W, _W, _K, _W,   # minors: Cm..Bm
]


def key_label_text_color(idx: int) -> tuple[int, int, int]:
    """Black or white text for a label on key ``idx``'s colour."""
    return KEY_LABEL_TEXT_COLORS[int(idx) % 24]


# OKLab (Björn Ottosson's perceptual colour space): lightness and chroma set as seen, on sRGB arrays.
_OKLAB_M1 = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_OKLAB_M2 = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
_OKLAB_M2_INV = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_OKLAB_M1_INV = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)


def srgb_to_oklab(rgb):
    """``[..., 3]`` sRGB 0..255 to OKLab (L 0..1, a, b)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(linear @ np.asarray(_OKLAB_M1).T) @ np.asarray(_OKLAB_M2).T


def oklab_to_srgb(lab):
    """``[..., 3]`` OKLab to sRGB 0..255 (uint8; outside the gamut clipped)."""
    linear = np.clip(((np.asarray(lab, dtype=np.float64) @ np.asarray(_OKLAB_M2_INV).T) ** 3)
                     @ np.asarray(_OKLAB_M1_INV).T, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
