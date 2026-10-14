from __future__ import annotations

# JumpCUE group colour presets. A group stores its preset index (graph_color_idx),
# independent of its group number; groups without one take preset group_number % N.
JUMPCUE_PAIR_COLORS: list[tuple[int, int, int]] = [
    (34, 139, 34),
    (0, 153, 204),
    (220, 120, 20),
    (148, 0, 211),
    (210, 105, 30),
    (0, 128, 128),
]

# CUE point colour presets (cue_color_idx); 0, the red, is the default.
CUE_POINT_COLORS: list[tuple[int, int, int]] = [(255, 40, 40), *JUMPCUE_PAIR_COLORS]


def get_jumpcue_pair_color(index: int) -> tuple[int, int, int]:
    if not JUMPCUE_PAIR_COLORS:
        return (34, 139, 34)
    return JUMPCUE_PAIR_COLORS[int(index) % len(JUMPCUE_PAIR_COLORS)]


def jumpcue_color_index_of(rgb, fallback: int) -> int:
    """Preset index of an RGB colour (older blocks store only RGB); ``fallback`` when
    it is not a preset."""
    try:
        key = tuple(int(c) for c in rgb)
    except (TypeError, ValueError):
        return int(fallback) % len(JUMPCUE_PAIR_COLORS)
    for i, color in enumerate(JUMPCUE_PAIR_COLORS):
        if color == key:
            return i
    return int(fallback) % len(JUMPCUE_PAIR_COLORS)


def get_cue_point_color(index: int) -> tuple[int, int, int]:
    i = int(index)
    return CUE_POINT_COLORS[i] if 0 <= i < len(CUE_POINT_COLORS) else CUE_POINT_COLORS[0]
