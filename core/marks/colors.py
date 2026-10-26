from __future__ import annotations

from analyzer.music.marks import JumpCue, JumpCueGroup, cues_by_point

# JumpCUE group colour presets. A group stores its preset index (graph_color_idx),
# independent of its group number; the analysis gives group n preset n % N. Hues apart
# from each other and from the CUE point red.
JUMPCUE_PAIR_COLORS: list[tuple[int, int, int]] = [
    (34, 139, 34),     # green
    (0, 153, 204),     # blue
    (220, 120, 20),    # orange
    (148, 0, 211),     # purple
    (230, 195, 0),     # yellow
    (0, 128, 128),     # teal
]

# CUE point colour presets (cue_color_idx); 0, the red, is the default.
CUE_POINT_COLORS: list[tuple[int, int, int]] = [(255, 40, 40), *JUMPCUE_PAIR_COLORS]


def get_jumpcue_pair_color(index: int) -> tuple[int, int, int]:
    if not JUMPCUE_PAIR_COLORS:
        return (34, 139, 34)
    return JUMPCUE_PAIR_COLORS[int(index) % len(JUMPCUE_PAIR_COLORS)]


def jump_cue_colors(groups: list[JumpCueGroup]) -> list[tuple[JumpCue, tuple[int, int, int]]]:
    """Every JumpCUE with its group's colour, by point (as the views draw them)."""
    return [(cue, get_jumpcue_pair_color(groups[gi].color_idx)) for cue, gi in cues_by_point(groups)]


def get_cue_point_color(index: int) -> tuple[int, int, int]:
    i = int(index)
    return CUE_POINT_COLORS[i] if 0 <= i < len(CUE_POINT_COLORS) else CUE_POINT_COLORS[0]
