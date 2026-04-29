from __future__ import annotations

JUMPCUE_PAIR_COLORS: list[tuple[int, int, int]] = [
    (34, 139, 34),
    (0, 153, 204),
    (220, 120, 20),
    (148, 0, 211),
    (210, 105, 30),
    (0, 128, 128),
]


def get_jumpcue_pair_color(index: int) -> tuple[int, int, int]:
    if not JUMPCUE_PAIR_COLORS:
        return (34, 139, 34)
    return JUMPCUE_PAIR_COLORS[int(index) % len(JUMPCUE_PAIR_COLORS)]
