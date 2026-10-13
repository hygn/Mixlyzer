"""Design tokens: the one place app colours and shape constants are defined.

The palette (ui.theme.palette) and the style sheet (ui.theme.qss) are both built
from a Theme, so a new theme (light, high contrast) is a new Theme instance.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool

    # surfaces, back to front
    window: str          # dialogs, panels
    base: str            # inputs, lists, tables
    alt_base: str        # alternating table rows
    raised: str          # buttons, combo boxes
    raised_hover: str
    raised_pressed: str
    popup: str           # menus, tooltips, combo popups
    selection: str       # selected rows / menu items (calmer than the accent)

    # lines
    border: str
    border_strong: str   # hover / focus-less emphasis
    separator: str

    # text
    text: str
    text_dim: str
    text_disabled: str

    # accent and status
    accent: str
    accent_hover: str
    accent_pressed: str
    on_accent: str
    warning: str
    danger: str
    success: str

    # shape
    radius: int = 4
    radius_small: int = 3
    scrollbar: int = 10


# After rekordbox's performance view: near-black surfaces, flat square controls
# with thin grey outlines, solid blue for "on", steel blue for selected rows.
DARK = Theme(
    name="dark",
    dark=True,
    window="#111111",
    base="#080808",
    alt_base="#101010",
    raised="#1b1b1b",
    raised_hover="#262626",
    raised_pressed="#0c0c0c",
    popup="#1a1a1a",
    selection="#234a72",
    border="#3a3a3a",
    border_strong="#626262",
    separator="#232323",
    text="#d8d8d8",
    text_dim="#8a8a8a",
    text_disabled="#4c4c4c",
    accent="#1f7fe5",
    accent_hover="#3b92ef",
    accent_pressed="#1667c2",
    on_accent="#ffffff",
    warning="#e6c229",
    danger="#e5484d",
    success="#3fbf5f",
    radius=2,
    radius_small=2,
    scrollbar=8,
)
