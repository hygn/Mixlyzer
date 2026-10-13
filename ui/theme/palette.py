"""QPalette from a Theme. Widgets that paint themselves (and QSS palette(...)
references) read these roles, so they follow the theme without their own colours."""
from __future__ import annotations

from PySide6 import QtGui

from ui.theme.tokens import Theme


def build_palette(theme: Theme) -> QtGui.QPalette:
    c = QtGui.QColor
    P = QtGui.QPalette
    palette = P()
    for group in (P.Active, P.Inactive, P.Disabled):
        disabled = group == P.Disabled
        fg = c(theme.text_disabled if disabled else theme.text)
        palette.setColor(group, P.Window, c(theme.window))
        palette.setColor(group, P.WindowText, fg)
        palette.setColor(group, P.Base, c(theme.base))
        palette.setColor(group, P.AlternateBase, c(theme.alt_base))
        palette.setColor(group, P.ToolTipBase, c(theme.popup))
        palette.setColor(group, P.ToolTipText, c(theme.text))
        palette.setColor(group, P.PlaceholderText, c(theme.text_disabled if disabled else theme.text_dim))
        palette.setColor(group, P.Text, fg)
        palette.setColor(group, P.Button, c(theme.raised))
        palette.setColor(group, P.ButtonText, fg)
        palette.setColor(group, P.BrightText, c(theme.danger))
        palette.setColor(group, P.Highlight, c(theme.separator if disabled else theme.selection))
        palette.setColor(group, P.HighlightedText, c(theme.on_accent))
        palette.setColor(group, P.Light, c(theme.raised_hover))
        palette.setColor(group, P.Midlight, c(theme.raised))
        palette.setColor(group, P.Mid, c(theme.border))
        palette.setColor(group, P.Dark, c(theme.raised_pressed))
        palette.setColor(group, P.Shadow, c("#000000"))
        palette.setColor(group, P.Link, c(theme.accent_hover))
        palette.setColor(group, P.LinkVisited, c(theme.accent))
    return palette
