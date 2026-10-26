"""App-wide style sheet: assets/styles/app.qss with a Theme's tokens filled in.

Rules here are the defaults every widget gets; a widget's own setStyleSheet
still wins where it sets the same property (deck buttons, edit panel states).

Ground rules, so the sheet stays safe to apply to the whole app:
- No fonts: the global font (utils.fonts) and per-widget setFont stay in charge.
- No min/max sizes: many widgets are fixed-size; only paddings that fit them.
- No QWidget / QFrame / QAbstractScrollArea rules: they would reach the
  pyqtgraph views and the custom-painted deck widgets.
- No ::item rules on item views: those make every cell go through the style
  sheet engine and slow the library table; selection colours come from the palette.
- Backgrounds of buttons / combos use palette(button) so widgets that colour
  themselves through their palette (JumpCUE combos) keep doing so.
"""
from __future__ import annotations

from dataclasses import asdict
from string import Template

from foundation.resource_paths import resource_path
from ui.theme.tokens import Theme

QSS_PATH = resource_path("assets/styles/app.qss")


def build_stylesheet(theme: Theme, icons: dict[str, str]) -> str:
    """The sheet with ``$token`` / ``${token}`` replaced by the theme's values and
    the glyph paths; a placeholder with no value raises KeyError."""
    values = {k: (f"{v}" if not isinstance(v, bool) else str(v)) for k, v in asdict(theme).items()}
    values.update(icons)
    return Template(QSS_PATH.read_text(encoding="utf-8")).substitute(values)
