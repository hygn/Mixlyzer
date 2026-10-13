"""Small SVG glyphs the style sheet needs (combo / spin arrows, check marks).

QSS can only take images from files, so they are written per theme from its
colours into a cache directory and referenced by absolute path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ui.theme.tokens import Theme

_CHEVRON_DOWN = '<path d="M3 5.5 L7 9.5 L11 5.5" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
_CHEVRON_UP = '<path d="M3 8.5 L7 4.5 L11 8.5" fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
_CHECK = '<path d="M3 7.2 L5.8 10 L11 4.2" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
_DASH = '<path d="M3.5 7 L10.5 7" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round"/>'


def _svg(body: str, color: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
            + body.format(c=color) + "</svg>")


def write_icons(theme: Theme, directory: Path | None = None) -> dict[str, str]:
    """Write the theme's glyphs; returns {name: path with forward slashes} for url(...)."""
    out = Path(directory) if directory else Path(tempfile.gettempdir()) / "mixlyzer_theme" / theme.name
    out.mkdir(parents=True, exist_ok=True)
    glyphs = {
        "chevron_down": (_CHEVRON_DOWN, theme.text_dim),
        "chevron_down_hover": (_CHEVRON_DOWN, theme.text),
        "chevron_down_disabled": (_CHEVRON_DOWN, theme.text_disabled),
        "chevron_up": (_CHEVRON_UP, theme.text_dim),
        "chevron_up_hover": (_CHEVRON_UP, theme.text),
        "chevron_up_disabled": (_CHEVRON_UP, theme.text_disabled),
        "check": (_CHECK, theme.on_accent),
        "check_disabled": (_CHECK, theme.text_disabled),
        "partial": (_DASH, theme.on_accent),
    }
    paths = {}
    for name, (body, color) in glyphs.items():
        path = out / f"{name}.svg"
        text = _svg(body, color)
        try:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
        except OSError:
            pass  # a missing glyph only drops the arrow / check image
        paths[name] = path.as_posix()
    return paths
