"""Draws the analysis pipeline map of Settings > Analysis > Global (ui/config/pipeline_map.py):
assets/images/analysis_pipeline.svg and analysis_pipeline.json (its size, each analysis' rectangle, and the words
drawn over the picture in the UI's language: the SVG holds no words). In the look of the audio overview's diagram
(ui/model_canvas.py): the dark theme's base with a dotted grid, rounded blocks with a thin border of their colour and
a faint glow, port marks, each analysis' icon and its number (the order it runs in), lines in the colour of what they
carry, routed at right angles with branch dots. Run after the pipeline changes:

    uv run python build/pipeline_map_svg.py

The words are pipeline_map.WORDS (the catalogs read them there)."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "assets" / "images" / "analysis_pipeline.svg"
JSON = ROOT / "assets" / "images" / "analysis_pipeline.json"

W, H = 780.0, 310.0
BASE, FACE, BORDER, TEXT, DIM = "#080808", "#1b1b1b", "#3a3a3a", "#d8d8d8", "#8a8a8a"   # ui/theme/tokens.py DARK
BLUE, AMBER, PINK, TEAL, GREEN = "#5aa9ff", "#f0a35a", "#e879b9", "#4fd1c5", "#7bd88f"   # model_canvas.DOMAINS
BW, BH = 136.0, 50.0
# name -> (x, y, colour, icon, number, inputs (fractions of the height), output)
BLOCKS = {
    "Audio": (58.0, 135.0, BLUE, "audio", 1, (0.5,), True),
    "Waveform": (58.0, 35.0, BLUE, "waveform", 2, (0.5,), False),
    "Beat grid": (236.0, 135.0, AMBER, "beat", 3, (0.5,), True),
    "Phrase": (446.0, 135.0, PINK, "phrase", 4, (0.5,), True),
    "CUE points": (632.0, 135.0, PINK, "cue", 5, (0.5,), False),
    "JumpCUE": (632.0, 35.0, TEAL, "jump", 6, (1 / 3, 2 / 3), False),
    "Key": (632.0, 235.0, GREEN, "key", 7, (1 / 3, 2 / 3), False),
}
SUBTITLES = {"Audio": "decoded, HPSS", "Waveform": "overview levels", "Beat grid": "beats, tempo, bars",
             "Phrase": "sections + labels", "CUE points": "by phrase label", "JumpCUE": "jump points",
             "Key": "key sections"}

out: list[str] = []


def el(text: str) -> None:
    out.append(text)


def f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def line(points, color: str, arrow: bool = True) -> None:
    d = "M" + " L".join(f"{f(x)},{f(y)}" for x, y in points)
    el(f'<path d="{d}" fill="none" stroke="{color}" stroke-opacity="0.9" stroke-width="1.15" '
       f'stroke-linejoin="round" stroke-linecap="round"/>')
    if arrow:
        (ax, ay), (bx, by) = points[-2], points[-1]
        length = math.hypot(bx - ax, by - ay) or 1.0
        ux, uy = (bx - ax) / length, (by - ay) / length
        pts = [(bx, by), (bx - 5.2 * ux + 2.6 * uy, by - 5.2 * uy - 2.6 * ux),
               (bx - 5.2 * ux - 2.6 * uy, by - 5.2 * uy + 2.6 * ux)]
        el(f'<polygon points="{" ".join(f"{f(x)},{f(y)}" for x, y in pts)}" fill="{color}"/>')


def dot(x: float, y: float, color: str) -> None:
    el(f'<circle cx="{f(x)}" cy="{f(y)}" r="2.1" fill="{color}"/>')


def chevron(x: float, y: float, color: str, point_right: bool = True) -> None:
    dx = 3.0 if point_right else -3.0
    el(f'<polyline points="{f(x)},{f(y - 2.7)} {f(x + dx)},{f(y)} {f(x)},{f(y + 2.7)}" fill="none" '
       f'stroke="{color}" stroke-opacity="0.75" stroke-width="0.9" stroke-linecap="round" stroke-linejoin="round"/>')


def icon(kind: str, x: float, y: float, c: str) -> None:
    """The analysis' icon in a 24 x 24 box at (x, y)."""
    stroke = f'fill="none" stroke="{c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"'
    if kind == "audio":   # a speaker
        el(f'<polygon points="{f(x + 2)},{f(y + 8.5)} {f(x + 6.5)},{f(y + 8.5)} {f(x + 12)},{f(y + 3)} '
           f'{f(x + 12)},{f(y + 21)} {f(x + 6.5)},{f(y + 15.5)} {f(x + 2)},{f(y + 15.5)}" fill="{c}"/>')
        for r in (4.5, 8.0):
            a0, a1 = math.radians(-50), math.radians(50)
            x0, y0 = x + 12 + r * math.cos(a0), y + 12 + r * math.sin(a0)
            x1, y1 = x + 12 + r * math.cos(a1), y + 12 + r * math.sin(a1)
            el(f'<path d="M{f(x0)},{f(y0)} A{f(r)},{f(r)} 0 0 1 {f(x1)},{f(y1)}" {stroke}/>')
    elif kind == "waveform":   # a waveform's bars
        for k, h in enumerate((5, 11, 17, 22, 13, 19, 8, 15, 4)):
            xx = x + 1.5 + k * 2.6
            el(f'<line x1="{f(xx)}" y1="{f(y + 12 - h / 2)}" x2="{f(xx)}" y2="{f(y + 12 + h / 2)}" {stroke}/>')
    elif kind == "beat":   # the grid's lines, the downbeats taller
        for k, h in enumerate((22, 12, 12, 12, 22, 12)):
            xx = x + 2 + k * 4.0
            el(f'<line x1="{f(xx)}" y1="{f(y + 12 - h / 2)}" x2="{f(xx)}" y2="{f(y + 12 + h / 2)}" {stroke}/>')
    elif kind == "phrase":   # sections
        for xx, w, a in ((x, 6.5, 1.0), (x + 7.5, 9.0, 0.5), (x + 17.5, 6.5, 1.0)):
            el(f'<rect x="{f(xx)}" y="{f(y + 7)}" width="{f(w)}" height="10" fill="{c}" fill-opacity="{a}"/>')
    elif kind == "cue":   # a flag
        el(f'<line x1="{f(x + 5)}" y1="{f(y + 2)}" x2="{f(x + 5)}" y2="{f(y + 22)}" {stroke}/>')
        el(f'<polygon points="{f(x + 5)},{f(y + 2)} {f(x + 21)},{f(y + 7)} {f(x + 5)},{f(y + 12)}" fill="{c}"/>')
    elif kind == "jump":   # a jump from one point to another
        el(f'<path d="M{f(x + 5)},{f(y + 13)} A8,7 0 0 1 {f(x + 19)},{f(y + 11)}" {stroke}/>')
        el(f'<polyline points="{f(x + 16.5)},{f(y + 8)} {f(x + 19.5)},{f(y + 11.5)} {f(x + 22)},{f(y + 8)}" {stroke}/>')
        el(f'<rect x="{f(x + 1)}" y="{f(y + 15)}" width="7" height="7" fill="{c}"/>')
        el(f'<rect x="{f(x + 16)}" y="{f(y + 15)}" width="7" height="7" fill="{c}"/>')
    elif kind == "key":   # piano keys
        el(f'<rect x="{f(x + 1)}" y="{f(y + 3)}" width="22" height="18" rx="1.5" {stroke}/>')
        for k in range(1, 4):
            el(f'<line x1="{f(x + 1 + 5.5 * k)}" y1="{f(y + 3)}" x2="{f(x + 1 + 5.5 * k)}" y2="{f(y + 21)}" {stroke}/>')
        for k in range(3):
            el(f'<rect x="{f(x + 4.5 + 5.5 * k)}" y="{f(y + 3)}" width="3.5" height="10" fill="{c}"/>')


def block(name: str) -> None:
    x, y, c, kind, _number, inputs, output = BLOCKS[name]
    el(f'<rect x="{f(x)}" y="{f(y)}" width="{f(BW)}" height="{f(BH)}" rx="5" fill="none" stroke="{c}" '
       f'stroke-opacity="0.12" stroke-width="5"/>')
    el(f'<rect x="{f(x)}" y="{f(y)}" width="{f(BW)}" height="{f(BH)}" rx="5" fill="{FACE}" stroke="{c}" '
       f'stroke-opacity="0.85" stroke-width="1.1"/>')
    for frac in inputs:
        chevron(x + 1.8, y + frac * BH, c)
    if output:
        chevron(x + BW - 4.8, y + BH / 2, c)
    icon(kind, x + 12, y + 13, c)
    el(f'<circle cx="{f(x + 1)}" cy="{f(y + 1)}" r="6.6" fill="{c}"/>')


def main() -> None:
    el(f'<svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="tiny" width="{f(W)}" '
       f'height="{f(H)}" viewBox="0 0 {f(W)} {f(H)}">')
    el(f'<rect x="0.5" y="0.5" width="{f(W - 1)}" height="{f(H - 1)}" rx="8" fill="{BASE}" stroke="{BORDER}" '
       f'stroke-width="0.8"/>')
    for gx in range(15, int(W), 15):
        for gy in range(15, int(H), 15):
            el(f'<circle cx="{gx}" cy="{gy}" r="0.55" fill="{BORDER}" fill-opacity="0.55"/>')
    y = 160.0
    # the file, decoded into the audio; the overview waveform from it
    el(f'<rect x="12" y="150" width="28" height="20" rx="10" fill="{FACE}" stroke="{TEXT}" stroke-opacity="0.6" '
       f'stroke-width="1"/>')
    el(f'<path d="M21,155 L27,155 L30,158 L30,165 L21,165 Z" fill="none" stroke="{TEXT}" stroke-opacity="0.8" '
       f'stroke-width="0.9" stroke-linejoin="round"/>')
    grey = "#9a9a9a"
    line([(40, y), (58, y)], grey)
    line([(47, y), (47, 60), (58, 60)], grey)
    dot(47, y, grey)
    # audio -> beat grid -> phrases -> CUE points; the beat grid and the phrases into JumpCUE and the key
    line([(194, y), (236, y)], BLUE)
    line([(372, y), (446, y)], AMBER)
    bx = 427.0
    line([(bx, y), (bx, 35 + BH / 3), (632, 35 + BH / 3)], AMBER)
    line([(bx, y), (bx, 235 + 2 * BH / 3), (632, 235 + 2 * BH / 3)], AMBER)
    dot(bx, y, AMBER)
    px = 602.0
    line([(582, y), (632, y)], PINK)
    line([(px, y), (px, 35 + 2 * BH / 3), (632, 35 + 2 * BH / 3)], PINK)
    line([(px, y), (px, 235 + BH / 3), (632, 235 + BH / 3)], PINK)
    dot(px, y, PINK)
    for name in BLOCKS:
        block(name)
    el("</svg>")
    SVG.write_text("\n".join(out) + "\n", encoding="utf-8")

    texts = [{"text": "file", "x": 26.0, "y": 181.4, "size": 9.167, "color": DIM, "align": "center", "room": 44}]
    for name, (x, y0, c, _kind, number, _ins, _out) in BLOCKS.items():
        texts.append({"text": name, "x": x + 48, "y": y0 + 20.8, "size": 10.833, "color": TEXT, "align": "left",
                      "room": BW - 54})
        texts.append({"text": SUBTITLES[name], "x": x + 48, "y": y0 + 36.2, "size": 8.333, "color": DIM,
                      "align": "left", "room": BW - 54})
        texts.append({"text": str(number), "x": x + 1, "y": y0 + 4.2, "size": 8.5, "color": "#0b0b0b",
                      "align": "center", "room": 12})
    texts += [{"text": "audio", "x": 215.0, "y": 155.3, "size": 8.0, "color": BLUE, "align": "center", "room": 40},
              {"text": "beat grid", "x": 399.5, "y": 155.3, "size": 8.0, "color": AMBER, "align": "center",
               "room": 70},
              {"text": "beat grid", "x": 515.0, "y": 46.9, "size": 8.0, "color": AMBER, "align": "center",
               "room": 100},
              {"text": "beat grid", "x": 515.0, "y": 278.6, "size": 8.0, "color": AMBER, "align": "center",
               "room": 100},
              {"text": "phrases", "x": 580.6, "y": 116.4, "size": 8.0, "color": PINK, "align": "center",
               "room": 40}]
    JSON.write_text(json.dumps({"size": [W, H], "blocks": {n: [b[0], b[1], BW, BH] for n, b in BLOCKS.items()},
                                "texts": texts}, indent=1) + "\n", encoding="utf-8")
    print(f"{SVG.relative_to(ROOT)}: {len(out)} elements; {JSON.relative_to(ROOT)}: {len(texts)} words")


if __name__ == "__main__":
    main()
