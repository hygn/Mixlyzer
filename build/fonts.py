"""The interface's fonts (assets/fonts, ui/fonts.py): static instances of Google Fonts' variable Noto Sans, Noto Sans
JP, KR, SC and TC (OFL-1.1) at the weights the interface draws, 400 Regular and 700 Bold. Static, not the variable
files: Qt's font database on Windows (DirectWrite) registers each named instance of a variable font as a copy of the
whole font, nine for each of these, about 530 MB of memory for the five; and it keeps two copies of each file it
registers, so every weight shipped costs twice its files (measured: the two weights about 115 MB). Run with the folder
holding the variable fonts (fonts.google.com's downloads, or github.com/google/fonts' ofl/notosans*; any file names: a
font is known by its family):

    uv run python build/fonts.py FOLDER

which writes assets/fonts/<NotoSansJP>-<Regular|Bold>.ttf (each CJK font takes a minute or so)."""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "fonts"
FAMILIES = ("Noto Sans", "Noto Sans JP", "Noto Sans KR", "Noto Sans SC", "Noto Sans TC")
WEIGHTS = {400: "Regular", 700: "Bold"}


def family_of(path: Path) -> str | None:
    """A variable font's family name (its name table's typographic family, else its family), None for another
    file."""
    from fontTools.ttLib import TTFont
    try:
        font = TTFont(path, lazy=True)
    except Exception:
        return None
    if "fvar" not in font:
        return None
    return font["name"].getDebugName(16) or font["name"].getDebugName(1)


def instance(source: Path, weight: int, out: Path) -> Path:
    """``source`` at ``weight`` (its other axes at their defaults), its names those of the weight's style."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import OverlapMode, instantiateVariableFont
    font = TTFont(source, recalcTimestamp=False)    # the same file from the same source
    axes = {axis.axisTag: axis.defaultValue for axis in font["fvar"].axes}
    axes["wght"] = weight
    instantiateVariableFont(font, axes, inplace=True, updateFontNames=True,
                            overlap=OverlapMode.KEEP_AND_SET_FLAGS)
    font.save(out)
    return out


def main(folder: Path) -> None:
    sources = {}
    for path in sorted(folder.iterdir()):
        family = family_of(path) if path.suffix.lower() in (".ttf", ".otf") else None
        if family in FAMILIES:
            sources[family] = path
    missing = [f for f in FAMILIES if f not in sources]
    if missing:
        sys.exit(f"no variable font of {', '.join(missing)} in {folder}")
    jobs = [(sources[family], weight, OUT / f"{family.replace(' ', '')}-{style}.ttf")
            for family in FAMILIES for weight, style in WEIGHTS.items()]
    with ProcessPoolExecutor(max_workers=4) as pool:    # a CJK font's instancing takes a GB or so of memory
        for out in pool.map(instance, *zip(*jobs)):
            print(f"{out.relative_to(ROOT)}: {out.stat().st_size / 2**20:.1f} MB", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_dir():
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
