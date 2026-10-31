"""The Cantonese readings' tables (core/search/reading.py): pycantonese's own (the ones its characters_to_jyutping
reads), written out so the app reads them without pycantonese, which builds them at each start by parsing the HKCanCor
corpus (a second and about 180 MB of memory at their peak):

    assets/readings/cantonese_words.tsv   word -> Jyutping (syllables spaced, with tones): HKCanCor's words, each
                                          its most common reading, rime-cantonese's over them
    assets/readings/cantonese_chars.tsv   character -> Jyutping: HKCanCor's characters so, rime-cantonese's over them

and the two data's licences beside them (HKCANCOR-LICENSE.txt, RIME-CANTONESE-LICENSE.txt). Run again after a
pycantonese update (a development dependency):

    uv run python build/cantonese.py"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "readings"
HEADER = ("# {what} from pycantonese {version}'s tables (the ones its characters_to_jyutping reads; build/cantonese.py):\n"
          "# {source}\n"
          "# Hong Kong Cantonese Corpus (HKCanCor), CC BY 4.0: HKCANCOR-LICENSE.txt. Luke, Kang-Kwong and May L.Y. Wong.\n"
          "# 2015. The Hong Kong Cantonese Corpus: Design and Uses. Journal of Chinese Linguistics.\n"
          "# rime-cantonese (github.com/rime/rime-cantonese), CC BY 4.0: RIME-CANTONESE-LICENSE.txt\n"
          "# {columns}\n")


def write(path: Path, table: dict[str, str], header: str) -> None:
    lines = []
    for key, value in sorted(table.items()):
        # read back by lines (str.splitlines: its separators), "#" a comment
        if not key or key.startswith("#") or len((key + "\t" + value).splitlines()) != 1 or value.count("\t"):
            raise ValueError(f"an entry the table cannot hold: {key!r} {value!r}")
        lines.append(f"{key}\t{value}\n")
    path.write_text(header + "".join(lines), encoding="utf-8", newline="\n")
    print(f"{path.relative_to(ROOT)}: {len(lines):,} entries, {path.stat().st_size / 2**20:.1f} MB")


def main() -> None:
    import pycantonese
    from pycantonese.jyutping.characters import _get_words_characters_to_jyutping
    words, chars = _get_words_characters_to_jyutping()
    data = Path(pycantonese.__file__).parent / "data"
    version = pycantonese.__version__
    write(OUT / "cantonese_words.tsv", words, HEADER.format(
        what="Cantonese words' readings", version=version, columns="word\tjyutping",
        source="HKCanCor's words, each its most common reading, then rime-cantonese's words over them."))
    write(OUT / "cantonese_chars.tsv", chars, HEADER.format(
        what="Cantonese characters' readings", version=version, columns="char\tjyutping",
        source="HKCanCor's characters, each its most common reading, then rime-cantonese's characters over them."))
    shutil.copyfile(data / "hkcancor" / "LICENSE.txt", OUT / "HKCANCOR-LICENSE.txt")
    shutil.copyfile(data / "rime_cantonese" / "LICENSE.txt", OUT / "RIME-CANTONESE-LICENSE.txt")


if __name__ == "__main__":
    main()
