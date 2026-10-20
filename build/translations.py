"""The translation catalogs updated from the source (Babel, a development dependency; run from anywhere):

    uv run python build/translations.py

extracts the texts of every tr / tr_n / tr_ctx / tr_mark call (core.i18n), with the comments above them that
start with ``TRANSLATORS:``, into assets/i18n/mixlyzer.pot and merges it into the catalog of each language of
assets/i18n/languages.json (assets/i18n/languages/<tag>/mixlyzer.po, made when missing; English is the source):
a new text comes in untranslated, a changed one keeps its old translation marked fuzzy (left out by the app until
checked) with the old English beside it, a removed one is dropped."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from babel.messages.frontend import CommandLineInterface

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.i18n import CATALOG_DIR, DOMAIN, LANGUAGES, SOURCE_LANGUAGE  # noqa: E402

SOURCES = ["app", "core", "ui", "views", "utils", "optimizer", "analyzer_core", "third_party"]
KEYWORDS = ["tr", "tr_n:1,2", "tr_ctx:1c,2", "tr_mark"]
TEMPLATE = Path("assets/i18n") / f"{DOMAIN}.pot"


def main() -> None:
    os.chdir(ROOT)   # the catalogs' source locations relative to the repo
    cli = CommandLineInterface()
    keywords = [arg for keyword in KEYWORDS for arg in ("-k", keyword)]
    cli.run(["pybabel", "-q", "extract", "--no-default-keywords", *keywords, "--add-comments=TRANSLATORS:",
             "--sort-by-file", "--project=Mixlyzer", "--copyright-holder=Mixlyzer contributors",
             "-o", str(TEMPLATE), *SOURCES])
    for language in LANGUAGES:
        if language == SOURCE_LANGUAGE:
            continue
        catalog = Path(CATALOG_DIR) / language / f"{DOMAIN}.po"
        catalog.parent.mkdir(parents=True, exist_ok=True)
        command = "update" if catalog.exists() else "init"
        args = ["pybabel", "-q", command, "-D", DOMAIN, "-i", str(TEMPLATE), "-o", str(catalog),
                "-l", language.replace("-", "_")]   # Babel names a locale ko_KR
        if command == "update":
            args += ["--previous", "--ignore-obsolete"]
        cli.run(args)
        print(f"{catalog.as_posix()}: {command}d")


if __name__ == "__main__":
    main()
