"""The UI's language: the languages are listed in assets/i18n/languages.json (BCP 47 tag, language-REGION as
en-US / ko-KR / ja-JP -> the language's name in itself; English, the source, needs no catalog), each one's
gettext catalog is assets/i18n/languages/<tag>/mixlyzer.po, read as it is (no compiled .mo: an edited
translation shows at the next start). The language is Settings > View > Language (``viewconfig.language``,
"" = the operating system's), set once at the start before the UI is imported (``app.main``); a change applies
at the next start.

``tr()`` translates a text, ``tr_n()`` one with a count, ``tr_ctx()`` one English word with two senses (the
context tells them apart); ``tr_mark()`` only marks a text for extraction, translated where it is shown (a table
in a module imported before the language is set). A fuzzy entry (its English changed since it was translated)
is left out until checked again, so the English shows. Spawned worker processes take the language from the
environment (``LANGUAGE_ENV``, set by ``install``), so the progress and the errors they send are translated too.

Updating the catalogs after the texts change (Babel, a development dependency): ``uv run python
build/translations.py``."""

from __future__ import annotations

import ast
import gettext
import json
import os
from pathlib import Path

from foundation.resource_paths import resource_path

SOURCE_LANGUAGE = "en-US"
LANGUAGE_LIST = "assets/i18n/languages.json"
CATALOG_DIR = "assets/i18n/languages"
DOMAIN = "mixlyzer"
LANGUAGE_ENV = "MIXLYZER_LANGUAGE"


def _languages() -> dict[str, str]:
    try:
        listed = json.loads(resource_path(LANGUAGE_LIST).read_text(encoding="utf-8"))
        languages = {str(code): str(name) for code, name in listed.items()}
    except (OSError, ValueError, AttributeError):
        languages = {}
    return languages or {SOURCE_LANGUAGE: "English"}


# the languages, each named in itself (Settings > View > Language)
LANGUAGES = _languages()


def catalog_path(language: str) -> Path:
    return resource_path(CATALOG_DIR) / language / f"{DOMAIN}.po"


class _Catalog(gettext.NullTranslations):
    """A .po file's translations: (context, English) -> the text, or its plural forms."""

    def __init__(self, messages: dict, plural) -> None:
        super().__init__()
        self._messages = messages
        self._plural = plural

    def _forms(self, context, singular, plural, n):
        forms = self._messages.get((context, singular))
        if isinstance(forms, list):
            index = self._plural(n)
            if 0 <= index < len(forms) and forms[index]:
                return forms[index]
        return singular if n == 1 else plural

    def gettext(self, message):
        return self._messages.get((None, message)) or message

    def ngettext(self, msgid1, msgid2, n):
        return self._forms(None, msgid1, msgid2, n)

    def pgettext(self, context, message):
        return self._messages.get((context, message)) or message

    def npgettext(self, context, msgid1, msgid2, n):
        return self._forms(context, msgid1, msgid2, n)


def _unquote(text: str) -> str:
    text = text.strip()
    return ast.literal_eval(text) if text.startswith('"') else ""


def read_po(path: Path) -> _Catalog:
    """The translated, not fuzzy entries of a .po file (obsolete ``#~`` entries left out)."""
    messages: dict = {}
    plural = lambda n: int(n != 1)
    entry: dict[str, str] = {}
    fuzzy = False
    field = ""

    def flush() -> None:
        nonlocal entry, fuzzy, plural
        msgid = entry.get("msgid")
        if msgid is not None and not fuzzy:
            context = entry.get("msgctxt")
            if msgid == "" and context is None:
                for line in entry.get("msgstr", "").splitlines():
                    name, _sep, value = line.partition(":")
                    if name.strip().lower() == "plural-forms" and "plural=" in value:
                        expression = value.split("plural=", 1)[1].strip().rstrip(";")
                        plural = gettext.c2py(expression)
            elif "msgid_plural" in entry:
                forms = [entry[key] for key in sorted(k for k in entry if k.startswith("msgstr["))]
                if any(forms):
                    messages[(context, msgid)] = forms
            elif entry.get("msgstr"):
                messages[(context, msgid)] = entry["msgstr"]
        entry, fuzzy = {}, False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if any(key.startswith("msgstr") for key in entry):
                flush()
            if line.startswith("#,") and "fuzzy" in line:
                fuzzy = True
            continue
        if line.startswith('"'):
            if field:
                entry[field] += _unquote(line)
            continue
        keyword, _sep, rest = line.partition(" ")
        if keyword in ("msgctxt", "msgid") and any(key.startswith("msgstr") for key in entry):
            flush()
        field = keyword
        entry[field] = _unquote(rest)
    flush()
    return _Catalog(messages, plural)


_language = SOURCE_LANGUAGE
_catalog: gettext.NullTranslations = gettext.NullTranslations()


def system_language() -> str:
    """The first of the operating system's UI languages there is a catalog for (its region, else the first of
    its language), else English."""
    try:
        from PySide6.QtCore import QLocale
        names = QLocale.system().uiLanguages()
    except Exception:
        return SOURCE_LANGUAGE
    by_language = {}
    for tag in LANGUAGES:
        by_language.setdefault(tag.split("-")[0].lower(), tag)
    for name in names:
        name = name.replace("_", "-")
        exact = next((tag for tag in LANGUAGES if tag.lower() == name.lower()), None)
        if exact or name.split("-")[0].lower() in by_language:
            return exact or by_language[name.split("-")[0].lower()]
    return SOURCE_LANGUAGE


def install(language: str) -> str:
    """Translate into ``language`` ("" or one without a catalog: the system's) from now on, in this process and
    in the worker processes it starts; the language used."""
    global _language, _catalog
    code = language if language in LANGUAGES else system_language()
    catalog: gettext.NullTranslations = gettext.NullTranslations()
    if code != SOURCE_LANGUAGE:
        try:
            catalog = read_po(catalog_path(code))
        except (OSError, ValueError, SyntaxError):
            code = SOURCE_LANGUAGE
    _language, _catalog = code, catalog
    os.environ[LANGUAGE_ENV] = code
    return code


def install_qt(app) -> None:
    """Qt's own texts (the standard buttons, the file dialogs) in the language set, and its number formats."""
    from PySide6 import QtCore
    locale = QtCore.QLocale(_language)
    QtCore.QLocale.setDefault(locale)
    translator = QtCore.QTranslator(app)
    if translator.load(locale, "qtbase", "_", QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.TranslationsPath)):
        app.installTranslator(translator)


def language() -> str:
    return _language


def tr(message: str) -> str:
    return _catalog.gettext(message)


def tr_n(singular: str, plural: str, n: int) -> str:
    return _catalog.ngettext(singular, plural, n)


def tr_ctx(context: str, message: str) -> str:
    return _catalog.pgettext(context, message)


def tr_mark(message: str) -> str:
    return message


if os.environ.get(LANGUAGE_ENV):   # a worker process: the language its app set
    install(os.environ[LANGUAGE_ENV])
