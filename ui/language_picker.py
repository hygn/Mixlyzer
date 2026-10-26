"""A language-REGION tag (ll-RR: ko-KR, ja-JP, en-US) picked in a combo box typed into: every locale Qt knows written
so (several hundred), the tie-ups' usual ones (core.tieup.LANGUAGES) first, each shown with its language's name in the
interface's language (the common ones), in itself and in English, and its region's; typing narrows the list to the
ones holding the text anywhere (일본, japan, 日本語, ja-), and a tag typed whole is taken as it is."""
from __future__ import annotations

from functools import lru_cache
import re

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QComboBox, QCompleter

from core import tieup
from core.i18n import tr, tr_mark

# the common languages' names, shown (and found) in the interface's language
NAMES = {
    "ja": tr_mark("Japanese"), "ko": tr_mark("Korean"), "en": tr_mark("English"), "zh": tr_mark("Chinese"),
    "fr": tr_mark("French"), "de": tr_mark("German"), "es": tr_mark("Spanish"), "it": tr_mark("Italian"),
    "pt": tr_mark("Portuguese"), "ru": tr_mark("Russian"), "uk": tr_mark("Ukrainian"), "pl": tr_mark("Polish"),
    "nl": tr_mark("Dutch"), "sv": tr_mark("Swedish"), "nb": tr_mark("Norwegian"), "da": tr_mark("Danish"),
    "fi": tr_mark("Finnish"), "cs": tr_mark("Czech"), "sk": tr_mark("Slovak"), "hu": tr_mark("Hungarian"),
    "ro": tr_mark("Romanian"), "bg": tr_mark("Bulgarian"), "el": tr_mark("Greek"), "tr": tr_mark("Turkish"),
    "ar": tr_mark("Arabic"), "he": tr_mark("Hebrew"), "fa": tr_mark("Persian"), "hi": tr_mark("Hindi"),
    "bn": tr_mark("Bengali"), "ta": tr_mark("Tamil"), "th": tr_mark("Thai"), "vi": tr_mark("Vietnamese"),
    "id": tr_mark("Indonesian"), "ms": tr_mark("Malay"), "km": tr_mark("Khmer"), "lo": tr_mark("Lao"),
    "my": tr_mark("Burmese"), "mn": tr_mark("Mongolian"), "hr": tr_mark("Croatian"), "sr": tr_mark("Serbian"),
    "sl": tr_mark("Slovenian"), "lt": tr_mark("Lithuanian"), "lv": tr_mark("Latvian"), "et": tr_mark("Estonian"),
    "ca": tr_mark("Catalan"), "eu": tr_mark("Basque"), "gl": tr_mark("Galician"), "is": tr_mark("Icelandic"),
    "ga": tr_mark("Irish"), "cy": tr_mark("Welsh"), "la": tr_mark("Latin"),
}
_TAG = re.compile(r"[a-z]{2}-[A-Z]{2}")


@lru_cache(maxsize=1)
def _locales() -> tuple[tuple[str, str], ...]:
    """(tag, its label) of every locale written ll-RR: the usual ones first, the others by tag."""
    found: dict[str, QLocale] = {}
    for loc in QLocale.matchingLocales(QLocale.AnyLanguage, QLocale.AnyScript, QLocale.AnyTerritory):
        name = loc.name()
        if re.fullmatch(r"[a-z]{2}_[A-Z]{2}", name):
            found.setdefault(name.replace("_", "-"), loc)
    # the usual ones first, then by language, each one's main region (Qt's default for it: de-DE, pt-PT) first
    main = {QLocale(loc.language()).name().replace("_", "-") for loc in found.values()}
    order = list(tieup.LANGUAGES) + sorted((t for t in found if t not in tieup.LANGUAGES),
                                           key=lambda t: (t[:2], t not in main, t))
    return tuple((t, _label(t, found.get(t))) for t in order)


def _label(tag: str, loc: QLocale | None) -> str:
    """"ja-JP  일본어 · 日本語 (日本) · Japanese (Japan)": the parts that differ, each once."""
    parts = []
    name = tr(NAMES[tag[:2]]) if tag[:2] in NAMES else ""
    if name and (loc is None or name != QLocale.languageToString(loc.language())):    # an English UI: once
        parts.append(name)
    if loc is not None:
        own = f"{loc.nativeLanguageName()} ({loc.nativeTerritoryName()})"
        english = f"{QLocale.languageToString(loc.language())} ({QLocale.territoryToString(loc.territory())})"
        parts += [own, english]
    return f"{tag}  " + " · ".join(dict.fromkeys(p for p in parts if p.strip(" ()")))


def tag_of(text: str) -> str:
    """The tag a combo's text names: a label's first word, or what was typed."""
    text = str(text or "").strip()
    first = text.split()[0] if text else ""
    return first if _TAG.fullmatch(first) else text


@lru_cache(maxsize=1)
def _model() -> QStandardItemModel:
    """The locales as one list every picker shares (a table holds a picker a row)."""
    model = QStandardItemModel()
    for t, label in _locales():
        item = QStandardItem(label)
        item.setData(t, Qt.UserRole)
        model.appendRow(item)
    return model


_WIDTHS: dict[str, int] = {}     # font -> the width the labels need (measured once: a table makes a picker a row)


def _list_width(font_key: str, metrics) -> int:
    if font_key not in _WIDTHS:
        _WIDTHS[font_key] = max(metrics.horizontalAdvance(label) for _t, label in _locales()) + 40
    return _WIDTHS[font_key]


class LanguageCombo(QComboBox):
    """The language picker (the module). Its typed search (a completer and its popup) is made when it first takes
    the focus: a table makes a picker a row, and most are never typed into."""

    def __init__(self, tag: str = "", parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setMaxVisibleItems(15)
        self.setModel(_model())
        # closed it is as narrow as a tag (the label starts with it); its lists open as wide as the labels
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(7)
        self.setCompleter(None)
        self._searchable = False
        self.setToolTip(tr("Type a language's name in any language, or its tag (language-REGION: ko-KR)."))
        self.set_tag(tag or tieup.LANGUAGES[0])

    def _wide(self) -> int:
        return _list_width(self.font().key(), self.fontMetrics())

    def showPopup(self) -> None:
        self.view().setMinimumWidth(self._wide())
        super().showPopup()

    def focusInEvent(self, event) -> None:
        if not self._searchable:
            self._searchable = True
            completer = QCompleter(_model(), self)
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setCompletionMode(QCompleter.PopupCompletion)
            completer.setMaxVisibleItems(15)
            completer.popup().setMinimumWidth(self._wide())
            self.setCompleter(completer)
        super().focusInEvent(event)

    def set_tag(self, tag: str) -> None:
        i = self.findData(tag)
        if i >= 0:
            self.setCurrentIndex(i)
        else:
            self.setEditText(tag)       # a tag Qt has no locale for, kept as written

    def tag(self) -> str:
        return tag_of(self.currentText())
