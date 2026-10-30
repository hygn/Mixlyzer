"""Tie-ups: the works a track is used in (an anime's opening, a game's title screen, a character song). The track owns
its tie-ups, as it owns everything else: what the library shows is its rows. This module is what they are: a work and
its names, a tie-up and its role, how they are shown and found. Where they are kept (library.db's tie-up tables) is
the library's: core.library.tieups, used through the library's owner (core.library.jobs).

Tagged automatically only where the user acts: a new track's first analysis takes the tie-ups of its versions in
the library (its Variants: the owner storing the analysis), and a tie-up added in the Tie-ups tab goes to the
track's versions too when Settings > Library says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Optional
import unicodedata

from core.search import reading
from system import i18n
from system.i18n import tr, tr_mark

# the languages a work's title can be shown in (Settings > Library: libconfig.tieup_language; "": the UI's), each by
# its name in itself
TITLE_LANGUAGES = (("ja-JP", "日本語"), ("ko-KR", "한국어"), ("en-US", "English"), ("zh-CN", "中文（简体）"),
                   ("zh-TW", "中文（繁體）"))
_display_language = ""


def set_display_language(tag: str) -> None:
    """The language works' titles are shown in (a TITLE_LANGUAGES tag; "": the UI's). Set at the start and when the
    setting changes; the library's tie-up texts are read again after (they are made in it)."""
    global _display_language
    _display_language = tag if tag in dict(TITLE_LANGUAGES) else ""


def display_language() -> str:
    return _display_language or i18n.language()


# (code, name): stored as the code, shown translated
CATEGORIES = (
    ("anime", tr_mark("Anime")),
    ("game", tr_mark("Game")),
    ("drama", tr_mark("Drama")),
    ("movie", tr_mark("Movie")),
    ("media_mix", tr_mark("Media mix")),
    ("idol_2d", tr_mark("2D idol")),
    ("commercial", tr_mark("Commercial")),
    ("other", tr_mark("Other")),
)
# (code, name, short: the library's column and the comment; none: the work alone, "Title")
UNSPECIFIED = "unspecified"
ROLES = (
    (UNSPECIFIED, tr_mark("Not specified"), ""),
    ("opening", tr_mark("Opening"), tr_mark("OP")),
    ("ending", tr_mark("Ending"), tr_mark("ED")),
    ("insert", tr_mark("Insert song"), tr_mark("Insert")),
    ("theme", tr_mark("Theme song"), tr_mark("Theme")),
    ("character_song", tr_mark("Character song"), tr_mark("Chara")),
    ("unit_song", tr_mark("Unit song"), tr_mark("Unit")),
    ("image_song", tr_mark("Image song"), tr_mark("Image")),
    ("title_screen", tr_mark("Title / login screen"), tr_mark("Title screen")),
    ("main_menu", tr_mark("Main screen"), tr_mark("Menu")),
    ("event", tr_mark("Event song"), tr_mark("Event")),
    ("bgm", tr_mark("BGM"), tr_mark("BGM")),
    ("other", tr_mark("Other"), tr_mark("Other")),
)
# a work's names: its titles, and the other names it is found by (never shown)
OFFICIAL = "official"
ALIAS_TYPES = (
    ("synonym", tr_mark("Synonym")),            # a name it is commonly called by; a romanization of its title
    ("short", tr_mark("Short")),                # an abbreviation
    ("search_hint", tr_mark("Search hint")),    # a misspelling, a typing slip: only found by
)
# the languages the editor offers (any other ll-RR tag may be typed)
LANGUAGES = ("ja-JP", "ko-KR", "en-US", "zh-CN", "zh-TW", "zh-HK")
# the references that name a work itself (finding a work again): its Wikidata item first, its Wikipedia page
IDENTITY_URLS = ("https://www.wikidata.org/wiki/", "https://ja.wikipedia.org/wiki/")
_LANG = re.compile(r"[a-z]{2}-[A-Z]{2}\Z")
_ROLE_SHORT = {code: short for code, _name, short in ROLES}
_ROLE_NAME = {code: name for code, name, _short in ROLES}
_CATEGORY_NAME = dict(CATEGORIES)


def is_lang(code: str) -> bool:
    """Whether ``code`` is an ll-RR language tag (ko-KR)."""
    return bool(_LANG.match(str(code or "")))


def guess_lang(text: str) -> str:
    """The language a name is likely in, from its script (the user corrects it)."""
    if any("぀" <= c <= "ヿ" for c in text):
        return "ja-JP"
    if any("가" <= c <= "힣" for c in text):
        return "ko-KR"
    if any("一" <= c <= "鿿" for c in text):
        return "ja-JP"
    return "en-US"


@dataclass
class Work:
    id: str = ""
    category: str = "other"
    titles: list[tuple[str, str]] = field(default_factory=list)          # (lang, title): a language's first its primary
    aliases: list[tuple[str, str, str]] = field(default_factory=list)    # (type, lang, name): searched, never shown
    urls: list[tuple[str, str]] = field(default_factory=list)            # (label, url)

    def title(self, lang: Optional[str] = None) -> str:
        """The title shown in ``lang`` (default: display_language()): that language and region's first title, else
        the language's in any region, else the first title."""
        want = display_language() if lang is None else lang
        for match in (lambda code: code == want, lambda code: code.split("-")[0] == want.split("-")[0]):
            for code, text in self.titles:
                if match(code):
                    return text
        return self.titles[0][1] if self.titles else ""

    def names(self) -> list[str]:
        """Every title, then every other name: what the work is found by."""
        return [t for _lang, t in self.titles] + [a for _type, _lang, a in self.aliases]


@dataclass
class TieUp:
    work: Work = field(default_factory=Work)
    role: str = UNSPECIFIED     # the work is what matters; the role when known
    seq: int = 0                # OP2: 2 (0: none)
    detail: str = ""            # "season 2, episodes 1-12", "TV size"

    def key(self) -> tuple[str, str, int]:
        """What makes it one tie-up of a track: its work, role and number."""
        return self.work.id, self.role, int(self.seq)

    def short(self, lang: Optional[str] = None) -> str:
        """The work's title (in ``lang``; default: display_language()) and the role, as the library shows it:
        "Title OP2"."""
        return f"{self.work.title(lang)} {_short(self.role, self.seq)}".strip()


def _short(role: str, seq: int) -> str:
    short = _ROLE_SHORT.get(role, "Other")
    return tr(short) + (str(seq) if seq else "") if short else ""


@dataclass
class Catalog:
    """Every work as the editors show and find them, read at once (by the library's owner): each with its names and
    references, the search's keys, how many tracks use each."""
    works: dict[str, Work]
    keys: list[tuple[str, tuple[str, str]]]
    uses: dict[str, int]


@dataclass(frozen=True)
class TrackTieups:
    shown: str                   # the library's column: "Title OP; Other ED"
    text: str                    # every name and role, for the plain text search
    readings: tuple[str, ...]    # the works' readings, for the search by sound
    works: tuple[str, ...] = ()  # the works' ids, for the tracks of the same work
    # the works' names' 초성 keys, and the sort keys of the title shown first by order (reading.name_keys: stored with
    # the names, core.library.tieups.track_texts)
    initials: tuple[str, ...] = ()
    sort_keys: dict = field(default_factory=dict, compare=False)


def describe(entries: Iterable[tuple]) -> TrackTieups:
    """A track's tie-ups as the library shows and searches them; ``entries``: (work, role, seq, detail, the work's
    readings) in the track's order."""
    shown, text, keys, works = [], [], [], []
    for w, role, seq, detail, readings in entries:
        if w.id and w.id not in works:
            works.append(w.id)
        short = _short(role, seq)
        shown.append(f"{w.title()} {short}".strip())
        # the text search finds the role and the category by their names too, translated and in English
        role_name, category = _ROLE_NAME.get(role, "Other"), _CATEGORY_NAME.get(w.category, "Other")
        text += w.names() + [short, tr(role_name), role_name, tr(category), category, detail]
        keys += [r for r in readings if r not in keys]
    return TrackTieups("; ".join(shown), " ".join(text), tuple(keys), tuple(works))


def name_key(name: str) -> str:
    """A name as two works' names are compared: NFKC, variant characters as their standard ones (髙 / 高:
    reading.fold_variants), case folded, its letters and digits in their own script (さくひん・ざ・わーく！ and
    さくひん・ざ・わーく! alike; two Han spellings read alike never: no reading involved)."""
    text = reading.fold_variants(unicodedata.normalize("NFKC", str(name or "")))
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def is_identity_url(url: str) -> bool:
    """A reference naming the work itself: its Wikidata item, its Wikipedia page."""
    return url.startswith(IDENTITY_URLS)


def name_readings(text: str, languages: Optional[list[str]], han_run: int) -> str:
    """A name's readings as kept with it (core.search.reading, one a line)."""
    return "\n".join(r for r, _source in reading.readings(text, languages, han_run=han_run))


def comment_suffix(tieups: Iterable[TieUp], lang: str = "") -> str:
    """"[Title OP; Other ED]": the works' titles in ``lang`` (a TITLE_LANGUAGES tag; "": display_language(), the
    Tie-up Titles setting's)."""
    shown = [t.short(lang if lang in dict(TITLE_LANGUAGES) else None) for t in tieups]
    return f"[{'; '.join(shown)}]" if shown else ""


def with_suffix(comment: str, suffix: str) -> str:
    """The comment with the tie-ups' suffix after it (once: not again when it already ends with it)."""
    comment = str(comment or "").rstrip()
    if not suffix or comment.endswith(suffix):
        return comment
    return f"{comment} {suffix}" if comment else suffix


def versions_of(uid: str, pairs) -> set[str]:
    """The track's versions: the tracks its Variants relations (core.search.variants.Pair) join it to, directly or
    through others."""
    neighbours: dict[str, set[str]] = {}
    for p in pairs:
        neighbours.setdefault(p.a, set()).add(p.b)
        neighbours.setdefault(p.b, set()).add(p.a)
    group, todo = {uid}, [uid]
    while todo:
        for v in neighbours.get(todo.pop(), ()):
            if v not in group:
                group.add(v)
                todo.append(v)
    return group - {uid}
