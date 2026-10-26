"""Tie-ups found on the web, from Wikimedia: what Japanese Wikipedia says a song is used in, and the work's names in
other languages from Wikidata (core.tieup). Off unless Settings > Library turns it on (it sends a track's title
and artist to Wikipedia and Wikidata), and asked only for a new track left without tie-ups (core.workflows.tieup_web)
or from the Edit tab ("Find on the Web").

What matters is the work (the IP a song comes from); its role is a bonus, "not specified" where the pages do not
agree on one. How a tie-up is read (Wikipedia: CC BY-SA, only facts are kept, the work's page as its reference;
Wikidata: CC0):
  1. The song's own Japanese Wikipedia page, else Japanese Wikipedia searched for the song's names (in quotes), with
     its artists (the voice actors a character credit names too), then its album; the first pages read.
  2. On a page, each mention of the song, with the lines describing it, read for a work and a role, as Japanese
     articles write them: テレビアニメ「[[作品]]」オープニングテーマ, 『作品』第2期エンディング, ゲーム「…」挿入歌 ... (the
     medium before the work gives its category, a number after the role its number, 第1話 - 第12話 its detail; a work
     inside the work, 作中作『…』, is not one). A work's page, or its discography page (作品のディスコグラフィ), listing
     the song is that work, the role the one it is listed under or none. The song's own page is read whole.
  3. The work's page (the link in the mention, else its title) gives its Wikidata item: its labels and aliases in
     Japanese, Korean, English and Chinese become its titles and other names, each language an ll-RR tag. A page is
     a work's only by its Wikipedia categories (page_category: not a singer's, a group's or an album's), and a 2D
     idol's by them too.
  4. The sure candidate (written by itself: core.workflows.tieup_web, and the one the Edit tab's Find on the Web
     checks) is, of the works whose pages are works', the one the song's own page names most, or the one the
     searches' pages name most (the searches go on until two agree): a tag a little wrong now and then is better
     than none. Its role the one named most, else none. The others are suggestions the Edit tab shows.
Requests go one at a time, at most one a second, with the User-Agent Wikimedia asks for and maxlag; answers are kept
for CACHE_DAYS (cache/tieup_web.db), so a song is not asked for twice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from typing import Callable, Optional

from core import reading, tieup

USER_AGENT = "Mixlyzer (https://github.com/hygn/Mixlyzer) tie-up lookup"
WIKIPEDIA = "https://ja.wikipedia.org/w/api.php"
WIKIDATA = "https://www.wikidata.org/w/api.php"
PAGES_READ = 3          # each search's first pages read
PAGES_MAX = 8           # pages read at most for a song
MIN_INTERVAL_SEC = 1.0  # between two requests
TIMEOUT_SEC = 10.0
CACHE_DAYS = 30
CACHE_NAME = os.path.join("cache", "tieup_web.db")
RULES = 3               # how pages are read and a tie-up picked: raise it when they change (the answers kept asked again)
# Wikidata's language codes -> ll-RR tags (others are not taken)
WIKIDATA_LANGS = {"ja": "ja-JP", "ko": "ko-KR", "en": "en-US", "zh-cn": "zh-CN", "zh-hans": "zh-CN", "zh": "zh-CN",
                  "zh-tw": "zh-TW", "zh-hant": "zh-TW", "zh-hk": "zh-HK"}
# the role words of Japanese articles -> (role, its number when the word says it)
ROLE_WORDS = (
    ("オープニングテーマ", "opening"), ("オープニング", "opening"), ("OP", "opening"),
    ("エンディングテーマ", "ending"), ("エンディング", "ending"), ("ED", "ending"),
    ("挿入歌", "insert"), ("劇中歌", "insert"), ("キャラクターソング", "character_song"),
    ("キャラソン", "character_song"), ("ユニットソング", "unit_song"), ("ユニット曲", "unit_song"),
    ("イメージソング", "image_song"), ("イメージテーマ", "image_song"), ("主題歌", "theme"), ("テーマソング", "theme"),
    ("テーマ曲", "theme"), ("CMソング", "theme"),
)
# a work inside the work (作中作『…』, 劇中アニメ「…」): not one of its own
_FICTION = re.compile(r"作中|劇中劇|劇中アニメ|劇中ゲーム|劇中ドラマ")
# a work's page by its Wikipedia categories: the works' indexes (アニメ作品 あ, 漫画作品 ゆ), a year's works (2019年の
# コンピュータゲーム), a company's works (アニプレックスのアニメ作品), a medium's (Android用ゲームソフト); else a franchise's
# own category (アイドルマスター) on a page that is no person's, group's or record's. Only such a work is tagged by
# itself (a singer, an album or an article's plain words never are).
_WORK_CATEGORY = re.compile(r"^(?:アニメ作品|漫画作品) |^\d{4}年の(?:テレビアニメ|アニメ映画|Webアニメ|OVA|漫画|小説|"
                            r"コンピュータゲーム|映画|テレビドラマ|ゲーム)$|ゲームソフト$|オンラインゲーム$|"
                            r"の(?:アニメ|漫画|映画)作品$|^メディアミックス作品$|ライトノベル")
_NOT_WORK_CATEGORY = re.compile(r"存命人物|\d{4}年生$|没年|音楽グループ|アルバム|シングル|アーティスト|ディスコグラフィ|"
                                r"歌手$|楽曲$|ユニット$")
# its categories' words -> the work's category (the first found; a manga's or a novel's songs are its anime's)
CATEGORY_WORDS = (("アニメ", "anime"), ("ゲーム", "game"), ("映画", "movie"), ("ドラマ", "drama"), ("漫画", "anime"),
                  ("小説", "anime"), ("ライトノベル", "anime"), ("メディアミックス", "media_mix"))
# a work's discography page: the work's own page title before it
_LISTING = re.compile(r"(.+?)の(?:ディスコグラフィ|楽曲一覧|楽曲|音楽作品|キャラクターソング(?:一覧)?|主題歌(?:一覧)?)\Z")
# the medium before a work -> its category
MEDIUM_WORDS = (
    ("テレビアニメ", "anime"), ("TVアニメ", "anime"), ("アニメ", "anime"), ("劇場版", "movie"), ("映画", "movie"),
    ("ゲーム", "game"), ("アプリ", "game"), ("ドラマ", "drama"), ("CM", "commercial"),
    ("メディアミックス", "media_mix"), ("プロジェクト", "media_mix"),
)
_ROLE_RE = "|".join(re.escape(w) for w, _r in ROLE_WORDS)
_MEDIUM_RE = "|".join(re.escape(w) for w, _c in MEDIUM_WORDS)
# a tie-up as written: (medium …)「work」 … role(number)
_MENTION = re.compile(rf"(?:(?P<medium>{_MEDIUM_RE})[^「『。\n]{{0,8}})?[「『](?P<work>[^」』\n]{{1,80}})[」』]"
                      rf"[^。\n「『]{{0,12}}?(?P<role>{_ROLE_RE})(?P<seq>[0-9０-９]?)")
# a work named by its medium alone: (medium …)「work」, with no role (○○のメディアミックスプロジェクト『作品』にて誕生):
# read on the song's own page only (elsewhere it is any page's prose)
_WORK_NAMED = re.compile(rf"(?P<medium>{_MEDIUM_RE})[^「『。\n]{{0,8}}[「『](?P<work>[^」』\n]{{1,80}})[」』]")
# a work's Wikidata classes (P31) -> its category, where the article named no medium; a page whose item has none of
# them is not a work (a singer, a group, an album). A manga's or a novel's songs are its anime's.
WIKIDATA_CLASSES = {"Q63952888": "anime", "Q1107": "anime", "Q220898": "anime", "Q20650540": "anime",
                    "Q1047299": "anime", "Q21198342": "anime", "Q8274": "anime", "Q865484": "anime",
                    "Q104213567": "anime", "Q747381": "anime", "Q7889": "game", "Q7058673": "game",
                    "Q11424": "movie", "Q5398426": "drama", "Q196600": "media_mix"}
_EPISODES = re.compile(r"第\s*\d+\s*話\s*[-–〜～]\s*第?\s*\d+\s*話")   # a range only: 第1話 alone may be "from"
# a sure candidate's order: categories of works a song is made for first, roles as a work credits them
CATEGORY_ORDER = ("anime", "game", "idol_2d", "drama", "movie", "media_mix", "other", "commercial")
ROLE_ORDER = ("opening", "ending", "theme", "insert", "character_song", "unit_song", "image_song")
_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_MARKUP = re.compile(r"'{2,}|<ref[^>]*/>|<ref[^>]*>.*?</ref>|<[^>]+>|\{\{[^{}]*\}\}")


@dataclass
class Candidate:
    work: tieup.Work                    # its titles, other names, category, the work's page as its reference
    role: str
    seq: int = 0
    detail: str = ""
    source: str = ""                    # the page the tie-up was read on
    mentions: int = 1
    sure: bool = False

    def as_tieup(self) -> tieup.TieUp:
        return tieup.TieUp(self.work, self.role, self.seq, self.detail)


class Wikimedia:
    """Wikipedia and Wikidata over HTTP: one request at a time, MIN_INTERVAL_SEC apart; ``fetch`` (url, params) ->
    JSON stands in for the network in tests."""

    def __init__(self, fetch: Optional[Callable[[str, dict], dict]] = None):
        self._fetch = fetch or self._http
        self._lock = threading.Lock()
        self._last = 0.0

    def _http(self, url: str, params: dict) -> dict:
        query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2", "maxlag": "5"})
        req = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            return json.load(resp)

    def get(self, url: str, **params) -> dict:
        with self._lock:
            wait = self._last + MIN_INTERVAL_SEC - time.monotonic()
            if wait > 0 and self._fetch is self._http:
                time.sleep(wait)
            try:
                return self._fetch(url, params)
            finally:
                self._last = time.monotonic()


_SHARED: Optional[Wikimedia] = None


def shared() -> Wikimedia:
    """The process's one Wikimedia client (its requests spaced across every lookup of the process)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = Wikimedia()
    return _SHARED


def _plain(text: str) -> tuple[str, dict[str, str]]:
    """Wikitext as read: links as their labels (label -> page kept), templates, references and markup out."""
    pages: dict[str, str] = {}

    def link(m: re.Match) -> str:
        page, label = m.group(1).strip(), (m.group(2) or m.group(1)).strip()
        pages.setdefault(label, page)
        return label

    previous = None
    while previous != text:     # nested templates from the inside out
        previous, text = text, _MARKUP.sub("", text)
    return _LINK.sub(link, text), pages


def _category(medium: Optional[str]) -> str:
    return next((c for w, c in MEDIUM_WORDS if w == medium), "other")


def _role(word: str) -> str:
    return next(r for w, r in ROLE_WORDS if w == word)


# a version note at the title's end: "(TV size)", "(Instrumental)", "(Club remix)", "(ふたりば～じょん)"
_NOTE_WORDS = (r"(?<![A-Za-z])(?:TV|instrumental|inst|off\s*vocal|karaoke|short|full|size|ver|version|remix|mix|edit|"
               r"arrange|cover)(?![A-Za-z])|テレビ|オフボーカル|カラオケ|バージョン|ば[ー～〜~]じょん")
_VERSION_NOTE = re.compile(rf"\s*[\(（\[【][^)）\]】]*?(?:{_NOTE_WORDS})[^)）\]】]*[\)）\]】]\s*$", re.IGNORECASE)
_FEAT = re.compile(r"(?:\s+|\s*[\(（]\s*)(?:feat\.|ft\.|feat\s|featuring\s)\s*(?P<who>[^)）]+)[\)）]?\s*$",
                   re.IGNORECASE)
_ARTIST_SPLIT = re.compile(r"\s+from\s+|\s*(?:feat\.?|ft\.|featuring|&|＆|×|/|／|,|、|・)\s*", re.IGNORECASE)
# a character's credit: "キャラA(CV.声優A)", "キャラB(CV:声優B)", "キャラC(cv.声優C)"
_VOICE = re.compile(r"(?P<who>[^\s(（\[【\]】&＆/／,、・]*)\s*[\(（]\s*(?:CV|C\.V\.|声)\s*[.:：．・]?\s*(?P<cast>[^)）]+)[\)）]",
                    re.IGNORECASE)


def song_title(title: str) -> tuple[str, list[str]]:
    """The song's name as an article writes it, and the artists the title credits: a version note at its end
    ("(TV size)", "(Instrumental)") and a "feat. …" left out (that artist kept)."""
    featured: list[str] = []
    previous = None
    while previous != title:
        previous = title
        title = _VERSION_NOTE.sub("", title)
        m = _FEAT.search(title)
        if m and m.start() > 0:
            featured.append(m.group("who").strip())
            title = title[:m.start()]
    return title.strip(), featured


def artist_names(artist: str, featured: list[str] = ()) -> list[str]:
    """The names an artist field holds ("A from B", "A feat. B", "A & B"), with the title's featured artists: an
    article may credit any of them. A character's credit, "Unit[キャラA(CV.声優A)&…]", gives the unit, then the
    voice actors (their pages list the songs they sang), then the characters."""
    names, cast, characters = [], [], []
    for part in [artist, *featured]:
        for m in _VOICE.finditer(part or ""):
            cast += _ARTIST_SPLIT.split(m.group("cast"))
            characters.append(m.group("who"))
        rest = re.sub(r"[\[\]【】]", " / ", _VOICE.sub(" / ", part or ""))
        names += _ARTIST_SPLIT.split(rest)
    names = [n.strip() for n in names + cast + characters if n and n.strip()]
    return [n for n in dict.fromkeys(names) if len(reading.fine_key(n)) >= 3]


def _role_in(label: str) -> Optional[tuple[str, int]]:
    """The role a heading or a label names (its number when it says one), or None."""
    for word, role in ROLE_WORDS:
        at = label.find(word)
        if at >= 0:
            digits = re.match(r"\s*([0-9０-９])", label[at + len(word):])
            seq = int(digits.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))) if digits else 0
            return role, seq
    return None


def listed_work(page: str) -> str:
    """The work a page is, where it lists a song: its title, a discography page's work (作品のディスコグラフィ:
    作品)."""
    title = re.sub(r"\s*\([^)]*\)$", "", page)
    m = _LISTING.match(title)
    return m.group(1) if m else title


def found_in(name: str, texts: list[str], similar: bool = True) -> list[int]:
    """The texts naming ``name`` as the library's search finds a title (core.reading.find): its letters contained,
    its sounds contained (最高潮! / 最高潮！, ヴァ / バ), or with ``similar`` nearly contained (one sound off per
    reading.SIMILAR_EVERY letters: a long vowel, a small kana, a typing slip)."""
    query = [(reading.fine_key(name), reading.normalize(name))]
    return sorted(reading.find(query, [(reading.fine_key(t), reading.normalize(t)) for t in texts], similar=similar))


def _mark(line: str) -> str:
    return re.match(r"[#*:;]*", line.strip()).group(0)


def _entry(lines: list[str], i: int) -> list[str]:
    """The line naming the song and the lines describing it: a table cell's row (its next cells, to |-), a list
    item's or a label's deeper lines (# song / #: …, ; 「song」 / : …), a heading's paragraph (=== song === / text);
    a line of running text alone (the next line is another paragraph, another song's)."""
    first = lines[i].strip()
    mark = _mark(first)
    base = mark[:-1] if mark.endswith(";") else mark
    out = [lines[i]]
    for line in lines[i + 1:i + (8 if first.startswith(("|", "!")) else 5)]:
        text = line.strip()
        if first.startswith(("|", "!")):
            if text.startswith(("|-", "|}", "{|")):
                break
            out.append(line)
            continue
        if first.startswith("="):
            if text.startswith("=") or _mark(text):
                break
            if text:
                out.append(line)
            continue
        m = _mark(text)
        if not text or not (m.startswith(base) and len(m) > len(base) and not m.endswith(";")
                            and (mark.endswith(";") or len(m) > len(mark))):
            break
        out.append(line)
    return out


def _listed_role(lines: list[str], i: int, entry: list[str]) -> Optional[tuple[str, int]]:
    """The role a work's page lists the song under: before the song on its line, else in the lines describing it
    (": …のオープニングテーマ。"; not a work inside the work's), else in the heading or label above it."""
    line = lines[i]
    before = line[:line.find("「")] if "「" in line else ""
    label = _role_in(before) if before.strip() else None
    for after in entry[1:]:
        if label is not None:
            break
        if not _FICTION.search(after):
            label = _role_in(after)
    for back in range(i - 1, max(-1, i - 7), -1):
        if label is not None:
            break
        prev = lines[back].strip()
        if prev.startswith(("=", ";")) and "「" not in prev:
            label = _role_in(prev)
    return label


def mentions(wikitext: str, song: str, whole: bool, page: str = "") -> list[tuple[str, str, str, int, str, str]]:
    """The tie-ups a page states for the song: (work, its page, role, number, detail, category), each once. ``whole``:
    every line (the song's own page; there a work named by its medium alone, メディアミックスプロジェクト『…』, too,
    with no role), else each line naming the song with the lines describing it (`_entry`).
    ``page``: the page read, the work (`listed_work`) where it lists the song without naming a work: under the role
    of its line, its description or its heading / label, else with none (tieup.UNSPECIFIED; the caller checks the
    page is a work)."""
    text, pages = _plain(wikitext)
    lines = text.split("\n")
    key = reading.fine_key(song)
    entries = [(-1, lines)] if whole else [(i, _entry(lines, i)) for i in found_in(song, lines)] if key else []
    found: dict[tuple, tuple] = {}
    work = listed_work(page) if page and not whole else ""
    work_page = page if work == re.sub(r"\s*\([^)]*\)$", "", page) else work
    for i, entry in entries:
        window = " ".join(entry)
        stated = False
        for m in _MENTION.finditer(window):
            name = m.group("work").strip()
            if not name or reading.fine_key(name) == key or (found_in(song, [name]) and found_in(name, [song])):
                continue                                          # the song itself, quoted
            if _FICTION.search(window[max(0, m.start("work") - 8):m.start("work")]):
                continue
            seq = int(m.group("seq").translate(str.maketrans("０１２３４５６７８９", "0123456789")) or 0)
            role = _role(m.group("role"))
            after = window[m.end():m.end() + 40].split("。")[0]     # the same sentence's episodes only
            episodes = _EPISODES.search(after)
            found.setdefault((name, role, seq), (name, pages.get(name, name), role, seq,
                                                 episodes.group(0) if episodes else "", _category(m.group("medium"))))
            stated = True
        for m in _WORK_NAMED.finditer(window) if whole else ():    # the song's own page: its work without a role
            name = m.group("work").strip()
            if (not name or reading.fine_key(name) == key or any(k[0] == name for k in found)
                    or _FICTION.search(window[max(0, m.start("work") - 8):m.start("work")])):
                continue
            found[(name, tieup.UNSPECIFIED, 0)] = (name, pages.get(name, name), tieup.UNSPECIFIED, 0, "",
                                                   _category(m.group("medium")))
        if work and not stated:      # listed here without a work named: this page's work
            role, seq = _listed_role(lines, i, entry) or (tieup.UNSPECIFIED, 0)
            found.setdefault((work, role, seq), (work, work_page, role, seq, "", "other"))
    return list(found.values())


class _Memo:
    """A search's requests answered once each (a page's item is asked for by the work check and by its names)."""

    def __init__(self, api: Wikimedia):
        self._api = api
        self._answers: dict[tuple, dict] = {}

    def get(self, url: str, **params) -> dict:
        key = (url, *sorted(params.items()))
        if key not in self._answers:
            self._answers[key] = self._api.get(url, **params)
        return self._answers[key]


def _entity(api, page: str) -> tuple[Optional[str], dict]:
    """A Japanese Wikipedia page's Wikidata item and its labels, aliases and claims; (None, {}) where it has none."""
    try:
        info = api.get(WIKIPEDIA, action="query", prop="pageprops", titles=page, redirects="1")
        item = (info.get("query", {}).get("pages") or [{}])[0].get("pageprops", {}).get("wikibase_item")
        if not item:
            return None, {}
        data = api.get(WIKIDATA, action="wbgetentities", ids=item, props="labels|aliases|claims",
                       languages="|".join(WIKIDATA_LANGS))
        return item, data.get("entities", {}).get(item, {})
    except Exception:
        return None, {}


def _categories(api, titles: list[str]) -> tuple[list[dict], list[str]]:
    """The pages (titles, at most 50) as the API gives them, and the categories they are in (no hidden ones)."""
    out = api.get(WIKIPEDIA, action="query", prop="categories", titles="|".join(titles), cllimit="max",
                  clshow="!hidden", redirects="1")
    pages = out.get("query", {}).get("pages") or []
    return pages, [c.get("title", "").split(":", 1)[-1] for p in pages for c in p.get("categories", [])]


def page_category(api, page: str) -> Optional[str]:
    """The category of the work a Japanese Wikipedia page is, by its Wikipedia categories and theirs, or None: not a
    work's page (a person's, a group's, a record's, or no page). A work (_WORK_CATEGORY, or its franchise's own
    category) about idols (アイドルを題材とした…, on it or on its categories: アイドルマスター's is) is a 2D idol, and so
    is a fictional idol group (架空の音楽グループ with idols in its name or categories: a school idol club); a
    fictional band with no idols is its work's, not a work."""
    try:
        pages, cats = _categories(api, [page])
        info = pages[0] if pages else {}
        if not info or info.get("missing") or "invalid" in info:
            return None
        parents = _categories(api, [f"Category:{c}" for c in cats[:50]])[1] if cats else []
    except Exception:
        return None
    title = re.sub(r"\s*\([^)]*\)$", "", info.get("title", page))
    works = [c for c in cats if _WORK_CATEGORY.search(c)]
    franchise = title in cats and not any(_NOT_WORK_CATEGORY.search(c) for c in cats)
    idols = any("アイドルを題材とした" in c for c in cats + parents)
    fictional_idols = (any(re.search(r"架空の(?:音楽グループ|アイドル)", c) for c in cats)
                       and (idols or "アイドル" in title or any("アイドル" in c for c in cats + parents)))
    if not works and not franchise and not fictional_idols:
        return None
    if idols or fictional_idols:
        return "idol_2d"
    words = " ".join(works)
    return next((c for w, c in CATEGORY_WORDS if w in words), "media_mix" if not works else "other")


def _work_class(entity: dict) -> Optional[str]:
    """The category of a work's item (WIKIDATA_CLASSES), or None: not a work."""
    classes = [c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
               for c in entity.get("claims", {}).get("P31", [])]
    return next((WIKIDATA_CLASSES[q] for q in classes if q in WIKIDATA_CLASSES), None)


def _work_names(api, page: str, category: str) -> tieup.Work:
    """The work of a Japanese Wikipedia page: its Wikidata item's labels and aliases as titles and other names (the
    page's title first, in Japanese), the page as its reference."""
    page = page.replace("_", " ")       # a link's spelling: Wikipedia's titles have spaces (Work_Name! -> Work Name!)
    w = tieup.Work("", category, [("ja-JP", re.sub(r"\s*\([^)]*\)$", "", page))], [],
                   [("Wikipedia", "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(page.replace(" ", "_")))])
    item, entity = _entity(api, page)
    if not item:
        return w
    if w.category == "other":
        w.category = _work_class(entity) or "other"
    seen = {reading.fine_key(t) for _l, t in w.titles}
    for code, label in entity.get("labels", {}).items():
        lang = WIKIDATA_LANGS.get(code)
        if lang and reading.fine_key(label["value"]) not in seen and lang not in {l for l, _t in w.titles}:
            w.titles.append((lang, label["value"]))
            seen.add(reading.fine_key(label["value"]))
    for code, aliases in entity.get("aliases", {}).items():
        lang = WIKIDATA_LANGS.get(code)
        for alias in aliases if lang else ():
            if reading.fine_key(alias["value"]) not in seen:
                w.aliases.append(("synonym", lang, alias["value"]))
                seen.add(reading.fine_key(alias["value"]))
    w.urls.append(("Wikidata", f"https://www.wikidata.org/wiki/{item}"))
    return w


def search(api: Wikimedia, title: str, artist: str = "", album: str = "") -> list[Candidate]:
    """The tie-ups Wikimedia gives for a song, the most mentioned first; a sure one marked (see the module)."""
    songs, artists, searches = queries(title, artist, album)
    if not songs:
        return []
    api = _Memo(api)
    # the song's own page first (Japanese Wikipedia titles a song "曲名", "曲名 (曲)", "曲名 (歌手の曲)", "曲名
    # (シングル)"): its lead says what the song is used in
    keys = {reading.fine_key(s) for s in songs}
    names = [n for s in songs for n in (s, f"{s} (曲)", f"{s} (シングル)", *(f"{s} ({a}の曲)" for a in artists[:2]))]
    own_pages = []
    for start in range(0, len(names), 40):     # the API takes 50 titles a request
        pages = api.get(WIKIPEDIA, action="query", titles="|".join(names[start:start + 40]), redirects="1",
                        prop="pageprops", ppprop="disambiguation").get("query", {}).get("pages", [])
        own_pages += [p["title"] for p in pages if not p.get("missing")
                      and "disambiguation" not in p.get("pageprops", {})
                      and reading.fine_key(re.sub(r"\s*\([^)]*\)$", "", p["title"])) in keys]
    read = _Reading()
    for page in own_pages:
        read.page(api, page, songs)
    own = bool(read.found)
    # else the search's pages, query after query (the song's names alone, with an artist, the album), until two of
    # them agree on one work or PAGES_MAX are read
    for query in searches if not own else ():
        hits = api.get(WIKIPEDIA, action="query", list="search", srsearch=query, srlimit=str(PAGES_READ),
                       srnamespace="0").get("query", {}).get("search", [])
        for hit in hits[:PAGES_READ]:
            if len(read.read) < PAGES_MAX:
                read.page(api, hit["title"], songs, artists)
        if _agreed(read, read.works_pages(api), make=False, agreed=True) is not None or len(read.read) >= PAGES_MAX:
            break
    for key, c in read.found.items():     # a 2D idol, else the medium the mention named, else Wikidata's class,
        page = read.work_pages[key]       # else the page's categories
        c.work = _work_names(api, page, c.work.category)
        if c.work.category == "other" or read.category(api, page) == "idol_2d":
            c.work.category = read.category(api, page) or "other"
    works = read.works_pages(api)
    sure = _sure(read, works) if own else _agreed(read, works)
    if sure is not None:
        read.found[sure].sure = True
    # a work with a role found is not suggested again without one
    named = {k[0] for k in read.found if k[1] != tieup.UNSPECIFIED}
    out = [c for k, c in read.found.items() if c.sure or k[1] != tieup.UNSPECIFIED or k[0] not in named]
    return sorted(out, key=lambda c: (not c.sure, -c.mentions))


def queries(title: str, artist: str = "", album: str = "") -> tuple[list[str], list[str], list[str]]:
    """What a song is looked up by: (its names as an article may write them: the title, without a "feat. …" or a
    version note, without a subtitle after 〜 / ~ / -; the artists the fields credit; the searches, in turn: each name
    quoted alone, with each artist, then the album's name when it is not the song's)."""
    cleaned, featured = song_title(title.strip())
    songs = [cleaned, title.strip()]
    sub = re.split(r"\s*[〜～~]|\s+-\s+", cleaned)[0]
    if sub.count("(") + sub.count("（") <= sub.count(")") + sub.count("）"):     # not cut inside a note
        songs.append(sub)
    songs = [s for s in dict.fromkeys(s.strip() for s in songs) if len(reading.fine_key(s)) >= 2]
    artists = artist_names(artist, featured)
    searches = [f'"{s}"' for s in songs[:2]] + [f'"{songs[0]}" {a}' for a in artists[:2]] if songs else []
    album_name, _ = song_title(album.strip())
    if album_name and reading.fine_key(album_name) not in {reading.fine_key(s) for s in songs}:
        searches.append(f'"{album_name}"')
    return songs, artists, list(dict.fromkeys(searches))


def _works(read: "_Reading", eligible: set[str]) -> dict[str, list[tuple]]:
    """The candidates by their work (its page's key), of the works a tag may be taken from (``eligible``)."""
    works: dict[str, list[tuple]] = {}
    for key in read.found:
        if key[0] in eligible:
            works.setdefault(key[0], []).append(key)
    return works


def _without_role(read: "_Reading", keys: list[tuple], make: bool = True) -> tuple:
    """The work of these candidates with no role: its candidate, made from theirs where none was found (``make``)."""
    key = (keys[0][0], tieup.UNSPECIFIED, 0)
    if make and key not in read.found:
        c = read.found[keys[0]]
        read.found[key] = Candidate(c.work, tieup.UNSPECIFIED, 0, "", c.source,
                                    len(set().union(*(read.stated_on[k] for k in keys))))
        read.work_pages[key] = read.work_pages[keys[0]]
        read.stated_on[key] = set().union(*(read.stated_on[k] for k in keys))
    return key


def _category_rank(read: "_Reading", keys: list[tuple]) -> int:
    category = read.found[keys[0]].work.category
    return -CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else -len(CATEGORY_ORDER)


def _agreed(read: "_Reading", eligible: set[str], make: bool = True, agreed: bool = False) -> Optional[tuple]:
    """The work the search's pages name most, of the ones whose pages are works' (``eligible``; each page one naming
    an artist; ties: the category nearer the front of CATEGORY_ORDER, then the one found first): whatever was found
    gives a tag, as the Edit tab's Find on the Web checks it (a tag a little wrong now and then is better than none:
    the user corrects it). Its role the one more of its pages name than any other, else none (the work is what
    matters). ``agreed``: only a work two pages name, more than any other (the searches go on until one is)."""
    works = _works(read, eligible)
    pages = {w: set().union(*(read.stated_on[k] for k in keys)) for w, keys in works.items()}
    best = sorted(pages, key=lambda w: (len(pages[w]), _category_rank(read, works[w])), reverse=True)
    if not best:
        return None
    if agreed and (len(pages[best[0]]) < 2 or (len(best) > 1 and len(pages[best[1]]) == len(pages[best[0]]))):
        return None
    keys = works[best[0]]
    roles = sorted((k for k in keys if k[1] != tieup.UNSPECIFIED), key=lambda k: len(read.stated_on[k]), reverse=True)
    if roles and (len(roles) == 1 or len(read.stated_on[roles[0]]) > len(read.stated_on[roles[1]])):
        return roles[0]
    return _without_role(read, keys, make)


def _sure(read: "_Reading", eligible: set[str]) -> Optional[tuple]:
    """The work of a song's own page to take: of the ones whose pages are works' (``eligible``), the one it names
    most (a work its page: the names linking one page are one work; ties: the category nearer the front of
    CATEGORY_ORDER, then the one named first), with the role it names first by ROLE_ORDER (an own page's lead names
    the song's main role), else none."""
    works = _works(read, eligible)
    best = sorted(works, key=lambda w: (len(works[w]), _category_rank(read, works[w])), reverse=True)
    if not best:
        return None
    keys = works[best[0]]
    roles = [k for k in keys if k[1] in ROLE_ORDER]
    if roles:
        return min(roles, key=lambda k: ROLE_ORDER.index(k[1]))
    named = [k for k in keys if k[1] != tieup.UNSPECIFIED]
    return named[0] if len(named) == 1 else _without_role(read, keys)


class _Reading:
    """What the pages read so far state: key (the work's page, role, number) -> candidate, its work's page, the pages
    stating it (a candidate's mentions)."""

    def __init__(self):
        self.found: dict[tuple, Candidate] = {}
        self.work_pages: dict[tuple, str] = {}
        self.stated_on: dict[tuple, set[str]] = {}
        self.read: set[str] = set()
        self.classes: dict[str, Optional[str]] = {}     # a page -> its work's category, or None: not a work's

    def category(self, api, page: str) -> Optional[str]:
        if page not in self.classes:
            self.classes[page] = page_category(api, page)
        return self.classes[page]

    def works_pages(self, api) -> set[str]:
        """The works found whose pages are works' (page_category): the ones a tag is taken from."""
        return {k[0] for k in self.found if self.category(api, self.work_pages[k]) is not None}

    def page(self, api, page: str, songs: list[str], artists: Optional[list[str]] = None) -> None:
        """One page read. ``artists``: a page a search found, read only where it names one of the artists (each
        mention of the song, and a work page listing it under a role); else the song's own page, read whole."""
        if page in self.read:
            return
        self.read.add(page)
        parsed = api.get(WIKIPEDIA, action="parse", page=page, prop="wikitext", redirects="1").get("parse", {})
        wikitext = parsed.get("wikitext", "")
        # its letters or its sounds only: one sound off is found anywhere in a whole page
        if artists and not any(found_in(a, [wikitext], similar=False) for a in artists):
            return
        own = artists is None
        for song in songs:
            for work, work_page, role, seq, detail, category in mentions(wikitext, song, own, "" if own else page):
                if work_page in (page, listed_work(page)):
                    # the page listing the song: a work only where its page is one (not a singer's, an album's)
                    category = self.category(api, work_page)
                    if category is None:
                        continue
                # a work is its page: the names linking one page, and the page listing the song itself, are one
                key = (reading.fine_key(re.sub(r"\s*\([^)]*\)$", "", work_page)), role, seq)
                if key not in self.found:
                    self.found[key] = Candidate(tieup.Work("", category, [("ja-JP", work)]), role, seq, detail,
                                                "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(page.replace(" ", "_")),
                                                mentions=0)
                    self.work_pages[key] = work_page
                if page not in self.stated_on.setdefault(key, set()):
                    self.stated_on[key].add(page)
                    self.found[key].mentions += 1


# ---- the answers kept ----

def _cache(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS web_query (query_key TEXT PRIMARY KEY, fetched_at REAL, result_json TEXT)")
    return con


def _to_json(cands: list[Candidate]) -> str:
    return json.dumps([asdict(c) for c in cands], ensure_ascii=False)


def _from_json(text: str) -> list[Candidate]:
    out = []
    for d in json.loads(text):
        w = d.pop("work")
        out.append(Candidate(tieup.Work(w["id"], w["category"], [tuple(t) for t in w["titles"]],
                                        [tuple(a) for a in w["aliases"]], [tuple(u) for u in w["urls"]]), **d))
    return out


def lookup(api: Wikimedia, title: str, artist: str = "", cache_path: str = CACHE_NAME,
           album: str = "") -> list[Candidate]:
    """`search`, its answer kept for CACHE_DAYS."""
    key = "\0".join((f"wikimedia{RULES}", reading.fine_key(title), reading.fine_key(artist), reading.fine_key(album)))
    con = _cache(cache_path)
    try:
        row = con.execute("SELECT fetched_at, result_json FROM web_query WHERE query_key=?", (key,)).fetchone()
        if row is not None and time.time() - row[0] < CACHE_DAYS * 86400:
            return _from_json(row[1])
        found = search(api, title, artist, album)
        with con:
            con.execute("INSERT OR REPLACE INTO web_query VALUES (?, ?, ?)", (key, time.time(), _to_json(found)))
        return found
    finally:
        con.close()


# (what is found goes into the library through its owner: core.library.jobs.add_found_tieups)

def sure_of(api: Wikimedia, song: tuple[str, str, str], cache_path: str = CACHE_NAME) -> Optional[Candidate]:
    """The sure tie-up Wikimedia has for a song ((title, artist, album): core.library.jobs.song_of), or None: the
    network only, no library."""
    title, artist, album = song
    return next((c for c in lookup(api, title, artist, cache_path, album) if c.sure), None)

