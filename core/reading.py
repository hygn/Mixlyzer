"""Readings for search: a track is found by how its title / artist / album sound, in any script --
あのね by "anone", "ano ne" or 아노네; 東京 by "tokyo" or 도쿄; 사랑해 by "saranghae".

Three layers:
1. Readings (``readings``): each script by its own reader into romaji -- the intermediate
   representation, stored in the library (up to READING_SLOTS (7) per text, the user's own too) and
   editable. Han characters are ambiguous, so a text of Han without kana gets one reading per
   language (the languages of Settings > Search, for the library and for a query); the same
   reading twice is kept once.
     Japanese          Sudachi (SudachiDict core) reads each word; kana become wapuro romaji
                       (toukyou, shi, tsu; っ doubles the next consonant, ー repeats the vowel);
                       the particles は / へ / を read wa / e / o.
     Mandarin          pypinyin (by word: 重庆 chong qing), tones dropped, ü as u.
     Taiwan Mandarin   the Mandarin reading with Taiwan's where Unihan's kMandarin differs
                       (assets/readings/unihan_taiwan_mandarin.tsv: 乾 gan, 万 mo).
     Cantonese         pycantonese (rime-cantonese, HKCanCor) Jyutping, tones dropped; a form it
                       lacks (会, 学: Japanese new / simplified) read as its Unihan variant
                       (assets/readings/unihan_cantonese_variants.tsv: 會, 學).
     Korean Hanja      Unihan kHangul (assets/readings/unihan_korean.tsv; the standard reading),
                       spelled as a word: the initial sound law on its first syllable (李 이, 女 여,
                       樂 낙), 렬 / 률 as 열 / 율 after a vowel or ㄴ (比率 비율), 不 as 부 before
                       ㄷ / ㅈ (不適合 부적합); then as Hangul (千本桜 cheonbonaeng).
     Sino-Vietnamese   Unihan kVietnamese (assets/readings/unihan_vietnamese.tsv: of several the
                       one closest to the Chinese readings -- Unihan mixes in Nom readings, so
                       some are not Sino-Vietnamese, 女 nua), diacritics dropped.
                       A character missing from these two tables is read in Mandarin, which shares
                       its Middle Chinese origin (愛 ai). Japanese new forms read via their old ones.
     Hangul            Revised Romanization (the official standard) with its sound changes:
                       liaison (읽어 ilgeo), nasalization (독립 dongnip), ㄹ as ㄴ (종로 jongno),
                       lateralization (신라 silla), aspiration (좋다 jota), palatalization (같이 gachi).
     other             Latin letters and digits kept as written, other scripts by anyascii
                       (Москва Moskva).
2. Normalization (``normalize``): romaji into a search form where spellings of one sound
   agree: shi = si, tsu = tu = 츠 (cheu), long vowels and doubled letters collapsed (toukyou =
   tokyo, kitto = kito), voicing ignored (k g / s z j ch ts / t d / h f b p v: Korean spells
   Japanese k and g alike, rendaku), l = r. Hangul in a query is read twice: as Korean (사회 by
   社会's Korean Hanja reading) and with Korean's spelling of foreign words (ㅓ o, ㅡ u, ㅐ e), as
   a Korean writes a Japanese title.
3. Matching (``find``): the query's forms against each reading's: contained exactly, or --
   from SIMILAR_MIN letters on -- nearly (RapidFuzz partial ratio, about one sound in five
   off), so a similar pronunciation is found too.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from functools import lru_cache
from typing import Callable, Iterable

from anyascii import anyascii
from rapidfuzz import fuzz, process

READING_SLOTS = 7  # readings stored per text (language readings and the user's own)
SOURCES = {"ja": "Japanese", "zh": "Mandarin", "zh-tw": "Taiwan Mandarin", "yue": "Cantonese",
           "ko-hanja": "Korean Hanja", "vi": "Sino-Vietnamese", "ko": "Korean", "text": "as written",
           "user": "set by the user"}
# the readings of Han without kana, in slot order (Taiwan Mandarin only where it differs from Mandarin)
HAN_LANGUAGES = ("ja", "zh", "yue", "ko-hanja", "vi", "zh-tw")
EXACT_MIN = 3     # a shorter normalized query would match nearly every track
SIMILAR_MIN = 6   # similar matching from this many letters of the normalized query on (5 found 26 more tracks)
SIMILAR_EVERY = 5  # one sound off allowed per this many letters

# ---- kana ----

_KANA_DIGRAPHS = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she", "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "じぇ": "je", "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che", "ぢゃ": "ja", "ぢゅ": "ju",
    "ぢょ": "jo", "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo", "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    # sounds written with small vowels (mostly loanwords)
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo", "ふゅ": "fyu", "てぃ": "ti", "でぃ": "di",
    "とぅ": "tu", "どぅ": "du", "てゅ": "tyu", "でゅ": "dyu", "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo", "ゔゅ": "vyu", "つぁ": "tsa", "つぃ": "tsi",
    "つぇ": "tse", "つぉ": "tso", "いぇ": "ye", "くぁ": "kwa", "くぃ": "kwi", "くぇ": "kwe", "くぉ": "kwo",
    "ぐぁ": "gwa", "すぃ": "si", "ずぃ": "zi",
}
_KANA = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko", "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so", "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to", "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho", "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo", "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o", "ん": "n", "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o", "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
    "ゕ": "ka", "ゖ": "ke",
}
_VOWELS = "aeiou"


def _to_hiragana(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:          # katakana -> hiragana
            out.append(chr(code - 0x60))
        elif ch in "ヷヸヹヺ":
            out.append({"ヷ": "ゔぁ", "ヸ": "ゔぃ", "ヹ": "ゔぇ", "ヺ": "ゔぉ"}[ch])
        else:
            out.append(ch)
    return "".join(out)


def kana_to_romaji(text: str) -> str:
    """Hiragana / katakana as wapuro romaji; anything else is kept."""
    s = _to_hiragana(text)
    out: list[str] = []
    double_next = False
    i = 0
    while i < len(s):
        pair, ch = s[i:i + 2], s[i]
        if ch == "っ":
            double_next = True
            i += 1
            continue
        if ch == "ー":
            last = next((c for c in reversed("".join(out)) if c.isalpha()), "")
            if last in _VOWELS:
                out.append(last)
            i += 1
            continue
        if pair in _KANA_DIGRAPHS:
            syl, i = _KANA_DIGRAPHS[pair], i + 2
        elif ch in _KANA:
            syl, i = _KANA[ch], i + 1
        else:
            syl, i = ch, i + 1
        if double_next and syl[:1].isalpha() and syl[0] not in _VOWELS:
            syl = ("t" + syl) if syl.startswith("ch") else (syl[0] + syl)
        double_next = False
        out.append(syl)
    return "".join(out)


# ---- hangul: Revised Romanization with its sound changes ----

_INITIALS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_MEDIALS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi",
            "yu", "eu", "ui", "i"]
_FINALS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
           "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_RR_INITIAL = {"ㄱ": "g", "ㄲ": "kk", "ㄴ": "n", "ㄷ": "d", "ㄸ": "tt", "ㄹ": "r", "ㅁ": "m", "ㅂ": "b", "ㅃ": "pp",
               "ㅅ": "s", "ㅆ": "ss", "ㅇ": "", "ㅈ": "j", "ㅉ": "jj", "ㅊ": "ch", "ㅋ": "k", "ㅌ": "t", "ㅍ": "p", "ㅎ": "h"}
# a final before a consonant or at the end: its representative sound
_RR_FINAL = {"": "", "ㄴ": "n", "ㄵ": "n", "ㄶ": "n", "ㄹ": "l", "ㄼ": "l", "ㄽ": "l", "ㄾ": "l", "ㅀ": "l",
             "ㅁ": "m", "ㄻ": "m", "ㅇ": "ng", **{f: "k" for f in "ㄱㄲㅋㄳㄺ"}, **{f: "t" for f in "ㄷㅅㅆㅈㅊㅌㅎ"},
             **{f: "p" for f in "ㅂㅍㄿㅄ"}}
_SPLIT = {"ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"),
          "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"),
          "ㅄ": ("ㅂ", "ㅅ")}
_ASPIRATED = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ", "ㅂ": "ㅍ"}
_K_FINALS, _T_FINALS, _P_FINALS = set("ㄱㄲㅋㄳㄺ"), set("ㄷㅅㅆㅈㅊㅌㅎ"), set("ㅂㅍㄼㄿㅄ")
# Korean's spelling of foreign words: a query's Hangul read as a Korean writes a Japanese title
_KOREAN_SPELLING = [("yeo", "yo"), ("yae", "ye"), ("wae", "we"), ("eo", "o"), ("eu", "u"), ("ae", "e"),
                    ("oe", "e"), ("ui", "i")]


def _korean_word(word: str) -> str:
    """One word of Hangul syllables in Revised Romanization, its sound changes applied."""
    syl = []
    for ch in word:
        code = ord(ch) - 0xAC00
        syl.append([_INITIALS[code // 588], _MEDIALS[(code % 588) // 28], _FINALS[code % 28]])
    for a, b in zip(syl, syl[1:]):
        fin, ini = a[2], b[0]
        if not fin:
            continue
        if fin in ("ㅎ", "ㄶ", "ㅀ"):                 # ㅎ: aspirates the next, or is silent
            rest = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[fin]
            if ini in _ASPIRATED:
                b[0], a[2] = _ASPIRATED[ini], rest
                continue
            a[2] = "ㄴ" if (fin == "ㅎ" and ini == "ㄴ") else rest
            fin = a[2]
            if not fin:
                continue
        elif ini == "ㅎ" and (fin in _ASPIRATED or fin in ("ㄺ", "ㄼ", "ㄵ")):   # aspiration: 축하 chuka
            keep, base = ("", fin) if fin in _ASPIRATED else _SPLIT[fin]
            b[0], a[2] = ("ㅊ" if base == "ㄷ" and b[1] == "i" else _ASPIRATED[base]), keep
            continue
        if ini == "ㅇ" and b[1] == "i" and fin in ("ㄷ", "ㅌ", "ㄾ"):     # palatalization: 같이 gachi
            b[0], a[2] = ("ㅈ" if fin == "ㄷ" else "ㅊ"), ("ㄹ" if fin == "ㄾ" else "")
            continue
        if ini == "ㅇ":                                                  # liaison: 읽어 ilgeo
            if fin != "ㅇ":
                a[2], b[0] = _SPLIT.get(fin, ("", fin))
            continue
        if ini == "ㄹ" and fin not in ("ㄹ", "ㄴ"):                       # ㄹ as ㄴ: 종로 jongno
            b[0] = ini = "ㄴ"
        if fin == "ㄴ" and ini == "ㄹ":                                    # lateralization: 신라 silla
            a[2] = "ㄹ"
        elif fin in ("ㄹ", "ㅀ", "ㄾ") and ini == "ㄴ":                   # 설날 seollal
            b[0] = "ㄹ"
        elif ini in ("ㄴ", "ㅁ"):                                          # nasalization: 독립 dongnip
            a[2] = "ㅇ" if fin in _K_FINALS else "ㄴ" if fin in _T_FINALS else "ㅁ" if fin in _P_FINALS else fin
    out = []
    for k, (ini, med, fin) in enumerate(syl):
        initial = "l" if (ini == "ㄹ" and k and syl[k - 1][2] == "ㄹ") else _RR_INITIAL[ini]
        out.append(initial + med + _RR_FINAL[fin])
    return "".join(out)


def korean_to_roman(text: str, *, foreign_spelling: bool = False) -> str:
    """Hangul in Revised Romanization (other characters by anyascii); ``foreign_spelling``: its
    vowels as Korean spells foreign words (ㅓ o, ㅡ u, ㅐ e), for a query."""
    out = re.sub(r"[가-힣]+", lambda m: _korean_word(m.group(0)), text)
    out = "".join(c if c.isascii() else anyascii(c) for c in out)
    if foreign_spelling:
        for old, new in _KOREAN_SPELLING:
            out = out.replace(old, new)
    return out


# ---- the 2-set (Dubeolsik) Korean keyboard: a query typed without the 한/영 key ----

_DUBEOLSIK = {"q": "ㅂ", "w": "ㅈ", "e": "ㄷ", "r": "ㄱ", "t": "ㅅ", "y": "ㅛ", "u": "ㅕ", "i": "ㅑ", "o": "ㅐ", "p": "ㅔ",
              "a": "ㅁ", "s": "ㄴ", "d": "ㅇ", "f": "ㄹ", "g": "ㅎ", "h": "ㅗ", "j": "ㅓ", "k": "ㅏ", "l": "ㅣ",
              "z": "ㅋ", "x": "ㅌ", "c": "ㅊ", "v": "ㅍ", "b": "ㅠ", "n": "ㅜ", "m": "ㅡ",
              "Q": "ㅃ", "W": "ㅉ", "E": "ㄸ", "R": "ㄲ", "T": "ㅆ", "O": "ㅒ", "P": "ㅖ"}   # with Shift
_DUBEOLSIK_KEYS = {jamo: key for key, jamo in _DUBEOLSIK.items()}
_MEDIAL_JAMO = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_VOWEL_PAIRS = {("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ", ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ",
                ("ㅜ", "ㅣ"): "ㅟ", ("ㅡ", "ㅣ"): "ㅢ"}
_FINAL_PAIRS = {parts: final for final, parts in _SPLIT.items()}
_JAMO_PARTS = {**{v: k for k, v in _VOWEL_PAIRS.items()}, **_SPLIT}   # compound jamo -> the keys' jamo
LAYOUT_MIN_SYLLABLES = 2   # fewer: short English words (go 해, dog 앻) would be read as Hangul


def _compose_jamo(jamo: str) -> tuple[str, int, int]:
    """Jamo as the 2-set input method composes them: (text, syllables, loose jamo left over)."""
    out: list[str] = []
    syllables = loose = 0
    ini = med = fin = ""

    def flush() -> None:
        nonlocal ini, med, fin, syllables, loose
        if ini and med:
            out.append(chr(0xAC00 + (_INITIALS.index(ini) * 21 + _MEDIAL_JAMO.index(med)) * 28 + _FINALS.index(fin)))
            syllables += 1
        elif ini or med:
            out.append(ini or med)
            loose += 1
        ini = med = fin = ""

    for j in jamo:
        if j in _MEDIAL_JAMO:
            if ini and med and fin:              # the final starts the next syllable (the second of a pair)
                keep, moved = _SPLIT.get(fin, ("", fin))
                fin = keep
                flush()
                ini, med = moved, j
            elif med and not fin and (med, j) in _VOWEL_PAIRS:
                med = _VOWEL_PAIRS[(med, j)]
            elif ini and not med:
                med = j
            else:
                flush()
                med = j
        elif j in _INITIALS:
            if ini and med and not fin and j in _FINALS and j not in "ㄸㅃㅉ":
                fin = j
            elif ini and med and fin and (fin, j) in _FINAL_PAIRS:
                fin = _FINAL_PAIRS[(fin, j)]
            else:
                flush()
                ini = j
        else:
            flush()
            out.append(j)
    flush()
    return "".join(out), syllables, loose


def keys_to_hangul(text: str) -> str | None:
    """Letters typed on the 2-set keyboard in English mode as the Hangul meant (tkghl 사회), word by
    word: a word is read so when it composes into LAYOUT_MIN_SYLLABLES or more whole syllables
    with no jamo left over (English and romaji mostly do not; dbfndbfl OVA 유루유리 OVA); None when
    no word is. A word in capitals is read as Caps Lock (no Shift)."""
    out, meant = [], False
    for word in re.split(r"(\s+)", str(text or "")):
        if not any(c.isascii() and c.isalpha() for c in word):
            out.append(word)
            continue
        keys = word.lower() if word.upper() == word else word
        hangul, syllables, loose = _compose_jamo("".join(
            _DUBEOLSIK.get(c, _DUBEOLSIK.get(c.lower(), c)) if c.isascii() else c for c in keys))
        ok = syllables >= LAYOUT_MIN_SYLLABLES and not loose
        out.append(hangul if ok else word)
        meant = meant or ok
    return "".join(out) if meant else None


def hangul_to_keys(text: str) -> str | None:
    """Hangul typed on the 2-set keyboard in Korean mode as the letters meant (무ㅐㅜㄷ anone): None
    unless it has loose jamo (a Korean word typed as meant has none)."""
    text = str(text or "")
    if not any(0x3131 <= ord(c) <= 0x318E for c in text):
        return None
    jamo = []
    for c in text:
        code = ord(c) - 0xAC00
        parts = [_INITIALS[code // 588], _MEDIAL_JAMO[(code % 588) // 28], _FINALS[code % 28]] if 0 <= code < 11172 else [c]
        for part in parts:
            jamo.extend(_JAMO_PARTS.get(part, (part,)))
    return "".join(_DUBEOLSIK_KEYS.get(j, j) for j in jamo if j)


# ---- Japanese (Sudachi) ----

_tokenizer = None
_tokenizer_lock = threading.RLock()   # the loader is also called with it held
_PARTICLES = {"は": "wa", "へ": "e", "を": "o"}


def _japanese_tokenizer():
    """SudachiDict core, loaded on first use (memory-mapped: about 0.15 s)."""
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            from sudachipy import Dictionary, SplitMode
            _tokenizer = Dictionary(dict="core").tokenizer(mode=SplitMode.C)
        return _tokenizer


def _is_kana(ch: str) -> bool:
    code = ord(ch)
    return 0x3041 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF


def _is_kanji(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF or ch in "々〆ヶ"


def _is_hangul(ch: str) -> bool:
    code = ord(ch)
    return 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F


def _script(ch: str) -> str:
    if _is_kana(ch) or _is_kanji(ch):
        return "ja"
    if _is_hangul(ch):
        return "ko"
    return "other"


def _read_japanese(text: str) -> str:
    words = []
    with _tokenizer_lock:   # a tokenizer is not for several threads at once
        tokens = [(m.surface(), m.reading_form(), m.part_of_speech()[0])
                  for m in _japanese_tokenizer().tokenize(text)]
    for surface, reading, pos in tokens:
        if pos == "助詞" and surface in _PARTICLES:
            words.append(_PARTICLES[surface])
        elif reading and all(_is_kana(c) for c in reading):
            words.append(kana_to_romaji(reading))
        else:   # unknown to the dictionary: kana by the table, the rest (hanzi: pinyin) by anyascii
            words.append("".join(kana_to_romaji(c) if _is_kana(c) else anyascii(c).lower() for c in surface))
    return " ".join(w for w in words if w.strip())


# ---- Chinese ----

_chinese_lock = threading.Lock()
_taiwan: dict[str, tuple[str, str]] | None = None


def _toneless(syllable: str) -> str:
    s = unicodedata.normalize("NFD", syllable)
    return "".join(c for c in s if not unicodedata.combining(c)).replace("ü", "u")


def _taiwan_readings() -> dict[str, tuple[str, str]]:
    """Han character -> (mainland, Taiwan) Mandarin, where they differ (Unihan kMandarin)."""
    global _taiwan
    if _taiwan is None:
        from core.resource_paths import resource_path
        table = {}
        for line in resource_path("assets/readings/unihan_taiwan_mandarin.tsv").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                ch, cn, tw = line.split("\t")
                table[ch] = (cn, tw)
        _taiwan = table
    return _taiwan


def _read_mandarin(text: str, *, taiwan: bool) -> str:
    from pypinyin import Style, lazy_pinyin
    syllables = lazy_pinyin(text, style=Style.TONE, errors=lambda chars: [anyascii(c).lower() for c in chars])
    out = []
    for ch, syllable in zip(text, syllables) if len(syllables) == len(text) else ():
        plain = _toneless(syllable)
        if taiwan and ch in _taiwan_readings() and _taiwan_readings()[ch][0] == plain:
            plain = _taiwan_readings()[ch][1]
        out.append(plain)
    if len(syllables) != len(text):   # pypinyin kept a non-Han run together
        out = [_toneless(x) for x in syllables]
    return " ".join(x for x in out if x.strip())


def _read_cantonese(text: str) -> str:
    import pycantonese

    def jyutping(run: str) -> list:
        with _chinese_lock:   # the first call builds its tables
            return pycantonese.characters_to_jyutping(run)

    def spelled(chars: str, reading: str | None) -> str:
        return re.sub(r"\d", "", reading) if reading else " ".join(anyascii(c).lower() for c in chars)

    variants = _unihan_table("unihan_cantonese_variants")
    out = []
    for chars, reading in jyutping(text):
        if not reading and any(c in variants for c in chars):
            # forms pycantonese lacks (会, 学: Japanese new / simplified): read as their Unihan variants
            out.extend(spelled(c, r) for c, r in jyutping("".join(variants.get(c, c) for c in chars)))
        else:
            out.append(spelled(chars, reading))
    return " ".join(x for x in out if x.strip())


_tables: dict[str, dict[str, str]] = {}


def _unihan_table(name: str) -> dict[str, str]:
    """Han character -> reading, from assets/readings/<name>.tsv (Unihan)."""
    if name not in _tables:
        from core.resource_paths import resource_path
        text = resource_path(f"assets/readings/{name}.tsv").read_text(encoding="utf-8")
        _tables[name] = dict(line.split("\t")[:2] for line in text.splitlines() if line and not line.startswith("#"))
    return _tables[name]


_Y_VOWELS = (2, 3, 6, 7, 12, 17, 20)   # ㅑ ㅒ ㅕ ㅖ ㅛ ㅠ ㅣ


def _sino_korean_word(chars: str, table: dict[str, str]) -> str:
    """Han characters (each in ``table``: one Hangul syllable) as the Hangul of one Sino-Korean
    word, spelled as Korean spells it: the initial sound law on its first syllable (李 이, 女 여,
    樂 낙), 렬 / 률 as 열 / 율 after a vowel or ㄴ (比率 비율, 旋律 선율), 不 as 부 before ㄷ / ㅈ
    (不足 부족, 不適合 부적합)."""
    syl = [[code // 588, (code % 588) // 28, code % 28] for code in (ord(table[ch]) - 0xAC00 for ch in chars)]
    for k in range(len(syl) - 1):
        if chars[k] == "不" and syl[k + 1][0] in (3, 12):       # ㄷ ㅈ
            syl[k][2] = 0
    for k in range(1, len(syl)):
        if syl[k][0] == 5 and syl[k][1] in (6, 17) and syl[k][2] == 8 and syl[k - 1][2] in (0, 4):   # 렬 률
            syl[k][0] = 11
    ini, med, _fin = syl[0]
    if ini == 5:                                   # ㄹ
        syl[0][0] = 11 if med in _Y_VOWELS else 2
    elif ini == 2 and med in _Y_VOWELS:            # ㄴ
        syl[0][0] = 11
    return "".join(chr(0xAC00 + ini * 588 + med * 28 + fin) for ini, med, fin in syl)


def _read_korean_hanja(text: str) -> str:
    table = _unihan_table("unihan_korean")
    out, word = [], ""
    for ch in text:
        if ch in table:
            word += ch
            continue
        if word:
            out.append(korean_to_roman(_sino_korean_word(word, table)))
            word = ""
        out.append(_read_mandarin(ch, taiwan=False))
    if word:
        out.append(korean_to_roman(_sino_korean_word(word, table)))
    return " ".join(x for x in out if x.strip())


# Han mixed with other characters is found by its Korean Hanja reading too: beside Latin letters or
# Hangul all of it (Han without kana is read in every language), beside kana -- read in Japanese
# with it -- a run of several Han characters, mostly a Sino-Japanese word (異世界生活, 社会不適合者);
# a shorter one there is mostly a native word read with its kana (君, 夜, 始める), left in Japanese.
# How many (Settings > Search; 0: none so read):
HAN_RUN_MIN = 3
HAN_RUN_LEAST = 2


@lru_cache(maxsize=16)
def _han_run(least: int) -> re.Pattern:
    return re.compile(r"[㐀-䶿一-鿿豈-﫿]{%d,}" % max(HAN_RUN_LEAST, least))


def _read_japanese_korean_hanja(run: str, han_run: int) -> str:
    """Japanese text with its runs of ``han_run`` or more Han characters in Korean Hanja, the
    rest in Japanese (Re:ゼロから始める異世界生活 zero kara hajimeru isegyesaenghwal)."""
    out, pos = [], 0
    for m in _han_run(han_run).finditer(run):
        if m.start() > pos:
            out.append(_read_japanese(run[pos:m.start()]))
        out.append(_read_korean_hanja(m.group(0)))
        pos = m.end()
    if pos < len(run):
        out.append(_read_japanese(run[pos:]))
    return " ".join(x for x in out if x.strip())


def _read_vietnamese(text: str) -> str:
    table = _unihan_table("unihan_vietnamese")
    return " ".join(anyascii(table[ch]).lower() if ch in table else _read_mandarin(ch, taiwan=False)
                    for ch in text if ch.strip())


def warm_up(progress: Callable[[float, str], None] | None = None) -> None:
    """Load the readers (Sudachi, pypinyin, pycantonese, the Unihan tables: about 2 s; the app does
    it while starting), so the first reading or search does not wait; ``progress(fraction done,
    reader)`` before each (by its usual share of the time)."""
    steps = [("Japanese (SudachiDict)", 0.04, _japanese_tokenizer),
             ("Mandarin (pypinyin)", 0.1, lambda: _read_mandarin("中", taiwan=True)),
             ("Cantonese (pycantonese)", 0.82, lambda: _read_cantonese("中")),
             ("Korean Hanja (Unihan)", 0.02, lambda: _unihan_table("unihan_korean")),
             ("Sino-Vietnamese (Unihan)", 0.02, lambda: _unihan_table("unihan_vietnamese"))]
    done = 0.0
    for name, share, load in steps:
        if progress is not None:
            progress(done, name)
        load()
        done += share


def _romanize(text: str, *, han: str, foreign_spelling: bool, han_run: int = 0) -> str:
    """``text`` in romaji, its Han characters read in ``han`` (one of HAN_LANGUAGES): kana with them
    reads them in Japanese -- in Korean Hanja, a text with kana has its runs of ``han_run`` or
    more Han characters read so and the rest in Japanese."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    japanese = any(_is_kana(c) for c in text)
    runs: list[list[str]] = []
    for ch in text:
        script = _script(ch)
        if runs and runs[-1][0] == script:
            runs[-1][1] += ch
        else:
            runs.append([script, ch])
    parts = []
    for script, run in runs:
        if script == "ja" and han == "ko-hanja" and japanese and han_run:
            parts.append(_read_japanese_korean_hanja(run, han_run))
        elif script == "ja" and (han == "ja" or any(_is_kana(c) for c in run)):
            parts.append(_read_japanese(run))
        elif script == "ja" and han == "yue":
            parts.append(_read_cantonese(run))
        elif script == "ja" and han == "ko-hanja":
            parts.append(_read_korean_hanja(run))
        elif script == "ja" and han == "vi":
            parts.append(_read_vietnamese(run))
        elif script == "ja":
            parts.append(_read_mandarin(run, taiwan=han == "zh-tw"))
        elif script == "ko":
            parts.append(korean_to_roman(run, foreign_spelling=foreign_spelling))
        else:
            parts.append(anyascii(run))
    # runs of different scripts meet without a space: keep the words apart
    out = ""
    for part in parts:
        if out and part and not out[-1].isspace() and not part[0].isspace():
            out += " "
        out += part
    return re.sub(r"\s+", " ", out).strip()


def han_languages(selection: Iterable[str] | None = None) -> list[str]:
    """The HAN_LANGUAGES in ``selection`` (the search settings), in slot order; all for None or
    none known."""
    chosen = set(selection) if selection is not None else set(HAN_LANGUAGES)
    return [lang for lang in HAN_LANGUAGES if lang in chosen] or list(HAN_LANGUAGES)


def _languages(text: str, han: Iterable[str] | None = None, han_run: int = 0) -> list[str]:
    """The readings a text gets: one per language ``han`` (han_languages) for Han without kana;
    Japanese for kana, and Korean Hanja too (when in ``han``) for its runs of ``han_run`` (0:
    none) or more Han characters; else one."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    if any(_is_kanji(c) for c in text) and not any(_is_kana(c) for c in text):
        return han_languages(han)
    if any(_is_kana(c) for c in text):
        korean = han_run and "ko-hanja" in han_languages(han) and _han_run(han_run).search(text)
        return ["ja"] + (["ko-hanja"] if korean else [])
    return ["ko"] if any(_is_hangul(c) for c in text) else ["text"]


def readings(text: str, languages: Iterable[str] | None = None, *,
             han_run: int = HAN_RUN_MIN) -> list[tuple[str, str]]:
    """Layer 1: the readings of ``text`` as (reading, source) -- source a SOURCES key -- each once,
    at most READING_SLOTS; none for an empty text. Han without kana is read in ``languages``
    (the stored languages of the search settings; all by default); beside kana, its runs of
    ``han_run`` or more Han characters in Korean Hanja too (0: not)."""
    out: list[tuple[str, str]] = []
    for lang in _languages(text, languages, han_run):
        reading = _romanize(text, han=lang if lang in HAN_LANGUAGES else "ja", foreign_spelling=False,
                            han_run=han_run)
        if reading and reading not in [r for r, _src in out]:
            out.append((reading, lang))
    return out[:READING_SLOTS]


def read(text: str) -> str:
    """The first reading of ``text`` (Japanese for Han), "" for none."""
    found = readings(text)
    return found[0][0] if found else ""


# ---- normalization ----

_NORMALIZE_STEPS = [(re.compile(pattern), repl) for pattern, repl in [
    (r"tch", "ch"), (r"(?:ch|sh|zh|ts|dz|j|z|c(?=[eiy]))", "s"),   # s z j sh ch zh ts as one
    (r"c|q|g|x", "k"), (r"d", "t"), (r"[fbpv]", "h"), (r"l", "r"),
    (r"(?<=s)y", ""),                                            # sha = sya = sa
    (r"m(?=h)", "n"),                                            # shimbun = shinbun
    (r"ou|oo", "o"), (r"uu", "u"), (r"ei|ee", "e"), (r"aa", "a"), (r"ii", "i"),
    (r"wo", "o"),
    (r"([a-z])\1+", r"\1"),                                      # doubled letters: っ, long vowels
]]
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def fine_key(romaji: str) -> str:
    """The romaji lower-cased, letters and digits only."""
    return _NOT_ALNUM.sub("", anyascii(unicodedata.normalize("NFKC", str(romaji or ""))).lower())


def normalize(romaji: str) -> str:
    """Layer 2: the search form of romaji, in which spellings of one sound agree (module notes)."""
    s = fine_key(romaji)
    for pattern, repl in _NORMALIZE_STEPS:
        s = pattern.sub(repl, s)
    return s


@lru_cache(maxsize=1 << 18)
def keys(reading: str) -> tuple[str, str]:
    """(fine, normalized) keys of a stored reading (kept: every library reload asks for the same again)."""
    return fine_key(reading), normalize(reading)


def query_keys(query: str, languages: Iterable[str] | None = None, *, hangul_korean: bool = True,
               hangul_foreign: bool = True, keyboard: bool = True,
               han_run: int = HAN_RUN_MIN) -> list[tuple[str, str]]:
    """(fine, normalized) keys of a search query, one pair per reading, read like a title (its Han
    in ``languages``, all by default; ``han_run`` as for ``readings``); its Hangul, for the
    normalized key, as Korean (``hangul_korean``: a Korean title, a Sino-Korean reading) and with
    Korean's spelling of foreign words (``hangul_foreign``: a Japanese title). With ``keyboard``, a
    query typed on the 2-set keyboard without the 한/영 key is also read as what was meant
    (``keys_to_hangul``, ``hangul_to_keys``)."""
    spellings: tuple[bool, ...] = (False,)
    if any(_is_hangul(c) for c in str(query or "")):
        spellings = tuple(f for f, on in ((False, hangul_korean), (True, hangul_foreign)) if on) or (False,)
    out = []
    for lang in _languages(query, languages, han_run):
        han = lang if lang in HAN_LANGUAGES else "ja"
        fine = fine_key(_romanize(query, han=han, foreign_spelling=False, han_run=han_run))
        for foreign in spellings:
            pair = (fine, normalize(_romanize(query, han=han, foreign_spelling=foreign, han_run=han_run)))
            if pair not in out:
                out.append(pair)
    if keyboard:
        for meant in (keys_to_hangul(query), hangul_to_keys(query)):
            if meant:
                out += [pair for pair in query_keys(meant, languages, hangul_korean=hangul_korean,
                                                    hangul_foreign=hangul_foreign, keyboard=False,
                                                    han_run=han_run) if pair not in out]
    return out


# ---- matching ----

def _similar_cutoff(n: int) -> float:
    """RapidFuzz partial ratio a query of ``n`` normalized letters needs: one sound off per
    SIMILAR_EVERY letters (a substitution costs one of n)."""
    return 100.0 * (1.0 - max(1, n // SIMILAR_EVERY) / n)


def matches(query: list[tuple[str, str]], reading_keys: tuple[str, str]) -> bool:
    """Whether a query (``query_keys``) is found in one reading (its ``keys``)."""
    return bool(find(query, [reading_keys]))


def find(query: list[tuple[str, str]], readings_keys: list[tuple[str, str]], *, similar: bool = True) -> set[int]:
    """Layer 3: the indices of the readings (``keys``) a query (``query_keys``: any of its readings)
    is found in: its fine key contained, its normalized key contained (EXACT_MIN letters on), or --
    with ``similar`` (the fuzzy search) -- nearly contained (SIMILAR_MIN letters on: a similar
    pronunciation)."""
    found: set[int] = set()
    norms = [norm for _fine, norm in readings_keys]
    for q_fine, q_norm in query:
        found.update(i for i, (fine, norm) in enumerate(readings_keys)
                     if (q_fine and q_fine in fine) or (len(q_norm) >= EXACT_MIN and q_norm in norm))
        if similar and len(q_norm) >= SIMILAR_MIN:
            # the query within a reading: partial_ratio would find a shorter reading within the
            # query instead (zaq in sahoebujeokapja), so the readings shorter than the query by
            # more than the sounds it may be off are left out
            least = len(q_norm) - max(1, len(q_norm) // SIMILAR_EVERY)
            hits = process.extract(q_norm, {i: norm for i, norm in enumerate(norms) if len(norm) >= least},
                                   scorer=fuzz.partial_ratio, score_cutoff=_similar_cutoff(len(q_norm)), limit=None)
            found.update(index for _choice, _score, index in hits)
    return found
