"""A title's version notes, the words saying which version of a song a track is, as Japanese, English, Korean and
Chinese titles write them: the instrumental's (an off vocal, a karaoke, a Korean MR / 반주, a Chinese 伴奏 / 纯音乐) and
the others (a TV size, a remix, a version: ver., バージョン, 버전, 版). The tie-up lookup leaves a note out of the song's
name (core.tieups.web.song_title); the Mix Candidates leave the instrumentals out (core.search.mixability)."""
from __future__ import annotations

import re

# the instrumental's words; "MR" only in capitals and not "Mr." / "MR." (a name: Mr.Children)
INSTRUMENTAL_WORDS = (r"(?<![A-Za-z])(?:instrumentals?|inst|off[\s_-]?vocal|karaoke)(?![A-Za-z])|"
                      r"(?-i:(?<![A-Za-z])MR(?![A-Za-z.]))|"
                      r"オフボーカル|カラオケ|インスト|반주|인스트|伴奏|纯音乐|純音樂")
# a version's words besides
NOTE_WORDS = (rf"{INSTRUMENTAL_WORDS}|(?<![A-Za-z])(?:TV|short|full|size|ver|version|remix|mix|edit|arrange|cover)"
              r"(?![A-Za-z])|テレビ|バージョン|ば[ー～〜~]じょん|사이즈|버전|版")
INSTRUMENTAL = re.compile(INSTRUMENTAL_WORDS, re.IGNORECASE)
NOTE = re.compile(NOTE_WORDS, re.IGNORECASE)
