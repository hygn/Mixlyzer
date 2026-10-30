import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from types import SimpleNamespace
from unittest import mock

from core.tieups import domain as tieup

from core.tieups import web as tw
from core.library import jobs, tieups
from core.library.db import LibraryDB
from core.library.records import TrackRow

SONG_PAGE = """{{Infobox Single
| Name = 青い鳥
}}
『'''青い鳥'''』は、[[結束バンド]]の楽曲。
[[テレビアニメ]]「[[ぼっち・ざ・ろっく! (アニメ)|ぼっち・ざ・ろっく!]]」第2期オープニングテーマ。<ref>{{Cite web|title=x}}</ref>
同作の第3話 - 第12話で[[エンディングテーマ]]としても使われたほか、劇場版『[[ぼっち・ざ・ろっく! (アニメ)|劇場総集編]]』の主題歌。
[[ゲーム]]「[[ボッチモバイル]]」の挿入歌でもある。
"""
ALBUM_PAGE = """# '''青い鳥'''
#* テレビアニメ「[[別の作品]]」エンディングテーマ。
# '''別の曲'''
#* テレビアニメ「[[関係ない作品]]」オープニングテーマ。
"""


class Fake:
    """Wikipedia / Wikidata answers for a few pages."""

    def __init__(self, own: bool = True):
        self.own = own
        self.calls = []

    def __call__(self, url: str, params: dict) -> dict:
        self.calls.append(params)
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "結束バンド (アルバム)"}]}}
        if params.get("prop") == "pageprops" and "ppprop" in params:
            pages = [{"title": "青い鳥 (曲)", "pageid": 1}] if self.own else [{"title": "青い鳥", "missing": True}]
            return {"query": {"pages": pages}}
        if params.get("action") == "parse":
            return {"parse": {"wikitext": SONG_PAGE if params["page"] == "青い鳥 (曲)" else ALBUM_PAGE}}
        if params.get("prop") == "categories":
            cats = {"ぼっち・ざ・ろっく! (アニメ)": ["アニメ作品 ほ"], "ボッチモバイル": ["2020年のコンピュータゲーム"],
                    "別の作品": ["アニメ作品 へ"]}.get(params["titles"])
            return {"query": {"pages": [{"title": params["titles"], "categories": [
                {"title": f"Category:{c}"} for c in cats]} if cats else {"title": params["titles"], "missing": True}]}}
        if params.get("prop") == "pageprops":
            item = {"ぼっち・ざ・ろっく! (アニメ)": "Q1", "ボッチモバイル": "Q2"}.get(params["titles"])
            return {"query": {"pages": [{"pageprops": {"wikibase_item": item} if item else {}}]}}
        if params.get("action") == "wbgetentities":
            entities = {
                "Q1": {"labels": {"ja": {"value": "ぼっち・ざ・ろっく!"}, "ko": {"value": "봇치 더 록!"},
                                  "en": {"value": "Bocchi the Rock!"}, "fr": {"value": "Bocchi (fr)"}},
                       "aliases": {"ja": [{"value": "ぼざろ"}]},
                       "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q63952888"}}}}]}},
                "Q2": {"labels": {"ja": {"value": "ボッチモバイル"}}, "claims": {}},
            }
            return {"entities": {params["ids"]: entities[params["ids"]]}}
        raise AssertionError(params)


class TieupWebTests(unittest.TestCase):
    def test_mentions_as_articles_write_them(self):
        got = {(w, r, s, d, c) for w, _p, r, s, d, c in tw.mentions(SONG_PAGE, "青い鳥", whole=True)}
        self.assertIn(("ぼっち・ざ・ろっく!", "opening", 0, "", "anime"), got)
        self.assertIn(("ボッチモバイル", "insert", 0, "", "game"), got)
        self.assertIn(("劇場総集編", "theme", 0, "", "movie"), got)
        # on another page only the lines naming the song are read
        album = {(w, r) for w, _p, r, _s, _d, _c in tw.mentions(ALBUM_PAGE, "青い鳥", whole=False)}
        self.assertEqual(album, {("別の作品", "ending")})

    def test_no_tie_up_where_the_page_states_none(self):
        # 粛聖!! ロリ神レクイエム☆ (an original song) as its page has it: a chart's TOP100 has no OP, an アニメーション no
        # アニメ, its covers' section is the covers' (a game's character's cover, in the game)
        page = ("同曲のミュージック・ビデオ自体も日本国外で認知度を高めているが、メンフィス・カルトの楽曲「9mm」の"
                "リズムに合わせて踊るように編集された9歳のしぐれういの二次創作動画も話題となった。公式SNSアカウントで、"
                "同曲のミュージック・ビデオに収録されている踊る9歳のしぐれういのアニメーションと「9mm」を組み合わせた"
                "動画を投稿。\n"
                "|scope=row |[[TBSテレビ|TBS]]の『[[COUNT DOWN TV]]』 - 今週のTOP100\n"
                "| style=\"text-align:center\"|27\n"
                "== カバー ==\n"
                "* 筆島しぐれ（[[吉岡麻耶]]） - 音楽ゲーム「[[ワールドダイスター 夢のステラリウム]]」に収録"
                "（2024年6月29日追加）。\n"
                "== 脚注 ==\n")
        self.assertEqual(tw.mentions(page, "粛聖!! ロリ神レクイエム☆", whole=True), [])

    def test_a_singles_other_songs_tie_ups_are_theirs(self):
        # S・M・L☆'s own page: its coupling song ミライボタン's tie-up (a game's ending) is not its own
        page = ("「S・M・L☆」（エス・エム・エル）は、アフィリア・サーガの楽曲。\n"
                "== 概要 ==\n"
                "TVアニメ『俺の脳内選択肢が、学園ラブコメを全力で邪魔している』のOPで使用された楽曲。\n"
                "== 収録曲 ==\n"
                "=== DVD付盤 ===\n"
                "# S・M・L☆ [4:50]\n"
                "#: 作詞 - 桃井はるこ / 作曲 - 濱田幹浩 / 編曲 - MACARONI☆\n"
                "#: テレビアニメ「俺の脳内選択肢が、学園ラブコメを全力で邪魔している」オープニングテーマ\n"
                "# ミライボタン [3:44]\n"
                "#: 作詞 - 林達也 / 作曲 - 濱田幹浩 / 編曲 - 河合英嗣\n"
                "#: PS Vita用ソフト「超次次元ゲイム ネプテューヌRe;Birth1」エンディングテーマ\n"
                "# S・M・L☆(off Vocal)\n"
                "# ミライボタン(off Vocal)\n")
        got = {(w, r) for w, _p, r, _s, _d, _c in tw.mentions(page, "S・M・L☆", whole=True)}
        self.assertEqual(got, {("俺の脳内選択肢が、学園ラブコメを全力で邪魔している", "opening")})
        # a double A-side single's section in Poppin'Party's discography: the tie-up is 「最高！」's, not 二重の虹's
        section = ("=== 二重の虹（ダブル レインボウ）/最高（さあ行こう）！ ===\n"
                   "10thシングル。2018年7月11日に発売された。\n"
                   "表題曲のひとつである「最高！」はテレビアニメ『フューチャーカード 神バディファイト』オープニングテーマに"
                   "起用されており、メンバーの西本りみが同番組の主人公・末門友牙の弟である末門晴役で出演している。\n")
        got = {(w, r) for w, _p, r, _s, _d, _c in tw.mentions(section, "二重の虹(ダブル レインボウ)", whole=False,
                                                                page="Poppin'Partyのディスコグラフィ")}
        # the page's own work listed, no role (the search then drops it: Poppin'Party's page is a group's)
        self.assertEqual(got, {("Poppin'Party", "unspecified")})

    @unittest.expectedFailure
    def test_a_franchise_pages_anime_section(self):
        # Re:ステージ!'s page lists 憧れFuture Sign under its TV anime's theme songs: the work is the anime
        # (Re:ステージ! ドリームデイズ♪, named in the section's first line), not the franchise its page is
        page = ("== テレビアニメ ==\n"
                "2019年7月より9月まで『Re:ステージ！ ドリームデイズ♪』（リステージ ドリームデイズ）のタイトルで、"
                "TOKYO MXほかにて放送された。\n"
                "=== 主題歌 ===\n"
                "; エンディングテーマ\n"
                ":; 「憧れFuture Sign」（第1話 - 第7話、第9話 - 第11話）\n"
                ":: 作詞 - やしきん / 作曲・編曲 - 伊藤翼 / 歌 - KiRaRe\n")
        works = {w for w, *_rest in tw.mentions(page, "憧れFuture Sign", whole=False, page="Re:ステージ!")}
        self.assertEqual(works, {"Re:ステージ！ ドリームデイズ♪"})

    def test_a_titles_own_words_on_its_tie_up(self):
        # left out of the names searched; the work they name is the sure one (幼女社長R, not 幼女社長)
        self.assertEqual(tw.song_title("オ・ヒ・メ・サ・マ！(アニメ幼女社長R EDテーマ)")[0], "オ・ヒ・メ・サ・マ！")
        self.assertEqual(tw.title_note("オ・ヒ・メ・サ・マ！(アニメ幼女社長R EDテーマ)"), "(アニメ幼女社長R EDテーマ)")
        self.assertEqual(tw.song_title("ふゆびより (포근한 겨울날) - TV애니메이션 『유루캠프』 ED테마")[0], "ふゆびより")
        # a Korean translation after the title is left out; a Korean title's own parenthesis kept
        self.assertEqual(tw.song_title("ツナグ、ソラモヨウ (이어진 하늘 모양)")[0], "ツナグ、ソラモヨウ")
        self.assertEqual(tw.song_title("다시 만난 세계 (소녀시대)")[0], "다시 만난 세계 (소녀시대)")

    def test_the_own_page_naming_its_work_without_a_role(self):
        page = ("『Yes! BanG_Dream!』は、Poppin'Partyのデビューシングルである。\n"
                "このユニットは、ブシロードのメディアミックスプロジェクト『[[BanG_Dream!]]』にて誕生しており、…\n")
        got = {(w, p, r, c) for w, p, r, _s, _d, c in tw.mentions(page, "Yes! BanG_Dream!", whole=True)}
        self.assertEqual(got, {("BanG_Dream!", "BanG_Dream!", tieup.UNSPECIFIED, "media_mix")})
        # elsewhere a work named by its medium is any page's prose
        self.assertEqual(tw.mentions("* 「Yes! BanG_Dream!」 テレビアニメ『[[別作品]]』に出演\n", "Yes! BanG_Dream!",
                                     False), [])

    def test_the_names_a_song_is_looked_up_by(self):
        self.assertEqual(tw.song_title("影響力∞アイドル feat.ハコニワリリィ"), ("影響力∞アイドル", ["ハコニワリリィ"]))
        self.assertEqual(tw.song_title("Song (feat. Someone)"), ("Song", ["Someone"]))
        self.assertEqual(tw.song_title("紅蓮華 (TV size)"), ("紅蓮華", []))
        self.assertEqual(tw.song_title("Defeat the Beast"), ("Defeat the Beast", []))   # not a feat.
        songs, artists, searches = tw.queries("恋の歌 〜Ver.2〜 feat.Singer", "MARUMOCHI from HoneyWorks",
                                              "TVアニメ「X」OP")
        self.assertEqual(songs, ["恋の歌", "恋の歌 〜Ver.2〜 feat.Singer"])     # a version note between wave dashes
        self.assertEqual(artists, ["MARUMOCHI", "HoneyWorks", "Singer"])
        self.assertEqual(searches, ['"恋の歌"', '"恋の歌 〜Ver.2〜 feat.Singer"', '"恋の歌" MARUMOCHI',
                                    '"恋の歌" HoneyWorks', '"TVアニメ「X」OP"'])
        # version notes in any place of the parenthesis, a title's own parenthesis kept
        self.assertEqual(tw.song_title("よしなしOKI☆DOKI(さくひまば～じょん)")[0], "よしなしOKI☆DOKI")
        self.assertEqual(tw.song_title("100%ちゅ～学生 (Kijibato remix)")[0], "100%ちゅ～学生")
        self.assertEqual(tw.song_title("Love Song (Forever)")[0], "Love Song (Forever)")
        self.assertNotIn("Song (Hoge", tw.queries("Song (Hoge～Fuga)")[0])     # not cut inside it
        # between dashes or wave dashes; Korean and Chinese notes
        self.assertEqual(tw.song_title("紅蓮華 -TV size-")[0], "紅蓮華")
        self.assertEqual(tw.song_title("残酷な天使のテーゼ～TV size～")[0], "残酷な天使のテーゼ")
        self.assertEqual(tw.song_title("Butter-Fly -instrumental-")[0], "Butter-Fly")
        self.assertEqual(tw.song_title("めざせポケモンマスター -20th Anniversary-")[0],
                         "めざせポケモンマスター -20th Anniversary-")      # a subtitle, no version's word
        self.assertEqual(tw.song_title("Butter-Fly - Live")[0], "Butter-Fly - Live")
        for title, song in (("君の知らない物語 (TV 사이즈)", "君の知らない物語"), ("다시 만난 세계 (MR)", "다시 만난 세계"),
                            ("다시 만난 세계 [반주]", "다시 만난 세계"), ("孤勇者 (伴奏)", "孤勇者"),
                            ("孤勇者 (纯音乐)", "孤勇者"), ("紅蓮華 (TV版)", "紅蓮華")):
            self.assertEqual(tw.song_title(title)[0], song, title)
        # "A meets B", a credit after a bracketed name; a colon inside a name kept
        self.assertEqual(tw.artist_names("HoneyWorks meets TrySail"), ["HoneyWorks", "TrySail"])
        self.assertEqual(tw.artist_names("SawanoHiroyuki[nZk]:Aimer"), ["SawanoHiroyuki", "nZk", "Aimer"])
        self.assertEqual(tw.artist_names("Re:vale"), ["Re:vale"])
        # a character's credit: the unit, the voice actors, the characters
        self.assertEqual(tw.artist_names("さくひま*ひまさく[大室櫻子(CV:加藤英美里)&古谷向日葵(CV:三森すずこ)]"),
                         ["さくひま*ひまさく", "加藤英美里", "三森すずこ", "大室櫻子", "古谷向日葵"])
        self.assertEqual(tw.artist_names("あおい(CV.井口裕香)/ひなた(CV.阿澄佳奈)"),
                         ["井口裕香", "阿澄佳奈", "あおい", "ひなた"])
        self.assertEqual(tw.artist_names("Aqu3ra & 月見ヤチヨ(cv.早見沙織)"), ["Aqu3ra", "早見沙織", "月見ヤチヨ"])
        # a parenthesis without "CV": the voice actor or the members after the name before it
        self.assertEqual(tw.artist_names("シャロ(内田真礼)"), ["シャロ", "内田真礼"])
        self.assertEqual(tw.artist_names("ゆいかおり(小倉唯&石原夏織)"), ["ゆいかおり", "小倉唯", "石原夏織"])
        self.assertEqual(tw.artist_names("レンチンメリケンサック (Mono. + Rostel)"),
                         ["レンチンメリケンサック", "Mono.", "Rostel"])
        self.assertEqual(tw.artist_names("部(沼倉･赤崎)"), ["沼倉", "赤崎"])
        self.assertEqual(tw.artist_names("コーロ(シャミ子・リリス)/CV:小原好美・鬼頭明里"),
                         ["コーロ", "小原好美", "鬼頭明里", "シャミ子", "リリス"])
        self.assertEqual(tw.artist_names("Artist (Club remix)"), ["Artist"])      # a version note, no name

    def test_a_name_kept_whole(self):
        """The names an artist field holds checked on 48,932 names of real artists (one person or one group each):
        the rules keep a name's own separators, the shipped list (assets/artists/default_artist_names.tsv) the
        names they cannot tell from a credit; collaborations and lists still split."""
        tw.kept_names.cache_clear()
        # featuring is not "feat" + "uring"
        self.assertEqual(tw.artist_names("ROUND TABLE featuring Nino"), ["ROUND TABLE", "Nino"])
        self.assertEqual(tw.artist_names("FictionJunction featuring YUUKA"), ["FictionJunction", "YUUKA"])
        # a middle dot between two katakana words, or with a "the", is a name's; between single letters too
        self.assertEqual(tw.artist_names("アフィリア・サーガ"), ["アフィリア・サーガ"])
        self.assertEqual(tw.artist_names("アフター・ザ・レイン"), ["アフター・ザ・レイン"])
        self.assertEqual(tw.artist_names("リリー・ベイカー(CV：天城サリー)、草壁葵衣(CV：小山内怜央)"),
                         ["天城サリー", "小山内怜央", "リリー・ベイカー", "草壁葵衣"])
        self.assertEqual(tw.artist_names("桜木ひな子(CV:M・A・O)/夏川くいな(CV:富田美憂)"),
                         ["M・A・O", "富田美憂", "桜木ひな子", "夏川くいな"])
        self.assertEqual(tw.artist_names("22/7"), ["22/7"])
        # a list of names still split at its dots, a collaboration at its "&", "×", ","
        self.assertEqual(tw.artist_names("コーロまちカド(シャミ子・桃・リリス・ミカン)/CV:小原好美・鬼頭明里・高橋未奈美・高柳知葉"),
                         ["コーロまちカド", "小原好美", "鬼頭明里", "高橋未奈美", "高柳知葉", "シャミ子", "桃", "リリス", "ミカン"])
        self.assertEqual(tw.artist_names("KOTONOHOUSE & Neko Hacker"), ["KOTONOHOUSE", "Neko Hacker"])
        self.assertEqual(tw.artist_names("Neko Hacker, ななひら, mochari, をとは"),
                         ["Neko Hacker", "ななひら", "mochari", "をとは"])
        self.assertEqual(tw.artist_names("카스미×란×아야×유키나×코코로"), ["카스미", "란", "아야", "유키나", "코코로"])
        # the shipped list: a separator in the name, a name under 3 letters
        for name in ("Wake Up, Girls!", "Hello, Happy World!", "ハロー、ハッピーワールド!", "MYTH & ROID", "CROSS×KEY",
                     "ひみつ×戦士 ファントミラージュ!", "Can/goo", "(K)NoW_NAME", "μ's", "≠ME"):
            self.assertEqual(tw.artist_names(name), [name], name)
        self.assertEqual(tw.artist_names("MYTH&ROID"), ["MYTH&ROID"])          # its spellings, any case
        self.assertEqual(tw.artist_names("Run Girls, Run! & Wake Up, Girls!"), ["Run Girls, Run!", "Wake Up, Girls!"])
        self.assertEqual(tw.artist_names("μ's & Aqours"), ["μ's", "Aqours"])
        self.assertEqual(tw.artist_names("Wake Up, Girls! feat. Hello"), ["Wake Up, Girls!", "Hello"])

    def test_the_users_names_after_the_shipped_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            shipped, users = Path(tmp) / "default.tsv", Path(tmp) / "user.tsv"
            shipped.write_text("# a comment\nMYTH & ROID\twikidata Q24175566\n", encoding="utf-8")
            users.write_text("Daisy×Daisy\n\n", encoding="utf-8")
            with mock.patch.object(tw, "NAMES_FILE", str(shipped)), mock.patch.object(tw, "USER_NAMES_FILE", str(users)):
                tw.kept_names.cache_clear()
                try:
                    self.assertEqual(set(tw.kept_names()), {"MYTH & ROID", "Daisy×Daisy"})
                    self.assertEqual(tw.artist_names("Daisy×Daisy & MYTH & ROID"), ["Daisy×Daisy", "MYTH & ROID"])
                finally:
                    tw.kept_names.cache_clear()

    def test_the_song_found_in_a_page_as_the_search_finds_a_title(self):
        lines = ["「シュガーソング・ビターステップ」はテレビアニメ「[[作品]]」オープニングテーマ。", "関係ない行。"]
        self.assertEqual(tw.found_in("ウィッチクラフト", ["ウイッチクラフト"], similar=False), [0])   # its sounds
        self.assertEqual(tw.found_in("シュガーソングとビターステップ", lines), [0])        # a sound off
        self.assertEqual(tw.found_in("シュガーソングとビターステップ", lines, similar=False), [])
        got = {(w, r) for w, _p, r, *_ in tw.mentions("\n".join(lines), "シュガーソングとビターステップ", False)}
        self.assertEqual(got, {("作品", "opening")})          # the song's own spelling never taken for a work

    def test_a_work_inside_the_work_and_a_discography(self):
        page = ("== Bird ==\n=== 星の歌 ===\n2020年に発売された1stシングル。\n"
                "# 星の歌\n#: 作詞：誰か\n#:* 作中作『劇中のアニメ』主題歌。\n# 次の歌\n#* テレビアニメ「[[別作品]]」挿入歌。\n")
        got = {(w, p, r) for w, p, r, *_ in tw.mentions(page, "星の歌", False, "作品のディスコグラフィ")}
        self.assertEqual(got, {("作品", "作品", tieup.UNSPECIFIED)})        # the IP, no role; not the next song's
        table = ("{| class=\"wikitable\"\n|-\n| rowspan=\"2\"|2022年 || 想いのち晴れ/別の歌\n| 歌手\n| 「想いのち晴れ」\n"
                 "| テレビアニメ『[[山の作品]]』オープニングテーマ\n|-\n| 2023年 || 次の歌\n| 「次の歌」\n| 『別作品』挿入歌\n|}\n")
        got = {(w, r) for w, _p, r, *_ in tw.mentions(table, "想いのち晴れ", False, "歌手")}
        self.assertEqual(got, {("山の作品", "opening")})                   # its row's cells, not the next row's
        label = "; 「星降る海」{{R|x}}\n: 月見ヤチヨ（早見沙織）による劇中歌。\n; 「別の歌」\n: 主題歌。\n"
        got = {(w, r) for w, _p, r, *_ in tw.mentions(label, "星降る海", False, "作品 (アニメ)")}
        self.assertEqual(got, {("作品", "insert")})

    def test_a_work_page_listing_the_song_under_a_role(self):
        page = ("=== 主題歌 ===\n; 「影響力∞アイドル」\n: HoneyWorks によるオープニングテーマ。\n"
                "; 「星を巡る」\n: エンディングテーマ。\n== 挿入歌 ==\n* 「別の歌」\n")
        got = {(w, r) for w, _p, r, _s, _d, _c in tw.mentions(page, "影響力∞アイドル", False, "作品 (アニメ)")}
        self.assertEqual(got, {("作品", "opening")})
        got = {(w, r) for w, _p, r, _s, _d, _c in tw.mentions(page, "別の歌", False, "作品 (アニメ)")}
        self.assertEqual(got, {("作品", "insert")})                 # the heading it is listed under

    def test_the_songs_own_page_and_a_sure_one(self):
        api = tw.Wikimedia(Fake())
        cands = tw.search(api, "青い鳥", "結束バンド")
        sure = [c for c in cands if c.sure]
        self.assertEqual(len(sure), 1)
        c = sure[0]
        self.assertEqual((c.role, c.work.category), ("opening", "anime"))
        self.assertEqual(c.work.titles, [("ja-JP", "ぼっち・ざ・ろっく!"), ("ko-KR", "봇치 더 록!"), ("en-US", "Bocchi the Rock!")])
        self.assertEqual(c.work.aliases, [("synonym", "ja-JP", "ぼざろ")])   # French left out: no ll-RR for it here
        self.assertTrue(all(tieup.is_lang(lang) for lang, _t in c.work.titles))
        self.assertIn(("Wikidata", "https://www.wikidata.org/wiki/Q1"), c.work.urls)
        # no search asked: the song's page said it
        self.assertFalse(any(p.get("list") == "search" for p in api._fetch.calls))

    def test_a_search_alone_gives_its_best_guess(self):
        api = tw.Wikimedia(Fake(own=False))
        cands = tw.search(api, "青い鳥", "")
        self.assertEqual([(c.work.title("ja-JP"), c.role, c.sure) for c in cands], [("別の作品", "ending", True)])
        page = "* 「青い鳥」 - テレビアニメ「[[作品A]]」オープニング、ゲーム「[[作品B]]」挿入歌\n"
        read = tw._Reading()
        read.page(tw.Wikimedia(lambda _u, _p: {"parse": {"wikitext": page}}), "ページ", ["青い鳥"], ["青い鳥"])
        self.assertEqual(len(read.found), 2)
        works = {k[0] for k in read.found}
        self.assertIsNone(tw._agreed(read, works, agreed=True))     # not agreed: the searches go on
        picked = read.found[tw._agreed(read, works)]                # but a tag all the same: the anime first
        self.assertEqual((picked.work.title("ja-JP"), picked.role), ("作品A", "opening"))
        not_a_work = {k[0] for k in read.found if read.found[k].work.title("ja-JP") == "作品B"}
        picked = read.found[tw._agreed(read, not_a_work)]            # the anime's page no work's: the game
        self.assertEqual(picked.work.title("ja-JP"), "作品B")
        self.assertIsNone(tw._agreed(read, set()))                  # no work's page: no tag

    def test_a_works_page_by_its_categories(self):
        pages = {"CUE!": ["2019年のコンピュータゲーム", "2022年のテレビアニメ", "アニメ作品 き"],
                 "ゆるゆり": ["2008年の漫画", "ゆるゆり", "漫画作品 ゆ"],
                 "アイドルマスター": ["アイドルマスター"],
                 "世界最強の魔女": ["2022年の小説", "漫画作品 せ"],
                 "井口裕香": ["1988年生", "アニメソング歌手", "存命人物", "日本の女性声優"],
                 "結束バンド (アルバム)": ["2022年のアルバム", "ぼっち・ざ・ろっく!の音楽", "テレビアニメ主題歌"],
                 "坂上忍": ["1967年生", "坂上忍", "存命人物"],
                 "ClariS": ["3人組の音楽グループ", "アニメソング歌手"],
                 "Re:ステージ!": ["2015年の小説", "アイドルを題材とした小説", "メディアミックス作品"],
                 "蓮ノ空女学院スクールアイドルクラブ": ["8人組の音楽グループ", "声優ユニット", "架空の音楽グループ",
                                                    "蓮ノ空女学院スクールアイドルクラブ"],
                 "結束バンド": ["4人組の音楽グループ", "架空の音楽グループ", "結束バンド"],
                 # a franchise's virtual idols: its project; its concert, a game's virtual idol character and a
                 # virtual idol of no franchise are not (their categories as Wikipedia has them)
                 "Vα-liv": ["アイドルマスター", "バーチャルYouTuber", "バーチャルYouTuber関連の組織", "バーチャルアイドル"],
                 "アイカツアカデミー!": ["アイカツ!", "バーチャルYouTuber", "バーチャルアイドル"],
                 "THE IDOLM@STER M@STERS OF IDOL WORLD!!": ["アイドルマスター"],
                 "永瀬麗子": ["ナムコキャラクター", "バーチャルアイドル", "リッジレーサー"],
                 "キズナアイ": ["Upd8のバーチャルYouTuber", "エレクトロニック・ミュージックの歌手", "キズナアイ",
                           "バーチャルYouTuber"],
                 # their categories' categories
                 "Category:アイドルマスター": ["アイドルを題材とした作品", "メディアミックス作品"],
                 "Category:蓮ノ空女学院スクールアイドルクラブ": ["ラブライブ!のユニット"],
                 "Category:結束バンド": ["ぼっち・ざ・ろっく!"],
                 "Category:声優を題材とした作品": ["声優"],
                 "Category:バーチャルアイドル": ["アイドル", "架空の芸能人"],
                 "Category:アイカツ!": ["アイドルを題材としたコンピュータゲーム", "メディアミックス作品"],
                 "Category:リッジレーサー": ["カーレースゲーム", "ナムコのゲームソフト"],
                 "Category:ナムコキャラクター": ["コンピュータゲームメーカー別の登場人物", "ナムコ"],
                 "Category:バーチャルYouTuber": ["アバター", "キャラクター", "ジャンル別のYouTuber"]}

        def fetch(_url, params):
            return {"query": {"pages": [{"title": t, "categories": [{"title": f"Category:{c}"} for c in pages.get(t, [])]}
                                        for t in params["titles"].split("|")]}}

        api = tw.Wikimedia(fetch)
        pages["CUE!"].append("声優を題材とした作品")
        works = ("CUE!", "ゆるゆり", "アイドルマスター", "世界最強の魔女", "井口裕香", "結束バンド (アルバム)", "坂上忍",
                 "ClariS", "Re:ステージ!", "蓮ノ空女学院スクールアイドルクラブ", "結束バンド", "Vα-liv",
                 "アイカツアカデミー!", "THE IDOLM@STER M@STERS OF IDOL WORLD!!", "永瀬麗子", "キズナアイ")
        got = {p: tw.page_category(api, p) for p in works}
        self.assertEqual(got, {"CUE!": "anime", "ゆるゆり": "anime", "アイドルマスター": "idol_2d",
                               "世界最強の魔女": "anime", "井口裕香": None, "結束バンド (アルバム)": None,
                               "坂上忍": None, "ClariS": None, "Re:ステージ!": "idol_2d",
                               "蓮ノ空女学院スクールアイドルクラブ": "idol_2d", "結束バンド": None, "Vα-liv": "idol_2d",
                               "アイカツアカデミー!": "idol_2d", "THE IDOLM@STER M@STERS OF IDOL WORLD!!": None,
                               "永瀬麗子": None, "キズナアイ": None})

    def test_pages_agreeing_on_the_work_only(self):
        """Two pages naming one work under different roles: the work is sure, its role not specified; a singer's page
        listing the song is no work."""
        texts = {"声優A": "* 「星の歌」 - テレビアニメ「[[作品]]」オープニングテーマ\n",
                 "声優B": "* 「星の歌」 - テレビアニメ「[[作品]]」エンディングテーマ\n",
                 "歌手": "== シングル ==\n* 星の歌\n"}
        items = {"作品": "Q1", "歌手": "Q2"}

        def fetch(_url, params):
            if params.get("list") == "search":
                return {"query": {"search": [{"title": t} for t in ("歌手", "声優A", "声優B")]}}
            if params.get("ppprop"):
                return {"query": {"pages": [{"title": "星の歌", "missing": True}]}}
            if params.get("action") == "parse":
                return {"parse": {"wikitext": texts[params["page"]] + "声優A 声優B"}}
            if params.get("prop") == "categories":
                cats = {"作品": ["漫画作品 さ"], "歌手": ["存命人物", "日本の女性歌手"]}.get(params["titles"], [])
                return {"query": {"pages": [{"title": params["titles"],
                                             "categories": [{"title": f"Category:{c}"} for c in cats]}]}}
            if params.get("prop") == "pageprops":
                item = items.get(params["titles"])
                return {"query": {"pages": [{"pageprops": {"wikibase_item": item} if item else {}}]}}
            classes = {"Q1": "Q21198342", "Q2": "Q215380"}
            return {"entities": {params["ids"]: {"labels": {"ja": {"value": "作品"}}, "claims": {"P31": [
                {"mainsnak": {"datavalue": {"value": {"id": classes[params["ids"]]}}}}]}}}}

        cands = tw.search(tw.Wikimedia(fetch), "星の歌", "歌手A(CV:声優A)")
        sure = [c for c in cands if c.sure]
        self.assertEqual([(c.work.title("ja-JP"), c.role, c.work.category) for c in sure],
                         [("作品", tieup.UNSPECIFIED, "anime")])
        self.assertEqual(tieup.TieUp(sure[0].work, sure[0].role).short(), "作品")    # shown as the work alone
        self.assertNotIn("歌手", [c.work.title("ja-JP") for c in cands])

    def test_roles_on_a_track_that_has_the_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LibraryDB(str(Path(tmp) / "library.db"))
            db.connect()
            uid = str(uuid4())
            db.upsert(TrackRow(uid=uid, path="a.wav", title="歌", added_ts=1))
            db.conn.commit()
            work = tieup.Work("", "anime", [("ja-JP", "作品")])
            with db.conn:
                tieups.save_tieup(db.conn, uid, tieup.TieUp(work, "opening"))
            found = lambda role: tw.Candidate(tieup.Work("", "anime", [("ja-JP", "作品")]), role)  # noqa: E731
            roles = lambda: [(t.work.id, t.role) for t in tieups.track_tieups(db.conn, uid)]  # noqa: E731
            with db.conn:
                tieups.add_found(db.conn, uid, found(tieup.UNSPECIFIED).as_tieup())   # the track has the work: nothing more
            self.assertEqual(roles(), [(work.id, "opening")])
            with db.conn:
                tieups.add_found(db.conn, uid, found("ending").as_tieup())            # another role: added
            self.assertEqual(roles(), [(work.id, "opening"), (work.id, "ending")])
            other = tieup.Work("", "game", [("ja-JP", "ゲーム")])
            with db.conn:
                tieups.save_tieup(db.conn, uid, tieup.TieUp(other, tieup.UNSPECIFIED))
                tieups.add_found(db.conn, uid,
                                 tw.Candidate(tieup.Work("", "game", [("ja-JP", "ゲーム")]), "theme").as_tieup())
            self.assertEqual(roles(), [(work.id, "opening"), (work.id, "ending"), (other.id, "theme")])  # its place
            db.close()

    def test_added_into_the_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LibraryDB(str(Path(tmp) / "library.db"))
            db.connect()
            uid = str(uuid4())
            db.upsert(TrackRow(uid=uid, path="a.wav", title="青い鳥", artist="結束バンド", added_ts=1))
            db.conn.commit()
            # the library has the work under one of its names already: that work, its other names added
            with db.conn:
                mine = tieups.save_work(db.conn, tieup.Work("", "anime", [("ja-JP", "ぼっち・ざ・ろっく！")]))
            api = tw.Wikimedia(Fake())
            cache = str(Path(tmp) / "cache.db")
            lib = SimpleNamespace(db=db)      # the jobs' library (core.library.service.Library)
            # as the lookup's thread does it (core.workflows.tieup_web): the song asked of the library's owner,
            # Wikimedia asked, the sure one written by the owner
            song = jobs.song_of(lib, uid, untagged=True)
            sure = tw.sure_of(api, song, cache)
            with db.conn:
                jobs.add_found_tieups(lib, [uid], [sure.as_tieup()])
            calls = len(api._fetch.calls)
            self.assertIsNone(jobs.song_of(lib, uid, untagged=True))                          # has one now
            self.assertEqual(tw.lookup(api, *song[:2], cache, song[2])[0].role, "opening")
            self.assertEqual(len(api._fetch.calls), calls)                                    # the cache answered
            got = tieups.track_tieups(db.conn, uid)
            self.assertEqual([(t.work.id, t.role) for t in got], [(mine, "opening")])
            self.assertEqual(got[0].work.title("ko-KR"), "봇치 더 록!")
            self.assertEqual(got[0].work.title("ja-JP"), "ぼっち・ざ・ろっく！")   # the library's own kept first
            db.close()


if __name__ == "__main__":
    unittest.main()
