import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from core.search import reading
from core.library.db import LibraryDB
from core.library.records import TrackRow


def refresh_library_readings(path: str, languages, progress=None, han_run: int = reading.HAN_RUN_MIN):
    """Re-read Library as the app runs it (core.workflows.reading_refresh): the owner's jobs give the library's
    texts, the process's reading reads them, the owner's jobs write them back; here on one connection. (tracks,
    fields changed)."""
    from types import SimpleNamespace
    from core.library import jobs
    from core.workers.reading_refresh import read_texts
    db = LibraryDB(path, reading_languages=list(languages), han_run=han_run)
    db.connect()
    lib = SimpleNamespace(db=db)       # the jobs' library (core.library.service.Library): the connection is all they use
    try:
        uids, texts = jobs.reading_texts(lib)
        read = read_texts(texts, languages, han_run, progress)
        changed = jobs.apply_readings(lib, uids, read)
        jobs.apply_work_readings(lib, read)
        db.conn.commit()
        return len(uids), changed
    finally:
        db.close()


def found(query: str, title: str) -> bool:
    """Whether the library's search finds ``title`` by ``query`` (any of its readings)."""
    return bool(reading.find(reading.query_keys(query), [reading.keys(r) for r, _src in reading.readings(title)]))


class ReadingTests(unittest.TestCase):
    def test_readings(self):
        self.assertEqual(reading.read("あのね"), "ano ne")
        self.assertEqual(reading.read("東京テディベア"), "toukyou tedibea")
        self.assertEqual(reading.read("きっと"), "kitto")
        self.assertEqual(reading.read("ファンタジー"), "fantajii")
        self.assertEqual(reading.read("사랑해"), "saranghae")
        self.assertEqual(reading.read("잇쇼"), "itsyo")
        self.assertEqual(reading.read("Москва"), "Moskva")
        # Latin inside Japanese kept as written (not read as a word), its symbols left out; は read wa
        self.assertTrue(reading.read("U.N.オーエンは彼女なのか？").startswith("U N ooen wa kanojo"))
        self.assertEqual(reading.read("†吸tie Ladies†"), "kyuu tie Ladies")        # † is no sound (anyascii: +)
        # hanzi the dictionary does not know: pinyin
        self.assertEqual(reading.fine_key(reading.read("洛天依")), "luotianyi")

    def test_found_by_sound_in_any_script(self):
        for query, title in [
            ("anone", "あのね"), ("ano ne", "あのね"), ("아노네", "あのね"), ("ANONE", "あのね"),
            ("tokyo", "東京テディベア"), ("도쿄", "東京テディベア"), ("테디베아", "東京テディベア"),
            ("isshoni", "いっしょにいようよ"), ("잇쇼니", "いっしょにいようよ"),
            ("senbonzakura", "千本桜"), ("센본자쿠라", "千本桜"),
            ("하츠네", "初音ミク"), ("하쓰네", "初音ミク"), ("hatsune", "初音ミク"),
            ("요루니카케루", "夜に駆ける"), ("판타지", "ファンタジー"), ("샤를", "シャルル"),
            ("saranghae", "사랑해"), ("luotianyi", "洛天依"), ("cafe", "Café de Paris"),
            ("bocchi", "ぼっち"), ("botchi", "ぼっち"), ("봇치", "ぼっち"),   # っち spelt cchi or tchi
        ]:
            with self.subTest(query=query, title=title):
                self.assertTrue(found(query, title))

    def test_korean_revised_romanization_sound_changes(self):
        for word, rr in [("신라", "silla"), ("같이", "gachi"), ("독립", "dongnip"), ("좋다", "jota"),
                         ("종로", "jongno"), ("읽어", "ilgeo"), ("설날", "seollal"), ("국물", "gungmul"),
                         ("맛있다", "masitda"), ("읽다", "ikda"), ("사랑해", "saranghae"), ("축하", "chuka")]:
            with self.subTest(word=word):
                self.assertEqual(reading.korean_to_roman(word), rr)

    def test_similar_pronunciation(self):
        # the dictionary misreads 鏡音 (kagami on): a query by its sound still finds it
        self.assertEqual(reading.read("鏡音リン"), "kagami on rin")
        self.assertTrue(found("kagamine rin", "鏡音リン"))
        self.assertTrue(found("카가미네 린", "鏡音リン"))
        self.assertFalse(found("hatsune", "鏡音リン"))

    def test_variant_characters_read_as_the_standard_ones(self):
        from core.tieups.domain import name_key
        self.assertEqual(reading.read("髙橋ミナミ"), reading.read("高橋ミナミ"))
        self.assertTrue(found("takahashi", "髙橋ミナミ"))
        self.assertTrue(found("yamazaki", "山﨑はるか"))
        self.assertIn("tanezaki", reading.read("種﨑敦美"))   # a word the dictionary knows as written: kept
        self.assertEqual(reading.read("𠮷田"), "yoshida")     # beyond the BMP: Han too
        for variant, standard in [("髙橋ミナミ", "高橋ミナミ"), ("山﨑はるか", "山崎はるか"), ("𠮷田", "吉田"),
                                  ("渡邉", "渡邊")]:
            with self.subTest(variant=variant):
                self.assertEqual(name_key(variant), name_key(standard))
        self.assertNotEqual(name_key("澤"), name_key("沢"))     # an old form, a character of its own in names

    def test_not_found(self):
        self.assertFalse(found("sakura", "あのね"))
        self.assertFalse(found("ka", "東京"))   # a coarse key under three letters matches nothing
        self.assertFalse(found("사회부적합자", "ZAQ"))   # a short reading is not looked for in the query


class EnglishReadingTests(unittest.TestCase):
    """A Latin title found by its katakana or Hangul: its English words spelled from their sounds (CMUdict)."""

    def test_spelled_as_each_language_spells_a_loanword(self):
        table = reading._english_table()
        for word, japanese, hangul in [
            ("brave", "bureibu", "브레이브"), ("shine", "shain", "샤인"), ("dream", "doriimu", "드림"),
            ("heart", "haato", "하트"), ("love", "rabu", "러브"), ("girl", "gaaru", "걸"), ("star", "sutaa", "스타"),
            ("music", "myuuziku", "뮤직"), ("queen", "kuiin", "퀸"), ("street", "sutoriito", "스트리트"),
            ("cat", "kyato", "캣"), ("fly", "furai", "플라이"), ("world", "waarudo", "월드"), ("song", "soongu", "송"),
        ]:
            with self.subTest(word=word):
                phones = table[word].split()
                self.assertEqual(reading.english_japanese(phones), japanese)
                self.assertEqual(reading.english_hangul(phones), hangul)

    def test_readings_after_the_first(self):
        self.assertEqual(reading.readings("Brave Shine"), [
            ("Brave Shine", "text"), ("bureibu shain", "en-ja"), ("bureibu syain", "en-ko")])
        self.assertEqual(reading.readings("恋のMagic")[0], ("koi no Magic", "ja"))
        self.assertIn(("koi no majiku", "en-ja"), reading.readings("恋のMagic"))
        self.assertEqual(reading.readings("あのね"), [("ano ne", "ja")])                # no English: none added
        # an apostrophe inside a word; abbreviations and words not in the dictionary as written
        self.assertIn(("doonto sei reizii", "en-ja"), reading.readings("Don't say lazy"))
        self.assertEqual(reading.readings("CV Mixlyzer"), [("CV Mixlyzer", "text")])

    def test_found_by_katakana_and_hangul(self):
        for query, title in [
            ("ブレイブシャイン", "Brave Shine"), ("브레이브 샤인", "Brave Shine"), ("샤인", "Brave Shine"),
            ("ドリーム", "Dream"), ("드림", "Dream"), ("ラブ", "Love"), ("러브", "Love"), ("ハート", "Heart"),
            ("하트", "Heart"), ("ミュージック", "Music"), ("뮤직", "Music"), ("キャット", "Cat"), ("캣", "Cat"),
            ("ドントセイレイジー", "Don't say lazy"), ("돈트 세이 레이지", "Don't say lazy"),
            ("メロディー", "Melody"), ("멜로디", "Melody"), ("ワールド", "World"), ("월드", "World"),
        ]:
            with self.subTest(query=query, title=title):
                self.assertTrue(found(query, title))
        self.assertFalse(found("드림", "Brave Shine"))
        self.assertFalse(found("シーブイ", "CV Mixlyzer"))      # letters are not spelled by their names


class ChineseReadingTests(unittest.TestCase):
    def test_han_without_kana_reads_in_each_language(self):
        found_readings = dict((src, r) for r, src in reading.readings("乾杯"))
        self.assertEqual(found_readings, {"ja": "kanpai", "zh": "qian bei", "yue": "gon bui", "ko-hanja": "geonbae",
                                          "vi": "kien boi", "zh-tw": "gan bei"})
        self.assertEqual([src for _r, src in reading.readings("東京テディベア")], ["ja"])   # kana: Japanese
        self.assertLessEqual(len(reading.readings("千本桜")), reading.READING_SLOTS)

    def test_cantonese_reads_forms_it_lacks_by_their_variants(self):
        def yue(text):
            return dict((src, r) for r, src in reading.readings(text))["yue"]
        self.assertEqual(yue("社会"), "se wui")     # 会: a Japanese new / simplified form of 會
        self.assertEqual(yue("学校"), "hok haau")
        self.assertEqual(yue("千本桜"), "cin bun jing")
        self.assertEqual(yue("学生"), "hok saang")   # the word found in its variants (生 alone: sang)

    def test_cantonese_by_the_longest_words_without_the_segmenter(self):
        from pycantonese.word_segmentation.segmenter import _get_default_segmenter
        self.assertEqual(reading._cantonese_split("香港人講廣東話"), ["香港人", "講", "廣東話"])
        self.assertEqual(reading._read_cantonese("銀行"), "ngan hong")   # 行 by its word (alone: hang)
        self.assertEqual(reading._read_cantonese("行人"), "hang jan")
        self.assertEqual(_get_default_segmenter.cache_info().currsize, 0)   # its model (160 MB) never loaded

    def test_found_by_chinese_readings(self):
        for query, title in [("qianbenying", "千本桜"), ("cin bun jing", "千本桜"), ("senbonzakura", "千本桜"),
                             ("luotianyi", "洛天依"), ("lok tin ji", "洛天依"), ("chongqing", "重慶森林")]:
            with self.subTest(query=query, title=title):
                keys = [reading.keys(r) for r, _src in reading.readings(title)]
                self.assertTrue(reading.find(reading.query_keys(query), keys))


class HanjaAndSinoVietnameseTests(unittest.TestCase):
    def test_korean_hanja(self):
        def ko(text):
            return dict((src, r) for r, src in reading.readings(text))["ko-hanja"]
        # the initial sound law on the first syllable, then Revised Romanization's sound changes
        self.assertEqual(ko("李理"), "iri")
        self.assertEqual(ko("樂園"), "nagwon")
        self.assertEqual(ko("女神"), "yeosin")
        self.assertEqual(ko("独立"), "dongnip")   # a Japanese new form, read via its old one
        self.assertEqual(ko("新羅"), "silla")
        self.assertTrue(found("천본앵", "千本桜"))
        # spelled as a word: 不 부 before ㄷ / ㅈ, 렬 / 률 after a vowel or ㄴ 열 / 율
        self.assertEqual(ko("不適合"), "bujeokap")
        self.assertEqual(ko("不安"), "buran")
        self.assertEqual(ko("比率"), "biyul")
        self.assertEqual(ko("旋律"), "seonyul")
        self.assertEqual(ko("法律"), "beomnyul")

    def test_found_by_korean_hanja_reading(self):
        # a Hangul query read as Korean as well (its foreign spelling folds 적 jeok into jok)
        for query in ("사회", "사회부적합자", "부적합", "shakai"):
            with self.subTest(query=query):
                self.assertTrue(found(query, "社会不適合者☆"))
                self.assertTrue(reading.find(reading.query_keys(query),
                                             [reading.keys(r) for r, _src in reading.readings("社会不適合者☆")],
                                             similar=False))

    def test_sino_vietnamese(self):
        def vi(text):
            return dict((src, r) for r, src in reading.readings(text))["vi"]
        self.assertEqual(vi("東京"), "dong kinh")   # đông, not the Nom đang
        self.assertEqual(vi("愛國"), "ai quoc")     # 愛 missing from Unihan: read in Mandarin
        self.assertTrue(found("dong kinh", "東京"))

    def test_seven_slots(self):
        self.assertEqual(reading.READING_SLOTS, 7)
        self.assertLessEqual(len(reading.readings("乾杯")), reading.READING_SLOTS)

    def test_fuzzy_off(self):
        keys = [reading.keys(r) for r, _src in reading.readings("鏡音リン")]
        query = reading.query_keys("kagamine rin")
        self.assertTrue(reading.find(query, keys, similar=True))
        self.assertFalse(reading.find(query, keys, similar=False))


class ReadingDatabaseTests(unittest.TestCase):
    def test_readings_follow_the_text_after_the_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LibraryDB(str(Path(tmp) / "library.db"))
            db.connect()
            try:
                uid = str(uuid4())
                db.upsert(TrackRow(uid=uid, path="a.wav", title="あのね", artist="千本桜", added_ts=1))
                got = db.readings(uid)
                self.assertEqual(got["title"], [("ano ne", "ja")])
                self.assertEqual([src for _r, src in got["artist"]][:3], ["ja", "zh", "yue"])
                # added by the user: kept first when the title changes; the automatic ones follow the text
                db.set_user_readings(uid, "title", ["anone desu", "a no ne"])
                self.assertEqual(db.readings(uid)["title"], [("anone desu", "user"), ("a no ne", "user"), ("ano ne", "ja")])
                db.upsert(TrackRow(uid=uid, path="a.wav", title="アノネ!", artist="東京"))
                got = db.readings(uid)
                # its "!" left out (a reading is how the text sounds)
                self.assertEqual(got["title"], [("anone desu", "user"), ("a no ne", "user"), ("anone", "ja")])
                self.assertEqual(got["artist"], [("toukyou", "ja"), ("dong jing", "zh"), ("dung ging", "yue"),
                                                 ("donggyeong", "ko-hanja"), ("dong kinh", "vi")])
                self.assertEqual(db.all_readings()[uid]["title"], ["anone desu", "a no ne", "anone"])
                # an automatic reading the user also typed: once, as the user's
                db.set_user_readings(uid, "artist", ["dong jing"])
                self.assertEqual(db.readings(uid)["artist"][:3], [("dong jing", "user"), ("toukyou", "ja"), ("dung ging", "yue")])
                # removed: only the automatic ones
                db.set_user_readings(uid, "title", None)
                self.assertEqual(db.readings(uid)["title"], [("anone", "ja")])
                # gone with the track
                db.delete_tracks([uid])
                self.assertEqual(db.all_readings(), {})
            finally:
                db.close()


    def test_removed_automatic_readings_stay_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "library.db")
            db = LibraryDB(path)
            db.connect()
            try:
                uid = str(uuid4())
                db.upsert(TrackRow(uid=uid, path="a.wav", title="東京", added_ts=1))
                db.set_user_readings(uid, "title", ["tokio"], removed=["dong jing"])
                self.assertEqual([r for r, _s in db.readings(uid)["title"]],
                                 ["tokio", "toukyou", "dung ging", "donggyeong", "dong kinh"])
                db.upsert(TrackRow(uid=uid, path="a.wav", title="東京", added_ts=1))   # written again
                self.assertNotIn("dong jing", [r for r, _s in db.readings(uid)["title"]])
                self.assertEqual(db.removed_readings(uid)["title"], ["dong jing"])
            finally:
                db.close()
            refresh_library_readings(path, ("ja", "zh"))                       # whatever the languages
            db = LibraryDB(path, reading_languages=["ja", "zh"])
            db.connect()
            try:
                self.assertEqual(db.readings(uid)["title"], [("tokio", "user"), ("toukyou", "ja")])
                # typed again: the user's, no longer removed
                db.set_user_readings(uid, "title", ["tokio", "dong jing"], removed=["dong jing"])
                self.assertEqual(db.readings(uid)["title"], [("tokio", "user"), ("dong jing", "user"), ("toukyou", "ja")])
                self.assertEqual(db.removed_readings(uid)["title"], [])
                db.set_user_readings(uid, "title", [], removed=["toukyou"])
                db.delete_tracks([uid])                                         # gone with the track
                self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM track_reading_removals").fetchone()[0], 0)
            finally:
                db.close()


class ReadingLanguageTests(unittest.TestCase):
    def test_stored_languages(self):
        self.assertEqual([src for _r, src in reading.readings("東京", ["vi", "ja"])], ["ja", "vi"])   # slot order
        self.assertEqual([src for _r, src in reading.readings("東京", [])], ["ja", "zh", "yue", "ko-hanja", "vi"])
        self.assertEqual([src for _r, src in reading.readings("あのね", ["zh"])], ["ja"])   # kana: Japanese
        self.assertEqual([src for _r, src in reading.readings("사랑", ["zh"])], ["ko"])

    def test_query_languages(self):
        keys = [reading.keys(r) for r, _src in reading.readings("東京")]
        self.assertTrue(reading.find(reading.query_keys("東京", ["ja"]), keys))
        self.assertEqual(len(reading.query_keys("東京", ["ja"])), 1)
        # Hangul: as Korean (社会's Korean Hanja reading) / as a foreign word (東京's Japanese reading)
        social = [reading.keys(r) for r, _src in reading.readings("社会不適合者")]
        self.assertTrue(reading.find(reading.query_keys("사회부적합자", hangul_foreign=False), social, similar=False))
        self.assertTrue(reading.find(reading.query_keys("도쿄", hangul_korean=False), keys, similar=False))
        # the exact romanization (fine key) is kept either way; the normalized key follows the choice
        self.assertEqual(reading.query_keys("사회", hangul_foreign=False), [("sahoe", "sahoe")])
        self.assertEqual(reading.query_keys("사회", hangul_korean=False), [("sahoe", "sahe")])

    def test_japanese_text_han_runs_in_korean_hanja(self):
        title = "Re:ゼロから始める異世界生活"
        self.assertEqual(reading.readings(title)[:2], [("Re zero kara hajimeru isekai seikatsu", "ja"),
                                                       ("Re zero kara hajimeru isegyesaenghwal", "ko-hanja")])
        self.assertTrue(found("이세계생활", title))
        self.assertTrue(found("사회부적합자", "ハイ!社会不適合者"))
        self.assertEqual([s for _r, s in reading.readings("夜に駆ける", han_run=2)], ["ja"])     # no run of 2
        self.assertEqual([s for _r, s in reading.readings("不思議なクスリ", han_run=4)], ["ja"])  # 3 < 4
        # (its Re read as English too, after them)
        self.assertEqual([s for _r, s in reading.readings(title, han_run=0)], ["ja", "en-ja"])             # off
        self.assertEqual([s for _r, s in reading.readings(title, ["ja", "zh"])], ["ja", "en-ja"])  # no Korean Hanja
        # 々 repeats a Japanese word, not a Han character to read
        self.assertEqual(reading.readings("時々不思議だね", han_run=2)[1], ("tokidoki bulsaui da ne", "ko-hanja"))
        # the settings: off is 0; a config.json without the fields takes the defaults
        from core.settings.config import default_cfg, search_settings
        s = default_cfg().searchconfig
        self.assertEqual((s.han_run(), s.stored_readings()[1]), (3, 3))
        s.han_runs = False
        self.assertEqual(s.han_run(), 0)
        self.assertEqual(search_settings({"db_languages": ["ja"]}).stored_readings(), (("ja",), 3))

    def test_typed_without_the_korean_english_key(self):
        # English mode, Korean meant: composed as the 2-set input method does (Shift: double consonants)
        self.assertEqual(reading.keys_to_hangul("tkghl"), "사회")
        self.assertEqual(reading.keys_to_hangul("tkghlqnwjrgkqwk"), "사회부적합자")
        self.assertEqual(reading.keys_to_hangul("dkssudgktpdy"), "안녕하세요")
        self.assertEqual(reading.keys_to_hangul("TKGHL"), "사회")        # Caps Lock
        self.assertEqual(reading.keys_to_hangul("Tkfkd"), "싸랑")        # Shift
        self.assertEqual(reading.keys_to_hangul("dbfndbfl OVA"), "유루유리 OVA")   # word by word
        for text in ("anone", "tokyo", "go", "dog", "love song", "hatsune miku", "senbonzakura", "kagamine rin"):
            with self.subTest(text=text):
                self.assertIsNone(reading.keys_to_hangul(text))       # English / romaji left as typed
        # Korean mode, letters meant: loose jamo give it away
        self.assertEqual(reading.hangul_to_keys("무ㅐㅜㄷ"), "anone")
        self.assertEqual(reading.hangul_to_keys("ㅗㅁㅅ녀ㅜㄷ"), "hatsune")
        self.assertIsNone(reading.hangul_to_keys("사회"))
        social = [reading.keys(r) for r, _src in reading.readings("社会不適合者☆")]
        anone = [reading.keys(r) for r, _src in reading.readings("あのね")]
        self.assertTrue(reading.find(reading.query_keys("tkghl"), social, similar=False))
        self.assertTrue(reading.find(reading.query_keys("dkshsp"), anone, similar=False))   # 아노네
        self.assertTrue(reading.find(reading.query_keys("무ㅐㅜㄷ"), anone, similar=False))
        self.assertFalse(reading.find(reading.query_keys("tkghl", keyboard=False), social))

    def test_library_reads_in_its_languages_and_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "library.db")
            db = LibraryDB(path, reading_languages=["ja"])
            db.connect()
            try:
                uid = str(uuid4())
                db.upsert(TrackRow(uid=uid, path="a.wav", title="社会", artist="東京", added_ts=1))
                self.assertEqual(db.readings(uid)["title"], [("shakai", "ja")])
                db.set_user_readings(uid, "artist", ["tokyo"])
                self.assertEqual(db.readings(uid)["artist"], [("tokyo", "user"), ("toukyou", "ja")])
            finally:
                db.close()
            # another language checked: its readings added after the user's
            done = []
            result = refresh_library_readings(path, ("ja", "ko-hanja"), lambda d, t: done.append((d, t)))
            self.assertEqual(result, (1, 2))                 # tracks, fields changed
            self.assertEqual(done[-1][0], done[-1][1])       # every text read
            db = LibraryDB(path)
            db.connect()
            try:
                self.assertEqual(db.readings(uid)["title"], [("shakai", "ja"), ("sahoe", "ko-hanja")])
                self.assertEqual(db.readings(uid)["artist"],
                                 [("tokyo", "user"), ("toukyou", "ja"), ("donggyeong", "ko-hanja")])
            finally:
                db.close()
            # one unchecked: its readings removed, the user's kept
            refresh_library_readings(path, ("ko-hanja",))
            db = LibraryDB(path)
            db.connect()
            try:
                self.assertEqual(db.readings(uid)["title"], [("sahoe", "ko-hanja")])
                self.assertEqual(db.readings(uid)["artist"], [("tokyo", "user"), ("donggyeong", "ko-hanja")])
            finally:
                db.close()


def initials_found(query: str, text: str) -> bool:
    """Whether the 초성 search finds ``text`` by ``query`` (its Japanese reading as the library stores it)."""
    japanese = [r for r, src in reading.readings(text, ["ja"]) if src == "ja"]
    pattern = reading.initials_pattern(query)
    return any(pattern.search(k) for k in reading.initials_keys(text, japanese))


class InitialsTests(unittest.TestCase):
    def test_a_query_of_consonants_alone(self):
        self.assertTrue(reading.is_initials_query("ㄷㅅㅁㄴ"))
        self.assertTrue(reading.is_initials_query("ㄷㅅ ㅁㄴ"))
        self.assertFalse(reading.is_initials_query("ㄷ"))            # one would match nearly everything
        self.assertFalse(reading.is_initials_query("ㄷㅏ"))           # a vowel: a mistyped query, read as one

    def test_typed_without_the_korean_english_key(self):
        self.assertEqual(reading.keys_to_initials("etastr"), "ㄷㅅㅁㄴㅅㄱ")    # 다시 만난 세계
        self.assertEqual(reading.keys_to_initials("etas tr"), "ㄷㅅㅁㄴㅅㄱ")
        self.assertEqual(reading.keys_to_initials("ETAS"), "ㄷㅅㅁㄴ")           # Caps Lock: no Shift
        self.assertEqual(reading.keys_to_initials("Rt"), None)                  # fewer than 3
        self.assertIsNone(reading.keys_to_initials("Butter-Fly"))               # a vowel's key, a hyphen
        self.assertIsNone(reading.keys_to_initials("ㄷㅅㅁ"))

    def test_hangul_hanja_and_a_japanese_reading(self):
        self.assertTrue(initials_found("ㄷㅅㅁㄴㅅㄱ", "다시 만난 세계"))
        self.assertTrue(initials_found("ㅁㄴㅅㄱ", "다시 만난 세계"))        # inside it
        self.assertTrue(initials_found("ㅊㅂㅇ", "千本桜"))                  # Korean Hanja: 천본앵
        self.assertTrue(initials_found("ㅅㅂㅅㅋㄹ", "千本桜"))              # Japanese: 센본사쿠라
        self.assertTrue(initials_found("ㄷㅋ", "東京"))                      # 도쿄, the long vowels left out
        self.assertTrue(initials_found("ㅇㅅㄱㅅㅎ", "Re:ゼロから始める異世界生活"))   # its Han run: 이세계생활
        # a plain, aspirated or tense consonant alike, つ as ㅅ or ㅈ (쓰 / 츠)
        self.assertTrue(initials_found("ㄱㅁㄴㅅㄹㄴㅇㅁㄴㄱㅌㄹ", "君の知らない物語"))
        self.assertTrue(initials_found("ㅅㄱㅁ", "月見"))
        self.assertTrue(initials_found("ㅊㅋㅁ", "月見"))
        self.assertFalse(initials_found("ㄷㅅㅁㄴㅅㄱ", "Butter-Fly"))


class NameSortTests(unittest.TestCase):
    TITLES = ["残酷な天使のテーゼ", "愛にできることはまだあるかい", "かえるのうた", "学園天国", "君の知らない物語",
              "キャッツ・アイ", "千本桜", "私は、わたしの事が好き。", "Butter-Fly", "다시 만난 세계", "1/6の夢旅人"]

    def order(self, order: str) -> list[str]:
        def key(title: str) -> str:
            japanese = [r for r, src in reading.readings(title, ["ja"]) if src == "ja"]
            return reading.sort_key(title, japanese, japanese, order)
        return sorted(self.TITLES, key=key)

    def test_gojuon(self):
        # one order by sound: digits first; か before が (かえる, がくえん), き + み before き + ゃ; 다시 (dasi) among
        # the た, Butter-Fly (ブッテル) among the は
        self.assertEqual(self.order("gojuon"), ["1/6の夢旅人", "愛にできることはまだあるかい", "かえるのうた", "学園天国",
                                                 "君の知らない物語", "キャッツ・アイ", "残酷な天使のテーゼ", "千本桜",
                                                 "다시 만난 세계", "Butter-Fly", "私は、わたしの事が好き。"])

    def test_hangul(self):
        # 가쿠엔, 다시, 붓테르, 센본, 아이, 와타쿠시, 잔코쿠, 카에루, 캿츠, 키미: か き aspirated wherever they are
        self.assertEqual(self.order("hangul"), ["1/6の夢旅人", "学園天国", "다시 만난 세계", "Butter-Fly", "千本桜",
                                                 "愛にできることはまだあるかい", "私は、わたしの事が好き。",
                                                 "残酷な天使のテーゼ", "かえるのうた", "キャッツ・アイ", "君の知らない物語"])

    @staticmethod
    def keys(text: str) -> dict:
        found = reading.readings(text)
        japanese = [r for r, src in reading.readings(text, ["ja"], han_run=0) if src == "ja"]
        return reading.name_keys(text, found[0][0] if found else "", japanese)[1]

    def test_the_librarys_names(self):
        # a small っ at the end kept (wapuro's xtu): つ in 五十音, the syllable before's ㅅ in 가나다
        self.assertTrue(reading.read("せーのっ!").endswith("xtu"))
        self.assertTrue(self.keys("せーのっ!")["gojuon"].startswith("せのつ "))
        self.assertTrue(self.keys("恋愛したいっ！")["hangul"].endswith("잇"))
        # English inside a Japanese reading by its spelling: Remix, Café Mix (x: くす)
        self.assertIn("きしはとれみくす", self.keys("100%ちゅ～学生 (Kijibato remix)")["gojuon"])
        self.assertIn("かふえみくす", self.keys("続く話 Café Mix")["gojuon"])
        # a short word in capitals spelled (DJ ディージェー), the ザ; a doubled letter one (Girl, Summit)
        self.assertIn("ていいしええ", self.keys("せーのっ! (DJ UTO Remix)")["gojuon"])
        self.assertTrue(self.keys("The Demon Girl Next Door")["gojuon"].startswith("さてもんきるねくすと"))
        self.assertIn("すみと", self.keys("ヤマノススメ Next Summit")["gojuon"])
        self.assertTrue(self.keys("Encouragement of Climb: Next Summit")["gojuon"].startswith("えんこうらけめんと"))
        # a Korean name's Latin words in Hangul for 가나다
        self.assertEqual(self.keys("그녀도 여친 Season 2")["hangul"], "그녀도여친세아손2")
        self.assertEqual(self.keys("외톨이 THE ROCK!")["hangul"], "외톨이자로크")

    def test_latin_names_by_their_spelling(self):
        # a Latin name among the kana by its letters read as Japanese takes a foreign word in
        self.assertEqual(reading.gojuon_key("kurarisu"), reading.sort_key("ClariS", ["ClariS"], [], "gojuon"))
        self.assertEqual(reading.sort_key("LiSA", ["LiSA"], [], "hangul"), "리사")

    def test_abc_and_as_written(self):
        self.assertEqual(self.order("abc")[:4], ["1/6の夢旅人", "愛にできることはまだあるかい", "Butter-Fly", "学園天国"])
        self.assertIsNone(reading.sort_key("千本桜", ["senbon sakura"], ["senbon sakura"], "text"))

    def test_japanese_as_korean_fans_spell_it(self):
        # か た ち aspirated wherever they are, つ 츠: as fans write them, not the loanword rule (기미, 고이, 쓰)
        self.assertEqual(reading.japanese_to_hangul("senbon sakura"), "센본 사쿠라")
        self.assertEqual(reading.japanese_to_hangul("kimi ga shinu made koi o shitai"), "키미 가 시누 마데 코이 오 시타이")
        self.assertEqual(reading.japanese_to_hangul("hatsune miku"), "하츠네 미쿠")
        self.assertEqual(reading.japanese_to_hangul("tsukimi"), "츠키미")
        self.assertEqual(reading.japanese_to_hangul("watashi"), "와타시")

    def test_places_as_the_loanword_rule_spells_them(self):
        # a place's name (Sudachi: 固有名詞 地名) as Korean knows it; the words beside it as fans write them
        self.assertEqual(reading.place_names("東京ラブストーリー"), {"toukyou"})
        self.assertEqual(NameSortTests.keys("東京ラブストーリー")["hangul"], "도쿄라부스토리")
        self.assertEqual(NameSortTests.keys("京都慕情")["hangul"], "교토보조")
        self.assertEqual(NameSortTests.keys("初音ミク")["hangul"], "하츠네미쿠")        # a person's: not a place
        self.assertEqual(reading.japanese_to_hangul("matsuyama", frozenset({"matsuyama"})), "마쓰야마")


if __name__ == "__main__":
    unittest.main()
