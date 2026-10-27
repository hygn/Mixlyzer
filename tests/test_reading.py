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
        # Latin inside Japanese kept as written (not read as a word); は read wa
        self.assertTrue(reading.read("U.N.オーエンは彼女なのか？").startswith("U.N. ooen wa kanojo"))
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

    def test_not_found(self):
        self.assertFalse(found("sakura", "あのね"))
        self.assertFalse(found("ka", "東京"))   # a coarse key under three letters matches nothing
        self.assertFalse(found("사회부적합자", "ZAQ"))   # a short reading is not looked for in the query


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
                db.set_user_readings(uid, "title", ["anone desu", "anone"])
                self.assertEqual(db.readings(uid)["title"], [("anone desu", "user"), ("anone", "user"), ("ano ne", "ja")])
                db.upsert(TrackRow(uid=uid, path="a.wav", title="アノネ!", artist="東京"))
                got = db.readings(uid)
                self.assertEqual(got["title"], [("anone desu", "user"), ("anone", "user"), ("anone !", "ja")])
                self.assertEqual(got["artist"], [("toukyou", "ja"), ("dong jing", "zh"), ("dung ging", "yue"),
                                                 ("donggyeong", "ko-hanja"), ("dong kinh", "vi")])
                self.assertEqual(db.all_readings()[uid]["title"], ["anone desu", "anone", "anone !"])
                # an automatic reading the user also typed: once, as the user's
                db.set_user_readings(uid, "artist", ["dong jing"])
                self.assertEqual(db.readings(uid)["artist"][:3], [("dong jing", "user"), ("toukyou", "ja"), ("dung ging", "yue")])
                # removed: only the automatic ones
                db.set_user_readings(uid, "title", None)
                self.assertEqual(db.readings(uid)["title"], [("anone !", "ja")])
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
        self.assertEqual(reading.readings(title), [("Re: zero kara hajimeru isekai seikatsu", "ja"),
                                                   ("Re: zero kara hajimeru isegyesaenghwal", "ko-hanja")])
        self.assertTrue(found("이세계생활", title))
        self.assertTrue(found("사회부적합자", "ハイ!社会不適合者"))
        self.assertEqual([s for _r, s in reading.readings("夜に駆ける", han_run=2)], ["ja"])     # no run of 2
        self.assertEqual([s for _r, s in reading.readings("不思議なクスリ", han_run=4)], ["ja"])  # 3 < 4
        self.assertEqual([s for _r, s in reading.readings(title, han_run=0)], ["ja"])            # off
        self.assertEqual([s for _r, s in reading.readings(title, ["ja", "zh"])], ["ja"])           # no Korean Hanja
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


if __name__ == "__main__":
    unittest.main()
