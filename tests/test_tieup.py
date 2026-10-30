"""Tie-ups: the domain (core.tieups.domain) and their storage in library.db (core.library.tieups)."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from core.search import reading

from core.tieups import domain as tieup
from core.library import tieups
from core.library.db import LibraryDB
from core.library.records import TrackRow
from core.search.variants import Pair


def found(query: str, readings) -> bool:
    return bool(reading.find(reading.query_keys(query), [reading.keys(r) for r in readings]))


def bocchi() -> tieup.Work:
    return tieup.Work("", "anime", [("ja-JP", "ぼっち・ざ・ろっく！"), ("ko-KR", "봇치 더 록!"), ("en-US", "Bocchi the Rock!")],
                      [("synonym", "ja-JP", "ぼざろ"), ("search_hint", "en-US", "BTR")], [("Official", "https://e.org")])


class TieupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = LibraryDB(str(Path(self._tmp.name) / "library.db"))
        self.db.connect()
        self.uids = [str(uuid4()) for _ in range(4)]
        for i, uid in enumerate(self.uids):
            self.db.upsert(TrackRow(uid=uid, path=f"{i}.wav", title=f"t{i}", added_ts=1))
        self.db.conn.commit()

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _add(self, uid: str, work: tieup.Work | None = None, role: str = "opening", seq: int = 0,
             versions=()) -> tieup.TieUp:
        t = tieup.TieUp(work or bocchi(), role, seq)
        with self.db.conn:
            tieups.save_tieup(self.db.conn, uid, t, versions=versions)
        return t

    def test_found_by_titles_aliases_and_their_readings(self):
        self._add(self.uids[0], role="ending", seq=2)
        t = tieups.track_texts(self.db.conn)[self.uids[0]]
        for query in ("bocchi", "봇치", "보자로", "bozaro", "BTR", "rokku"):
            with self.subTest(query=query):
                self.assertTrue(found(query, t.readings))
        self.assertFalse(found("sakura", t.readings))
        for text in ("Ending", "Anime", "ぼざろ"):
            self.assertIn(text, t.text)
        self.assertNotIn("ぼざろ", tieups.comment_suffixes(self.db.conn)[self.uids[0]])   # never shown

    def test_title_in_the_interfaces_language(self):
        w = tieup.Work("", "anime", [("ja-JP", "日本"), ("en-GB", "British"), ("en-US", "American"), ("zh-TW", "臺")])
        self.assertEqual(w.title("en-US"), "American")    # language and region
        self.assertEqual(w.title("en-AU"), "British")     # the language in another region
        self.assertEqual(w.title("ko-KR"), "日本")         # none: the first
        self.assertEqual(w.title("zh-CN"), "臺")

    def test_titles_shown_in_the_language_set(self):
        w = tieup.Work("", "anime", [("ja-JP", "ご注文はうさぎですか？"), ("ko-KR", "주문은 토끼입니까?"),
                                     ("en-US", "Is the Order a Rabbit?")])
        try:
            tieup.set_display_language("ko-KR")       # Settings > Library's Tie-up Titles, whatever the UI's
            self.assertEqual(w.title(), "주문은 토끼입니까?")
            self.assertEqual(tieup.describe([(w, "opening", 1, "", [])]).shown, "주문은 토끼입니까? OP1")
            tieup.set_display_language("xx-XX")       # not one of TITLE_LANGUAGES: the UI's
            self.assertEqual(tieup.display_language(), tieup.i18n.language())
        finally:
            tieup.set_display_language("")

    def test_languages_are_language_region(self):
        self.assertTrue(tieup.is_lang("ko-KR"))
        for bad in ("ja", "ja-Latn", "zh-Hans", "", "KO-kr", "ko_KR"):
            self.assertFalse(tieup.is_lang(bad), bad)
        with self.assertRaises(ValueError):
            self._add(self.uids[0], tieup.Work("", "anime", [("ja", "x")]))
        with self.assertRaises(sqlite3.IntegrityError):    # the table itself refuses one
            self.db.conn.execute("INSERT INTO tieup_work(id) VALUES('w');")
            self.db.conn.execute("INSERT INTO tieup_name(work_id, name, lang) VALUES('w', 'x', 'ja');")
        self.db.conn.rollback()
        self.assertEqual([tieup.guess_lang(t) for t in ("あのね", "사랑", "東京", "Love")],
                         ["ja-JP", "ko-KR", "ja-JP", "en-US"])

    def test_comment_suffix_once(self):
        self._add(self.uids[0])
        suffix = tieups.comment_suffixes(self.db.conn)[self.uids[0]]
        self.assertEqual(suffix, "[Bocchi the Rock! OP]")   # the tests run in en-US
        self.assertEqual(tieup.with_suffix(tieup.with_suffix("c", suffix), suffix), "c " + suffix)
        self.assertEqual(tieup.with_suffix("", suffix), suffix)
        self.assertEqual(tieup.with_suffix("c", ""), "c")

    def test_comment_suffix_in_a_language_of_its_own(self):
        # the XML's and the Export tab's comments in their languages (Settings > Library), whatever the shown one's
        self._add(self.uids[0])
        try:
            tieup.set_display_language("ko-KR")
            self.assertEqual(tieups.comment_suffixes(self.db.conn)[self.uids[0]], "[봇치 더 록! OP]")
            self.assertEqual(tieups.comment_suffixes(self.db.conn, "ja-JP")[self.uids[0]], "[ぼっち・ざ・ろっく！ OP]")
            self.assertEqual(tieup.comment_suffix(tieups.track_tieups(self.db.conn, self.uids[0]), "en-US"),
                             "[Bocchi the Rock! OP]")
        finally:
            tieup.set_display_language("")

    def test_edit_by_its_old_key_and_remove(self):
        t = self._add(self.uids[0], seq=1)
        self._add(self.uids[0], t.work, role="ending")
        old = t.key()
        t.seq, t.detail = 2, "TV size"
        with self.db.conn:
            self.assertEqual(tieups.save_tieup(self.db.conn, self.uids[0], t, old=old), [self.uids[0]])
        got = tieups.track_tieups(self.db.conn, self.uids[0])
        self.assertEqual([(x.role, x.seq, x.detail) for x in got], [("opening", 2, "TV size"), ("ending", 0, "")])
        # edited onto another tie-up's key: one row
        t2 = got[1]
        old2 = t2.key()
        t2.role, t2.seq = "opening", 2
        with self.db.conn:
            tieups.save_tieup(self.db.conn, self.uids[0], t2, old=old2)
        self.assertEqual(len(tieups.track_tieups(self.db.conn, self.uids[0])), 1)
        with self.db.conn:
            tieups.delete_tieup(self.db.conn, self.uids[0], t2.key())
        self.assertEqual(tieups.track_tieups(self.db.conn, self.uids[0]), [])

    def test_an_addition_goes_to_the_versions_where_they_lack_it(self):
        a, b, c, _d = self.uids
        mine = self._add(b, role="ending", seq=0)
        mine.detail = "theirs"
        with self.db.conn:
            tieups.save_tieup(self.db.conn, b, tieup.TieUp(mine.work, "opening", 0, "kept"))
        t = tieup.TieUp(mine.work, "opening", 0, "added")
        with self.db.conn:
            written = tieups.save_tieup(self.db.conn, a, t, versions=[b, c, "not-a-track"])
        self.assertEqual(written, [a, c])                                  # b had it already
        self.assertEqual([x.detail for x in tieups.track_tieups(self.db.conn, b) if x.role == "opening"], ["kept"])
        # an edit stays on the track
        old = t.key()
        t.seq = 3
        with self.db.conn:
            self.assertEqual(tieups.save_tieup(self.db.conn, a, t, old=old, versions=[c]), [a])
        self.assertEqual([x.seq for x in tieups.track_tieups(self.db.conn, c)], [0])

    def test_a_new_track_takes_its_versions_tieups(self):
        a, b, c, d = self.uids
        self._add(a, role="opening")
        self._add(c, role="ending")
        pairs = [Pair(a, b, "vocal", 1.0, 1.0, 0.9), Pair(b, c, "same", 1.0, 1.0, 1.0)]
        self.assertEqual(tieup.versions_of(b, pairs), {a, c})
        with self.db.conn:
            self.assertEqual(tieups.match_new_track(self.db.conn, b, pairs), 2)
            self.assertEqual(tieups.match_new_track(self.db.conn, b, pairs), 0)      # it has tie-ups now
            self.assertEqual(tieups.match_new_track(self.db.conn, d, pairs), 0)      # no versions
        self.assertEqual(sorted(x.role for x in tieups.track_tieups(self.db.conn, b)), ["ending", "opening"])

    def test_a_work_found_by_any_name(self):
        self._add(self.uids[0])
        w = tieups.track_tieups(self.db.conn, self.uids[0])[0].work
        named = lambda *titles: tieup.Work("", "anime", [("ja-JP", t) for t in titles])  # noqa: E731
        self.assertEqual(tieups.find_work(self.db.conn, named("ぼっち・ざ・ろっく!")), w.id)   # ！ / ! alike
        self.assertEqual(tieups.find_work(self.db.conn, named("Bocchi the Rock")), w.id)
        self.assertIsNone(tieups.find_work(self.db.conn, named("Another")))
        # a short name or a search hint never: short names collide
        self.assertIsNone(tieups.find_work(self.db.conn, named("BTR")))
        self.assertEqual(tieups.find_work(self.db.conn, tieup.Work("", "anime", [("ja-JP", "X")],
                                                                  [("synonym", "ja-JP", "ぼざろ")])), w.id)
        # names added, never overwritten; each language's first title its primary
        with self.db.conn:
            tieups.add_names(self.db.conn, w.id, [("ko-KR", "봇치"), ("zh-TW", "孤獨搖滾！")], [], [("Wiki", "https://w.org")])
        again = tieups.work(self.db.conn, w.id)
        self.assertEqual(again.title("ko-KR"), "봇치 더 록!")
        self.assertEqual(again.title("zh-TW"), "孤獨搖滾！")
        self.assertEqual([u for _l, u in again.urls], ["https://e.org", "https://w.org"])

    def test_works_told_apart(self):
        a = tieup.Work("", "anime", [("ja-JP", "天使")], [], [("Wikidata", "https://www.wikidata.org/wiki/Q1")])
        self._add(self.uids[0], a)
        # two Han spellings read alike are two names (no reading involved)
        self.assertIsNone(tieups.find_work(self.db.conn, tieup.Work("", "anime", [("ja-JP", "添使")])))
        # the Wikidata item decides, whatever the names
        same = tieup.Work("", "anime", [("en-US", "Angel")], [], [("Wikidata", "https://www.wikidata.org/wiki/Q1")])
        self.assertEqual(tieups.find_work(self.db.conn, same), a.id)
        other = tieup.Work("", "anime", [("ja-JP", "天使")], [], [("Wikidata", "https://www.wikidata.org/wiki/Q2")])
        self.assertEqual(tieups.find_work(self.db.conn, other), a.id)        # no work holds Q2: by its name
        # two works with the name: none taken (the user merges)
        self._add(self.uids[1], tieup.Work("", "game", [("ja-JP", "天使")]))
        self.assertIsNone(tieups.find_work(self.db.conn, tieup.Work("", "anime", [("ja-JP", "天使")])))

    def test_a_work_edited_for_every_track_using_it(self):
        t = self._add(self.uids[0])
        self._add(self.uids[1], t.work)
        t.work.titles = [("ja-JP", "新しい題")]
        with self.db.conn:
            tieups.save_work(self.db.conn, t.work)
        self.assertEqual({x.work.title() for u in self.uids[:2] for x in tieups.track_tieups(self.db.conn, u)},
                         {"新しい題"})

    def test_works_merged_and_deleted(self):
        t = self._add(self.uids[0])                                   # the work, opening
        other = tieup.Work("", "anime", [("ja-JP", "ぼっちざろっく別名")], [], [("Wiki", "https://w.org")])
        self._add(self.uids[0], other, "opening")                     # the same work entered twice: one track has both
        self._add(self.uids[1], other, "ending")
        with self.db.conn:
            moved = tieups.merge_works(self.db.conn, t.work.id, other.id)
        self.assertEqual(sorted(moved), sorted(self.uids[:2]))
        self.assertIsNone(tieups.work(self.db.conn, other.id))
        kept = tieups.work(self.db.conn, t.work.id)
        self.assertIn("ぼっちざろっく別名", kept.names())
        self.assertIn("https://w.org", [u for _l, u in kept.urls])
        self.assertEqual([(x.work.id, x.role) for x in tieups.track_tieups(self.db.conn, self.uids[0])],
                         [(t.work.id, "opening")])                    # kept once
        self.assertEqual([(x.work.id, x.role) for x in tieups.track_tieups(self.db.conn, self.uids[1])],
                         [(t.work.id, "ending")])
        self.assertEqual(tieups.work_tracks(self.db.conn, t.work.id), sorted(self.uids[:2]))
        with self.db.conn:
            gone = tieups.delete_work(self.db.conn, t.work.id)
        self.assertEqual(sorted(gone), sorted(self.uids[:2]))
        self.assertEqual(tieups.catalog(self.db.conn).works, {})
        self.assertEqual(tieups.track_texts(self.db.conn), {})

    def test_gone_with_the_track(self):
        self._add(self.uids[0])
        self.db.delete_tracks([self.uids[0]])
        self.assertEqual(tieups.track_texts(self.db.conn), {})


if __name__ == "__main__":
    unittest.main()
