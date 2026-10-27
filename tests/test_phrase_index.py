import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from core.library.db import LibraryDB, TrackRow
from core.library.segments import build_phrase_index_rows
from analyzer.music.key_names import idx_to_labels
from analyzer.music.marks import Phrase


class PhraseIndexBuilderTests(unittest.TestCase):
    def test_phrase_is_split_at_bpm_and_key_changes(self):
        phrases = [Phrase(0.0, 10.0, "CHORUS")]
        bpm = [
            {"seq_index": 0, "start_sec": 0.0, "end_sec": 4.0, "bpm": 120.0,
             "bpm_rounded": 120, "time_signature": 4},
            {"seq_index": 1, "start_sec": 4.0, "end_sec": 10.0, "bpm": 128.0,
             "bpm_rounded": 128, "time_signature": 4},
        ]
        keys = [
            {"seq_index": 0, "start_sec": 0.0, "end_sec": 7.0, "key_value": 0,
             "key_label": "8B"},
            {"seq_index": 1, "start_sec": 7.0, "end_sec": 10.0, "key_value": 1,
             "key_label": "3B"},
        ]

        rows = build_phrase_index_rows(phrases, bpm, keys)

        self.assertEqual([(row["start_sec"], row["end_sec"]) for row in rows],
                         [(0.0, 4.0), (4.0, 7.0), (7.0, 10.0)])
        self.assertEqual([row["bpm_rounded"] for row in rows], [120, 128, 128])
        self.assertEqual([row["key_value"] for row in rows], [0, 0, 1])
        self.assertTrue(all(row["phrase_seq_index"] == 0 for row in rows))
        self.assertTrue(all(row["phrase_label"] == "CHORUS" for row in rows))

    def test_missing_analysis_coverage_keeps_phrase_indexable(self):
        rows = build_phrase_index_rows(
            [Phrase(2.0, 6.0, "INTRO")], [], []
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["bpm"])
        self.assertIsNone(rows[0]["key_value"])


class PhraseIndexDatabaseTests(unittest.TestCase):
    def test_schema_replace_indexes_and_track_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LibraryDB(str(Path(tmp) / "library.db"))
            db.connect()
            try:
                uid = str(uuid4())
                db.upsert(TrackRow(path="song.wav", uid=uid, added_ts=1, duration=10.0))
                db.conn.commit()
                section = {
                    "phrase_seq_index": 0, "section_index": 0, "phrase_label": "INTRO",
                    "phrase_start_sec": 0.0, "phrase_end_sec": 10.0,
                    "phrase_duration_sec": 10.0, "start_sec": 0.0, "end_sec": 10.0,
                    "duration_sec": 10.0, "bpm_seq_index": 0, "bpm": 120.0,
                    "bpm_rounded": 120, "time_signature": 4, "key_seq_index": 0,
                    "key_value": 0, "key_label": "8B",
                }
                db.replace_phrase_index(uid, [section])
                row = db.conn.execute(
                    "SELECT * FROM track_phrase_sections WHERE track_uid=?", (uid,)
                ).fetchone()
                self.assertEqual(row["phrase_label"], "INTRO")
                self.assertEqual(row["bpm_rounded"], 120)
                self.assertEqual(row["key_value"], 0)

                index_names = {
                    row[0] for row in db.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' "
                        "AND tbl_name='track_phrase_sections'"
                    )
                }
                self.assertIn("idx_tps_label_bpm", index_names)
                self.assertIn("idx_tps_label_bpm_value", index_names)
                self.assertIn("idx_tps_label_key", index_names)

                db.delete_tracks([uid])
                count = db.conn.execute(
                    "SELECT COUNT(*) FROM track_phrase_sections WHERE track_uid=?", (uid,)
                ).fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                db.close()

    def test_phrase_search_combines_bpm_and_key_on_the_same_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LibraryDB(str(Path(tmp) / "library.db"))
            db.connect()
            try:
                uid_a, uid_b = str(uuid4()), str(uuid4())
                db.upsert(TrackRow(path="a.wav", uid=uid_a, title="A", added_ts=2))
                db.upsert(TrackRow(path="b.wav", uid=uid_b, title="B", added_ts=1))

                def section(index, bpm, key):
                    return {
                        "phrase_seq_index": 0,
                        "section_index": index,
                        "phrase_label": "CHORUS",
                        "phrase_start_sec": 0.0,
                        "phrase_end_sec": 16.0,
                        "phrase_duration_sec": 16.0,
                        "start_sec": index * 8.0,
                        "end_sec": (index + 1) * 8.0,
                        "duration_sec": 8.0,
                        "bpm_seq_index": index,
                        "bpm": bpm,
                        "bpm_rounded": round(bpm),
                        "time_signature": 4,
                        "key_seq_index": index,
                        "key_value": key,
                        "key_label": idx_to_labels(key)[0],
                    }

                db.replace_phrase_index(uid_a, [section(0, 128.0, 0)])
                # B contains both requested values, but never in one section.
                db.replace_phrase_index(uid_b, [
                    section(0, 128.0, 1),
                    section(1, 130.0, 0),
                ])

                phrase_only = db.search_phrase_sections("chorus")
                self.assertEqual({row.path for row in phrase_only}, {"a.wav", "b.wav"})

                match_all = db.search_phrase_sections(
                    "CHORUS", bpm_min=127.0, bpm_max=129.0, key_value=0,
                    match_all=True,
                )
                self.assertEqual([row.path for row in match_all], ["a.wav"])

                match_any = db.search_phrase_sections(
                    "CHORUS", bpm_min=127.0, bpm_max=129.0, key_value=0,
                    match_all=False,
                )
                self.assertEqual({row.path for row in match_any}, {"a.wav", "b.wav"})
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
