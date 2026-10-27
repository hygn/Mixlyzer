"""The library's parts agree: every track's uid is the one spelling, every track has its stored analysis and every
stored analysis its track, the stored files read without pickle, every row naming a track names one there."""
import re
import unittest

from core.analysis_library.stored import StoredAnalysis
from system.ids import canonical_uuid4

from . import _library as lib

# the tables whose rows name a track, and the column naming it (the library's schema; the ones there are checked)
TRACK_COLUMNS = {"playlist_tracks": "track_uid", "track_readings": "track_uid", "track_reading_removals": "track_uid",
                 "track_bpm_segments": "track_uid", "track_key_segments": "track_uid",
                 "track_phrase_sections": "track_uid", "track_tag_sources": "track_uid", "tieup": "track_uid"}


@lib.needs_library
class IntegrityTests(unittest.TestCase):
    def test_uids_are_canonical(self):
        problems = []
        for uid in lib.uids():
            try:
                if canonical_uuid4(uid) != uid:
                    problems.append((uid, "not in its canonical spelling"))
            except ValueError:
                problems.append((uid, "not a UUIDv4"))
        lib.report(self, problems, "uid")

    def test_every_track_has_its_analysis(self):
        folder = lib.library_dir()
        problems = [(uid, "no {uid}.npz") for uid in lib.uids() if not (folder / f"{uid}.npz").is_file()]
        lib.report(self, problems, "stored analysis missing")

    def test_no_orphan_analysis(self):
        """An NPZ named like a track's whose track is gone (a removal left half done)."""
        tracks = set(lib.uids())
        pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.npz$")
        orphans = [p.stem for p in lib.library_dir().glob("*.npz") if pattern.match(p.name) and p.stem not in tracks]
        self.assertEqual(len(orphans), 0, f"{len(orphans)} stored analyses without a track")

    def test_every_analysis_reads(self):
        stored = lib.stored()
        problems = [(uid, "does not read without pickle") for uid in lib.uids()
                    if (lib.library_dir() / f"{uid}.npz").is_file() and uid not in stored]
        for uid, arrays in stored.items():
            try:
                StoredAnalysis.from_arrays(arrays)
            except Exception as exc:                                             # its class only: no data
                problems.append((uid, f"StoredAnalysis.from_arrays: {type(exc).__name__}"))
        lib.report(self, problems, "stored analysis unreadable")

    def test_rows_name_tracks_there(self):
        tracks = set(lib.uids())
        found = []
        with lib.connect() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            for table, column in TRACK_COLUMNS.items():
                if table not in tables:
                    continue
                named = {r[0] for r in db.execute(f"SELECT DISTINCT {column} FROM {table}")}   # noqa: S608
                gone = len(named - tracks)
                if gone:
                    found.append(f"{table}: {gone} tracks not in the library")
            if "playlist_tracks" in tables:
                broken = db.execute("SELECT COUNT(*) FROM playlist_tracks e LEFT JOIN playlists p "
                                    "ON p.uid = e.playlist_uid WHERE p.uid IS NULL").fetchone()[0]
                if broken:
                    found.append(f"playlist_tracks: {broken} entries of no playlist")
                gaps = db.execute("SELECT COUNT(*) FROM (SELECT playlist_uid, COUNT(*) n, MIN(position) lo, "
                                  "MAX(position) hi FROM playlist_tracks GROUP BY playlist_uid) "
                                  "WHERE lo != 0 OR hi != n - 1").fetchone()[0]
                if gaps:
                    found.append(f"playlist_tracks: {gaps} playlists whose positions are not 0 .. n - 1")
        self.assertEqual(found, [])

    def test_one_track_one_file(self):
        with lib.connect() as db:
            paths = [r[0] for r in db.execute("SELECT path FROM tracks")]
        self.assertEqual(len(paths), len(set(paths)))
        folded = [p.replace("\\", "/").casefold() for p in paths]
        self.assertEqual(len(folded), len(set(folded)), "two tracks name one file in two spellings")


if __name__ == "__main__":
    unittest.main()
