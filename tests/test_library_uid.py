import sqlite3
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from core.library.db import LibraryDB, TrackRow


class UidKeyedLibraryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = LibraryDB(str(self.dir / "library.db"))
        self.db.connect()
        self.a, self.b = str(uuid4()), str(uuid4())
        self.db.upsert_many([
            TrackRow(uid=self.a, path=str(self.dir / "a.wav"), title="A", added_ts=1),
            TrackRow(uid=self.b, path=str(self.dir / "b.wav"), title="B", added_ts=2),
        ])

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def entries(self, playlist_uid):
        return [row.uid for _position, row in self.db.list_playlist_tracks(playlist_uid)]

    def test_playlist_holds_a_track_more_than_once(self):
        playlist = self.db.create_playlist("Set")
        self.assertEqual(self.db.add_playlist_tracks(playlist.uid, [self.a, self.b, self.a]), 3)
        self.assertEqual(self.entries(playlist.uid), [self.a, self.b, self.a])
        self.assertEqual([p for p, _row in self.db.list_playlist_tracks(playlist.uid)], [0, 1, 2])

        # the last entry (the second A) moves to the front; the first A stays where it was
        self.assertTrue(self.db.move_playlist_tracks_before(playlist.uid, [2], 0))
        self.assertEqual(self.entries(playlist.uid), [self.a, self.a, self.b])
        self.assertTrue(self.db.move_playlist_tracks_before(playlist.uid, [0], None))
        self.assertEqual(self.entries(playlist.uid), [self.a, self.b, self.a])

        self.db.remove_playlist_tracks(playlist.uid, [0])
        self.assertEqual(self.entries(playlist.uid), [self.b, self.a])
        self.assertEqual([p for p, _row in self.db.list_playlist_tracks(playlist.uid)], [0, 1])

    def test_deleting_a_track_closes_up_its_playlists(self):
        playlist = self.db.create_playlist("Set")
        self.db.add_playlist_tracks(playlist.uid, [self.a, self.b, self.a, self.b])
        self.db.delete_tracks([self.a])
        self.assertIsNone(self.db.get_by_uid(self.a))
        self.assertEqual(self.entries(playlist.uid), [self.b, self.b])
        self.assertEqual([p for p, _row in self.db.list_playlist_tracks(playlist.uid)], [0, 1])

    def test_path_is_an_editable_attribute(self):
        playlist = self.db.create_playlist("Set")
        self.db.add_playlist_tracks(playlist.uid, [self.a])
        moved = self.dir / "moved.wav"
        moved.write_bytes(b"x" * 10)
        self.db.set_path(self.a, str(moved))
        row = self.db.get_by_uid(self.a)
        self.assertEqual(row.path, str(moved))
        self.assertEqual(row.file_size, 10)
        self.assertEqual(self.db.get(str(moved)).uid, self.a)
        self.assertEqual(self.entries(playlist.uid), [self.a])

        # one file, one track
        with self.assertRaises(ValueError):
            self.db.set_path(self.b, str(moved))
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.upsert(TrackRow(uid=self.b, path=str(moved), added_ts=2))
        self.assertEqual(self.db.get_by_uid(self.b).path, str(self.dir / "b.wav"))

    def test_playlists_by_uid(self):
        first = self.db.create_playlist("One")
        second = self.db.create_playlist("Two")
        self.assertTrue(self.db.move_playlist_node("playlist", second.uid, None, ("playlist", first.uid)))
        self.assertEqual([p.uid for p in self.db.list_playlists()], [second.uid, first.uid])
        self.db.rename_playlist(first.uid, "First")
        self.db.delete_playlist(second.uid)
        playlists = self.db.list_playlists()
        self.assertEqual([(p.uid, p.name, p.sort_index) for p in playlists], [(first.uid, "First", 0)])


if __name__ == "__main__":
    unittest.main()
