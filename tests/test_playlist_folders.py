"""The playlists' folders (core.library.db): a tree of folders and playlists in one order per folder, moved within a
folder and between folders (a folder never into itself), deleted with what is in them (the tracks stay); the
Rekordbox XML's PLAYLISTS written from it (third_party.rekordbox: folders Type 0, playlists Type 1 keyed by their
tracks' Location), by every sync and on its own when the playlists change (core.workflows.rekordbox_sync)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

from core.library.db import LibraryDB
from core.library.records import FOLDER_NODE, PLAYLIST_NODE, PlaylistTree, TrackRow, playlist_children
from core.library import jobs
from core.settings.config import default_cfg
from core.workers.rekordbox_sync import RekordboxSyncRequest, _RekordboxSyncEngine
from core.workflows import rekordbox_sync as sync_workflow
from third_party.rekordbox import RekordboxXmlDocument

APP = QApplication.instance() or QApplication([])


class FolderTreeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = LibraryDB(str(self.dir / "library.db"))
        self.db.connect()
        self.tracks = [str(uuid4()) for _ in range(3)]
        self.db.upsert_many([TrackRow(uid=u, path=str(self.dir / f"{i}.wav"), title=str(i), added_ts=i)
                             for i, u in enumerate(self.tracks)])

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def tree(self, folder=None):
        """A folder's children by name, a folder's as (name, its children)."""
        children = playlist_children(self.db.list_playlist_folders(), self.db.list_playlists())

        def walk(uid):
            return [(row.name, walk(row.uid)) if kind == FOLDER_NODE else row.name for kind, row in children.get(uid, ())]
        return walk(folder)

    def test_nested_folders_in_one_order(self):
        top = self.db.create_playlist("Top")
        events = self.db.create_playlist_folder("Events")
        summer = self.db.create_playlist_folder("Summer", events.uid)
        warm = self.db.create_playlist("Warm-up", events.uid)
        peak = self.db.create_playlist("Peak", summer.uid)
        self.assertEqual(self.tree(), ["Top", ("Events", [("Summer", ["Peak"]), "Warm-up"])])
        self.assertEqual({p.name: p.folder_uid for p in self.db.list_playlists()},
                         {"Top": None, "Warm-up": events.uid, "Peak": summer.uid})
        # moved into a folder before its child, then out to the top's end
        self.assertTrue(self.db.move_playlist_node(PLAYLIST_NODE, top.uid, events.uid, (FOLDER_NODE, summer.uid)))
        self.assertEqual(self.tree(), [("Events", ["Top", ("Summer", ["Peak"]), "Warm-up"])])
        self.assertTrue(self.db.move_playlist_node(PLAYLIST_NODE, peak.uid, None))
        self.assertEqual(self.tree(), [("Events", ["Top", ("Summer", []), "Warm-up"]), "Peak"])
        # moved among its folder's children; numbered 0.. within each folder
        self.assertTrue(self.db.move_playlist_node_by(PLAYLIST_NODE, warm.uid, -2))
        self.assertEqual(self.tree(events.uid), ["Warm-up", "Top", ("Summer", [])])
        self.assertFalse(self.db.move_playlist_node_by(PLAYLIST_NODE, warm.uid, -1))
        orders = sorted((p.folder_uid or "", p.sort_index) for p in self.db.list_playlists()) + \
            sorted((f.parent_uid or "", f.sort_index) for f in self.db.list_playlist_folders())
        self.assertEqual(sorted(orders), sorted([(events.uid, 0), (events.uid, 1), ("", 1), ("", 0), (events.uid, 2)]))

    def test_a_folder_never_into_itself(self):
        a = self.db.create_playlist_folder("A")
        b = self.db.create_playlist_folder("B", a.uid)
        self.assertFalse(self.db.move_playlist_node(FOLDER_NODE, a.uid, b.uid))
        self.assertFalse(self.db.move_playlist_node(FOLDER_NODE, a.uid, a.uid))
        self.assertEqual(self.tree(), [("A", [("B", [])])])
        self.assertTrue(self.db.move_playlist_node(FOLDER_NODE, b.uid, None, (FOLDER_NODE, a.uid)))
        self.assertEqual(self.tree(), [("B", []), ("A", [])])
        with self.assertRaises(ValueError):
            self.db.create_playlist("X", str(uuid4()))        # no such folder
        with self.assertRaises(ValueError):
            self.db.create_playlist_folder("  ")

    def test_folder_deleted_with_what_is_in_it(self):
        keep = self.db.create_playlist("Keep")
        sets = self.db.create_playlist_folder("Sets")
        inner = self.db.create_playlist_folder("Inner", sets.uid)
        gone = self.db.create_playlist("Gone", inner.uid)
        self.db.add_playlist_tracks(gone.uid, self.tracks)
        self.db.add_playlist_tracks(keep.uid, self.tracks[:1])
        after = self.db.create_playlist("After")
        self.db.delete_playlist_folder(sets.uid)
        self.assertEqual(self.tree(), ["Keep", "After"])
        self.assertEqual([p.sort_index for p in self.db.list_playlists()], [0, 1])
        self.assertEqual(self.db.list_playlist_folders(), [])
        self.assertNotIn(gone.uid, self.db.playlist_entries())
        self.assertEqual(len(self.db.list_all()), 3)            # the tracks stay
        self.assertEqual(self.db.playlist_entries()[keep.uid], ((0, self.tracks[0]),))
        self.db.rename_playlist(after.uid, "Later")
        self.db.delete_playlist(keep.uid)
        self.assertEqual([(p.name, p.sort_index) for p in self.db.list_playlists()], [("Later", 0)])

    def test_a_name_is_one_in_its_folder(self):
        a = self.db.create_playlist_folder("A")
        b = self.db.create_playlist_folder("B")
        in_a = self.db.create_playlist("Warm-up", a.uid)
        in_b = self.db.create_playlist("warm-UP", b.uid)          # another folder: another playlist
        top = self.db.create_playlist("Warm-up")
        with self.assertRaises(ValueError):
            self.db.create_playlist("WARM-UP", a.uid)               # the same folder, letter case aside
        with self.assertRaises(ValueError):
            self.db.move_playlist_node(PLAYLIST_NODE, in_b.uid, a.uid)
        other = self.db.create_playlist("Peak", a.uid)
        with self.assertRaises(ValueError):
            self.db.rename_playlist(other.uid, "warm-up")
        self.db.rename_playlist(in_a.uid, "Warm-Up")                # its own name in another case
        self.assertTrue(self.db.move_playlist_node_by(PLAYLIST_NODE, in_a.uid, 1))   # within its folder
        self.assertEqual(self.tree(), [("A", ["Peak", "Warm-Up"]), ("B", ["warm-UP"]), "Warm-up"])
        self.assertEqual(top.folder_uid, None)

    def test_rekordbox_xml_playlists(self):
        sets = self.db.create_playlist_folder("Sets")
        sub = self.db.create_playlist_folder("Empty", sets.uid)
        a = self.db.create_playlist("A", sets.uid)
        b = self.db.create_playlist("B")
        self.db.add_playlist_tracks(a.uid, [self.tracks[0], self.tracks[1], self.tracks[0]])
        self.db.add_playlist_tracks(b.uid, [self.tracks[2]])
        doc = RekordboxXmlDocument.new()
        for uid in self.tracks[:2]:      # the third not in COLLECTION: its entry left out
            doc.append_track(ET.Element("TRACK", TrackID="1", MixlyzerUID=uid, Location=f"file://localhost/{uid}.wav"))
        tree = PlaylistTree(tuple(self.db.list_playlist_folders()), tuple(self.db.list_playlists()),
                            self.db.playlist_entries())
        doc.set_playlists(tree)
        doc.set_playlists(tree)          # written again: replaced, not added
        root = ET.fromstring(doc.to_xml())
        self.assertEqual(len(root.findall("PLAYLISTS")), 1)
        top = root.find("PLAYLISTS/NODE")
        self.assertEqual((top.get("Type"), top.get("Name"), top.get("Count")), ("0", "ROOT", "2"))
        folder, playlist_b = list(top)
        self.assertEqual((folder.get("Type"), folder.get("Name"), folder.get("Count")), ("0", "Sets", "2"))
        empty, playlist_a = list(folder)
        self.assertEqual((empty.get("Name"), empty.get("Count")), ("Empty", "0"))
        self.assertEqual((playlist_a.get("Type"), playlist_a.get("KeyType"), playlist_a.get("Entries")), ("1", "1", "3"))
        self.assertEqual([t.get("Key") for t in playlist_a],
                         [f"file://localhost/{u}.wav" for u in (self.tracks[0], self.tracks[1], self.tracks[0])])
        self.assertEqual((playlist_b.get("Name"), playlist_b.get("Entries"), len(playlist_b)), ("B", "0", 0))
        self.assertEqual(sub.parent_uid, sets.uid)


class _Library(QtCore.QObject):
    """The library's owner as the XML sync sees it: its snapshot, its updates, its calls."""
    updated = QtCore.Signal(object)

    def __init__(self, snapshot, tree=None):
        super().__init__()
        self.snapshot = snapshot
        self.tree = tree

    def call(self, job, *args):
        if job is jobs.playlist_tree:
            return self.tree
        raise AssertionError(job)


class _Snapshot:
    def __init__(self, playlists=(), folders=(), entries=None):
        self.playlists, self.folders, self.entries = playlists, folders, entries if entries is not None else {}


class XmlSyncTests(unittest.TestCase):
    def test_playlists_changed_are_written_once_they_rest(self):
        first = _Snapshot()
        library = _Library(first)
        cfg = default_cfg()
        cfg.libconfig.rekordbox_sync_enabled = True
        sync = sync_workflow.RekordboxXmlSync(cfg_getter=lambda: cfg, library=library)
        asked = []
        sync._enqueue = asked.append
        library.updated.emit(_Snapshot(first.playlists, first.folders, first.entries))   # the same parts
        self.assertFalse(sync._playlists_timer.isActive())
        library.updated.emit(_Snapshot((), (), {}))     # read again unchanged (an edit's save, an analysis stored)
        self.assertFalse(sync._playlists_timer.isActive())
        library.updated.emit(_Snapshot((), (), {"p": ()}))                               # entries read again
        self.assertTrue(sync._playlists_timer.isActive())
        sync._playlists_timer.stop()
        sync.sync_playlists_requested()
        self.assertEqual([r.mode for r in asked], ["playlists"])
        cfg.libconfig.rekordbox_sync_enabled = False
        sync.sync_playlists_requested()
        self.assertEqual(len(asked), 1)

    def test_playlists_mode_writes_the_tree_into_the_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "rekordbox.xml"
            doc = RekordboxXmlDocument.new()
            uid = str(uuid4())
            doc.append_track(ET.Element("TRACK", TrackID="1", MixlyzerUID=uid, Location="file://localhost/x.wav"))
            xml.write_text(doc.to_xml(), encoding="utf-8")
            from core.library.records import PlaylistRow
            tree = PlaylistTree((), (PlaylistRow("p", "Set", 0, None),), {"p": ((0, uid),)})
            request = RekordboxSyncRequest(library_dir=Path(tmp), xml_path=xml, mode="playlists")
            result = _RekordboxSyncEngine(request, lambda *_a: None, _Library(_Snapshot(), tree)).run()
            self.assertEqual(result.entry_count, 1)
            root = ET.fromstring(xml.read_text(encoding="utf-8"))
            node = root.find("PLAYLISTS/NODE/NODE")
            self.assertEqual((node.get("Name"), [t.get("Key") for t in node]), ("Set", ["file://localhost/x.wav"]))
            self.assertEqual(len(root.findall("COLLECTION/TRACK")), 1)


if __name__ == "__main__":
    unittest.main()
