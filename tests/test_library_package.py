"""The library package (core.library.package): a library of the test's own packed and unpacked whole (library.db's
rows, the stored analyses, thumbnails, indexes, settings, user files, caches), its music files joined to a folder
they moved to, and every package refused that is not what this app makes: not a zip, another version, a name no
package has (a path out of the folder), a file whose bytes differ, a compressed member, a database holding code, a
folder not empty; a cancel leaves no file. The settings restored keep this device's audio and file paths."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtGui
from PySide6.QtWidgets import QApplication

from core.analysis_library.store import FeatureNPZStore
from core.library import package, thumbnails
from core.library.db import LibraryDB
from core.library.package import PackageError
from core.library.records import TrackRow
from core.library.service import Library
from core.library.version import CURRENT_LIBRARY_VERSION, write_library_version
from core.settings.config import default_cfg, restored_settings
from core.workflows.library_backup import _music_folders

APP = QApplication.instance() or QApplication([])


class _Fixture:
    """A library in a folder of the test's own: three tracks whose music files are in ``music/``, a playlist, the
    analyses, two thumbnails, an orphaned analysis, a user file, a feature cache, a tie-up cache."""

    def __init__(self, root: Path, *, trigger: bool = False) -> None:
        self.root = root
        self.libpath = root / "library"
        self.libpath.mkdir(parents=True)
        write_library_version(self.libpath, CURRENT_LIBRARY_VERSION)
        self.music = root / "music"
        (self.music / "A").mkdir(parents=True)
        (self.music / "B").mkdir(parents=True)
        db = LibraryDB(str(self.libpath / "library.db"))
        db.connect()
        self.lib = Library(db, str(self.libpath), FeatureNPZStore(base_dir=str(self.libpath), compressed=True))
        self.uids = []
        for i, rel in enumerate(("A/one.flac", "A/two.mp3", "B/三.wav")):
            path = self.music / rel
            path.write_bytes(bytes([i]) * (1000 + i))
            uid = str(uuid4())
            db.upsert(TrackRow(uid=uid, path=str(path), title=f"曲{i}", artist="歌手", added_ts=i,
                               file_size=path.stat().st_size))
            self.lib.features.save(uid, {"beats": np.arange(4.0) + i, "duration": 10.0 + i})
            self.uids.append(uid)
        playlist = db.create_playlist("セット")
        db.add_playlist_tracks(playlist.uid, [self.uids[0], self.uids[2], self.uids[0]])
        image = QtGui.QImage(8, 8, QtGui.QImage.Format_RGB32)
        image.fill(0x336699)
        for uid in self.uids[:2]:
            thumbnails.save_thumbnail(str(self.libpath), uid, image)
        orphan = str(uuid4())
        (self.libpath / package.ORPHANED).mkdir()
        (self.libpath / package.ORPHANED / f"{orphan}.npz").write_bytes(b"kept")
        (self.libpath / "stray.txt").write_text("not packed")
        if trigger:
            db.conn.execute("CREATE TRIGGER evil AFTER UPDATE ON tracks BEGIN DELETE FROM playlists; END")
            db.conn.commit()
        self.shortcuts = root / "shortcuts.json"
        self.shortcuts.write_text('{"deck.play": ["Z"]}', encoding="utf-8")
        self.cache = root / "featurecache" / "key"
        (self.cache / "train").mkdir(parents=True)
        (self.cache / "train" / "x.npz").write_bytes(b"cache")

    def plan(self, dest: Path, *, feature_cache: bool = True) -> package.ExportPlan:
        cfg = default_cfg()
        cfg.playbackconfig.output_device = "Their Speakers"
        cfg.viewconfig.language = "ja-JP"
        cfg.libconfig.local_music_directories = [str(self.music)]

        def tieup_cache(target: str) -> None:
            con = sqlite3.connect(target)
            con.execute("CREATE TABLE web_query (query_key TEXT PRIMARY KEY, fetched_at REAL, result_json TEXT)")
            con.execute("INSERT INTO web_query VALUES ('k', 1.0, '[]')")
            con.commit()
            con.close()
        return package.ExportPlan(
            dest=str(dest), settings=cfg.to_dict(), deck=cfg.deck_dict(),
            files={package.SHORTCUTS: str(self.shortcuts), package.MIDI: str(self.root / "no-midi.json")},
            trees={package.FEATURE_CACHE + "key_feature_cache_path": str(self.cache)} if feature_cache else {},
            copies={package.TIEUP_CACHE: tieup_cache})

    def close(self) -> None:
        self.lib.db.close()


def _rows(db_path: Path) -> dict:
    con = sqlite3.connect(db_path)
    try:
        return {"tracks": sorted(con.execute("SELECT uid, path, title, file_size FROM tracks")),
                "playlists": sorted(con.execute("SELECT uid, name FROM playlists")),
                "entries": sorted(con.execute("SELECT playlist_uid, position, track_uid FROM playlist_tracks"))}
    finally:
        con.close()


def _rewrite(src: Path, dest: Path, *, change=None, add=(), manifest=None, deflate=()) -> None:
    """A copy of the package ``src`` with a member's bytes changed (``change``: name -> bytes), members added, the
    manifest edited (``manifest(dict)``), members compressed."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
        for info in zin.infolist():
            data = zin.read(info)
            if info.filename == package.MANIFEST and manifest is not None:
                payload = json.loads(data)
                manifest(payload)
                data = json.dumps(payload).encode("utf-8")
            if change and info.filename in change:
                data = change[info.filename]
            kind = zipfile.ZIP_DEFLATED if info.filename in deflate else zipfile.ZIP_STORED
            zout.writestr(zipfile.ZipInfo(info.filename), data, compress_type=kind)
        for name, data in add:
            zout.writestr(zipfile.ZipInfo(name), data)


class PackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="mxlib-test-"))
        self.fx = _Fixture(self.tmp)
        self.pkg = self.tmp / "backup.mxlib"
        self.result = package.export(self.fx.lib, self.fx.plan(self.pkg))

    def tearDown(self) -> None:
        self.fx.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_round_trip_keeps_every_file_and_row(self):
        self.assertEqual(self.result.tracks, 3)
        self.assertFalse(Path(str(self.pkg) + ".partial").exists())
        self.assertEqual([p.name for p in self.tmp.iterdir() if p.name.startswith(".mxlib-")], [])
        info = package.read_package(str(self.pkg))
        self.assertEqual(info.tracks, 3)
        self.assertTrue(info.feature_cache)
        self.assertNotIn("library/stray.txt", info.files)
        dest = self.tmp / "restored"
        result = package.unpack_library(info, str(dest), {})
        self.assertEqual((result.tracks, result.moved, result.without_analysis), (3, 0, 0))
        self.assertFalse((self.tmp / "restored.partial").exists())
        self.assertEqual(_rows(dest / "library.db"), _rows(self.fx.libpath / "library.db"))
        for uid in self.fx.uids:
            self.assertEqual((dest / f"{uid}.npz").read_bytes(), (self.fx.libpath / f"{uid}.npz").read_bytes())
        for uid in self.fx.uids[:2]:
            self.assertEqual((dest / "thumbs" / f"{uid}.jpg").read_bytes(),
                             Path(thumbnails.thumbnail_path(str(self.fx.libpath), uid)).read_bytes())
        self.assertEqual(len(list((dest / package.ORPHANED).iterdir())), 1)
        self.assertEqual((dest / "VERSION").read_text().strip(), CURRENT_LIBRARY_VERSION)
        self.assertFalse((dest / "stray.txt").exists())
        # the settings, the user's file, the caches
        self.assertEqual(package.read_json(info, package.CONFIG)["viewconfig"]["language"], "ja-JP")
        self.assertFalse(info.has(package.MIDI))       # no file there: none packed
        target = self.tmp / "here" / "shortcuts.json"
        package.extract_file(info, package.SHORTCUTS, str(target))
        self.assertEqual(target.read_bytes(), self.fx.shortcuts.read_bytes())
        folder = self.tmp / "here" / "cache"
        self.assertEqual(package.extract_tree(info, package.FEATURE_CACHE + "key_feature_cache_path", str(folder)), 1)
        self.assertEqual((folder / "train" / "x.npz").read_bytes(), b"cache")
        self.assertTrue(info.has(package.TIEUP_CACHE))

    def test_feature_cache_left_out_unless_asked(self):
        pkg = self.tmp / "small.mxlib"
        package.export(self.fx.lib, self.fx.plan(pkg, feature_cache=False))
        info = package.read_package(str(pkg))
        self.assertFalse(info.feature_cache)
        self.assertEqual(info.names(package.FEATURE_CACHE), [])

    def test_music_moved_to_another_folder_is_joined(self):
        info = package.read_package(str(self.pkg))
        roots = package.path_roots(p for p, _s in info.paths)
        self.assertEqual([(r.path, r.tracks) for r in roots], [(str(self.fx.music), 3)])
        moved = self.tmp / "elsewhere" / "Music"
        shutil.move(str(self.fx.music), str(moved))
        (moved / "A" / "two.mp3").write_bytes(b"another size")
        (moved / "B" / "三.wav").unlink()
        mapping = {roots[0].path: str(moved)}
        check = package.check_mapping(info.paths, mapping)[roots[0].path]
        self.assertEqual((check.found, check.differs, check.missing), (1, 1, 1))
        dest = self.tmp / "restored"
        result = package.unpack_library(info, str(dest), mapping)
        self.assertEqual(result.moved, 3)
        paths = sorted(p for _u, p, _t, _s in _rows(dest / "library.db")["tracks"])
        self.assertEqual(paths, sorted(str(moved / rel) for rel in ("A/one.flac", "A/two.mp3", "B/三.wav")))
        self.assertEqual(_music_folders([str(self.fx.music)], mapping), [str(moved)])

    def test_same_folder_keeps_the_paths_as_stored(self):
        info = package.read_package(str(self.pkg))
        root = package.path_roots(p for p, _s in info.paths)[0].path
        dest = self.tmp / "restored"
        result = package.unpack_library(info, str(dest), {root: root})
        self.assertEqual(result.moved, 0)
        self.assertEqual(_rows(dest / "library.db")["tracks"], _rows(self.fx.libpath / "library.db")["tracks"])

    def test_cancel_leaves_no_file(self):
        pkg = self.tmp / "cancelled.mxlib"
        with self.assertRaises(PackageError) as caught:
            package.export(self.fx.lib, self.fx.plan(pkg), cancelled=lambda: True)
        self.assertEqual(caught.exception.reason, PackageError.CANCELLED)
        self.assertFalse(pkg.exists())
        self.assertFalse(Path(str(pkg) + ".partial").exists())
        self.assertEqual([p.name for p in self.tmp.iterdir() if p.name.startswith(".mxlib-")], [])

    # ---- refusals ----
    def _refused(self, path: Path, reason: str, *, unpack: bool = False) -> None:
        dest = self.tmp / "restored"
        with self.assertRaises(PackageError) as caught:
            info = package.read_package(str(path))
            if unpack:
                package.unpack_library(info, str(dest), {})
        self.assertEqual(caught.exception.reason, reason, caught.exception.detail)
        self.assertFalse(dest.exists())
        self.assertFalse((self.tmp / "restored.partial").exists())

    def test_not_a_package(self):
        bad = self.tmp / "bad.mxlib"
        bad.write_bytes(b"not a zip")
        self._refused(bad, PackageError.NOT_A_PACKAGE)
        plain = self.tmp / "plain.mxlib"
        with zipfile.ZipFile(plain, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"format": "something else"}))
        self._refused(plain, PackageError.NOT_A_PACKAGE)

    def test_another_library_version(self):
        other = self.tmp / "other.mxlib"
        _rewrite(self.pkg, other, manifest=lambda m: m.update(library_version="0.0.1"))
        self._refused(other, PackageError.LIBRARY_VERSION)
        newer = self.tmp / "newer.mxlib"
        _rewrite(self.pkg, newer, manifest=lambda m: m.update(format_version=package.FORMAT_VERSION + 1))
        self._refused(newer, PackageError.NEWER_FORMAT)

    def test_names_out_of_the_folder(self):
        for name in ("../evil.txt", "library/../../evil.npz", "/etc/evil", "C:/evil", "library\\evil.npz",
                     "featurecache/key_feature_cache_path/../x", "user/models/../../config.json"):
            self.assertFalse(package.allowed(name), name)
            listed = self.tmp / "listed.mxlib"

            def add(m, name=name):
                m["files"][name] = {"size": 4, "sha256": hashlib.sha256(b"evil").hexdigest()}
            _rewrite(self.pkg, listed, add=[(name, b"evil")], manifest=add)
            self._refused(listed, PackageError.UNEXPECTED)
            unlisted = self.tmp / "unlisted.mxlib"
            _rewrite(self.pkg, unlisted, add=[(name, b"evil")])
            self._refused(unlisted, PackageError.UNEXPECTED)

    def test_bytes_changed_refused_at_unpack(self):
        name = f"library/{self.fx.uids[0]}.npz"
        data = (self.fx.libpath / f"{self.fx.uids[0]}.npz").read_bytes()
        changed = self.tmp / "changed.mxlib"
        _rewrite(self.pkg, changed, change={name: bytes(b ^ 1 for b in data)})
        self._refused(changed, PackageError.DAMAGED, unpack=True)
        shorter = self.tmp / "shorter.mxlib"
        _rewrite(self.pkg, shorter, change={name: data[:-1]})
        self._refused(shorter, PackageError.DAMAGED)

    def test_compressed_member_refused(self):
        deflated = self.tmp / "deflated.mxlib"
        _rewrite(self.pkg, deflated, deflate={package.LIBRARY_DB})
        self._refused(deflated, PackageError.DAMAGED)

    def test_database_with_code_refused(self):
        fx = _Fixture(self.tmp / "evil", trigger=True)
        try:
            pkg = self.tmp / "evil.mxlib"
            package.export(fx.lib, fx.plan(pkg))
        finally:
            fx.close()
        self._refused(pkg, PackageError.SCHEMA, unpack=True)

    def test_folder_not_empty_refused(self):
        dest = self.tmp / "full"
        dest.mkdir()
        (dest / "something").write_text("mine")
        info = package.read_package(str(self.pkg))
        with self.assertRaises(PackageError) as caught:
            package.unpack_library(info, str(dest), {})
        self.assertEqual(caught.exception.reason, PackageError.DESTINATION)
        self.assertEqual((dest / "something").read_text(), "mine")


class PathTests(unittest.TestCase):
    def test_roots_and_remap_of_another_system(self):
        paths = ["D:\\Music\\Anison\\a.flac", "d:\\music\\Doujin\\b.flac", "D:\\Other\\c.mp3",
                 "/home/u/Music/x/d.mp3", "/home/u/Music/y/e.mp3"]
        roots = {r.path: r.tracks for r in package.path_roots(paths)}
        self.assertEqual(roots, {"D:\\Music": 2, "D:\\Other": 1, "/home/u/Music": 2})
        mapping = {"D:\\Music": os.path.join("mnt", "music")}
        self.assertEqual(package.remap_path("d:\\MUSIC\\Anison\\a.flac", mapping),
                         os.path.join("mnt", "music", "Anison", "a.flac"))
        self.assertIsNone(package.remap_path("D:\\Other\\c.mp3", mapping))
        self.assertIsNone(package.remap_path("/home/u/Music/x/d.mp3", mapping))
        # a POSIX path is matched in its own letter case
        self.assertIsNone(package.remap_path("/home/u/music/x/d.mp3", {"/home/u/Music": "/m"}))


class RestoredSettingsTests(unittest.TestCase):
    def test_device_and_paths_kept(self):
        here = default_cfg()
        here.playbackconfig.output_device = "My Speakers"
        here.analysisconfig.beat_tcn_parameter_path = "assets/weights/mine.npz"
        here.libconfig.logpath = "here.log"
        here.deckconfig.music_input_device = "My Mic"
        theirs = default_cfg()
        theirs.playbackconfig.output_device = "Their Speakers"
        theirs.analysisconfig.beat_tcn_parameter_path = "C:/elsewhere/evil.npz"
        theirs.libconfig.logpath = "C:/Windows/evil.log"
        theirs.viewconfig.language = "ja-JP"
        theirs.deckconfig.tempo_range = 16
        theirs.deckconfig.music_input_device = "Their Mic"
        new = restored_settings(here, theirs.to_dict(), theirs.deck_dict(), libpath="restored",
                                local_music_directories=["/m"], rekordbox_xml_path="x.xml")
        self.assertEqual(new.playbackconfig.output_device, "My Speakers")
        self.assertEqual(new.analysisconfig.beat_tcn_parameter_path, "assets/weights/mine.npz")
        self.assertEqual(new.libconfig.logpath, "here.log")
        self.assertEqual(new.deckconfig.music_input_device, "My Mic")
        self.assertEqual(new.viewconfig.language, "ja-JP")
        self.assertEqual(new.deckconfig.tempo_range, 16)
        self.assertEqual((new.libconfig.libpath, new.libconfig.local_music_directories,
                          new.libconfig.rekordbox_xml_path), ("restored", ["/m"], "x.xml"))


if __name__ == "__main__":
    unittest.main()
