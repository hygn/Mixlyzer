"""The library package (.mxlib): the library and what goes with it (the settings, the user's shortcuts, MIDI mapping and
retrained models, the caches) in one file, for a backup and for a move to another device: unpacked on the same device
it is a restore, on another a move (docs: core/library/AGENTS.md, Backup and moving).

    export      jobs.export_package(lib, plan, ...)       a job of the library's owner: library.db and the stored
                                                          analyses of one moment
    read        read_package(path) -> PackageInfo         every name, size and the library version checked
    unpack      unpack_library(info, folder, mapping)     into a new folder (never the library open), each file's
                                                          SHA-256 and the databases' schema checked, the music
                                                          files' paths rewritten under the roots the user mapped
    the rest    read_json / extract_file / extract_tree   the settings and the user's files, by the restore

A zip, stored (the analyses and thumbnails are compressed already), its names only the fixed ones and patterns below:
nothing is written where a name in the package says (a package may come from elsewhere). A folder being unpacked is
no library yet: its files are the unpacker's until the app is started on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Iterable, Optional

from core.library import music_index, thumbnails, variants
from core.library.version import CURRENT_LIBRARY_VERSION, VERSION_FILENAME, write_library_version
from system import os_support
from system.atomic_io import atomic_output_path

FORMAT = "mixlyzer-library"
FORMAT_VERSION = 1
SUFFIX = ".mxlib"
MANIFEST = "manifest.json"
JSON_MAX = 64 << 20          # a JSON member read at most (the manifest lists every track's path)
CHUNK = 1 << 20
ORPHANED = "orphaned_analyses"     # core.library.jobs.ORPHANED_ANALYSES

LIBRARY_DB = "library/library.db"
MUSIC_DB = "library/" + music_index.FILE_NAME
VARIANTS_DB = "library/" + variants.FILE_NAME
CONFIG = "settings/config.json"
DECK = "settings/deck.json"
SHORTCUTS = "user/shortcuts.json"
MIDI = "user/midi.json"
TIEUP_CACHE = "cache/tieup_web.db"
MODELS = "user/models/"            # + an analysisconfig setting's name (*_parameter_path)
FEATURE_CACHE = "featurecache/"    # + an analysisconfig setting's name (*_feature_cache_path) + the file's own path

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_FIXED = frozenset({LIBRARY_DB, "library/" + VERSION_FILENAME, MUSIC_DB, VARIANTS_DB, CONFIG, DECK, SHORTCUTS, MIDI,
                    TIEUP_CACHE})
_PATTERNS = (
    re.compile(rf"library/{_UUID}\.npz"),
    re.compile(rf"library/{ORPHANED}/{_UUID}\.npz"),
    re.compile(rf"library/{thumbnails.THUMBS_DIR}/{_UUID}\.jpg"),
    re.compile(r"user/models/[a-z][a-z0-9_]{0,63}_parameter_path"),
    # a cache's own files: each part a plain name (no "..", no drive, no separator), at most 8 deep
    re.compile(r"featurecache/[a-z][a-z0-9_]{0,63}_feature_cache_path(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127}){1,8}"),
)
_DATABASES = (LIBRARY_DB, MUSIC_DB, VARIANTS_DB)


def allowed(name: str) -> bool:
    """Whether ``name`` may be in a package."""
    return name in _FIXED or any(p.fullmatch(name) for p in _PATTERNS)


class PackageError(Exception):
    """A package refused or a step that failed: ``reason`` one of the codes below (worded by the interface),
    ``detail`` for the log."""
    NOT_A_PACKAGE = "not_a_package"      # not a zip, or no manifest of this format
    DAMAGED = "damaged"                  # its manifest and its files disagree, a file's hash differs
    NEWER_FORMAT = "newer_format"        # made by a later version of the format
    LIBRARY_VERSION = "library_version"  # its library of another version than this app's
    UNEXPECTED = "unexpected"            # a name no package has
    SCHEMA = "schema"                    # a database not of the app's own schema, or damaged
    DESTINATION = "destination"          # the folder to unpack into is not new and empty
    CANCELLED = "cancelled"

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


Progress = Optional[Callable[[float], None]]      # the share done, 0 .. 1
Cancelled = Optional[Callable[[], bool]]


# ---- export ----

@dataclass(frozen=True)
class ExportPlan:
    """What goes into a package beside the library (made by the GUI from the settings, written by the library's
    owner): ``settings`` / ``deck`` config.json's and deck.json's content; ``files`` a name in the package -> a file on
    this device; ``trees`` a name prefix (FEATURE_CACHE + a setting) -> a folder, its files under it; ``copies`` a
    name -> a function writing a consistent copy of a database to the path it is given (the tie-up cache's)."""
    dest: str
    settings: dict
    deck: dict
    files: dict = field(default_factory=dict)
    trees: dict = field(default_factory=dict)
    copies: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExportResult:
    path: str
    size: int
    tracks: int
    files: int


def export(lib, plan: ExportPlan, progress: Progress = None, cancelled: Cancelled = None) -> ExportResult:
    """The package ``plan.dest`` written from the library ``lib`` (core.library.service.Library, on its owner's
    thread: no other job runs meanwhile, so library.db and the stored analyses are of one moment). Written to
    ``<dest>.partial`` and renamed at its end; a failure or a cancel leaves no file."""
    dest = Path(plan.dest)
    partial = dest.with_name(dest.name + ".partial")
    scratch = Path(tempfile.mkdtemp(prefix=".mxlib-", dir=str(dest.parent)))
    made: list[Path] = []
    try:
        entries = _entries(lib, plan, scratch, made)
        total = max(1, sum(os.path.getsize(src) for _name, src in entries))
        tracks = lib.db.list_all()
        files: dict[str, dict] = {}
        done = 0
        last = -1
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name, src in entries:
                info = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
                digest, size = hashlib.sha256(), 0
                with open(src, "rb") as fin, zf.open(info, "w", force_zip64=True) as fout:
                    while True:
                        if cancelled is not None and cancelled():
                            raise PackageError(PackageError.CANCELLED)
                        block = fin.read(CHUNK)
                        if not block:
                            break
                        fout.write(block)
                        digest.update(block)
                        size += len(block)
                        done += len(block)
                        share = int(100 * done / total)
                        if progress is not None and share != last:
                            last = share
                            progress(min(1.0, done / total))
                files[name] = {"size": size, "sha256": digest.hexdigest()}
            manifest = {
                "format": FORMAT, "format_version": FORMAT_VERSION, "library_version": CURRENT_LIBRARY_VERSION,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "os": os_support.OS,
                "tracks": len(tracks), "feature_cache": bool(plan.trees),
                "paths": [[t.path, int(t.file_size or 0)] for t in tracks],
                "files": files,
            }
            zf.writestr(zipfile.ZipInfo(MANIFEST, date_time=time.localtime()[:6]),
                        json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
        os.replace(partial, dest)
        return ExportResult(str(dest), os.path.getsize(dest), len(tracks), len(files))
    except BaseException:
        _remove(partial)
        raise
    finally:
        for path in made:
            _remove(path)
        try:
            scratch.rmdir()
        except OSError:
            pass


def _entries(lib, plan: ExportPlan, scratch: Path, made: list[Path]) -> list[tuple[str, str]]:
    """(name in the package, file to copy) of everything packed: the databases first copied into ``scratch`` through
    SQLite's backup API (each copy listed in ``made``, removed after)."""
    entries: list[tuple[str, str]] = []

    def scratch_file(name: str) -> Path:
        path = scratch / name.replace("/", "__")
        made.append(path)
        return path

    def backup(source: sqlite3.Connection, name: str) -> None:
        target = scratch_file(name)
        dst = sqlite3.connect(target)
        try:
            source.backup(dst)
        finally:
            dst.close()
        entries.append((name, str(target)))

    backup(lib.db.conn, LIBRARY_DB)
    for name, path in ((MUSIC_DB, music_index.path_for(lib.path)), (VARIANTS_DB, variants.path_for(lib.path))):
        if os.path.isfile(path):
            src = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
            try:
                backup(src, name)
            finally:
                src.close()
    version = os.path.join(lib.path, VERSION_FILENAME)
    if os.path.isfile(version):
        entries.append(("library/" + VERSION_FILENAME, version))
    for uid in sorted(lib.features.uids()):
        entries.append((f"library/{uid}.npz", lib.features.path(uid)))
    entries += _named_files(os.path.join(lib.path, ORPHANED), f"library/{ORPHANED}/")
    entries += _named_files(os.path.join(lib.path, thumbnails.THUMBS_DIR), f"library/{thumbnails.THUMBS_DIR}/")
    for name, write in plan.copies.items():
        target = scratch_file(name)
        write(str(target))
        if target.is_file():
            entries.append((name, str(target)))
    for name, payload in ((CONFIG, plan.settings), (DECK, plan.deck)):
        target = scratch_file(name)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        entries.append((name, str(target)))
    for name, path in plan.files.items():
        if os.path.isfile(path):
            entries.append((name, str(path)))
    for prefix, folder in plan.trees.items():
        entries += _tree_files(folder, prefix.rstrip("/") + "/")
    bad = [name for name, _src in entries if not allowed(name)]
    if bad:
        raise ValueError(f"names no package may hold: {bad[:5]}")
    return entries


def _named_files(folder: str, prefix: str) -> list[tuple[str, str]]:
    """The files of ``folder`` (not below it) whose name in the package is allowed: a stray file is left out."""
    out = []
    try:
        with os.scandir(folder) as found:
            for entry in sorted(found, key=lambda e: e.name):
                if entry.is_file(follow_symlinks=False) and allowed(prefix + entry.name):
                    out.append((prefix + entry.name, entry.path))
    except FileNotFoundError:
        pass
    return out


def _tree_files(folder: str, prefix: str) -> list[tuple[str, str]]:
    """Every file under ``folder`` (links not followed) whose name in the package is allowed."""
    out = []
    root = Path(folder)
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not os.path.islink(os.path.join(base, d)))
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink() or not path.is_file():
                continue
            packed = prefix + path.relative_to(root).as_posix()
            if allowed(packed):
                out.append((packed, str(path)))
    return out


# ---- reading ----

@dataclass(frozen=True)
class PackageInfo:
    """A package read and checked (its files not yet): where it is, what its manifest says."""
    path: str
    created: str
    os: str
    tracks: int
    feature_cache: bool
    size: int
    paths: tuple            # ((the track's path as stored, its file's size), ...)
    files: dict             # name -> (size, sha256)

    def has(self, name: str) -> bool:
        return name in self.files

    def names(self, prefix: str) -> list[str]:
        return sorted(n for n in self.files if n.startswith(prefix))


def read_package(path: str) -> PackageInfo:
    """The package at ``path`` read and checked: a zip of this format and library version, its names all allowed,
    each file stored and of the manifest's size. PackageError otherwise."""
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise PackageError(PackageError.NOT_A_PACKAGE, str(exc)) from exc
    with zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        if MANIFEST not in names:
            raise PackageError(PackageError.NOT_A_PACKAGE, "no manifest")
        if len(set(names)) != len(names):
            raise PackageError(PackageError.DAMAGED, "a name twice")
        manifest = _json_member(zf, zf.getinfo(MANIFEST))
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise PackageError(PackageError.NOT_A_PACKAGE, "not a library package")
        if not isinstance(manifest.get("format_version"), int) or manifest["format_version"] > FORMAT_VERSION:
            raise PackageError(PackageError.NEWER_FORMAT, str(manifest.get("format_version")))
        version = str(manifest.get("library_version") or "")
        if version != CURRENT_LIBRARY_VERSION:
            raise PackageError(PackageError.LIBRARY_VERSION, f"{version} (this app: {CURRENT_LIBRARY_VERSION})")
        listed = manifest.get("files")
        if not isinstance(listed, dict):
            raise PackageError(PackageError.DAMAGED, "no file list")
        files: dict[str, tuple[int, str]] = {}
        for name, entry in listed.items():
            if not isinstance(name, str) or not allowed(name):
                raise PackageError(PackageError.UNEXPECTED, repr(name)[:200])
            size = entry.get("size") if isinstance(entry, dict) else None
            digest = entry.get("sha256") if isinstance(entry, dict) else None
            if not isinstance(size, int) or size < 0 or not isinstance(digest, str) \
                    or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise PackageError(PackageError.DAMAGED, f"{name}: no size or hash")
            files[name] = (size, digest)
        for info in infos:
            if info.filename == MANIFEST:
                continue
            if info.filename not in files:
                raise PackageError(PackageError.UNEXPECTED, repr(info.filename)[:200])
            if info.is_dir() or info.compress_type != zipfile.ZIP_STORED or info.file_size != files[info.filename][0]:
                raise PackageError(PackageError.DAMAGED, info.filename)
        missing = set(files) - set(names)
        if missing or LIBRARY_DB not in files:
            raise PackageError(PackageError.DAMAGED, f"missing: {sorted(missing | {LIBRARY_DB} - set(files))[:5]}")
        paths = []
        for item in manifest.get("paths") or ():
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], int):
                paths.append((item[0], item[1]))
        tracks = manifest.get("tracks")
        return PackageInfo(
            path=str(path), created=str(manifest.get("created") or ""), os=str(manifest.get("os") or ""),
            tracks=tracks if isinstance(tracks, int) and tracks >= 0 else len(paths),
            feature_cache=bool(manifest.get("feature_cache")), size=os.path.getsize(path), paths=tuple(paths),
            files=files)


def _json_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo):
    if info.file_size > JSON_MAX or info.compress_type != zipfile.ZIP_STORED:
        raise PackageError(PackageError.DAMAGED, f"{info.filename}: too large")
    try:
        return json.loads(zf.read(info).decode("utf-8"))
    except (ValueError, UnicodeError, zipfile.BadZipFile, OSError) as exc:
        raise PackageError(PackageError.DAMAGED, f"{info.filename}: {exc}") from exc


def read_json(info: PackageInfo, name: str) -> dict:
    """A JSON file of the package (its hash checked); {} when the package has none, or it is no JSON object."""
    if not info.has(name):
        return {}
    with zipfile.ZipFile(info.path) as zf:
        data = _read_checked(zf, info, name)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_checked(zf: zipfile.ZipFile, info: PackageInfo, name: str) -> bytes:
    size, digest = info.files[name]
    if size > JSON_MAX:
        raise PackageError(PackageError.DAMAGED, f"{name}: too large")
    data = zf.read(name)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise PackageError(PackageError.DAMAGED, f"{name}: hash")
    return data


def _copy_checked(zf: zipfile.ZipFile, info: PackageInfo, name: str, target: Path,
                  advance: Callable[[int], None] = lambda _n: None, cancelled: Cancelled = None) -> None:
    """The package's file ``name`` written to ``target`` (a new file), its size and hash checked as it is copied."""
    size, digest = info.files[name]
    got, written = hashlib.sha256(), 0
    with zf.open(name) as fin, open(target, "xb") as fout:
        while True:
            if cancelled is not None and cancelled():
                raise PackageError(PackageError.CANCELLED)
            block = fin.read(CHUNK)
            if not block:
                break
            written += len(block)
            if written > size:
                break
            fout.write(block)
            got.update(block)
            advance(len(block))
    if written != size or got.hexdigest() != digest:
        raise PackageError(PackageError.DAMAGED, f"{name}: hash")


def extract_file(info: PackageInfo, name: str, target: str) -> None:
    """The package's file ``name`` (hash checked) put at ``target``, replacing it whole."""
    with zipfile.ZipFile(info.path) as zf, atomic_output_path(target) as temporary:
        temporary.unlink()
        _copy_checked(zf, info, name, temporary)


def extract_tree(info: PackageInfo, prefix: str, folder: str, cancelled: Cancelled = None) -> int:
    """The package's files under ``prefix`` put into ``folder`` (each replaced whole, the folder's other files kept);
    how many."""
    prefix = prefix.rstrip("/") + "/"
    count = 0
    with zipfile.ZipFile(info.path) as zf:
        for name in info.names(prefix):
            target = Path(folder).joinpath(*name[len(prefix):].split("/"))
            with atomic_output_path(target) as temporary:
                temporary.unlink()
                _copy_checked(zf, info, name, temporary, cancelled=cancelled)
            count += 1
    return count


# ---- the music files' folders ----

@dataclass(frozen=True)
class Root:
    """A folder the library's music files are under, as the stored paths spell it, and how many tracks it holds."""
    path: str
    tracks: int


def _pure(path: str):
    """The path in the spelling of the system that stored it (a drive or UNC path: Windows')."""
    return PureWindowsPath(path) if re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", path) else PurePosixPath(path)


def _key(part: str, pure) -> str:
    return part.casefold() if isinstance(pure, PureWindowsPath) else part


def path_roots(paths: Iterable[str]) -> list[Root]:
    """The folders the files are under: the files grouped by their drive (or root) and first folder, each group's
    common folder; the most tracks first."""
    groups: dict[tuple, list] = {}
    for path in paths:
        pure = _pure(path)
        parts = pure.parent.parts
        if not parts:
            continue
        head = tuple(_key(p, pure) for p in parts[:2])
        groups.setdefault((type(pure), head), []).append(pure.parent)
    roots = []
    for (kind, _head), parents in groups.items():
        common = list(parents[0].parts)
        for parent in parents[1:]:
            parts = parent.parts
            n = 0
            while n < min(len(common), len(parts)) and _key(common[n], parent) == _key(parts[n], parent):
                n += 1
            common = common[:n]
        roots.append(Root(str(kind(*common)), len(parents)))
    return sorted(roots, key=lambda r: (-r.tracks, r.path))


def remap_path(path: str, mapping: dict) -> Optional[str]:
    """``path`` under the folder ``mapping`` gives the root it is under (a root -> a folder on this device; the
    longest root first); None when it is under none mapped."""
    pure = _pure(path)
    parts = pure.parts
    best = None
    for root, folder in mapping.items():
        if not folder:
            continue
        rp = _pure(root)
        if type(rp) is not type(pure):
            continue
        rparts = rp.parts
        if len(rparts) <= len(parts) and all(_key(a, pure) == _key(b, pure) for a, b in zip(rparts, parts)):
            if best is None or len(rparts) > best[0]:
                best = (len(rparts), root, folder)
    if best is None:
        return None
    if best[2] == best[1]:      # the root where it was (a restore on the same device): the path as stored
        return path
    rest = parts[best[0]:]
    return os.path.join(best[2], *rest) if rest else str(best[2])


@dataclass(frozen=True)
class RootCheck:
    found: int = 0          # the file there, of the size stored
    differs: int = 0        # a file there of another size
    missing: int = 0


def check_mapping(paths: Iterable[tuple[str, int]], mapping: dict) -> dict:
    """Per root of ``mapping``, how many of the tracks under it are found where it maps them (``paths``: (path as
    stored, size stored)); the size only confirms the user's mapping, nothing is looked for."""
    out: dict[str, list[int]] = {root: [0, 0, 0] for root in mapping}
    for path, size in paths:
        for root in mapping:
            target = remap_path(path, {root: mapping[root]})
            if target is None:
                continue
            try:
                st = os.stat(target)
                out[root][0 if not size or st.st_size == size else 1] += 1
            except (OSError, ValueError):
                out[root][2] += 1
            break
    return {root: RootCheck(*counts) for root, counts in out.items()}


# ---- unpacking ----

@dataclass(frozen=True)
class UnpackResult:
    folder: str
    tracks: int
    moved: int              # paths rewritten under a mapped root
    clashes: int            # paths left as they were: another track already had the new one
    without_analysis: int   # tracks the package had no stored analysis of


def unpack_library(info: PackageInfo, folder: str, mapping: dict, progress: Progress = None,
                   cancelled: Cancelled = None) -> UnpackResult:
    """The package's library unpacked into ``folder`` (new, or an empty folder): unpacked beside it first and
    renamed at the end, each file's hash checked, the databases' integrity and schema checked (a trigger, view or
    table the app's own schema lacks refuses the package), the tracks' paths rewritten through ``mapping`` (root ->
    folder on this device)."""
    dest = Path(folder)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise PackageError(PackageError.DESTINATION, str(dest))
    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        raise PackageError(PackageError.DESTINATION, str(staging))
    staging.mkdir(parents=True)
    names = info.names("library/")
    total = max(1, sum(info.files[n][0] for n in names))
    done = [0, -1]

    def advance(n: int) -> None:
        done[0] += n
        share = int(100 * done[0] / total)
        if progress is not None and share != done[1]:
            done[1] = share
            progress(min(1.0, done[0] / total))
    try:
        with zipfile.ZipFile(info.path) as zf:
            for name in names:
                target = staging.joinpath(*name[len("library/"):].split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_checked(zf, info, name, target, advance, cancelled)
        for name in _DATABASES:
            path = staging.joinpath(*name[len("library/"):].split("/"))
            if path.is_file():
                _check_database(str(path), name)
        if not (staging / VERSION_FILENAME).is_file():     # read_package checked the library is of this version
            write_library_version(staging, CURRENT_LIBRARY_VERSION)
        moved, clashes, uids = _rewrite_paths(str(staging / "library.db"), mapping)
        without = sum(1 for uid in uids if not (staging / f"{uid}.npz").is_file())
        if dest.exists():
            dest.rmdir()
        os.replace(staging, dest)
        return UnpackResult(str(dest), len(uids), moved, clashes, without)
    except BaseException:
        _remove_tree(staging)
        raise


def _check_database(path: str, name: str) -> None:
    """The database is whole and holds no code: the app makes no trigger or view, and one in another's package would
    run when the app reads or writes it (a table or index an earlier version left is only data: kept). Read with
    untrusted schema code off."""
    try:
        con = sqlite3.connect(path)
        try:
            con.execute("PRAGMA trusted_schema=OFF")
            code = con.execute("SELECT type, name FROM sqlite_master WHERE type IN ('trigger', 'view')").fetchall()
            if code:
                raise PackageError(PackageError.SCHEMA, f"{name}: {sorted(code)[:5]}")
            result = con.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise PackageError(PackageError.SCHEMA, f"{name}: {result}")
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise PackageError(PackageError.SCHEMA, f"{name}: {exc}") from exc


def _rewrite_paths(path: str, mapping: dict) -> tuple[int, int, list[str]]:
    """The tracks' paths under a mapped root rewritten (a path another track already has: left, counted); (moved,
    clashes, every track's uid)."""
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA trusted_schema=OFF")
        rows = con.execute("SELECT uid, path FROM tracks").fetchall()
        moved = clashes = 0
        with con:
            for uid, old in rows:
                new = remap_path(str(old), mapping)
                if new is None or new == old:
                    continue
                try:
                    con.execute("UPDATE tracks SET path = ? WHERE uid = ?", (new, uid))
                    moved += 1
                except sqlite3.IntegrityError:
                    clashes += 1
        return moved, clashes, [str(uid) for uid, _p in rows]
    finally:
        con.close()


def _remove(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _remove_tree(folder: Path) -> None:
    """A folder the unpacking made, emptied and removed: its files and folders only, a link never followed."""
    if not folder.is_dir() or folder.is_symlink():
        return
    for base, dirs, names in os.walk(folder, topdown=False, followlinks=False):
        for name in names:
            _remove(Path(base) / name)
        for name in dirs:
            path = Path(base) / name
            if path.is_symlink():
                _remove(path)
            else:
                try:
                    path.rmdir()
                except OSError:
                    pass
    try:
        folder.rmdir()
    except OSError:
        pass
