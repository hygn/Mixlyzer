"""Settings > Library's Backup: the library packed into one file (core.library.package), and a package restored into
a new library folder, on this device or another.

    backup(dest, feature_cache)       a job of the library's owner writes the package: progress, backed_up / failed
    read(path, done, failed)          a package read and checked (a thread of its own)
    check(info, mapping, done)        how many tracks each music folder mapped finds (a thread)
    restore(info, folder, mapping)    the library unpacked into ``folder`` (new: the library in use is never
                                      touched), the user's files and the caches put in place: restored / failed

What a package holds beside the library is named here: the settings as the store holds them, the user's shortcuts,
MIDI mapping and retrained models, the tie-up lookups' cache, the feature caches when asked. Restored, each goes to
this device's own place for it (never where the package says): a file there already is kept beside it as
``<name>.before-restore-<time>``. The window then saves the settings (config.restored_settings) and starts the app
again on the new library.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

from PySide6 import QtCore

from core.controls import midi, shortcuts
from core.library import jobs, package
from core.settings.config import analysisconfig, config
from core.settings.store import current
from core.tieups import web
from system.model_files import has_user_model, is_shipped
from system.resource_paths import resource_path

PARAMETERS = tuple(f.name for f in fields(analysisconfig) if f.name.endswith("_parameter_path"))
FEATURE_CACHES = tuple(f.name for f in fields(analysisconfig) if f.name.endswith("_feature_cache_path"))


def user_files(cfg: config) -> dict[str, str]:
    """The user's files a package holds -> where they are on this device: the shortcuts, the MIDI mapping, each
    retrained model (a setting naming a shipped file, or no file there, has none)."""
    out = {package.SHORTCUTS: str(resource_path(shortcuts.USER_FILE)), package.MIDI: str(resource_path(midi.USER_FILE))}
    for name in PARAMETERS:
        path = str(getattr(cfg.analysisconfig, name) or "").strip()
        if path and not is_shipped(path):
            out[package.MODELS + name] = str(resource_path(path))
    return out


def feature_caches(cfg: config) -> dict[str, str]:
    """The feature caches' folders by their name prefix in a package."""
    out = {}
    for name in FEATURE_CACHES:
        folder = str(getattr(cfg.analysisconfig, name) or "").strip()
        if folder:
            out[package.FEATURE_CACHE + name] = folder
    return out


@dataclass(frozen=True)
class RestoreResult:
    unpacked: package.UnpackResult
    settings: dict                  # the package's config.json
    deck: dict                      # its deck.json
    music_folders: list             # its music folders on this device (joined by the mapping, those that exist)
    rekordbox_xml_path: str
    kept_aside: list                # this device's files a restored one took the place of, where they were kept


class LibraryBackup(QtCore.QObject):
    progress = QtCore.Signal(float)           # the share done (0 .. 1) of the backup or restore running
    backed_up = QtCore.Signal(object)         # package.ExportResult
    restored = QtCore.Signal(object)          # RestoreResult
    failed = QtCore.Signal(object)            # the exception (package.PackageError: its reason)
    _deliver = QtCore.Signal(object)          # a callable run on the GUI thread

    def __init__(self, library, parent=None) -> None:
        """``library``: the library's owner (core.library.service)."""
        super().__init__(parent)
        self._library = library
        self._cancel = threading.Event()
        self._deliver.connect(lambda fn: fn())

    def cancel(self) -> None:
        """The backup or restore running stopped at its next block (its file, its folder removed)."""
        self._cancel.set()

    # ---- backup ----
    def backup(self, dest: str, feature_cache: bool) -> None:
        cfg = current()
        plan = package.ExportPlan(
            dest=str(dest), settings=cfg.to_dict(), deck=cfg.deck_dict(),
            files={name: path for name, path in user_files(cfg).items()
                   if not name.startswith(package.MODELS) or has_user_model(path)},
            trees={p: f for p, f in feature_caches(cfg).items() if os.path.isdir(f)} if feature_cache else {},
            copies={package.TIEUP_CACHE: web.copy_cache})
        self._cancel.clear()
        self._library.read(jobs.export_package, plan, self.progress.emit, self._cancel.is_set,
                           done=self.backed_up.emit, failed=self.failed.emit)

    # ---- restore ----
    def read(self, path: str, done: Callable[[package.PackageInfo], None],
             failed: Callable[[Exception], None]) -> None:
        self._run(lambda: package.read_package(path), done, failed)

    def check(self, info: package.PackageInfo, mapping: dict, done: Callable[[dict], None]) -> None:
        self._run(lambda: package.check_mapping(info.paths, mapping), done, lambda _exc: done({}))

    def restore(self, info: package.PackageInfo, folder: str, mapping: dict) -> None:
        cfg = current()
        self._cancel.clear()
        self._run(lambda: self._restore(info, folder, mapping, cfg), self.restored.emit, self.failed.emit)

    def _restore(self, info: package.PackageInfo, folder: str, mapping: dict, cfg: config) -> RestoreResult:
        unpacked = package.unpack_library(info, folder, mapping, self.progress.emit, self._cancel.is_set)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kept: list[str] = []
        for name, target in user_files(cfg).items():
            if info.has(name):
                if os.path.exists(target):
                    aside = f"{target}.before-restore-{stamp}"
                    os.replace(target, aside)
                    kept.append(aside)
                package.extract_file(info, name, target)
        if info.has(package.TIEUP_CACHE):
            with tempfile.TemporaryDirectory(prefix="mxlib-") as tmp:
                copy = os.path.join(tmp, "tieup_web.db")
                package.extract_file(info, package.TIEUP_CACHE, copy)
                web.restore_cache(copy)
        for prefix, target in feature_caches(cfg).items():
            if info.names(prefix + "/"):
                package.extract_tree(info, prefix, target)
        settings, deck = package.read_json(info, package.CONFIG), package.read_json(info, package.DECK)
        lib = settings.get("libconfig") if isinstance(settings.get("libconfig"), dict) else {}
        return RestoreResult(unpacked, settings, deck, _music_folders(lib.get("local_music_directories"), mapping),
                             _rekordbox_xml(str(lib.get("rekordbox_xml_path") or ""), mapping, cfg), kept)

    def _run(self, work: Callable, done: Callable, failed: Callable) -> None:
        def body() -> None:
            try:
                result = work()
            except Exception as exc:
                self._deliver.emit(lambda error=exc: failed(error))
                return
            self._deliver.emit(lambda: done(result))
        threading.Thread(target=body, name="library-backup", daemon=True).start()


def _music_folders(folders, mapping: dict) -> list[str]:
    """The package's music folders as this device has them: joined by the mapping, else as written, those that
    exist here."""
    out: list[str] = []
    for folder in folders if isinstance(folders, list) else []:
        folder = str(folder or "").strip()
        if not folder:
            continue
        here = package.remap_path(folder, mapping) or folder
        if os.path.isdir(here) and here not in out:
            out.append(here)
    return out


def _rekordbox_xml(path: str, mapping: dict, cfg: config) -> str:
    """The package's Rekordbox XML on this device: joined by the mapping, else as written, where its folder exists;
    else this device's."""
    for here in (package.remap_path(path, mapping) if path else None, path):
        if here and os.path.isdir(os.path.dirname(os.path.abspath(here))):
            return here
    return cfg.libconfig.rekordbox_xml_path


def package_suffix() -> str:
    """A backup file's suffix."""
    return package.SUFFIX


def default_folder(libpath: str) -> str:
    """Where a restored library goes unless the user says otherwise: beside the library in use, named by the time."""
    base = Path(libpath or "library").resolve()
    return str(base.parent / f"{base.name}-restored-{time.strftime('%Y%m%d-%H%M')}")
