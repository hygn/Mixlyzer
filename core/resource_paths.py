from __future__ import annotations

import sys
from pathlib import Path

# Single source of truth for locating read-only resources that ship with the
# application (e.g. the External Sync process denylist).
#
# These files must NOT be resolved relative to the current working directory:
# if the app is launched from another folder the CWD-relative lookup silently
# fails, and security-critical files like ``process_denylist.json`` would then
# load as empty and fail OPEN. Anchoring to this module's location keeps the
# lookup stable regardless of CWD.
#
# This module lives at ``<project_root>/core/resource_paths.py`` so the project
# root is ``parents[1]``; in a PyInstaller 6 build that is the package's ``_internal`` folder, where the
# bundled files are (the module's path is there), so no sys._MEIPASS handling is needed.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """Directory that bundles read-only application resources."""
    return _PROJECT_ROOT


def app_folder() -> Path:
    """The folder the app is in: the package folder (the exe's) in a PyInstaller build, else the project root.
    It holds config.json, the library and the feature caches: the app runs with it as the current folder."""
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _PROJECT_ROOT


def resource_path(name: str) -> Path:
    """Absolute path to a bundled resource by file name."""
    return _PROJECT_ROOT / name


def setting_path(text: str) -> str:
    """A file path from the settings: as written when it is absolute or found from the current folder, else the
    bundled resource of that name (the shipped assets/sound/click.wav, assets/images/vinyl.png: in a PyInstaller
    build they are with the app's resources, not in the folder it runs from)."""
    text = str(text or "").strip()
    if not text or Path(text).is_absolute() or Path(text).exists():
        return text
    bundled = resource_path(text)
    return str(bundled) if bundled.exists() else text


def process_denylist_path() -> Path:
    """Absolute path to the External Sync process denylist."""
    return resource_path("process_denylist.json")
