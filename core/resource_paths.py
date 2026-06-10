from __future__ import annotations

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
# root is ``parents[1]``. If a frozen (PyInstaller) build is added later, add
# the ``sys._MEIPASS`` handling here so every caller benefits.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """Directory that bundles read-only application resources."""
    return _PROJECT_ROOT


def resource_path(name: str) -> Path:
    """Absolute path to a bundled resource by file name."""
    return _PROJECT_ROOT / name


def process_denylist_path() -> Path:
    """Absolute path to the External Sync process denylist."""
    return resource_path("process_denylist.json")
