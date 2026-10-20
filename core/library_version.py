from __future__ import annotations

from pathlib import Path

from utils.atomic_io import atomic_write_text


DEFAULT_LIBRARY_VERSION = "0.1.0"
CURRENT_LIBRARY_VERSION = "0.7.0"
VERSION_FILENAME = "VERSION"


def version_file_path(lib_path: str | Path) -> Path:
    return Path(lib_path) / VERSION_FILENAME


def read_library_version(lib_path: str | Path) -> str:
    version_path = version_file_path(lib_path)
    try:
        text = version_path.read_text(encoding="utf-8").strip()
    except Exception:
        return DEFAULT_LIBRARY_VERSION
    return text or DEFAULT_LIBRARY_VERSION


def write_library_version(lib_path: str | Path, version: str) -> None:
    version_path = version_file_path(lib_path)
    atomic_write_text(version_path, str(version).strip() + "\n")


def ensure_current_version_file(lib_path: str | Path) -> None:
    version_path = version_file_path(lib_path)
    if version_path.exists():
        return
    write_library_version(lib_path, CURRENT_LIBRARY_VERSION)
