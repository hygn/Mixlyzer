from __future__ import annotations

from pathlib import Path


DEFAULT_LIBRARY_VERSION = "0.1.0"
CURRENT_LIBRARY_VERSION = "0.3.0"
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
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(str(version).strip() + "\n", encoding="utf-8")


def ensure_current_version_file(lib_path: str | Path) -> None:
    version_path = version_file_path(lib_path)
    if version_path.exists():
        return
    write_library_version(lib_path, CURRENT_LIBRARY_VERSION)
