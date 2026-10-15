"""Stable boundary for future library migration support.

The application performs no automatic migration.  It can inspect library
compatibility through this adapter without coupling startup to one-off
development migration utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.library_version import CURRENT_LIBRARY_VERSION, read_library_version


@dataclass(frozen=True)
class LibraryMigrationStatus:
    library_path: Path
    current_version: str
    required_version: str

    @property
    def is_current(self) -> bool:
        return self.current_version == self.required_version


class LibraryMigrationAdapter:
    """Read-only startup adapter; development migrations remain explicit."""

    def inspect(
        self,
        library_path: str | Path,
        required_version: str = CURRENT_LIBRARY_VERSION,
    ) -> LibraryMigrationStatus:
        path = Path(library_path)
        return LibraryMigrationStatus(
            library_path=path,
            current_version=read_library_version(path),
            required_version=str(required_version).strip(),
        )


library_migration = LibraryMigrationAdapter()


__all__ = [
    "LibraryMigrationAdapter",
    "LibraryMigrationStatus",
    "library_migration",
]
