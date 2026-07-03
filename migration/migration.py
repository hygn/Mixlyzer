from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from core.library_version import CURRENT_LIBRARY_VERSION, read_library_version, write_library_version
from migration import lib_0_1_0_to_0_1_1
from migration import lib_0_1_1_to_0_2_0
from migration import lib_0_2_0_to_0_3_0


MigrationLogger = Callable[[str], None]
MigrationProgress = Callable[[int], None]


# Supported chain: 0.1.0 -> 0.1.1 -> 0.2.0 -> 0.3.0.
MIGRATIONS: dict[tuple[str, str], object] = {
    (lib_0_1_0_to_0_1_1.SOURCE_VERSION, lib_0_1_0_to_0_1_1.TARGET_VERSION): lib_0_1_0_to_0_1_1,
    (lib_0_1_1_to_0_2_0.SOURCE_VERSION, lib_0_1_1_to_0_2_0.TARGET_VERSION): lib_0_1_1_to_0_2_0,
    (lib_0_2_0_to_0_3_0.SOURCE_VERSION, lib_0_2_0_to_0_3_0.TARGET_VERSION): lib_0_2_0_to_0_3_0,
}


def get_migration_path(from_version: str, to_version: str = CURRENT_LIBRARY_VERSION) -> list[object]:
    current = str(from_version).strip()
    target = str(to_version).strip()
    modules: list[object] = []
    visited: set[str] = set()
    while current != target:
        if current in visited:
            raise RuntimeError(f"Migration loop detected at version {current}")
        visited.add(current)
        next_module = None
        for (src, _dst), module in MIGRATIONS.items():
            if src == current:
                next_module = module
                break
        if next_module is None:
            raise RuntimeError(f"No migration path from {current} to {target}")
        modules.append(next_module)
        current = str(next_module.TARGET_VERSION)
    return modules


def migrate_library(
    lib_path: str | Path,
    logger: MigrationLogger | None = None,
    progress_callback: MigrationProgress | None = None,
) -> None:
    lib_root = Path(lib_path)
    current = read_library_version(lib_root)
    modules = get_migration_path(current, CURRENT_LIBRARY_VERSION)
    total_modules = max(1, len(modules))
    for module_index, module in enumerate(modules):
        if logger is not None:
            logger(f"Migrating library {module.SOURCE_VERSION} -> {module.TARGET_VERSION}")
        if progress_callback is None:
            result = module.migrate_library_with_progress(lib_root, logger=logger or print, progress_callback=None)
        else:
            start_pct = int(round((module_index * 100) / total_modules))
            end_pct = int(round(((module_index + 1) * 100) / total_modules))

            def _module_progress(local_pct: int, start=start_pct, end=end_pct) -> None:
                span = max(0, end - start)
                overall = start + int(round((max(0, min(100, int(local_pct))) / 100.0) * span))
                progress_callback(min(100, max(0, overall)))

            result = module.migrate_library_with_progress(lib_root, logger=logger or print, progress_callback=_module_progress)
        if int(result or 0) != 0:
            raise RuntimeError(
                f"Migration {module.SOURCE_VERSION} -> {module.TARGET_VERSION} finished with errors."
            )
        write_library_version(lib_root, module.TARGET_VERSION)
    if progress_callback is not None:
        progress_callback(100)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Mixlyzer library migrations")
    parser.add_argument(
        "--libpath",
        default="library",
        help="Path to library directory (default: ./library)",
    )
    args = parser.parse_args()
    lib_path = Path(args.libpath).resolve()
    before = read_library_version(lib_path)
    print(f"Library version: {before} -> {CURRENT_LIBRARY_VERSION}")
    migrate_library(lib_path, logger=print)
    print("Migration finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
