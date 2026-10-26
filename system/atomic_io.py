"""Small, format-agnostic helpers for durable atomic file replacement."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically replace ``path`` on success.

    The caller may use any writer that accepts a filesystem path. The completed
    temporary file is flushed to disk before replacement; failures leave the
    previous destination untouched.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=target.suffix or ".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        yield temporary
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        _replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


REPLACE_TRIES = 20          # Windows refuses to replace a file another process has open (a reader, the antivirus,
REPLACE_WAIT_SEC = 0.025    # the indexer) for a moment: tried again for up to half a second


def _replace(temporary: Path, target: Path) -> None:
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_WAIT_SEC)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    with atomic_output_path(path) as temporary:
        temporary.write_text(text, encoding=encoding, newline="")


def atomic_write_json(
    path: str | Path,
    payload: object,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=ensure_ascii, indent=indent)
