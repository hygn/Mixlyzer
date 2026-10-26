"""Files the analyzer writes (a retrained model, the training's caches): replaced whole or not at all."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPLACE_TRIES = 20          # Windows refuses to replace a file another process has open (a reader, the antivirus,
REPLACE_WAIT_SEC = 0.025    # the indexer) for a moment: tried again for up to half a second


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    """A temporary path beside ``path`` to write to, replacing ``path`` once written (flushed to disk first); a
    failure leaves ``path`` as it was."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix or ".tmp",
                                             dir=target.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        yield temporary
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        for attempt in range(REPLACE_TRIES):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == REPLACE_TRIES - 1:
                    raise
                time.sleep(REPLACE_WAIT_SEC)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    with atomic_output_path(path) as temporary:
        temporary.write_text(text, encoding=encoding, newline="")


def atomic_write_json(path: str | Path, payload: object, *, ensure_ascii: bool = False, indent: int | None = 2) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=ensure_ascii, indent=indent)
