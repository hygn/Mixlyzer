"""Files the analyzer writes (a retrained model, the training's caches): replaced whole or not at all."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

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


def atomic_save_npz(path: str | Path, arrays: dict, *, compressed: bool = True,
                    check: Callable[[Path], object] | None = None) -> None:
    """``arrays`` as an NPZ at ``path``, replaced whole; ``check`` reads the written file before it replaces ``path``
    (raising leaves ``path`` as it was). Written through a handle: given a path not ending in ".npz" (a model named
    "x.NPZ" or "x"), numpy writes "<path>.npz" instead, and the empty temporary file would replace ``path``."""
    save = np.savez_compressed if compressed else np.savez
    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as handle:
            save(handle, **arrays)
        if check is not None:
            check(temporary)


def atomic_save_npy(path: str | Path, values: np.ndarray) -> None:
    """``values`` as an NPY at ``path``, replaced whole (through a handle, as ``atomic_save_npz``)."""
    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
