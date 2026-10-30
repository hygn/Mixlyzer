"""The backups made before a file's tags are written (the Sync tab's Write Tags, as its confirmation says), in the
app folder's ``backups/tags/<date>/`` (gitignored), named after the file with the time, and put back by ``restore``:

- the tag block (``tag_block_backup``, "<name> <HHMMSS>.mxtags"): the bytes of the file the write replaces, the rest
  (its audio) named by where the written file has it (core.tags.writer's segments: the parts of the file kept as
  they are), with the whole file's SHA-256. Small (the tags, a cover picture they rewrite); put back onto the file
  while it has the size that write left and the parts it kept as they were (tags written again since in the same
  room are what the backup holds itself): the file rebuilt checked against the hash, anything else refused.
- the whole file (``backup_copy``, "<name> <HHMMSS>.<ext>"): a copy, its times kept.

``restore`` makes the file anew beside it and moves it over it: the file is either as it was or untouched."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import time
from pathlib import Path

from system.i18n import tr
from system.resource_paths import app_folder

FOLDER = Path("backups") / "tags"
SUFFIX = ".mxtags"
MAGIC = b"MXTAGS1\n"
_BLOCK = 1 << 20
_HEADER_MAX = 64 << 20      # a header longer than this is no backup of ours


class BackupError(Exception):
    """A backup that cannot be made or put back (its message for the user)."""


def folder() -> Path:
    return app_folder() / FOLDER


def _target(source: Path, suffix: str) -> Path:
    day = folder() / time.strftime("%Y%m%d")
    day.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    target, n = day / f"{source.stem} {stamp}{suffix}", 2
    while target.exists():
        target, n = day / f"{source.stem} {stamp}-{n}{suffix}", n + 1
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def backup_copy(path: str | Path) -> Path:
    """A copy of the whole file made now (its times kept); OSError when it cannot be (nothing left behind)."""
    source = Path(path)
    target = _target(source, source.suffix)
    try:
        shutil.copy2(source, target)
    except OSError:
        target.unlink(missing_ok=True)
        raise
    return target


def tag_block_backup(path: str | Path, segments: list) -> Path:
    """The tag block backup of ``path`` before it becomes ``segments`` (core.tags.writer.Segment: new bytes, or
    (offset, length) of the file as it is); OSError when it cannot be made (nothing left behind)."""
    source = Path(path)
    size = source.stat().st_size
    kept, new_pos = [], 0               # (offset in the file, offset in the file to come, length)
    for seg in segments:
        if isinstance(seg, tuple):
            kept.append((seg[0], new_pos, seg[1]))
            new_pos += seg[1]
        else:
            new_pos += len(seg)
    parts, blob, pos = [], bytearray(), 0
    with open(source, "rb") as f:
        for old, new, length in sorted(kept):
            if old > pos:
                f.seek(pos)
                parts.append(["bytes", len(blob), old - pos])
                blob += f.read(old - pos)
                pos = old
            if old + length > pos:
                skip = pos - old
                parts.append(["kept", new + skip, length - skip])
                pos = old + length
        if pos < size:
            f.seek(pos)
            parts.append(["bytes", len(blob), size - pos])
            blob += f.read(size - pos)
    header = json.dumps({"name": source.name, "size": size, "sha256": _sha256(source), "written_size": new_pos,
                         "parts": parts}).encode("utf-8")
    target = _target(source, SUFFIX)
    try:
        with open(target, "wb") as out:
            out.write(MAGIC + struct.pack("<I", len(header)) + header + bytes(blob))
    except OSError:
        target.unlink(missing_ok=True)
        raise
    return target


def _read_tag_block(backup: Path) -> tuple[dict, int]:
    """The backup's header and where its bytes start."""
    with open(backup, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise BackupError(tr("This is not a tag backup of Mixlyzer's."))
        (n,) = struct.unpack("<I", f.read(4))
        if n > _HEADER_MAX:
            raise BackupError(tr("This is not a tag backup of Mixlyzer's."))
        try:
            header = json.loads(f.read(n).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise BackupError(tr("This is not a tag backup of Mixlyzer's.")) from exc
    return header, len(MAGIC) + 4 + n


def backed_up_name(backup: str | Path) -> str:
    """The file name a tag block backup was made of ("" for a whole file's copy: its own name)."""
    backup = Path(backup)
    return _read_tag_block(backup)[0].get("name", "") if backup.suffix.lower() == SUFFIX else ""


def backups_of(path: str | Path) -> list[Path]:
    """The backups made of the file ``path`` names, newest first: its tag block backups (by the name they hold) and
    its whole copies ("<stem> <HHMMSS>[-n]<ext>" in a day's folder); links and backups not read left out."""
    source = Path(path)
    root = folder()
    if not root.is_dir():
        return []
    copy = re.compile(re.escape(source.stem) + r" \d{6}(-\d+)?" + re.escape(source.suffix) + r"$", re.IGNORECASE)
    found = []
    for day in root.iterdir():
        if not day.is_dir() or day.is_symlink():
            continue
        for entry in day.iterdir():
            if entry.is_symlink() or not entry.is_file() or not entry.name.startswith(source.stem + " "):
                continue
            if entry.suffix.lower() == SUFFIX:
                try:
                    if backed_up_name(entry) == source.name:
                        found.append(entry)
                except (BackupError, OSError):
                    continue
            elif copy.match(entry.name):
                found.append(entry)
    return sorted(found, key=lambda entry: (entry.parent.name, entry.name.rsplit(" ", 1)[-1]), reverse=True)


def restore(backup: str | Path, path: str | Path) -> None:
    """The file ``path`` put back as ``backup`` holds it (a tag block backup, or a whole file's copy). BackupError
    when the tag block backup is not of this file as its write left it, or the rebuilt file is not the one backed
    up; OSError when the file cannot be written. Either way the file is untouched."""
    backup, path = Path(backup), Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=".mixlyzer-restore-", suffix=path.suffix, dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            if backup.suffix.lower() != SUFFIX:
                with open(backup, "rb") as src:
                    shutil.copyfileobj(src, out, _BLOCK)
            else:
                header, start = _read_tag_block(backup)
                if path.stat().st_size != header.get("written_size"):
                    raise BackupError(tr("The file has changed since this backup was made: it is not put back."))
                with open(path, "rb") as now, open(backup, "rb") as blob:
                    for kind, offset, length in header.get("parts", ()):
                        src = now if kind == "kept" else blob
                        src.seek(offset if kind == "kept" else start + offset)
                        while length > 0:
                            block = src.read(min(length, _BLOCK))
                            if not block:
                                raise BackupError(tr("The file has changed since this backup was made: it is "
                                                     "not put back."))
                            out.write(block)
                            length -= len(block)
            out.flush()
            os.fsync(out.fileno())
        if backup.suffix.lower() == SUFFIX and _sha256(tmp) != header.get("sha256"):
            raise BackupError(tr("The file has changed since this backup was made: it is not put back."))
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
