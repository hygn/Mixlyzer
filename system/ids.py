"""The app's ids: a track's, a playlist's, a work's are UUIDv4s, kept in one spelling (lower case, hyphenated) so the
library's rows, its {uid}.npz files and its thumbnails name the same track."""
from __future__ import annotations

from typing import Optional
from uuid import UUID


def canonical_uuid4(value: object) -> str:
    parsed = UUID(str(value).strip())
    if parsed.version != 4:
        raise ValueError(f"Expected UUIDv4, got version {parsed.version}")
    return str(parsed)


def canonical_uuid4_or_none(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return canonical_uuid4(text)
