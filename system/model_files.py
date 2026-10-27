"""Learned model files: the shipped model is pinned as ``default_<name>`` next to the user's ``<name>``.

Reoptimize writes the user's file and never the shipped one, so the shipped model is always there to start
from again; a user's file that does not exist (yet) means the shipped model.
"""

from __future__ import annotations

from pathlib import Path

from system.resource_paths import resource_path

SHIPPED_PREFIX = "default_"


def shipped_path(path: str | Path) -> Path:
    """The shipped model the file at ``path`` replaces (``path`` itself when it names a shipped file)."""
    p = resource_path(str(path))
    return p if is_shipped(p) else p.with_name(SHIPPED_PREFIX + p.name)


def resolve_model(path: str | Path) -> Path:
    """The model file to read: the user's file when it exists, else the shipped one."""
    p = resource_path(str(path))
    return p if p.is_file() else shipped_path(p)


def is_shipped(path: str | Path) -> bool:
    """Whether ``path`` names a shipped file, in any letter case (Windows and macOS find it so)."""
    return Path(path).name.casefold().startswith(SHIPPED_PREFIX)


def has_user_model(path: str | Path) -> bool:
    """Whether a retrained (user's) model is in use at ``path``."""
    p = resource_path(str(path))
    return not is_shipped(p) and p.is_file()
