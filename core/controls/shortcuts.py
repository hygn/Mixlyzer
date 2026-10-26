"""Keyboard shortcuts: the shipped ones in ``assets/shortcuts/default_shortcuts.json``, the user's changes in
``assets/shortcuts/shortcuts.json`` (the shipped-file convention of the models, ``system.model_files``).

Both map an action (``ui.shortcuts.ACTIONS``) to its key sequences in Qt's portable text ("Ctrl+Shift+Del"); an
empty list is no shortcut. The user's file holds only the actions changed from the shipped ones, so a missing file
is the shipped shortcuts and reverting an action drops its entry. The shipped file is only read: saving to it is
refused.
"""

from __future__ import annotations

import json
from pathlib import Path

from system.model_files import is_shipped
from system.resource_paths import resource_path
from system.atomic_io import atomic_write_json

DEFAULT_FILE = "assets/shortcuts/default_shortcuts.json"
USER_FILE = "assets/shortcuts/shortcuts.json"

Keymap = dict[str, list[str]]


def _read(path: Path) -> Keymap:
    """The action -> key sequences entries of a shortcut file (malformed entries left out)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[Shortcuts] {path.name} not read: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[Shortcuts] {path.name} not read: not an object")
        return {}
    out: Keymap = {}
    for action, keys in data.items():
        if isinstance(keys, list) and all(isinstance(k, str) for k in keys):
            out[str(action)] = [k for k in keys if k.strip()]
        else:
            print(f"[Shortcuts] {path.name}: {action} left out (not a list of key sequences)")
    return out


def defaults() -> Keymap:
    return _read(resource_path(DEFAULT_FILE))


def user_changes() -> Keymap:
    return _read(resource_path(USER_FILE))


def keymap() -> Keymap:
    """The shortcuts in use: the shipped ones with the user's changes."""
    return {**defaults(), **user_changes()}


def save(keys: Keymap, path: str | None = None) -> None:
    """Keep ``keys`` (every action's shortcuts) as the user's (``path``: the user's file): only the actions that
    differ from the shipped shortcuts are written, and with none the user's file is removed (the shipped ones
    again)."""
    target = resource_path(path or USER_FILE)
    if is_shipped(target):
        raise ValueError(f"the shipped shortcuts ({target.name}) are not written")
    shipped = defaults()
    changes = {action: list(seqs) for action, seqs in keys.items() if list(seqs) != shipped.get(action, [])}
    if not changes:
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, changes)
