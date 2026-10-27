import json
import os
import tempfile
import unittest
from unittest import mock

from PySide6 import QtGui

from core.controls import shortcuts
from ui.shortcuts import ACTIONS

# the shipped keys' tiers: Ctrl+Shift destroys data, Ctrl edits data (in memory; saving and a new playlist write
# it), no Ctrl changes none
DESTRUCTIVE = {f"cue.delete.{c}" for c in "ABCDEFGH"} | {
    "cue.memory_delete", "library.reanalyze", "library.remove_from_library", "library.remove_from_playlist",
    "library.write_tags"}
EDITS_PREFIXES = ("beatgrid.", "key.", "phrase.", "cue.set.", "cue.memory", "editor.undo", "editor.redo",
                  "editor.save", "editor.start_", "editor.end_", "editor.split", "editor.merge_", "editor.reanalyze",
                  "library.new_playlist")
NO_DATA_IN_EDIT_GROUPS = {"beatgrid.tap", "beatgrid.zero", "key.pick", "key.root", "key.drone", "key.chord"}
# OS-wide keys on a supported system: Ctrl+Alt+Del / arrows / F-keys (Windows, Linux consoles and workspaces),
# Ctrl+Alt+L / T (Linux lock screen, terminal), Alt+F2 / F4, Cmd(Ctrl)+Q / W / H (macOS quit, close, hide)
RESERVED = {"Ctrl+Alt+Del", "Ctrl+Alt+Left", "Ctrl+Alt+Right", "Ctrl+Alt+Up", "Ctrl+Alt+Down", "Ctrl+Alt+L",
            "Ctrl+Alt+T", "Alt+F2", "Alt+F4", "Ctrl+Q", "Ctrl+W", "Ctrl+H", "Ctrl+Space", "Ctrl+Esc",
            *(f"Ctrl+Alt+F{n}" for n in range(1, 13)),
            # graphics card overlays and Steam (NVIDIA, AMD), found held on a user's computer
            "Alt+R", "Alt+Z", "Alt+F1", "Alt+F3", "Alt+F9", "Alt+F10", "Ctrl+Alt+M", "Ctrl+Alt+R", "Alt+Shift+R",
            "Alt+Shift+F10", "Shift+F12", "F12"}


class ShippedShortcutTests(unittest.TestCase):
    def test_every_action_has_keys(self):
        defaults = shortcuts.defaults()
        self.assertEqual(set(defaults), {a.id for a in ACTIONS})
        self.assertEqual([a for a, keys in defaults.items() if not keys], [])

    def test_keys_unique_and_single_combinations(self):
        seen = {}
        for action, keys in shortcuts.defaults().items():
            for k in keys:
                seq = QtGui.QKeySequence(k)
                self.assertEqual(seq.count(), 1, (action, k))
                self.assertEqual(seq.toString(QtGui.QKeySequence.PortableText), k, (action, k))
                self.assertNotIn(k, seen, (action, k, seen.get(k)))
                self.assertNotIn(k, RESERVED, action)
                seen[k] = action

    def test_modifier_tiers(self):
        for action, keys in shortcuts.defaults().items():
            for k in keys:
                ctrl, shift = "Ctrl+" in k, "Shift+" in k
                if action in DESTRUCTIVE:
                    self.assertTrue(ctrl and shift, (action, k))
                elif action.startswith(EDITS_PREFIXES) and action not in NO_DATA_IN_EDIT_GROUPS:
                    self.assertTrue(ctrl and not shift, (action, k))
                else:
                    self.assertFalse(ctrl, (action, k))


class UserFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.user = os.path.join(self.dir.name, "shortcuts.json")
        patcher = mock.patch.object(shortcuts, "USER_FILE", self.user)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.dir.cleanup)

    def test_only_changes_written_and_none_removes_the_file(self):
        keys = shortcuts.defaults()
        keys["library.reveal"] = ["I"]
        keys["library.edit_track"] = []
        shortcuts.save(keys)
        with open(self.user, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"library.reveal": ["I"], "library.edit_track": []})
        self.assertEqual(shortcuts.keymap()["library.reveal"], ["I"])
        shortcuts.save(shortcuts.defaults())
        self.assertFalse(os.path.exists(self.user))
        self.assertEqual(shortcuts.keymap(), shortcuts.defaults())

    def test_shipped_file_never_written(self):
        before = (shortcuts.resource_path(shortcuts.DEFAULT_FILE)).read_bytes()
        with self.assertRaises(ValueError):
            shortcuts.save({"playback.play_pause": []}, path=shortcuts.DEFAULT_FILE)
        self.assertEqual(shortcuts.resource_path(shortcuts.DEFAULT_FILE).read_bytes(), before)

    def test_malformed_entries_left_out(self):
        with open(self.user, "w", encoding="utf-8") as fh:
            json.dump({"library.reveal": "I", "library.load": ["Return"]}, fh)
        self.assertEqual(shortcuts.user_changes(), {"library.load": ["Return"]})


if __name__ == "__main__":
    unittest.main()
