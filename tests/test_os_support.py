import sys
import unittest
from unittest import mock

from system import os_support


class OsFlagTests(unittest.TestCase):
    def test_one_flag_per_system(self):
        flags = [os_support.IS_WINDOWS, os_support.IS_LINUX, os_support.IS_MACOS]
        self.assertLessEqual(sum(flags), 1)
        self.assertEqual(os_support.IS_SUPPORTED, sum(flags) == 1)

    def test_detect(self):
        for platform, expected in (("win32", "windows"), ("linux", "linux"), ("darwin", "macos"),
                                   ("freebsd14", "other")):
            with mock.patch.object(sys, "platform", platform):
                self.assertEqual(os_support._detect(), expected)

    def test_hotkey_taken(self):
        self.assertFalse(os_support.hotkey_taken(""))
        self.assertFalse(os_support.hotkey_taken("Hyper+Q"))
        if not os_support.IS_WINDOWS:
            self.assertFalse(os_support.hotkey_taken("Ctrl+Alt+Shift+F11"))
            return
        import ctypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.assertFalse(os_support.hotkey_taken("Ctrl+Alt+Shift+F11"))
        self.assertTrue(user32.RegisterHotKey(None, 0x1234, 0x2 | 0x1 | 0x4, 0x7A))   # held here: as by another
        try:
            self.assertTrue(os_support.hotkey_taken("Ctrl+Alt+Shift+F11"))
        finally:
            user32.UnregisterHotKey(None, 0x1234)
        self.assertFalse(os_support.hotkey_taken("Ctrl+Alt+Shift+F11"))

    def test_hidden_subprocess_kwargs(self):
        kwargs = os_support.hidden_subprocess_kwargs()
        if os_support.IS_WINDOWS:
            self.assertIn("creationflags", kwargs)
        else:
            self.assertEqual(kwargs, {})
        with mock.patch.object(os_support, "IS_WINDOWS", False):
            self.assertEqual(os_support.hidden_subprocess_kwargs(), {})


if __name__ == "__main__":
    unittest.main()
