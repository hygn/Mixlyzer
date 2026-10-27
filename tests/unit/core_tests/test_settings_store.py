"""core.settings.store: the settings' one owner. A save writes both files and tells it; what playing changes is
written a second after it last did (or at once when the app quits); current() is the installed store's settings."""
import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.settings import config as settings_config
from core.settings import store

APP = QApplication.instance() or QApplication([])


class SettingsStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config_file = os.path.join(self._tmp.name, "config.json")
        self.deck_file = os.path.join(self._tmp.name, "deck.json")
        for name, value in (("CONFIG_FILE", self.config_file), ("DECK_FILE", self.deck_file)):
            patcher = mock.patch.object(settings_config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(store, "_store", None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.settings = store.SettingsStore(settings_config.default_cfg())

    def read(self, path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_a_save_writes_both_files_and_tells_it(self):
        seen = []
        self.settings.changed.connect(lambda old, new: seen.append((old, new)))
        old = self.settings.current
        new = settings_config.default_cfg()
        new.analysisconfig.max_parallel_analyses = 7
        new.deckconfig.volume_trim_dbfs = -3.0
        self.settings.save(new)
        self.assertIs(self.settings.current, new)
        self.assertEqual(seen, [(old, new)])
        self.assertEqual(self.read(self.config_file)["analysisconfig"]["max_parallel_analyses"], 7)
        self.assertEqual(self.read(self.deck_file), new.deck_dict())

    def test_what_playing_changes_written_once_it_rests(self):
        deck = self.settings.current.deckconfig
        deck.volume_trim_dbfs = -6.0
        with mock.patch.object(store, "DECK_SAVE_DELAY_MS", 1000):
            self.settings.deck_changed()
        self.assertFalse(os.path.exists(self.deck_file))           # not yet: a knob still turning
        self.assertTrue(self.settings._deck_save.isActive())
        deck.volume_trim_dbfs = -4.5
        self.settings.flush()                                       # the app quitting
        self.assertEqual(self.read(self.deck_file)["volume_trim_dbfs"], -4.5)
        self.assertFalse(os.path.exists(self.config_file))          # config.json only by a save
        self.assertFalse(self.settings._deck_save.isActive())

    def test_current_is_the_installed_stores(self):
        self.assertIsNot(store.current(), self.settings.current)    # none installed: the defaults
        store.install(self.settings)
        self.assertIs(store.current(), self.settings.current)
        store.current().deckconfig.volume_trim_dbfs = -1.0
        store.deck_changed()
        self.assertTrue(self.settings._deck_save.isActive())


if __name__ == "__main__":
    unittest.main()
