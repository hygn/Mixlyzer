import os
import re
import tempfile
import unittest
from pathlib import Path

from system import i18n

PO = r'''
msgid ""
msgstr ""
"Plural-Forms: nplurals=2; plural=(n != 1);\n"

msgid "Settings"
msgstr "설정"

#, python-brace-format
msgid ""
"Two\n"
"lines {name}"
msgstr ""
"두\n"
"줄 {name}"

#, fuzzy
msgid "Changed"
msgstr "old"

msgid "{count} track"
msgid_plural "{count} tracks"
msgstr[0] "one {count}"
msgstr[1] "many {count}"

msgctxt "key"
msgid "Key"
msgstr "키"

msgid "Untranslated"
msgstr ""

#~ msgid "Gone"
#~ msgstr "obsolete"
'''


class ReadPoTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "x.po"
        self.path.write_text(PO, encoding="utf-8")
        self.catalog = i18n.read_po(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_entries(self):
        c = self.catalog
        self.assertEqual(c.gettext("Settings"), "설정")
        self.assertEqual(c.gettext("Two\nlines {name}"), "두\n줄 {name}")
        self.assertEqual(c.gettext("Changed"), "Changed")            # fuzzy: left out
        self.assertEqual(c.gettext("Untranslated"), "Untranslated")
        self.assertEqual(c.gettext("Gone"), "Gone")                  # obsolete
        self.assertEqual(c.pgettext("key", "Key"), "키")
        self.assertEqual(c.gettext("Key"), "Key")                    # another context

    def test_plural(self):
        c = self.catalog
        self.assertEqual(c.ngettext("{count} track", "{count} tracks", 1), "one {count}")
        self.assertEqual(c.ngettext("{count} track", "{count} tracks", 5), "many {count}")


class CatalogTests(unittest.TestCase):
    def test_languages_listed(self):
        self.assertIn(i18n.SOURCE_LANGUAGE, i18n.LANGUAGES)
        for tag in i18n.LANGUAGES:
            self.assertRegex(tag, r"^[a-z]{2,3}-[A-Z]{2}$")
            if tag != i18n.SOURCE_LANGUAGE:
                self.assertTrue(i18n.catalog_path(tag).is_file(), tag)

    def test_catalogs_complete_and_placeholders_kept(self):
        try:
            from babel.messages.pofile import read_po
        except ImportError:
            self.skipTest("Babel (the dev group: uv sync --group dev) reads the catalogs as the build does")
        placeholder = re.compile(r"\{[^{}]*\}")
        for tag in i18n.LANGUAGES:
            if tag == i18n.SOURCE_LANGUAGE:
                continue
            with open(i18n.catalog_path(tag), "rb") as fh:
                catalog = read_po(fh)
            for message in catalog:
                if not message.id:
                    continue
                english = message.id if isinstance(message.id, str) else message.id[0]
                texts = [message.string] if isinstance(message.string, str) else list(message.string)
                for text in texts:
                    self.assertTrue(text, f"{tag}: untranslated {english!r}")
                    self.assertFalse(message.fuzzy, f"{tag}: fuzzy {english!r}")
                    self.assertEqual(set(placeholder.findall(text)), set(placeholder.findall(english)),
                                     f"{tag}: {english!r}")

    def test_install_sets_the_environment(self):
        before = os.environ.get(i18n.LANGUAGE_ENV)
        try:
            self.assertEqual(i18n.install("ja-JP"), "ja-JP")
            self.assertEqual(os.environ[i18n.LANGUAGE_ENV], "ja-JP")
            self.assertEqual(i18n.tr("Settings"), "環境設定")
            self.assertEqual(i18n.install(i18n.SOURCE_LANGUAGE), i18n.SOURCE_LANGUAGE)
            self.assertEqual(i18n.tr("Settings"), "Settings")
        finally:
            i18n.install(i18n.SOURCE_LANGUAGE)
            if before is None:
                os.environ.pop(i18n.LANGUAGE_ENV, None)
            else:
                os.environ[i18n.LANGUAGE_ENV] = before


if __name__ == "__main__":
    unittest.main()
