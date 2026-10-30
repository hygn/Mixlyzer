"""The third-party notices (assets/notices/THIRD_PARTY_NOTICES.txt, build/notices.py) name what the package carries:
every dependency the project declares, the native engine's VOLK and crates, the fonts, the data; and the spec ships
the file. A dependency added without running build/notices.py again fails here."""
import re
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTICES = ROOT / "assets" / "notices" / "THIRD_PARTY_NOTICES.txt"


def entries() -> set[str]:
    """The notices' entries, by their names lower-cased (--- name version)."""
    text = NOTICES.read_text(encoding="utf-8")
    return {m.group(1).lower() for m in re.finditer(r"^--- (\S+)", text, re.M)}


class NoticesTests(unittest.TestCase):
    def test_every_declared_dependency_is_named(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        names = {re.split(r"[<>=!~ ;\[]", d, 1)[0].lower() for d in project["dependencies"]}
        named = {n.replace("_", "-") for n in entries()}
        self.assertEqual({n for n in names if n.replace("_", "-") not in named}, set())

    def test_the_bundled_works_and_their_texts(self):
        text = NOTICES.read_text(encoding="utf-8")
        for name in ("Python", "Qt", "libsndfile", "VOLK", "cpu_features", "The Rust standard library", "Noto Sans",
                     "Unicode Character Database", "Hong Kong Cantonese Corpus", "rime-cantonese",
                     "The GNU General Public License v3"):
            self.assertIn(f"--- {name}", text, name)
        self.assertIn("GNU LESSER GENERAL PUBLIC LICENSE", text)
        self.assertIn("SIL OPEN FONT LICENSE", text.upper())
        # the engine's crates (Cargo.lock), each named
        lock = (ROOT / "audioengine" / "native" / "Cargo.lock").read_text(encoding="utf-8")
        crates = set(re.findall(r'^name = "([^"]+)"', lock, re.M)) - {"mixlyzer-engine", "mixlyzer_engine"}
        for crate in ("cpal", "volk", "midir"):
            if crate in crates:
                self.assertIn(crate, entries(), crate)

    def test_the_spec_ships_them_and_not_the_unread_corpora(self):
        spec = (ROOT / "build" / "mixlyzer.spec").read_text(encoding="utf-8")
        self.assertIn("shipped('assets/notices')", spec)
        for corpus in ("cantomap", "common_voice", "ctcpc"):
            self.assertIn(f"data/{corpus}", spec)
        # FFmpeg checked by the notices' own build before it is shipped
        self.assertIn("+ ffmpeg()", spec)
        self.assertIn("notices.ffmpeg_build(", spec)


class FFmpegBuildTests(unittest.TestCase):
    """An FFmpeg is shipped only as FFmpeg's own code under the LGPL (build/notices.py's ffmpeg_build, which the spec
    runs too): no GPL or nonfree build, no library of another project linked into it."""

    OWN = ("--disable-everything", "--disable-autodetect", "--extra-version=mixlyzer", "--enable-decoder=aac",
           "--enable-demuxer=mov", "--enable-filter=aresample")

    def version_text(self, *options: str) -> str:
        return ("ffmpeg version n7.1.3-19-gdac2a1116d-mixlyzer Copyright (c) 2000-2025 the FFmpeg developers\n"
                "built with clang version 21.1.2\n"
                f"configuration: {' '.join(options)}\n"
                "libavutil      59. 39.100 / 59. 39.100\n")

    def setUp(self):
        sys.path.insert(0, str(ROOT / "build"))
        import notices
        self.notices = notices

    def tearDown(self):
        sys.path.remove(str(ROOT / "build"))

    def test_its_own_build_is_read(self):
        version, compiler, options = self.notices.ffmpeg_build(self.version_text(*self.OWN, "--enable-small"))
        self.assertEqual(version, "n7.1.3-19-gdac2a1116d-mixlyzer")
        self.assertEqual(compiler, "clang version 21.1.2")
        self.assertIn("--enable-decoder=aac", options)

    def test_a_gpl_or_nonfree_build_is_refused(self):
        for option in ("--enable-gpl", "--enable-nonfree"):
            with self.assertRaises(SystemExit, msg=option):
                self.notices.ffmpeg_build(self.version_text(*self.OWN, option))

    def test_another_projects_library_is_refused(self):
        # an "LGPL" build may link a GPL library inside another (BtbN's chromaprint, built on FFTW)
        for option in ("--enable-chromaprint", "--enable-libopus", "--enable-zlib", "--enable-schannel"):
            with self.assertRaises(SystemExit, msg=option):
                self.notices.ffmpeg_build(self.version_text(*self.OWN, option))
        with self.assertRaises(SystemExit):     # the libraries it would find by itself
            self.notices.ffmpeg_build(self.version_text("--disable-everything", "--enable-decoder=aac"))

    def test_the_notices_name_the_root_ffmpeg(self):
        if not self.notices.FFMPEG.is_file():
            self.skipTest("no FFmpeg at the repository's root")
        version, _compiler, _options = self.notices.ffmpeg_build(self.notices.ffmpeg_version_text())
        self.assertIn(f"--- FFmpeg {version}\n", NOTICES.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
