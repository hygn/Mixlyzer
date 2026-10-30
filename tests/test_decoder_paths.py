"""The decoder's FFmpeg calls on the paths a library holds: a Windows path over 260 characters, a name in NFD (as
macOS gives it), a name starting with "-", and on Linux and macOS a name FFmpeg would take for another protocol's URL
("pipe:1.mp3"): each decoded, its tags, its cover and its length read, through FFmpeg's file protocol alone."""
import os
import shutil
import subprocess
import tempfile
import unicodedata
import unittest
from pathlib import Path

from analyzer.audio import decoder
from system import os_support

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = next((p for p in (ROOT / "ffmpeg.exe", ROOT / "ffmpeg") if p.is_file()), None) or shutil.which("ffmpeg")
TITLE = "ロキ"


class DecoderInputTests(unittest.TestCase):
    def test_the_file_protocol_alone(self):
        args = decoder._input("pipe:1.mp3")
        self.assertEqual(args[:3], ["-protocol_whitelist", "file", "-i"])
        self.assertEqual(args[3], "file:" + os.path.abspath("pipe:1.mp3"))


@unittest.skipUnless(FFMPEG, "FFmpeg makes the test file and reads it")
class DecoderPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cover = cls.dir / "cover.png"
        cls.source = cls.dir / "source.mp3"
        run = lambda *args: subprocess.run([str(FFMPEG), "-v", "error", *args], check=True,  # noqa: E731
                                           **os_support.hidden_subprocess_kwargs())
        run("-f", "lavfi", "-i", "color=c=red:s=32x32", "-frames:v", "1", str(cover))
        run("-f", "lavfi", "-i", "sine=f=440:d=1", "-i", str(cover), "-map", "0", "-map", "1", "-c:a", "libmp3lame",
            "-c:v", "copy", "-disposition:v", "attached_pic", "-id3v2_version", "3", "-metadata", f"title={TITLE}",
            str(cls.source))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def check(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, path)
        pcm = decoder.decode_to_memmap(str(path), 44100, 2)
        self.assertGreater(pcm.shape[0], 40000)
        self.assertEqual(decoder.read_metadata(str(path)).get("title"), TITLE)
        self.assertTrue(decoder.read_cover(str(path)))
        self.assertEqual(decoder._probe_audio_stream(str(path))[0], 44100)

    def test_a_long_path(self):
        if os_support.path_too_long(self.dir / ("x" * 300)):
            self.skipTest("Windows' long paths are off here")
        path = self.dir.joinpath(*["ラブライブ！サンシャイン!!_" + "x" * 30] * 6, "青空Jumping Heart.mp3")
        self.assertGreater(len(str(path)), os_support.WINDOWS_MAX_PATH)
        self.check(path)

    def test_a_name_in_nfd(self):
        self.check(self.dir / unicodedata.normalize("NFD", "ギガンティックO.T.N.mp3"))

    def test_a_name_starting_with_a_dash(self):
        self.check(self.dir / "-i.mp3")

    @unittest.skipIf(os_support.IS_WINDOWS, "Windows' names hold no ':'")
    def test_a_name_like_a_protocol(self):
        self.check(self.dir / "pipe:1.mp3")


if __name__ == "__main__":
    unittest.main()
