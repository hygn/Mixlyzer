"""analyzer.audio.decoder.audio_packet_hash on files made by an FFmpeg (ffmpeg_fixtures) and read by the app's: MP3,
M4A (AAC) and FLAC, each with tags and a cover. The hash is the definition its docstring gives (another FFmpeg's
packets copied out, hashed, the same); tags written into the file (core.tags.writer: in the padding, and a large
comment that makes the file be written anew) leave it as it is; the audio encoded again changes it; a file without
audio has none."""
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from analyzer.audio import decoder
from core.tags.writer import write_tags
from examples import jams_mxl_sbx1
from ffmpeg_fixtures import APP_FFMPEG, MAKER
from system import os_support

FORMATS = {".mp3": ("-c:a", "libmp3lame", "-b:a", "192k", "-id3v2_version", "3"),
           ".m4a": ("-c:a", "aac", "-b:a", "192k"),
           ".flac": ("-c:a", "flac")}


@unittest.skipUnless(APP_FFMPEG and MAKER, "an FFmpeg makes the test files (ffmpeg_fixtures), the app's reads them")
class AudioPacketHashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.cover = cls.dir / "cover.png"
        cls.make("-f", "lavfi", "-i", "color=c=red:s=32x32", "-frames:v", "1", str(cls.cover))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    @staticmethod
    def make(*args) -> bytes:
        return subprocess.run([str(MAKER), "-v", "error", "-y", *args], check=True, capture_output=True,
                              **os_support.hidden_subprocess_kwargs()).stdout

    def encoded(self, suffix: str, name: str, *extra: str) -> Path:
        out = self.dir / f"{name}{suffix}"
        cover = ("-i", str(self.cover), "-map", "0", "-map", "1", "-c:v", "copy", "-disposition:v", "attached_pic")
        self.make("-f", "lavfi", "-i", "sine=f=440:d=3", *cover, *FORMATS[suffix], *extra,
                  "-metadata", "title=ロキ", str(out))
        return out

    def test_tags_leave_it_the_audio_changes_it(self):
        for suffix in FORMATS:
            with self.subTest(suffix):
                path = self.encoded(suffix, "track")
                first = decoder.audio_packet_hash(str(path))
                self.assertIsNotNone(first)
                self.assertEqual(len(first.sha256), 64)
                self.assertGreater(first.size, 0)
                self.assertEqual(first.sample_rate, 44100)
                self.assertEqual(first.channels, "mono")
                self.assertEqual(first.codec, {".mp3": "mp3", ".m4a": "aac", ".flac": "flac"}[suffix])
                self.assertTrue(first.ffmpeg)
                # the definition: another FFmpeg's packets, copied out and hashed
                copied = self.make("-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "data", "-")
                self.assertEqual(hashlib.sha256(copied).hexdigest(), first.sha256)
                size = path.stat().st_size
                write_tags(path, {"comment": "a comment", "title": "ロキ (TV size)"})
                self.assertEqual(decoder.audio_packet_hash(str(path)), first)
                write_tags(path, {"comment": "x" * 70000})               # past the padding: written anew
                self.assertGreater(path.stat().st_size, size + 60000)
                self.assertEqual(decoder.audio_packet_hash(str(path)), first)
                again = self.encoded(suffix, "again", *(("-compression_level", "0") if suffix == ".flac"
                                                       else ("-b:a", "128k")))
                self.assertNotEqual(decoder.audio_packet_hash(str(again)).sha256, first.sha256)
                # the format's reference reader (examples/jams_mxl_sbx1.py) by another FFmpeg: the same hash
                self.assertEqual(jams_mxl_sbx1.packets_sha256(str(path), str(MAKER)), first.sha256)

    def test_no_audio_no_hash(self):
        self.assertIsNone(decoder.audio_packet_hash(str(self.cover)))
        self.assertIsNone(decoder.audio_packet_hash(str(self.dir / "missing.mp3")))


if __name__ == "__main__":
    unittest.main()
