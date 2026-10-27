import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from core.tags import writer as tw
from core.library.db import LibraryDB, TrackRow

AUDIO = (0.3 * np.sin(np.linspace(0, 200 * np.pi, 22050))).astype(np.float32)
VALUES = {"title": "テスト ☆ 제목", "artist": "Ａｒｔｉｓｔ", "album": "Album", "comment": "c1\nc2", "bpm": 174.5,
          "key": "F#m"}
ROOT = Path(__file__).resolve().parents[1]   # the app's own FFmpeg beside it (as the decoder looks), else PATH's
FFMPEG = next((p for p in (ROOT / "ffmpeg.exe", ROOT / "ffmpeg") if p.is_file()), None) or shutil.which("ffmpeg")


def ss(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def id3_frame(fid: bytes, data: bytes) -> bytes:
    return fid + len(data).to_bytes(4, "big") + b"\x00\x00" + data


MP3_FRAMES = (bytes([0xFF, 0xFB, 0x90, 0x64]) + bytes(413)) * 20


def mp3(body: bytes, trailer: bytes = b"") -> bytes:
    return b"ID3\x03\x00\x00" + ss(len(body)) + body + MP3_FRAMES + trailer


class TagWriterTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def roundtrip(self, path: Path, audio_from) -> None:
        """Every field written where the format keeps it, read back, the audio the same; then fields removed,
        the others kept."""
        before = audio_from(path)
        tw.write_tags(path, VALUES)
        got = tw.read_tags(path)
        expect = {**{k: str(v) for k, v in VALUES.items()}, "bpm": "174.5"}
        if tw.tag_format(path) == "mp4":
            expect["bpm"] = "174"
        self.assertEqual({k: got.get(k) for k in expect}, expect)
        self.assertEqual(audio_from(path), before)
        tw.write_tags(path, {"comment": "", "key": ""})
        found = tw.read_spellings(path)
        for places in tw.targets(tw.tag_format(path), ["comment", "key"]).values():
            for spelling in places:
                self.assertNotIn(spelling, found)
        self.assertEqual(tw.read_tags(path)["title"], VALUES["title"])
        self.assertEqual(audio_from(path), before)
        self.assertEqual(list(self.dir.glob(".mixlyzer-tags-*")), [])

    def sf_audio(self, path: Path) -> bytes:
        return sf.read(path, dtype="float32")[0].tobytes()

    def test_flac_wav_aiff(self):
        for name, fmt in (("a.flac", "FLAC"), ("a.wav", "WAV"), ("a.aiff", "AIFF")):
            path = self.dir / name
            sf.write(path, AUDIO, 22050, format=fmt)
            with self.subTest(fmt):
                self.roundtrip(path, self.sf_audio)

    def test_flac_grows_beyond_padding_and_back_in_place(self):
        path = self.dir / "a.flac"
        sf.write(path, AUDIO, 22050, format="FLAC")
        before = self.sf_audio(path)
        tw.write_tags(path, {"comment": "x" * 50_000})
        size = path.stat().st_size
        tw.write_tags(path, {"comment": "short"})
        self.assertEqual(path.stat().st_size, size)   # in the padding left the first time
        self.assertEqual(tw.read_tags(path)["comment"], "short")
        self.assertEqual(self.sf_audio(path), before)

    def test_written_where_the_value_came_from(self):
        # FLAC: the artist read from PERFORMER (MediaInfo takes it first) goes back there; ARTIST stays
        path = self.dir / "a.flac"
        sf.write(path, AUDIO, 22050, format="FLAC")
        tw.write_tags(path, {"artist": "Composer"})
        tw.write_tags(path, {"artist": "Orchestra"}, {"artist": ["vorbis:PERFORMER"]})
        sources = tw.tag_sources(path, {"title": "", "artist": "Orchestra", "album": "", "comment": ""})
        self.assertEqual(sources["artist"], ["vorbis:PERFORMER"])
        self.assertEqual(sources["title"], [])
        tw.write_tags(path, {"artist": "New Orchestra"}, sources)
        found = tw.read_spellings(path)
        self.assertEqual(found["vorbis:PERFORMER"], ["New Orchestra"])
        self.assertEqual(found["vorbis:ARTIST"], ["Composer"])

    def test_a_source_the_format_lacks_goes_to_its_default_places(self):
        path = self.dir / "a.wav"   # the file was a FLAC once: its sources say PERFORMER
        sf.write(path, AUDIO, 22050, format="WAV")
        self.assertEqual(tw.targets("wav", ["artist"], {"artist": ["vorbis:PERFORMER"]})["artist"],
                         ("info:IART", "id3:TPE1"))
        tw.write_tags(path, {"artist": "A"}, {"artist": ["vorbis:PERFORMER"]})
        found = tw.read_spellings(path)
        self.assertEqual((found["info:IART"], found["id3:TPE1"]), (["A"], ["A"]))

    def test_mp3_keeps_other_frames_and_the_audio_bytes(self):
        other = id3_frame(b"GEOB", b"\x00x\x00\x00Serato Markers2\x00" + bytes(range(256)) * 4)
        txxx = id3_frame(b"TXXX", b"\x00comment\x00ffmpeg's comment")
        album_artist = id3_frame(b"TPE2", b"\x00Album Artist")
        path = self.dir / "a.mp3"
        path.write_bytes(mp3(other + txxx + album_artist + bytes(64)))
        tw.write_tags(path, {"comment": "c"})   # no source: the comment's copies the file has (FFmpeg's TXXX)
        found = tw.read_spellings(path)
        self.assertEqual(found["id3:TXXX:comment"], ["c"])
        self.assertNotIn("id3:COMM", found)
        self.roundtrip(path, lambda p: p.read_bytes()[-len(MP3_FRAMES):])
        found = tw.read_spellings(path)
        self.assertNotIn("id3:TXXX:comment", found)   # emptied with its copies
        self.assertIn(other, path.read_bytes())
        # an artist read from TPE1: the album artist (a fallback) is left, also when the artist is emptied
        tw.write_tags(path, {"artist": "A"}, {"artist": ["id3:TPE1"]})
        tw.write_tags(path, {"artist": ""}, {"artist": ["id3:TPE1"]})
        found = tw.read_spellings(path)
        self.assertNotIn("id3:TPE1", found)
        self.assertEqual(found["id3:TPE2"], ["Album Artist"])
        # an artist read from the album artist goes back there, and is emptied there
        tw.write_tags(path, {"artist": "AA"}, {"artist": ["id3:TPE2"]})
        self.assertEqual(tw.read_spellings(path)["id3:TPE2"], ["AA"])
        tw.write_tags(path, {"artist": ""}, {"artist": ["id3:TPE2"]})
        self.assertNotIn("id3:TPE2", tw.read_spellings(path))

    def test_a_value_read_from_id3v1_goes_to_id3v2(self):
        v1 = b"TAG" + b"v1 title".ljust(30, b"\x00") + bytes(64) + b"v1 comment".ljust(30, b"\x00") + b"\x00"
        path = self.dir / "a.mp3"
        path.write_bytes(mp3(id3_frame(b"TPE1", b"\x00A") + bytes(32), v1))
        sources = tw.tag_sources(path, {"title": "v1 title", "artist": "A", "album": "", "comment": "v1 comment"})
        self.assertEqual((sources["title"], sources["comment"]), (["id3v1:title"], ["id3v1:comment"]))
        tw.write_tags(path, {"title": "長いタイトル", "comment": "c"}, sources)
        found = tw.read_spellings(path)
        self.assertEqual((found["id3:TIT2"], found["id3:COMM"]), (["長いタイトル"], ["c"]))
        self.assertEqual(found["id3v1:title"], ["v1 title"])   # ID3v1 is not written
        self.assertTrue(path.read_bytes().endswith(v1))

    def test_unsupported_and_damaged_files_stay(self):
        cases = {
            "a.ogg": b"OggS" + bytes(100),
            "v22.mp3": b"ID3\x02\x00\x00" + ss(20) + bytes(20) + bytes([0xFF, 0xFB]) + bytes(100),
            "bad.mp3": b"ID3\x03\x00\x00" + ss(30) + b"tit2" + bytes(26) + bytes([0xFF, 0xFB]) + bytes(100),
            "cut.flac": b"fLaC" + bytes([0x00, 0x00, 0x00, 0x22]) + bytes(10),
        }
        for name, data in cases.items():
            path = self.dir / name
            path.write_bytes(data)
            with self.subTest(name):
                with self.assertRaises(tw.TagWriteError):
                    tw.write_tags(path, VALUES)
                self.assertEqual(path.read_bytes(), data)
        self.assertIsNone(tw.tag_format(self.dir / "a.ogg"))
        self.assertIsNone(tw.tag_format(self.dir / "v22.mp3"))

    @unittest.skipUnless(FFMPEG, "FFmpeg makes the MP4 files")
    def test_mp4_moov_first_and_last(self):
        def ff_audio(p: Path) -> bytes:
            return subprocess.run([str(FFMPEG), "-v", "error", "-i", str(p), "-f", "f32le", "-"],
                                  capture_output=True, check=True).stdout

        for flags in ([], ["-movflags", "+faststart"]):
            path = self.dir / ("fast.m4a" if flags else "end.m4a")
            subprocess.run([str(FFMPEG), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=2", "-c:a", "aac",
                            *flags, "-metadata", "title=old", str(path)], check=True)
            with self.subTest(path.name):
                self.roundtrip(path, ff_audio)
                tw.write_tags(path, {"comment": "y" * 20_000})   # moov grows: the offsets follow
                self.assertEqual(tw.read_tags(path)["comment"], "y" * 20_000)
                self.assertEqual(ff_audio(path), ff_audio(path))


def apic(ptype: int, image: bytes) -> bytes:
    return id3_frame(b"APIC", b"\x00image/jpeg\x00" + bytes([ptype]) + b"\x00" + image)


def flac_with_picture(path: Path, mime: bytes, image: bytes) -> None:
    """A FLAC file with a PICTURE block (front cover) right after its STREAMINFO."""
    sf.write(path, AUDIO, 22050, format="FLAC")
    data = bytearray(path.read_bytes())
    last = data[4] & 0x80
    data[4] &= 0x7F
    body = ((3).to_bytes(4, "big") + len(mime).to_bytes(4, "big") + mime + (0).to_bytes(4, "big") + bytes(16)
            + len(image).to_bytes(4, "big") + image)
    block = bytes([6 | last]) + len(body).to_bytes(3, "big") + body
    path.write_bytes(bytes(data[:42]) + block + bytes(data[42:]))


class TagReaderTests(unittest.TestCase):
    """The analysis's own tag and picture readers (no native tag library parses an outside file)."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_levels_values_joined_and_kept_as_written(self):
        path = self.dir / "a.mp3"
        path.write_bytes(mp3(id3_frame(b"TPE1", b"\x03A") + id3_frame(b"TPE1", b"\x03B")
                             + id3_frame(b"TIT2", b"\x03X / Y")))
        self.assertEqual(tw.read_fields(path), {"artist": "A / B", "title": "X / Y"})

    def test_pictures_front_cover_first_whatever_the_mime(self):
        path = self.dir / "a.mp3"
        path.write_bytes(mp3(apic(0, b"\xff\xd8\xffother") + apic(3, b"\xff\xd8\xfffront")))
        self.assertEqual(tw.read_picture(path), b"\xff\xd8\xfffront")
        path = self.dir / "a.flac"
        flac_with_picture(path, b"application/jpg", b"\xff\xd8\xffcover")   # a MIME FFmpeg does not take
        self.assertEqual(tw.read_picture(path), b"\xff\xd8\xffcover")
        self.assertEqual(tw.read_tags(path), {})

    def test_a_compressed_frame_unpacking_too_far_is_not_read(self):
        import time
        import zlib
        bomb = zlib.compress(b"\x03" + b"a" * (100 << 20), 9)
        frame = b"TIT2" + ss(len(bomb)) + b"\x00\x08" + bomb   # v2.4: compressed
        path = self.dir / "bomb.mp3"
        path.write_bytes(b"ID3\x04\x00\x00" + ss(len(frame)) + frame + MP3_FRAMES)
        start = time.perf_counter()
        self.assertNotIn("id3:TIT2", tw.read_spellings(path))
        self.assertLess(time.perf_counter() - start, 5.0)

    def test_ffmetadata(self):
        from analyzer.audio.decoder import _ffmetadata
        bs = chr(92)
        text = (f";FFMETADATA1\ntitle=a{bs}=b{bs};c{bs}#d{bs}{bs}e\ncomment=line1{bs}\nline2\n; a comment\n"
                "#another\n[STREAM]\ntitle=not global\n")
        self.assertEqual(_ffmetadata(text), {"title": f"a=b;c#d{bs}e", "comment": "line1\nline2"})
        self.assertEqual(_ffmetadata("not metadata"), {})


class TagSourceStoreTests(unittest.TestCase):
    def test_sources_kept_cleared_with_the_file_and_the_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LibraryDB(str(Path(tmp) / "library.db"))
            db.connect()
            try:
                uid = str(uuid4())
                db.upsert_many([TrackRow(uid=uid, path=str(Path(tmp) / "a.flac"), title="T")])
                db.set_tag_sources(uid, {"title": [], "artist": ["vorbis:PERFORMER", "vorbis:ARTIST"]})
                self.assertEqual(db.tag_sources(uid), {"artist": ["vorbis:PERFORMER", "vorbis:ARTIST"], "title": []})
                db.set_tag_sources(uid, {"artist": ["vorbis:ARTIST"]})   # the other fields as they are
                self.assertEqual(db.tag_sources(uid), {"artist": ["vorbis:ARTIST"], "title": []})
                other = Path(tmp) / "b.m4a"
                other.write_bytes(b"x")
                db.set_path(uid, str(other))   # another file: found again
                self.assertEqual(db.tag_sources(uid), {})
                db.set_tag_sources(uid, {"title": ["mp4:titl"]})
                db.delete_tracks([uid])
                count = db.conn.execute("SELECT COUNT(*) FROM track_tag_sources").fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
