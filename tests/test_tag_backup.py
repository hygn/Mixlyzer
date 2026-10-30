"""core.tags.backup: a file's tags written with a tag block backup made first, then put back by it, is the file it
was byte for byte (FLAC in place and grown, WAV, AIFF, MP3 with its other frames and ID3v1, MP4 with its moov first
and last); the backup holds the tags, not the audio; a file changed since is refused and left as it is; a whole
file's copy is put back too. The backups go to a temporary folder (never the app folder's)."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import soundfile as sf

from core.tags import backup
from core.tags import writer as tw
from test_tag_writer import AUDIO, FFMPEG, MP3_FRAMES, VALUES, id3_frame, mp3


class TagBackupTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.app = Path(tempfile.mkdtemp())
        patcher = mock.patch.object(backup, "app_folder", return_value=self.app)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.app, ignore_errors=True)

    def write_and_restore(self, path: Path, values: dict = VALUES) -> Path:
        original = path.read_bytes()
        made = []
        tw.write_tags(path, values, about_to_write=lambda p, segments: made.append(backup.tag_block_backup(p, segments)))
        self.assertNotEqual(path.read_bytes(), original)
        (saved,) = made
        self.assertTrue(saved.is_relative_to(self.app / "backups" / "tags"))
        self.assertEqual(backup.backed_up_name(saved), path.name)
        backup.restore(saved, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual([p for p in self.dir.iterdir() if p.name.startswith(".mixlyzer-")], [])
        return saved

    def test_each_format_put_back_as_it_was(self):
        for name, fmt in (("a.flac", "FLAC"), ("a.wav", "WAV"), ("a.aiff", "AIFF")):
            path = self.dir / name
            sf.write(path, AUDIO, 22050, format=fmt)
            with self.subTest(fmt):
                saved = self.write_and_restore(path)
                self.assertLess(saved.stat().st_size, path.stat().st_size // 4)    # the tags, not the audio

    def test_flac_in_place_and_grown(self):
        path = self.dir / "a.flac"
        sf.write(path, AUDIO, 22050, format="FLAC")
        tw.write_tags(path, {"comment": "c"})
        self.write_and_restore(path, {"comment": "d"})              # within the padding: in place
        self.write_and_restore(path, {"comment": "x" * 50_000})     # beyond it: the file written anew

    def test_mp3_with_other_frames_and_id3v1(self):
        other = id3_frame(b"GEOB", b"\x00x\x00\x00Serato Markers2\x00" + bytes(range(256)) * 4)
        v1 = b"TAG" + b"v1 title".ljust(30, b"\x00") + bytes(64) + b"v1 comment".ljust(30, b"\x00") + b"\x00"
        path = self.dir / "a.mp3"
        path.write_bytes(mp3(other + id3_frame(b"TIT2", b"\x00old") + bytes(64), v1))
        saved = self.write_and_restore(path)
        self.assertLess(saved.stat().st_size, len(MP3_FRAMES))

    @unittest.skipUnless(FFMPEG, "FFmpeg makes the MP4 files")
    def test_mp4_moov_first_and_last(self):
        for flags in ([], ["-movflags", "+faststart"]):
            path = self.dir / ("fast.m4a" if flags else "end.m4a")
            subprocess.run([str(FFMPEG), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=2", "-c:a", "aac",
                            *flags, "-metadata", "title=old", str(path)], check=True)
            with self.subTest(path.name):
                self.write_and_restore(path, {**VALUES, "comment": "y" * 20_000})

    def test_tags_written_again_since_still_put_back(self):
        # the second write changes what the backup holds itself (the tags in the padding): the file rebuilt is
        # the one backed up, its hash says so
        path = self.dir / "a.flac"
        sf.write(path, AUDIO, 22050, format="FLAC")
        original = path.read_bytes()
        made = []
        tw.write_tags(path, {"title": "ヴヴヴ"}, about_to_write=lambda p, s: made.append(backup.tag_block_backup(p, s)))
        tw.write_tags(path, {"title": "ヴヴヴ (TV size)"})
        backup.restore(made[0], path)
        self.assertEqual(path.read_bytes(), original)

    def test_a_file_changed_since_refused(self):
        path = self.dir / "a.flac"
        sf.write(path, AUDIO, 22050, format="FLAC")
        made = []
        tw.write_tags(path, {"title": "ヴヴヴ"}, about_to_write=lambda p, s: made.append(backup.tag_block_backup(p, s)))
        data = bytearray(path.read_bytes())
        data[-100] ^= 0xFF                                          # its audio changed since
        path.write_bytes(bytes(data))
        changed = path.read_bytes()
        with self.assertRaises(backup.BackupError):
            backup.restore(made[0], path)
        self.assertEqual(path.read_bytes(), changed)
        path.write_bytes(changed + b"x")                            # another size
        with self.assertRaises(backup.BackupError):
            backup.restore(made[0], path)
        changed = path.read_bytes()
        not_ours = self.dir / "x.mxtags"
        not_ours.write_bytes(b"something else")
        with self.assertRaises(backup.BackupError):
            backup.restore(not_ours, path)
        self.assertEqual(path.read_bytes(), changed)

    def test_nothing_written_no_backup_left(self):
        path = self.dir / "a.flac"
        sf.write(path, AUDIO, 22050, format="FLAC")

        def refuse(p, segments):
            backup.tag_block_backup(p, segments)
            raise OSError("disk full")

        original = path.read_bytes()
        with self.assertRaises(OSError):
            tw.write_tags(path, VALUES, about_to_write=refuse)
        self.assertEqual(path.read_bytes(), original)

    def test_a_whole_file_copy(self):
        path = self.dir / "a.wav"
        sf.write(path, AUDIO, 22050, format="WAV")
        original = path.read_bytes()
        saved = backup.backup_copy(path)
        self.assertEqual((saved.suffix, saved.read_bytes()), (".wav", original))
        self.assertEqual(backup.backed_up_name(saved), "")
        tw.write_tags(path, VALUES)
        backup.restore(saved, path)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
