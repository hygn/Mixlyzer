"""third_party.rekordbox_phrases: rekordbox's beats read from a .DAT's PQTZ, the phrases put on them in the mid mood
(a run of verses one number, SILENCE going on with the phrase before), the .EXT's PSSI replaced with its other tags
as they are, and a write refused while rekordbox runs or after the file changed. Synthetic ANLZ files in a
temporary folder; rekordbox's library is never opened."""
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.music.marks import Phrase
from third_party import rekordbox_phrases as rp


def tag(fourcc: bytes, header_rest: bytes, body: bytes) -> bytes:
    return fourcc + struct.pack(">II", 12 + len(header_rest), 12 + len(header_rest) + len(body)) + header_rest + body


def anlz(*tags: bytes) -> bytes:
    body = b"".join(tags)
    return b"PMAI" + struct.pack(">IIIIII", 28, 28 + len(body), 1, 0x10000, 0x10000, 0) + body


def dat(beats_ms) -> bytes:
    body = b"".join(struct.pack(">HHI", i % 4 + 1, 12000, ms) for i, ms in enumerate(beats_ms))
    return anlz(tag(b"PQTZ", struct.pack(">III", 0, 0x80000, len(beats_ms)), body))


BEATS = 0.25 + 0.5 * np.arange(200)          # rekordbox's beats: 120 BPM from 0.25 s


class PhraseTests(unittest.TestCase):
    def test_the_beats_of_the_dat(self):
        np.testing.assert_allclose(rp.beat_times(dat([250 + 500 * i for i in range(200)])), BEATS)

    def test_the_phrases_on_rekordboxs_beats(self):
        phrases = [Phrase(0.0, 8.2, "INTRO"), Phrase(8.2, 24.3, "VERSE"), Phrase(24.3, 40.2, "VERSE"),
                   Phrase(40.2, 56.0, "CHORUS"), Phrase(56.0, 58.0, "SILENCE"), Phrase(58.0, 72.0, "VERSE"),
                   Phrase(72.0, 90.0, "OUTRO")]
        entries, end, losses, off = rp.phrase_entries(phrases, BEATS)
        # each from the beat nearest its start (1-based); the second run of verses Verse 2
        self.assertEqual(entries, [(1, 1), (17, 2), (49, 2), (81, 9), (116, 3), (144, 10)])
        self.assertEqual(end, 180)
        self.assertEqual(losses, ["continued:SILENCE"])
        self.assertAlmostEqual(off, 0.25)                    # the VERSE at 0:58.0, between two beats
        # on the beats: an INTRO from 0:00 (put on the first beat, 0.25 s, by its nature) not counted; a grid 0.1 s
        # off rekordbox's: every phrase after the first that far from its beat
        on_beats = [Phrase(0.0, 8.25, "INTRO"), Phrase(8.25, 24.25, "VERSE"), Phrase(24.25, 40.0, "CHORUS")]
        self.assertAlmostEqual(rp.phrase_entries(on_beats, BEATS)[3], 0.0)
        shifted = [Phrase(p.start + 0.1 if p.start else 0.0, p.end + 0.1, p.label) for p in on_beats]
        self.assertAlmostEqual(rp.phrase_entries(shifted, BEATS)[3], 0.1)
        # the high mood, the verses Down, SILENCE an Outro: as the settings say
        mapping = rp.phrase_map(rp.MOOD_HIGH, "VERSE=down,SILENCE=outro,CHORUS=bogus")
        entries, _end, losses, _off = rp.phrase_entries(phrases, BEATS, rp.MOOD_HIGH, mapping)
        self.assertEqual(entries, [(1, 1), (17, 3), (49, 3), (81, 5), (112, 6), (116, 3), (144, 6)])
        self.assertEqual(losses, [])
        # the low mood's two verses: the third run Verse 2 again, said
        three = phrases + [Phrase(90.0, 92.0, "CHORUS"), Phrase(92.0, 99.0, "VERSE")]
        entries, _end, losses, _off = rp.phrase_entries(three, BEATS, rp.MOOD_LOW, rp.phrase_map(rp.MOOD_LOW))
        self.assertEqual([kind for _b, kind in entries if kind in (2, 5)], [2, 2, 5, 5])     # a run of two, then two
        self.assertIn("verses", losses)

    def test_the_mapping_as_the_setting_holds_it(self):
        mapping = rp.phrase_map(rp.MOOD_MID, "VERSE=verse3")
        self.assertEqual(rp.map_text(mapping, rp.MOOD_MID), "VERSE=verse3")       # the defaults left out
        self.assertEqual(rp.map_text(rp.phrase_map(rp.MOOD_MID), rp.MOOD_MID), "")
        self.assertEqual(rp.phrase_map(rp.MOOD_HIGH, "VERSE=verse3")["VERSE"], "up")   # not the mood's: its default
        self.assertEqual(rp.choice_name(rp.MOOD_MID, rp.BY_RUN), "Verse 1-6")

    def test_the_pssi_replaced_and_the_other_tags_kept(self):
        wave = tag(b"PWV3", struct.pack(">III", 1, 4, 0), b"\x01\x02\x03\x04")
        old = rp.pssi_tag([(1, 1), (33, 9)], 64, rp.MOOD_HIGH, bank=3)
        ext = anlz(wave, old, tag(b"PVB2", b"", b"\x00" * 8))
        new = rp.with_phrases(ext, rp.pssi_tag([(1, 1), (17, 2), (49, 9)], 80, rp.MOOD_MID, bank=3))
        self.assertEqual(rp.read_phrases(new), (rp.MOOD_MID, 3, 80, [(1, 1), (17, 2), (49, 9)]))
        self.assertEqual(struct.unpack(">I", new[8:12])[0], len(new))
        self.assertEqual([t for t, _a, _l in rp._tags(new)], [b"PWV3", b"PSSI", b"PVB2"])
        self.assertIn(wave, new)
        # a file without one: added at the end
        added = rp.with_phrases(anlz(wave), rp.pssi_tag([(1, 1)], 8))
        self.assertEqual([t for t, _a, _l in rp._tags(added)], [b"PWV3", b"PSSI"])
        with self.assertRaises(rp.RekordboxError):
            rp.with_phrases(b"not an analysis file", old)

    def test_written_with_a_backup_and_refused_while_rekordbox_runs_or_after_a_change(self):
        with tempfile.TemporaryDirectory() as folder:
            share = Path(folder) / "USBANLZ" / "abc"
            share.mkdir(parents=True)
            (share / "ANLZ0000.DAT").write_bytes(dat([250 + 500 * i for i in range(200)]))
            (share / "ANLZ0000.EXT").write_bytes(anlz(rp.pssi_tag([(1, 1)], 200)))
            files = (share / "ANLZ0000.DAT", share / "ANLZ0000.EXT")
            phrases = [Phrase(0.0, 16.2, "INTRO"), Phrase(16.2, 48.0, "CHORUS")]
            with mock.patch.object(rp, "_analysis_files", return_value=files), \
                    mock.patch.object(rp, "app_folder", return_value=Path(folder) / "app"), \
                    mock.patch.object(rp, "running", return_value=False) as running:
                plan = rp.plan("ヴヴヴ.flac", phrases)
                self.assertEqual([name for name, _t in plan.now], ["Intro"])
                self.assertEqual([name for name, _t in plan.after], ["Intro", "Chorus"])
                running.return_value = True
                with self.assertRaises(rp.RekordboxError):
                    rp.write(plan)
                running.return_value = False
                backup = rp.write(plan)
                self.assertEqual(backup.read_bytes(), plan.data)
                self.assertTrue(str(backup).startswith(os.path.join(folder, "app", "backups", "rekordbox")))
                self.assertEqual(rp.read_phrases(files[1].read_bytes())[3], [(1, 1), (33, 9)])
                with self.assertRaises(rp.RekordboxError):       # the file changed since the plan was made
                    rp.write(plan)
                # the whole analysis folder kept; none
                plan = rp.plan("ヴヴヴ.flac", phrases[:1])
                folder_copy = rp.write(plan, "folder")
                self.assertEqual(sorted(p.name for p in folder_copy.iterdir()), ["ANLZ0000.DAT", "ANLZ0000.EXT"])
                self.assertEqual((folder_copy / "ANLZ0000.EXT").read_bytes(), plan.data)
                plan = rp.plan("ヴヴヴ.flac", phrases)
                self.assertIsNone(rp.write(plan, ""))
                self.assertEqual(files[1].read_bytes(), plan.new)
                # the track's backups, newest first (the .EXT, the folder); the first put back
                found = rp.backups_of(share)
                self.assertEqual(len(found), 2)
                self.assertEqual(sorted(rp.restore(backup, share)), ["ANLZ0000.EXT"])
                self.assertEqual(files[1].read_bytes(), backup.read_bytes())
                self.assertEqual(sorted(rp.restore(folder_copy, share)), ["ANLZ0000.DAT", "ANLZ0000.EXT"])
                running.return_value = True
                with self.assertRaises(rp.RekordboxError):
                    rp.restore(backup, share)
                running.return_value = False
                junk = Path(folder) / "app" / "backups" / "rekordbox" / "x" / "abc 000000 notes.txt"
                junk.parent.mkdir(parents=True)
                junk.write_text("no analysis here")
                with self.assertRaises(rp.RekordboxError):       # not an ANLZ file: nothing put back
                    rp.restore(junk, share)


if __name__ == "__main__":
    unittest.main()
