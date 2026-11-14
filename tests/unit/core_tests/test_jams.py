"""third_party.jams: a track's marks as a JAMS document (JAMS-mxl-sbx1), checked against JAMS 0.3's schema as its
text states it (every observation's four fields, times not negative, each namespace's values: beat a number, tempo a
BPM with a confidence 0-1, key_mode its pattern, segment_open a string) and the extension's own parts: the beats'
places in their bars (a pickup counted back from the first downbeat, a junction's short bar), the keys spelled as
their signatures, a ramp's end BPM beside its row, the phrases without UNASSIGNED, the CUE points and JumpCUEs as
blobs, each time exactly as the library stores it (a float32 mark widened, equal to the beat it sits on), the
file's identity in the sandbox, the text ASCII."""
import json
import math
import re
import unittest

import numpy as np

from analyzer.audio.decoder import AudioPacketHash
from analyzer.music.marks import CuePoint, JumpCue, JumpCueGroup, Phrase
from core.marks.phrases import UNASSIGNED
from third_party.jams import FORMAT, SANDBOX, beat_positions, jams_document, jams_text, key_mode_value

# key_mode's value pattern as JAMS 0.3's namespace states it
KEY_MODE = re.compile(r"^N|([A-G][b#]?)(:(major|minor|ionian|dorian|phrygian|lydian|mixolydian|aeolian|locrian))?$")
AUDIO = AudioPacketHash("ab" * 32, 1234, "flac", 44100, "stereo", "n7.1-test")


def document(**changes) -> dict:
    marks = dict(
        beats=0.5 * np.arange(1, 20),                                   # 120 BPM from 0.5 s
        tempo_segments=np.asarray([[0.0, 8.0, 120.0, 1.5, 4.0, 120.0],   # its first downbeat the 3rd beat
                                   [8.0, 10.0, 120.0, 8.0, 3.0, 132.0]]),
        key_segments=np.asarray([[1.0, 0.0, 0.0, 6.0], [10.0, 1.0, 6.0, 10.0], [3.0, 1.0, np.nan, 11.0]]),
        phrases=[Phrase(0.0, 1.5, UNASSIGNED), Phrase(1.5, 5.5, "INTRO"), Phrase(5.5, 10.0, "CHORUS")],
        cue_points=[CuePoint(5.5, "CHORUS", "drop", 0)],
        jump_cues=[JumpCueGroup((JumpCue("A", "", 1.5, 3.5, 1.5), JumpCue("B", "last", 5.5, 9.5, 5.5)), 1)],
        duration_sec=10.5, title="ヴヴヴ", artist="アーティスト", album="", path="C:/music/ヴヴヴ.flac",
        file_size=99, uid="0f8fad5b-d9cb-469f-a165-70867728950e", audio=AUDIO, total_samples=463050,
        app_version="0.1.0")
    marks.update(changes)
    return jams_document(**marks)


def annotation(doc: dict, namespace: str, kind: str | None = None) -> dict:
    found = [a for a in doc["annotations"] if a["namespace"] == namespace
             and (kind is None or a.get("sandbox", {}).get(SANDBOX, {}).get("kind") == kind)]
    assert len(found) == 1, (namespace, kind)
    return found[0]


class SchemaTests(unittest.TestCase):
    def test_jams_shape_and_namespaces(self):
        doc = document()
        self.assertEqual(set(doc), {"file_metadata", "annotations", "sandbox"})
        meta = doc["file_metadata"]
        self.assertRegex(meta["jams_version"], r"^[0-9]\.[0-9]\.[0-9]$")
        self.assertEqual((meta["title"], meta["release"], meta["duration"]), ("ヴヴヴ", "", 10.5))
        for a in doc["annotations"]:
            self.assertTrue({"annotation_metadata", "data", "namespace"} <= set(a))
            self.assertEqual(a["annotation_metadata"]["annotation_tools"], "Mixlyzer 0.1.0")
            for o in a["data"]:
                self.assertEqual(set(o), {"time", "duration", "value", "confidence"})
                self.assertGreaterEqual(o["time"], 0.0)
                self.assertGreaterEqual(o["duration"], 0.0)
        self.assertTrue(all(isinstance(o["value"], int) for o in annotation(doc, "beat")["data"]))
        for o in annotation(doc, "tempo")["data"]:
            self.assertGreater(o["value"], 0)
            self.assertTrue(0.0 <= o["confidence"] <= 1.0)
        for o in annotation(doc, "key_mode")["data"]:
            self.assertRegex(o["value"], KEY_MODE)
        self.assertTrue(all(isinstance(o["value"], str) for o in annotation(doc, "segment_open")["data"]))

    def test_text_is_ascii_json_without_nan(self):
        text = jams_text(document())
        self.assertTrue(text.isascii())

        def refuse(name):
            raise AssertionError(f"{name} in the JSON")
        self.assertEqual(json.loads(text, parse_constant=refuse)["file_metadata"]["title"], "ヴヴヴ")


class MarksTests(unittest.TestCase):
    def test_beats_in_their_bars(self):
        beats = annotation(document(), "beat")["data"]
        # the pickup (0.5, 1.0) counted back from the downbeat at 1.5; at the junction a bar of one beat (7.5), then
        # the second row's bars of 3 from 8.0
        self.assertEqual([o["value"] for o in beats],
                         [3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 2, 3, 1])
        self.assertEqual([o["time"] for o in beats][:3], [0.5, 1.0, 1.5])
        self.assertEqual(beat_positions(np.empty(0), None).size, 0)

    def test_tempo_rows_with_their_end_bpm(self):
        tempo = annotation(document(), "tempo")
        self.assertEqual([(o["time"], o["duration"], o["value"]) for o in tempo["data"]],
                         [(0.0, 8.0, 120.0), (8.0, 2.0, 120.0)])
        self.assertEqual(tempo["sandbox"][SANDBOX]["bpm_end"], [120.0, 132.0])

    def test_keys_spelled_as_their_signatures(self):
        keys = annotation(document(), "key_mode")["data"]
        self.assertEqual([(o["time"], o["value"]) for o in keys], [(0.0, "Db:major"), (6.0, "Bb:minor")])  # NaN out
        names = [key_mode_value(p, m) for m in (False, True) for p in range(12)]
        self.assertEqual(names, ["C:major", "Db:major", "D:major", "Eb:major", "E:major", "F:major", "F#:major",
                                 "G:major", "Ab:major", "A:major", "Bb:major", "B:major",
                                 "C:minor", "C#:minor", "D:minor", "D#:minor", "E:minor", "F:minor", "F#:minor",
                                 "G:minor", "G#:minor", "A:minor", "Bb:minor", "B:minor"])

    def test_phrases_without_unassigned(self):
        phrases = annotation(document(), "segment_open")["data"]
        self.assertEqual([(o["time"], o["duration"], o["value"]) for o in phrases],
                         [(1.5, 4.0, "INTRO"), (5.5, 4.5, "CHORUS")])

    def test_cues_as_blobs(self):
        doc = document()
        (cue,) = annotation(doc, "blob", "cue_points")["data"]
        self.assertEqual((cue["time"], cue["duration"]), (5.5, 0.0))
        self.assertEqual(cue["value"], {"name": "CHORUS", "comment": "drop", "color": "#ff2828"})
        jumps = annotation(doc, "blob", "jump_cues")["data"]
        self.assertEqual([o["time"] for o in jumps], [1.5, 5.5])
        self.assertEqual(jumps[1]["value"], {"slot": "B", "name": "last", "group": 0, "color": "#0099cc",
                                             "passage": [5.5, 9.5]})

    def test_times_exactly_as_stored(self):
        # a phrase and a JumpCUE on a beat whose float64 is a float32's (as the library has them), a CUE point and
        # a passage's end edited in the deck (float64) written as saving stores them (float32)
        start = float(np.float32(1.306975))
        end, cue = 16.075950123456, 88.960311234567
        doc = document(beats=np.asarray([start, 1.7685058520316548]), phrases=[Phrase(start, end, "INTRO")],
                       cue_points=[CuePoint(cue)],
                       jump_cues=[JumpCueGroup((JumpCue("A", "", start, end, start),), 0)])
        text = json.loads(jams_text(doc))      # as a JSON reader takes them (float64)
        beat = annotation(text, "beat")["data"][0]["time"]
        (phrase,) = annotation(text, "segment_open")["data"]
        (jump,) = annotation(text, "blob", "jump_cues")["data"]
        self.assertEqual(annotation(text, "beat")["data"][1]["time"], 1.7685058520316548)   # float64 in full
        self.assertEqual(phrase["time"], beat)                       # on the beat, the very value
        self.assertEqual(jump["time"], beat)
        self.assertEqual(jump["value"]["passage"][1], float(np.float32(end)))
        self.assertEqual(annotation(text, "blob", "cue_points")["data"][0]["time"], float(np.float32(cue)))
        self.assertEqual(phrase["duration"], float(np.float32(end)) - start)

    def test_the_file_identified(self):
        mx = document()["sandbox"][SANDBOX]
        self.assertEqual(mx["format"], FORMAT)
        self.assertEqual(mx["uid"], "0f8fad5b-d9cb-469f-a165-70867728950e")
        self.assertEqual(mx["file"], {"path": "C:/music/ヴヴヴ.flac", "size": 99, "audio": {
            "packets_sha256": "ab" * 32, "packets_bytes": 1234, "codec": "flac", "sample_rate": 44100,
            "channels": "stereo", "samples": 463050, "ffmpeg": "n7.1-test"}})
        self.assertIsNone(document(total_samples=None)["sandbox"][SANDBOX]["file"]["audio"]["samples"])

    def test_path_and_uid_left_out(self):
        mx = document(path=None, uid=None)["sandbox"][SANDBOX]
        self.assertNotIn("uid", mx)
        self.assertNotIn("path", mx["file"])
        self.assertEqual(mx["file"]["audio"]["packets_sha256"], "ab" * 32)

    def test_a_track_without_marks(self):
        doc = document(beats=np.empty(0), tempo_segments=np.empty((0, 6)), key_segments=np.empty((0, 4)),
                       phrases=[], cue_points=[], jump_cues=[])
        self.assertTrue(all(a["data"] == [] for a in doc["annotations"]))
        self.assertFalse(math.isnan(doc["file_metadata"]["duration"]))


class ReferenceReaderTests(unittest.TestCase):
    """examples/jams_mxl_sbx1.py, the format's reference reader, reads back what the writer writes."""

    def test_round_trip(self):
        from examples import jams_mxl_sbx1 as reader
        track = reader.parse(json.loads(jams_text(document())))
        self.assertEqual((track.title, track.artist, track.duration, track.app_version), ("ヴヴヴ", "アーティスト", 10.5,
                                                                                          "0.1.0"))
        self.assertEqual((track.uid, track.path, track.size), ("0f8fad5b-d9cb-469f-a165-70867728950e",
                                                               "C:/music/ヴヴヴ.flac", 99))
        self.assertEqual(track.audio, reader.AudioIdentity("ab" * 32, 1234, "flac", 44100, "stereo", 463050,
                                                           "n7.1-test"))
        self.assertEqual(track.beats[:3], (reader.Beat(0.5, 3), reader.Beat(1.0, 4), reader.Beat(1.5, 1)))
        self.assertEqual(track.downbeats, (1.5, 3.5, 5.5, 7.5, 8.0, 9.5))
        self.assertEqual(track.tempo, (reader.TempoRow(0.0, 8.0, 120.0, 120.0), reader.TempoRow(8.0, 10.0, 120.0, 132.0)))
        self.assertEqual([r.is_ramp for r in track.tempo], [False, True])
        self.assertEqual([(k.start, k.tonic, k.minor) for k in track.keys], [(0.0, "Db", False), (6.0, "Bb", True)])
        self.assertEqual(track.phrases, (reader.Phrase(1.5, 5.5, "INTRO"), reader.Phrase(5.5, 10.0, "CHORUS")))
        self.assertEqual(track.cue_points, (reader.CuePoint(5.5, "CHORUS", "drop", "#ff2828"),))
        self.assertEqual(track.jump_groups(), {0: (reader.JumpCue(1.5, "A", "", 0, "#0099cc", (1.5, 3.5)),
                                                   reader.JumpCue(5.5, "B", "last", 0, "#0099cc", (5.5, 9.5)))})
        bare = reader.parse(json.loads(jams_text(document(path=None, uid=None))))
        self.assertEqual((bare.uid, bare.path), (None, None))

    def test_the_standard_library_alone(self):
        import ast
        import sys
        from pathlib import Path
        source = Path(__file__).resolve().parents[3] / "examples" / "jams_mxl_sbx1.py"
        names = set()
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertEqual(names - set(sys.stdlib_module_names) - {"__future__"}, set())

    def test_another_programs_annotations_left_alone(self):
        from examples import jams_mxl_sbx1 as reader
        doc = json.loads(jams_text(document()))
        doc["annotations"].append({"namespace": "beat", "annotation_metadata": {"annotation_tools": "madmom"},
                                   "data": [{"time": 0.1, "duration": 0.0, "value": 1, "confidence": 0.5}]})
        self.assertEqual(len(reader.parse(doc).beats), 19)

    def test_other_files_refused(self):
        from examples import jams_mxl_sbx1 as reader
        doc = json.loads(jams_text(document()))
        plain = {k: v for k, v in doc.items() if k != "sandbox"}             # a JAMS file of another program
        for bad in (plain, {**doc, "sandbox": {SANDBOX: {**doc["sandbox"][SANDBOX], "format": "JAMS-mxl-sbx2"}}}):
            with self.assertRaises(reader.FormatError):
                reader.parse(bad)
        broken = json.loads(jams_text(document()))
        broken["annotations"][0]["data"][0]["time"] = -1.0
        with self.assertRaises(reader.FormatError):
            reader.parse(broken)


if __name__ == "__main__":
    unittest.main()
