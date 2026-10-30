"""third_party.serato: the beatgrid as Serato's markers (from the first downbeat; a steady stretch one marker, a ramp a
marker a beat, every beat within TOLERANCE_SEC of the grid's), the JumpCUEs as hot cues (A-H, Serato's colours, the
names as Serato takes them), Markers2 laid out as Serato reads it (padded, the file's loops and colour kept,
BPMLOCK set), and what Serato loses."""
import base64
import struct
import unittest

import numpy as np

from analyzer.music.marks import CuePoint, JumpCue, JumpCueGroup
from analyzer.music.tempo_rows import ramp_beat_times
from third_party import serato


def row(start, end, bpm, inizio, ts=4, bpm_end=None):
    return [start, end, bpm, inizio, ts, bpm if bpm_end is None else bpm_end]


def beats_of(markers, until):
    """The beats Serato spaces from the markers (evenly between two, the last BPM run on)."""
    out = []
    for (pos, count), (nxt, _v) in zip(markers, markers[1:]):
        out += list(np.linspace(pos, nxt, int(count) + 1)[:-1])
    pos, bpm = markers[-1]
    out += list(pos + 60.0 / bpm * np.arange(int((until - pos) * bpm / 60.0) + 1))
    return np.array(out)


class GridTests(unittest.TestCase):
    def test_a_steady_grid_one_marker_from_the_first_downbeat(self):
        period = 60 / 174
        beats = 0.37 + period * np.arange(400)
        markers, losses = serato.grid_markers(beats, [row(0.37, beats[-1], 174, 0.37 + period)])
        self.assertEqual(len(markers), 1)
        self.assertAlmostEqual(markers[0][0], 0.37 + period)              # the first downbeat, not the first beat
        self.assertAlmostEqual(markers[0][1], 174.0, places=6)
        self.assertEqual(losses, [])

    def test_a_ramp_a_marker_a_beat_and_every_beat_kept(self):
        steady = 0.5 * np.arange(8)
        ramp = ramp_beat_times(4.0, 12.0, 120.0, 150.0, 4.0, 12.0)
        ramp = ramp[ramp < 12.0]
        after = ramp[-1] + 0.4 + 0.4 * np.arange(20)
        beats = np.concatenate([steady, ramp, after])
        rows = [row(0.0, 4.0, 120, 0.0), row(4.0, 12.0, 120, 4.0, 4, 150), row(after[0], after[-1], 150, after[0])]
        markers, _losses = serato.grid_markers(beats, rows)
        self.assertGreaterEqual(len(markers), ramp.size - 1)
        self.assertLess(len(markers), ramp.size + 4)
        got = beats_of(markers, beats[-1] + 0.01)[:beats.size]
        np.testing.assert_allclose(got, beats, atol=serato.TOLERANCE_SEC + 1e-9)
        # through the payload (f32 positions)
        parsed = serato.parse_beatgrid(serato.build_beatgrid(markers))
        np.testing.assert_allclose(np.array(parsed), np.array(markers), atol=1e-4)

    def test_the_tempo_map_as_serato_writes_it(self):
        # a Tempo Map Serato wrote (one segment: beat 0 at 0.586838 s, 86.999756 BPM), made again from its markers
        # byte for byte
        serato_s = bytes.fromhex("0200 0001 3fe2c75fb6038d07 00000001 0000000000000000 4055bffc01f9ce94 "
                                 "3fd0000000000000 01 0000000000000000".replace(" ", ""))
        anchor, segments = serato.parse_tempo_map(serato_s)
        self.assertEqual(serato.build_tempo_map([(anchor, segments[0][1])]), serato_s)

    def test_the_tempo_map_of_a_ramp_gives_its_beats(self):
        steady = 0.5 * np.arange(8)
        ramp = ramp_beat_times(4.0, 12.0, 120.0, 150.0, 4.0, 12.0)
        ramp = ramp[ramp < 12.0]
        beats = np.concatenate([steady, ramp, ramp[-1] + 0.4 + 0.4 * np.arange(20)])
        rows = [row(0.0, 4.0, 120, 0.0), row(4.0, 12.0, 120, 4.0, 4, 150), row(ramp[-1] + 0.4, 30.0, 150, ramp[-1] + 0.4)]
        markers, _ = serato.grid_markers(beats, rows)
        payload = serato.build_tempo_map(markers)
        anchor, segments = serato.parse_tempo_map(payload)
        self.assertEqual(len(segments), len(markers))
        self.assertEqual([b for b, _bpm in segments][:2], [0, int(markers[0][1])])
        got = serato.tempo_map_beats(payload, beats[-1] + 0.01)[:beats.size]
        np.testing.assert_allclose(got, beats, atol=serato.TOLERANCE_SEC + 1e-9)

    def test_a_bar_of_another_length_said(self):
        beats = 0.5 * np.arange(24)
        rows = [row(0.0, 4.0, 120, 0.0), row(4.0, 5.5, 120, 4.0, 3), row(5.5, 12.0, 120, 5.5)]
        _markers, losses = serato.grid_markers(beats, rows)
        self.assertEqual(losses, ["bars"])

    def test_no_grid(self):
        self.assertEqual(serato.grid_markers([1.0], []), ([], []))
        self.assertEqual(serato.parse_beatgrid(b"\x01\x00\x00\x00\x00\x09"), [])     # a count past its data


class TimeTests(unittest.TestCase):
    def mp3(self, folder, encoder):
        """An MP3 whose first frame is an Info frame with ``encoder``'s LAME tag (all flags), after an ID3 tag."""
        info = b"Info" + struct.pack(">I", 15) + bytes(4 + 4 + 100 + 4) + encoder + bytes(40)
        frame = b"\xff\xfb\x90\x64" + bytes(32) + info
        tag = b"ID3\x04\x00\x00\x00\x00\x00\x10" + bytes(16)
        path = f"{folder}/{encoder.decode() or 'plain'}.mp3"
        with open(path, "wb") as f:
            f.write(tag + frame + bytes(400))
        return path

    def test_serato_time_by_format(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            self.assertAlmostEqual(serato.time_offset(self.mp3(folder, b"LAME"), 44100), 0.050)
            self.assertAlmostEqual(serato.time_offset(self.mp3(folder, b"Lavc"), 48000), 0.04474)
            self.assertAlmostEqual(serato.time_offset(self.mp3(folder, b""), 44100), -0.00506)     # no delay given
            self.assertAlmostEqual(serato.time_offset(f"{folder}/a.m4a", 44100), -2112 / 44100)
            self.assertEqual(serato.time_offset(f"{folder}/a.flac", 44100), 0.0)
        markers = serato.shifted([(1.0, 8), (5.0, 128.0)], -0.05)
        self.assertEqual(markers, [(0.95, 8), (4.95, 128.0)])
        cue = JumpCue("A", "", 12.5, 12.5, 12.5)
        cues, _ = serato.hot_cues([JumpCueGroup((cue,), 0)], offset=0.05)
        self.assertAlmostEqual(cues[0].time_sec, 12.55)


class CueTests(unittest.TestCase):
    def groups(self):
        cue = lambda label, point, comment="": JumpCue(label, comment, point, point, point)   # noqa: E731
        return [JumpCueGroup((cue("A", 12.5, "サビ"), cue("B", 40.0)), 0),
                JumpCueGroup((cue("C", 61.25, "x" * 30 + "ヴ" * 10), cue("I", 80.0)), 3)]

    def points(self, n):
        return [CuePoint(10.0 * (i + 1), f"Cue {i + 1}", "", i % 8) for i in range(n)]

    def test_jumpcues_as_hot_cues(self):
        cues, losses = serato.hot_cues(self.groups(), self.points(2))
        self.assertEqual([(c.slot, c.time_sec, c.name) for c in cues][:2], [(0, 12.5, "サビ"), (1, 40.0, "B")])
        self.assertTrue(all(c.color in serato.CUE_COLORS for c in cues))
        self.assertEqual(losses, ["slots", "cue_points"])            # I past H; the CUE points
        payload = serato.build_markers2(cues)
        self.assertGreaterEqual(len(payload), serato.MARKERS2_MIN)
        back = serato.parse_cues(payload)
        self.assertEqual([(c.slot, c.time_sec, c.color) for c in back], [(c.slot, c.time_sec, c.color) for c in cues])
        long_name = back[2].name.encode("utf-8")
        self.assertLessEqual(len(long_name), serato.NAME_MAX)          # cut between characters
        self.assertTrue(back[2].name.startswith("x" * 30 + "ヴ"))
        self.assertTrue(serato.bpm_locked(payload))

    def test_cue_points_as_hot_cues(self):
        cues, losses = serato.hot_cues(self.groups(), self.points(3), "cue_points")
        self.assertEqual([(c.slot, c.time_sec, c.name) for c in cues],
                         [(0, 10.0, "Cue 1"), (1, 20.0, "Cue 2"), (2, 30.0, "Cue 3")])
        self.assertEqual(losses, [])
        # both: the JumpCUEs A, B, C in their slots, the CUE points in the slots after them, in time order
        cues, losses = serato.hot_cues(self.groups(), self.points(7), "both")
        self.assertEqual([(c.slot, c.name) for c in cues][:4], [(0, "サビ"), (1, "B"), (2, cues[2].name), (3, "Cue 1")])
        self.assertEqual(len(cues), 8)
        self.assertEqual(losses, ["slots", "full"])            # I past H; two CUE points with no slot left

    def test_the_files_hot_cues_kept_or_replaced(self):
        old = serato.build_markers2([serato.HotCue(5, 99.0, serato.CUE_COLORS[2], "Serato's own"),
                                     serato.HotCue(0, 1.0, serato.CUE_COLORS[2], "taken")])
        kept, _ = serato.hot_cues(self.groups(), (), "jumpcues", old, keep=True)
        self.assertEqual([(c.slot, c.name) for c in kept], [(0, "サビ"), (1, "B"), (2, kept[2].name),
                                                            (5, "Serato's own")])
        replaced, _ = serato.hot_cues(self.groups(), (), "jumpcues", old, keep=False)
        self.assertEqual([c.slot for c in replaced], [0, 1, 2])
        self.assertFalse(serato.bpm_locked(serato.build_markers2(replaced, old, bpm_lock=False)))

    def test_the_files_loops_and_colour_kept(self):
        def entry(name, data):
            return name.encode() + b"\x00" + struct.pack(">I", len(data)) + data
        loop = b"\x00\x00" + struct.pack(">II", 1000, 5000) + b"\xff\xff\xff\xff\x00\xaa\xe1\xff\x00\x00Loop\x00"
        body = (b"\x01\x01" + entry("COLOR", b"\x00\x99\xff\x99") + entry("CUE", b"\x00\x05" + bytes(20))
                + entry("LOOP", loop) + entry("BPMLOCK", b"\x00") + b"\x00")
        existing = b"\x01\x01" + base64.b64encode(body).rstrip(b"=") + b"\x00"
        payload = serato.build_markers2(serato.hot_cues(self.groups())[0], existing)
        names = [n for n, _d in serato._entries(payload)]
        self.assertEqual(names, ["COLOR", "CUE", "CUE", "CUE", "LOOP", "BPMLOCK"])      # the file's cue replaced
        self.assertEqual(dict(serato._entries(payload))["LOOP"], loop)
        self.assertEqual(dict(serato._entries(payload))["COLOR"], b"\x00\x99\xff\x99")


if __name__ == "__main__":
    unittest.main()
