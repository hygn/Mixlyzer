"""third_party.midi_tempo_map, read back by a reader of its own (the Standard MIDI File's chunks and meta events, not
the writer's code): each beat of the grid at its tick within TOLERANCE_SEC, a steady grid one tempo, a ramp a tempo
a beat, every downbeat a bar's start, the beats before the first a pickup, the keys' signatures and markers, the
track ending after the audio."""
import struct
import unittest

import numpy as np

from analyzer.music.tempo_rows import ramp_beat_times
from third_party.midi_tempo_map import PPQ, TOLERANCE_SEC, TempoMapError, midi_tempo_map


def read(data: bytes) -> dict:
    """The file's one track: its meta events as (tick, type, data), its division."""
    assert data[:4] == b"MThd"
    length, fmt, ntracks, division = struct.unpack(">IHHH", data[4:14])
    assert (length, fmt, ntracks) == (6, 0, 1)
    assert data[14:18] == b"MTrk"
    (size,) = struct.unpack(">I", data[18:22])
    body, pos, tick, events = data[22:22 + size], 0, 0, []
    assert len(body) == size

    def vlq() -> int:
        nonlocal pos
        value = 0
        while True:
            byte = body[pos]
            pos += 1
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value

    while pos < len(body):
        tick += vlq()
        assert body[pos] == 0xFF          # meta events only
        kind = body[pos + 1]
        pos += 2
        n = vlq()
        events.append((tick, kind, body[pos:pos + n]))
        pos += n
    assert events[-1][1] == 0x2F
    return {"division": division, "events": events}


def beat_seconds(track: dict, beats: int) -> np.ndarray:
    """Each beat's time (its tick a multiple of the division) by the tempo events."""
    tempos = [(t, int.from_bytes(d, "big")) for t, k, d in track["events"] if k == 0x51]
    out, at, tempo, ti = [], 0.0, 500000, 0
    for b in range(beats):
        while ti < len(tempos) and tempos[ti][0] <= b * track["division"]:
            tempo = tempos[ti][1]
            ti += 1
        out.append(at)
        at += tempo / 1e6
    return np.array(out)


def bar_starts(track: dict, beats: int) -> list[int]:
    sigs = {t // track["division"]: d[0] for t, k, d in track["events"] if k == 0x58}
    starts, num, pos = [], 4, 0
    for b in range(beats):
        if b in sigs:
            num, pos = sigs[b], 0
        if pos % num == 0:
            starts.append(b)
        pos += 1
    return starts


def row(start, end, bpm, inizio, ts=4, bpm_end=None):
    return [start, end, bpm, inizio, ts, bpm if bpm_end is None else bpm_end]


class TempoMapTests(unittest.TestCase):
    def test_a_steady_grid_after_a_lead_in(self):
        period = 60 / 128
        beats = 0.37 + period * np.arange(400)
        track = read(midi_tempo_map(beats, [row(0.37, beats[-1] + period, 128, 0.37 + period)], [], "ヴヴヴ", 200.0))
        self.assertEqual(track["division"], PPQ)
        tempos = [e for e in track["events"] if e[1] == 0x51]
        # the lead-in (0.37 s, under a beat) is one beat of its own; then the grid's one tempo
        self.assertEqual(len(tempos), 2)
        self.assertEqual(int.from_bytes(tempos[1][2], "big"), round(period * 1e6))
        times = beat_seconds(track, 401)
        np.testing.assert_allclose(times[1:], beats, atol=TOLERANCE_SEC + 1e-9)
        # the downbeat a beat after the first: a pickup bar of 2 (the lead-in beat and the first), then 4/4
        self.assertEqual([(t // PPQ, d[0]) for t, k, d in track["events"] if k == 0x58], [(0, 2), (2, 4)])
        self.assertTrue(set(range(2, 400, 4)) <= set(bar_starts(track, 401)))
        self.assertEqual(track["events"][0][1:], (0x03, "ヴヴヴ".encode("utf-8")))
        self.assertGreaterEqual(track["events"][-1][0], 1 + (200.0 - beats[-1]) / period * PPQ + 399 * PPQ - PPQ)

    def test_a_sliver_before_the_first_beat_joins_the_next(self):
        period = 0.5
        beats = 1.05 + period * np.arange(16)          # 0.05 s left at the start: under half a beat
        track = read(midi_tempo_map(beats, [row(1.05, 9.05, 120, 1.05)], []))
        tempos = [int.from_bytes(d, "big") for _t, k, d in track["events"] if k == 0x51]
        self.assertEqual(tempos[0], 550000)             # 0.55 s: the sliver with a beat
        times = beat_seconds(track, 3 + 16)
        np.testing.assert_allclose(times[2:2 + 16], beats, atol=TOLERANCE_SEC + 1e-9)

    def test_a_ramp_and_a_short_bar(self):
        steady = 0.0 + 0.5 * np.arange(8)                                      # 120 BPM, two bars
        ramp = ramp_beat_times(4.0, 12.0, 120.0, 150.0, 4.0, 12.0)              # a ramp to 150
        ramp = ramp[ramp < 12.0]
        after = ramp[-1] + 0.4 + 0.4 * np.arange(11)                           # 150 BPM: a bar of 3, then 4/4
        beats = np.concatenate([steady, ramp, after])
        rows = [row(0.0, 4.0, 120, 0.0), row(4.0, 12.0, 120, 4.0, 4, 150), row(after[0], after[-1] + 0.4, 150,
                                                                                  after[0], 3),
                row(after[3], after[-1] + 0.4, 150, after[3], 4)]
        track = read(midi_tempo_map(beats, rows, []))
        times = beat_seconds(track, beats.size)
        np.testing.assert_allclose(times, beats, atol=TOLERANCE_SEC + 1e-9)
        tempos = [t for t, k, _d in track["events"] if k == 0x51]
        self.assertGreaterEqual(len(tempos), ramp.size - 1)                     # a tempo a beat in the ramp
        self.assertLessEqual(len([t for t in tempos if t >= (8 + ramp.size) * PPQ]), 2)
        starts = bar_starts(track, beats.size)
        bar3 = 8 + ramp.size
        self.assertIn(bar3, starts)
        self.assertIn(bar3 + 3, starts)                                         # the bar of 3 ends at its downbeat

    def test_keys_as_signatures_and_markers(self):
        beats = 0.5 * np.arange(64)
        keys = [[9, 1, 0.0, 10.0], [9, 1, 10.0, 12.0], [0, 0, 12.0, 20.0], [6, 0, 20.0, 32.0]]
        track = read(midi_tempo_map(beats, [row(0.0, 32.0, 120, 0.0)], keys))
        signatures = [(t, struct.unpack(">bB", d)) for t, k, d in track["events"] if k == 0x59]
        markers = [(t, d.decode("utf-8")) for t, k, d in track["events"] if k == 0x06]
        # A minor, then C major at 12 s (beat 24), then F# major at 20 s (beat 40); the same key twice once
        self.assertEqual(signatures, [(0, (0, 1)), (24 * PPQ, (0, 0)), (40 * PPQ, (6, 0))])
        self.assertEqual(markers, [(0, "Am (8A)"), (24 * PPQ, "C (8B)"), (40 * PPQ, "F# (2B)")])

    def test_at_another_speed(self):
        # the audio played 1.25 times as fast (time-stretched): every time / 1.25, every tempo x 1.25, the bars and the
        # keys on the same beats, the track ending after the faster audio
        beats = 0.25 + 0.5 * np.arange(64)
        rows = [row(0.25, 32.25, 120, 0.25)]
        keys = [[9, 1, 0.0, 12.25], [0, 0, 12.25, 32.25]]
        plain, fast = (read(midi_tempo_map(beats, rows, keys, duration_sec=40.0, speed=s)) for s in (1.0, 1.25))
        self.assertTrue(np.allclose(beat_seconds(fast, 64), beat_seconds(plain, 64) / 1.25, atol=TOLERANCE_SEC))
        for kind in (0x58, 0x59, 0x06):
            self.assertEqual([(t, d) for t, k, d in fast["events"] if k == kind],
                             [(t, d) for t, k, d in plain["events"] if k == kind])
        tempos = [60e6 / int.from_bytes(d, "big") for _t, k, d in fast["events"] if k == 0x51]
        self.assertAlmostEqual(tempos[-1], 150.0, places=2)
        self.assertEqual(fast["events"][-1][0], plain["events"][-1][0])    # 40 s at 120, 32 s at 150: one beat count

    def test_no_beatgrid(self):
        for beats in ([], [1.0], [np.nan, 2.0]):
            with self.assertRaises(TempoMapError):
                midi_tempo_map(beats, [], [])


if __name__ == "__main__":
    unittest.main()
