import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

from analyzer.music.beat_geometry import build_downbeats_from_segments, downbeat_times
from analyzer.music.marks import Phrase
from core.library.db import LibraryDB, TrackRow
from core.library.segments import build_bpm_segments, build_phrase_index_rows
from analyzer.music.tempo_rows import (
    bpm_at,
    is_ramp_row,
    ramp_beat_times,
    rebuild_grid_from_segments,
    shift_row_beats,
    split_row,
)
from third_party.rekordbox import _tempo_entries_for_xml

# steady 180 -> ramp 180..160 over 10 s -> steady 160, phase-continuous enough for the tests
SEGMENTS = np.array([
    [0.0, 100.0, 180.0, 0.3, 4.0, 180.0],
    [100.0, 110.0, 180.0, 100.1, 4.0, 160.0],
    [110.0, 200.0, 160.0, 110.2, 4.0, 160.0],
])


class TempoRowTests(unittest.TestCase):
    def test_ramp_beats_follow_the_integrated_tempo(self):
        beats = ramp_beat_times(100.0, 110.0, 180.0, 160.0, 100.1, 110.0)
        t = np.linspace(100.0, 110.0, 200001)
        phase = np.concatenate([[0.0], np.cumsum((180.0 - 2.0 * (t[1:] - 100.0)) / 60.0 * np.diff(t))])
        start = np.interp(100.1, t, phase)
        expected = np.interp(start + np.arange(beats.size), phase, t)
        self.assertLess(float(np.max(np.abs(beats - expected))), 1e-5)

    def test_steady_rows_rebuild_the_same_with_or_without_the_end_column(self):
        steady = SEGMENTS[[0, 2]]
        np.testing.assert_allclose(rebuild_grid_from_segments(steady), rebuild_grid_from_segments(steady[:, :5]))

    def test_split_keeps_the_beats(self):
        left, right = split_row(SEGMENTS[1], 104.0)
        rows = np.vstack([SEGMENTS[0], left, right, SEGMENTS[2]])
        np.testing.assert_allclose(rebuild_grid_from_segments(rows), rebuild_grid_from_segments(SEGMENTS), atol=1e-6)
        self.assertAlmostEqual(left[5], bpm_at(SEGMENTS[1], 104.0))

    def test_half_beat_shift_lands_between_the_beats(self):
        row = SEGMENTS[1].copy()
        old = ramp_beat_times(100.0, 110.0, 180.0, 160.0, row[3], 110.0)
        row[3] = shift_row_beats(row, 0.5)
        new = ramp_beat_times(100.0, 110.0, 180.0, 160.0, row[3], 110.0)
        mids = 0.5 * (old[1:] + old[:-1])
        self.assertLess(max(float(np.min(np.abs(mids - x))) for x in new[:-1]), 5e-4)

    def test_downbeat_fallback_matches_the_beat_based_downbeats(self):
        beats = rebuild_grid_from_segments(SEGMENTS)
        np.testing.assert_allclose(build_downbeats_from_segments(SEGMENTS), downbeat_times(beats, SEGMENTS), atol=1e-6)


class RampIndexTests(unittest.TestCase):
    def test_ramp_rows_keep_both_tempos_and_never_merge(self):
        rows = build_bpm_segments(SEGMENTS)
        self.assertEqual([(r["bpm"], r["bpm_end"]) for r in rows], [(180.0, None), (180.0, 160.0), (160.0, None)])
        legacy = build_bpm_segments(SEGMENTS[:, :5])
        self.assertTrue(all(r["bpm_end"] is None for r in legacy))

    def test_phrase_section_in_a_ramp_gets_its_midpoint_tempo(self):
        sections = build_phrase_index_rows([Phrase(100.0, 110.0, "BUILD")], build_bpm_segments(SEGMENTS), [])
        self.assertAlmostEqual(sections[0]["bpm"], 170.0)

    def test_transition_search_finds_a_ramp(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = LibraryDB(str(Path(tmp) / "library.db"))
            db.connect()
            try:
                uid = str(uuid4())
                db.upsert_many([TrackRow(path="x.flac", uid=uid, title="T", artist="A")])
                db.replace_bpm_segments(uid, build_bpm_segments(SEGMENTS))
                hits = db.search_bpm_transitions(175, 185, 155, 165)
                self.assertEqual([(h.from_seq_index, h.to_seq_index) for h in hits], [(1, 1)])
                stored = db.get_bpm_segments(uid)
                self.assertEqual([s.bpm_end for s in stored], [None, 160.0, None])
            finally:
                db.close()


def _xml_beats(entries, stop):
    """The beats the TEMPO entries give: each entry's beats up to the next entry's Inizio."""
    beats = []
    for i, entry in enumerate(entries):
        until = entries[i + 1]["start"] if i + 1 < len(entries) else stop
        t = entry["start"]
        while t < until - 1e-9:
            beats.append(t)
            t += 60.0 / entry["bpm"]
    return np.asarray(beats)


class RampExportTests(unittest.TestCase):
    def test_rekordbox_entries_reproduce_the_grid(self):
        # a step per beat: every beat as written (Inizio in ms), none added or lost
        entries = _tempo_entries_for_xml(SEGMENTS, 170.0)
        grid = rebuild_grid_from_segments(SEGMENTS)
        beats = _xml_beats(entries, 200.0)
        self.assertEqual(beats.size, grid.size)
        self.assertLessEqual(float(np.max(np.abs(beats - grid))), 0.0005 + 1e-9)
        ramp = (grid >= 100.0) & (grid < 110.0)
        self.assertEqual(sum(1 for e in entries if 100.0 <= e["start"] < 110.0), int(np.count_nonzero(ramp)))

    def test_rekordbox_ramp_steps_merge_within_the_tolerance(self):
        grid = rebuild_grid_from_segments(SEGMENTS)
        per_beat = _tempo_entries_for_xml(SEGMENTS, 170.0)
        previous = len(per_beat)
        for tolerance in (0.001, 0.002, 0.005):
            entries = _tempo_entries_for_xml(SEGMENTS, 170.0, tolerance)
            beats = _xml_beats(entries, 200.0)
            self.assertEqual(beats.size, grid.size)
            self.assertLessEqual(float(np.max(np.abs(beats - grid))), tolerance + 1e-9)
            self.assertLess(len(entries), previous)
            previous = len(entries)
            # a ramp step's BPM never puts a beat just before the next step's first
            for k in range(len(entries) - 1):
                first, nxt = entries[k]["start"], entries[k + 1]["start"]
                if not (100.0 <= first and nxt < 110.0):
                    continue
                n = round((nxt - first) * entries[k]["bpm"] / 60.0)
                self.assertGreaterEqual(first + n * 60.0 / entries[k]["bpm"], nxt - 1e-9)

    def test_cue_points_are_memory_cues(self):
        from third_party.rekordbox import build_rekordbox_track_element
        from analyzer.music.marks import CuePoint
        from core.analysis_library.cue_points import build_cue_points_np
        from core.marks.colors import get_cue_point_color
        from core.analysis_library.stored import StoredAnalysis

        cues = build_cue_points_np([
            CuePoint(id=0, time_sec=12.5, label="Drop", color_idx=2),
            CuePoint(id=1, time_sec=0.3, label="", color_idx=0),
        ])
        element = build_rekordbox_track_element(
            TrackRow(path="x.flac", uid=str(uuid4()), title="T", artist="A"),
            StoredAnalysis.from_arrays({"tempo_segments": SEGMENTS, "cue_points_np": cues}),
            resolve_path=lambda _path: None,
        )
        memory = [m.attrib for m in element.findall("POSITION_MARK") if m.attrib["Num"] == "-1"]
        self.assertEqual([(m["Start"], m["Name"], m["Type"]) for m in memory], [("0.300", "", "0"), ("12.500", "Drop", "0")])
        self.assertEqual(tuple(int(memory[1][c]) for c in ("Red", "Green", "Blue")), get_cue_point_color(2))


class GridRefineTests(unittest.TestCase):
    def test_slow_tail_is_held_at_the_main_tempo(self):
        from analyzer.beat.grid_refine import hold_slow_tail

        rows = np.array([[0.0, 189.9, 206.0, 0.3, 4.0, 206.0], [189.9, 190.8, 206.0, 190.2, 4.0, 206.0],
                         [190.8, 210.4, 136.1, 191.0, 4.0, 136.1]])
        held, done = hold_slow_tail(rows, 210.4)
        self.assertTrue(done)
        self.assertEqual(held.shape[0], 2)
        self.assertAlmostEqual(held[-1, 1], 210.4)
        # a slowdown that starts earlier than the last 30 s is a real section: kept
        kept, done = hold_slow_tail(np.array([[0.0, 150.0, 206.0, 0.3, 4.0, 206.0], [150.0, 210.4, 136.1, 150.2, 4.0, 136.1]]), 210.4)
        self.assertFalse(done)
        self.assertEqual(kept.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
