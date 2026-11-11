"""The track editor (offscreen Qt) over a DataModel loaded with a synthetic analysis: the edit lock, the beatgrid's
shift / BPM x2, x1/2 / split / merge / time signature / assign on the segment under the playhead, the page-wide
shortcuts working on the current page, keys and phrases split, merged, moved and relabelled, CUE points and
JumpCUEs set and deleted by their slot rules, undo, the marks sent to the deck, and the editor's save."""
import os
import tempfile
import unittest
from unittest import mock
from uuid import uuid4

import numpy as np

from unit.ui_tests.qt_env import APP, drain, isolated

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QMessageBox

from analyzer.music.marks import CuePoint, JumpCue, Phrase
from analyzer.music.tempo_rows import rebuild_grid_from_segments, with_end_bpm
from audioengine import controls as deck
from core.analysis_library.cue_points import build_cue_points_np
from core.analysis_library.jump_cues import build_jump_cues_np
from core.analysis_library.phrases import build_phrase_segments_np
from core.analysis_library.store import FeatureNPZStore
from core.analysis_library.stored import StoredAnalysis
from core.library.db import LibraryDB
from core.library.records import TrackRow
from core.library.service import LibraryService
from core.marks.phrases import UNASSIGNED
from core.track.model import DataModel
from core.workflows.event_bus import EventBus
from ui.edit_panel.cue_slots import SLOT_MIME, CueSlot
from ui.edit_panel.panel import EditPanel

_ISOLATION = None
SEGMENTS = np.asarray([[0.0, 32.0, 120.0, 0.0, 4.0], [32.0, 64.0, 120.0, 32.0, 4.0]])     # 120 BPM, 64 s
BEATS = rebuild_grid_from_segments(with_end_bpm(SEGMENTS))         # every half second, the track end's too
PHRASE_PAGE = EditPanel.PAGES.index("Phrase")


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


def _pair(a: float, b: float) -> dict:
    cue = lambda p, label: {"label": label, "comment": label, "start": p, "end": p, "point": p, "color_idx": 0}  # noqa: E731
    return {"forward": cue(a, "A"), "backward": cue(b, "B"), "lag_beats": 0.0, "lag_sec": b - a, "score": 1.0,
            "confidence": 1.0}


def arrays() -> dict:
    return {
        "beats_time_sec": BEATS,
        "tempo_segments": SEGMENTS,
        "key_segments": np.asarray([[0.0, 0.0, 0.0, 32.0], [9.0, 1.0, 32.0, 64.0]]),       # 1st C, then A minor
        "phrase_segments_np": build_phrase_segments_np([Phrase(0.0, 16.0, "INTRO"), Phrase(16.0, 48.0, "VERSE"),
                                                        Phrase(48.0, 64.0, "OUTRO")]),
        "jump_cues_np": build_jump_cues_np([_pair(8.0, 40.0)]),
        "cue_points_np": build_cue_points_np([CuePoint(time_sec=16.0)]),
        "duration_sec": 64.0,
    }


class EditPanelCase(unittest.TestCase):
    UID = str(uuid4())

    def setUp(self):
        self.bus = EventBus()
        self.model = DataModel()
        self.panel = EditPanel(self.bus, self.model, library=self.library())
        self.panel.set_column_count(4)
        self.model.load(StoredAnalysis.from_arrays(arrays()), {"uid": self.UID, "path": "track.wav"}, 64.0)
        self.h = self.panel.shortcut_handlers()

    def library(self):
        return None

    def tearDown(self):
        self.panel._marks_timer.stop()
        self.panel.close()
        self.panel.deleteLater()
        drain()

    def at(self, t: float) -> None:
        self.panel.update_time(t)

    def segments(self) -> list:
        return np.asarray(self.model.tempo_segments).round(6).tolist()

    def keys(self) -> list:
        return [(float(s[0]), float(s[1]), float(s[2]), float(s[3])) for s in self.model.key_segments]

    def phrases(self) -> list:
        return [(p.start, p.end, p.label) for p in self.model.phrases]

    def jump_cues(self) -> dict:
        return {c.label: (c.point, gi) for gi, g in enumerate(self.model.jump_cues) for c in g.cues}


class EditingTests(EditPanelCase):
    def test_editing_on_from_the_start(self):
        self.at(10.0)
        for button in (self.panel.btn_shift_fine_fwd, self.panel.btn_bpm_double, self.panel.key_split,
                       self.panel.phrase_split, self.panel.cp_memory, self.panel.btn_save):
            self.assertTrue(button.isEnabled())
        self.assertIsNone(self.h["beatgrid.shift_fine_forward"]())
        self.assertTrue(self.model.can_undo())

    def test_a_running_reanalysis_turns_editing_off(self):
        self.bus.sig_segment_reanalyze_state.emit(False)     # running
        self.at(10.0)
        for button in (self.panel.btn_shift_fine_fwd, self.panel.key_split, self.panel.phrase_split,
                       self.panel.cp_memory, self.panel.btn_save):
            self.assertFalse(button.isEnabled())
        waits = "Editing waits for the segment reanalysis to finish."
        self.assertEqual(self.h["beatgrid.shift_fine_forward"](), waits)
        self.assertEqual(self.h["editor.split"](), waits)
        self.assertEqual(self.h["cue.memory_delete"](), waits)
        self.assertEqual(self.h["cue.set.C"](), waits)
        self.panel._jc_set_or_jump(2)                   # an empty slot's SET
        self.panel._rename_cue_point(0, "Name")
        self.assertFalse(self.model.can_undo())
        self.assertTrue(self.panel.jc_slots[0].btn_action.isEnabled())       # JUMP stays usable
        self.bus.sig_segment_reanalyze_state.emit(True)
        self.assertIsNone(self.h["beatgrid.shift_fine_forward"]())
        self.assertTrue(self.model.can_undo())

    def test_undo_redo_buttons_follow_the_history_and_a_reanalysis(self):
        self.assertFalse(self.panel.btn_undo.isEnabled())
        self.at(10.0)
        self.panel.btn_shift_fine_fwd.click()
        self.assertTrue(self.panel.btn_undo.isEnabled())
        self.bus.sig_segment_reanalyze_state.emit(False)
        self.assertFalse(self.panel.btn_undo.isEnabled())

    def test_no_page_emphasized(self):
        """The edit focus is kept but off (EditPanel.EMPHASIZE_SHOWN_PAGES): no page is ever emphasized."""
        seen = []
        self.bus.sig_edit_focus_changed.connect(seen.append)
        for i in range(len(self.panel.PAGES)):
            self.panel.show_page(i)
        self.assertEqual((seen, self.panel._edit_focus), ([], frozenset()))
        self.assertFalse(self.panel.EMPHASIZE_SHOWN_PAGES)


class CuePageTests(EditPanelCase):
    """The side strip: two squares joined pick the list, under them the list's one tool, as wide as both."""

    def test_two_squares_joined_and_the_tool_as_wide_as_both(self):
        page = self.panel.cue_page
        page.show()
        APP.processEvents()
        jump, cue = page.tab_jump.geometry(), page.tab_cue.geometry()
        self.assertEqual(jump.width(), jump.height())
        self.assertEqual(cue.size(), jump.size())
        self.assertEqual(cue.left(), jump.right() + 1)                    # nothing between them
        self.assertEqual(cue.top(), jump.top())
        tool = page.tool_slot.geometry()
        self.assertEqual((tool.left(), tool.width()), (jump.left(), jump.width() * 2))
        self.assertGreater(tool.top(), jump.bottom())
        self.assertEqual(page.tab_jump.width() * 2, page.btn_next.geometry().right() + 1 - page.btn_prev.geometry().left())
        page.hide()

    def test_a_square_shows_its_list_and_tool(self):
        page, panel = self.panel.cue_page, self.panel
        self.assertEqual(page.current_list, 0)
        self.assertIs(page.tool_slot.currentWidget(), panel.jc_reanalyze)
        page.tab_cue.click()
        self.assertEqual(page.current_list, 1)
        self.assertIs(page.tool_slot.currentWidget(), panel.cp_memory)
        self.assertFalse(page.tab_jump.isChecked())
        page.tab_jump.click()
        self.assertEqual(page.current_list, 0)

    def test_the_tool_works(self):
        panel = self.panel
        panel.cue_page.show_list(1)
        self.assertTrue(panel.cue_page.tab_cue.isChecked())
        self.at(20.2)
        count = len(self.model.cue_points)
        panel.cp_memory.click()
        self.assertEqual(len(self.model.cue_points), count + 1)


class BeatgridTests(EditPanelCase):
    def setUp(self):
        super().setUp()
        self.at(10.0)                                 # in the first segment
        self.original = self.segments()

    @staticmethod
    def on_grid(beats, offset: float) -> bool:
        """Every beat on the half-second grid moved by ``offset``."""
        steps = (np.asarray(beats) - offset) / 0.5
        return bool(np.allclose(steps, np.round(steps), atol=1e-6))

    def test_shift_moves_the_current_segment_or_the_whole_grid(self):
        self.panel.btn_shift_fine_fwd.click()
        beats = np.asarray(self.model.beats)
        self.assertTrue(self.on_grid(beats[beats < 31.9], 0.001))
        self.assertTrue(self.on_grid(beats[beats > 32.4], 0.0))           # the next segment stays
        self.assertAlmostEqual(self.model.tempo_segments[0, 3], 0.001)
        self.assertEqual(self.model.tempo_segments[1, 3], 32.0)
        self.panel.btn_scope_all.click()
        self.panel.btn_shift_coarse_back.click()
        beats = np.asarray(self.model.beats)
        self.assertTrue(self.on_grid(beats[beats < 31.9], 0.001 - 0.010))     # every segment moved
        self.assertTrue(self.on_grid(beats[beats > 32.4], -0.010))
        self.panel.btn_undo.click()
        self.panel.btn_undo.click()
        np.testing.assert_allclose(self.model.beats, BEATS)
        self.assertEqual(self.segments(), self.original)
        self.panel.btn_redo.click()
        self.assertAlmostEqual(self.model.tempo_segments[0, 3], 0.001)

    def test_shifting_a_segment_leaves_the_next_ones_beats(self):
        self.panel.btn_shift_fine_fwd.click()
        beats = np.asarray(self.model.beats)
        np.testing.assert_allclose(beats[beats > 31.9], BEATS[BEATS > 31.9])
        np.testing.assert_allclose(self.model.beats, rebuild_grid_from_segments(self.model.tempo_segments))

    def test_half_beat_shift(self):
        self.h["beatgrid.shift_half_forward"]()
        self.assertAlmostEqual(self.model.tempo_segments[0, 3] % 0.5, 0.25)

    def test_count_held_ends_when_the_beatgrid_changes(self):
        self.panel.beat_counter_button.pressed.emit()          # Count held
        self.at(12.0)
        self.assertEqual(self.panel.beat_counter_display.text(), "4")
        self.h["beatgrid.shift_half_forward"]()                # another grid: the count over the old beats ends
        self.at(14.0)
        self.assertEqual(self.panel.beat_counter_display.text(), "4")
        self.assertEqual(self.panel.beat_counter_barbeat_display.text(), "--")
        self.panel.beat_counter_button.released.emit()
        self.panel.beat_counter_button.pressed.emit()          # pressed again: counted anew on this grid
        self.at(15.0)
        self.assertEqual(self.panel.beat_counter_display.text(), "2")
        self.assertAlmostEqual(float(np.asarray(self.model.beats)[1]), 0.75)

    def test_bpm_doubled_and_halved_keeps_the_downbeat(self):
        self.panel.btn_bpm_double.click()
        self.assertEqual(self.model.tempo_segments[0, 2], 240.0)
        self.assertEqual(self.model.tempo_segments[0, 3], 0.0)
        self.assertEqual(self.model.tempo_segments[1, 2], 120.0)      # the other segment as it was
        self.assertEqual(int((np.asarray(self.model.beats) < 32.0).sum()), 128)
        self.panel.btn_bpm_half.click()
        np.testing.assert_allclose(self.model.beats, BEATS)
        self.assertEqual(self.segments(), self.original)
        self.panel.btn_scope_all.click()
        self.panel.btn_bpm_half.click()
        self.assertEqual(self.model.tempo_segments[:, 2].tolist(), [60.0, 60.0])
        self.panel.btn_undo.click()
        self.panel.btn_undo.click()
        self.panel.btn_undo.click()
        self.assertEqual(self.segments(), self.original)
        self.assertFalse(self.model.can_undo())

    def test_split_at_the_playhead_and_merge_back(self):
        self.at(10.3)
        self.h["editor.split"]()                       # the Beatgrid page is the current one
        self.assertEqual(self.model.tempo_segments[:, 0].tolist(), [0.0, 10.3, 32.0])
        self.assertEqual(self.model.tempo_segments[1, 3], 12.0)       # its first downbeat on the old bars
        np.testing.assert_allclose(self.model.beats, BEATS)           # no beat moved
        self.h["editor.merge_previous"]()
        self.assertEqual(self.segments(), self.original)
        np.testing.assert_allclose(self.model.beats, BEATS)
        self.at(40.0)
        self.h["editor.merge_previous"]()
        self.assertEqual(self.model.tempo_segments[:, :2].tolist(), [[0.0, 64.0]])

    def test_split_at_a_segment_edge_does_nothing(self):
        self.at(32.0)
        self.panel.btn_cutseg.click()
        self.at(0.0)
        self.panel.btn_cutseg.click()
        self.assertFalse(self.model.can_undo())

    def test_time_signature_and_downbeat(self):
        self.h["beatgrid.time_signature_3"]()
        self.assertEqual(self.model.tempo_segments[:, 4].tolist(), [3.0, 4.0])
        self.at(10.6)                                  # nearest beat 10.5: its downbeat
        self.panel.btn_set_downbeat.click()
        self.assertAlmostEqual(self.model.tempo_segments[0, 3] % 1.5, 10.5 % 1.5)
        np.testing.assert_allclose(self.model.beats, BEATS)

    def test_assign_the_reference_bpm(self):
        self.panel.inp_refBPM.setText("128")
        self.panel.btn_setBPM.click()
        self.assertEqual(self.model.tempo_segments[0, 2], 128.0)
        self.assertEqual(self.model.tempo_segments[0, 5], 128.0)
        self.panel.inp_refBPM.setText("abc")
        self.panel.btn_setBPM.click()
        self.assertEqual(self.model.tempo_segments[0, 2], 128.0)
        self.panel.btn_zeroBPM.click()
        self.assertEqual(self.panel.inp_refBPM.text(), "")

    def test_ramp_from_the_reference_bpm(self):
        self.panel.inp_refBPM.setText("100")
        self.panel.btn_ramp_start.click()
        row = self.model.tempo_segments[0]
        self.assertEqual((row[2], row[5]), (100.0, 120.0))
        self.assertEqual(float(self.model.beats[0]), 0.0)              # its first beat kept

    def test_the_status_line_follows_the_playhead(self):
        self.at(40.0)
        self.assertIn("Seg 2", self.panel.beatgrid_status.text())


class PageShortcutTests(EditPanelCase):
    def setUp(self):
        super().setUp()

    def test_a_page_wide_shortcut_works_on_the_current_page(self):
        self.at(10.2)
        self.h["editor.page_key"]()
        self.h["editor.split"]()
        self.assertEqual(len(self.model.tempo_segments), 2)                # the beatgrid untouched
        self.assertEqual([k[2] for k in self.keys()], [0.0, 10.0, 32.0])   # at the nearest beat
        self.h["editor.page_phrase"]()
        self.h["editor.split"]()
        self.assertEqual([p[0] for p in self.phrases()], [0.0, 10.0, 16.0, 48.0])
        self.h["editor.page_cue"]()
        self.assertEqual(self.h["editor.split"](), "This shortcut does nothing on the CUE page.")

    def test_a_pages_own_tool_shows_its_page(self):
        self.h["editor.page_beatgrid"]()
        self.at(20.0)
        self.h["phrase.label.CHORUS"]()
        self.assertEqual(self.panel._context, PHRASE_PAGE)
        self.assertEqual(self.phrases()[1][2], "CHORUS")


class KeyTests(EditPanelCase):
    def setUp(self):
        super().setUp()
        self.h["editor.page_key"]()

    def test_keys_changed_on_the_segment_or_the_whole_track(self):
        self.at(10.0)
        self.h["key.sharp"]()
        self.assertEqual(self.keys(), [(1.0, 0.0, 0.0, 32.0), (9.0, 1.0, 32.0, 64.0)])
        self.h["key.relative"]()                       # C# major -> A# minor
        self.assertEqual(self.keys()[0][:2], (10.0, 1.0))
        self.panel.key_scope_all.click()
        self.h["key.flat"]()
        self.assertEqual([k[:2] for k in self.keys()], [(9.0, 1.0), (8.0, 1.0)])
        self.h["key.merge_same"]()                     # nothing alike next to each other
        self.assertEqual(len(self.keys()), 2)
        self.h["key.sharp"]()
        self.h["key.flat"]()
        self.panel._write_key_scoped(5)                # picked for the whole track: one segment
        self.assertEqual(self.keys(), [(5.0, 0.0, 0.0, 64.0)])
        self.assertNotEqual(self.panel.key_pick.text(), "--")       # the picker shows the key under the playhead

    def test_key_changes_lit_while_the_key_heard_is_not_the_tracks(self):
        from ui.edit_panel.styles import WARNING_BUTTON_STYLESHEET
        tools = (self.panel.key_flat, self.panel.key_sharp, self.panel.key_fifth_down, self.panel.key_fifth_up,
                 self.panel.key_relative)
        self.at(10.0)
        self.assertTrue(all(t.styleSheet() == "" for t in tools))
        self.bus.sig_played_changed.emit(1.0, 2)        # the key shifted by 2 semitones
        self.assertTrue(all(t.styleSheet() == WARNING_BUTTON_STYLESHEET for t in tools))
        self.assertIn("WARNING", self.panel.key_sharp.toolTip())
        self.assertNotEqual(self.panel.key_pick.text(), "--")          # its key's colour kept, framed
        self.h["key.sharp"]()                          # still the track's own keys that change
        self.assertEqual(self.keys()[0][:2], (1.0, 0.0))
        self.bus.sig_played_changed.emit(1.06, 12)      # an octave: the same key
        self.assertTrue(all(t.styleSheet() == "" for t in tools))
        self.assertNotIn("WARNING", self.panel.key_sharp.toolTip())

    def test_split_merge_and_merge_same(self):
        self.at(40.3)
        self.h["editor.split"]()
        self.assertEqual([k[2:] for k in self.keys()], [(0.0, 32.0), (32.0, 40.5), (40.5, 64.0)])
        self.h["key.merge_same"]()
        self.assertEqual([k[2:] for k in self.keys()], [(0.0, 32.0), (32.0, 64.0)])
        self.at(10.0)
        self.h["editor.merge_next"]()
        self.assertEqual(self.keys(), [(0.0, 0.0, 0.0, 64.0)])            # the current segment's key kept
        self.h["editor.merge_next"]()                                     # no neighbour
        self.model.undo()
        self.model.undo()
        self.model.undo()
        self.assertEqual(self.keys(), [(0.0, 0.0, 0.0, 32.0), (9.0, 1.0, 32.0, 64.0)])

    def test_a_boundary_moved_by_a_beat_or_a_bar(self):
        self.at(40.0)
        self.h["editor.start_earlier"]()
        self.assertEqual(self.keys()[1][2], 31.5)
        self.assertEqual(self.keys()[0][3], 31.5)
        self.h["editor.start_later"]()
        self.assertEqual(self.keys()[1][2], 32.0)
        self.h["editor.step_bar"]()
        self.h["editor.start_later"]()
        self.assertEqual(self.keys()[1][2], 34.0)
        self.h["editor.start_earlier"]()
        self.h["editor.start_earlier"]()
        self.assertEqual(self.keys()[1][2], 30.0)
        self.at(10.0)
        self.h["editor.start_earlier"]()               # the track start does not move
        self.assertEqual(self.keys()[0][2], 0.0)

    def test_one_listening_aid_held_at_a_time(self):
        held = []
        self.bus.sig_tone_hold.connect(held.append)
        self.at(10.0)                                   # C major
        self.h["key.root"]()
        self.h["key.chord"]()                           # the chord instead of the root
        self.assertEqual((self.panel.key_root.isChecked(), self.panel.key_chord.isChecked()), (False, True))
        self.assertEqual(held[-1], (48, 60, 64, 67))     # the root in the bass, the triad an octave up
        self.h["key.chord"]()                           # again: none
        self.assertFalse(any(b.isChecked() for b in (self.panel.key_root, self.panel.key_drone,
                                                       self.panel.key_chord)))
        self.assertEqual(held[-1], ())

    def test_the_whole_track_reanalysis_asked(self):
        asked = []
        self.bus.sig_segment_reanalyze_requested.connect(asked.append)
        self.h["editor.reanalyze_all"]()
        self.assertEqual(len(asked), 1)
        np.testing.assert_allclose(asked[0].beats, BEATS)
        self.assertEqual(asked[0].path, "track.wav")


class PhraseTests(EditPanelCase):
    def setUp(self):
        super().setUp()
        self.h["editor.page_phrase"]()
        self.at(20.0)                                   # in VERSE

    def test_label_split_merge_and_unassign(self):
        self.h["phrase.label.CHORUS"]()
        self.at(30.1)
        self.h["editor.split"]()
        self.assertEqual(self.phrases(), [(0.0, 16.0, "INTRO"), (16.0, 30.0, "CHORUS"), (30.0, 48.0, "CHORUS"),
                                          (48.0, 64.0, "OUTRO")])
        self.h["editor.merge_next"]()
        self.assertEqual(self.phrases()[2], (30.0, 64.0, "CHORUS"))
        self.h["phrase.unassign"]()
        self.assertEqual(self.phrases()[2][2], UNASSIGNED)
        self.assertEqual([p.label for p in self.model.stored_edits().phrases], ["INTRO", "CHORUS"])  # not stored

    def test_boundaries_moved_on_the_grid(self):
        self.h["editor.end_later"]()
        self.assertEqual(self.phrases()[1][1], 48.5)
        self.assertEqual(self.phrases()[2][0], 48.5)
        self.h["editor.step_bar"]()
        self.h["editor.start_earlier"]()
        self.assertEqual(self.phrases()[1][0], 14.0)
        self.assertEqual(self.phrases()[0][1], 14.0)

    def test_the_cue_points_made_from_the_phrases(self):
        self.h["phrase.cue_points"]()
        times = [p.time_sec for p in self.model.cue_points]
        self.assertIn(48.0, times)                     # OUTRO's start (the default rules)
        self.assertNotIn(16.0, times)                  # the earlier ones replaced
        self.model.undo()
        self.assertEqual([p.time_sec for p in self.model.cue_points], [16.0])


class CueTests(EditPanelCase):
    def setUp(self):
        super().setUp()

    def test_memory_on_the_nearest_beat_once(self):
        self.at(20.2)
        self.h["cue.memory"]()
        self.assertEqual([p.time_sec for p in self.model.cue_points], [16.0, 20.0])
        self.h["cue.memory"]()
        self.assertEqual(len(self.model.cue_points), 2)
        self.at(63.9)                                  # the track end is a snap point too
        self.h["cue.memory"]()
        self.assertEqual(self.model.cue_points[-1].time_sec, 64.0)

    def test_delete_the_cue_point_at_the_playhead(self):
        self.at(30.0)
        self.assertEqual(self.h["cue.memory_delete"](), "No CUE point at the playhead.")
        self.at(16.02)
        self.assertIsNone(self.h["cue.memory_delete"]())
        self.assertEqual(self.model.cue_points, [])

    def test_rename_and_colour(self):
        self.panel._rename_cue_point(0, "Drop")
        self.assertEqual(self.model.cue_points[0].label, "Drop")
        self.panel._update_cue_point(self.model.cue_points[0], color_idx=3)
        self.assertEqual(self.model.cue_points[0].color_idx, 3)
        self.assertEqual(self.panel.cp_slots[0].name.text(), "Drop")

    def test_call_moves_the_deck_while_stopped(self):
        actions = []
        self.bus.sig_deck_action.connect(lambda code, on: actions.append((code, on)))
        self.at(10.0)
        self.h["cue.call_next"]()
        self.assertEqual(actions, [(deck.ACT_CALL_NEXT, True)])
        self.h["cue.call_previous"]()                  # none before the playhead
        self.assertEqual(len(actions), 1)
        self.bus.sig_playback_status.emit(True)
        self.assertEqual(self.h["cue.call_next"](), "CALL works while stopped.")
        self.assertEqual(len(actions), 1)
        called = []
        self.bus.sig_call_requested.connect(called.append)
        self.panel._cp_call(0)
        self.assertEqual(called, [])
        self.bus.sig_playback_status.emit(False)
        self.panel._cp_call(0)
        self.assertEqual(called, [16.0])


class JumpCueTests(EditPanelCase):
    def setUp(self):
        super().setUp()

    def test_set_in_an_empty_slot_in_a_colour_of_its_own(self):
        self.at(20.2)
        self.assertIsNone(self.h["cue.set.C"]())
        cues = self.jump_cues()
        self.assertEqual(cues["C"][0], 20.0)
        self.assertNotEqual(cues["C"][1], cues["A"][1])                 # a group of its own
        self.assertEqual(self.h["cue.set.C"](), "JumpCUE C is already set.")
        self.assertEqual(self.panel.jc_slots[2].letter.text(), "C")

    def test_two_cues_never_share_a_point(self):
        self.at(8.1)                                   # A is at 8.0
        with mock.patch.object(QMessageBox, "warning") as warned:
            self.h["cue.set.D"]()
        self.assertTrue(warned.called)
        self.assertEqual(sorted(self.jump_cues()), ["A", "B"])

    def test_jump_asks_the_deck(self):
        jumps = []
        self.bus.sig_jump_to_requested.connect(jumps.append)
        self.h["cue.jump.B"]()
        self.assertEqual(jumps, [40.0])
        self.assertEqual(self.h["cue.jump.E"](), "JumpCUE E is not set.")

    def test_a_cue_dragged_onto_another_slot(self):
        self.panel._move_jumpcue("A", "D")            # into an empty slot
        self.assertEqual(self.jump_cues(), {"D": (8.0, 0), "B": (40.0, 0)})
        self.panel._rename_jumpcue(1, "Drop")          # slot B
        self.panel._move_jumpcue("B", "D")             # onto a cue: the two swap, a name stays with its cue
        self.assertEqual(self.jump_cues(), {"B": (8.0, 0), "D": (40.0, 0)})
        self.assertEqual({c.label: c.comment for g in self.model.jump_cues for c in g.cues}, {"B": "B", "D": "Drop"})
        self.model.undo()
        self.assertEqual(self.jump_cues(), {"D": (8.0, 0), "B": (40.0, 0)})
        self.panel._move_jumpcue("C", "A")             # no cue in C: nothing
        self.panel._move_jumpcue("D", "D")
        self.assertEqual(self.jump_cues(), {"D": (8.0, 0), "B": (40.0, 0)})

    def test_a_drop_on_a_slot(self):
        def drop(slot, mime_type: str, payload: bytes):
            mime = QtCore.QMimeData()
            mime.setData(mime_type, QtCore.QByteArray(payload))
            slot.dropEvent(QtGui.QDropEvent(QtCore.QPointF(4, 4), QtCore.Qt.MoveAction, mime, QtCore.Qt.LeftButton,
                                            QtCore.Qt.NoModifier))
        drop(self.panel.jc_slots[5], "text/plain", b"A")       # not a dragged cue
        self.assertEqual(sorted(self.jump_cues()), ["A", "B"])
        drop(self.panel.jc_slots[5], SLOT_MIME, b"A")          # slot F
        self.assertEqual(self.jump_cues(), {"F": (8.0, 0), "B": (40.0, 0)})

    def test_a_drag_from_the_squircle_or_the_name(self):
        def press_move(widget, at: QtCore.QPoint):
            far = at + QtCore.QPoint(QtWidgets.QApplication.startDragDistance() + 2, 0)
            for kind, pos, button, buttons in ((QtCore.QEvent.MouseButtonPress, at, QtCore.Qt.LeftButton,
                                                QtCore.Qt.LeftButton),
                                               (QtCore.QEvent.MouseMove, far, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)):
                QtWidgets.QApplication.sendEvent(widget, QtGui.QMouseEvent(
                    kind, QtCore.QPointF(pos), QtCore.QPointF(widget.mapToGlobal(pos)), button, buttons,
                    QtCore.Qt.NoModifier))
        slot = self.panel.jc_slots[1]                  # B
        menus = []
        slot.squircle_clicked.connect(lambda: menus.append(1))
        with mock.patch.object(CueSlot, "drag") as drag:
            press_move(slot.squircle, QtCore.QPoint(3, 3))
            self.assertEqual(drag.call_args.args[0], "B")
            self.assertFalse(slot.squircle.isDown())   # dragged, not clicked: no menu
            press_move(slot, slot.name_stack.geometry().center())   # the name leaves its press to the slot
            self.assertEqual(drag.call_count, 2)
            press_move(self.panel.jc_slots[4], QtCore.QPoint(40, 4))     # an empty slot: nothing to drag
            self.assertEqual(drag.call_count, 2)
            self.bus.sig_segment_reanalyze_state.emit(False)               # editing off: no drag
            press_move(slot.squircle, QtCore.QPoint(3, 3))
            self.assertEqual(drag.call_count, 2)
            self.assertFalse(slot.acceptDrops())
        self.assertEqual(menus, [])

    def test_a_dragged_cue_held_over_a_page_arrow_turns_the_page(self):
        self.at(20.0)
        self.h["cue.set.H"]()                          # A..H in use: page 2 (I-P) to move to
        page = self.panel.cue_page
        mime = QtCore.QMimeData()
        mime.setData(SLOT_MIME, QtCore.QByteArray(b"A"))
        QtWidgets.QApplication.sendEvent(page.btn_next, QtGui.QDragEnterEvent(
            QtCore.QPoint(2, 2), QtCore.Qt.MoveAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        self.assertEqual(page.pages[0], 1)
        self.assertTrue(page._turn_timer.isActive())
        page._turn_timer.timeout.emit()                # the last page: it stops
        self.assertEqual(page.pages[0], 1)
        self.assertFalse(page._turn_timer.isActive())
        self.panel._jc_drop(2, "A")                    # slot K
        self.assertEqual(self.jump_cues()["K"][0], 8.0)

    def test_a_colour_is_a_group(self):
        self.at(20.0)
        self.h["cue.set.C"]()
        color_a = self.model.jump_cues[self.jump_cues()["A"][1]].color_idx
        self.panel._set_jumpcue_color(20.0, color_a)
        self.assertEqual(len(self.model.jump_cues), 1)
        self.assertEqual([c.label for c in self.model.jump_cues[0].cues], ["A", "C", "B"])     # by point

    def test_rename_and_a_group_moved_by_a_beat(self):
        self.panel._rename_jumpcue(0, "Drop")
        self.assertEqual({c.label: c.comment for c in self.model.jump_cues[0].cues}["A"], "Drop")
        self.panel._rename_jumpcue(0, "")
        self.assertEqual({c.label: c.comment for c in self.model.jump_cues[0].cues}["A"], "A")
        self.panel._shift_jumpcue_group(0, 1)
        self.assertEqual(self.jump_cues(), {"A": (8.5, 0), "B": (40.5, 0)})

    def test_the_editor_stays_open_through_its_moves(self):
        editor = self.panel._jc_editor
        self.panel._jc_open_editor(0)                  # A's colour square
        self.assertEqual(editor.letter, "A")
        editor.btn_beat_later.click()
        editor.btn_beat_later.click()                  # clicked again: no reopening
        self.assertEqual(self.jump_cues(), {"A": (9.0, 0), "B": (41.0, 0)})
        editor.btn_bar_later.click()                   # to its place in the next bar (4 beats of 0.5 s)
        self.assertEqual(self.jump_cues(), {"A": (11.0, 0), "B": (43.0, 0)})
        editor.btn_bar_earlier.click()
        editor.btn_beat_earlier.click()
        self.assertEqual(self.jump_cues(), {"A": (8.5, 0), "B": (40.5, 0)})
        self.assertEqual(editor.letter, "A")
        self.panel._jc_open_editor(1)                  # another cue's square: the editor turns to it
        self.assertEqual(editor.letter, "B")
        self.panel._jc_open_editor(1)                  # its own square clicked again: closed
        self.assertIsNone(editor.letter)

    def test_a_bar_move_in_a_shorter_bar_stops_at_its_last_beat(self):
        # 3/4 from 16 s (beat 32) on: A on the 4th beat of its 4/4 bar goes to the 3rd of the 3/4 bar after
        self.model.set_beatgrid(self.model.beats, with_end_bpm(np.asarray([[0.0, 16.0, 120.0, 0.0, 4.0],
                                                                           [16.0, 64.0, 120.0, 16.0, 3.0]])))
        self.panel._set_jumpcue_groups([([JumpCue(label="A", comment="A", start=15.5, end=15.5, point=15.5)], 0)])
        self.panel._shift_jumpcue_group(0, 1, by_bar=True)
        self.assertEqual(self.jump_cues(), {"A": (17.0, 0)})

    def test_the_editor_follows_its_cue_into_other_slots(self):
        editor = self.panel._jc_editor
        self.panel._jc_open_editor(0)
        self.panel._move_jumpcue("A", "E")             # dragged
        self.assertEqual(editor.letter, "E")
        self.panel._move_jumpcue("B", "E")             # another cue dragged onto it: the two swap
        self.assertEqual(editor.letter, "B")
        self.assertEqual(self.jump_cues()["B"][0], 8.0)

    def test_the_editors_comment(self):
        editor = self.panel._jc_editor
        self.panel._rename_jumpcue(1, "Drop")
        self.panel._jc_open_editor(1)                  # B, named
        self.assertEqual(editor.comment.text(), "Drop")
        self.panel._jc_open_editor(0)                  # A: its comment just its letter, shown empty
        self.assertEqual(editor.comment.text(), "")
        editor.comment.setText("Chorus")
        editor.comment.editingFinished.emit()
        self.assertEqual({c.label: c.comment for g in self.model.jump_cues for c in g.cues}, {"A": "Chorus", "B": "Drop"})
        self.assertEqual(self.panel.jc_slots[0].name.text(), "Chorus")
        changes = []
        self.model.sig_jump_cues_changed.connect(lambda *_: changes.append(1))
        editor.comment.editingFinished.emit()          # kept again unchanged: no edit
        self.assertEqual(changes, [])
        editor.comment.setText("")
        editor.comment.editingFinished.emit()          # no comment: back to its letter
        self.assertEqual({c.label: c.comment for g in self.model.jump_cues for c in g.cues}["A"], "A")
        editor.comment.setText("typing")               # Esc: the comment as kept again
        QtWidgets.QApplication.sendEvent(editor.comment, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape,
                                                                         QtCore.Qt.NoModifier))
        self.assertEqual(editor.comment.text(), "")
        self.assertEqual({c.label: c.comment for g in self.model.jump_cues for c in g.cues}["A"], "A")

    def test_the_editor_picks_the_colour_and_closes_with_its_cue_or_the_track(self):
        editor = self.panel._jc_editor
        self.at(20.0)
        self.h["cue.set.C"]()
        self.panel._jc_open_editor(2)
        color_a = self.model.jump_cues[self.jump_cues()["A"][1]].color_idx
        editor.color_cells[color_a].click()            # C joins A's group, the editor still open
        self.assertEqual(len(self.model.jump_cues), 1)
        self.assertEqual(editor.letter, "C")
        editor.btn_delete.click()
        self.assertEqual(sorted(self.jump_cues()), ["A", "B"])
        self.assertIsNone(editor.letter)
        self.panel._jc_open_editor(0)
        self.model.load(StoredAnalysis.from_arrays(arrays()), {"uid": self.UID, "path": "track.wav"}, 64.0)
        self.assertIsNone(editor.letter)

    def test_the_editors_shortcuts_and_editing_off(self):
        editor = self.panel._jc_editor
        self.assertEqual(self.h["cue.color_beat_later"](),
                         "No JumpCUE is open in the editor: click a JumpCUE's color square.")
        self.panel._jc_open_editor(1)                  # B
        self.assertIsNone(self.h["cue.color_bar_later"]())
        self.assertEqual(self.jump_cues(), {"A": (10.0, 0), "B": (42.0, 0)})
        self.bus.sig_segment_reanalyze_state.emit(False)
        self.assertFalse(editor.btn_beat_later.isEnabled())
        self.assertTrue(editor.btn_close.isEnabled())
        self.assertIsNotNone(self.h["cue.color_beat_later"]())
        self.assertEqual(self.jump_cues(), {"A": (10.0, 0), "B": (42.0, 0)})

    def test_the_delete_shortcut_deletes_the_slots_cue(self):
        self.assertIsNone(self.h["cue.delete.B"]())
        self.assertEqual(sorted(self.jump_cues()), ["A"])

    def test_the_deck_given_the_marks_as_edited(self):
        sent = []
        self.bus.sig_deck_marks.connect(sent.append)
        self.panel._send_deck_marks()
        self.panel._send_deck_marks()                  # nothing replaced: not sent again
        self.assertEqual(len(sent), 1)
        beats, down, cues, jumps = sent[0]
        np.testing.assert_allclose(beats, BEATS)
        self.assertEqual(np.flatnonzero(down)[:3].tolist(), [0, 4, 8])
        self.assertEqual(cues.tolist(), [16.0])
        self.assertEqual(jumps[:2].tolist(), [8.0, 40.0])
        self.assertTrue(np.isnan(jumps[2:]).all())
        self.at(20.0)
        self.h["cue.memory"]()
        self.panel._send_deck_marks()
        self.assertEqual(sent[-1][2].tolist(), [16.0, 20.0])


class SaveTests(EditPanelCase):
    """The editor's save, through the library's owner, into a library of its own."""

    def library(self):
        self._tmp = tempfile.TemporaryDirectory()
        db = LibraryDB(os.path.join(self._tmp.name, "library.db"))
        db.connect()
        db.upsert(TrackRow(uid=self.UID, path="track.wav", title="t", added_ts=1))
        db.conn.commit()
        db.close()
        FeatureNPZStore(base_dir=self._tmp.name, compressed=True).save(self.UID, arrays())
        self.service = LibraryService(self._tmp.name)
        return self.service

    def tearDown(self):
        super().tearDown()
        self.service.close()
        self._tmp.cleanup()

    def settle(self) -> None:
        for _ in range(4):
            self.service.wait()

    def test_edits_saved_and_marked_saved(self):
        self.at(10.0)
        self.panel.btn_bpm_double.click()
        self.assertTrue(self.model.unsaved)
        self.h["editor.save"]()
        self.assertFalse(self.panel.btn_save.isEnabled())             # while it runs
        self.at(40.0)
        self.h["cue.memory"]()                                        # an edit while saving
        self.settle()
        self.assertTrue(self.panel.btn_save.isEnabled())
        self.assertTrue(self.model.unsaved)                           # the later edit not in that save
        stored = StoredAnalysis.from_arrays(FeatureNPZStore(base_dir=self._tmp.name).load(self.UID))
        self.assertEqual(stored.tempo_segments[0, 2], 240.0)
        self.assertEqual([p.time_sec for p in stored.cue_points], [16.0])
        self.h["editor.save"]()
        self.settle()
        self.assertFalse(self.model.unsaved)
        stored = StoredAnalysis.from_arrays(FeatureNPZStore(base_dir=self._tmp.name).load(self.UID))
        self.assertEqual([p.time_sec for p in stored.cue_points], [16.0, 40.0])
        db = LibraryDB(os.path.join(self._tmp.name, "library.db"))      # the transition search's rows follow
        db.connect()
        try:
            self.assertEqual([r.bpm for r in db.get_bpm_segments(self.UID)], [240.0, 120.0])
        finally:
            db.close()

    def test_nothing_saved_during_a_reanalysis(self):
        self.at(10.0)
        self.panel.btn_bpm_double.click()
        self.bus.sig_segment_reanalyze_state.emit(False)
        self.h["editor.save"]()
        self.settle()
        self.assertTrue(self.model.unsaved)

    def test_save_then_runs_after_the_save_only(self):
        """save_then: what comes after (a load the user asked for) runs once the edits are stored, never before,
        never when no save could start; a save already running keeps its own."""
        ran = []
        self.at(10.0)
        self.panel.btn_bpm_double.click()
        self.assertTrue(self.panel.save_then(lambda: ran.append("first")))
        self.assertEqual(ran, [])                                      # not before the save is stored
        self.assertFalse(self.panel.save_then(lambda: ran.append("second")))    # a save is running
        self.settle()
        self.assertEqual(ran, ["first"])
        self.assertFalse(self.model.unsaved)
        self.panel.btn_bpm_double.click()
        self.bus.sig_segment_reanalyze_state.emit(False)               # editing off: no save starts
        self.assertFalse(self.panel.save_then(lambda: ran.append("third")))
        self.settle()
        self.assertEqual(ran, ["first"])

    def test_after_a_failed_save_nothing_runs(self):
        ran = []
        self.at(10.0)
        self.panel.btn_bpm_double.click()
        with mock.patch.object(QMessageBox, "warning"):
            self.assertTrue(self.panel.save_then(lambda: ran.append("after")))
            self.assertTrue(self.panel.saving)
            self.panel._on_save_failed(OSError("disk full"))
        self.assertEqual(ran, [])
        self.assertFalse(self.panel.saving)
        self.assertTrue(self.model.unsaved)


if __name__ == "__main__":
    unittest.main()
