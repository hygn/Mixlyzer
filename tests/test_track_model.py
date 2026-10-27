"""The loaded track's one owner (core.track.model): its marks changed through its methods alone, each change told by
its own signal, undo / redo over every mark, the edits saved, and a segment reanalysis's result going in as an edit
whether or not an editor is shown."""
import os
from dataclasses import replace
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.workflows.event_bus import EventBus
from core.track.model import DataModel
from analyzer.music.marks import CuePoint, JumpCue, JumpCueGroup, Phrase
from core.analysis_library.cue_points import build_cue_points_np
from core.analysis_library.jump_cues import build_jump_cues_np
from core.analysis_library.phrases import build_phrase_segments_np
from core.marks.phrases import UNASSIGNED
from core.analysis_library.stored import StoredAnalysis
from core.track.reanalysis import ReanalysisResult

APP = QApplication.instance() or QApplication([])

SIGNALS = ("sig_loaded", "sig_beatgrid_changed", "sig_keys_changed", "sig_phrases_changed",
           "sig_jump_cues_changed", "sig_cue_points_changed", "sig_edits_changed")


def _pair(a: float, b: float, la: str = "A", lb: str = "B") -> dict:
    cue = lambda p, label: {"label": label, "comment": label, "start": p, "end": p, "point": p, "color_idx": 0}  # noqa: E731
    return {"forward": cue(a, la), "backward": cue(b, lb), "lag_beats": 0.0, "lag_sec": b - a, "score": 1.0,
            "confidence": 1.0}


def _analysis() -> StoredAnalysis:
    return StoredAnalysis.from_arrays({
        "beats_time_sec": np.arange(0.0, 60.0, 0.5),
        "tempo_segments": np.asarray([[30.0, 60.0, 120.0, 30.0, 4.0], [0.0, 30.0, 120.0, 0.0, 4.0]]),  # unsorted
        "key_segments": np.asarray([[9.0, 1.0, 20.0, 60.0], [0.0, 0.0, 0.0, 20.0]]),
        "phrase_segments_np": build_phrase_segments_np([Phrase(8.0, 24.0, "VERSE")]),
        "jump_cues_np": build_jump_cues_np([_pair(8.0, 24.0)]),
        "cue_points_np": build_cue_points_np([CuePoint(time_sec=24.0), CuePoint(time_sec=8.0)]),
        "duration_sec": 60.0,
    })


class TrackModelTests(unittest.TestCase):
    def setUp(self):
        self.model = DataModel()
        self.told = []
        for name in SIGNALS:
            getattr(self.model, name).connect(lambda name=name: self.told.append(name))
        self.model.load(_analysis(), {"uid": "u", "title": "t"}, 60.0)

    def test_loaded_marks_in_the_editors_form(self):
        m = self.model
        self.assertEqual(m.tempo_segments[:, 0].tolist(), [0.0, 30.0])            # sorted, 6 columns
        self.assertEqual(m.tempo_segments.shape[1], 6)
        self.assertEqual(m.key_segments[:, 2].tolist(), [0.0, 20.0])
        self.assertEqual([(r.start, r.label) for r in m.phrases],          # covering the track
                         [(0.0, UNASSIGNED), (8.0, "VERSE"), (24.0, UNASSIGNED)])
        self.assertEqual([p.time_sec for p in m.cue_points], [8.0, 24.0])
        self.assertEqual(len(m.jump_cues), 1)
        self.assertFalse(m.can_undo() or m.unsaved)

    def test_nothing_changes_it_but_its_methods(self):
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            self.model.analysis.beats = np.zeros(3)
        with self.assertRaises(TypeError):
            self.model.properties.update(title="x")
        with self.assertRaises(ValueError):                  # held arrays are read-only: copy, change, set
            self.model.tempo_segments[0, 2] = 99.0

    def test_an_edit_is_told_stored_and_undone(self):
        m = self.model
        self.told.clear()
        segments = np.array(m.tempo_segments)
        segments[0, 2] = 128.0
        beats = np.array(m.beats) + 0.01
        m.set_beatgrid(beats, segments)
        self.assertEqual(self.told, ["sig_beatgrid_changed", "sig_edits_changed"])
        self.assertTrue(m.unsaved and m.can_undo())
        m.set_key_segments(np.asarray([[2.0, 0.0, 0.0, 60.0]]))
        self.told.clear()
        m.undo()
        self.assertEqual(m.key_segments[:, 0].tolist(), [0.0, 9.0])
        self.assertEqual(m.tempo_segments[0, 2], 128.0)                          # the earlier edit stays
        self.assertEqual(set(self.told), set(SIGNALS) - {"sig_loaded"})           # every mark told again
        m.undo()
        self.assertEqual(m.tempo_segments[0, 2], 120.0)
        self.assertFalse(m.can_undo())
        m.redo()
        self.assertEqual(m.tempo_segments[0, 2], 128.0)

    def test_invalid_jump_cue_letters_change_nothing(self):
        before = self.model.jump_cues
        with self.assertRaises(ValueError):
            self.model.set_jump_cues([JumpCueGroup((JumpCue("A", "A", 8.0, 8.0, 8.0),
                                                    JumpCue("AA", "AA", 24.0, 24.0, 24.0)), 0)])
        self.assertIs(self.model.jump_cues, before)
        self.assertFalse(self.model.unsaved)

    def test_the_blocks_saved(self):
        m = self.model
        m.set_phrases([replace(r, label="CHORUS") if r.label == "VERSE" else r for r in m.phrases])
        m.set_automatic(key_segments=np.asarray([[1.0, 0.0, 0.0, 60.0]]))
        saved = m.stored_edits()
        self.assertEqual([p.label for p in saved.phrases], ["CHORUS"])          # UNASSIGNED not stored
        self.assertEqual(saved.key_segments_auto.tolist(), [[1.0, 0.0, 0.0, 60.0]])
        self.assertIsNone(saved.jump_cues_auto)                                  # not given again: as stored
        self.assertEqual(saved.tempo_segments.shape, (2, 6))
        self.assertIsNone(m.analysis.key_segments_auto)        # the analysis as loaded; the result is the model's
        m.mark_saved(m.save_point())
        self.assertFalse(m.unsaved)

    def test_an_edit_while_saving_stays_unsaved(self):
        m = self.model
        m.set_phrases([replace(r, label="CHORUS") if r.label == "VERSE" else r for r in m.phrases])
        point = m.save_point()                                   # the save taken here
        m.set_key_segments(np.asarray([[2.0, 1.0, 0.0, 60.0]]))  # an edit while it runs
        m.mark_saved(point)
        self.assertTrue(m.unsaved)
        m.undo()                                                 # back to what was stored
        self.assertFalse(m.unsaved)
        m.redo()
        self.assertTrue(m.unsaved)
        m.set_automatic(key_segments=np.asarray([[1.0, 0.0, 0.0, 60.0]]))
        m.mark_saved(m.save_point())
        self.assertFalse(m.unsaved)
        m.set_automatic(key_segments=np.asarray([[3.0, 0.0, 0.0, 60.0]]))
        self.assertTrue(m.unsaved)

    def test_letters_the_load_gave_are_unsaved(self):
        def block(labels):
            return {"cue_id": np.arange(len(labels)), "cue_label": np.asarray(labels),
                    "cue_comment": np.asarray(labels), "cue_start": np.arange(len(labels), dtype=float),
                    "cue_end": np.arange(len(labels), dtype=float), "cue_point": np.arange(len(labels), dtype=float),
                    "graph_cue_id": np.arange(len(labels)), "graph_idx": np.zeros(len(labels), int),
                    "graph_id": np.asarray([0]), "graph_color_idx": np.asarray([3])}
        m = self.model
        # another spelling of the slots: read as A and B, as stored
        m.load(StoredAnalysis.from_arrays({"jump_cues_np": block(["a", " B"])}), {}, 60.0)
        self.assertEqual([c.label for g in m.jump_cues for c in g.cues], ["A", "B"])
        self.assertFalse(m.unsaved)
        # letters that are not slots: lettered by point, not saved yet
        m.load(StoredAnalysis.from_arrays({"jump_cues_np": block(["X", "X"])}), {}, 60.0)
        self.assertEqual([c.label for g in m.jump_cues for c in g.cues], ["A", "B"])
        self.assertTrue(m.unsaved)
        self.assertEqual(len(m.stored_edits().jump_cues[0].cues), 2)            # the letters shown are saved
        # more cues than slots: the first 26 by point shown, the load not refused, the stored block kept as it is
        # (a save of other marks never drops the 4 left out) until the JumpCUEs are edited
        labels = [f"{i}" for i in range(30)]
        m.load(StoredAnalysis.from_arrays({"jump_cues_np": block(labels)}), {}, 60.0)
        letters = [c.label for g in m.jump_cues for c in g.cues]
        self.assertEqual(letters, [chr(ord("A") + i) for i in range(26)])
        self.assertEqual(m.jump_cues_left_out, 4)
        self.assertFalse(m.unsaved)
        m.set_key_segments(np.asarray([[2.0, 1.0, 0.0, 60.0]]))
        self.assertIsNone(m.stored_edits().jump_cues)
        from core.analysis_library.stored import apply_edits
        arrays = {"jump_cues_np": block(labels)}
        apply_edits(arrays, m.stored_edits())
        self.assertEqual(list(arrays["jump_cues_np"]["cue_label"]), labels)     # all 30, as stored
        m.set_jump_cues([JumpCueGroup(m.jump_cues[0].cues[:3], 1)])          # the JumpCUEs edited: as shown
        self.assertEqual(len(m.stored_edits().jump_cues[0].cues), 3)
        m.undo()                                                                 # back to as loaded: kept again
        self.assertIsNone(m.stored_edits().jump_cues)

    def test_a_reanalysis_result_is_an_edit_without_an_editor(self):
        from core.workflows.segment_reanalysis import SegmentReanalysisWorkflow
        tasks = SimpleNamespace(updatetask=lambda *a: None, rmtask=lambda *a: None)
        flow = SegmentReanalysisWorkflow(bus=EventBus(), model=self.model, taskmanager=tasks,
                                         get_current_path=lambda: None)
        self.told.clear()
        flow._handle_segment_result(1, ReanalysisResult(
            beats=np.arange(0.0, 60.0, 0.25), tempo_segments=np.asarray([[0.0, 60.0, 240.0, 0.0, 4.0, 240.0]])))
        flow._handle_phrase_result(2, ReanalysisResult(phrases=[Phrase(0.0, 30.0, "INTRO")]))
        self.assertEqual(self.model.tempo_segments[:, 2].tolist(), [240.0])
        self.assertEqual(self.model.phrases[0].label, "INTRO")
        self.assertIn("sig_beatgrid_changed", self.told)
        self.assertIn("sig_phrases_changed", self.told)
        self.assertTrue(self.model.can_undo())
        self.model.undo()
        self.model.undo()
        self.assertEqual(self.model.tempo_segments[:, 2].tolist(), [120.0, 120.0])


if __name__ == "__main__":
    unittest.main()
