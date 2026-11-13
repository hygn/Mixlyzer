"""The Export tab's Render Audio (ui.library.tabs.export, ui.render.track_audio): the track written as a WAV on a
thread of its own: whole, with a JumpCUE's jump baked in (the cue jumped from and the cue jumped to picked apart,
any two), at the tempo changed by the native engine (where it is built) or the track's own."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from unit.ui_tests.qt_env import APP, isolated  # noqa: F401
from unit.ui_tests.test_export_tempo_map import UID, _Library, finish

from PySide6 import QtWidgets

from analyzer.music.marks import JumpCue, JumpCueGroup
from core.analysis_library.stored import StoredAnalysis
from core.library.records import TrackRow
from core.playback.player import native_engine_built
from ui.library.tabs.export import ExportTrackPanel

_ISOLATION = None
RATE = 48000


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


def cue(label: str, point: float) -> JumpCue:
    return JumpCue(label=label, comment="", start=point, end=point + 1.0, point=point)


class ExportRenderTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        boxes = mock.patch.multiple(QtWidgets.QMessageBox, information=mock.DEFAULT, warning=mock.DEFAULT,
                                    critical=mock.DEFAULT, question=mock.DEFAULT)
        self.boxes = boxes.start()
        self.addCleanup(boxes.stop)
        # 8 s of a tone whose pitch tells the second (440 Hz + 110 Hz a second)
        t = np.arange(8 * RATE) / RATE
        tone = 0.3 * np.sin(2 * np.pi * np.cumsum(440.0 + 110.0 * np.floor(t)) / RATE)
        self.source = Path(self.dir.name) / "曲.wav"
        sf.write(str(self.source), np.stack([tone, tone], axis=1).astype(np.float32), RATE)
        groups = (JumpCueGroup(cues=(cue("A", 2.0), cue("B", 6.0)), color_idx=0),
                  JumpCueGroup(cues=(cue("C", 4.0),), color_idx=1))
        library = _Library(StoredAnalysis(duration_sec=8.0, jump_cues=groups))
        self.panel = ExportTrackPanel(library)
        self.panel.set_track(TrackRow(uid=UID, path=str(self.source), title="曲", bpm=120.0, duration=8.0))
        self.panel.dest_edit.setText(self.dir.name)
        library.answer()

    def rendered(self, name: str) -> tuple[np.ndarray, int]:
        self.panel.render_audio()
        finish(self.panel)
        self.boxes["information"].assert_called()
        pcm, rate = sf.read(str(Path(self.dir.name) / name))
        return pcm, rate

    def pick(self, picker, label):
        picker.setCue(next(cue for cue in picker.cues() if cue.label == label))

    def test_every_cue_in_both_pickers_by_letter(self):
        self.assertEqual([cue.label for cue in self.panel.btn_jump_to.cues()], ["A", "B", "C"])
        self.assertIsNone(self.panel.btn_jump_from.cue())       # no jump by default
        self.assertFalse(self.panel.btn_jump_to.isEnabled())
        self.pick(self.panel.btn_jump_from, "A")                 # the cue jumped to: the next one by default
        self.assertTrue(self.panel.btn_jump_to.isEnabled())
        self.assertEqual(self.panel.btn_jump_to.cue().label, "B")

    def test_the_whole_track(self):
        self.panel.spn_tempo.setValue(0.0)
        pcm, rate = self.rendered("曲_render.wav")
        self.assertEqual((rate, pcm.shape), (RATE, (8 * RATE, 2)))

    def test_a_jump_between_cues_of_two_groups(self):
        self.panel.spn_tempo.setValue(0.0)
        self.pick(self.panel.btn_jump_from, "C")        # 4 s, jumped back to A at 2 s
        self.pick(self.panel.btn_jump_to, "A")
        pcm, _rate = self.rendered("曲_jump_C_to_A.wav")
        self.assertEqual(pcm.shape[0], 4 * RATE + 6 * RATE)      # 0 - 4 s, then 2 - 8 s
        # after the jump (4 s in), the tone of the source's third second (A, 2 s): 660 Hz
        spectrum = np.abs(np.fft.rfft(pcm[int(4.2 * RATE):int(4.7 * RATE), 0]))
        self.assertAlmostEqual(np.argmax(spectrum) * RATE / (0.5 * RATE), 660.0, delta=4.0)

    def test_the_same_cue_twice_refused(self):
        self.pick(self.panel.btn_jump_from, "B")
        self.pick(self.panel.btn_jump_to, "B")
        self.panel.render_audio()
        self.boxes["warning"].assert_called_once()
        self.assertEqual(sorted(p.name for p in Path(self.dir.name).iterdir()), ["曲.wav"])

    @unittest.skipUnless(native_engine_built(), "the native engine is not built")
    def test_at_the_changed_tempo(self):
        self.panel.spn_bpm.setValue(96.0)               # -20 % of the stored 120 BPM
        self.assertAlmostEqual(self.panel.spn_tempo.value(), -20.0)
        self.pick(self.panel.btn_jump_from, "A")
        self.pick(self.panel.btn_jump_to, "B")
        pcm, _rate = self.rendered("曲_jump_A_to_B_96BPM.wav")             # named by the target BPM
        self.assertAlmostEqual(pcm.shape[0] / RATE, (8.0 - 4.0) / 0.8, delta=0.05)
        self.panel.chk_render_tempo.setChecked(False)   # the track's own tempo
        pcm, _rate = self.rendered("曲_jump_A_to_B.wav")
        self.assertEqual(pcm.shape[0], 4 * RATE)

    def pitch(self, pcm: np.ndarray, start: float) -> float:
        """The strongest frequency of 0.5 s of ``pcm`` from ``start`` s."""
        x = pcm[int(start * RATE):int((start + 0.5) * RATE), 0]
        spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size), 8 * x.size))
        return np.argmax(spectrum) * RATE / (8 * x.size)

    @unittest.skipUnless(native_engine_built(), "the native engine is not built")
    def test_at_the_changed_key_alone_or_with_the_tempo(self):
        self.panel.spn_tempo.setValue(0.0)
        self.panel.spn_key.setValue(2.0)
        pcm, _rate = self.rendered("曲_key+2.wav")                       # named by the shift
        self.assertAlmostEqual(pcm.shape[0] / RATE, 8.0, delta=0.05)    # the tempo kept
        self.assertAlmostEqual(self.pitch(pcm, 0.25), 440.0 * 2 ** (2 / 12), delta=1.5)
        self.panel.spn_bpm.setValue(96.0)                               # both at once: -20 % and 3 down
        self.panel.spn_key.setValue(-3.0)
        pcm, _rate = self.rendered("曲_96BPM_key-3.wav")
        self.assertAlmostEqual(pcm.shape[0] / RATE, 8.0 / 0.8, delta=0.05)
        self.assertAlmostEqual(self.pitch(pcm, 0.25), 440.0 * 2 ** (-3 / 12), delta=1.5)
        # the source's third second (660 Hz) at 2 / 0.8 s, three down
        self.assertAlmostEqual(self.pitch(pcm, 2.6), 660.0 * 2 ** (-3 / 12), delta=1.5)
        self.panel.chk_render_key.setChecked(False)                     # the track's own key, at the tempo
        pcm, _rate = self.rendered("曲_96BPM.wav")
        self.assertAlmostEqual(self.pitch(pcm, 0.25), 440.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
